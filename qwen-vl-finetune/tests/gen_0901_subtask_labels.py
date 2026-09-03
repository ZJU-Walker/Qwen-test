#!/usr/bin/env python3
"""Generate full-episode labels for the 0901 standalone-pick datasets.

The three collections stop at or just after one grasp.  They do not contain a
subsequent transport/place action and do not have a human-prompt waiting prefix,
so every episode has exactly one inclusive segment:

    0901ball   -> pick up the ball
    0901green  -> pick up the green block
    0901grey   -> pick up the grey box

The current collection was visually audited and has 30/30/31 episodes.  This
generator validates metadata, parquet lengths, and all three camera frame counts
before staging any output.

Dry run (writes only under /tmp)::

    python tests/gen_0901_subtask_labels.py

Install into all three datasets::

    python tests/gen_0901_subtask_labels.py --install

Installed files are protected from overwrite unless ``--force`` is explicit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("/iris/projects/humanoid/trossen_data")
DATASETS = {
    "0901ball": {
        "task": "pick up the ball",
        "episodes": 30,
        "frames": 3030,
    },
    "0901green": {
        "task": "pick up the green block",
        "episodes": 30,
        "frames": 3786,
    },
    "0901grey": {
        "task": "pick up the grey box",
        "episodes": 31,
        "frames": 3595,
    },
}
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
TASK_RE = re.compile(r"pick up the (?:ball|green block|grey box)")
R_GRIPPER = 13
MIN_QC_CLOSE_RUN = 8
# Cam-high audit: ball 8 contains a regrasp plus human intervention; ball 11
# reopens and leaves the ball on the table.  Keep them labeled for manual review,
# but require the automatic QC queue to continue surfacing exactly these clips.
EXPECTED_QC_FLAGS = {
    "0901ball": {"episode_000008.mp4", "episode_000011.mp4"},
    "0901green": set(),
    "0901grey": set(),
}
EXPECTED_TRAIN_EXCLUSIONS = {
    "0901ball": {"episode_000008.mp4", "episode_000011.mp4"},
    "0901green": set(),
    "0901grey": set(),
}


def _episode_lengths(root: Path) -> dict[str, int]:
    lengths = {}
    with (root / "meta" / "episodes.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = f"episode_{int(row['episode_index']):06d}.mp4"
            assert key not in lengths, f"duplicate metadata episode: {root.name}/{key}"
            lengths[key] = int(row["length"])
    return lengths


def _video_frame_count(path: Path) -> int:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode or not proc.stdout.strip().isdigit():
        raise AssertionError(
            f"cannot read video frame count for {path}: {proc.stderr.strip()}"
        )
    return int(proc.stdout.strip())


def _gripper_qc(parquet_path: Path) -> list[str]:
    states = np.asarray(
        pq.read_table(parquet_path, columns=["observation.state"])
        .column("observation.state")
        .to_pylist(),
        dtype=np.float32,
    )
    assert states.ndim == 2 and states.shape[1] > R_GRIPPER, parquet_path
    gripper = states[:, R_GRIPPER]
    lo, hi = np.percentile(gripper, [2, 98])
    assert hi - lo > 0.005, f"right gripper never actuates in {parquet_path}"
    closed = gripper < (lo + hi) / 2

    runs = []
    start = None
    for frame, is_closed in enumerate(closed):
        if is_closed and start is None:
            start = frame
        elif not is_closed and start is not None:
            runs.append((start, frame))
            start = None
    if start is not None:
        runs.append((start, len(closed)))

    # The gripper may start closed before opening.  Only later sustained closures
    # are grasp attempts for these standalone skills.
    attempts = [
        run
        for run in runs
        if run[0] > 0 and run[1] - run[0] >= MIN_QC_CLOSE_RUN
    ]
    if not attempts:
        return ["no sustained post-prefix gripper closure"]
    chosen = max(attempts, key=lambda run: run[1] - run[0])
    reasons = []
    if len(attempts) > 1:
        reasons.append(f"multiple grasp attempts {attempts}")
    if chosen[1] != len(closed):
        reasons.append(f"chosen closure {chosen} does not reach episode end {len(closed)}")
    return reasons


def generate(name: str, root: Path) -> dict[str, list[dict]]:
    spec = DATASETS[name]
    task = str(spec["task"])
    assert TASK_RE.fullmatch(task), (name, task)
    lengths = _episode_lengths(root)
    return {
        key: [{"task": task, "start": 0, "end": nframes - 1}]
        for key, nframes in sorted(lengths.items())
    }


def validate(name: str, root: Path, labels: dict[str, list[dict]]) -> None:
    spec = DATASETS[name]
    info = json.loads((root / "meta" / "info.json").read_text())
    assert float(info["fps"]) == 30.0, (name, info.get("fps"))
    assert int(info["total_episodes"]) == spec["episodes"], (name, info)
    assert int(info["total_frames"]) == spec["frames"], (name, info)

    lengths = _episode_lengths(root)
    assert len(lengths) == spec["episodes"], (name, len(lengths), spec["episodes"])
    assert sum(lengths.values()) == spec["frames"], (
        name,
        sum(lengths.values()),
        spec["frames"],
    )
    expected_keys = {
        f"episode_{index:06d}.mp4" for index in range(int(spec["episodes"]))
    }
    assert set(lengths) == expected_keys, (
        name,
        sorted(expected_keys - set(lengths)),
        sorted(set(lengths) - expected_keys),
    )
    assert set(labels) == expected_keys, (name, "label key mismatch")

    exclusion_path = root / "meta" / "train_exclude_episodes.json"
    expected_exclusions = EXPECTED_TRAIN_EXCLUSIONS[name]
    if expected_exclusions:
        assert exclusion_path.is_file(), (
            name,
            f"missing required training exclusion sidecar {exclusion_path}",
        )
        exclusion = json.loads(exclusion_path.read_text())
        assert set(exclusion) == {"version", "reason", "episodes"}, exclusion
        assert exclusion["version"] == 1, exclusion
        assert isinstance(exclusion["reason"], str) and exclusion["reason"].strip()
        assert set(exclusion["episodes"]) == expected_exclusions, exclusion
    else:
        assert not exclusion_path.exists(), (
            name,
            f"unexpected exclusion sidecar in clean dataset: {exclusion_path}",
        )

    parquet_dir = root / "data" / "chunk-000"
    parquet_keys = {path.stem + ".mp4" for path in parquet_dir.glob("episode_*.parquet")}
    assert parquet_keys == expected_keys, (name, "parquet key mismatch")

    qc_flags = {}
    for key, nframes in sorted(lengths.items()):
        parquet_path = parquet_dir / key.replace(".mp4", ".parquet")
        parquet_frames = int(pq.ParquetFile(parquet_path).metadata.num_rows)
        assert parquet_frames == nframes, (name, key, parquet_frames, nframes)
        reasons = _gripper_qc(parquet_path)
        if reasons:
            qc_flags[key] = reasons

        segments = labels[key]
        assert segments == [
            {
                "task": spec["task"],
                "start": 0,
                "end": nframes - 1,
            }
        ], (name, key, segments)

        for camera in CAMERAS:
            camera_dir = (
                root
                / "videos"
                / "chunk-000"
                / f"observation.images.{camera}"
            )
            video_keys = {
                path.name for path in camera_dir.glob("episode_*.mp4")
            }
            assert video_keys == expected_keys, (name, camera, "video key mismatch")
            video_path = camera_dir / key
            video_frames = _video_frame_count(video_path)
            assert video_frames == nframes, (
                name,
                camera,
                key,
                video_frames,
                nframes,
            )

    assert set(qc_flags) == EXPECTED_QC_FLAGS[name], (
        name,
        "gripper QC queue changed",
        qc_flags,
        EXPECTED_QC_FLAGS[name],
    )
    for key, reasons in qc_flags.items():
        print(f"  manual review priority: {name}/{key}: {'; '.join(reasons)}")


def _sidecars(target: Path) -> tuple[Path, Path]:
    return (
        target.with_name("subtask_labels_autogen_backup.json"),
        target.with_name("subtask_labels_review.json"),
    )


def _preflight(targets: list[Path], force: bool, protect_existing: bool) -> None:
    if not protect_existing:
        return
    stale_sidecars = [path for target in targets for path in _sidecars(target) if path.exists()]
    if stale_sidecars:
        details = "\n".join(f"  {path}" for path in stale_sidecars)
        raise SystemExit(
            "refusing to install while manual-review sidecars exist:\n"
            f"{details}\nresolve or archive them explicitly before regenerating labels"
        )
    if force:
        return
    existing = [target for target in targets if target.exists()]
    if existing:
        details = "\n".join(f"  {target}" for target in existing)
        raise SystemExit(
            "refusing to overwrite installed labels that may contain manual edits:\n"
            f"{details}\nrerun with --force only if those edits should be replaced"
        )


def _write_batch(
    outputs: list[tuple[Path, dict]], force: bool, protect_existing: bool
) -> None:
    targets = [target for target, _ in outputs]
    _preflight(targets, force, protect_existing)
    staged = []
    try:
        for target, labels in outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(f".{target.name}.autogen.tmp")
            temp.write_text(json.dumps(labels, indent=2) + "\n")
            staged.append((temp, target))
        _preflight(targets, force, protect_existing)
        for temp, target in staged:
            temp.replace(target)
    finally:
        for temp, _ in staged:
            if temp.exists():
                temp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--only", choices=tuple(DATASETS))
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite installed labels")
    args = parser.parse_args()

    names = [args.only] if args.only else list(DATASETS)
    targets = [
        (
            args.data_root / name / "videos" / "chunk-000" / "subtask_labels.json"
            if args.install
            else Path("/tmp/0901_labels") / f"{name}.json"
        )
        for name in names
    ]
    _preflight(targets, args.force, args.install)

    outputs = []
    for name, target in zip(names, targets):
        root = args.data_root / name
        labels = generate(name, root)
        validate(name, root, labels)
        outputs.append((target, labels))
        print(
            f"{name}: {len(labels)} episodes, {sum(len(v) for v in labels.values())} "
            f"segments, task={DATASETS[name]['task']!r}"
        )

    _write_batch(outputs, args.force, args.install)
    for target, _ in outputs:
        print(f"  wrote {target}" + ("" if args.install else "  (dry run)"))
    if not args.install:
        print("\ndry run only -- rerun with --install to write into the datasets")


if __name__ == "__main__":
    main()
