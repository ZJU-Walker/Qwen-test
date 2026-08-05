# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///
"""Build the episode-selection manifest for the ABC-130k -> LeRobot conversion.

Selection rule (agreed 2026-07-29): tasks with >= --min-annotated annotated
episodes (annotation.mcap present, i.e. subtask labels exist); per task, up to
--per-task episodes that have BOTH episode.mcap and annotation.mcap, sorted by
episode uuid (uuids are random -> unbiased deterministic sample), train split
first, topped up from val if train alone can't fill the quota. --spares extra
episodes per task are recorded as fallbacks for episodes that fail conversion.

Output: manifest.json in --out-dir:
{
  "repo": "XDOF/ABC-130k",
  "tasks": [
    {"task": "...", "instruction_fallback": "...", "episodes": [
        {"path": "data/train/<task>/episode_<uuid>", "split": "train",
         "size": <episode.mcap bytes>, "role": "primary"|"spare"},
        ...
    ]}
  ]
}

Usage:
  uv run scripts/abc_select_episodes.py \
      --summary-csv /iris/projects/humanoid/ke/openpi_trossen_brian/abc130k_task_summary.csv \
      --out-dir /iris/projects/humanoid/abc_data/abc130k_annot20
"""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi

REPO = "XDOF/ABC-130k"


def list_task_episodes(api: HfApi, split: str, task: str):
    """Episodes of a task that have BOTH mcaps, with episode.mcap sizes."""
    eps = {}
    for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True,
                                path_in_repo=f"data/{split}/{task}"):
        parts = e.path.split("/")
        if len(parts) != 5:
            continue
        d = eps.setdefault(parts[3], {})
        if parts[4] == "episode.mcap":
            d["size"] = e.size
        elif parts[4] == "annotation.mcap":
            d["annotated"] = True
    return [
        {"path": f"data/{split}/{task}/{name}", "split": split, "size": d["size"]}
        for name, d in sorted(eps.items())
        if d.get("annotated") and "size" in d
    ]


def select_for_task(api, task, per_task, spares):
    pool = list_task_episodes(api, "train", task)
    if len(pool) < per_task + spares:
        try:
            pool += list_task_episodes(api, "val", task)
        except Exception:
            pass  # some tasks have no val split
    take = pool[: per_task + spares]
    for i, e in enumerate(take):
        e["role"] = "primary" if i < per_task else "spare"
    return take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--spares", type=int, default=10)
    ap.add_argument("--min-annotated", type=int, default=20)
    args = ap.parse_args()

    tasks = sorted(
        r["task"] for r in csv.DictReader(open(args.summary_csv))
        if int(r["annotated_eps"]) >= args.min_annotated
    )
    print(f"{len(tasks)} tasks pass the >={args.min_annotated}-annotated filter")

    api = HfApi()
    api.whoami()  # fail fast if the token is missing/invalid

    manifest = {"repo": REPO, "per_task": args.per_task, "tasks": []}
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(select_for_task, api, t, args.per_task, args.spares): t
                for t in tasks}
        for i, fut in enumerate(as_completed(futs)):
            t = futs[fut]
            results[t] = fut.result()
            n_primary = sum(1 for e in results[t] if e["role"] == "primary")
            if n_primary < args.per_task:
                print(f"  WARNING {t}: only {n_primary} annotated episodes with both files")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(tasks)} tasks listed")

    for t in tasks:
        manifest["tasks"].append({
            "task": t,
            "instruction_fallback": t.replace("_", " "),
            "episodes": results[t],
        })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    n_primary = sum(1 for t in manifest["tasks"] for e in t["episodes"] if e["role"] == "primary")
    total_gb = sum(e["size"] for t in manifest["tasks"] for e in t["episodes"]
                   if e["role"] == "primary") / 1e9
    print(f"\nwrote {out / 'manifest.json'}: {len(tasks)} tasks, "
          f"{n_primary} primary episodes, {total_gb:.0f} GB raw to download")


if __name__ == "__main__":
    main()
