# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub",
#   "mcap",
#   "mcap-protobuf-support",
#   "numpy",
#   "pyarrow",
#   "av",
# ]
# ///
"""Convert selected ABC-130k MCAP episodes into a LeRobot-v2.1-style dataset
that RobotFlowMatchingDataset (qwenvl/data/robot_data.py) trains on directly.

Reads manifest.json (from abc_select_episodes.py), then per episode:
  download episode.mcap + annotation.mcap -> extract three H.264 camera streams
  -> re-encode fitted inside --box-w x --box-h (aspect kept, NEVER upscaled,
  even dims, yuv420p, 30 fps container timestamps) -> 14-dim state/action
  arrays paired by tick index -> subtask segments (inclusive frame ranges)
  -> staged episode. Raw MCAPs are deleted immediately after conversion.

Finalize pass orders successful episodes by (task, uuid), substitutes spares
for failed primaries (capped at per_task per task), assigns global episode
indices, and writes:
  data/chunk-XXX/episode_XXXXXX.parquet     (schema identical to the trossen ref)
  videos/chunk-XXX/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_XXXXXX.mp4
  videos/chunk-XXX/subtask_labels.json      ({"episode_XXXXXX.mp4": [{"task","start","end"},...]})
  videos/chunk-XXX/instructions.json        ({"episode_XXXXXX.mp4": "roll the ties", ...})
  meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}
  conversion_report.json

State/action layout (matches the trossen 14-dim convention):
  [left-arm 6 joints, left gripper, right-arm 6 joints, right gripper]
Pairing is by tick index: ABC's 8 state/action topics share bit-identical
per-index log_times (verified), and camera message counts equal state counts;
any residual mismatch is truncated to the common minimum (episode fails if
the mismatch exceeds --max-count-slack).

Usage:
  uv run scripts/abc_convert_lerobot.py --manifest <dir>/manifest.json --out <dir> \
      --workers 12                       # full run (resumable; reruns skip staged episodes)
  ... --only-tasks roll_the_ties --limit 3   # smoke test
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FPS = 30
CHUNK_SIZE = 1000
CAMERA_MAP = [
    # (output camera key, ordered list of source topics; first present wins)
    ("cam_high", ["/top-left-camera", "/top-camera"]),
    ("cam_left_wrist", ["/left-wrist-camera"]),
    ("cam_right_wrist", ["/right-wrist-camera"]),
]
STATE_TOPICS = ["/left-arm-state", "/left-ee-state", "/right-arm-state", "/right-ee-state"]
ACTION_TOPICS = ["/left-arm-action", "/left-ee-action", "/right-arm-action", "/right-ee-action"]
DIMS = [6, 1, 6, 1]

HF_SCHEMA_METADATA = (
    '{"info": {"features": {"action": {"feature": {"dtype": "float32", "_type": "Value"}, '
    '"length": 14, "_type": "Sequence"}, "observation.state": {"feature": {"dtype": "float32", '
    '"_type": "Value"}, "length": 14, "_type": "Sequence"}, "timestamp": {"dtype": "float32", '
    '"_type": "Value"}, "frame_index": {"dtype": "int64", "_type": "Value"}, "episode_index": '
    '{"dtype": "int64", "_type": "Value"}, "index": {"dtype": "int64", "_type": "Value"}, '
    '"task_index": {"dtype": "int64", "_type": "Value"}}}}'
)


# ----------------------------------------------------------------------------
# per-episode worker
# ----------------------------------------------------------------------------

def ffprobe_frames(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    w, h, n = int(out[0]), int(out[1]), int(out[2])
    return n, w, h


def floor_indices(source_ts, target_ts):
    """Index of the latest source message at or before each target tick (official
    ABC exporter semantics)."""
    return np.clip(np.searchsorted(source_ts, target_ts, side="right") - 1,
                   0, len(source_ts) - 1)


def encode_resampled(raw_path, codec, out_mp4, needed, box_w, box_h, crf):
    """Decode the raw stream sequentially, emit frame needed[i] at output index i
    (duplicating frames as required), scale into the box, encode h264. Used for
    episodes whose streams run at different rates and must be resampled onto the
    30 Hz tick grid; mirrors the official ABC exporter's encode_aligned."""
    import av
    fmt = "hevc" if "265" in codec or "hevc" in codec.lower() else "h264"
    vf = (f"scale='min({box_w},iw)':'min({box_h},ih)':"
          f"force_original_aspect_ratio=decrease:force_divisible_by=2")
    enc = None
    src_idx, cur = -1, None
    with av.open(str(raw_path), format=fmt) as container:
        dec = container.decode(container.streams.video[0])
        try:
            for want in needed:
                while src_idx < want:
                    frame = next(dec, None)
                    if frame is None:
                        break
                    src_idx += 1
                    cur = frame
                if cur is None:
                    raise RuntimeError("decoder produced no frames")
                arr = cur.to_ndarray(format="rgb24")
                if enc is None:
                    h, w = arr.shape[:2]
                    enc = subprocess.Popen(
                        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
                         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                         "-crf", str(crf), "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", "-threads", "2", str(out_mp4)],
                        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                enc.stdin.write(arr.tobytes())
        finally:
            if enc is not None:
                enc.stdin.close()
                stderr = enc.stderr.read().decode("utf-8", "replace")
                if enc.wait() != 0:
                    raise RuntimeError(f"rawvideo encode failed: {stderr[-1000:]}")


def encode(raw_path, codec, out_mp4, box_w, box_h, crf, head_trim=0, max_frames=None):
    filters = []
    if head_trim > 0:
        # Drop leading frames (camera-vs-tick shift correction) and rebase pts to 0 so the
        # loader's windowed decode (start_pts = t / fps) still lands on row-aligned frames.
        filters.append(f"select='gte(n,{head_trim})',setpts=PTS-STARTPTS")
    filters.append(f"scale='min({box_w},iw)':'min({box_h},ih)':"
                   f"force_original_aspect_ratio=decrease:force_divisible_by=2")
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "hevc" if "265" in codec or "hevc" in codec.lower() else "h264",
           "-r", str(FPS), "-i", str(raw_path)]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-vf", ",".join(filters), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-threads", "2", str(out_mp4)]
    subprocess.run(cmd, check=True, capture_output=True)


def sample_image_stats(mp4, n_samples=4):
    """Per-channel min/max/mean/std over a few evenly spaced frames, values in [0,1]."""
    import av
    frames = []
    with av.open(str(mp4)) as c:
        stream = c.streams.video[0]
        total = stream.frames or 0
        want = set(np.linspace(0, max(total - 1, 0), n_samples, dtype=int).tolist()) if total else {0, 60, 120}
        for i, frame in enumerate(c.decode(stream)):
            if i in want or not total:
                frames.append(frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0)
                if len(frames) >= n_samples:
                    break
    px = np.concatenate([f.reshape(-1, 3) for f in frames], axis=0)
    fmt = lambda v: [[[float(x)]] for x in v]  # LeRobot nests image stats as [3][1][1]
    return {"min": fmt(px.min(0)), "max": fmt(px.max(0)),
            "mean": fmt(px.mean(0)), "std": fmt(px.std(0)), "count": [len(frames)]}


def convert_episode(job):
    """Download + convert one episode into staging. Returns an outcome dict."""
    ep, task, instruction_fallback, cfg = job
    uuid = ep["path"].rsplit("/episode_", 1)[1]
    stage = Path(cfg["staging"]) / uuid
    if (stage / "_DONE").exists():
        return {"uuid": uuid, "task": task, "status": "cached"}
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(dir=cfg["scratch"], prefix=f"abc_{uuid[:8]}_"))
    try:
        from huggingface_hub import hf_hub_download
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory

        mcap_path = hf_hub_download(cfg["repo"], f"{ep['path']}/episode.mcap",
                                    repo_type="dataset", cache_dir=str(tmp / "hf"))
        ann_path = hf_hub_download(cfg["repo"], f"{ep['path']}/annotation.mcap",
                                   repo_type="dataset", cache_dir=str(tmp / "hf"))

        # -------- pass over episode.mcap: stream camera bytes to disk, gather arrays
        cam_files, cam_meta, scalars, tick_times = {}, {}, {}, None
        instruction = None
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            meta_records = {m.name: dict(m.metadata) for m in reader.iter_metadata()}
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            topics = {ch.topic for ch in summary.channels.values()}
            src_topics = {}
            for cam_key, candidates in CAMERA_MAP:
                src = next((t for t in candidates if t in topics), None)
                if src is None:
                    raise RuntimeError(f"missing camera for {cam_key}; topics={sorted(topics)}")
                src_topics[src] = cam_key
                cam_files[cam_key] = open(tmp / f"{cam_key}.raw", "wb")
                cam_meta[cam_key] = {"codec": "h264", "count": 0, "src": src, "log_times": []}
            wanted = set(src_topics) | set(STATE_TOPICS) | set(ACTION_TOPICS) | {"/instruction"}
            for _, channel, message, decoded in reader.iter_decoded_messages(topics=sorted(wanted)):
                t = channel.topic
                if t in src_topics:
                    key = src_topics[t]
                    cam_files[key].write(decoded.data)
                    cam_meta[key]["count"] += 1
                    cam_meta[key]["log_times"].append(message.log_time)
                    fmt = getattr(decoded, "format", "") or "h264"
                    cam_meta[key]["codec"] = fmt
                elif t in STATE_TOPICS or t in ACTION_TOPICS:
                    d = scalars.setdefault(t, {"times": [], "vals": []})
                    d["times"].append(message.log_time)
                    d["vals"].append(np.asarray(decoded.position, dtype=np.float64))
                elif t == "/instruction" and instruction is None:
                    instruction = (getattr(decoded, "data", "") or "").strip()
        for fh in cam_files.values():
            fh.close()

        instruction = instruction or meta_records.get("episode-metadata", {}).get("task_name", "").strip() \
            or instruction_fallback

        # -------- assemble the 30 Hz timeline. Two recording variants exist in ABC:
        # (a) synchronized: all 8 state/action topics tick together at ~30 Hz with
        #     equal counts, cameras 1 frame per tick (verified bit-identical tick
        #     log_times) -> pair by index after a per-camera integer shift, keeping
        #     original frames 1:1.
        # (b) mixed-rate: states ~120+ Hz, actions at another rate, cameras ~30 Hz
        #     with heavy drops and mutually unequal counts -> resample EVERY stream
        #     onto a fresh 30 Hz tick grid by log_time (floor: latest message at or
        #     before each tick; cameras decode-and-duplicate), mirroring the
        #     official ABC exporter.
        for t in STATE_TOPICS + ACTION_TOPICS:
            if t not in scalars or not scalars[t]["vals"]:
                raise RuntimeError(f"missing topic {t}")
        counts = {t: len(scalars[t]["vals"]) for t in STATE_TOPICS + ACTION_TOPICS}
        counts.update({f"cam:{k}": m["count"] for k, m in cam_meta.items()})

        def stack_block(topic, dim, sel):
            arr = np.stack(scalars[topic]["vals"])
            if arr.shape[1] != dim:
                raise RuntimeError(f"{topic} dim {arr.shape[1]} != {dim}")
            return arr[sel]

        alignment = None
        shifts, frac_offsets, tick_trim = {}, {}, 0

        def build_synchronized():
            """Index pairing with per-camera integer shift from fitted linear clocks.
            log_times carry +-17 ms delivery jitter (dt bursts 0..67 ms), so fit
            t = a + b*i per stream; the intercept gap gives the shift (~ -3 frames:
            cameras lead the ticks by ~105 ms, exactly what the official exporter's
            floor alignment produces). Raises on drift/slack violations."""
            nonlocal shifts, frac_offsets, tick_trim
            if max(counts.values()) - min(counts.values()) > cfg["max_count_slack"]:
                raise RuntimeError(f"count mismatch beyond slack: {counts}")
            T = min(counts[t] for t in STATE_TOPICS + ACTION_TOPICS)
            ticks = np.asarray(scalars[STATE_TOPICS[0]]["times"][:T], dtype=np.int64)
            t_ref = int(ticks[0])
            b_tick, a_tick = np.polyfit(np.arange(T, dtype=np.float64),
                                        (ticks - t_ref).astype(np.float64), 1)
            period = b_tick if T > 1 else 1e9 / FPS
            for cam_key, m in cam_meta.items():
                ct = np.asarray(m["log_times"], dtype=np.int64)
                n_cmp = min(len(ct), T)
                b_cam, a_cam = np.polyfit(np.arange(n_cmp, dtype=np.float64),
                                          (ct[:n_cmp] - t_ref).astype(np.float64), 1)
                drift = (b_cam - b_tick) * n_cmp
                if abs(drift) > 0.5 * period:
                    raise RuntimeError(
                        f"{cam_key} clock drifts {drift / 1e6:.1f}ms over the episode")
                rel = (a_cam - a_tick) / period
                shifts[cam_key] = int(round(rel))
                frac_offsets[cam_key] = round(float(rel - round(rel)), 3)
            if max(abs(s) for s in shifts.values()) > cfg["max_count_slack"]:
                raise RuntimeError(f"camera-tick shift beyond slack: {shifts}")
            tick_trim = max(0, max(shifts.values()))      # drop leading ticks if a cam lags
            cam_trim = {k: tick_trim - s for k, s in shifts.items()}
            avail = {k: cam_meta[k]["count"] - cam_trim[k] for k in cam_meta}
            T_final = min([T - tick_trim] + list(avail.values()))
            sel = slice(tick_trim, tick_trim + T_final)
            state = np.concatenate(
                [stack_block(t, d, sel) for t, d in zip(STATE_TOPICS, DIMS)], axis=1)
            action = np.concatenate(
                [stack_block(t, d, sel) for t, d in zip(ACTION_TOPICS, DIMS)], axis=1)
            cam_plan = {k: {"head_trim": cam_trim[k], "avail": avail[k], "needed": None}
                        for k in cam_meta}
            return state, action, ticks[sel], T_final, cam_plan

        def build_resampled():
            """Fresh 30 Hz tick grid; every stream floor-indexed by log_time."""
            tick_ns = int(round(1e9 / FPS))
            firsts, lasts = [], []
            for t in STATE_TOPICS + ACTION_TOPICS:
                firsts.append(scalars[t]["times"][0])
                lasts.append(scalars[t]["times"][-1])
            for m in cam_meta.values():
                firsts.append(m["log_times"][0])
                lasts.append(m["log_times"][-1])
            ticks = np.arange(max(firsts), min(lasts) + 1, tick_ns, dtype=np.int64)
            T_final = len(ticks)
            if T_final < cfg["min_frames"]:
                raise RuntimeError(f"resample grid too short: {T_final} ticks")

            def resampled(topic, dim):
                ts = np.asarray(scalars[topic]["times"], dtype=np.int64)
                return stack_block(topic, dim, floor_indices(ts, ticks))

            state = np.concatenate(
                [resampled(t, d) for t, d in zip(STATE_TOPICS, DIMS)], axis=1)
            action = np.concatenate(
                [resampled(t, d) for t, d in zip(ACTION_TOPICS, DIMS)], axis=1)
            cam_plan = {}
            for cam_key, m in cam_meta.items():
                ct = np.asarray(m["log_times"], dtype=np.int64)
                cam_plan[cam_key] = {"head_trim": 0, "avail": T_final,
                                     "needed": floor_indices(ct, ticks)}
            return state, action, ticks, T_final, cam_plan

        try:
            state, action, tick_times, T_final, cam_plan = build_synchronized()
            alignment = "index_synchronized"
        except RuntimeError as sync_err:
            state, action, tick_times, T_final, cam_plan = build_resampled()
            alignment = f"resampled_30hz ({sync_err})"
        if T_final < cfg["min_frames"]:
            raise RuntimeError(f"too short after alignment: {T_final} frames ({alignment})")

        # -------- subtask segments from annotation.mcap point events
        events = []
        with open(ann_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, channel, message, decoded in reader.iter_decoded_messages():
                if channel.topic == "/subtask-annotation":
                    events.append((message.log_time, (getattr(decoded, "data", "") or "").strip()))
        events.sort()
        starts = [int(np.searchsorted(tick_times, lt, side="left")) for lt, _ in events]
        segments = []
        for i, ((_, text), s) in enumerate(zip(events, starts)):
            e = (starts[i + 1] - 1) if i + 1 < len(starts) else T_final - 1
            s, e = min(s, T_final - 1), min(e, T_final - 1)
            if text and s <= e:
                segments.append({"task": text, "start": s, "end": e})
        if not segments:
            raise RuntimeError("annotation.mcap yielded no usable segments")

        # -------- encode videos and verify EXACT frame counts (must equal T_final).
        # Synchronized path: uncapped first pass; decoded frames must equal message
        # count minus head trim, else the 1-message-=-1-frame assumption is violated
        # (official exporter warns h264 chunks can differ from frames) and the episode
        # fails rather than shipping a silently shifted mapping. Resampled path:
        # decode-and-duplicate onto the tick grid emits exactly T_final frames.
        probe = {}
        for cam_key, m in cam_meta.items():
            out_mp4 = tmp / f"{cam_key}.mp4"
            plan = cam_plan[cam_key]
            if plan["needed"] is not None:
                encode_resampled(tmp / f"{cam_key}.raw", m["codec"], out_mp4,
                                 plan["needed"], cfg["box_w"], cfg["box_h"], cfg["crf"])
                n, w, h = ffprobe_frames(out_mp4)
                if n != T_final:
                    raise RuntimeError(f"{cam_key}: resampled encode gave {n} frames, "
                                       f"want {T_final}")
            else:
                encode(tmp / f"{cam_key}.raw", m["codec"], out_mp4, cfg["box_w"],
                       cfg["box_h"], cfg["crf"], head_trim=plan["head_trim"])
                n, w, h = ffprobe_frames(out_mp4)
                if n != plan["avail"]:
                    raise RuntimeError(
                        f"{cam_key}: decoded {n} frames != {plan['avail']} "
                        f"messages-after-trim (h264 chunks != frames)")
                if n > T_final:
                    encode(tmp / f"{cam_key}.raw", m["codec"], out_mp4, cfg["box_w"],
                           cfg["box_h"], cfg["crf"], head_trim=plan["head_trim"],
                           max_frames=T_final)
                    n, w, h = ffprobe_frames(out_mp4)
                    if n != T_final:
                        raise RuntimeError(
                            f"{cam_key}: re-encode gave {n} frames, want {T_final}")
            probe[cam_key] = (n, w, h)

        # -------- image stats samples (for episodes_stats.jsonl)
        img_stats = {f"observation.images.{k}": sample_image_stats(tmp / f"{k}.mp4")
                     for k in cam_meta}

        # -------- stage atomically
        stage.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stage / "arrays.npz",
                            state=state.astype(np.float32), action=action.astype(np.float32))
        for cam_key in cam_meta:
            shutil.move(str(tmp / f"{cam_key}.mp4"), stage / f"{cam_key}.mp4")
        meta = {
            "uuid": uuid, "task": task, "instruction": instruction, "num_frames": T_final,
            "subtasks": segments, "source_path": ep["path"],
            "resolutions": {k: [probe[k][2], probe[k][1]] for k in cam_meta},  # [h, w]
            "source_topics": {k: m["src"] for k, m in cam_meta.items()},
            "codec_in": {k: m["codec"] for k, m in cam_meta.items()},
            "counts_raw": counts, "alignment": alignment,
            "cam_tick_shifts": shifts, "tick_trim": tick_trim,
            "cam_frac_offsets": frac_offsets, "duration_s": round(T_final / FPS, 2),
            "operator_id": meta_records.get("episode-metadata", {}).get("operator_id"),
            "image_stats": img_stats,
        }
        (stage / "meta.json").write_text(json.dumps(meta))
        (stage / "_DONE").touch()
        return {"uuid": uuid, "task": task, "status": "ok", "frames": T_final,
                "shifts": shifts, "counts": counts, "secs": round(time.time() - t0, 1)}
    except Exception as exc:
        shutil.rmtree(stage, ignore_errors=True)
        err = f"{type(exc).__name__}: {exc}"
        stderr = getattr(exc, "stderr", None)  # CalledProcessError from ffmpeg/ffprobe
        if stderr:
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            err += " | stderr: " + stderr.strip()[-2000:]
        return {"uuid": uuid, "task": task, "status": "failed", "error": err,
                "trace": traceback.format_exc(limit=3)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# finalize: staged episodes -> LeRobot layout
# ----------------------------------------------------------------------------

def vector_stats(arr):
    return {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
            "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
            "count": [int(arr.shape[0])]}


def scalar_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    return {"min": [float(a.min())], "max": [float(a.max())],
            "mean": [float(a.mean())], "std": [float(a.std())],
            "count": [int(a.shape[0])]}


def finalize(out_root, staging, manifest, report):
    import pyarrow as pa
    import pyarrow.parquet as pq

    per_task = manifest.get("per_task", 20)
    chosen = []  # (task, uuid, stage_dir, meta)
    for tinfo in manifest["tasks"]:
        picked = 0
        for ep in tinfo["episodes"]:  # primaries first, then spares (manifest order)
            if picked >= per_task:
                break
            uuid = ep["path"].rsplit("/episode_", 1)[1]
            stage = staging / uuid
            if (stage / "_DONE").exists():
                meta = json.loads((stage / "meta.json").read_text())
                chosen.append((tinfo["task"], uuid, stage, meta))
                picked += 1
        if picked < per_task:
            report["finalize_warnings"].append(
                f"{tinfo['task']}: only {picked}/{per_task} episodes converted")
    if not chosen:
        raise RuntimeError("finalize: 0 episodes chosen (bad filters?); refusing to write")

    # Round-robin over tasks (rank-within-task major) so a fractional loader
    # train_split (prefix of sorted episodes) cuts WITHIN every task instead of
    # silently dropping the alphabetically-last tasks wholesale.
    chosen.sort(key=lambda c: (c[0], c[1]))
    rank, keyed = {}, []
    for c in chosen:
        r = rank.get(c[0], 0)
        rank[c[0]] = r + 1
        keyed.append((r, c[0], c))
    chosen = [c for _, _, c in sorted(keyed, key=lambda k: (k[0], k[1]))]

    # Purge outputs from any previous (possibly larger) finalize: the training loader
    # enumerates by glob, so stale episode files would silently train with empty
    # subtask/instruction labels. Videos are hard links into _staging, so re-linking
    # is cheap; parquets are rewritten below anyway.
    for sub in ("data", "videos"):
        shutil.rmtree(out_root / sub, ignore_errors=True)

    tasks_sorted = sorted({c[0] for c in chosen})
    instr_by_task = {}
    for task, _, _, meta in chosen:
        instr_by_task.setdefault(task, meta["instruction"])
    task_index = {t: i for i, t in enumerate(tasks_sorted)}

    schema = pa.schema(
        [pa.field("action", pa.list_(pa.float32(), 14)),
         pa.field("observation.state", pa.list_(pa.float32(), 14)),
         pa.field("timestamp", pa.float32()),
         pa.field("frame_index", pa.int64()),
         pa.field("episode_index", pa.int64()),
         pa.field("index", pa.int64()),
         pa.field("task_index", pa.int64())],
        metadata={"huggingface": HF_SCHEMA_METADATA},
    )

    global_index = 0
    episodes_jsonl, episodes_stats_jsonl = [], []
    subtask_labels_by_chunk, instructions_by_chunk = {}, {}
    res_counter = {}
    for ep_idx, (task, uuid, stage, meta) in enumerate(chosen):
        chunk = ep_idx // CHUNK_SIZE
        chunk_name = f"chunk-{chunk:03d}"
        stem = f"episode_{ep_idx:06d}"
        T = meta["num_frames"]
        arrays = np.load(stage / "arrays.npz")
        state, action = arrays["state"], arrays["action"]
        assert state.shape == (T, 14) and action.shape == (T, 14), (uuid, state.shape, T)

        data_dir = out_root / "data" / chunk_name
        data_dir.mkdir(parents=True, exist_ok=True)
        cols = {
            "action": pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1)), 14),
            "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1)), 14),
            "timestamp": pa.array((np.arange(T) / FPS).astype(np.float32)),
            "frame_index": pa.array(np.arange(T, dtype=np.int64)),
            "episode_index": pa.array(np.full(T, ep_idx, dtype=np.int64)),
            "index": pa.array(np.arange(global_index, global_index + T, dtype=np.int64)),
            "task_index": pa.array(np.full(T, task_index[task], dtype=np.int64)),
        }
        pq.write_table(pa.Table.from_arrays([cols[f.name] for f in schema], schema=schema),
                       data_dir / f"{stem}.parquet")

        for cam_key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            vdir = out_root / "videos" / chunk_name / f"observation.images.{cam_key}"
            vdir.mkdir(parents=True, exist_ok=True)
            target = vdir / f"{stem}.mp4"
            if target.exists():
                target.unlink()
            os.link(stage / f"{cam_key}.mp4", target)

        subtask_labels_by_chunk.setdefault(chunk_name, {})[f"{stem}.mp4"] = meta["subtasks"]
        instructions_by_chunk.setdefault(chunk_name, {})[f"{stem}.mp4"] = meta["instruction"]

        episodes_jsonl.append({"episode_index": ep_idx, "tasks": [meta["instruction"]],
                               "length": T})
        st = {"action": vector_stats(action), "observation.state": vector_stats(state),
              "timestamp": scalar_stats(np.arange(T) / FPS),
              "frame_index": scalar_stats(np.arange(T)),
              "episode_index": scalar_stats([ep_idx] * T),
              "index": scalar_stats(np.arange(global_index, global_index + T)),
              "task_index": scalar_stats([task_index[task]] * T)}
        st.update(meta["image_stats"])
        episodes_stats_jsonl.append({"episode_index": ep_idx, "stats": st})

        hw = tuple(meta["resolutions"]["cam_high"])
        res_counter[hw] = res_counter.get(hw, 0) + 1
        global_index += T

    n_eps = len(chosen)
    main_hw = max(res_counter, key=res_counter.get) if res_counter else (480, 640)

    def video_feature(h, w):
        return {"dtype": "video", "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": float(FPS), "video.height": h, "video.width": w,
                         "video.channels": 3, "video.codec": "h264",
                         "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                         "has_audio": False}}

    joint_names = [f"{side}_joint_{i}" for side in ("left", "right") for i in range(7)]
    info = {
        "codebase_version": "v2.1",
        "robot_type": "i2rt_yam_bimanual",
        "source_dataset": manifest["repo"],
        "total_episodes": n_eps,
        "total_frames": int(global_index),
        "total_tasks": len(tasks_sorted),
        "total_videos": n_eps * 3,
        "total_chunks": (n_eps + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [14], "names": joint_names},
            "observation.state": {"dtype": "float32", "shape": [14], "names": joint_names},
            "observation.images.cam_high": video_feature(*main_hw),
            "observation.images.cam_left_wrist": video_feature(480, 640),
            "observation.images.cam_right_wrist": video_feature(480, 640),
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in tasks_sorted:
            f.write(json.dumps({"task_index": task_index[t], "task": instr_by_task[t]}) + "\n")
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for row in episodes_jsonl:
            f.write(json.dumps(row) + "\n")
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for row in episodes_stats_jsonl:
            f.write(json.dumps(row) + "\n")
    for chunk_name, labels in subtask_labels_by_chunk.items():
        vdir = out_root / "videos" / chunk_name
        (vdir / "subtask_labels.json").write_text(json.dumps(labels, indent=1))
        (vdir / "instructions.json").write_text(
            json.dumps(instructions_by_chunk[chunk_name], indent=1))

    report["finalized"] = {
        "episodes": n_eps, "frames": int(global_index), "tasks": len(tasks_sorted),
        "hours": round(global_index / FPS / 3600, 2),
        "cam_high_resolutions": {f"{h}x{w}": c for (h, w), c in sorted(res_counter.items())},
    }
    return report


# ----------------------------------------------------------------------------

def save_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, report_path)  # atomic: readers see old or new, never partial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--box-w", type=int, default=960)
    ap.add_argument("--box-h", type=int, default=600)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--max-count-slack", type=int, default=5)
    ap.add_argument("--min-frames", type=int, default=90)
    ap.add_argument("--only-tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="episodes per task cap (smoke tests)")
    ap.add_argument("--scratch",
                    default=os.environ.get("ABC_SCRATCH", f"/tmp/abc_convert_{os.getuid()}"))
    ap.add_argument("--skip-convert", action="store_true", help="finalize only")
    ap.add_argument("--no-finalize", action="store_true",
                    help="convert/stage only (multi-node shards); run a final "
                         "--skip-convert pass afterwards to write the dataset")
    ap.add_argument("--report-name", default="conversion_report.json",
                    help="per-shard report filename (avoids clobbering when "
                         "multiple shard jobs share one --out)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt episodes recorded as failed in a previous run")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out_root = Path(args.out)
    staging = out_root / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    for stale in scratch.glob("abc_*"):  # leftovers from a killed previous run
        shutil.rmtree(stale, ignore_errors=True)

    if args.only_tasks:
        manifest["tasks"] = [t for t in manifest["tasks"] if t["task"] in set(args.only_tasks)]
    if args.limit:
        manifest["per_task"] = min(manifest.get("per_task", 20), args.limit)
        for t in manifest["tasks"]:
            t["episodes"] = (
                [e for e in t["episodes"] if e["role"] == "primary"][: args.limit]
                + [e for e in t["episodes"] if e["role"] == "spare"][:3])

    cfg = {"repo": manifest["repo"], "staging": str(staging), "scratch": str(scratch),
           "box_w": args.box_w, "box_h": args.box_h, "crf": args.crf,
           "max_count_slack": args.max_count_slack, "min_frames": args.min_frames}

    report_path = out_root / args.report_name
    report = {"outcomes": {}, "finalize_warnings": []}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            print("WARN: conversion_report.json corrupt; starting fresh report", flush=True)
            report = {}
        report.setdefault("outcomes", {})
        report["finalize_warnings"] = []

    if not args.skip_convert:
        def job_uuid(j):
            return j[0]["path"].rsplit("/episode_", 1)[1]

        def previously_failed(uuid):
            prev = report["outcomes"].get(uuid)
            return bool(prev and prev.get("status") == "failed") and not args.retry_failed

        # Primaries first; spares of a task run only if a primary of that task failed.
        primary_jobs, spare_pool, skipped_failed_tasks = [], {}, []
        for t in manifest["tasks"]:
            spare_pool[t["task"]] = [e for e in t["episodes"] if e["role"] == "spare"]
            for e in (e for e in t["episodes"] if e["role"] == "primary"):
                j = (e, t["task"], t["instruction_fallback"], cfg)
                if previously_failed(job_uuid(j)):
                    skipped_failed_tasks.append(t["task"])  # still triggers spare rounds
                else:
                    primary_jobs.append(j)
        if skipped_failed_tasks:
            print(f"skipping {len(skipped_failed_tasks)} previously-failed episodes "
                  f"(--retry-failed to re-attempt)", flush=True)

        def run_jobs(jobs, tag):
            from concurrent.futures.process import BrokenProcessPool
            failed_tasks = []
            done = 0
            pending = list(jobs)
            handled = set()  # job uuids with a recorded outcome this call
            for attempt in range(4):
                if not pending:
                    break
                if attempt:
                    print(f"[{tag}] pool broke; retrying {len(pending)} jobs "
                          f"(attempt {attempt + 1}/4)", flush=True)
                try:
                    with ProcessPoolExecutor(max_workers=args.workers) as ex:
                        futs = {ex.submit(convert_episode, j): j for j in pending}
                        for fut in as_completed(futs):
                            j = futs[fut]
                            try:
                                r = fut.result()
                            except BrokenProcessPool:
                                raise
                            except Exception as exc:  # worker raised (not crashed)
                                r = {"uuid": job_uuid(j), "task": j[1],
                                     "status": "failed", "error": repr(exc)}
                            handled.add(job_uuid(j))
                            report["outcomes"][r["uuid"]] = r
                            done += 1
                            if r["status"] == "failed":
                                failed_tasks.append(r["task"])
                                print(f"[{tag} {done}/{len(jobs)}] FAIL "
                                      f"{r['task']}/{r['uuid'][:8]}: {r['error']}", flush=True)
                            else:
                                print(f"[{tag} {done}/{len(jobs)}] {r['status']} "
                                      f"{r['task']}/{r['uuid'][:8]} "
                                      f"({r.get('frames', '?')}f {r.get('secs', '?')}s)",
                                      flush=True)
                            if done % 25 == 0:
                                save_report(report_path, report)
                    pending = []
                except BrokenProcessPool:
                    # completed-but-unconsumed futures are lost; their _DONE staging
                    # markers make the retry near-free for episodes that finished.
                    pending = [j for j in pending if job_uuid(j) not in handled]
            else:
                if pending:
                    raise RuntimeError(
                        f"[{tag}] process pool kept breaking; {len(pending)} jobs unattempted")
            save_report(report_path, report)
            return failed_tasks

        failed_tasks = run_jobs(primary_jobs, "primary") + skipped_failed_tasks
        # one spare per failure, then keep going while spares remain and failures persist
        round_num = 0
        while failed_tasks and round_num < 3:
            round_num += 1
            jobs = []
            for task in failed_tasks:
                pool = spare_pool.get(task, [])
                e = None
                while pool:  # skip spares already recorded as failed
                    cand = pool.pop(0)
                    if not previously_failed(cand["path"].rsplit("/episode_", 1)[1]):
                        e = cand
                        break
                if e is not None:
                    tinfo = next(t for t in manifest["tasks"] if t["task"] == task)
                    jobs.append((e, task, tinfo["instruction_fallback"], cfg))
                else:
                    print(f"[spares] {task}: no spares left", flush=True)
            if not jobs:
                break
            failed_tasks = run_jobs(jobs, f"spare-r{round_num}")

    if args.no_finalize:
        save_report(report_path, report)
        n_ok = sum(1 for r in report["outcomes"].values() if r["status"] in ("ok", "cached"))
        n_failed = sum(1 for r in report["outcomes"].values() if r["status"] == "failed")
        print(f"shard done (no finalize): {n_ok} ok, {n_failed} failed", flush=True)
        return 1 if n_failed else 0

    # Final pass: merge outcomes from any shard reports so the consolidated
    # conversion_report.json documents every episode's conversion.
    for shard_report in sorted(out_root.glob("conversion_report_*.json")):
        try:
            report["outcomes"].update(json.loads(shard_report.read_text()).get("outcomes", {}))
        except json.JSONDecodeError:
            print(f"WARN: unreadable shard report {shard_report}", flush=True)

    report = finalize(out_root, staging, manifest, report)
    save_report(report_path, report)
    print(json.dumps(report["finalized"], indent=2))
    n_failed = sum(1 for r in report["outcomes"].values() if r["status"] == "failed")
    print(f"failures: {n_failed}; warnings: {len(report['finalize_warnings'])}")
    for w in report["finalize_warnings"]:
        print("  WARN", w)
    return 1 if (report["finalized"]["episodes"] == 0 or report["finalize_warnings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
