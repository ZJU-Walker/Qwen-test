# Client patch: send the left wrist camera (for 2-wrist checkpoints)

**Not applied — for your review.** Five small edits to `client_action_expert_rtc_use_me.py`.
The server's `/health` now reports `wrist_cameras`; if it lists two, the server requires a
`wrist_left` upload (400 otherwise). Old 1-wrist checkpoints need none of this (set
`SEND_LEFT_WRIST = False` or serve an old checkpoint — the extra field is simply not sent).

### 1. Config (next to `WRIST_CAM`, line ~71)
```python
WRIST_CAM = "cam_right_wrist"
WRIST_LEFT_CAM = "cam_left_wrist"
SEND_LEFT_WRIST = True   # 2-wrist checkpoints (ee6d run onward); False for older ones
```

### 2. `post_request` signature (line ~181)
```python
def post_request(session, cam_high_frames, frame_ids, wrist_rgb, state7, action_prefix,
                 wrist_left_rgb=None, ...):        # <- add kwarg (keep the rest as-is)
```

### 3. `post_request` files block (line ~202)
```python
        files.append(("wrist", ("wrist.jpg", io.BytesIO(encode_jpeg(wrist_rgb)), "image/jpeg")))
        if wrist_left_rgb is not None:
            files.append(("wrist_left", ("wrist_left.jpg",
                          io.BytesIO(encode_jpeg(wrist_left_rgb)), "image/jpeg")))
```

### 4. `capture()` return (line ~314)
```python
        left = cam_rgb(obs, WRIST_LEFT_CAM) if SEND_LEFT_WRIST else None
        return frames, ids, cam_rgb(obs, WRIST_CAM), left, \
            obs["observation.state"].detach().cpu().numpy().astype(np.float32)
```

### 5. The three call sites (lines ~318-319, ~346-347, ~441-443)
```python
        frames, ids, wrist, wrist_left, state7 = capture()
        ... post_request(session, frames, ids, wrist, state7, <prefix>, wrist_left_rgb=wrist_left, ...)
```
(one extra unpacked value + one extra kwarg at each site; nothing else changes —
history cadence, dedup, RTC prefix handling are all untouched.)

### Notes
- The RTC/action interface is unchanged even for ee6d checkpoints: the client still sends
  7-dim joint state and receives (50, 7) joint actions; all representation conversion is
  server-side. `DELAY_STEPS` stays <= 10 (rtc10 training).
- Latency cost: one extra JPEG encode+upload per request (~1-2 ms + a few KB).
