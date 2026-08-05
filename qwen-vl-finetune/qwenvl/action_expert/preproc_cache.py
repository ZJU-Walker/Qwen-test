"""Per-frame video-preprocessing cache for serving (bit-exact, keyed by dedup frame ids).

The RTC client's history window advances by <=1 frame between requests, yet the video
processor re-runs resize+normalize+patchify on all 10 frames every request (~100 ms CPU,
the dominant templatize cost). This wrapper replaces `processor.video_processor` at SERVE
time only: it caches each frame's resized+NORMALIZED tensor keyed by the client's dedup
frame id, preprocesses only unseen frames, and replays the remaining stages (temporal
padding + patchify) through the processor's own `_preprocess` with the already-done stages
disabled (do_convert_rgb/do_resize/do_rescale/do_normalize=False).

Bit-exactness: every stage runs the processor's OWN methods with its own parameters --
`_prepare_input_videos` (PIL->tensor), `resize` (same smart_resize target, same
interpolation/antialias), `rescale_and_normalize` (same fused op) -- so the assembled
output is torch.equal to a vanilla call (gated by tests/test_video_preproc_cache.py,
which also covers cold-start duplicate windows, +1/+2 advances, and config changes).

Safety: `current_ids=None` (or any input shape surprise) falls through to the vanilla
processor, so warmup, no-history mode, and non-dedup clients are untouched. Any change in
input resolution, window length, or processor size config invalidates the whole cache.
"""

from typing import Optional

import torch

from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize


class CachingVideoWrapper:
    def __init__(self, vp):
        self._vp = vp
        self._cache = {}          # frame_id -> (C, H', W') resized+normalized float tensor
        self._cache_key = None    # (T, H, W, target_h, target_w, size_cfg) -> full invalidation
        self.current_ids: Optional[list] = None  # set by the server around templatize; None -> vanilla
        self.stats = {"hits": 0, "misses": 0, "passthrough": 0}

    # The wrapper must look like the wrapped processor to the rest of the stack
    # (templatize reads vp.temporal_patch_size / vp.fps; Qwen3VLProcessor reads more).
    def __getattr__(self, name):
        return getattr(self._vp, name)

    def reset(self):
        self._cache.clear()
        self._cache_key = None

    def _interpolation(self):
        # Same mapping preprocess() applies to the instance's `resample` setting.
        from transformers.image_utils import PILImageResampling
        try:
            from transformers.image_utils import pil_torch_interpolation_mapping
        except ImportError:  # moved in some versions
            from transformers.image_processing_utils_fast import pil_torch_interpolation_mapping
        return pil_torch_interpolation_mapping[PILImageResampling(self._vp.resample)]

    def __call__(self, videos=None, **kwargs):
        ids = self.current_ids
        frames = None
        if ids is not None and videos is not None and len(videos) == 1:
            frames = videos[0]
            # apply_chat_template batches one level deeper than a direct call: the batch
            # list holds one conversation's video list, which holds THE video (its frame
            # list). Unwrap that extra level -- this is the PRODUCTION calling convention
            # (gated by the apply_chat_template integration test).
            if hasattr(frames, "__len__") and len(frames) == 1 \
                    and isinstance(frames[0], (list, tuple)):
                frames = frames[0]
            if not hasattr(frames, "__len__") or len(frames) != len(ids):
                frames = None
        if frames is None:
            self.stats["passthrough"] += 1
            return self._vp(videos=videos, **kwargs)

        # Geometry from the PIL headers only -- pixels of cached frames are never touched.
        if not hasattr(frames[0], "size") or not isinstance(frames[0].size, tuple):
            self.stats["passthrough"] += 1
            return self._vp(videos=videos, **kwargs)
        W, H = frames[0].size
        T = len(frames)
        size = self._vp.size
        target_h, target_w = smart_resize(
            num_frames=T, height=H, width=W,
            temporal_factor=self._vp.temporal_patch_size,
            factor=self._vp.patch_size * self._vp.merge_size,
            min_pixels=size["shortest_edge"], max_pixels=size["longest_edge"],
        )
        key = (T, H, W, target_h, target_w, size["shortest_edge"], size["longest_edge"])
        if key != self._cache_key:
            self.reset()
            self._cache_key = key

        from transformers.image_processing_utils_fast import SizeDict
        interp = self._interpolation()
        out_frames = []
        seen_this_call = {}
        for fid, frame in zip(ids, frames):
            fid = int(fid)
            if fid in seen_this_call:            # cold-start windows repeat one id
                out_frames.append(seen_this_call[fid])
                self.stats["hits"] += 1
                continue
            hit = self._cache.get(fid)
            if hit is None:
                self.stats["misses"] += 1
                # PIL -> tensor via the processor's OWN front half, for this frame only.
                vids, _ = self._vp._decode_and_sample_videos(
                    [[frame]], video_metadata=None, do_sample_frames=False, sample_indices_fn=None)
                prep = self._vp._prepare_input_videos(videos=vids, input_data_format=None, device=None)[0]
                if isinstance(prep, (list, tuple)):
                    prep = torch.stack(list(prep), dim=0)
                resized = self._vp.resize(prep,
                                          SizeDict(height=target_h, width=target_w),
                                          interpolation=interp)
                # Fused rescale+normalize, same call _preprocess makes (expects (B, T, ...)).
                hit = self._vp.rescale_and_normalize(
                    resized.unsqueeze(0), True, self._vp.rescale_factor,
                    True, tuple(self._vp.image_mean), tuple(self._vp.image_std),
                )[0, 0]
            else:
                self.stats["hits"] += 1
            seen_this_call[fid] = hit
            out_frames.append(hit)
        # Keep exactly the ids the current window can still reference.
        self._cache = dict(seen_this_call)

        stacked = torch.stack(out_frames, dim=0)  # (T, C, H', W') float, normalized
        kwargs.update(do_convert_rgb=False, do_resize=False, do_rescale=False,
                      do_normalize=False, do_sample_frames=False)
        return self._vp(videos=[stacked], **kwargs)

    # Qwen3VLProcessor calls video_processor(...) directly; some paths use .preprocess.
    preprocess = __call__
