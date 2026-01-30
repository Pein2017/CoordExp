import contextlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Literal

import torch
import torch.nn.functional as F

from swift.trainers.rlhf_trainer.utils import replace_assistant_response_with_ids

from .rollout_matching_sft import (
    GTObject,
    RolloutMatchingSFTTrainer,
    _PendingTrainRolloutLog,
    _WindowedMicroBatch,
    _find_desc_value_token_positions,
    _fingerprint_raw_micro_batch,
    _points_from_coord_tokens,
    _serialize_append_fragment,
    hungarian_match_maskiou,
    parse_rollout_for_matching,
)
from ..common.geometry.coord_utils import decode_coord


_OBJECT_KEY_RE = re.compile(r"^object_(\d+)$")


@dataclass
class _PendingStage2Log:
    """Accumulate Stage-2 AB logs across micro-batches for one optimizer step."""

    n_micro: int = 0
    sums: Dict[str, float] = field(default_factory=dict)

    def add(self, metrics: Mapping[str, float]) -> None:
        self.n_micro += 1
        for k, v in metrics.items():
            try:
                self.sums[str(k)] = float(self.sums.get(str(k), 0.0)) + float(v)
            except Exception:
                continue

    def finalize(self) -> Dict[str, float]:
        if self.n_micro <= 0:
            return {}
        out: Dict[str, float] = {}
        for k, v in self.sums.items():
            # Average losses/ratios; keep counters as totals.
            if (
                k.startswith("loss/")
                or k.startswith("stage2/channel_")
                or k.startswith("rollout/")
            ):
                out[k] = float(v) / float(self.n_micro)
            else:
                out[k] = float(v)
        return out


def _expectation_decode_coords(
    *,
    coord_logits: torch.Tensor,  # [N, 1000]
    temperature: float,
) -> torch.Tensor:
    """Decode normalized coordinates c_hat = E[k]/999 from coord-bin logits."""
    if coord_logits.numel() == 0:
        return coord_logits.new_zeros((0,), dtype=torch.float32)
    temp = float(temperature)
    if temp <= 0:
        raise ValueError(f"temperature must be > 0; got {temp}")
    probs = torch.softmax(coord_logits.float() / temp, dim=-1)
    bins = torch.arange(0, 1000, device=coord_logits.device, dtype=torch.float32)
    exp = (probs * bins).sum(dim=-1)
    return exp / 999.0


def _bbox_l1_giou_loss(
    *,
    pred_xyxy: torch.Tensor,  # [N,4] normalized
    gt_xyxy: torch.Tensor,  # [N,4] normalized
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (l1_mean, giou_mean) for normalized bboxes.

    Pred boxes are canonicalized and clipped to [0,1] before GIoU.
    """
    if pred_xyxy.numel() == 0:
        z = pred_xyxy.new_tensor(0.0)
        return z, z

    px1, py1, px2, py2 = pred_xyxy.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(dim=-1)

    # L1 on raw coords (but still normalized).
    l1 = (pred_xyxy - gt_xyxy).abs().mean()

    # Canonicalize + clip pred for stable IoU.
    x1 = torch.minimum(px1, px2).clamp(0.0, 1.0)
    y1 = torch.minimum(py1, py2).clamp(0.0, 1.0)
    x2 = torch.maximum(px1, px2).clamp(0.0, 1.0)
    y2 = torch.maximum(py1, py2).clamp(0.0, 1.0)

    gx1 = gx1.clamp(0.0, 1.0)
    gy1 = gy1.clamp(0.0, 1.0)
    gx2 = gx2.clamp(0.0, 1.0)
    gy2 = gy2.clamp(0.0, 1.0)

    inter_x1 = torch.maximum(x1, gx1)
    inter_y1 = torch.maximum(y1, gy1)
    inter_x2 = torch.minimum(x2, gx2)
    inter_y2 = torch.minimum(y2, gy2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area_p = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    area_g = (gx2 - gx1).clamp(min=0.0) * (gy2 - gy1).clamp(min=0.0)
    union = (area_p + area_g - inter).clamp(min=eps)
    iou = inter / union

    enc_x1 = torch.minimum(x1, gx1)
    enc_y1 = torch.minimum(y1, gy1)
    enc_x2 = torch.maximum(x2, gx2)
    enc_y2 = torch.maximum(y2, gy2)
    enc_area = (enc_x2 - enc_x1).clamp(min=0.0) * (enc_y2 - enc_y1).clamp(min=0.0)
    enc_area = enc_area.clamp(min=eps)

    giou = iou - (enc_area - union) / enc_area
    giou_loss = (1.0 - giou).mean()

    giou_loss = torch.nan_to_num(giou_loss, nan=0.0, posinf=0.0, neginf=0.0)
    l1 = torch.nan_to_num(l1, nan=0.0, posinf=0.0, neginf=0.0)
    return l1, giou_loss


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Return start index of the last occurrence of needle in haystack."""
    if not needle:
        raise ValueError("needle is empty")
    if len(needle) > len(haystack):
        raise ValueError("needle longer than haystack")
    last = -1
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            last = i
    if last < 0:
        raise ValueError("unable to locate assistant token ids inside encoded input_ids")
    return int(last)


def _coerce_bbox_bins(values: Any) -> Optional[List[int]]:
    if not isinstance(values, Sequence):
        return None
    if len(values) != 4:
        return None
    out: List[int] = []
    for v in values:
        k = decode_coord(v)
        if k is None:
            return None
        out.append(int(k))
    x1, y1, x2, y2 = out
    if x2 < x1 or y2 < y1:
        return None
    return out


def _extract_gt_bboxonly(sample: Mapping[str, Any]) -> List[GTObject]:
    payload = sample.get("assistant_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("stage2-ab requires assistant_payload in each sample")

    objs: List[GTObject] = []
    for key, entry in payload.items():
        if not isinstance(key, str):
            continue
        m = _OBJECT_KEY_RE.match(key)
        if not m:
            continue
        idx = int(m.group(1))
        if not isinstance(entry, Mapping):
            raise ValueError(f"assistant_payload[{key}] must be a mapping")

        # Enforce exactly one geometry key, bbox-only.
        geom_keys = [
            k
            for k in ("bbox_2d", "poly")
            if k in entry and entry.get(k) is not None
        ]
        if len(geom_keys) != 1:
            raise ValueError(
                f"bbox-only v1 requires exactly one geometry field per object; got keys={geom_keys} for {key}"
            )
        if geom_keys[0] != "bbox_2d":
            raise ValueError(
                f"bbox-only v1 requires filtering out polygons upstream; found {geom_keys[0]} in {key}"
            )

        pts = _coerce_bbox_bins(entry.get("bbox_2d"))
        if pts is None:
            raise ValueError(f"invalid bbox_2d for {key}; expected 4 bins in [0,999] and ordered xyxy")

        desc = entry.get("desc", "")
        objs.append(
            GTObject(
                index=int(idx),
                geom_type="bbox_2d",
                points_norm1000=list(pts),
                desc=str(desc) if desc is not None else "",
            )
        )

    objs.sort(key=lambda o: int(o.index))
    if not objs:
        raise ValueError("no valid GT objects found in assistant_payload")
    return objs


def _bbox_groups_from_token_ids(
    *,
    token_ids: Sequence[int],
    coord_id_set: set[int],
    gt_objs: Sequence[GTObject],
) -> List[List[int]]:
    coord_pos = [i for i, tid in enumerate(token_ids) if int(tid) in coord_id_set]
    exp = int(len(gt_objs)) * 4
    if len(coord_pos) != exp:
        raise ValueError(
            f"unexpected coord-token count in teacher-forced ids: got={len(coord_pos)} expected={exp}"
        )
    groups: List[List[int]] = []
    for i in range(0, len(coord_pos), 4):
        groups.append([int(p) for p in coord_pos[i : i + 4]])
    return groups


class Stage2ABTrainingTrainer(RolloutMatchingSFTTrainer):
    """Stage-2 AB trainer: Channel-A iterative soft self-context + Channel-B rollout matching.

    This is bbox-only v1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stage2_pending_train_logs: Dict[int, _PendingStage2Log] = {}
        self._stage2_channel_override: Optional[str] = None

    def _maybe_seed_hf_sampling_rollout(
        self, *, seed_base: int, backend: str, do_sample: bool
    ) -> bool:
        """Deprecated helper retained for backward compatibility within this file.

        Prefer `_hf_sampling_seed_context(...)` so the training RNG state is restored
        after HF rollout generation.
        """
        if str(backend).lower() != "hf" or not bool(do_sample):
            return False
        # Best-effort seed (state is *not* restored here).
        sb = int(seed_base)
        try:
            from transformers.trainer_utils import set_seed

            set_seed(sb)
            return True
        except Exception:
            pass

        try:
            import random

            random.seed(sb)
        except Exception:
            pass
        try:
            import numpy as np

            np.random.seed(sb)
        except Exception:
            pass
        try:
            torch.manual_seed(sb)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sb)
        except Exception:
            pass
        return True

    @contextlib.contextmanager
    def _hf_sampling_seed_context(
        self, *, seed_base: int, backend: str, do_sample: bool
    ):
        """Context manager: seed HF stochastic rollout generation without perturbing training RNG.

        - If backend!=hf or do_sample==False, yields False and does nothing.
        - Otherwise saves python/numpy/torch(+cuda) RNG state, seeds deterministically,
          yields True, then restores the saved RNG states.

        This satisfies the stage2-ab spec requirement (seed stochastic HF rollouts)
        while avoiding unintended effects on later training randomness.
        """
        if str(backend).lower() != "hf" or not bool(do_sample):
            yield False
            return

        sb = int(seed_base)

        # Save RNG states.
        py_state = None
        np_state = None
        torch_state = None
        cuda_state = None

        try:
            import random

            py_state = random.getstate()
        except Exception:
            py_state = None

        try:
            import numpy as np

            np_state = np.random.get_state()
        except Exception:
            np_state = None

        try:
            torch_state = torch.get_rng_state()
        except Exception:
            torch_state = None

        try:
            if torch.cuda.is_available():
                cuda_state = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_state = None

        # Seed (prefer Transformers helper; fall back to best-effort seeding).
        try:
            from transformers.trainer_utils import set_seed

            set_seed(sb)
        except Exception:
            try:
                import random

                random.seed(sb)
            except Exception:
                pass
            try:
                import numpy as np

                np.random.seed(sb)
            except Exception:
                pass
            try:
                torch.manual_seed(sb)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sb)
            except Exception:
                pass

        try:
            yield True
        finally:
            # Restore RNG states.
            if py_state is not None:
                try:
                    import random

                    random.setstate(py_state)
                except Exception:
                    pass
            if np_state is not None:
                try:
                    import numpy as np

                    np.random.set_state(np_state)
                except Exception:
                    pass
            if torch_state is not None:
                try:
                    torch.set_rng_state(torch_state)
                except Exception:
                    pass
            if cuda_state is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(cuda_state)
                except Exception:
                    pass

    def _ab_cfg(self) -> Mapping[str, Any]:
        cfg = getattr(self, "stage2_ab_cfg", None)
        return cfg if isinstance(cfg, Mapping) else {}

    def _ab_get(self, key: str, default: Any) -> Any:
        try:
            cfg = self._ab_cfg()
            if key in cfg:
                return cfg[key]
        except Exception:
            pass
        return default

    def _ab_schedule_pattern(self) -> List[str]:
        cfg = self._ab_cfg()
        sched = cfg.get("schedule", {})
        pattern = None
        if isinstance(sched, Mapping):
            pattern = sched.get("pattern")
        if not isinstance(pattern, list) or not pattern:
            return ["A"]
        out: List[str] = []
        for x in pattern:
            s = str(x).strip().upper()
            if s in {"A", "B"}:
                out.append(s)
        return out if out else ["A"]

    def _stage2_channel_for_step(self, global_step: int) -> Literal["A", "B"]:
        if isinstance(self._stage2_channel_override, str) and self._stage2_channel_override in {
            "A",
            "B",
        }:
            return "A" if self._stage2_channel_override == "A" else "B"
        pattern = self._ab_schedule_pattern()
        s = int(global_step)
        return "A" if pattern[s % len(pattern)] == "A" else "B"

    def log(self, logs: Dict[str, float]) -> None:
        try:
            if (
                isinstance(logs, dict)
                and "loss" in logs
                and not any(str(k).startswith("eval_") for k in logs.keys())
            ):
                step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
                pending = self._stage2_pending_train_logs.pop(step, None)
                if pending is not None:
                    logs.update(pending.finalize())
        except Exception:
            pass
        return super().log(logs)

    def training_step(self, model, inputs, *args, **kwargs):
        # When using identity collator, `inputs` is a list of raw samples.
        if not isinstance(inputs, list):
            return super(RolloutMatchingSFTTrainer, self).training_step(
                model, inputs, *args, **kwargs
            )

        # Buffering is a training-only optimization.
        if not bool(getattr(model, "training", False)):
            self._rm_rollout_buffer = None
            ch = self._stage2_channel_for_step(
                int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            )
            prev = self._stage2_channel_override
            self._stage2_channel_override = ch
            try:
                prepared = self._prepare_batch_inputs(inputs)
            finally:
                self._stage2_channel_override = prev
            return super(RolloutMatchingSFTTrainer, self).training_step(
                model, prepared, *args, **kwargs
            )

        gs = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        ch_sched = self._stage2_channel_for_step(gs)

        buf = self._maybe_init_rollout_buffer()
        if buf is not None:
            buf.on_micro_step_start(global_step=gs)

        reuse_active = bool(
            buf is not None
            and getattr(buf, "window_step0", None) is not None
            and int(getattr(buf, "completed_optimizer_steps", 0) or 0) > 0
            and int(getattr(buf, "m_steps", 1) or 1) > 1
        )

        if reuse_active:
            ch_sched = "B"

        if ch_sched == "A":
            prev = self._stage2_channel_override
            self._stage2_channel_override = "A"
            try:
                prepared = self._prepare_batch_inputs(inputs)
            finally:
                self._stage2_channel_override = prev
            return super(RolloutMatchingSFTTrainer, self).training_step(
                model, prepared, *args, **kwargs
            )

        if buf is None:
            prev = self._stage2_channel_override
            self._stage2_channel_override = "B"
            try:
                packing_enabled = bool(self._packing_enabled())
                if (
                    packing_enabled
                    and self._post_rollout_pack_scope() == "window"
                    and isinstance(inputs, _WindowedMicroBatch)
                ):
                    win = inputs.rm_window
                    idx = int(getattr(inputs, "rm_window_idx", 0) or 0)

                    def _build_all() -> List[Dict[str, Any]]:
                        prev2 = self._stage2_channel_override
                        self._stage2_channel_override = "B"
                        try:
                            return self._prepare_window_packed_batches(
                                window_raw_micro_batches=win.raw_micro_batches,
                                global_step=gs,
                            )
                        finally:
                            self._stage2_channel_override = prev2

                    prepared = win.get_prepared(idx=idx, build_all_prepared=_build_all)
                else:
                    prepared = self._prepare_batch_inputs(inputs)
            finally:
                self._stage2_channel_override = prev
            return super(RolloutMatchingSFTTrainer, self).training_step(
                model, prepared, *args, **kwargs
            )

        raw_fp = _fingerprint_raw_micro_batch(inputs)

        def _build_prepared() -> Dict[str, Any]:
            packing_enabled = bool(self._packing_enabled())
            if (
                packing_enabled
                and self._post_rollout_pack_scope() == "window"
                and isinstance(inputs, _WindowedMicroBatch)
            ):
                win = inputs.rm_window
                idx = int(getattr(inputs, "rm_window_idx", 0) or 0)

                def _build_all() -> List[Dict[str, Any]]:
                    prev2 = self._stage2_channel_override
                    self._stage2_channel_override = "B"
                    try:
                        return self._prepare_window_packed_batches(
                            window_raw_micro_batches=win.raw_micro_batches,
                            global_step=gs,
                        )
                    finally:
                        self._stage2_channel_override = prev2

                prepared2 = win.get_prepared(idx=idx, build_all_prepared=_build_all)
                return prepared2

            prev2 = self._stage2_channel_override
            self._stage2_channel_override = "B"
            try:
                prepared2 = self._prepare_batch_inputs(inputs)
            finally:
                self._stage2_channel_override = prev2
            if not isinstance(prepared2, dict):
                raise ValueError("prepared batch must be a dict")
            return prepared2

        batch, reused = buf.select_batch(
            global_step=gs, raw_fp=raw_fp, build_prepared=_build_prepared
        )

        try:
            bm = batch.get("_rollout_matching_batch_metrics")
            if not isinstance(bm, dict):
                bm = {}
                batch["_rollout_matching_batch_metrics"] = bm

            bm["rollout/buffer_reuse"] = float(1.0 if reused else 0.0)
            bm["rollout/buffer_window_step0"] = float(
                -1 if buf.window_step0 is None else int(buf.window_step0)
            )
            bm["rollout/buffer_completed_steps"] = float(buf.completed_optimizer_steps)

            if reused:
                bm["time/rollout_generate_s"] = 0.0
                bm["time/rollout_parse_match_s"] = 0.0
                bm["time/rollout_teacher_encode_s"] = 0.0
                bm["time/post_rollout_pack_s"] = 0.0
        except Exception:
            pass

        return super(RolloutMatchingSFTTrainer, self).training_step(
            model, batch, *args, **kwargs
        )


    def _prepare_window_packed_batches(
        self,
        *,
        window_raw_micro_batches: List[List[Mapping[str, Any]]],
        global_step: int,
    ) -> List[Dict[str, Any]]:
        """Build packed prepared batches for one full accumulation window.

        Stage2-AB specialization: batch rollout generation across the whole window so
        rollout request batch size can be > learner microbatch size.

        - Learner microbatch stays small (typically 1 packed sequence per backward).
        - Rollouts are generated for the whole window, optionally chunked by
          `custom.extra.rollout_matching.rollout_infer_batch_size`.
        """

        gas = int(len(window_raw_micro_batches))
        if gas <= 0:
            raise ValueError("window_raw_micro_batches is empty")

        # Flatten window samples (preserve micro-step order).
        flat_inputs: List[Mapping[str, Any]] = []
        per_micro_n: List[int] = []
        for mb in window_raw_micro_batches:
            per_micro_n.append(int(len(mb)))
            flat_inputs.extend(list(mb))

        # Build post-rollout segments for the whole window in one pass.
        # This triggers rollout generation on `flat_inputs` and then parses/matches
        # per sample to produce segments.
        segs, bm_total = self._prepare_batch_inputs_b(flat_inputs, _segments_only=True)

        # Schedule segments into exactly `gas` micro-packs.
        t_pack0 = time.perf_counter()
        packs, window_pack_metrics = self._schedule_post_rollout_packs_window(
            window_segments=segs,
            gas=gas,
        )
        t_pack_s = float(time.perf_counter() - t_pack0)

        template = self.template
        from swift.llm import to_device

        prepared: List[Dict[str, Any]] = []
        packing_length = int(self._packing_length())

        # Put aggregated rollout/match metrics on the first micro-step only to avoid
        # spamming the log with duplicated window totals.
        for i, selected in enumerate(packs):
            if not selected:
                raise ValueError(
                    "window post-rollout packing produced an empty micro-pack. "
                    "This indicates the window did not produce enough post-rollout segments."
                )

            with self._template_packing_enabled():
                packed = template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            bm: Dict[str, float] = {}
            if i == 0 and isinstance(bm_total, Mapping):
                bm.update({k: float(v) for k, v in bm_total.items() if isinstance(v, (int, float))})
                bm.update(window_pack_metrics)
                bm["time/post_rollout_pack_s"] = float(t_pack_s)
                bm["packing/post_rollout_scope_window"] = 1.0
            else:
                bm["time/post_rollout_pack_s"] = 0.0
                bm["packing/post_rollout_scope_window"] = 0.0

            sel_total = int(sum(int(sl) for _, _, sl in selected))
            fill = float(sel_total) / float(packing_length) if packing_length > 0 else 0.0
            bm.update(
                {
                    "packing/post_rollout_fill": float(fill),
                    "packing/post_rollout_selected_total_len": float(sel_total),
                    "packing/post_rollout_segments": float(len(selected)),
                    "packing/post_rollout_buffer": float(0.0),
                }
            )

            batch["_rollout_matching_batch_metrics"] = bm
            batch["_stage2_ab_channel"] = "B"
            prepared.append(batch)

        return prepared

    def _prepare_batch_inputs(
        self,
        inputs: List[Mapping[str, Any]],
        _segments_only: bool = False,
    ) -> Any:
        ch = self._stage2_channel_for_step(
            int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        )
        if ch == "A":
            return self._prepare_batch_inputs_a(inputs, _segments_only=_segments_only)
        return self._prepare_batch_inputs_b(inputs, _segments_only=_segments_only)

    def _prepare_batch_inputs_a(
        self, inputs: List[Mapping[str, Any]], *, _segments_only: bool
    ) -> Any:
        template = self.template
        tok = template.tokenizer

        coord_token_ids = self._get_coord_token_ids()
        coord_id_set = set(int(i) for i in coord_token_ids if int(i) >= 0)

        packing_enabled = self._packing_enabled()
        if packing_enabled and not self._packing_drop_last():
            raise ValueError(
                "stage2-ab packing uses carry-only mode and requires training.packing_drop_last: true"
            )
        if packing_enabled and self._packing_buffer_cap() <= 0:
            raise ValueError(
                "training.packing_buffer must be a positive int when packing is enabled"
            )
        if packing_enabled and self._packing_length() <= 0:
            raise ValueError(
                "packing is enabled but no valid packing_length/template.max_length is set (check global_max_length)"
            )

        encoded_batch: List[Dict[str, Any]] = []
        meta_unpacked: List[Dict[str, Any]] = []
        segments: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []

        t_encode_s = 0.0

        for sample in inputs:
            if "messages" not in sample:
                raise ValueError("stage2-ab requires 'messages' in dataset samples")

            gts = _extract_gt_bboxonly(sample)

            payload: Dict[str, Any] = {}
            for obj in gts:
                payload[f"object_{int(obj.index)}"] = {
                    "desc": str(obj.desc),
                    "bbox_2d": [f"<|coord_{int(v)}|>" for v in obj.points_norm1000],
                }
            assistant_text = json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
            y_train_ids = tok.encode(assistant_text, add_special_tokens=False)

            # Desc spans for CE weighting (relative to tail ids).
            tail_desc_pos = _find_desc_value_token_positions(
                tokenizer=tok, token_ids=y_train_ids
            )

            # Bbox groups (relative positions in y_train_ids); convert to full positions after we know prompt_len.
            rel_groups = _bbox_groups_from_token_ids(
                token_ids=y_train_ids, coord_id_set=coord_id_set, gt_objs=gts
            )

            t_enc0 = time.perf_counter()
            data_for_encode = dict(sample)
            messages = json.loads(json.dumps(sample["messages"]))
            has_assistant = False
            try:
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        has_assistant = True
                        break
            except Exception:
                has_assistant = False

            if has_assistant:
                data_for_encode["messages"] = replace_assistant_response_with_ids(
                    messages, y_train_ids
                )
            else:
                data_for_encode["messages"] = list(messages) + [
                    {"role": "assistant", "content": y_train_ids}
                ]
            with self._template_train_mode():
                encoded = template.encode(data_for_encode, return_length=True)
            t_encode_s += time.perf_counter() - t_enc0

            encoded_len = self._extract_encoded_len(encoded)

            enc_ids = encoded.get("input_ids")
            if not isinstance(enc_ids, Sequence):
                raise ValueError("template.encode did not return sequence input_ids")
            enc_ids_list = [int(x) for x in enc_ids]

            # Robustly locate assistant span (prompt_len) even under truncation.
            prompt_len: Optional[int] = None
            train_len_eff: Optional[int] = None
            labels = encoded.get("labels")
            if isinstance(labels, Sequence):
                try:
                    lbl = [int(x) for x in labels]
                    first = next((i for i, x in enumerate(lbl) if int(x) != -100), None)
                    if first is not None:
                        last = max(i for i, x in enumerate(lbl) if int(x) != -100)
                        prompt_len = int(first)
                        train_len_eff = int(last - first + 1)
                except Exception:
                    prompt_len = None
                    train_len_eff = None

            if prompt_len is None:
                prompt_len = _find_subsequence(enc_ids_list, y_train_ids)
                train_len_eff = int(len(y_train_ids))

            # Clamp to encoded length (e.g. when template truncates).
            prompt_len = int(prompt_len)
            train_len_eff = int(train_len_eff or 0)
            train_len_eff = max(
                0, min(train_len_eff, int(encoded_len) - int(prompt_len))
            )

            prompt_ids = enc_ids_list[:prompt_len]

            tail_desc_pos_eff: List[int] = []
            for rel in tail_desc_pos:
                try:
                    rel_i = int(rel)
                except Exception:
                    continue
                if 0 <= rel_i < train_len_eff:
                    tail_desc_pos_eff.append(rel_i)

            bbox_groups_fn: List[Dict[str, Any]] = []
            for obj, rel_pos in zip(gts, rel_groups):
                try:
                    rel_pos_int = [int(p) for p in rel_pos]
                except Exception:
                    continue
                if len(rel_pos_int) != 4:
                    continue
                if any(p < 0 or p >= train_len_eff for p in rel_pos_int):
                    continue
                abs_pos = [int(prompt_len + p) for p in rel_pos_int]
                if any(p >= int(encoded_len) for p in abs_pos):
                    continue
                bbox_groups_fn.append(
                    {
                        "pos": abs_pos,
                        "gt_bins": list(obj.points_norm1000),
                    }
                )

            meta_entry: Dict[str, Any] = {
                "stage2_channel": "A",
                "prompt_len": int(prompt_len),
                "prompt_ids": prompt_ids,
                "rollout_len": int(0),
                "prefix_len": int(0),
                "train_len": int(train_len_eff),
                "encoded_len": int(encoded_len),
                "decode_mode": "none",
                "parse_dropped_invalid": int(0),
                "parse_dropped_ambiguous": int(0),
                "parse_truncated": bool(False),
                "valid_pred_objects": int(0),
                "matched_for_supervision": int(0),
                "matched_maskiou_sum": float(0.0),
                "matched_maskiou_count": int(0),
                "gt_objects": int(len(gts)),
                "fn_count": int(0),
                "gating_rejections": int(0),
                "excluded_from_supervision": int(0),
                "prefix_coord_pos": [],
                "prefix_coord_target_bins": [],
                "tail_ignore_pos": [],
                "tail_desc_pos": tail_desc_pos_eff,
                "bbox_groups_prefix": [],
                "bbox_groups_fn": bbox_groups_fn,
            }

            segments.append((encoded, meta_entry, int(encoded_len)))
            if not packing_enabled:
                encoded_batch.append(encoded)
                meta_unpacked.append(meta_entry)

        from swift.llm import to_device

        batch_metrics: Dict[str, float] = {
            "stage2/channel_a": float(1.0),
            "stage2/channel_b": float(0.0),
            "time/rollout_generate_s": float(0.0),
            "time/rollout_parse_match_s": float(0.0),
            "time/rollout_teacher_encode_s": float(t_encode_s),
        }

        if bool(_segments_only):
            return segments, batch_metrics

        if packing_enabled:
            self._append_post_rollout_segments(segments)

            t_pack0 = time.perf_counter()
            selected, pack_metrics = self._pop_post_rollout_pack()
            with self._template_packing_enabled():
                packed = template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            batch_metrics.update(pack_metrics)
            batch_metrics["time/post_rollout_pack_s"] = float(
                time.perf_counter() - t_pack0
            )
            batch["_rollout_matching_batch_metrics"] = batch_metrics
            batch["_stage2_ab_channel"] = "A"
            return batch

        with self._template_packing_disabled():
            batch = to_device(template.data_collator(encoded_batch), self.model.device)
        batch["_rollout_matching_meta"] = meta_unpacked
        batch["_rollout_matching_batch_metrics"] = batch_metrics
        batch["_stage2_ab_channel"] = "A"
        return batch

    def _prepare_batch_inputs_b(
        self, inputs: List[Mapping[str, Any]], *, _segments_only: bool
    ) -> Any:
        template = self.template
        tok = template.tokenizer

        coord_token_ids = self._get_coord_token_ids()
        coord_id_set = set(int(i) for i in coord_token_ids if int(i) >= 0)
        coord_id_to_bin = self._coord_id_map()

        gate_thr = float(self._cfg("maskiou_gate", 0.3))
        top_k = int(self._cfg("candidate_top_k", 10))
        mask_res = int(self._cfg("maskiou_resolution", 256))

        fp_cost = float(self._cfg("fp_cost", 1.0))
        fn_cost = float(self._cfg("fn_cost", 1.0))

        packing_enabled = self._packing_enabled()
        if packing_enabled and not self._packing_drop_last():
            raise ValueError(
                "stage2-ab post-rollout packing uses carry-only mode and requires training.packing_drop_last: true"
            )
        if packing_enabled and self._packing_buffer_cap() <= 0:
            raise ValueError(
                "training.packing_buffer must be a positive int when packing is enabled"
            )
        if packing_enabled and self._packing_length() <= 0:
            raise ValueError(
                "packing is enabled but no valid packing_length/template.max_length is set (check global_max_length)"
            )

        gs = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        seed_base = int(self._derive_rollout_seed_base(global_step=gs))

        backend = self._rollout_backend()
        decode_mode = str(self._cfg("decode_mode", "greedy")).lower()
        max_new_tokens = int(self._cfg("max_new_tokens", 512))
        num_beams = int(self._cfg("num_beams", 1))
        temperature = float(self._cfg("temperature", 0.0))
        repetition_penalty = float(self._cfg("repetition_penalty", 1.0) or 1.0)
        do_sample = bool(float(temperature) > 0.0)

        with self._hf_sampling_seed_context(
            seed_base=seed_base, backend=backend, do_sample=do_sample
        ) as seeded:
            hf_seeded_global = float(1.0 if seeded else 0.0)

            t_gen0 = time.perf_counter()

            # Allow batching rollout requests (server/hf) independently of learner microbatch.
            # This is most useful when using window packing (post_rollout_pack_scope='window'),
            # where `inputs` may contain an entire accumulation window of samples.
            rollout_infer_bs_raw = self._cfg("rollout_infer_batch_size", None)
            if rollout_infer_bs_raw is None:
                rollout_infer_bs_raw = self._cfg("rollout_generate_batch_size", None)
            try:
                rollout_infer_bs = int(rollout_infer_bs_raw) if rollout_infer_bs_raw is not None else 0
            except Exception:
                rollout_infer_bs = 0
            if rollout_infer_bs <= 0:
                rollout_infer_bs = int(len(inputs))

            rollout_results = []
            for off in range(0, int(len(inputs)), int(rollout_infer_bs)):
                chunk = inputs[int(off) : int(off + rollout_infer_bs)]
                if not chunk:
                    continue
                rollout_results.extend(self._rollout_many(chunk))

            if len(rollout_results) != len(inputs):
                raise RuntimeError(
                    "rollout backend returned unexpected number of results"
                )
            t_gen_s = time.perf_counter() - t_gen0

        encoded_batch: List[Dict[str, Any]] = []
        meta_unpacked: List[Dict[str, Any]] = []
        segments: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []

        t_parse_match_s = 0.0
        t_encode_s = 0.0

        drop_poly_total = 0
        drop_unknown_total = 0
        drop_bbox_invalid_total = 0
        invalid_rollout_total = 0

        for sample, (resp_ids, _resp_text, decode_mode, prompt_ids) in zip(
            inputs, rollout_results
        ):
            if "messages" not in sample:
                raise ValueError("stage2-ab requires 'messages' in dataset samples")

            t_pm0 = time.perf_counter()
            parse = parse_rollout_for_matching(
                tokenizer=tok, response_token_ids=resp_ids
            )

            # Invalid rollout fallback detection: prefix is exactly "{" and no valid objects.
            invalid_rollout = 0
            try:
                brace_ids = tok.encode("{", add_special_tokens=False)
                if (
                    list(parse.prefix_token_ids) == [int(x) for x in brace_ids]
                    and not parse.valid_objects
                ):
                    invalid_rollout = 1
            except Exception:
                invalid_rollout = 0

            gts = _extract_gt_bboxonly(sample)

            # Filter preds to bbox-only.
            pred_meta = []
            preds: List[GTObject] = []
            for pobj in list(parse.valid_objects):
                if pobj.geom_type != "bbox_2d":
                    if pobj.geom_type == "poly":
                        drop_poly_total += 1
                    else:
                        drop_unknown_total += 1
                    continue
                pts = _points_from_coord_tokens(
                    response_token_ids=parse.response_token_ids,
                    coord_token_indices=pobj.coord_token_indices,
                    coord_id_to_bin=coord_id_to_bin,
                )
                if pts is None or len(pts) != 4:
                    drop_bbox_invalid_total += 1
                    continue
                pred_meta.append(pobj)
                preds.append(
                    GTObject(
                        index=int(pobj.index),
                        geom_type="bbox_2d",
                        points_norm1000=list(pts),
                        desc="",
                    )
                )

            match = hungarian_match_maskiou(
                preds=preds,
                gts=gts,
                top_k=top_k,
                gate_threshold=gate_thr,
                mask_resolution=mask_res,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
            )

            prefix_bbox_groups: List[Dict[str, Any]] = []
            prefix_pos: List[int] = []
            prefix_bins: List[int] = []
            matched_gt_for_supervision: set[int] = set()
            for pred_i, gt_i in match.matched_pairs:
                if pred_i < 0 or pred_i >= len(pred_meta):
                    continue
                if gt_i < 0 or gt_i >= len(gts):
                    continue
                pobj = pred_meta[pred_i]
                if len(pobj.coord_token_indices) != 4:
                    continue
                matched_gt_for_supervision.add(int(gt_i))
                gt_bins = list(gts[gt_i].points_norm1000)
                pos_seg = [int(len(prompt_ids) + int(p)) for p in pobj.coord_token_indices]
                prefix_bbox_groups.append({"pos": pos_seg, "gt_bins": gt_bins})
                for local_idx, tbin in zip(pobj.coord_token_indices, gt_bins):
                    prefix_pos.append(int(local_idx))
                    prefix_bins.append(int(tbin))

            fn_gt_indices_final = [
                i for i in range(len(gts)) if i not in matched_gt_for_supervision
            ]
            fn_objs = [gts[i] for i in fn_gt_indices_final]

            max_idx = parse.max_object_index_in_prefix
            start_idx = (max_idx + 1) if max_idx is not None else 1
            append_text = _serialize_append_fragment(
                fn_objects=fn_objs, start_index=start_idx, prefix_text=parse.prefix_text
            )
            append_ids = tok.encode(append_text, add_special_tokens=False)

            tail_desc_pos = _find_desc_value_token_positions(
                tokenizer=tok, token_ids=append_ids
            )

            y_train_ids = list(parse.prefix_token_ids) + [int(t) for t in append_ids]

            # FN bbox groups in the appended tail.
            rel_groups = _bbox_groups_from_token_ids(
                token_ids=append_ids, coord_id_set=coord_id_set, gt_objs=fn_objs
            )
            fn_bbox_groups: List[Dict[str, Any]] = []
            for obj, rel_pos in zip(fn_objs, rel_groups):
                fn_bbox_groups.append(
                    {
                        "pos": [
                            int(len(prompt_ids) + int(len(parse.prefix_token_ids)) + p)
                            for p in rel_pos
                        ],
                        "gt_bins": list(obj.points_norm1000),
                    }
                )

            t_parse_match_s += time.perf_counter() - t_pm0

            # Teacher-forced encode.
            t_enc0 = time.perf_counter()
            data_for_encode = dict(sample)
            messages = json.loads(json.dumps(sample["messages"]))
            has_assistant = False
            try:
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        has_assistant = True
                        break
            except Exception:
                has_assistant = False

            if has_assistant:
                data_for_encode["messages"] = replace_assistant_response_with_ids(
                    messages, y_train_ids
                )
            else:
                data_for_encode["messages"] = list(messages) + [
                    {"role": "assistant", "content": y_train_ids}
                ]
            with self._template_train_mode():
                encoded = template.encode(data_for_encode, return_length=True)
            t_encode_s += time.perf_counter() - t_enc0

            encoded_len = self._extract_encoded_len(encoded)

            enc_ids = encoded.get("input_ids")
            if not isinstance(enc_ids, Sequence):
                raise ValueError("template.encode did not return sequence input_ids")
            enc_ids_list = [int(x) for x in enc_ids]

            # Robustly locate assistant span (prompt_len) even under truncation.
            prompt_len: Optional[int] = None
            train_len_eff: Optional[int] = None
            labels = encoded.get("labels")
            if isinstance(labels, Sequence):
                try:
                    lbl = [int(x) for x in labels]
                    first = next((i for i, x in enumerate(lbl) if int(x) != -100), None)
                    if first is not None:
                        last = max(i for i, x in enumerate(lbl) if int(x) != -100)
                        prompt_len = int(first)
                        train_len_eff = int(last - first + 1)
                except Exception:
                    prompt_len = None
                    train_len_eff = None

            if prompt_len is None:
                prompt_len = _find_subsequence(enc_ids_list, y_train_ids)
                train_len_eff = int(len(y_train_ids))

            prompt_len = int(prompt_len)
            train_len_eff = int(train_len_eff or 0)
            train_len_eff = max(
                0, min(train_len_eff, int(encoded_len) - int(prompt_len))
            )

            prefix_len_raw = int(len(parse.prefix_token_ids))
            prefix_len_eff = int(min(prefix_len_raw, train_len_eff))

            if int(encoded_len) <= int(prompt_len):
                raise ValueError(
                    "teacher-forced encode produced no assistant span: "
                    f"prompt_len={int(prompt_len)} encoded_len={int(encoded_len)} train_len={int(train_len_eff)}"
                )

            prompt_ids_local = enc_ids_list[:prompt_len]
            delta_prompt = int(prompt_len) - int(len(prompt_ids))

            # Shift bbox groups based on local prompt_len, and drop any out-of-range groups.
            def _shift_groups(
                groups: Sequence[Mapping[str, Any]], *, lower: int, upper: int
            ) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for g in groups:
                    if not isinstance(g, Mapping):
                        continue
                    pos = g.get("pos")
                    gb = g.get("gt_bins")
                    if not isinstance(pos, Sequence) or not isinstance(gb, Sequence):
                        continue
                    if len(pos) != 4 or len(gb) != 4:
                        continue
                    try:
                        pos_i = [int(p) for p in pos]
                        gb_i = [int(x) for x in gb]
                    except Exception:
                        continue
                    pos_i = [int(p + delta_prompt) for p in pos_i]
                    if any(p < int(lower) or p >= int(upper) for p in pos_i):
                        continue
                    if any(p >= int(encoded_len) for p in pos_i):
                        continue
                    out.append({"pos": pos_i, "gt_bins": gb_i})
                return out

            bbox_groups_prefix = _shift_groups(
                prefix_bbox_groups,
                lower=int(prompt_len),
                upper=int(prompt_len + prefix_len_eff),
            )
            bbox_groups_fn = _shift_groups(
                fn_bbox_groups,
                lower=int(prompt_len + prefix_len_eff),
                upper=int(prompt_len + train_len_eff),
            )

            # Tail desc spans for CE weighting (relative to tail ids).
            tail_desc_pos_eff: List[int] = []
            tail_cap = max(0, int(train_len_eff) - int(prefix_len_eff))
            for rel in tail_desc_pos:
                try:
                    rel_i = int(rel)
                except Exception:
                    continue
                if 0 <= rel_i < tail_cap:
                    tail_desc_pos_eff.append(rel_i)

            invalid_rollout_total += int(invalid_rollout)

            meta_entry: Dict[str, Any] = {
                "stage2_channel": "B",
                "stage2_invalid_rollout": int(invalid_rollout),
                "rollout_seed_base": int(seed_base),
                "prompt_len": int(prompt_len),
                "prompt_ids": prompt_ids_local,
                "rollout_len": int(len(parse.response_token_ids)),
                "prefix_len": int(prefix_len_eff),
                "train_len": int(train_len_eff),
                "encoded_len": int(encoded_len),
                "decode_mode": str(decode_mode),
                "parse_dropped_invalid": int(parse.dropped_invalid),
                "parse_dropped_ambiguous": int(parse.dropped_ambiguous),
                "parse_truncated": bool(parse.truncated),
                "valid_pred_objects": int(len(preds)),
                "matched_for_supervision": int(len(matched_gt_for_supervision)),
                "matched_maskiou_sum": float(match.matched_maskiou_sum),
                "matched_maskiou_count": int(match.matched_maskiou_count),
                "gt_objects": int(len(gts)),
                "fn_count": int(len(fn_objs)),
                "gating_rejections": int(match.gating_rejections),
                "excluded_from_supervision": int(0),
                "prefix_coord_pos": prefix_pos,
                "prefix_coord_target_bins": prefix_bins,
                "tail_ignore_pos": [],
                "tail_desc_pos": tail_desc_pos_eff,
                "bbox_groups_prefix": bbox_groups_prefix,
                "bbox_groups_fn": bbox_groups_fn,
            }

            segments.append((encoded, meta_entry, int(encoded_len)))
            if not packing_enabled:
                encoded_batch.append(encoded)
                meta_unpacked.append(meta_entry)

        from swift.llm import to_device

        batch_metrics: Dict[str, float] = {
            "stage2/channel_a": float(0.0),
            "stage2/channel_b": float(1.0),
            "stage2/invalid_rollout": float(invalid_rollout_total),
            "stage2/drop_poly": float(drop_poly_total),
            "stage2/drop_unknown": float(drop_unknown_total),
            "stage2/drop_bbox_invalid": float(drop_bbox_invalid_total),
            "rollout/seed_base": float(seed_base),
            "rollout/backend_hf": float(1.0 if backend == "hf" else 0.0),
            "rollout/backend_vllm": float(1.0 if backend == "vllm" else 0.0),
            "rollout/decode_mode_greedy": float(1.0 if decode_mode == "greedy" else 0.0),
            "rollout/decode_mode_beam": float(1.0 if decode_mode == "beam" else 0.0),
            "rollout/hf_seeded_global": float(hf_seeded_global),
            "rollout/temperature": float(temperature),
            "rollout/do_sample": float(1.0 if do_sample else 0.0),
            "rollout/max_new_tokens": float(max_new_tokens),
            "rollout/num_beams": float(num_beams),
            "rollout/repetition_penalty": float(repetition_penalty),
            "time/rollout_generate_s": float(t_gen_s),
            "time/rollout_parse_match_s": float(t_parse_match_s),
            "time/rollout_teacher_encode_s": float(t_encode_s),
        }

        if bool(_segments_only):
            return segments, batch_metrics

        if packing_enabled:
            self._append_post_rollout_segments(segments)

            t_pack0 = time.perf_counter()
            selected, pack_metrics = self._pop_post_rollout_pack()
            with self._template_packing_enabled():
                packed = template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            batch_metrics.update(pack_metrics)
            batch_metrics["time/post_rollout_pack_s"] = float(
                time.perf_counter() - t_pack0
            )
            batch["_rollout_matching_batch_metrics"] = batch_metrics
            batch["_stage2_ab_channel"] = "B"
            return batch

        with self._template_packing_disabled():
            batch = to_device(template.data_collator(encoded_batch), self.model.device)
        batch["_rollout_matching_meta"] = meta_unpacked
        batch["_rollout_matching_batch_metrics"] = batch_metrics
        batch["_stage2_ab_channel"] = "B"
        return batch

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        channel = inputs.pop("_stage2_ab_channel", None)
        if channel not in {"A", "B"}:
            channel = "B"

        meta = inputs.pop("_rollout_matching_meta", None)
        if not isinstance(meta, list):
            raise ValueError("stage2-ab trainer requires _rollout_matching_meta")

        batch_metrics = inputs.pop("_rollout_matching_batch_metrics", None)

        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("stage2-ab compute_loss requires input_ids tensor")

        coord_token_ids = self._get_coord_token_ids()
        coord_id_set = set(int(i) for i in coord_token_ids if int(i) >= 0)
        coord_ids_t = torch.tensor(
            coord_token_ids, device=input_ids.device, dtype=torch.long
        )

        # Config.
        n_softctx_iter = int(self._ab_get("n_softctx_iter", 1) or 1)
        n_softctx_iter = max(1, n_softctx_iter)
        desc_ce_weight = float(self._ab_get("desc_ce_weight", 1.0) or 0.0)
        desc_ce_weight = max(0.0, desc_ce_weight)
        bbox_l1_w = float(self._ab_get("bbox_l1_weight", 1.0) or 0.0)
        bbox_l1_w = max(0.0, bbox_l1_w)
        bbox_giou_w = float(self._ab_get("bbox_giou_weight", 1.0) or 0.0)
        bbox_giou_w = max(0.0, bbox_giou_w)
        temperature = float(self._ab_get("softctx_temperature", 1.0) or 1.0)
        if temperature <= 0:
            raise ValueError(f"softctx_temperature must be > 0; got {temperature}")

        # Always compute logits; do not rely on model.loss.
        ignored_keys = {
            "labels",
            "compute_loss_func",
            "loss_scale",
            "text_position_ids",
            "channel",
            "logits_to_keep",
        }
        inputs_for_model = {k: v for k, v in inputs.items() if k not in ignored_keys}

        # NOTE: under multi-GPU training, `model` may be wrapped in DistributedDataParallel.
        # Access config/embeddings from the underlying module, but run the forward via the
        # wrapper so gradients synchronize correctly.
        core_model = getattr(model, "module", model)

        # Qwen-VL (mRoPE) + padding_free packing: build 4-row position_ids.
        try:
            model_type = str(
                getattr(getattr(core_model, "config", None), "model_type", "") or ""
            )
        except Exception:
            model_type = ""

        text_position_ids = inputs.get("text_position_ids")
        position_ids = inputs_for_model.get("position_ids")
        if (
            model_type.startswith("qwen")
            and isinstance(text_position_ids, torch.Tensor)
            and isinstance(position_ids, torch.Tensor)
            and position_ids.ndim == 3
            and position_ids.shape[0] == 3
            and text_position_ids.ndim == 2
            and text_position_ids.shape == position_ids.shape[1:]
        ):
            inputs_for_model["position_ids"] = torch.cat(
                [text_position_ids.unsqueeze(0), position_ids], dim=0
            )

        # Fail fast: padding-free packing needs the Qwen 4-row position_ids contract.
        packing_enabled = False
        try:
            packing_enabled = bool(self._packing_enabled())
        except Exception:
            packing_enabled = False
        if packing_enabled and model_type.startswith("qwen"):
            pos4 = inputs_for_model.get("position_ids")
            if not (
                isinstance(pos4, torch.Tensor)
                and pos4.ndim == 3
                and int(pos4.shape[0]) == 4
            ):
                raise ValueError(
                    "stage2-ab packing enabled but missing Qwen3-VL padding-free packing metadata: "
                    "expected either position_ids[4,bsz,seqlen] or (text_position_ids[bsz,seqlen] + position_ids[3,bsz,seqlen]). "
                    "Mitigations: disable training.packing or check the chat template/packing pipeline."
                )

        t_fwd0 = time.perf_counter()
        outputs = None

        if channel == "A":
            embed = core_model.get_input_embeddings()
            if embed is None:
                raise ValueError("core_model.get_input_embeddings() returned None")

            # Build coord-slot update indices (segment-aware for packing).
            def _collect_coord_slots() -> Tuple[torch.Tensor, torch.Tensor]:
                bsz, seqlen = input_ids.shape
                b_list: List[int] = []
                p_list: List[int] = []
                if len(meta) == bsz:
                    for b in range(bsz):
                        m = meta[b]
                        prompt_len = int(m.get("prompt_len", 0))
                        train_len = int(m.get("train_len", 0))
                        start = max(1, prompt_len)
                        end = min(seqlen, prompt_len + train_len)
                        for p in range(start, end):
                            if int(input_ids[b, p].item()) in coord_id_set:
                                b_list.append(int(b))
                                p_list.append(int(p))
                    return (
                        torch.tensor(b_list, device=input_ids.device, dtype=torch.long),
                        torch.tensor(p_list, device=input_ids.device, dtype=torch.long),
                    )

                if bsz != 1:
                    raise ValueError("packed-mode meta requires bsz==1")
                offset = 0
                for seg in meta:
                    encoded_len = int(seg.get("encoded_len") or 0)
                    if encoded_len <= 0:
                        raise ValueError("packed-mode segment missing encoded_len")
                    prompt_len = int(seg.get("prompt_len", 0))
                    train_len = int(seg.get("train_len", 0))
                    seg_start = offset
                    seg_end = offset + encoded_len
                    start = max(seg_start + 1, seg_start + prompt_len)
                    end = min(seg_end, seg_start + prompt_len + train_len)
                    for p in range(start, end):
                        if int(input_ids[0, p].item()) in coord_id_set:
                            b_list.append(0)
                            p_list.append(int(p))
                    offset += encoded_len
                return (
                    torch.tensor(b_list, device=input_ids.device, dtype=torch.long),
                    torch.tensor(p_list, device=input_ids.device, dtype=torch.long),
                )

            b_slots, p_slots = _collect_coord_slots()
            coord_table = embed(coord_ids_t)
            coord_table_f = coord_table.float()

            logits_prev: Optional[torch.Tensor] = None
            for it in range(int(n_softctx_iter)):
                grad_on = bool(it == int(n_softctx_iter) - 1)
                ctx = torch.enable_grad() if grad_on else torch.no_grad()
                with ctx:
                    base_embeds = embed(input_ids)
                    embeds = base_embeds
                    if it > 0 and logits_prev is not None and p_slots.numel() > 0:
                        logit_pos = (p_slots - 1).clamp(min=0)
                        logits_next = logits_prev[b_slots, logit_pos]
                        coord_logits = logits_next.index_select(dim=-1, index=coord_ids_t)
                        probs = torch.softmax(coord_logits.float() / temperature, dim=-1)
                        exp_emb = probs @ coord_table_f
                        exp_emb = exp_emb.to(base_embeds.dtype)
                        exp_emb = exp_emb.detach()
                        embeds = base_embeds.clone()
                        embeds[b_slots, p_slots] = exp_emb

                    fwd_inputs = dict(inputs_for_model)
                    fwd_inputs.pop("input_ids", None)
                    fwd_inputs["inputs_embeds"] = embeds
                    fwd_inputs["use_cache"] = False
                    fwd_inputs.pop("past_key_values", None)
                    out = model(**fwd_inputs)
                    if not hasattr(out, "logits") or out.logits is None:
                        raise ValueError("model did not return logits")
                    if getattr(out, "past_key_values", None) is not None:
                        raise ValueError("past_key_values must be None for Channel-A")
                    logits_prev = out.logits
                    outputs = out

        else:
            fwd_inputs = dict(inputs_for_model)
            fwd_inputs["use_cache"] = False
            fwd_inputs.pop("past_key_values", None)
            outputs = model(**fwd_inputs)

        t_fwd_s = time.perf_counter() - t_fwd0

        if outputs is None or outputs.logits is None:
            raise ValueError("model did not return logits")
        logits = outputs.logits
        if logits.shape[:2] != input_ids.shape[:2]:
            raise ValueError(
                "model returned sliced logits (logits_to_keep-style). Disable logits slicing for stage2-ab training."
            )

        bsz, seq_len, vocab = logits.shape

        # Build labels + weights for CE.
        labels_masked = torch.full_like(input_ids, -100)
        weights_masked = input_ids.new_zeros(input_ids.shape, dtype=torch.float32)

        def _apply_seg(
            *,
            b: int,
            offset: int,
            encoded_len: int,
            m: Mapping[str, Any],
        ) -> None:
            prompt_len = int(m.get("prompt_len", 0))
            prefix_len = int(m.get("prefix_len", 0))
            train_len = int(m.get("train_len", 0))
            tail_desc_pos = list(m.get("tail_desc_pos") or [])

            tail_start = int(prompt_len + prefix_len)
            tail_end = int(prompt_len + train_len)

            seg_start = int(offset)
            seg_end = int(offset + encoded_len)

            tail_start = max(seg_start + 1, min(seg_start + tail_start, seg_end))
            tail_end = max(tail_start, min(seg_start + tail_end, seg_end))

            for p in range(tail_start, tail_end):
                tok_id = int(input_ids[b, p].item())
                if tok_id in coord_id_set:
                    continue
                labels_masked[b, p] = input_ids[b, p]
                weights_masked[b, p] = 1.0

            # Apply desc weighting on tail desc value tokens.
            for rel in tail_desc_pos:
                try:
                    rel_i = int(rel)
                except Exception:
                    continue
                p = int(seg_start + prompt_len + prefix_len + rel_i)
                if p < tail_start or p >= tail_end:
                    continue
                if labels_masked[b, p].item() == -100:
                    continue
                weights_masked[b, p] = float(desc_ce_weight)

        if len(meta) == bsz:
            for b in range(bsz):
                _apply_seg(b=b, offset=0, encoded_len=seq_len, m=meta[b])
        else:
            if bsz != 1:
                raise ValueError("packed-mode meta requires bsz==1")
            offset = 0
            for seg in meta:
                enc_len = int(seg.get("encoded_len") or 0)
                if enc_len <= 0:
                    raise ValueError("packed-mode segment missing encoded_len")
                _apply_seg(b=0, offset=offset, encoded_len=enc_len, m=seg)
                offset += enc_len

        # Weighted CE over supervised (non-coord) tokens.
        logits_next = logits[:, :-1, :]
        labels_next = labels_masked[:, 1:]
        weights_next = weights_masked[:, 1:]
        per_tok = F.cross_entropy(
            logits_next.reshape(-1, vocab),
            labels_next.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(bsz, -1)
        denom = float(weights_next.sum().detach().cpu().item())
        if denom <= 0:
            ce_loss = per_tok.new_tensor(0.0)
        else:
            ce_loss = (per_tok * weights_next).sum() / weights_next.sum().clamp(min=1.0)

        def _flatten_groups(key: str) -> Tuple[List[int], List[int], List[int]]:
            b_list: List[int] = []
            pos_list: List[int] = []
            bins_list: List[int] = []
            if len(meta) == bsz:
                for b in range(bsz):
                    groups = meta[b].get(key) or []
                    for g in groups:
                        if not isinstance(g, Mapping):
                            continue
                        pos = g.get("pos")
                        gb = g.get("gt_bins")
                        if not isinstance(pos, Sequence) or not isinstance(gb, Sequence):
                            continue
                        if len(pos) != 4 or len(gb) != 4:
                            continue
                        for p, tbin in zip(pos, gb):
                            b_list.append(int(b))
                            pos_list.append(int(p))
                            bins_list.append(int(tbin))
                return b_list, pos_list, bins_list

            if bsz != 1:
                raise ValueError("packed-mode meta requires bsz==1")
            offset = 0
            for seg in meta:
                enc_len = int(seg.get("encoded_len") or 0)
                groups = seg.get(key) or []
                for g in groups:
                    if not isinstance(g, Mapping):
                        continue
                    pos = g.get("pos")
                    gb = g.get("gt_bins")
                    if not isinstance(pos, Sequence) or not isinstance(gb, Sequence):
                        continue
                    if len(pos) != 4 or len(gb) != 4:
                        continue
                    for p, tbin in zip(pos, gb):
                        b_list.append(0)
                        pos_list.append(int(offset + int(p)))
                        bins_list.append(int(tbin))
                offset += enc_len
            return b_list, pos_list, bins_list

        def _decode_groups(key: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
            b_list, pos_list, bins_list = _flatten_groups(key)
            if not pos_list:
                z = logits.new_tensor(0.0)
                return z, z, 0
            if len(pos_list) % 4 != 0:
                raise ValueError(f"{key} coord slots must be a multiple of 4 (bbox_2d)")

            b_t = torch.tensor(b_list, device=logits.device, dtype=torch.long)
            pos_t = torch.tensor(pos_list, device=logits.device, dtype=torch.long)
            bin_t = torch.tensor(bins_list, device=logits.device, dtype=torch.long).clamp(
                min=0, max=999
            )

            # Drop any groups with invalid positions to avoid device-side asserts.
            b_g = b_t.reshape(-1, 4)
            pos_g = pos_t.reshape(-1, 4)
            bin_g = bin_t.reshape(-1, 4)

            valid = (pos_g > 0).all(dim=1)
            valid &= (pos_g < int(seq_len)).all(dim=1)
            if not bool(valid.all().item()):
                b_g = b_g[valid]
                pos_g = pos_g[valid]
                bin_g = bin_g[valid]

            n_groups = int(pos_g.shape[0])
            if n_groups == 0:
                z = logits.new_tensor(0.0)
                return z, z, 0

            b_t = b_g.reshape(-1)
            pos_t = pos_g.reshape(-1)
            bin_t = bin_g.reshape(-1)

            logits_prev = logits[b_t, pos_t - 1]
            coord_logits = logits_prev.index_select(dim=-1, index=coord_ids_t)
            pred = _expectation_decode_coords(
                coord_logits=coord_logits, temperature=temperature
            )
            gt = bin_t.float() / 999.0
            pred_xyxy = pred.reshape(-1, 4)
            gt_xyxy = gt.reshape(-1, 4)
            l1, giou = _bbox_l1_giou_loss(pred_xyxy=pred_xyxy, gt_xyxy=gt_xyxy)
            return l1, giou, n_groups

        l1_p, giou_p, n_p = _decode_groups("bbox_groups_prefix")
        l1_t, giou_t, n_t = _decode_groups("bbox_groups_fn")

        n_all = int(n_p + n_t)
        if n_all > 0:
            l1 = (l1_p * float(n_p) + l1_t * float(n_t)) / float(n_all)
            giou = (giou_p * float(n_p) + giou_t * float(n_t)) / float(n_all)
        else:
            l1 = logits.new_tensor(0.0)
            giou = logits.new_tensor(0.0)

        bbox_loss = bbox_l1_w * l1 + bbox_giou_w * giou
        total = ce_loss + bbox_loss

        # Buffer Stage-2 logs to merge into post-optimizer-step train log line.
        try:
            step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            target_step = step + 1

            pending2 = self._stage2_pending_train_logs.get(target_step)
            if pending2 is None:
                pending2 = _PendingStage2Log()
                self._stage2_pending_train_logs[target_step] = pending2
            # Counters should be summed across micro-batches; losses averaged.
            stage2_logs = {
                "stage2/channel_a": float(1.0 if channel == "A" else 0.0),
                "stage2/channel_b": float(1.0 if channel == "B" else 0.0),
                "loss/bbox_l1": float(l1.detach().cpu().item()),
                "loss/bbox_giou": float(giou.detach().cpu().item()),
            }
            if isinstance(batch_metrics, Mapping):
                for k in (
                    "stage2/invalid_rollout",
                    "stage2/drop_poly",
                    "stage2/drop_unknown",
                    "stage2/drop_bbox_invalid",
                    "rollout/seed_base",
                    "rollout/backend_hf",
                    "rollout/backend_vllm",
                    "rollout/decode_mode_greedy",
                    "rollout/decode_mode_beam",
                    "rollout/hf_seeded_global",
                    "rollout/temperature",
                    "rollout/do_sample",
                    "rollout/max_new_tokens",
                    "rollout/num_beams",
                    "rollout/repetition_penalty",
                ):
                    if k in batch_metrics:
                        try:
                            stage2_logs[k] = float(batch_metrics.get(k) or 0.0)
                        except Exception:
                            pass
            pending2.add(stage2_logs)
        except Exception:
            pass

        # Also feed the base rollout-matching pending log so its timing/packing/buffer plots stay intact.
        try:
            pending = self._rm_pending_train_logs.get(int(getattr(getattr(self, "state", None), "global_step", 0) or 0) + 1)
            if pending is None:
                pending = _PendingTrainRolloutLog()
                self._rm_pending_train_logs[int(getattr(getattr(self, "state", None), "global_step", 0) or 0) + 1] = pending
            pending.add_micro(
                meta=meta,
                ce_loss=float(ce_loss.detach().cpu().item()),
                coord_loss=float(bbox_loss.detach().cpu().item()),
                coord_prefix=float((bbox_l1_w * l1_p + bbox_giou_w * giou_p).detach().cpu().item()),
                coord_tail=float((bbox_l1_w * l1_t + bbox_giou_w * giou_t).detach().cpu().item()),
                time_forward_s=float(t_fwd_s),
                time_mask_build_s=float(0.0),
                batch_metrics=batch_metrics if isinstance(batch_metrics, Mapping) else None,
            )
        except Exception:
            pass

        return (total, outputs) if return_outputs else total
