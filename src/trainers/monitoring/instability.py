from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InstabilityMonitorMixin:
    """Per-step monitor + optional guard for catastrophic batches.

    Enabled via `custom.instability_monitor` (stored under `CustomConfig.extra`).
    The collator attaches `instability_meta_json` (batch sample_id/base_idx info),
    which this mixin consumes before model forward.

    When triggered, writes a JSON event to `<output_dir>/instability_dumps/events.jsonl`
    (or `dump_dir` if provided). If guard is enabled, replaces the loss with 0.0 to
    avoid poisoning weights for the current step.

    NOTE: This module is best-effort. Failures must not block training.
    """

    meta_field = "instability_meta_json"

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        # Batch extras are stripped by the Stage-1 extras contract (usually in
        # GradAccumLossScaleMixin). Fetch the meta json from the stash.
        meta_json = None
        if isinstance(inputs, dict):
            labels_snapshot = inputs.get("labels")
        else:
            labels_snapshot = None

        try:
            from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

            extras = maybe_pop_and_stash_batch_extras(self, inputs)
            meta_json = extras.instability_meta_json
        except Exception:
            # Back-compat fallback (should be rare): pop directly.
            if isinstance(inputs, dict):
                meta_json = inputs.pop(self.meta_field, None)

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            from src.metrics.reporter import best_effort_value

            loss = best_effort_value(
                self,
                name="instability_monitor",
                fn=lambda: self._monitor_and_guard(
                    loss=loss, outputs=outputs, labels=labels_snapshot, meta_json=meta_json
                ),
                default=loss,
            )
        except Exception:
            raise

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)
        if not math.isfinite(out):
            return float(default)
        return float(out)

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _get_cfg(self) -> Mapping[str, Any]:
        raw = getattr(self, "instability_monitor_cfg", None)
        return raw if isinstance(raw, Mapping) else {}

    def _rank(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0
        try:
            return int(dist.get_rank())
        except RuntimeError:
            return 0

    def _is_main_process(self) -> bool:
        return self._rank() == 0

    def _dump_dir(self, cfg: Mapping[str, Any]) -> str | None:
        raw = cfg.get("dump_dir")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        output_dir = getattr(getattr(self, "args", None), "output_dir", None)
        if isinstance(output_dir, str) and output_dir:
            return os.path.join(output_dir, "instability_dumps")
        return None

    def _append_event(self, dump_dir: str, event: Mapping[str, Any]) -> None:
        if not self._is_main_process():
            return
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, "events.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(event), ensure_ascii=True, sort_keys=True))
            f.write("\n")

    @staticmethod
    def _load_jsonl_records_by_index(
        jsonl_path: str, indices: Sequence[int]
    ) -> dict[int, Any]:
        """Load a subset of JSONL rows by 0-based line index (best-effort)."""

        wanted = sorted({int(i) for i in indices if isinstance(i, int) or str(i).isdigit()})
        if not wanted:
            return {}
        out: dict[int, Any] = {}
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                current = 0
                target_pos = 0
                target = wanted[target_pos]
                for line in f:
                    if current == target:
                        raw = line.strip("\n")
                        try:
                            out[current] = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            out[current] = {"_raw": raw, "_parse_error": True}
                        target_pos += 1
                        if target_pos >= len(wanted):
                            break
                        target = wanted[target_pos]
                    current += 1
        except (OSError, UnicodeError) as exc:
            return {"_error": str(exc)}  # type: ignore[return-value]
        return out

    def _maybe_dump_samples(
        self, dump_dir: str, *, mode: str, step: Any, meta_json: str | None
    ) -> None:
        cfg = self._get_cfg()
        if not bool(cfg.get("dump_samples", False)):
            return
        if not self._is_main_process():
            return
        if not isinstance(meta_json, str) or not meta_json.strip():
            return

        jsonl_path = None
        if mode == "train":
            jp = getattr(self, "instability_train_jsonl", None)
            if isinstance(jp, str) and jp:
                jsonl_path = jp
        else:
            jp = getattr(self, "instability_val_jsonl", None)
            if isinstance(jp, str) and jp:
                jsonl_path = jp
        if jsonl_path is None:
            return

        try:
            meta = json.loads(meta_json)
        except (json.JSONDecodeError, TypeError):
            meta = None
        if not isinstance(meta, list):
            return

        base_idxs: list[int] = []
        # meta is a list of packs; each contains {"samples":[{"base_idx":...}, ...]}
        for pack in meta:
            if not isinstance(pack, dict):
                continue
            samples = pack.get("samples")
            if not isinstance(samples, list):
                continue
            for s in samples:
                if not isinstance(s, dict):
                    continue
                bi = s.get("base_idx")
                try:
                    bi_i = int(bi)
                except (TypeError, ValueError):
                    continue
                if bi_i >= 0:
                    base_idxs.append(bi_i)

        if not base_idxs:
            return

        records = self._load_jsonl_records_by_index(jsonl_path, base_idxs)
        payload = {
            "mode": mode,
            "global_step": step,
            "jsonl_path": jsonl_path,
            "base_idxs": sorted(set(base_idxs)),
            "records": records,
            "meta": meta,
        }

        os.makedirs(dump_dir, exist_ok=True)
        out_path = os.path.join(dump_dir, f"samples_step{step}_{mode}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, sort_keys=True)

    def _ema_update(self, name: str, value: float, decay: float) -> float:
        key = f"_instab_ema_{name}"
        prev = getattr(self, key, None)
        ema = (
            float(value)
            if prev is None
            else float(decay) * float(prev) + (1.0 - float(decay)) * float(value)
        )
        setattr(self, key, ema)
        return ema

    def _request_training_stop(self, *, reason: str, mode: str, step: Any) -> None:
        """Best-effort early stop request that works with HF Trainer-style loops."""

        if getattr(self, "_instab_stop_requested", False):
            return
        setattr(self, "_instab_stop_requested", True)

        # Mark the control/state so Trainer breaks out cleanly at the next check.
        try:
            control = getattr(self, "control", None)
            if control is not None:
                try:
                    control.should_training_stop = True
                except Exception:
                    raise
                try:
                    control.should_epoch_stop = True
                except Exception:
                    raise
        except Exception:
            raise
        try:
            state = getattr(self, "state", None)
            if state is not None:
                setattr(state, "should_training_stop", True)
        except Exception:
            raise

        if self._is_main_process():
            logger.error(
                "InstabilityMonitor early stop requested (mode=%s step=%s reason=%s)",
                mode,
                step,
                reason,
            )

    def _monitor_and_guard(
        self, *, loss: torch.Tensor, outputs: Any, labels: Any, meta_json: Any
    ) -> torch.Tensor:
        cfg = self._get_cfg()
        if not bool(cfg.get("enabled", False)):
            return loss

        guard_cfg = cfg.get("guard", {}) if isinstance(cfg.get("guard"), Mapping) else {}
        guard_enabled = bool(guard_cfg.get("enabled", False))

        # Optional early-stop (abort training cleanly) when a catastrophic signal is detected.
        early_cfg = cfg.get("early_stop", {}) if isinstance(cfg.get("early_stop"), Mapping) else {}
        early_enabled = bool(early_cfg.get("enabled", False))

        mode = (
            "train" if getattr(self, "model", None) is None or self.model.training else "eval"
        )  # type: ignore[attr-defined]
        if mode != "train" and not bool(guard_cfg.get("guard_in_eval", False)):
            guard_enabled = False

        ema_decay = self._as_float(cfg.get("ema_decay", 0.98), 0.98)
        ema_decay = min(max(ema_decay, 0.0), 0.9999)
        spike_factor = self._as_float(cfg.get("spike_factor", 8.0), 8.0)
        abs_loss_threshold = self._as_float(cfg.get("abs_loss_threshold", 0.2), 0.2)
        min_supervised_tokens = self._as_int(cfg.get("min_supervised_tokens", 64), 64)
        min_token_acc = self._as_float(guard_cfg.get("min_token_acc", 0.01), 0.01)
        max_abs_logit = self._as_float(guard_cfg.get("max_abs_logit", 200.0), 200.0)

        loss_val = float(loss.detach().float().cpu().item())
        finite_loss = bool(torch.isfinite(loss).all().item())

        logits = getattr(outputs, "logits", None)
        token_acc = None
        supervised_tokens = None
        max_logit_abs = None
        finite_logits = None

        if isinstance(logits, torch.Tensor) and isinstance(labels, torch.Tensor):
            seq_len = min(int(logits.shape[1]), max(int(labels.shape[1] - 1), 0))
            if seq_len > 0:
                logits_next = logits[:, :seq_len, :]
                labels_next = labels[:, 1 : seq_len + 1]
                mask = labels_next != -100
                supervised_tokens = int(mask.sum().detach().cpu().item())
                with torch.no_grad():
                    if supervised_tokens > 0:
                        logits_sup = logits_next[mask]
                        labels_sup = labels_next[mask]
                        finite_logits = bool(torch.isfinite(logits_sup).all().item())
                        safe_logits = torch.nan_to_num(
                            logits_sup, nan=0.0, posinf=1e4, neginf=-1e4
                        )
                        max_logit_abs = float(
                            safe_logits.abs().amax().detach().cpu().item()
                        )
                        preds = safe_logits.argmax(dim=-1)
                        token_acc = float(
                            (preds == labels_sup).float().mean().detach().cpu().item()
                        )
                    else:
                        # No supervised tokens -> ignore logits checks for this batch.
                        finite_logits = True

        ema_loss = None
        if mode == "train":
            ema_loss = self._ema_update("loss", loss_val, ema_decay)

        reasons: list[str] = []
        if not finite_loss and bool(guard_cfg.get("guard_on_nonfinite", True)):
            reasons.append("nonfinite_loss")
        if finite_logits is False and bool(guard_cfg.get("guard_on_nonfinite", True)):
            reasons.append("nonfinite_logits")
        if max_logit_abs is not None and max_logit_abs > max_abs_logit:
            reasons.append("logit_overflow_like")

        is_spike = False
        if mode == "train" and ema_loss is not None:
            if (
                loss_val >= abs_loss_threshold
                and ema_loss > 0
                and loss_val > float(spike_factor) * float(ema_loss)
            ):
                is_spike = True
            if loss_val >= abs_loss_threshold and ema_loss < abs_loss_threshold / 4:
                is_spike = True
        if is_spike and bool(guard_cfg.get("guard_on_spike", True)):
            reasons.append("loss_spike")

        if (
            token_acc is not None
            and supervised_tokens is not None
            and supervised_tokens >= min_supervised_tokens
            and token_acc <= 0.0
            and bool(guard_cfg.get("guard_on_zero_acc", True))
        ):
            reasons.append("zero_token_acc")
        elif (
            token_acc is not None
            and supervised_tokens is not None
            and supervised_tokens >= min_supervised_tokens
            and token_acc < min_token_acc
            and bool(guard_cfg.get("guard_on_zero_acc", True))
            and is_spike
        ):
            reasons.append("low_token_acc")

        if not reasons:
            return loss

        dump_dir = self._dump_dir(cfg)
        step = getattr(getattr(self, "state", None), "global_step", None)
        epoch = getattr(getattr(self, "state", None), "epoch", None)

        # Decide whether to request an early stop.
        should_early_stop = False
        stop_reason = None
        if mode == "train":
            if bool(cfg.get("early_stop_on_zero_acc", False)) and "zero_token_acc" in reasons:
                should_early_stop = True
                stop_reason = "zero_token_acc"
            if early_enabled:
                raw_on = early_cfg.get("on_reasons")
                if raw_on is None:
                    on_reasons = ("zero_token_acc",)
                elif isinstance(raw_on, str):
                    on_reasons = (raw_on,)
                elif isinstance(raw_on, (list, tuple, set)):
                    on_reasons = tuple(str(r) for r in raw_on)
                else:
                    on_reasons = ("zero_token_acc",)
                for r in reasons:
                    if r in on_reasons:
                        should_early_stop = True
                        stop_reason = r
                        break

        lr = None
        try:
            opt = getattr(self, "optimizer", None)
            if opt is not None and getattr(opt, "param_groups", None):
                lr = opt.param_groups[0].get("lr", None)
        except Exception:
            lr = None
        if lr is None:
            try:
                sched = getattr(self, "lr_scheduler", None)
                if sched is not None:
                    last = sched.get_last_lr()
                    if last:
                        lr = last[0]
            except Exception:
                lr = None
        if lr is None:
            lr = getattr(getattr(self, "args", None), "learning_rate", None)

        meta_str = meta_json if isinstance(meta_json, str) else None
        if isinstance(meta_str, str) and len(meta_str) > 20000:
            meta_str = meta_str[:20000] + "...(truncated)"

        event = {
            "mode": mode,
            "reasons": reasons,
            "global_step": step,
            "epoch": epoch,
            "loss": loss_val,
            "ema_loss": ema_loss,
            "learning_rate": lr,
            "supervised_tokens": supervised_tokens,
            "token_acc": token_acc,
            "max_abs_logit": max_logit_abs,
            "finite_loss": finite_loss,
            "finite_logits": finite_logits,
            "meta_json": meta_str,
        }
        if dump_dir is not None:
            try:
                self._append_event(dump_dir, event)
                self._maybe_dump_samples(
                    dump_dir,
                    mode=mode,
                    step=step,
                    meta_json=meta_str,
                )
            except Exception:
                raise

        if self._is_main_process():
            ema_s = None if ema_loss is None else f"{float(ema_loss):.6f}"
            acc_s = None if token_acc is None else f"{float(token_acc):.4f}"
            maxlog_s = None if max_logit_abs is None else f"{float(max_logit_abs):.2f}"
            logger.warning(
                "InstabilityMonitor triggered (mode=%s step=%s): %s (loss=%.6f ema=%s acc=%s sup=%s max|logit|=%s)",
                mode,
                step,
                ",".join(reasons),
                loss_val,
                ema_s,
                acc_s,
                supervised_tokens,
                maxlog_s,
            )

        if should_early_stop and stop_reason is not None:
            # Ensure we don't backprop through a bad step; stop ASAP after dumping.
            guard_enabled = True
            try:
                self._request_training_stop(reason=stop_reason, mode=mode, step=step)
            except Exception:
                raise

        if not guard_enabled:
            return loss

        # IMPORTANT: return a differentiable zero.
        logits = getattr(outputs, "logits", None)
        if isinstance(logits, torch.Tensor):
            safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            return safe_logits.sum() * 0.0
        return loss * 0.0
