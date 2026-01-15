import json
import math
import os
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.coord_tokens.codec import get_coord_token_ids
from src.coord_tokens.soft_ce_w1 import coord_soft_ce_w1
from src.data_collators.token_types import TokenType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AggregateTokenTypeMetricsMixin:
    """Trainer mixin to log aggregate loss/accuracy and token-type metrics.

    - Aggregate only (no per-dataset buckets)
    - Safe under packing when token_types are pre-concatenated; skips on mismatch
    - Skips metrics when no supervised tokens to avoid NaNs
    """

    label_field = "dataset_labels"
    segment_field = "dataset_segments"

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        _ = inputs.pop(self.label_field, None)
        _ = inputs.pop(self.segment_field, None)  # Optional legacy field
        labels_for_metrics = inputs.get("labels")
        token_types = inputs.pop("token_types", None)

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            self._log_aggregate_metrics(outputs, labels_for_metrics, token_types)
            self._sync_dataset_metrics()
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

    def _log_aggregate_metrics(
        self,
        outputs: Any,
        labels: Any,
        token_types: Any,
    ) -> None:
        logits = getattr(outputs, "logits", None)
        if logits is None or labels is None or not isinstance(labels, torch.Tensor):
            return

        mode = (
            "train"
            if getattr(self, "model", None) is None or self.model.training
            else "eval"
        )  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        logits_next = logits[:, :-1, :]
        labels_next = labels[:, 1:]
        mask = labels_next != -100
        supervised_count = int(mask.sum().detach())
        if supervised_count == 0:
            return

        token_types_next = None
        if isinstance(token_types, torch.Tensor) and token_types.shape == labels.shape:
            token_types_next = token_types[:, 1:]
            if not (token_types_next != TokenType.IGNORE).any().item():
                token_types_next = None

        coord_mask = None
        if token_types_next is not None:
            coord_mask = token_types_next == TokenType.COORD
        else:
            coord_mask = self._infer_coord_mask(labels_next, logits_next)

        with torch.no_grad():
            preds = logits_next.argmax(dim=-1)

            labels_masked = labels_next[mask]
            preds_masked = preds[mask]

            topk_indices = None
            flat_logits = logits_next[mask]
            if flat_logits.numel() > 0:
                k = min(5, flat_logits.shape[-1])
                topk_indices = flat_logits.topk(k=k, dim=-1).indices
                top5_acc = (
                    (topk_indices == labels_masked.unsqueeze(-1))
                    .any(dim=-1)
                    .float()
                    .mean()
                )
            else:
                top5_acc = logits_next.new_tensor(0.0)

        metrics["token_acc_top5"].update(float(top5_acc.detach().item()))

        if token_types_next is not None:
            types_masked = token_types_next[mask]
            for name, type_id in (
                ("desc", TokenType.DESC),
                ("format", TokenType.FORMAT),
                ("coord", TokenType.COORD),
            ):
                sel = types_masked == type_id
                if not sel.any().item():
                    continue
                metrics[f"{name}_token_frac"].update(float(sel.float().mean().detach().item()))
            for name, type_id in (
                ("desc", TokenType.DESC),
                ("format", TokenType.FORMAT),
                ("coord", TokenType.COORD),
            ):
                type_sel = types_masked == type_id
                if not type_sel.any().item():
                    continue
                with torch.no_grad():
                    acc = (
                        (preds_masked[type_sel] == labels_masked[type_sel])
                        .float()
                        .mean()
                    )
                metrics[f"{name}_token_acc"].update(float(acc.detach().item()))

                if topk_indices is None:
                    continue
                with torch.no_grad():
                    top5 = (
                        (topk_indices[type_sel] == labels_masked[type_sel].unsqueeze(-1))
                        .any(dim=-1)
                        .float()
                        .mean()
                )
                metrics[f"{name}_token_acc_top5"].update(float(top5.detach().item()))

        # Token-type metrics (optional)
        if token_types_next is not None:
            text_mask = mask & (token_types_next != TokenType.COORD)
            text_mask = text_mask & (token_types_next != TokenType.IGNORE)
        elif coord_mask is not None:
            text_mask = mask & (~coord_mask)
        else:
            text_mask = mask

        if text_mask is not None and text_mask.any().item():
            with torch.no_grad():
                text_acc = (preds[text_mask] == labels_next[text_mask]).float().mean()
            metrics["text_token_acc"].update(float(text_acc.detach().item()))

        # Only split text vs coord; do not emit per-subtype metrics.

    def _infer_coord_mask(
        self, labels_next: torch.Tensor, logits_next: torch.Tensor
    ) -> torch.Tensor | None:
        coord_ids_fn = getattr(self, "_get_coord_token_ids", None)
        coord_map_fn = getattr(self, "_get_coord_id_map", None)
        if not callable(coord_ids_fn) or not callable(coord_map_fn):
            return None
        coord_token_ids = coord_ids_fn()
        if not coord_token_ids:
            return None
        coord_id_map = coord_map_fn(logits_next.shape[-1], logits_next.device)
        if coord_id_map is None:
            return None
        labels_safe = labels_next
        if labels_safe.min().item() < 0:
            labels_safe = labels_safe.clamp(min=0)
        coord_mask = coord_id_map >= 0
        max_label = int(labels_safe.max().item()) if labels_safe.numel() else -1
        if max_label < 0 or max_label >= coord_mask.numel():
            return None
        return coord_mask[labels_safe]

    def _sync_dataset_metrics(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        mode = (
            "train"
            if getattr(self, "model", None) is None or self.model.training
            else "eval"
        )  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        local_keys = list(metrics.keys())
        key_cache = getattr(self, "_dataset_metric_key_cache", {})
        cached = key_cache.get(mode, set())
        local_set = set(local_keys)
        if local_set.issubset(cached):
            return

        gathered_keys = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_keys, local_keys)

        union_keys = set()
        for keys in gathered_keys:
            if keys:
                union_keys.update(keys)

        for key in sorted(union_keys):
            if key not in metrics:
                _ = metrics[key]

        cached.update(union_keys)
        key_cache[mode] = cached
        setattr(self, "_dataset_metric_key_cache", key_cache)


class CoordSoftCEW1LossMixin:
    """Trainer mixin to compute coord-token supervision from logits.

    Behaviour when enabled:
      - Masks coord-token targets to `ignore_index` for the base full-vocab CE loss.
      - Computes coord supervision from the same forward logits (no second forward),
        by slicing logits to ordered coord-token ids (0..999) and applying:
          softCE(Gaussian soft target) + W1(CDF).
    """

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return super().compute_loss(  # type: ignore[misc]
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        labels = inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            return super().compute_loss(  # type: ignore[misc]
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        coord_token_ids = self._get_coord_token_ids()
        if not coord_token_ids:
            return super().compute_loss(  # type: ignore[misc]
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        labels_orig = labels
        masked_labels = self._mask_coord_targets(labels_orig, coord_token_ids)
        inputs["labels"] = masked_labels

        passed_num_items = num_items_in_batch
        try:
            # Swift/Transformers may call `compute_loss(..., num_items_in_batch=...)` where
            # `num_items_in_batch` is the *number of sequences* in the micro-batch (e.g. 1),
            # not the number of supervised tokens. In that case, upstream may rescale the
            # model's per-token mean loss by (token_count / num_items_in_batch), effectively
            # turning it into a token-sum and making the logged loss depend on packing length.
            #
            # For Stage-1 (Scheme A), we want all loss terms to be mean-normalized:
            #   - base CE: mean over *non-coord* supervised tokens (coord targets are masked)
            #   - coord loss: mean over coord-token positions (handled below)
            if num_items_in_batch is not None:
                passed_num_items = self._count_supervised_tokens(masked_labels)
        except Exception:
            passed_num_items = num_items_in_batch

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=passed_num_items
        )

        try:
            loss = self._maybe_add_coord_softce_w1_loss(
                loss=loss,
                outputs=outputs,
                labels=labels_orig,
                masked_labels=masked_labels,
                coord_token_ids=coord_token_ids,
                num_items_in_batch=passed_num_items,
            )
        except Exception as exc:
            if not getattr(self, "_coord_softce_w1_error_warned", False):
                logger.warning(
                    "coord_soft_ce_w1 loss computation failed; aborting to avoid silently "
                    "training without coord supervision.",
                    exc_info=True,
                )
                setattr(self, "_coord_softce_w1_error_warned", True)
            raise
        finally:
            inputs["labels"] = labels_orig

        return (loss, outputs) if return_outputs else loss

    def _maybe_add_coord_softce_w1_loss(
        self,
        *,
        loss: torch.Tensor,
        outputs: Any,
        labels: torch.Tensor,
        masked_labels: torch.Tensor,
        coord_token_ids: list[int],
        num_items_in_batch: Any,
    ) -> torch.Tensor:
        cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return loss

        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            return loss

        seq_len = min(logits.shape[1], max(labels.shape[1] - 1, 0))
        if seq_len <= 0:
            return loss

        logits_next = logits[:, :seq_len, :]
        labels_next = labels[:, 1 : seq_len + 1]
        masked_labels_next = masked_labels[:, 1 : seq_len + 1]

        vocab_size = int(logits_next.shape[-1])
        if max(coord_token_ids) >= vocab_size:
            raise ValueError(
                f"coord_soft_ce_w1 enabled but coord token ids exceed vocab_size={vocab_size}. "
                "Ensure the tokenizer provides a full 1000-bin coord vocab."
            )

        coord_id_map = self._get_coord_id_map(vocab_size, logits_next.device)
        if coord_id_map is None:
            return loss

        labels_safe = labels_next
        if labels_safe.numel() > 0 and labels_safe.min().item() < 0:
            labels_safe = labels_safe.clamp(min=0)
        target_bins_all = coord_id_map[labels_safe].to(dtype=torch.long)
        coord_positions_mask = (target_bins_all >= 0) & (labels_next != -100)
        if not coord_positions_mask.any().item():
            return loss

        flat_logits_full = logits_next[coord_positions_mask]
        flat_target_bins = target_bins_all[coord_positions_mask]

        idx = torch.tensor(
            coord_token_ids, device=flat_logits_full.device, dtype=torch.long
        )
        flat_logits = flat_logits_full.index_select(-1, idx)

        temperature = float(getattr(cfg, "temperature", 1.0))
        soft_ce_weight = float(getattr(cfg, "soft_ce_weight", 1.0))
        w1_weight = float(getattr(cfg, "w1_weight", 1.0))
        gate_weight = float(getattr(cfg, "gate_weight", 0.0))

        out = coord_soft_ce_w1(
            flat_logits,
            flat_target_bins,
            sigma=float(getattr(cfg, "target_sigma", 2.0)),
            truncate=getattr(cfg, "target_truncate", None),
            temperature=temperature,
            # Weighting is applied at the trainer level so we can include the gate term.
            soft_ce_weight=1.0,
            w1_weight=1.0,
            normalize_w1=True,
        )

        softce_sum = out.soft_ce_per_token.sum()
        w1_sum = out.w1_per_token.sum()
        gate_sum = softce_sum.new_tensor(0.0)
        gate_mass_mean = None
        if gate_weight != 0.0:
            gate_per_token = self._coord_vocab_gate_loss(
                flat_logits_full, flat_logits, temperature=temperature
            )
            gate_sum = gate_per_token.sum()
            with torch.no_grad():
                gate_mass_mean = torch.exp((-gate_per_token).clamp(min=-50.0, max=50.0)).mean()
        # Coord supervision is averaged over coord-token positions only (not over
        # non-coord tokens), so its scale is independent of sequence packing length.
        denom = coord_positions_mask.sum().to(dtype=torch.float32)
        if (
            getattr(getattr(self, "args", None), "average_tokens_across_devices", False)
            and getattr(self, "model_accepts_loss_kwargs", False)
            and dist.is_available()
            and dist.is_initialized()
        ):
            denom_global = denom.detach().clone()
            dist.all_reduce(denom_global, op=dist.ReduceOp.SUM)
            denom = denom_global
        denom = torch.where(denom > 0, denom, denom.new_tensor(1.0))

        softce_loss = softce_sum / denom
        w1_loss = w1_sum / denom
        gate_loss = gate_sum / denom

        softce_contrib = soft_ce_weight * softce_loss
        w1_contrib = w1_weight * w1_loss
        gate_contrib = gate_weight * gate_loss
        coord_loss = softce_contrib + w1_contrib + gate_contrib

        if (
            getattr(getattr(self, "args", None), "average_tokens_across_devices", False)
            and getattr(self, "model_accepts_loss_kwargs", False)
        ):
            try:
                if dist.is_available() and dist.is_initialized():
                    scale = float(dist.get_world_size())
                else:
                    scale = float(self.accelerator.num_processes)
            except Exception:
                scale = 1.0
            coord_loss = coord_loss * scale
            softce_contrib = softce_contrib * scale
            w1_contrib = w1_contrib * scale
            gate_contrib = gate_contrib * scale

        coord_loss = torch.nan_to_num(coord_loss, nan=0.0, posinf=1e4, neginf=0.0)
        softce_contrib = torch.nan_to_num(softce_contrib, nan=0.0, posinf=1e4, neginf=0.0)
        w1_contrib = torch.nan_to_num(w1_contrib, nan=0.0, posinf=1e4, neginf=0.0)
        gate_contrib = torch.nan_to_num(gate_contrib, nan=0.0, posinf=1e4, neginf=0.0)

        self._log_coord_softce_w1_metrics(
            logits_next=logits_next,
            labels_next=labels_next,
            coord_positions_mask=coord_positions_mask,
            loss_total=coord_loss,
            loss_softce=softce_contrib,
            loss_w1=w1_contrib,
            loss_gate=gate_contrib,
            gate_mass_mean=gate_mass_mean,
        )

        return loss + coord_loss.to(dtype=loss.dtype)

    def _log_coord_softce_w1_metrics(
        self,
        *,
        logits_next: torch.Tensor,
        labels_next: torch.Tensor,
        coord_positions_mask: torch.Tensor,
        loss_total: torch.Tensor,
        loss_softce: torch.Tensor,
        loss_w1: torch.Tensor,
        loss_gate: torch.Tensor,
        gate_mass_mean: torch.Tensor | None,
    ) -> None:
        mode = (
            "train"
            if getattr(self, "model", None) is None or self.model.training
            else "eval"
        )  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        metrics["coord_softce_w1/loss"].update(float(loss_total.detach().cpu().item()))
        metrics["coord_softce_w1/soft_ce"].update(float(loss_softce.detach().cpu().item()))
        metrics["coord_softce_w1/w1"].update(float(loss_w1.detach().cpu().item()))
        metrics["coord_softce_w1/gate"].update(float(loss_gate.detach().cpu().item()))
        metrics["coord_softce_w1/coord_tokens"].update(
            float(int(coord_positions_mask.sum().detach().item()))
        )
        if gate_mass_mean is not None:
            metrics["coord_softce_w1/coord_vocab_mass"].update(
                float(gate_mass_mean.detach().cpu().item())
            )

    def _coord_vocab_gate_loss(
        self,
        logits_full: torch.Tensor,
        logits_coord: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        """Negative log probability mass of the coord sub-vocabulary.

        For each position:
          mass_coord = sum_{i in coord_vocab} softmax(logits_full / T)[i]
          gate_loss = -log(mass_coord) = logsumexp(all/T) - logsumexp(coord/T)

        Args:
            logits_full: [N, V] full-vocab logits at coord positions
            logits_coord: [N, K] coord-only logits (same positions, sliced)
            temperature: softmax temperature (>0)

        Returns:
            gate_loss_per_token: [N] non-negative tensor (fp32), NaN-safe
        """

        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        full = torch.nan_to_num(
            logits_full.float(), nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(min=-1e4, max=1e4) / float(temperature)
        coord = torch.nan_to_num(
            logits_coord.float(), nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(min=-1e4, max=1e4) / float(temperature)
        lse_all = torch.logsumexp(full, dim=-1)
        lse_coord = torch.logsumexp(coord, dim=-1)
        loss = (lse_all - lse_coord).clamp(min=0.0)
        return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

    def _count_supervised_tokens(self, labels: torch.Tensor) -> int:
        if labels.ndim < 2:
            return 0
        labels_next = labels[:, 1:]
        return int((labels_next != -100).sum().detach().item())

    def _mask_coord_targets(
        self, labels: torch.Tensor, coord_token_ids: list[int]
    ) -> torch.Tensor:
        if labels.numel() == 0:
            return labels
        labels_safe = labels
        if labels_safe.min().item() < 0:
            labels_safe = labels_safe.clamp(min=0)
        max_label = int(labels_safe.max().item()) if labels_safe.numel() else -1
        max_coord = max(coord_token_ids) if coord_token_ids else -1
        size = max(max_label, max_coord) + 1
        if size <= 0:
            return labels

        lookup = torch.zeros(int(size), dtype=torch.bool, device=labels.device)
        ids = torch.tensor(coord_token_ids, device=labels.device, dtype=torch.long)
        valid = (ids >= 0) & (ids < lookup.numel())
        if valid.any().item():
            lookup[ids[valid]] = True

        mask = lookup[labels_safe] & (labels != -100)
        if not mask.any().item():
            return labels
        out = labels.clone()
        out[mask] = -100
        return out

    def _get_coord_token_ids(self) -> list[int]:
        cached = getattr(self, "_coord_token_ids", None)
        if cached is not None:
            return cached
        tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
        if tokenizer is None:
            return []
        ids = get_coord_token_ids(tokenizer)
        setattr(self, "_coord_token_ids", ids)
        return ids

    def _get_coord_id_map(
        self, vocab_size: int, device: torch.device
    ) -> torch.Tensor | None:
        cache = getattr(self, "_coord_id_map_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_coord_id_map_cache", cache)
        key = (str(device), int(vocab_size))
        if key in cache:
            return cache[key]
        coord_token_ids = self._get_coord_token_ids()
        if not coord_token_ids:
            return None
        id_map = torch.full((int(vocab_size),), -1, dtype=torch.long, device=device)
        coord_ids = torch.tensor(coord_token_ids, device=device, dtype=torch.long)
        values = torch.arange(coord_ids.numel(), device=device, dtype=torch.long)
        valid = coord_ids < int(vocab_size)
        if valid.any().item():
            id_map[coord_ids[valid]] = values[valid]
        cache[key] = id_map
        return id_map


class InstabilityMonitorMixin:
    """Per-step monitor + optional guard for catastrophic batches.

    Enabled via `custom.instability_monitor` (stored under `CustomConfig.extra`).
    The collator attaches `instability_meta_json` (batch sample_id/base_idx info),
    which this mixin consumes before model forward.

    When triggered, writes a JSON event to `<output_dir>/instability_dumps/events.jsonl`
    (or `dump_dir` if provided). If guard is enabled, replaces the loss with 0.0 to
    avoid poisoning weights for the current step.
    """

    meta_field = "instability_meta_json"

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        meta_json = inputs.pop(self.meta_field, None)
        labels_snapshot = inputs.get("labels")

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            loss = self._monitor_and_guard(
                loss=loss, outputs=outputs, labels=labels_snapshot, meta_json=meta_json
            )
        except Exception:
            # Best-effort only; never block training.
            pass

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(out):
            return float(default)
        return float(out)

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _get_cfg(self) -> Mapping[str, Any]:
        raw = getattr(self, "instability_monitor_cfg", None)
        return raw if isinstance(raw, Mapping) else {}

    def _rank(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0
        try:
            return int(dist.get_rank())
        except Exception:
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
                        except Exception:
                            out[current] = {"_raw": raw, "_parse_error": True}
                        target_pos += 1
                        if target_pos >= len(wanted):
                            break
                        target = wanted[target_pos]
                    current += 1
        except Exception as exc:
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
        except Exception:
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
                except Exception:
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
        ema = float(value) if prev is None else float(decay) * float(prev) + (1.0 - float(decay)) * float(value)
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
                    pass
                try:
                    control.should_epoch_stop = True
                except Exception:
                    pass
        except Exception:
            pass
        try:
            state = getattr(self, "state", None)
            if state is not None:
                setattr(state, "should_training_stop", True)
        except Exception:
            pass

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
            "train"
            if getattr(self, "model", None) is None or self.model.training
            else "eval"
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
            if loss_val >= abs_loss_threshold and ema_loss > 0 and loss_val > float(spike_factor) * float(ema_loss):
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

        # Decide whether to request an early stop. Default is off, but when enabled we
        # stop on zero token acc (strong crash indicator) to save time.
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
                pass

        if self._is_main_process():
            ema_s = None if ema_loss is None else f"{float(ema_loss):.6f}"
            acc_s = None if token_acc is None else f"{float(token_acc):.4f}"
            maxlog_s = (
                None if max_logit_abs is None else f"{float(max_logit_abs):.2f}"
            )
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
                pass

        if not guard_enabled:
            return loss

        # IMPORTANT: return a differentiable zero. A detached 0.0 breaks
        # DeepSpeed/Accelerate backward with:
        #   "element 0 of tensors does not require grad and does not have a grad_fn"
        #
        # Also avoid `loss * 0.0` when loss is NaN/Inf (NaN*0 -> NaN).
        logits = getattr(outputs, "logits", None)
        if isinstance(logits, torch.Tensor):
            safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            return safe_logits.sum() * 0.0
        return loss * 0.0
