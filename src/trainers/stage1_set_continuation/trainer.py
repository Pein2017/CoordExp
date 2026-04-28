from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from swift.trainers import TrainerFactory

from src.coord_tokens.codec import get_coord_token_ids
from src.trainers.metrics.mixins import _validate_batch_contract
from src.trainers.teacher_forcing.forwards import (
    assert_unsliced_logits,
    prepare_forward_inputs,
    run_no_cache_forward,
)

from .branch_scorer import (
    LogitsMode,
    TensorBranchScoreInput,
    crop_tensors_for_logits,
    score_tensor_batch_retained,
    score_branch_checkpointed_exact,
    score_branch_retained_graph,
    supervised_suffix_start,
)
from .branch_batcher import BranchBatchWorkItem, plan_smart_branch_batches
from .branch_encoder import EncodedSetContinuationBranch, encode_set_continuation_branch
from .budget import plan_candidate_execution
from .losses import (
    CandidateLogProbResult,
    compute_close_sequence_nll,
    compute_close_start_nll,
    compute_mp_pem_losses,
)
from .metrics import (
    BRANCH_ISOLATION_CODES,
    BRANCH_BATCH_SCHEDULER_CODES,
    BRANCH_RUNTIME_MODE_CODES,
    CANDIDATE_SCORING_MODE_CODES,
    DDP_CANDIDATE_PADDING_POLICY_CODES,
    LOGZ_ESTIMATOR_CODES,
    PREFIX_ATTACH_MODE_CODES,
    PREFIX_GRADIENT_CODES,
    emit_stage1_set_continuation_metrics,
    mean_numeric_metrics,
    metric_code,
)
from .sampling import Stage1SetContinuationSample, sample_subset_and_candidates


class Stage1SetContinuationTrainer(
    TrainerFactory.get_cls(
        type("Args", (), {"task_type": "causal_lm"})(),
        TrainerFactory.TRAINER_MAPPING,
    )
):
    """Stage-1 subset-conditioned full-entry multi-positive trainer.

    V1 intentionally uses repeated independent forwards. Prefix gradients are
    recomputed per branch and are not detached; no prefix cache or branch mask is
    used.
    """

    _coord_token_ids: list[int] | None = None

    def _cfg(self) -> Any:
        cfg = getattr(self, "stage1_set_continuation_cfg", None)
        if cfg is None:
            raise ValueError(
                "stage1_set_continuation trainer requires stage1_set_continuation_cfg"
            )
        return cfg

    def _get_coord_token_ids(self) -> list[int]:
        if self._coord_token_ids is not None:
            return self._coord_token_ids
        tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
        if tokenizer is None:
            raise ValueError("set-continuation trainer requires template.tokenizer")
        ids = get_coord_token_ids(tokenizer, validate=True)
        self._coord_token_ids = [int(token_id) for token_id in ids]
        return self._coord_token_ids

    def _get_coord_id_map(
        self,
        *,
        vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache = getattr(self, "_coord_id_map_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_coord_id_map_cache", cache)
        key = (str(device), int(vocab_size))
        if key in cache:
            return cache[key]
        from src.trainers.losses.coord_soft_ce_w1 import build_coord_id_map

        coord_id_map = build_coord_id_map(
            vocab_size=int(vocab_size),
            device=device,
            coord_token_ids=self._get_coord_token_ids(),
        )
        cache[key] = coord_id_map
        return coord_id_map

    def _special_token_ids_for_gate(self, *, device: torch.device) -> torch.Tensor:
        cache = getattr(self, "_stage1_set_continuation_special_token_ids", None)
        if not isinstance(cache, list):
            tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
            ids: set[int] = set()
            raw_special_ids = getattr(tokenizer, "all_special_ids", None)
            if isinstance(raw_special_ids, Sequence):
                for token_id in cast(Sequence[Any], raw_special_ids):
                    if isinstance(token_id, int) and token_id >= 0:
                        ids.add(int(token_id))
            for attr in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
            ):
                token_id = getattr(tokenizer, attr, None)
                if isinstance(token_id, int) and token_id >= 0:
                    ids.add(int(token_id))
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                for token in ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>"):
                    token_id = convert(token)
                    if isinstance(token_id, int) and token_id >= 0:
                        ids.add(int(token_id))
            cache = sorted(ids)
            setattr(self, "_stage1_set_continuation_special_token_ids", cache)
        return torch.tensor(cache, dtype=torch.long, device=device)

    def _bidirectional_gate_kwargs(self, *, device: torch.device) -> dict[str, Any]:
        gate_cfg = getattr(self._cfg(), "bidirectional_token_gate", None)
        enabled = bool(gate_cfg is not None and getattr(gate_cfg, "enabled", False))
        return {
            "bidirectional_gate_enabled": enabled,
            "bidirectional_gate_temperature": float(
                getattr(gate_cfg, "temperature", 1.0) or 1.0
            ),
            "special_token_ids": (
                self._special_token_ids_for_gate(device=device) if enabled else None
            ),
        }

    def _ddp_max_int(self, value: int, model: Any) -> int:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() <= 1
        ):
            return int(value)
        device = None
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                device = parameter.device
                break
        if device is None:
            device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
        tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return int(tensor.item())

    def _seed_parts(
        self, meta: Mapping[str, Any], sample_offset: int
    ) -> tuple[Any, ...]:
        args = getattr(self, "args", None)
        state = getattr(self, "state", None)
        seed = int(getattr(args, "seed", 0) or 0)
        rank_raw = getattr(args, "process_index", None)
        if rank_raw is None:
            rank_raw = getattr(args, "local_rank", None)
        if rank_raw is None or int(rank_raw) < 0:
            rank_raw = os.environ.get("RANK", "0")
        rank = int(rank_raw or 0)
        sample_identity = (
            meta.get("sample_id")
            if meta.get("sample_id") is not None
            else meta.get("base_idx", sample_offset)
        )
        return (
            seed,
            getattr(state, "epoch", 0) or 0,
            getattr(state, "global_step", 0) or 0,
            sample_identity,
            rank,
            sample_offset,
        )

    def _prepare_branch_inputs(
        self,
        *,
        model: Any,
        branch: EncodedSetContinuationBranch,
    ) -> dict[str, Any]:
        branch_inputs = dict(branch.branch_inputs)
        _validate_batch_contract(
            model=model, inputs=branch_inputs, template=self.template
        )
        prepare_inputs = getattr(self, "_prepare_inputs", None)
        if callable(prepare_inputs) and hasattr(getattr(self, "args", None), "device"):
            branch_inputs = dict(prepare_inputs(branch_inputs))
        return branch_inputs

    def _forward_branch(
        self,
        *,
        model: Any,
        branch: EncodedSetContinuationBranch,
        logits_to_keep: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        branch_inputs = self._prepare_branch_inputs(model=model, branch=branch)
        ignored_keys = {
            "labels",
            "compute_loss_func",
            "loss_scale",
            "text_position_ids",
            "channel",
            "logits_to_keep",
        }
        _core_model, inputs_for_model, _model_type = prepare_forward_inputs(
            model=model,
            inputs=branch_inputs,
            ignored_keys=tuple(ignored_keys),
            packing_enabled=False,
            where="stage1-set-continuation",
        )
        if logits_to_keep is not None:
            inputs_for_model["logits_to_keep"] = int(logits_to_keep)
        outputs = run_no_cache_forward(model=model, inputs_for_model=inputs_for_model)
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise ValueError(
                "stage1-set-continuation model forward did not return logits"
            )
        input_ids = branch_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError(
                "encoded set-continuation branch requires input_ids tensor"
            )
        if logits_to_keep is None:
            assert_unsliced_logits(
                logits=logits,
                input_ids=input_ids,
                where="stage1-set-continuation training",
            )
        elif int(logits.shape[1]) != int(logits_to_keep):
            raise ValueError(
                "stage1-set-continuation training: model returned logits with "
                f"sequence length {int(logits.shape[1])}, expected logits_to_keep="
                f"{int(logits_to_keep)}"
            )
        return outputs, branch_inputs

    def _encode_branch(
        self,
        *,
        meta: Mapping[str, Any],
        prefix_indices: tuple[int, ...],
        candidate_index: int | None,
    ) -> EncodedSetContinuationBranch:
        return encode_set_continuation_branch(
            meta=meta,
            template=self.template,
            prefix_indices=prefix_indices,
            candidate_index=candidate_index,
            object_field_order=str(getattr(self, "object_field_order", "desc_first")),
        )

    def _score_candidate_branch(
        self,
        *,
        model: Any,
        branch: EncodedSetContinuationBranch,
        coord_token_ids: torch.Tensor,
    ) -> tuple[CandidateLogProbResult, Any, dict[str, Any]]:
        train_forward = self._cfg().train_forward
        runtime = train_forward.branch_runtime
        logits_mode = cast(LogitsMode, str(train_forward.logits.mode))
        gate_kwargs = self._bidirectional_gate_kwargs(device=coord_token_ids.device)
        if runtime.mode == "retained_graph":
            bundle = score_branch_retained_graph(
                trainer=self,
                model=model,
                branch=branch,
                coord_token_ids=coord_token_ids,
                logits_mode=logits_mode,
                **gate_kwargs,
            )
            return bundle.logprob, bundle.outputs, bundle.branch_inputs
        if runtime.mode == "checkpointed_exact":
            bundle = score_branch_checkpointed_exact(
                trainer=self,
                model=model,
                branch=branch,
                coord_token_ids=coord_token_ids,
                use_reentrant=runtime.checkpoint_use_reentrant,
                preserve_rng_state=runtime.preserve_rng_state,
                logits_mode=logits_mode,
                **gate_kwargs,
            )
            return bundle.logprob, bundle.outputs, bundle.branch_inputs
        if runtime.mode == "smart_batched_exact":
            bundle = score_branch_retained_graph(
                trainer=self,
                model=model,
                branch=branch,
                coord_token_ids=coord_token_ids,
                logits_mode=logits_mode,
                **gate_kwargs,
            )
            return bundle.logprob, bundle.outputs, bundle.branch_inputs
        raise ValueError(f"unsupported branch runtime mode: {runtime.mode}")

    def _tensor_branch_score_input(
        self,
        *,
        model: Any,
        branch: EncodedSetContinuationBranch,
    ) -> TensorBranchScoreInput:
        branch_inputs = self._prepare_branch_inputs(model=model, branch=branch)
        _core_model, inputs_for_model, _model_type = prepare_forward_inputs(
            model=model,
            inputs=branch_inputs,
            ignored_keys=(
                "labels",
                "compute_loss_func",
                "loss_scale",
                "text_position_ids",
                "channel",
                "logits_to_keep",
            ),
            packing_enabled=False,
            where="stage1-set-continuation-smart-batched",
        )
        tensor_inputs = {
            key: value
            for key, value in inputs_for_model.items()
            if isinstance(value, torch.Tensor)
        }
        return TensorBranchScoreInput(
            model_inputs=tensor_inputs,
            labels=branch_inputs["labels"],
            candidate_entry_label_mask=branch.candidate_entry_label_mask,
            coord_label_mask=branch.coord_label_mask,
            objective_label_mask=branch.objective_label_mask,
            schema_open_label_mask=branch.schema_open_label_mask,
            json_structural_label_mask=branch.json_structural_label_mask,
        )

    def _score_candidate_branches_smart_batched(
        self,
        *,
        model: Any,
        branches: Sequence[EncodedSetContinuationBranch],
        coord_token_ids: torch.Tensor,
    ) -> tuple[list[CandidateLogProbResult], dict[str, float]]:
        cfg = self._cfg()
        train_forward = cfg.train_forward
        logits_mode = cast(LogitsMode, str(train_forward.logits.mode))
        batching = train_forward.branch_batching
        if self._branch_aux_enabled():
            raise ValueError(
                "smart_batched_exact does not yet support branch-local aux losses; "
                "use retained_graph or disable branch-local aux modules"
            )
        score_inputs = [
            self._tensor_branch_score_input(model=model, branch=branch)
            for branch in branches
        ]
        work_items: list[BranchBatchWorkItem] = []
        for index, item in enumerate(score_inputs):
            suffix_start = (
                supervised_suffix_start(
                    labels=item.labels,
                    supervised_label_mask=item.candidate_entry_label_mask,
                )
                if logits_mode == "supervised_suffix"
                else 0
            )
            sequence_length = int(item.labels.shape[-1])
            work_items.append(
                BranchBatchWorkItem(
                    index=index,
                    sequence_length=sequence_length,
                    suffix_keep=max(1, sequence_length - suffix_start),
                )
            )
        plan = plan_smart_branch_batches(
            work_items,
            max_branch_rows=(
                batching.max_branch_rows if bool(batching.enabled) else len(work_items)
            ),
            max_branch_tokens=(
                batching.max_branch_tokens if bool(batching.enabled) else None
            ),
            min_fill_ratio=float(batching.min_fill_ratio),
        )
        results: list[CandidateLogProbResult | None] = [None for _ in score_inputs]
        gate_kwargs = self._bidirectional_gate_kwargs(device=coord_token_ids.device)

        def _no_cache_forward(**kwargs: torch.Tensor) -> Any:
            return run_no_cache_forward(
                model=model,
                inputs_for_model=dict(kwargs),
            )

        for batch in plan.batches:
            batch_inputs = [score_inputs[item.index] for item in batch.items]
            batch_results = score_tensor_batch_retained(
                model=model,
                items=batch_inputs,
                coord_token_ids=coord_token_ids,
                logits_mode=logits_mode,
                forward_fn=_no_cache_forward,
                **gate_kwargs,
            )
            for item, result in zip(batch.items, batch_results, strict=True):
                results[item.index] = result
        resolved = [result for result in results if result is not None]
        if len(resolved) != len(score_inputs):
            raise ValueError("smart branch batching did not score every candidate")
        metrics = {
            "mp/smart_batched_branch_forwards": float(plan.batch_count),
            "mp/branch_batch_count": float(plan.batch_count),
            "mp/branch_batch_rows_mean": plan.rows_mean,
            "mp/branch_batch_rows_max": plan.rows_max,
            "mp/branch_batch_tokens_mean": plan.tokens_mean,
            "mp/branch_batch_tokens_max": plan.tokens_max,
            "mp/branch_batch_padding_fraction": float(plan.padding_fraction),
            "mp/branch_batch_scheduler": metric_code(
                plan.scheduler,
                BRANCH_BATCH_SCHEDULER_CODES,
                metric_name="mp/branch_batch_scheduler",
            ),
        }
        return resolved, metrics

    def _candidate_scoped_labels(
        self,
        *,
        labels: torch.Tensor,
        branch: EncodedSetContinuationBranch,
    ) -> torch.Tensor:
        candidate_mask = branch.candidate_object_label_mask.to(device=labels.device)
        if tuple(candidate_mask.shape) != tuple(labels.shape):
            raise ValueError(
                "candidate_object_label_mask must align with branch labels for aux losses"
            )
        scoped = labels.clone()
        scoped[~candidate_mask] = -100
        return scoped

    def _add_aux_metric_atom(
        self,
        accum: dict[str, dict[str, Any]],
        *,
        name: str,
        loss: torch.Tensor | None,
        position_count: int,
        skipped: bool,
    ) -> None:
        state = accum.setdefault(
            name,
            {
                "losses": [],
                "candidate_count": 0.0,
                "position_count": 0.0,
                "skipped_candidates": 0.0,
                "contributing_candidates": 0.0,
            },
        )
        state["candidate_count"] = float(state["candidate_count"]) + 1.0
        if skipped or not isinstance(loss, torch.Tensor):
            state["skipped_candidates"] = float(state["skipped_candidates"]) + 1.0
            return
        state["losses"].append(loss)
        state["position_count"] = float(state["position_count"]) + float(position_count)
        state["contributing_candidates"] = float(state["contributing_candidates"]) + 1.0

    def _branch_aux_enabled(self) -> bool:
        return any(
            bool(getattr(getattr(self, attr, None), "enabled", False))
            for attr in (
                "coord_soft_ce_w1_cfg",
                "bbox_geo_cfg",
                "bbox_size_aux_cfg",
            )
        )

    def _compute_candidate_aux_atoms(
        self,
        *,
        branch: EncodedSetContinuationBranch,
        branch_inputs: Mapping[str, Any],
        outputs: Any,
        aux_accum: dict[str, dict[str, Any]],
    ) -> None:
        if not self._branch_aux_enabled():
            return
        logits = getattr(outputs, "logits", None)
        labels = branch_inputs.get("labels")
        if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError(
                "branch-local aux losses require logits and labels tensors"
            )
        labels = labels.to(device=logits.device)
        candidate_labels = self._candidate_scoped_labels(labels=labels, branch=branch)
        coord_token_ids = self._get_coord_token_ids()
        coord_id_map = self._get_coord_id_map(
            vocab_size=int(logits.shape[-1]),
            device=logits.device,
        )
        tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
        object_field_order = str(
            getattr(self, "object_field_order", "desc_first") or "desc_first"
        )
        bbox_format = str(getattr(self, "bbox_format", "xyxy") or "xyxy")
        coord_cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
        if coord_cfg is not None and bool(getattr(coord_cfg, "enabled", False)):
            from src.trainers.losses.coord_soft_ce_w1 import (
                compute_coord_soft_ce_w1_loss,
            )
            from src.trainers.teacher_forcing.stage1 import (
                mask_stage1_coord_targets,
            )

            masked_labels = mask_stage1_coord_targets(candidate_labels, coord_token_ids)
            result = compute_coord_soft_ce_w1_loss(
                logits=logits,
                labels=candidate_labels,
                masked_labels=masked_labels,
                coord_token_weights=None,
                coord_token_ids=coord_token_ids,
                coord_id_map=coord_id_map,
                tokenizer=tokenizer,
                token_types=None,
                cfg=coord_cfg,
                average_tokens_across_devices=False,
                model_accepts_loss_kwargs=False,
                accelerator_num_processes=None,
                object_field_order=object_field_order,
                bbox_format=bbox_format,
            )
            self._add_aux_metric_atom(
                aux_accum,
                name="coord_soft_ce_w1",
                loss=getattr(result, "coord_loss", None),
                position_count=int(getattr(result, "coord_tokens", 0) or 0)
                if result is not None
                else 0,
                skipped=result is None,
            )

        bbox_geo_cfg = getattr(self, "bbox_geo_cfg", None)
        if bbox_geo_cfg is not None and bool(getattr(bbox_geo_cfg, "enabled", False)):
            from src.trainers.losses.bbox_geo import compute_stage1_bbox_geo_loss

            decode_temperature = float(getattr(coord_cfg, "temperature", 1.0) or 1.0)
            result = compute_stage1_bbox_geo_loss(
                logits=logits,
                labels=candidate_labels,
                coord_token_ids=coord_token_ids,
                coord_id_map=coord_id_map,
                tokenizer=tokenizer,
                cfg=bbox_geo_cfg,
                decode_temperature=decode_temperature,
                object_field_order=object_field_order,
                bbox_format=bbox_format,
            )
            self._add_aux_metric_atom(
                aux_accum,
                name="bbox_geo",
                loss=getattr(result, "total_loss", None),
                position_count=int(getattr(result, "coord_slots", 0) or 0)
                if result is not None
                else 0,
                skipped=result is None,
            )

        bbox_size_cfg = getattr(self, "bbox_size_aux_cfg", None)
        if bbox_size_cfg is not None and bool(getattr(bbox_size_cfg, "enabled", False)):
            from src.trainers.losses.bbox_size_aux import (
                compute_stage1_bbox_size_aux_loss,
            )

            decode_temperature = float(getattr(coord_cfg, "temperature", 1.0) or 1.0)
            result = compute_stage1_bbox_size_aux_loss(
                logits=logits,
                labels=candidate_labels,
                coord_token_ids=coord_token_ids,
                coord_id_map=coord_id_map,
                tokenizer=tokenizer,
                cfg=bbox_size_cfg,
                decode_temperature=decode_temperature,
                object_field_order=object_field_order,
                bbox_format=bbox_format,
            )
            self._add_aux_metric_atom(
                aux_accum,
                name="bbox_size",
                loss=getattr(result, "total_loss", None),
                position_count=int(getattr(result, "coord_slots", 0) or 0)
                if result is not None
                else 0,
                skipped=result is None,
            )

    def _aggregate_candidate_aux_atoms(
        self,
        aux_accum: dict[str, dict[str, Any]],
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        total_aux_loss: torch.Tensor | None = None
        metrics: dict[str, Any] = {}
        for name, state in sorted(aux_accum.items()):
            losses = state.get("losses", [])
            if losses:
                loss = torch.stack(list(losses)).mean()
                total_aux_loss = (
                    loss if total_aux_loss is None else total_aux_loss + loss
                )
                metrics[f"loss/aux_{name}"] = float(loss.detach().item())
            else:
                metrics[f"loss/aux_{name}"] = 0.0
            metrics[f"aux/{name}/candidate_count"] = float(
                state.get("candidate_count", 0.0)
            )
            metrics[f"aux/{name}/position_count"] = float(
                state.get("position_count", 0.0)
            )
            metrics[f"aux/{name}/skipped_candidates"] = float(
                state.get("skipped_candidates", 0.0)
            )
            metrics[f"aux/{name}/contributing_candidates"] = float(
                state.get("contributing_candidates", 0.0)
            )
        return total_aux_loss, metrics

    def _close_branch_stats(
        self,
        *,
        model: Any,
        meta: Mapping[str, Any],
        prefix_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        branch = self._encode_branch(
            meta=meta,
            prefix_indices=prefix_indices,
            candidate_index=None,
        )
        if not bool(branch.structural_close_start_mask.any().item()):
            raise ValueError("structural close-start mask is empty")
        logits_mode = str(self._cfg().train_forward.logits.mode)
        suffix_start = (
            supervised_suffix_start(
                labels=branch.labels,
                supervised_label_mask=branch.structural_close_sequence_mask,
            )
            if logits_mode == "supervised_suffix"
            else 0
        )
        logits_to_keep = (
            max(1, int(branch.labels.shape[-1]) - int(suffix_start))
            if logits_mode == "supervised_suffix"
            else None
        )
        outputs, branch_inputs = self._forward_branch(
            model=model,
            branch=branch,
            logits_to_keep=logits_to_keep,
        )
        logits = outputs.logits
        labels = branch_inputs["labels"].to(device=logits.device)
        close_start_mask = branch.structural_close_start_mask.to(device=logits.device)
        close_sequence_mask = branch.structural_close_sequence_mask.to(
            device=logits.device
        )
        prefix_token_positions = close_start_mask.nonzero(as_tuple=False)
        prefix_tokens = (
            int(prefix_token_positions[:, 1].min().detach().cpu().item())
            if int(prefix_token_positions.shape[0]) > 0
            else 0
        )
        (
            labels,
            close_start_mask,
            close_sequence_mask,
            _objective_mask,
            _schema_open_mask,
            _json_structural_mask,
        ) = crop_tensors_for_logits(
            suffix_start=suffix_start,
            labels=labels,
            candidate_entry_label_mask=close_start_mask,
            coord_label_mask=close_sequence_mask,
        )
        close_start_nll = compute_close_start_nll(
            logits=logits,
            labels=labels,
            structural_close_start_mask=close_start_mask,
        )
        close_sequence_nll = compute_close_sequence_nll(
            logits=logits,
            labels=labels,
            structural_close_sequence_mask=close_sequence_mask,
        )
        final_mask = torch.zeros_like(close_sequence_mask)
        true_positions = close_sequence_mask.nonzero(as_tuple=False)
        if int(true_positions.shape[0]) > 0:
            final_row = true_positions[-1]
            final_mask[int(final_row[0]), int(final_row[1])] = True
        final_token_nll = compute_close_sequence_nll(
            logits=logits,
            labels=labels,
            structural_close_sequence_mask=final_mask,
        )
        p_close_start = torch.exp(-close_start_nll).clamp(min=0.0, max=1.0)
        return (
            close_start_nll,
            close_sequence_nll,
            final_token_nll,
            p_close_start,
            prefix_tokens,
            outputs,
        )

    @staticmethod
    def _tensor_stats(values: torch.Tensor, *, prefix: str) -> dict[str, float]:
        if values.numel() == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_std": 0.0,
            }
        std = values.std(unbiased=False) if values.numel() > 1 else values.new_zeros(())
        return {
            f"{prefix}_mean": float(values.mean().detach().item()),
            f"{prefix}_min": float(values.min().detach().item()),
            f"{prefix}_max": float(values.max().detach().item()),
            f"{prefix}_std": float(std.detach().item()),
        }

    @staticmethod
    def _annotation_completeness_weight(*, object_count: int, cfg: Any) -> float:
        if not bool(getattr(cfg, "enabled", False)):
            return 1.0
        by_max_gt_raw = getattr(cfg, "by_max_gt", {})
        if isinstance(by_max_gt_raw, Mapping):
            by_max_gt = cast(Mapping[Any, Any], by_max_gt_raw)
            for max_gt, weight in sorted(
                ((int(key), float(value)) for key, value in by_max_gt.items()),
                key=lambda item: item[0],
            ):
                if int(object_count) <= max_gt:
                    return float(weight)
        return float(getattr(cfg, "default_weight", 1.0))

    @staticmethod
    def _json_structural_aux_loss(
        *,
        candidate_results: Sequence[CandidateLogProbResult],
        weight: float,
    ) -> torch.Tensor | None:
        if not candidate_results or float(weight) <= 0.0:
            return None
        reference = candidate_results[0].score
        total_tokens = sum(
            int(result.json_structural_tokens) for result in candidate_results
        )
        if total_tokens <= 0:
            return reference.new_zeros(())
        total_score = sum(
            (result.json_structural_score for result in candidate_results),
            start=reference.new_zeros(()),
        )
        return (-total_score / max(total_tokens, 1)) * float(weight)

    @staticmethod
    def _bidirectional_gate_aux_loss(
        *,
        candidate_results: Sequence[CandidateLogProbResult],
        gate_cfg: Any,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if not candidate_results or not bool(getattr(gate_cfg, "enabled", False)):
            return None, {}
        reference = candidate_results[0].score
        zero = reference.new_zeros(())
        coord_tokens = sum(
            int(result.coord_gate_tokens) for result in candidate_results
        )
        text_tokens = sum(int(result.text_gate_tokens) for result in candidate_results)
        coord_nll_sum = sum(
            (result.coord_gate_nll_sum for result in candidate_results),
            start=zero,
        )
        text_nll_sum = sum(
            (result.text_gate_nll_sum for result in candidate_results),
            start=zero,
        )
        coord_mass_sum = sum(
            (result.coord_gate_coord_mass_sum for result in candidate_results),
            start=zero,
        )
        text_mass_sum = sum(
            (result.text_gate_coord_mass_sum for result in candidate_results),
            start=zero,
        )
        coord_loss = coord_nll_sum / max(coord_tokens, 1)
        text_loss = text_nll_sum / max(text_tokens, 1)
        total = coord_loss * float(getattr(gate_cfg, "coord_gate_weight", 0.0))
        total = total + text_loss * float(getattr(gate_cfg, "text_gate_weight", 0.0))
        metrics = {
            "loss/coord_gate": float(coord_loss.detach().item()),
            "loss/text_gate": float(text_loss.detach().item()),
            "gate/coord_slot_coord_mass_mean": float(
                (coord_mass_sum / max(coord_tokens, 1)).detach().item()
            ),
            "gate/text_slot_coord_mass_mean": float(
                (text_mass_sum / max(text_tokens, 1)).detach().item()
            ),
            "gate/coord_tokens_count": float(coord_tokens),
            "gate/text_tokens_count": float(text_tokens),
        }
        return total, metrics

    @staticmethod
    def _score_metrics(
        *,
        scores: torch.Tensor,
        candidate_results: list[CandidateLogProbResult],
    ) -> dict[str, float]:
        lengths = scores.new_tensor(
            [max(int(result.tokens), 1) for result in candidate_results],
            dtype=scores.dtype,
        )
        coord_counts = scores.new_tensor(
            [int(result.coord_tokens) for result in candidate_results],
            dtype=scores.dtype,
        )
        non_coord_counts = scores.new_tensor(
            [int(result.non_coord_tokens) for result in candidate_results],
            dtype=scores.dtype,
        )
        coord_scores = torch.stack([result.coord_score for result in candidate_results])
        non_coord_scores = torch.stack(
            [result.non_coord_score for result in candidate_results]
        )
        per_token = scores / lengths.clamp_min(1.0)
        coord_fraction = coord_counts / lengths.clamp_min(1.0)
        metrics: dict[str, float] = {}
        metrics.update(
            Stage1SetContinuationTrainer._tensor_stats(
                lengths,
                prefix="mp/candidate_entry_tokens",
            )
        )
        metrics.update(
            Stage1SetContinuationTrainer._tensor_stats(
                scores,
                prefix="mp/candidate_logprob_sum",
            )
        )
        metrics.update(
            Stage1SetContinuationTrainer._tensor_stats(
                per_token,
                prefix="mp/candidate_logprob_per_token",
            )
        )
        metrics.update(
            Stage1SetContinuationTrainer._tensor_stats(
                coord_fraction,
                prefix="mp/candidate_coord_token_fraction",
            )
        )
        if bool(coord_counts.gt(0).any().item()):
            coord_per = coord_scores[coord_counts > 0] / coord_counts[coord_counts > 0]
            metrics["mp/candidate_logprob_per_coord_token_mean"] = float(
                coord_per.mean().detach().item()
            )
        else:
            metrics["mp/candidate_logprob_per_coord_token_mean"] = 0.0
        if bool(non_coord_counts.gt(0).any().item()):
            non_coord_per = (
                non_coord_scores[non_coord_counts > 0]
                / non_coord_counts[non_coord_counts > 0]
            )
            metrics["mp/candidate_logprob_per_noncoord_token_mean"] = float(
                non_coord_per.mean().detach().item()
            )
        else:
            metrics["mp/candidate_logprob_per_noncoord_token_mean"] = 0.0
        schema_tokens = sum(int(result.schema_tokens) for result in candidate_results)
        schema_score = sum(
            (result.schema_score for result in candidate_results),
            start=scores.new_zeros(()),
        )
        metrics["loss/schema_open"] = float(
            (-schema_score / max(schema_tokens, 1)).detach().item()
        )
        metrics["mp/schema_open_tokens_scored_mean"] = float(
            schema_tokens / max(len(candidate_results), 1)
        )
        json_structural_tokens = sum(
            int(result.json_structural_tokens) for result in candidate_results
        )
        json_structural_score = sum(
            (result.json_structural_score for result in candidate_results),
            start=scores.new_zeros(()),
        )
        metrics["loss/json_structural_raw"] = float(
            (-json_structural_score / max(json_structural_tokens, 1)).detach().item()
        )
        metrics["mp/json_structural_tokens_scored_mean"] = float(
            json_structural_tokens / max(len(candidate_results), 1)
        )
        return metrics

    def _sample_state(
        self,
        *,
        meta: Mapping[str, Any],
        sample_offset: int,
    ) -> Stage1SetContinuationSample:
        payload = meta.get("assistant_payload")
        if not isinstance(payload, Mapping):
            raise ValueError("set-continuation metadata requires assistant_payload")
        objects = payload.get("objects")
        if not isinstance(objects, list):
            raise ValueError("assistant_payload.objects must be a list")
        cfg = self._cfg()
        return sample_subset_and_candidates(
            object_count=len(objects),
            subset_sampling_cfg=cfg.subset_sampling,
            candidates_cfg=cfg.candidates,
            seed_parts=self._seed_parts(meta, sample_offset),
        )

    def _process_sample(
        self,
        *,
        model: Any,
        meta: Mapping[str, Any],
        sample_offset: int,
        coord_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, Any], Any]:
        cfg = self._cfg()
        sample = self._sample_state(meta=meta, sample_offset=sample_offset)
        prefix_indices = tuple(int(index) for index in sample.prefix_indices)
        remaining_indices = tuple(int(index) for index in sample.remaining_indices)
        candidate_indices = tuple(int(index) for index in sample.candidate_indices)
        if remaining_indices and not candidate_indices:
            raise ValueError(
                "set-continuation sampler produced remaining objects but no candidates"
            )
        execution_plan = plan_candidate_execution(
            sample=sample,
            cfg=cfg,
            sample_seed_parts=(
                *self._seed_parts(meta=meta, sample_offset=sample_offset),
                "candidate_budget",
            ),
            prefix_tokens=0,
            candidate_token_lengths=[],
            memory_free_gib=None,
            enabled_budget_kinds=("candidate",),
        )
        candidate_indices = tuple(
            int(index) for index in execution_plan.selected_candidate_indices
        )
        runtime_mode = str(cfg.train_forward.branch_runtime.mode)

        metrics: dict[str, Any] = {
            "mp/num_prefix_objects": float(len(prefix_indices)),
            "mp/num_remaining_objects": float(len(remaining_indices)),
            "mp/num_candidates_scored": float(len(candidate_indices)),
            "mp/scored_candidate_fraction": float(
                len(candidate_indices)
                / max(1, execution_plan.remaining_candidate_count)
            ),
            "mp/samples_with_candidates": float(bool(candidate_indices)),
            "mp/samples_full_prefix": float(len(remaining_indices) == 0),
            "mp/branch_forwards_per_sample": float(len(candidate_indices) + 1),
            "mp/branch_runtime_mode": metric_code(
                runtime_mode,
                BRANCH_RUNTIME_MODE_CODES,
                metric_name="mp/branch_runtime_mode",
            ),
            "mp/retained_graph_branch_forwards": 0.0,
            "mp/checkpointed_branch_forwards": 0.0,
            "mp/smart_batched_branch_forwards": 0.0,
            "mp/branch_batch_count": 0.0,
            "mp/branch_batch_rows_mean": 0.0,
            "mp/branch_batch_rows_max": 0.0,
            "mp/branch_batch_tokens_mean": 0.0,
            "mp/branch_batch_tokens_max": 0.0,
            "mp/branch_batch_padding_fraction": 0.0,
            "mp/branch_batch_scheduler": metric_code(
                "disabled",
                BRANCH_BATCH_SCHEDULER_CODES,
                metric_name="mp/branch_batch_scheduler",
            ),
            "mp/objective_fidelity_exact_samples": float(
                execution_plan.objective_fidelity == "exact"
            ),
            "mp/objective_fidelity_approx_samples": float(
                execution_plan.objective_fidelity != "exact"
            ),
            "mp/fallback_applied_samples": float(execution_plan.fallback_applied),
            "mp/fallback_reason_candidate_budget": float(
                execution_plan.fallback_reason == "candidate_budget"
            ),
            "mp/fallback_reason_token_budget": float(
                execution_plan.fallback_reason == "token_budget"
            ),
            "mp/fallback_reason_memory_budget": float(
                execution_plan.fallback_reason == "memory_budget"
            ),
            "mp/selected_mode_empty_prefix": float(
                sample.selected_mode == "empty_prefix"
            ),
            "mp/selected_mode_random_subset": float(
                sample.selected_mode == "random_subset"
            ),
            "mp/selected_mode_leave_one_out": float(
                sample.selected_mode == "leave_one_out"
            ),
            "mp/selected_mode_full_prefix": float(
                sample.selected_mode == "full_prefix"
            ),
            "mp/candidate_scoring_mode": metric_code(
                execution_plan.authored_candidate_scoring_mode,
                CANDIDATE_SCORING_MODE_CODES,
                metric_name="mp/candidate_scoring_mode",
            ),
            "mp/prefix_attach_mode": metric_code(
                "repeated_forward",
                PREFIX_ATTACH_MODE_CODES,
                metric_name="mp/prefix_attach_mode",
            ),
            "mp/branch_isolation": metric_code(
                "independent_forward",
                BRANCH_ISOLATION_CODES,
                metric_name="mp/branch_isolation",
            ),
            "mp/prefix_gradient": metric_code(
                "non_detached_recomputed_per_branch",
                PREFIX_GRADIENT_CODES,
                metric_name="mp/prefix_gradient",
            ),
            "mp/prefix_encoding_cache_hits": 0.0,
            "mp/prefix_encoding_cache_misses": 0.0,
        }
        object_count = len(prefix_indices) + len(remaining_indices)
        completeness_weight = self._annotation_completeness_weight(
            object_count=object_count,
            cfg=cfg.structural_close.annotation_completeness_weight,
        )
        tail_count = max(0, int(getattr(cfg.candidates, "tail_positive_count", 1)))
        tail_indices = (
            tuple(sorted(remaining_indices)[-tail_count:])
            if tail_count and remaining_indices
            else ()
        )
        metrics["mp/annotation_completeness_weight_mean"] = float(completeness_weight)
        metrics["mp/tail_positive_samples"] = float(
            bool(tail_indices)
            and bool(set(tail_indices).intersection(candidate_indices))
        )
        metrics["mp/final_gt_object_scored_samples"] = float(
            bool(remaining_indices) and max(remaining_indices) in set(candidate_indices)
        )
        metrics["mp/final_close_weight_mean"] = 0.0
        for mode in (
            "empty_prefix",
            "random_subset",
            "leave_one_out",
            "full_prefix",
        ):
            metrics[f"mp/configured_ratio_{mode}"] = float(
                sample.configured_mixture.get(mode, 0.0)
            )
            metrics[f"mp/resolved_valid_ratio_{mode}"] = float(
                sample.resolved_valid_mixture.get(mode, 0.0)
            )
        total_loss: torch.Tensor | None = None
        objective_contributes = False
        sync_contributes = False
        last_outputs: Any = None
        aux_accum: dict[str, dict[str, Any]] = {}
        ddp_padding_policy = str(cfg.train_forward.ddp_sync.candidate_padding)
        max_candidate_forwards = self._ddp_max_int(len(candidate_indices), model)
        padding_forwards = (
            max(0, max_candidate_forwards - len(candidate_indices))
            if ddp_padding_policy == "max_count"
            else 0
        )
        metrics["mp/ddp_candidate_forward_sync_count"] = float(max_candidate_forwards)
        metrics["mp/ddp_candidate_forward_local_count"] = float(len(candidate_indices))
        metrics["mp/ddp_candidate_forward_max_count"] = float(max_candidate_forwards)
        metrics["mp/ddp_candidate_padding_policy"] = metric_code(
            ddp_padding_policy,
            DDP_CANDIDATE_PADDING_POLICY_CODES,
            metric_name="mp/ddp_candidate_padding_policy",
        )
        metrics["mp/ddp_candidate_padding_forwards"] = float(padding_forwards)
        metrics["mp/retained_graph_branch_forwards"] = float(
            (len(candidate_indices) + padding_forwards)
            if runtime_mode == "retained_graph"
            else 0
        )
        metrics["mp/checkpointed_branch_forwards"] = float(
            (len(candidate_indices) + padding_forwards)
            if runtime_mode == "checkpointed_exact"
            else 0
        )

        if candidate_indices:
            candidate_results: list[CandidateLogProbResult] = []
            candidate_lengths: list[int] = []
            scores: list[torch.Tensor] = []
            if runtime_mode == "smart_batched_exact":
                branches = [
                    self._encode_branch(
                        meta=meta,
                        prefix_indices=prefix_indices,
                        candidate_index=candidate_index,
                    )
                    for candidate_index in candidate_indices
                ]
                batch_results, batch_metrics = (
                    self._score_candidate_branches_smart_batched(
                        model=model,
                        branches=branches,
                        coord_token_ids=coord_token_ids,
                    )
                )
                metrics.update(batch_metrics)
                candidate_results.extend(batch_results)
                candidate_lengths.extend(int(result.tokens) for result in batch_results)
                scores.extend(result.score for result in batch_results)
            else:
                for candidate_index in candidate_indices:
                    branch = self._encode_branch(
                        meta=meta,
                        prefix_indices=prefix_indices,
                        candidate_index=candidate_index,
                    )
                    result, outputs, branch_inputs = self._score_candidate_branch(
                        model=model,
                        branch=branch,
                        coord_token_ids=coord_token_ids,
                    )
                    last_outputs = outputs
                    candidate_results.append(result)
                    candidate_lengths.append(int(result.tokens))
                    scores.append(result.score)
                    self._compute_candidate_aux_atoms(
                        branch=branch,
                        branch_inputs=branch_inputs,
                        outputs=outputs,
                        aux_accum=aux_accum,
                    )

            score_tensor = torch.stack(scores)
            raw_log_z = torch.logsumexp(score_tensor, dim=0)
            estimator = execution_plan.logz_estimator
            mp_result = compute_mp_pem_losses(
                scores=score_tensor,
                pem_mode=str(cfg.positive_evidence_margin.objective),
                rho=cfg.positive_evidence_margin.rho,
                log_rho=cfg.positive_evidence_margin.log_rho,
                estimator=estimator,
                remaining_count=execution_plan.remaining_candidate_count,
                scored_count=len(candidate_indices),
                candidate_lengths=score_tensor.new_tensor(candidate_lengths),
            )
            total_loss = mp_result.total_objective
            objective_contributes = True
            metrics.update(mp_result.metrics)
            metrics["mp/logZ_estimator"] = metric_code(
                estimator,
                LOGZ_ESTIMATOR_CODES,
                metric_name="mp/logZ_estimator",
            )
            metrics.update(
                self._score_metrics(
                    scores=score_tensor, candidate_results=candidate_results
                )
            )
            json_structural_loss = self._json_structural_aux_loss(
                candidate_results=candidate_results,
                weight=float(cfg.structural_close.json_structural_weight),
            )
            if json_structural_loss is not None:
                metrics["loss/json_structural"] = float(
                    json_structural_loss.detach().item()
                )
                if float(cfg.structural_close.json_structural_weight) > 0.0:
                    total_loss = total_loss + json_structural_loss
                    objective_contributes = True
            else:
                metrics["loss/json_structural"] = 0.0
            gate_loss, gate_metrics = self._bidirectional_gate_aux_loss(
                candidate_results=candidate_results,
                gate_cfg=cfg.bidirectional_token_gate,
            )
            metrics.update(gate_metrics)
            if gate_loss is not None:
                total_loss = total_loss + gate_loss
                objective_contributes = True
            metrics["mp/logZ_scored_raw"] = float(raw_log_z.detach().item())
            metrics["mp/logZ_remaining_est"] = float(
                mp_result.log_z_remaining.detach().item()
            )
            if execution_plan.objective_fidelity == "exact":
                metrics["mp/logZ_remaining_exact"] = float(
                    mp_result.log_z_remaining.detach().item()
                )
            metrics["mp/responsibility_entropy_scored"] = float(
                metrics.get("mp/responsibility_entropy", 0.0)
            )
            metrics["mp/max_responsibility_scored"] = float(
                metrics.get("mp/max_responsibility", 0.0)
            )
            metrics["mp/min_responsibility_scored"] = float(
                metrics.get("mp/min_responsibility", 0.0)
            )
            metrics["mp/valid_length_corr_samples"] = float(
                metrics.get("mp/responsibility_length_corr_valid", 0.0)
            )
            if "mp/responsibility_length_corr" in metrics:
                metrics["mp/responsibility_vs_length_corr"] = float(
                    metrics["mp/responsibility_length_corr"]
                )
            metrics["loss/candidate_balanced"] = float(
                mp_result.loss_candidate_balanced.detach().item()
            )
            metrics["loss/mp_diagnostic"] = float(mp_result.loss_mp.detach().item())
            metrics["loss/pem"] = float(mp_result.loss_pem.detach().item())
            metrics["loss/mp"] = (
                float(mp_result.loss_candidate_balanced.detach().item())
                if cfg.positive_evidence_margin.objective == "disabled"
                else 0.0
            )
            metrics["mp/loss_mp_denominator_samples"] = 1.0
            aux_loss, aux_metrics = self._aggregate_candidate_aux_atoms(aux_accum)
            metrics.update(aux_metrics)
            if aux_loss is not None:
                total_loss = total_loss + aux_loss
                objective_contributes = True
        else:
            candidate_results = []
            metrics.update(
                {
                    "loss/candidate_balanced": 0.0,
                    "loss/mp": 0.0,
                    "loss/mp_diagnostic": 0.0,
                    "loss/pem": 0.0,
                    "loss/schema_open": 0.0,
                    "loss/json_structural": 0.0,
                    "mp/loss_mp_denominator_samples": 0.0,
                    "mp/logZ_scored_raw": 0.0,
                    "mp/logZ_remaining_est": 0.0,
                    "mp/logZ_estimator": metric_code(
                        "exact",
                        LOGZ_ESTIMATOR_CODES,
                        metric_name="mp/logZ_estimator",
                    ),
                    "mp/responsibility_entropy": 0.0,
                    "mp/responsibility_entropy_scored": 0.0,
                    "mp/effective_candidate_count": 0.0,
                    "mp/effective_candidate_fraction": 0.0,
                    "mp/max_responsibility": 0.0,
                    "mp/max_responsibility_scored": 0.0,
                    "mp/min_responsibility": 0.0,
                    "mp/min_responsibility_scored": 0.0,
                    "mp/valid_length_corr_samples": 0.0,
                    "mp/schema_open_tokens_scored_mean": 0.0,
                    "mp/json_structural_tokens_scored_mean": 0.0,
                }
            )

        for _ in range(padding_forwards):
            # DDP ranks must execute the same number of branch graphs before the
            # shared backward. Padding forwards are zero-loss and metric-free.
            padding_branch = self._encode_branch(
                meta=meta,
                prefix_indices=prefix_indices,
                candidate_index=None,
            )
            padding_result, padding_outputs, _padding_inputs = (
                self._score_candidate_branch(
                    model=model,
                    branch=padding_branch,
                    coord_token_ids=coord_token_ids,
                )
            )
            if padding_outputs is not None:
                last_outputs = padding_outputs
            padding_loss = padding_result.score * 0.0
            total_loss = (
                padding_loss if total_loss is None else total_loss + padding_loss
            )
            sync_contributes = True

        (
            close_start_nll,
            close_sequence_nll,
            final_token_nll,
            p_close,
            prefix_tokens,
            close_outputs,
        ) = self._close_branch_stats(
            model=model,
            meta=meta,
            prefix_indices=prefix_indices,
        )
        last_outputs = close_outputs if last_outputs is None else last_outputs
        metrics["mp/prefix_tokens_mean"] = float(prefix_tokens)
        p_continue = (1.0 - p_close).clamp(min=0.0, max=1.0)
        if remaining_indices:
            anti_loss = -torch.log((1.0 - p_close).clamp_min(1e-7))
            weighted_anti = anti_loss * float(
                cfg.structural_close.close_start_suppression_weight
            )
            if total_loss is None:
                total_loss = weighted_anti
            else:
                total_loss = total_loss + weighted_anti
            metrics["loss/anti_close_start"] = float(weighted_anti.detach().item())
            metrics["loss/anti_stop"] = metrics["loss/anti_close_start"]
            metrics["loss/weak_schema_close"] = 0.0
            metrics["loss/eod"] = 0.0
            metrics["stop/p_close_start_when_remaining_exists"] = float(
                p_close.detach().item()
            )
            metrics["stop/p_continue_start_when_remaining_exists"] = float(
                p_continue.detach().item()
            )
            metrics["stop/p_stop_when_remaining_exists"] = metrics[
                "stop/p_close_start_when_remaining_exists"
            ]
            metrics["stop/p_continue_when_remaining_exists"] = metrics[
                "stop/p_continue_start_when_remaining_exists"
            ]
        else:
            final_close_weight = float(
                cfg.structural_close.final_schema_close_weight
            ) * float(completeness_weight)
            metrics["mp/final_close_weight_mean"] = float(final_close_weight)
            weak_loss = close_sequence_nll * final_close_weight
            if final_close_weight > 0.0:
                total_loss = weak_loss if total_loss is None else total_loss + weak_loss
                objective_contributes = True
            metrics["loss/anti_close_start"] = 0.0
            metrics["loss/anti_stop"] = 0.0
            metrics["loss/weak_schema_close"] = float(weak_loss.detach().item())
            metrics["loss/eod"] = metrics["loss/weak_schema_close"]
            metrics["stop/p_close_start_when_remaining_empty"] = float(
                p_close.detach().item()
            )
            metrics["stop/p_stop_when_remaining_empty"] = metrics[
                "stop/p_close_start_when_remaining_empty"
            ]
            metrics["stop/logp_close_sequence_when_remaining_empty"] = float(
                (-close_sequence_nll).detach().item()
            )
        metrics["stop/p_final_schema_token_teacher_forced"] = float(
            torch.exp(-final_token_nll).clamp(min=0.0, max=1.0).detach().item()
        )
        metrics["mp/total_candidate_tokens_scored"] = float(
            sum(int(result.tokens) for result in candidate_results)
            if candidate_indices
            else 0.0
        )
        metrics["mp/candidate_tokens_scored_mean"] = metrics[
            "mp/total_candidate_tokens_scored"
        ] / max(len(candidate_indices), 1)
        repeated_forward_tokens = (
            float(prefix_tokens) * float(len(candidate_indices))
            + metrics["mp/total_candidate_tokens_scored"]
        )
        single_sequence_tokens = (
            float(prefix_tokens) + metrics["mp/total_candidate_tokens_scored"]
        )
        metrics["mp/repeated_forward_token_ratio_vs_baseline"] = float(
            repeated_forward_tokens / max(single_sequence_tokens, 1.0)
        )
        metrics["mp/objective_contributing_samples"] = float(objective_contributes)
        if not objective_contributes and not sync_contributes:
            total_loss = None
        return total_loss, metrics, last_outputs

    def compute_loss(
        self,
        model: Any,
        inputs: Mapping[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ):
        del num_items_in_batch
        if getattr(self, "model", None) is None:
            self.model = model
        batch = dict(inputs)
        meta_list = batch.pop("set_continuation_meta", None)
        if not isinstance(meta_list, list):
            raise ValueError(
                "stage1_set_continuation trainer requires set_continuation_meta from its collator"
            )
        if not meta_list:
            raise ValueError("set_continuation_meta must contain at least one sample")

        coord_token_ids = torch.tensor(self._get_coord_token_ids(), dtype=torch.long)
        sample_losses: list[torch.Tensor] = []
        sample_metrics: list[dict[str, Any]] = []
        last_outputs: Any = None
        for sample_offset, meta in enumerate(meta_list):
            if not isinstance(meta, Mapping):
                raise ValueError("set_continuation_meta entries must be mappings")
            sample_loss, metrics, outputs = self._process_sample(
                model=model,
                meta=meta,
                sample_offset=sample_offset,
                coord_token_ids=coord_token_ids,
            )
            if sample_loss is not None:
                sample_losses.append(sample_loss)
            sample_metrics.append(metrics)
            if outputs is not None:
                last_outputs = outputs

        if sample_losses:
            total_loss = torch.stack(sample_losses).mean()
        else:
            logits = getattr(last_outputs, "logits", None)
            total_loss = (
                logits.sum() * 0.0
                if isinstance(logits, torch.Tensor)
                else torch.tensor(0.0, dtype=torch.float32)
            )
        metrics = mean_numeric_metrics(sample_metrics)
        emit_stage1_set_continuation_metrics(self, metrics)
        if return_outputs:
            aggregate_outputs = {
                "loss": total_loss,
                "logits": total_loss.new_empty((len(meta_list), 0, 0)),
                "set_continuation_outputs": "aggregate_loss_only",
            }
            return total_loss, aggregate_outputs
        return total_loss

    def _apply_rank_symmetric_save_delay_best_metric_guard(
        self, metrics: Mapping[str, Any]
    ) -> None:
        """Re-apply save-delay best-metric guard after rank-0 eval metrics are shared."""

        if not metrics:
            return
        try:
            from src.callbacks.save_delay_callback import SaveDelayCallback
            from transformers.trainer_utils import SaveStrategy
        except ImportError:
            return

        args = getattr(self, "args", None)
        state = getattr(self, "state", None)
        control = getattr(self, "control", None)
        if args is None or state is None or control is None:
            return
        if getattr(args, "save_strategy", None) != SaveStrategy.BEST:
            return

        callback_handler = getattr(self, "callback_handler", None)
        callbacks = getattr(callback_handler, "callbacks", ()) or ()
        for callback in callbacks:
            if isinstance(callback, SaveDelayCallback):
                cast(Any, callback).on_evaluate(args, state, control, metrics=metrics)

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: Any = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Run Stage-1 detection callbacks without HF teacher-forced prediction.

        Set-continuation batches carry branch metadata for the MP objective, and
        the ordinary HuggingFace prediction loop does not know how to forward
        those examples. Evaluation for this trainer is therefore callback-owned:
        in production that means Stage1DetectionEvalCallback performs rollout,
        parsing, and detector metrics using the live model.
        """

        del eval_dataset, ignore_keys
        start = time.perf_counter()
        metrics: dict[str, float] = {}
        callback_handler = getattr(self, "callback_handler", None)
        if callback_handler is not None:
            if hasattr(self.control, "should_evaluate"):
                self.control.should_evaluate = False
            call_event = getattr(callback_handler, "call_event", None)
            if callable(call_event):
                self.control = call_event(
                    "on_evaluate",
                    self.args,
                    self.state,
                    self.control,
                    metrics=metrics,
                )
            else:
                self.control = callback_handler.on_evaluate(
                    self.args,
                    self.state,
                    self.control,
                    metrics=metrics,
                    model=getattr(self, "model", None),
                )

        try:
            import torch.distributed as dist
        except (ImportError, TypeError, ValueError):
            dist = None  # type: ignore[assignment]
        if dist is not None and dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            payload: list[dict[str, float] | None] = [
                dict(metrics) if rank == 0 else None
            ]
            dist.broadcast_object_list(payload, src=0)
            if isinstance(payload[0], dict):
                metrics.update(payload[0])

        self._apply_rank_symmetric_save_delay_best_metric_guard(metrics)

        metrics.setdefault(f"{metric_key_prefix}/runtime", time.perf_counter() - start)
        log_fn = getattr(self, "log", None)
        if callable(log_fn):
            log_fn(metrics)
        return metrics


__all__ = ["Stage1SetContinuationTrainer"]
