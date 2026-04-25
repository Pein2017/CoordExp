from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
from swift.trainers import TrainerFactory

from src.coord_tokens.codec import get_coord_token_ids
from src.trainers.metrics.mixins import _validate_batch_contract
from src.trainers.teacher_forcing.forwards import (
    assert_unsliced_logits,
    prepare_forward_inputs,
    run_no_cache_forward,
)

from .branch_encoder import EncodedSetContinuationBranch, encode_set_continuation_branch
from .losses import (
    CandidateLogProbResult,
    compute_candidate_full_entry_logprob,
    compute_close_sequence_nll,
    compute_close_start_nll,
    compute_mp_pem_losses,
)
from .metrics import (
    BRANCH_ISOLATION_CODES,
    CANDIDATE_SCORING_MODE_CODES,
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
        assert_unsliced_logits(
            logits=logits,
            input_ids=input_ids,
            where="stage1-set-continuation training",
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
    ) -> tuple[CandidateLogProbResult, Any]:
        outputs, branch_inputs = self._forward_branch(model=model, branch=branch)
        logits = outputs.logits
        labels = branch_inputs["labels"].to(device=logits.device)
        result = compute_candidate_full_entry_logprob(
            logits=logits,
            labels=labels,
            candidate_entry_label_mask=branch.candidate_entry_label_mask.to(
                device=logits.device
            ),
            coord_label_mask=branch.coord_label_mask.to(device=logits.device),
            coord_token_ids=coord_token_ids.to(device=logits.device),
        )
        return result, outputs

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
        outputs, branch_inputs = self._forward_branch(model=model, branch=branch)
        logits = outputs.logits
        labels = branch_inputs["labels"].to(device=logits.device)
        close_start_mask = branch.structural_close_start_mask.to(device=logits.device)
        close_sequence_mask = branch.structural_close_sequence_mask.to(
            device=logits.device
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
        prefix_token_positions = close_start_mask.nonzero(as_tuple=False)
        prefix_tokens = (
            int(prefix_token_positions[:, 1].min().detach().cpu().item())
            if int(prefix_token_positions.shape[0]) > 0
            else 0
        )
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

        metrics: dict[str, Any] = {
            "mp/num_prefix_objects": float(len(prefix_indices)),
            "mp/num_remaining_objects": float(len(remaining_indices)),
            "mp/num_candidates_scored": float(len(candidate_indices)),
            "mp/scored_candidate_fraction": float(sample.scored_candidate_fraction),
            "mp/samples_with_candidates": float(bool(candidate_indices)),
            "mp/samples_full_prefix": float(len(remaining_indices) == 0),
            "mp/branch_forwards_per_sample": float(len(candidate_indices) + 1),
            "mp/candidate_scoring_mode": metric_code(
                sample.candidate_scoring_mode,
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
        }
        total_loss: torch.Tensor | None = None
        objective_contributes = False
        last_outputs: Any = None

        if candidate_indices:
            candidate_results: list[CandidateLogProbResult] = []
            candidate_lengths: list[int] = []
            scores: list[torch.Tensor] = []
            for candidate_index in candidate_indices:
                branch = self._encode_branch(
                    meta=meta,
                    prefix_indices=prefix_indices,
                    candidate_index=candidate_index,
                )
                result, outputs = self._score_candidate_branch(
                    model=model,
                    branch=branch,
                    coord_token_ids=coord_token_ids,
                )
                last_outputs = outputs
                candidate_results.append(result)
                candidate_lengths.append(int(result.tokens))
                scores.append(result.score)

            score_tensor = torch.stack(scores)
            raw_log_z = torch.logsumexp(score_tensor, dim=0)
            estimator = "exact"
            if sample.candidate_scoring_mode == "uniform_subsample":
                estimator = (
                    "uniform_importance"
                    if cfg.positive_evidence_margin.mode == "replace_mp"
                    else "sampled_raw"
                )
            mp_result = compute_mp_pem_losses(
                scores=score_tensor,
                pem_mode=str(cfg.positive_evidence_margin.mode),
                rho=cfg.positive_evidence_margin.rho,
                log_rho=cfg.positive_evidence_margin.log_rho,
                estimator=estimator,
                remaining_count=len(remaining_indices),
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
            metrics["mp/logZ_scored_raw"] = float(raw_log_z.detach().item())
            metrics["mp/logZ_remaining_est"] = float(
                mp_result.log_z_remaining.detach().item()
            )
            if sample.candidate_scoring_mode == "exact":
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
            metrics["loss/mp_diagnostic"] = float(mp_result.loss_mp.detach().item())
            metrics["loss/pem"] = float(mp_result.loss_pem.detach().item())
            metrics["loss/mp"] = (
                float(mp_result.loss_mp.detach().item())
                if cfg.positive_evidence_margin.mode == "disabled"
                else 0.0
            )
            metrics["mp/loss_mp_denominator_samples"] = 1.0
        else:
            metrics.update(
                {
                    "loss/mp": 0.0,
                    "loss/mp_diagnostic": 0.0,
                    "loss/pem": 0.0,
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
                    "mp/max_responsibility": 0.0,
                    "mp/max_responsibility_scored": 0.0,
                    "mp/min_responsibility": 0.0,
                    "mp/min_responsibility_scored": 0.0,
                    "mp/valid_length_corr_samples": 0.0,
                }
            )

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
            weighted_anti = anti_loss * float(cfg.structural_close.anti_close_weight)
            if total_loss is None:
                total_loss = weighted_anti
            else:
                total_loss = total_loss + weighted_anti
            metrics["loss/anti_close_start"] = float(weighted_anti.detach().item())
            metrics["loss/weak_schema_close"] = 0.0
            metrics["stop/p_close_start_when_remaining_exists"] = float(
                p_close.detach().item()
            )
            metrics["stop/p_continue_start_when_remaining_exists"] = float(
                p_continue.detach().item()
            )
        else:
            weak_loss = close_sequence_nll * float(
                cfg.structural_close.final_close_weight
            )
            if float(cfg.structural_close.final_close_weight) > 0.0:
                total_loss = weak_loss if total_loss is None else total_loss + weak_loss
                objective_contributes = True
            metrics["loss/anti_close_start"] = 0.0
            metrics["loss/weak_schema_close"] = float(weak_loss.detach().item())
            metrics["stop/p_close_start_when_remaining_empty"] = float(
                p_close.detach().item()
            )
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
        metrics["mp/repeated_forward_token_ratio_vs_baseline"] = float(
            (metrics["mp/total_candidate_tokens_scored"] + len(prefix_indices))
            / max(len(prefix_indices) + len(remaining_indices), 1)
        )
        metrics["mp/objective_contributing_samples"] = float(objective_contributes)
        if not objective_contributes:
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


__all__ = ["Stage1SetContinuationTrainer"]
