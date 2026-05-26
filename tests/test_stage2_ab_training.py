import json
import math
import types
from contextlib import nullcontext
from typing import List, Sequence

import pytest
import torch
import torch.nn as nn

from src.trainers.stage2_rollout_runtime import (
    GTObject,
    _serialize_append_fragment,
    parse_rollout_for_matching,
)
from src.common.detection_sequence import (
    BOX_START_TOKEN,
    OBJECT_REF_START_TOKEN,
)
from src.training.stage2.rollout_codec import (
    Stage2RolloutTemplateMismatchError,
    resolve_stage2_rollout_template_policy,
)
from src.training.stage2.assignment import GreedyIoUAssignment
from src.trainers.rollout_matching.contracts import MatchResult
from src.trainers.stage2_rollout_correction import (
    Stage2RolloutCorrectionTrainer,
    _PendingStage2Log,
    _assign_stage2_rollout_correction_objects,
    _bbox_groups_from_token_ids,
    _bbox_smoothl1_ciou_loss,
    _build_teacher_forced_payload,
    _expectation_decode_coords,
    _extract_gt_bboxonly,
    _is_rollout_correction_direct_batch_metric_key,
    _matched_prefix_structure_positions,
    _sample_identifier_or_index,
    _stage2_ab_tail_closure_positions,
    _stage2_compact_semantic_stop_branch_metadata,
    _stage2_compact_tail_closure_positions,
    _stage2_gt_consumed_members,
    _stage2_ul_geometry_from_options,
    _stage2_ul_promoted_targets,
    _stage2_ul_rollout_evidence,
    _stage2_ul_rollout_id,
    _stage2_rollout_attempt_monitor_record,
    _stage2_rollout_temperature_bucket,
    _stage2_rollout_temperature_metrics,
    _stage2_update_rollout_temperature_stats,
)
from src.trainers.rollout_correction.target_builder import (
    _apply_rollout_correction_duplicate_control,
    _attach_stage2_trie_sidecar_to_meta,
    _build_canonical_prefix_data,
    _build_canonical_prefix_text_data,
    _build_rollout_correction_meta_entry,
    _build_rollout_correction_supervision_targets,
    _build_rollout_correction_triage,
    _build_duplicate_control_divergence_diagnostics,
    _build_residual_set_correction_events,
    _rollout_correction_residual_set_enabled,
    _compute_duplicate_diagnostics,
    _sequential_dedup_bbox_objects,
)
from src.trainers.rollout_correction.teacher_forcing_adapter import (
    build_residual_set_target_ir,
)
from src.trainers.rollout_correction.objective_runner import (
    build_stage2_core_loss_logs,
    run_stage2_objective_pipelines,
)
from src.trainers.rollout_correction.rollout_views import (
    extract_compact_full_object_token_spans,
)
from src.trainers.rollout_correction.trie_supervision import (
    Stage2TrieCandidate,
    Stage2TrieObjectSpan,
    Stage2TrieTokenTarget,
    Stage2TrieTargets,
)
from src.training.teacher_forcing.constants import (
    TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
)
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab


def _apply_test_duplicate_control(
    parsed_bbox_objects_raw: Sequence[GTObject], *, duplicate_iou_threshold: float
):
    result = _apply_rollout_correction_duplicate_control(
        anchor_objects_raw=parsed_bbox_objects_raw,
        explorer_objects_raw_by_view=[],
        duplicate_iou_threshold=float(duplicate_iou_threshold),
        center_radius_scale=0.0,
        unlabeled_consistent_iou_threshold=0.0,
    )
    return (
        list(result.kept_anchor_objects),
        dict(result.suppressed_duplicate_objects_by_boundary),
    )


def test_rollout_correction_temperature_bucket_metrics_track_invalid_and_diversity() -> None:
    stats = {}
    view_a = {
        "rollout_temperature": 0.3,
        "invalid_rollout": 0,
        "parse_truncated": 0,
        "pred_objects": 3,
        "n_valid_pred": 2,
        "gen_new_tokens": 40,
        "parse": types.SimpleNamespace(
            response_text="first",
            dropped_invalid=0,
            dropped_ambiguous=0,
        ),
    }
    view_b = {
        "rollout_temperature": 0.3,
        "invalid_rollout": 1,
        "parse_truncated": 1,
        "pred_objects": 4,
        "n_valid_pred": 1,
        "gen_new_tokens": 80,
        "parse": types.SimpleNamespace(
            response_text="second",
            dropped_invalid=2,
            dropped_ambiguous=1,
        ),
    }

    _stage2_update_rollout_temperature_stats(stats, view_a)
    _stage2_update_rollout_temperature_stats(stats, view_b)
    metrics = _stage2_rollout_temperature_metrics(stats)

    prefix = "rollout/by_temperature/t0p3"
    assert _stage2_rollout_temperature_bucket(0.3) == "t0p3"
    assert metrics[f"{prefix}/raw_rollouts"] == 2.0
    assert metrics[f"{prefix}/invalid_rollout"] == 1.0
    assert metrics[f"{prefix}/invalid_rollout_rate"] == 0.5
    assert metrics[f"{prefix}/parse_dropped_invalid"] == 2.0
    assert metrics[f"{prefix}/gen_new_tokens_mean"] == 60.0
    assert metrics[f"{prefix}/gen_new_tokens_p90"] == pytest.approx(76.0)
    assert metrics[f"{prefix}/unique_sequence_count"] == 2.0
    assert metrics[f"{prefix}/unique_sequence_rate"] == 1.0
    assert _is_rollout_correction_direct_batch_metric_key(
        f"{prefix}/invalid_rollout_rate"
    )


def test_rollout_correction_temperature_monitor_record_preserves_attempt_decode_params() -> None:
    record = _stage2_rollout_attempt_monitor_record(
        label="peer",
        view={
            "rollout_index": 2,
            "rollout_role": "peer_attempt",
            "decode_mode": "sampling",
            "rollout_decode_mode": "sampling",
            "rollout_temperature": 0.7,
            "rollout_top_p": 0.9,
            "rollout_top_k": 50,
            "invalid_rollout": 1,
            "parse_truncated": 0,
            "pred_objects": 5,
            "n_valid_pred": 3,
            "gen_new_tokens": 128,
            "parse": types.SimpleNamespace(
                response_text="object row",
                dropped_invalid=4,
                dropped_ambiguous=1,
            ),
        },
    )

    assert record["label"] == "peer"
    assert record["rollout_index"] == 2
    assert record["temperature"] == 0.7
    assert record["top_p"] == 0.9
    assert record["top_k"] == 50.0
    assert record["invalid_rollout"] is True
    assert record["parse_dropped_invalid"] == 4
    assert record["response_text"] == "object row"


def test_rollout_correction_assignment_helper_supports_greedy_iou_provenance() -> None:
    match = _assign_stage2_rollout_correction_objects(
        strategy=GreedyIoUAssignment(iou_threshold=0.5),
        preds=(
            GTObject(
                index=7,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="matched pred",
            ),
            GTObject(
                index=8,
                geom_type="bbox_2d",
                points_norm1000=[800, 800, 900, 900],
                desc="false positive",
            ),
        ),
        gts=(
            GTObject(
                index=3,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="matched gt",
            ),
            GTObject(
                index=4,
                geom_type="bbox_2d",
                points_norm1000=[300, 300, 400, 400],
                desc="false negative",
            ),
        ),
    )

    assert match.strategy_id == "greedy_iou"
    assert match.iou_threshold == pytest.approx(0.5)
    assert match.matched_pairs == ((0, 0),)
    assert match.fp_pred_indices == (1,)
    assert match.fn_gt_indices == (1,)
    assert match.gating_rejections == 0
    assert match.matched_maskiou_count == 1


class _DummyOut:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.past_key_values = None


class _DummyModel(nn.Module):
    def __init__(
        self, *, vocab: int = 1200, hidden: int = 8, model_type: str = "qwen3_vl"
    ):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "has_input_ids": input_ids is not None,
                "has_inputs_embeds": inputs_embeds is not None,
                "position_ids": position_ids,
                "use_cache": use_cache,
                "past_key_values": past_key_values,
            }
        )
        assert (input_ids is None) ^ (inputs_embeds is None)
        assert use_cache is False
        assert past_key_values is None
        if position_ids is not None:
            # Qwen packing contract: 4-row position_ids ([text_position_ids; mRoPE]).
            assert position_ids.shape[0] == 4

        x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        return _DummyOut(self.lm_head(x))


class _DummySlicedModel(_DummyModel):
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        # Simulate logits_to_keep-style slicing (seq_len shrinks), which stage2-ab forbids.
        out.logits = out.logits[:, :-1, :]
        return out


class _DummyAlwaysTokenModel(nn.Module):
    def __init__(
        self,
        *,
        pred_id: int = 1100,
        vocab: int = 1200,
        model_type: str = "qwen3_vl",
    ):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.pred_id = int(pred_id)
        self.vocab = int(vocab)

    def forward(
        self,
        *,
        input_ids=None,
        position_ids=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        assert input_ids is not None
        assert use_cache is False
        assert past_key_values is None
        if position_ids is not None:
            assert position_ids.shape[0] == 4

        bsz, seqlen = input_ids.shape
        logits = torch.full(
            (bsz, seqlen, self.vocab),
            -100.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., self.pred_id] = 0.0
        return _DummyOut(logits)


class _DummyCallIndexedTokenModel(nn.Module):
    """Dummy model that returns different constant logits per forward call."""

    def __init__(
        self,
        *,
        pred_ids: list[int],
        vocab: int = 1200,
        hidden: int = 8,
        model_type: str = "qwen3_vl",
    ):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.embed = nn.Embedding(int(vocab), int(hidden))
        self.vocab = int(vocab)
        self.pred_ids = [int(x) for x in list(pred_ids)]
        if not self.pred_ids:
            raise ValueError("pred_ids must be non-empty")
        self.calls = 0

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        assert (input_ids is None) ^ (inputs_embeds is None)
        assert use_cache is False
        assert past_key_values is None
        if position_ids is not None:
            assert position_ids.shape[0] == 4

        idx = min(int(self.calls), int(len(self.pred_ids) - 1))
        pred_id = int(self.pred_ids[idx])
        self.calls += 1

        if input_ids is not None:
            bsz, seqlen = input_ids.shape
            device = input_ids.device
        else:
            bsz, seqlen = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        logits = torch.full(
            (int(bsz), int(seqlen), int(self.vocab)),
            -100.0,
            dtype=torch.float32,
            device=device,
        )
        logits[..., int(pred_id)] = 0.0
        return _DummyOut(logits)


class _DummyConstantCoord999Model(nn.Module):
    """Dummy model that makes softctx updates deterministic.

    It always assigns the highest coord-bin logit to bin 999, so the expected
    coord embedding is embedding(999) for every coord slot on iteration >=1.
    """

    def __init__(
        self, *, vocab: int = 1200, hidden: int = 8, model_type: str = "qwen3_vl"
    ):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.embed = nn.Embedding(vocab, hidden)
        # Make embeddings deterministic/unique so bitwise equality checks are meaningful.
        with torch.no_grad():
            w = torch.arange(vocab, dtype=torch.float32).unsqueeze(1).repeat(1, hidden)
            self.embed.weight.copy_(w)
        self.vocab = int(vocab)
        self.calls = []
        self.inputs_embeds_calls = []

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "has_input_ids": input_ids is not None,
                "has_inputs_embeds": inputs_embeds is not None,
                "position_ids": position_ids,
                "use_cache": use_cache,
                "past_key_values": past_key_values,
            }
        )
        assert (input_ids is None) ^ (inputs_embeds is None)
        assert use_cache is False
        assert past_key_values is None
        if position_ids is not None:
            assert position_ids.shape[0] == 4

        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        self.inputs_embeds_calls.append(x.detach().cpu().clone())

        bsz, seqlen = x.shape[:2]
        logits = torch.full(
            (bsz, seqlen, self.vocab),
            -100.0,
            dtype=torch.float32,
            device=x.device,
        )
        logits[..., 999] = 0.0
        return _DummyOut(logits)


class _DummyTokenizer:
    def __init__(self):
        # Reserve [0,999] for coord tokens.
        self._next_id = 1000
        self._tok_to_id: dict[str, int] = {}
        self._id_to_tok: dict[int, str] = {}
        # Make '{' stable.
        self._id_for("{")

    def _id_for(self, tok: str) -> int:
        if tok not in self._tok_to_id:
            idx = int(self._next_id)
            self._next_id += 1
            self._tok_to_id[tok] = idx
            self._id_to_tok[idx] = tok
        return int(self._tok_to_id[tok])

    def encode(self, text: str, add_special_tokens: bool = False):
        # Keep coord tokens as single ids when present; otherwise char-level.
        if (
            isinstance(text, str)
            and text.startswith("<|coord_")
            and text.endswith("|>")
        ):
            try:
                n = int(text[len("<|coord_") : -len("|>")])
                return [int(n)]
            except Exception:
                return [self._id_for(text)]
        return [self._id_for(ch) for ch in str(text)]

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        out = []
        for t in token_ids:
            tid = int(t)
            if 0 <= tid <= 999:
                out.append(f"<|coord_{tid}|>")
            else:
                out.append(self._id_to_tok.get(tid, "?"))
        return "".join(out)

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
        **_kwargs,
    ):
        s = str(text)
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        i = 0
        special_tokens = (
            OBJECT_REF_START_TOKEN,
            BOX_START_TOKEN,
            "<|im_end|>",
            "<|endoftext|>",
        )
        while i < len(s):
            token_id: int | None = None
            token_end = i + 1
            for special in special_tokens:
                if s.startswith(special, i):
                    token_id = self._id_for(special)
                    token_end = i + len(special)
                    break
            if token_id is None and s.startswith("<|coord_", i):
                j = s.find("|>", i)
                if j >= 0:
                    token_text = s[i : j + 2]
                    token_id = int(self.encode(token_text, add_special_tokens=False)[0])
                    token_end = j + 2
            if token_id is None:
                token_id = self._id_for(s[i])
            input_ids.append(int(token_id))
            offsets.append((int(i), int(token_end)))
            i = int(token_end)
        out = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            toks = [tokens]
            scalar = True
        else:
            toks = list(tokens)
            scalar = False

        ids: list[int] = []
        for tok in toks:
            s = str(tok)
            if s.startswith("<|coord_") and s.endswith("|>"):
                try:
                    n = int(s[len("<|coord_") : -len("|>")])
                except Exception:
                    n = -1
                ids.append(int(n))
            else:
                ids.append(self._id_for(s))
        return ids[0] if scalar else ids


class _CoordLiteralTokenizer(_DummyTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False):
        s = str(text)
        out: list[int] = []
        i = 0
        special_tokens = (
            OBJECT_REF_START_TOKEN,
            BOX_START_TOKEN,
            "<|im_end|>",
            "<|endoftext|>",
        )
        while i < len(s):
            matched_special = False
            for special in special_tokens:
                if s.startswith(special, i):
                    out.append(self._id_for(special))
                    i += len(special)
                    matched_special = True
                    break
            if matched_special:
                continue
            if s.startswith("<|coord_", i):
                j = s.find("|>", i)
                if j >= 0:
                    out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                    i = j + 2
                    continue
            out.append(self._id_for(s[i]))
            i += 1
        return out


class _CompactMarkerDescMergingTokenizer(_CoordLiteralTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False):
        marker_desc = f"{OBJECT_REF_START_TOKEN}unmatched-anchor"
        s = str(text)
        if s.startswith(marker_desc):
            return [self._id_for(marker_desc)] + [
                int(token_id)
                for token_id in super().encode(
                    s[len(marker_desc) :],
                    add_special_tokens=add_special_tokens,
                )
            ]

        return super().encode(text, add_special_tokens=add_special_tokens)


class _BareDescMergingTokenizer(_CoordLiteralTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False):
        if str(text) == "cat":
            return [self._id_for("cat")]
        return super().encode(text, add_special_tokens=add_special_tokens)


class _PieceFrameMismatchTokenizer(_DummyTokenizer):
    """Tokenizer stub where per-token decode and full decode have different lengths."""

    def __init__(self):
        super().__init__()
        self._mismatch_id = self._id_for("~")

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        ids = [int(t) for t in token_ids]
        if len(ids) == 1 and int(ids[0]) == int(self._mismatch_id):
            # Per-token decode expands this token to two chars.
            return "~~"

        out = []
        for tid in ids:
            if 0 <= tid <= 999:
                out.append(f"<|coord_{tid}|>")
            elif int(tid) == int(self._mismatch_id):
                # Full decode contracts the same token to one char.
                out.append("~")
            else:
                out.append(self._id_to_tok.get(int(tid), "?"))
        return "".join(out)


class _BoundaryMergingTokenizer(_DummyTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False):
        if (
            isinstance(text, str)
            and text.startswith("<|coord_")
            and text.endswith("|>")
        ):
            return super().encode(text, add_special_tokens=add_special_tokens)

        s = str(text)
        merged_pieces = (
            '[{"desc": "book", "bbox_2d": [',
            ', {"desc": "book", "bbox_2d": [',
            "]}",
        )
        out: list[int] = []
        i = 0
        while i < len(s):
            match = None
            for piece in merged_pieces:
                if s.startswith(piece, i) and (
                    match is None or len(piece) > len(match)
                ):
                    match = piece
            if match is not None:
                out.append(self._id_for(match))
                i += len(match)
                continue
            out.append(self._id_for(s[i]))
            i += 1
        return out


def test_legacy_stop_neutral_key_is_rejected() -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {
        "correction": {
            "stop_neutral": {"enabled": True},
        },
    }
    with pytest.raises(ValueError, match="stop_neutral"):
        _ = t._rollout_correction_cfg_get("anything", None)


def test_expectation_decode_is_mean_not_argmax():
    logits = torch.full((1, 1000), -100.0)
    logits[0, 0] = 0.0
    logits[0, 999] = 0.0
    out = _expectation_decode_coords(coord_logits=logits, temperature=1.0)
    assert float(out.item()) == pytest.approx(0.5, abs=1e-6)


def test_expectation_decode_st_has_hard_forward_and_soft_grad() -> None:
    logits = torch.full((1, 1000), -100.0)
    logits[0, 0] = 0.0
    logits[0, 999] = 0.0

    out_exp = _expectation_decode_coords(
        coord_logits=logits,
        temperature=1.0,
        mode="exp",
    )
    out_st = _expectation_decode_coords(
        coord_logits=logits,
        temperature=1.0,
        mode="st",
    )

    # Forward values differ: exp is mean-like, ST uses hard argmax forward.
    assert float(out_exp.item()) == pytest.approx(0.5, abs=1e-6)
    assert float(out_st.item()) == pytest.approx(0.0, abs=1e-6)


def test_expectation_decode_st_propagates_gradients() -> None:
    logits = torch.randn(2, 1000, requires_grad=True)
    out_st = _expectation_decode_coords(
        coord_logits=logits,
        temperature=1.0,
        mode="st",
    )

    loss = out_st.sum()
    loss.backward()

    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_bbox_losses_stable_on_noncanonical_pred():
    pred = torch.tensor([[1.2, -0.1, -0.2, 0.5]], dtype=torch.float32)
    gt = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    smoothl1, ciou = _bbox_smoothl1_ciou_loss(pred_xyxy=pred, gt_xyxy=gt)
    assert torch.isfinite(smoothl1).item()
    assert torch.isfinite(ciou).item()


def _make_stage2_pipeline_manifest(
    *,
    token_ce_enabled: bool = True,
    token_ce_weight: float = 1.0,
    desc_ce_weight: float = 1.0,
    rollout_fn_desc_weight: float | None = None,
    rollout_global_prefix_struct_ce_weight: float = 1.0,
    **_ignored_legacy_kwargs: object,
) -> dict:
    token_cfg: dict[str, object] = {
        "desc_ce_weight": float(desc_ce_weight),
        "rollout_global_prefix_struct_ce_weight": float(
            rollout_global_prefix_struct_ce_weight
        ),
    }
    if rollout_fn_desc_weight is not None:
        token_cfg["rollout_fn_desc_weight"] = float(rollout_fn_desc_weight)

    return {
        "objective": [
            {
                "name": "token_ce",
                "enabled": bool(token_ce_enabled),
                "weight": float(token_ce_weight),
                "application": {"preset": "anchor_text_only"},
                "config": token_cfg,
            },
        ],
        "diagnostics": [],
    }


def _make_residual_set_pipeline_manifest(
    *,
    enabled: bool = True,
    base_seed: int = 17,
    **config_overrides: object,
) -> dict:
    config: dict[str, object] = {
        "base_seed": int(base_seed),
    }
    config.update(config_overrides)
    return {
        "objective": [
            {
                "name": "residual_set_correction",
                "enabled": bool(enabled),
                "weight": 1.0,
                "application": {"preset": "rollout_self_prefix"},
                "config": config,
            },
        ],
        "diagnostics": [],
    }


def _make_min_trainer():
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    desc_ce_weight = 1.0
    t.stage2_rollout_correction_cfg = {
        "correction": {},
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
        "desc_ce_weight": desc_ce_weight,
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=float(desc_ce_weight),
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)
    return t


def _make_compact_rollout_correction_trainer(
    *,
    rollout_text: str,
    rollout_texts_by_call: list[str] | None = None,
    fallback_loss_weight: float = 1.0,
    pseudo_positive_enabled: bool = False,
    num_rollouts: int = 1,
) -> Stage2RolloutCorrectionTrainer:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {
        "correction": {
            "rollout_template_family": "compact_full",
            "fallback_loss_weight": float(fallback_loss_weight),
            "pseudo_positive": {"enabled": bool(pseudo_positive_enabled)},
            "triage_posterior": {"num_rollouts": int(num_rollouts)},
        }
    }
    t.rollout_matching_cfg = {"detection_sequence_format": "compact_full"}
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t.state = types.SimpleNamespace(global_step=0)

    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 64,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            content = data["messages"][-1]["content"]
            assistant_ids = (
                [int(x) for x in content]
                if isinstance(content, list)
                else [int(x) for x in tok.encode(str(content), add_special_tokens=False)]
            )
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    schema_ids = {
        tok._id_for(OBJECT_REF_START_TOKEN),
        tok._id_for(BOX_START_TOKEN),
    }
    stop_id = tok._id_for("<|im_end|>")
    t.teacher_forcing_role_vocab = RoleVocab(
        schema_token_ids=frozenset(schema_ids),
        text_token_ids=frozenset(set(range(1000, 5000)) - schema_ids - {stop_id}),
        coord_token_ids=frozenset(range(1000)),
        stop_token_id=stop_id,
    )
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 128
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._prepare_samples_for_rollout = lambda samples, rollout_backend: list(samples)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    def _decode_request(decode_override=None):
        override = dict(decode_override or {})
        return types.SimpleNamespace(
            decode_mode=str(override.get("decode_mode", "sampling")),
            temperature=float(override.get("temperature", 0.7)),
            top_p=float(override.get("top_p", 1.0)),
            top_k=int(override.get("top_k", -1)),
            repetition_penalty=1.0,
            max_new_tokens=64,
            num_beams=1,
        )

    t._resolve_rollout_decode_request = _decode_request

    rollout_calls = 0

    def _rollout_many(chunk, decode_override=None, request_index_offset=0):
        nonlocal rollout_calls
        mode = str((decode_override or {}).get("decode_mode", "sampling"))
        text = (
            str(rollout_texts_by_call[int(rollout_calls)])
            if rollout_texts_by_call is not None
            and int(rollout_calls) < len(rollout_texts_by_call)
            else str(rollout_text)
        )
        rollout_calls += 1
        ids = tok.encode(text, add_special_tokens=False)
        return [(list(ids), text, mode, []) for _ in chunk]

    t._rollout_many = _rollout_many
    return t


def _single_bbox_sample() -> dict:
    return {
        "messages": [],
        "assistant_payload": {
            "objects": [
                {"bbox_2d": [10, 20, 30, 40], "desc": "cat"},
            ],
        },
    }


def test_compact_tail_closure_and_stop_metadata_use_turn_end_token() -> None:
    tok = _CoordLiteralTokenizer()
    assistant_text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    assistant_ids = list(tok.encode(assistant_text, add_special_tokens=False))
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))

    closure_pos = _stage2_compact_tail_closure_positions(
        tokenizer=tok,
        assistant_span_ids=assistant_ids + [im_end_id],
        prefix_len=0,
    )
    stop_meta = _stage2_compact_semantic_stop_branch_metadata(
        tokenizer=tok,
        assistant_span_ids=assistant_ids + [im_end_id],
        prefix_len=0,
    )

    assert closure_pos == [len(assistant_ids)]
    assert stop_meta["stop_rel_pos"] == len(assistant_ids)
    assert stop_meta["stop_token_id"] == im_end_id
    assert stop_meta["continue_token_id"] is None


def test_compute_loss_rejects_legacy_stage2_ab_channel_marker():
    trainer = _make_min_trainer()
    model = _DummyModel()

    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
        }
    ]

    with pytest.raises(ValueError, match="retired A/B phase marker"):
        trainer.compute_loss(
            model,
            {
                "_stage2_ab_channel": "A",
                "_rollout_matching_meta": meta,
                "input_ids": input_ids,
            },
        )


def test_parse_rollout_fallback_prefix_brace_is_deterministic():
    tok = _CoordLiteralTokenizer()
    resp_ids = tok.encode("hello", add_special_tokens=False)

    p1 = parse_rollout_for_matching(tokenizer=tok, response_token_ids=list(resp_ids))
    p2 = parse_rollout_for_matching(tokenizer=tok, response_token_ids=list(resp_ids))

    assert p1.prefix_token_ids == tok.encode('{"objects": [', add_special_tokens=False)
    assert p1.prefix_token_ids == p2.prefix_token_ids
    assert p1.prefix_text == p2.prefix_text == '{"objects": ['
    assert p1.valid_objects == []
    assert p1.invalid_rollout is True


def test_rollout_correction_trie_sidecar_uses_prompt_len_as_segment_local_label_start() -> None:
    meta = {
        "rollout_context": "rollout_valid_with_fn_append",
        "rollout_fallback_loss_weight": 0.25,
    }

    _attach_stage2_trie_sidecar_to_meta(
        meta_entry=meta,
        y_train_ids=[101, 102, 103],
        prompt_len=4,
        sample_id="sample-1",
        rollout_index=2,
    )

    targets = meta["stage2_trie_targets"]
    assert isinstance(targets, Stage2TrieTargets)
    assert [target.position for target in targets.token_targets] == [4, 5, 6]
    assert [target.positive_token_ids for target in targets.token_targets] == [
        (101,),
        (102,),
        (103,),
    ]
    assert targets.summary.candidate_count == 1
    assert targets.summary.fallback_candidate_count == 0
    summary = meta["stage2_trie_candidate_summary"]
    assert summary == {
        "sample_id": "sample-1",
        "candidate_count": 1,
        "fallback_candidate_count": 0,
        "fallback_loss_weight_sum": 0.0,
        "weak_positive_fp_count": 0,
        "label_position_start": 4,
        "target_positions": 3,
        "branch_points": 0,
        "max_branching_factor": 1,
        "rollout_indices": [2],
        "rollout_index": 2,
        "source": "valid_rollout",
        "token_count": 3,
        "loss_weight": 1.0,
    }
    assert meta["stage2_trie_span_scores"] == []


def test_rollout_correction_trie_sidecar_fallback_context_uses_fallback_source_and_weight() -> None:
    meta = {
        "rollout_context": "fallback_gt_fn_append_only",
        "rollout_fallback_loss_weight": 0.25,
    }

    _attach_stage2_trie_sidecar_to_meta(
        meta_entry=meta,
        y_train_ids=[201, 202],
        prompt_len=3,
        sample_id="fallback-sample",
        rollout_index=-1,
    )

    targets = meta["stage2_trie_targets"]
    assert isinstance(targets, Stage2TrieTargets)
    assert targets.summary.fallback_candidate_count == 1
    assert targets.summary.fallback_loss_weight_sum == pytest.approx(0.25)
    assert targets.token_targets[0].source_weights == (0.25,)
    assert meta["stage2_trie_candidate_summary"]["source"] == (
        "fallback_gt_fn_append_only"
    )
    assert meta["stage2_trie_candidate_summary"]["loss_weight"] == pytest.approx(0.25)


def test_rollout_correction_trie_sidecar_attaches_explicit_grouped_candidates() -> None:
    meta = {
        "rollout_context": "rollout_valid_with_fn_append",
        "rollout_fallback_loss_weight": 0.25,
    }
    candidates = [
        Stage2TrieCandidate(
            sample_id="group-sample",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 11, 12],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="group-sample",
            rollout_index=1,
            source="valid_rollout",
            token_ids=[101, 11, 13],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="group-sample",
            rollout_index=-1,
            source="fallback_gt_fn_append_only",
            token_ids=[101, 11, 14],
            loss_weight=0.25,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="fallback_fn",
                    token_start=2,
                    token_end=3,
                    object_iou=None,
                    support_count=0,
                    loss_weight=0.25,
                )
            ],
        ),
    ]

    _attach_stage2_trie_sidecar_to_meta(
        meta_entry=meta,
        y_train_ids=[999],
        prompt_len=5,
        sample_id="group-sample",
        rollout_index=9,
        stage2_trie_candidates=candidates,
    )

    targets = meta["stage2_trie_targets"]
    assert isinstance(targets, Stage2TrieTargets)
    assert targets.summary.candidate_count == 3
    assert targets.summary.fallback_candidate_count == 1
    assert targets.summary.branch_points == 1
    assert targets.token_targets[2].positive_token_ids == (12, 13, 14)
    summary = meta["stage2_trie_candidate_summary"]
    assert summary["sample_id"] == "group-sample"
    assert summary["candidate_count"] == 3
    assert summary["fallback_candidate_count"] == 1
    assert summary["fallback_loss_weight_sum"] == pytest.approx(0.25)
    assert summary["weak_positive_fp_count"] == 0
    assert summary["target_positions"] == 3
    assert summary["branch_points"] == 1
    assert summary["max_branching_factor"] == 3
    assert summary["label_position_start"] == 5
    assert summary["rollout_indices"] == [0, 1, -1]
    assert "rollout_index" not in summary
    assert "token_count" not in summary
    assert meta["stage2_trie_span_scores"][0]["sample_id"] == "group-sample"


def test_rollout_correction_trie_sidecar_rejects_explicit_grouped_candidate_sample_mismatch() -> None:
    meta = {
        "rollout_context": "rollout_valid_with_fn_append",
        "rollout_fallback_loss_weight": 0.25,
    }
    candidates = [
        Stage2TrieCandidate(
            sample_id="candidate-sample",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101],
            loss_weight=1.0,
            object_spans=[],
        )
    ]

    with pytest.raises(ValueError, match="sample_id.*metadata"):
        _attach_stage2_trie_sidecar_to_meta(
            meta_entry=meta,
            y_train_ids=[],
            prompt_len=5,
            sample_id="metadata-sample",
            rollout_index=0,
            stage2_trie_candidates=candidates,
        )


def test_rollout_correction_trie_sidecar_rejects_empty_explicit_grouped_candidates() -> None:
    meta = {
        "rollout_context": "rollout_valid_with_fn_append",
        "rollout_fallback_loss_weight": 0.25,
    }

    with pytest.raises(ValueError, match="rollout group candidates.*non-empty"):
        _attach_stage2_trie_sidecar_to_meta(
            meta_entry=meta,
            y_train_ids=[],
            prompt_len=5,
            sample_id="group-sample",
            rollout_index=0,
            stage2_trie_candidates=[],
        )


def test_rollout_correction_trie_sidecar_rejects_explicit_group_with_invalid_prompt_len() -> None:
    meta = {
        "rollout_context": "rollout_valid_with_fn_append",
        "rollout_fallback_loss_weight": 0.25,
    }
    candidates = [
        Stage2TrieCandidate(
            sample_id="group-sample",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101],
            loss_weight=1.0,
            object_spans=[],
        )
    ]

    with pytest.raises(ValueError, match="label_position_start.*> 0"):
        _attach_stage2_trie_sidecar_to_meta(
            meta_entry=meta,
            y_train_ids=[],
            prompt_len=0,
            sample_id="group-sample",
            rollout_index=0,
            stage2_trie_candidates=candidates,
        )


def test_stage2_rollout_correction_sample_identifier_preserves_falsy_ids() -> None:
    assert _sample_identifier_or_index({"sample_id": 0, "image_id": 7}, 99) == 0
    assert _sample_identifier_or_index({"sample_id": None, "image_id": 0}, 99) == 0
    assert _sample_identifier_or_index({"base_idx": 0}, 99) == 0
    assert _sample_identifier_or_index({}, 99) == 99


@pytest.mark.parametrize(
    ("prompt_len", "y_train_ids"),
    [
        (0, [301]),
        (-1, [301]),
        (3, []),
    ],
)
def test_rollout_correction_trie_sidecar_skips_invalid_prompt_or_empty_targets(
    prompt_len: int,
    y_train_ids: list[int],
) -> None:
    meta = {
        "rollout_context": "rollout_valid_with_fn_append",
        "rollout_fallback_loss_weight": 0.25,
    }

    _attach_stage2_trie_sidecar_to_meta(
        meta_entry=meta,
        y_train_ids=y_train_ids,
        prompt_len=prompt_len,
        sample_id="skip-sample",
        rollout_index=0,
    )

    assert "stage2_trie_targets" not in meta
    assert "stage2_trie_candidate_summary" not in meta
    assert "stage2_trie_span_scores" not in meta


def _minimal_rollout_correction_meta_entry_kwargs(**overrides):
    kwargs = {
        "tokenizer": _DummyTokenizer(),
        "enc_ids_list": [100, 201],
        "prompt_len": 1,
        "prompt_ids": [100],
        "train_len_eff": 1,
        "prefix_len_eff": 0,
        "encoded_len": 2,
        "parse": types.SimpleNamespace(
            response_token_ids=[],
            dropped_invalid=0,
            dropped_ambiguous=0,
            truncated=False,
        ),
        "invalid_rollout": 0,
        "seed_base": 0,
        "decode_mode": "sampling",
        "n_drop_invalid": 0,
        "valid_pred_objects": 0,
        "matched_for_supervision_count": 0,
        "match": MatchResult(
            matched_pairs=[],
            fn_gt_indices=[],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
        "gt_objects_count": 0,
        "fn_count_for_meta": 0,
        "prefix_pos": [],
        "prefix_bins": [],
        "prefix_struct_pos": [],
        "prefix_desc_pos": [],
        "prefix_desc_weights": [],
        "prefix_bbox_groups": [],
        "fn_bbox_groups": [],
        "tail_desc_pos": [],
        "tail_desc_weights": [],
        "fn_object_weights": [],
        "anchor_decode_mode": "sampling",
        "explorer_decode_mode": "sampling",
        "valid_explorer_count": 0,
        "duplicate_clusters_total": 0,
        "duplicate_clusters_exempt": 0,
        "duplicate_clusters_suppressed": 0,
        "duplicate_objects_suppressed": 0,
        "duplicate_survivor_anchor_indices": [],
        "duplicate_exempt_anchor_indices": [],
        "duplicate_suppressed_anchor_indices": [],
        "anchor_gt_backed_indices": [],
        "anchor_support_counts": [],
        "anchor_support_rates": [],
        "shielded_anchor_indices": [],
        "dead_anchor_indices": [],
        "lvis_verified_positive_dead_anchor_indices": [],
        "lvis_verified_negative_dead_anchor_indices": [],
        "lvis_not_exhaustive_anchor_indices": [],
        "lvis_unevaluable_anchor_indices": [],
        "pseudo_positive_anchor_indices": [],
        "dead_explorer_indices_by_view": [],
        "recovered_gt_indices": [],
        "recovered_gt_support_counts": [],
        "recovered_gt_support_rates": [],
        "duplicate_control_first_divergence_diagnostics": [],
        "duplicate_control_first_divergence_boundary_count": 0,
        "duplicate_control_first_divergence_skipped_no_divergence": 0,
        "assignment_strategy": "greedy_iou",
        "assignment_iou_threshold": 0.3,
        "rollout_template_family": "coordjson",
        "rollout_parser_id": "coordjson_legacy",
        "rollout_append_policy_id": "coordjson_legacy_fn_append",
        "rollout_context": "rollout_valid_with_fn_append",
        "rollout_fallback_reason": None,
        "rollout_fallback_loss_weight": 0.0,
        "rollout_counts_as_valid_rollout": True,
        "y_train_ids": [201],
        "sample_id": "sample-1",
        "rollout_index": 0,
        "stage2_trie_candidates": None,
        "stage2_trie_object_spans": [],
        "stage2_trie_weak_fp_span_level_fallback": False,
        "stage2_tail_closure_positions_fn": lambda **_kwargs: [],
        "stage2_semantic_stop_branch_metadata_fn": lambda **_kwargs: None,
    }
    kwargs.update(overrides)
    return kwargs


def test_rollout_correction_meta_entry_non_residual_does_not_attach_legacy_stage2_trie_targets() -> None:
    meta, _drop_count = _build_rollout_correction_meta_entry(
        **_minimal_rollout_correction_meta_entry_kwargs()
    )

    assert "stage2_trie_targets" not in meta
    assert "stage2_trie_candidate_summary" not in meta
    assert "residual_set_target_ir" not in meta


def test_rollout_correction_meta_entry_residual_refuses_missing_event_sidecar() -> None:
    with pytest.raises(ValueError, match="CorrectionEvent-derived target IR"):
        _build_rollout_correction_meta_entry(
            **_minimal_rollout_correction_meta_entry_kwargs(
                residual_set_selected=True,
                residual_set_rollin_policy="random_valid_branch",
                residual_set_base_seed=17,
            )
        )


def test_rollout_correction_residual_objective_detection_honors_enabled_and_channel() -> None:
    manifest = _make_residual_set_pipeline_manifest()
    assert _rollout_correction_residual_set_enabled(
        manifest["objective"]
    )
    assert not _rollout_correction_residual_set_enabled(
        _make_residual_set_pipeline_manifest(enabled=False)["objective"]
    )

def test_rollout_correction_compact_full_rollout_template_uses_compact_parser_and_targets(
    monkeypatch,
) -> None:
    row = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    t = _make_compact_rollout_correction_trainer(rollout_text=row)
    parser_called = False

    def _fail_if_legacy_parser_called(**kwargs):
        nonlocal parser_called
        parser_called = True
        raise AssertionError("legacy CoordJSON parser must not run for compact_full")

    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        _fail_if_legacy_parser_called,
    )

    segments, metrics = t._prepare_rollout_correction_inputs(
        [_single_bbox_sample()],
        _segments_only=True,
    )

    assert parser_called is False
    assert len(segments) == 1
    _encoded, meta, _length = segments[0]
    assert meta["rollout_template_family"] == "compact_full"
    assert meta["rollout_parser_id"] == "compact_full"
    assert meta["rollout_context"] == "rollout_valid_with_fn_append"
    assert meta["rollout_counts_as_valid_rollout"] is True
    assert meta["fn_count"] == 0
    assert metrics["rollout/template_family_compact_full"] == pytest.approx(1.0)
    assert metrics["rollout/invalid_fallback_gt_fn_count"] == pytest.approx(0.0)
    assert metrics["rollout/fallback_loss_share"] == pytest.approx(0.0)


def test_ul_consensus_rollout_evidence_skips_invalid_unmatched_boxes() -> None:
    evidence = _stage2_ul_rollout_evidence(
        sample_id="sample-invalid-ul-box",
        view={
            "rollout_id": "r0",
            "rollout_counts_as_valid_rollout": True,
            "parsed_bbox_objects_raw": [
                GTObject(
                    index=0,
                    geom_type="bbox_2d",
                    points_norm1000=[100, 200, 50, 300],
                    desc="dog",
                ),
            ],
        },
        gts=[],
        assignment_iou_threshold=0.5,
    )

    assert evidence.is_valid is True
    assert evidence.unmatched_members == ()
    assert evidence.member_drop_reasons == {"invalid_bbox": 1}

    from src.trainers.rollout_correction.ul_consensus import mine_ul_consensus

    ul_result = mine_ul_consensus(
        [evidence],
        min_ul_valid_rollouts=1,
        consensus_ratio=1.0,
        geometry=_stage2_ul_geometry_from_options({}),
    )
    assert ul_result.skip_reasons == {"member_drop/invalid_bbox": 1}


def test_ul_consensus_rollout_evidence_drops_spatial_wrong_desc_gt_conflicts() -> None:
    gts = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[100, 100, 200, 220],
            desc="cat",
        ),
    ]
    rollouts = [
        _stage2_ul_rollout_evidence(
            sample_id="sample-wrong-desc-conflict",
            view={
                "rollout_id": rollout_id,
                "rollout_counts_as_valid_rollout": True,
                "parsed_bbox_objects_raw": [
                    GTObject(
                        index=0,
                        geom_type="bbox_2d",
                        points_norm1000=list(box),
                        desc="dog",
                    ),
                ],
            },
            gts=gts,
            assignment_iou_threshold=0.5,
        )
        for rollout_id, box in (
            ("r0", (100, 100, 200, 220)),
            ("r1", (101, 101, 201, 221)),
        )
    ]
    from src.trainers.rollout_correction.ul_consensus import mine_ul_consensus

    ul_result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=2,
        consensus_ratio=1.0,
        geometry=_stage2_ul_geometry_from_options({}),
    )

    assert all(evidence.unmatched_members == () for evidence in rollouts)
    assert [evidence.member_drop_reasons for evidence in rollouts] == [
        {"spatial_wrong_desc_conflict": 1},
        {"spatial_wrong_desc_conflict": 1},
    ]
    assert ul_result.promoted_clusters == ()
    assert ul_result.rejected_clusters == ()
    assert ul_result.quarantined_clusters == ()
    assert ul_result.skip_reasons == {
        "member_drop/spatial_wrong_desc_conflict": 2,
    }


def test_residual_set_ul_consensus_promotes_into_residual_atoms_and_artifacts() -> None:
    tok = _CoordLiteralTokenizer()
    rollouts = [
        _stage2_ul_rollout_evidence(
            sample_id="sample-ul",
            view={
                "rollout_id": rollout_id,
                "rollout_counts_as_valid_rollout": True,
                "parsed_bbox_objects_raw": [
                    GTObject(
                        index=0,
                        geom_type="bbox_2d",
                        points_norm1000=list(box),
                        desc="dog",
                    ),
                ],
            },
            gts=[],
            assignment_iou_threshold=0.5,
        )
        for rollout_id, box in (
            ("r0", (50, 60, 150, 180)),
            ("r1", (51, 61, 151, 181)),
        )
    ]
    geometry = _stage2_ul_geometry_from_options(
        {
            "ul_cluster_iou_threshold": 0.9,
            "ul_gray_iou_low": 0.3,
            "duplicate_burst_iou_threshold": 0.95,
            "commit_iou_threshold": 0.75,
        }
    )
    from src.trainers.rollout_correction.ul_consensus import (
        mine_ul_consensus,
        ul_cluster_artifact_rows,
    )

    ul_result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=2,
        consensus_ratio=1.0,
        geometry=geometry,
    )
    promoted_objects = _stage2_ul_promoted_targets(
        ul_result.promoted_clusters,
        lambda_ul_promoted=0.25,
    )

    clean_prefix = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    parsed = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="cat",
        ),
    ]
    result = _build_residual_set_correction_events(
        tokenizer=tok,
        response_token_ids=list(tok.encode(clean_prefix, add_special_tokens=False)),
        parsed_bbox_objects_raw=parsed,
        compact_full_object_spans=[],
        gts=parsed,
        accepted_objects_clean=parsed,
        match=MatchResult(
            matched_pairs=[(0, 0)],
            fn_gt_indices=[],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
        ul_promoted_objects=promoted_objects,
        assignment_iou_threshold=0.5,
        sample_id="sample-ul",
        rollout_index=0,
        lambda_ul_promoted=0.25,
    )

    assert len(promoted_objects) == 1
    assert result.metrics["residual_object_count"] == pytest.approx(2.0)
    assert result.metrics["ul_promoted_object_count"] == pytest.approx(1.0)
    assert result.events
    atom = result.events[0].atom_drafts[0]
    assert atom.metadata["selected_object_id"].startswith("ul:")
    assert atom.selected_action is not None
    assert atom.selected_action.metadata["support_provenance"] == (
        "ul",
        "cluster:0",
        "rollout:r0",
        "member:0",
    )
    assert atom.selected_action.metadata["loss_weight"] == pytest.approx(0.25)

    rows = ul_cluster_artifact_rows(ul_result, image_id="image-ul")
    assert rows[0]["decision"] == "promoted"
    assert rows[0]["reason"] == "consensus"
    assert rows[0]["support_rollout_ids"] == ["r0", "r1"]
    assert [member["bbox_norm1000"] for member in rows[0]["member_boxes"]] == [
        [50.0, 60.0, 150.0, 180.0],
        [51.0, 61.0, 151.0, 181.0],
    ]
    assert "pairwise_geometry" in rows[0]
    assert "consumed_overlap" in rows[0]


def test_residual_set_ul_consensus_materializes_rollout_local_promoted_boxes() -> None:
    tok = _CoordLiteralTokenizer()
    prepared_attempt_views = [
        {
            "rollout_id": "r0",
            "rollout_counts_as_valid_rollout": True,
            "parsed_bbox_objects_raw": [
                GTObject(
                    index=0,
                    geom_type="bbox_2d",
                    points_norm1000=[50, 60, 150, 180],
                    desc="dog",
                ),
            ],
        },
        {
            "rollout_id": "r1",
            "rollout_counts_as_valid_rollout": True,
            "parsed_bbox_objects_raw": [
                GTObject(
                    index=0,
                    geom_type="bbox_2d",
                    points_norm1000=[52, 62, 152, 182],
                    desc="dog",
                ),
            ],
        },
    ]
    rollouts = [
        _stage2_ul_rollout_evidence(
            sample_id="sample-ul-local",
            view=view,
            gts=[],
            assignment_iou_threshold=0.5,
        )
        for view in prepared_attempt_views
    ]
    geometry = _stage2_ul_geometry_from_options(
        {
            "ul_cluster_iou_threshold": 0.9,
            "ul_gray_iou_low": 0.3,
            "duplicate_burst_iou_threshold": 0.95,
            "commit_iou_threshold": 0.75,
        }
    )
    from src.trainers.rollout_correction.ul_consensus import mine_ul_consensus

    ul_result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=2,
        consensus_ratio=1.0,
        geometry=geometry,
    )

    clean_prefix = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    parsed = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="cat",
        ),
    ]
    match = MatchResult(
        matched_pairs=[(0, 0)],
        fn_gt_indices=[],
        fp_pred_indices=[],
        gating_rejections=0,
        matched_maskiou_sum=1.0,
        matched_maskiou_count=1,
    )

    def _coord_targets_for_rollout(rollout_id: str) -> dict[str, int]:
        promoted_objects = _stage2_ul_promoted_targets(
            ul_result.promoted_clusters,
            lambda_ul_promoted=0.25,
            rollout_id=rollout_id,
        )
        result = _build_residual_set_correction_events(
            tokenizer=tok,
            response_token_ids=list(tok.encode(clean_prefix, add_special_tokens=False)),
            parsed_bbox_objects_raw=parsed,
            compact_full_object_spans=[],
            gts=parsed,
            accepted_objects_clean=parsed,
            match=match,
            ul_promoted_objects=promoted_objects,
            assignment_iou_threshold=0.5,
            sample_id="sample-ul-local",
            rollout_index=0,
            rollout_id=rollout_id,
            lambda_ul_promoted=0.25,
        )
        assert len(promoted_objects) == 1
        assert promoted_objects[0]["object"].points_norm1000 == (
            [50, 60, 150, 180] if rollout_id == "r0" else [52, 62, 152, 182]
        )
        atoms = result.events[0].atom_drafts
        selected = {
            str(atom.metadata["slot"]): int(atom.selected_action.token_id)
            for atom in atoms
            if atom.selected_action is not None
            and str(atom.metadata["slot"]) in {"x1", "y1", "x2", "y2"}
        }
        assert atoms[0].selected_action.metadata["support_provenance"] == (
            "ul",
            "cluster:0",
            f"rollout:{rollout_id}",
            "member:0",
        )
        return selected

    assert _coord_targets_for_rollout("r0") == {
        "x1": 50,
        "y1": 60,
        "x2": 150,
        "y2": 180,
    }
    assert _coord_targets_for_rollout("r1") == {
        "x1": 52,
        "y1": 62,
        "x2": 152,
        "y2": 182,
    }


def test_residual_set_ul_consensus_materializes_rollout_local_desc_text() -> None:
    prepared_attempt_views = [
        {
            "rollout_id": "r0",
            "rollout_counts_as_valid_rollout": True,
            "parsed_bbox_objects_raw": [
                GTObject(
                    index=0,
                    geom_type="bbox_2d",
                    points_norm1000=[50, 60, 150, 180],
                    desc="Dog",
                ),
            ],
        },
        {
            "rollout_id": "r1",
            "rollout_counts_as_valid_rollout": True,
            "parsed_bbox_objects_raw": [
                GTObject(
                    index=0,
                    geom_type="bbox_2d",
                    points_norm1000=[52, 62, 152, 182],
                    desc="dog",
                ),
            ],
        },
    ]
    rollouts = [
        _stage2_ul_rollout_evidence(
            sample_id="sample-ul-local-desc",
            view=view,
            gts=[],
            assignment_iou_threshold=0.5,
        )
        for view in prepared_attempt_views
    ]
    geometry = _stage2_ul_geometry_from_options(
        {
            "ul_cluster_iou_threshold": 0.9,
            "ul_gray_iou_low": 0.3,
            "duplicate_burst_iou_threshold": 0.95,
            "commit_iou_threshold": 0.75,
        }
    )
    from src.trainers.rollout_correction.ul_consensus import mine_ul_consensus

    assert {
        member.desc_text
        for rollout in rollouts
        for member in rollout.unmatched_members
    } == {"Dog", "dog"}
    ul_result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=2,
        consensus_ratio=1.0,
        geometry=geometry,
    )

    r0_target = _stage2_ul_promoted_targets(
        ul_result.promoted_clusters,
        lambda_ul_promoted=0.25,
        rollout_id="r0",
    )
    r1_target = _stage2_ul_promoted_targets(
        ul_result.promoted_clusters,
        lambda_ul_promoted=0.25,
        rollout_id="r1",
    )

    assert r0_target[0]["object"].desc == "Dog"
    assert r1_target[0]["object"].desc == "dog"


def test_ul_consensus_gt_overlap_reaches_miner_quarantine_and_artifacts() -> None:
    gt = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[100, 100, 200, 220],
            desc="dog",
        ),
    ]
    rollouts = [
        _stage2_ul_rollout_evidence(
            sample_id="sample-ul-consumed",
            view={
                "rollout_id": rollout_id,
                "rollout_counts_as_valid_rollout": True,
                "parsed_bbox_objects_raw": [
                    GTObject(
                        index=0,
                        geom_type="bbox_2d",
                        points_norm1000=list(box),
                        desc="dog",
                    ),
                ],
            },
            gts=gt,
            assignment_iou_threshold=0.5,
        )
        for rollout_id, box in (
            ("r0", (100, 100, 200, 220)),
            ("r1", (101, 101, 201, 221)),
        )
    ]
    geometry = _stage2_ul_geometry_from_options(
        {
            "ul_cluster_iou_threshold": 0.9,
            "ul_gray_iou_low": 0.3,
            "duplicate_burst_iou_threshold": 0.95,
            "commit_iou_threshold": 0.75,
        }
    )
    from src.trainers.rollout_correction.ul_consensus import (
        mine_ul_consensus,
        ul_cluster_artifact_rows,
    )

    ul_result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=2,
        consensus_ratio=1.0,
        geometry=geometry,
        consumed_members=_stage2_gt_consumed_members(gts=gt),
    )

    assert ul_result.promoted_clusters == ()
    assert len(ul_result.quarantined_clusters) == 1
    assert ul_result.quarantined_clusters[0].reason == "consumed_target_overlap"
    rows = ul_cluster_artifact_rows(ul_result, image_id="image-consumed")
    assert rows[0]["reason"] == "consumed_target_overlap"
    assert rows[0]["consumed_overlap"]


def test_residual_events_use_compact_row_context_desc_tokens() -> None:
    tok = _BareDescMergingTokenizer()
    raw_text = (
        f"{OBJECT_REF_START_TOKEN}cab{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_41|>"
    )
    response_token_ids = list(tok.encode(raw_text, add_special_tokens=False))
    parsed = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 41],
            desc="cab",
        )
    ]
    spans = extract_compact_full_object_token_spans(
        tokenizer=tok,
        response_token_ids=response_token_ids,
        parsed_objects=parsed,
    )

    result = _build_residual_set_correction_events(
        tokenizer=tok,
        response_token_ids=response_token_ids,
        parsed_bbox_objects_raw=parsed,
        compact_full_object_spans=spans,
        gts=[
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="cat",
            )
        ],
        accepted_objects_clean=parsed,
        match=MatchResult(
            matched_pairs=[(0, 0)],
            fn_gt_indices=[],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
        assignment_iou_threshold=0.5,
        sample_id="sample-context-desc",
        rollout_index=0,
        lambda_ul_promoted=0.5,
    )

    assert result.events
    atom = result.events[0].atom_drafts[0]
    assert atom.correction_kind == "spatial_wrong_desc_conflict"
    assert atom.metadata["slot"] == "desc"
    assert atom.selected_action is not None
    assert atom.selected_action.token_id == tok._id_for("t")
    assert atom.metadata["observed_desc_token_position"] == response_token_ids.index(tok._id_for("b"))
    assert result.y_train_ids[atom.target_position] == tok._id_for("t")

    schema_token_ids = frozenset(
        {
            tok._id_for(OBJECT_REF_START_TOKEN),
            tok._id_for(BOX_START_TOKEN),
        }
    )
    coord_token_ids = frozenset(range(1000))
    role_vocab = RoleVocab(
        schema_token_ids=schema_token_ids,
        text_token_ids=frozenset(
            int(token_id)
            for token_id in result.y_train_ids
            if int(token_id) not in schema_token_ids
            and int(token_id) not in coord_token_ids
        ),
        coord_token_ids=coord_token_ids,
        stop_token_id=tok._id_for("<|im_end|>"),
    )
    input_ids = torch.tensor([result.y_train_ids], dtype=torch.long)
    target_ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=result.events,
        role_vocab=role_vocab,
        position_space="segment_local",
    )

    spatial_atoms = [
        atom
        for atom in target_ir.atoms
        if atom.provenance["correction_builder"] == "stage2_residual_dirty_prefix_scan_v1"
    ]
    assert len(spatial_atoms) == 1
    ir_atom = spatial_atoms[0]
    assert ir_atom.target_position == atom.target_position
    assert ir_atom.selected_token_id == input_ids[0, ir_atom.target_position].item()
    assert ir_atom.selected_token_id == tok._id_for("t")


def _residual_result_for_compact_text(
    *,
    tok: _CoordLiteralTokenizer,
    raw_text: str,
    gts: Sequence[GTObject],
    parsed: Sequence[GTObject],
    match: MatchResult,
    sample_id: str = "sample-dirty-prefix",
):
    response_token_ids = list(tok.encode(raw_text, add_special_tokens=False))
    try:
        spans = extract_compact_full_object_token_spans(
            tokenizer=tok,
            response_token_ids=response_token_ids,
            parsed_objects=list(parsed),
        )
    except ValueError:
        spans = []
    return _build_residual_set_correction_events(
        tokenizer=tok,
        response_token_ids=response_token_ids,
        parsed_bbox_objects_raw=list(parsed),
        compact_full_object_spans=spans,
        gts=list(gts),
        accepted_objects_clean=list(parsed),
        match=match,
        assignment_iou_threshold=0.5,
        sample_id=sample_id,
        rollout_index=0,
        lambda_ul_promoted=0.5,
    )


def test_residual_target_builder_retains_invalid_geometry_as_dirty_context() -> None:
    tok = _CoordLiteralTokenizer()
    dirty_text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_30|><|coord_20|><|coord_10|><|coord_40|>"
    )
    dirty_ids = list(tok.encode(dirty_text, add_special_tokens=False))

    result = _residual_result_for_compact_text(
        tok=tok,
        raw_text=dirty_text,
        gts=[
            GTObject(index=0, geom_type="bbox_2d", points_norm1000=[10, 20, 30, 40], desc="cat"),
        ],
        parsed=[],
        match=MatchResult(
            matched_pairs=[],
            fn_gt_indices=[0],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
    )

    assert result.y_train_ids[: len(dirty_ids)] == dirty_ids
    assert result.metrics["scanner_row_decision/invalid_geometry"] == pytest.approx(1.0)
    assert result.metrics["scanner_final_remaining_object_count"] == pytest.approx(1.0)
    assert result.events
    event = result.events[0]
    assert event.correction_kind == "invalid_geometry"
    atom = event.atom_drafts[0]
    assert atom.target_position == len(dirty_ids)
    assert atom.metadata["selected_object_id"] == "gt:0"


def test_residual_target_builder_removes_trailing_incomplete_object_span() -> None:
    tok = _CoordLiteralTokenizer()
    stable_text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    raw_text = f"{stable_text}{OBJECT_REF_START_TOKEN}dog{BOX_START_TOKEN}<|coord_50|>"
    stable_ids = list(tok.encode(stable_text, add_special_tokens=False))
    parsed = [
        GTObject(index=0, geom_type="bbox_2d", points_norm1000=[10, 20, 30, 40], desc="cat"),
    ]

    result = _residual_result_for_compact_text(
        tok=tok,
        raw_text=raw_text,
        gts=[
            GTObject(index=0, geom_type="bbox_2d", points_norm1000=[10, 20, 30, 40], desc="cat"),
            GTObject(index=1, geom_type="bbox_2d", points_norm1000=[50, 60, 70, 80], desc="dog"),
        ],
        parsed=parsed,
        match=MatchResult(
            matched_pairs=[(0, 0)],
            fn_gt_indices=[1],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
    )

    assert result.y_train_ids[: len(stable_ids)] == stable_ids
    assert result.y_train_ids[len(stable_ids)] == tok.convert_tokens_to_ids(OBJECT_REF_START_TOKEN)
    assert tok._id_for("d") in result.y_train_ids
    assert tok._id_for("g") in result.y_train_ids
    assert result.y_train_ids.count(tok.convert_tokens_to_ids(BOX_START_TOKEN)) == 2
    assert result.metrics["scanner_row_decision/trailing_incomplete"] == pytest.approx(1.0)
    assert result.events[0].correction_kind == "trailing_incomplete"
    assert result.events[0].atom_drafts[0].target_position == len(stable_ids)


def test_residual_target_builder_drops_unreliable_malformed_first_row_to_noop() -> None:
    tok = _CoordLiteralTokenizer()
    raw_text = "not a compact detection row"
    response_token_ids = list(tok.encode(raw_text, add_special_tokens=False))

    result = _build_residual_set_correction_events(
        tokenizer=tok,
        response_token_ids=response_token_ids,
        parsed_bbox_objects_raw=[],
        compact_full_object_spans=[],
        gts=[
            GTObject(index=0, geom_type="bbox_2d", points_norm1000=[10, 20, 30, 40], desc="cat"),
        ],
        accepted_objects_clean=[],
        match=MatchResult(
            matched_pairs=[],
            fn_gt_indices=[0],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
        assignment_iou_threshold=0.5,
        sample_id="sample-malformed",
        rollout_index=0,
        lambda_ul_promoted=0.5,
    )

    assert result.events == []
    assert result.y_train_ids == []
    assert result.metrics["scanner_dropped_sample"] == pytest.approx(1.0)
    assert result.metrics["scanner_no_atom_reason/unreliable_resync_boundary"] == pytest.approx(1.0)
    assert result.metrics["no_event_dropped_dirty_prefix"] == pytest.approx(1.0)


def test_residual_target_builder_committed_match_removes_object_through_scanner() -> None:
    tok = _CoordLiteralTokenizer()
    raw_text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_41|>"
    )
    parsed = [
        GTObject(index=0, geom_type="bbox_2d", points_norm1000=[10, 20, 30, 41], desc="cat"),
    ]

    result = _residual_result_for_compact_text(
        tok=tok,
        raw_text=raw_text,
        gts=[
            GTObject(index=0, geom_type="bbox_2d", points_norm1000=[10, 20, 30, 40], desc="cat"),
            GTObject(index=1, geom_type="bbox_2d", points_norm1000=[50, 60, 70, 80], desc="dog"),
        ],
        parsed=parsed,
        match=MatchResult(
            matched_pairs=[(0, 0)],
            fn_gt_indices=[1],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.95,
            matched_maskiou_count=1,
        ),
    )

    assert result.metrics["scanner_row_decision/committed"] == pytest.approx(1.0)
    assert result.metrics["scanner_final_remaining_object_count"] == pytest.approx(1.0)
    assert [event.correction_kind for event in result.events] == ["premature_stop"]
    slots = [str(atom.metadata["slot"]) for atom in result.events[0].atom_drafts]
    assert slots == [
        "object_start",
        "desc",
        "desc",
        "desc",
        "box_start",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    assert [
        atom.metadata["selected_object_id"] for atom in result.events[0].atom_drafts
    ] == ["gt:1"] * 9
    assert all(event.correction_kind != "matched_object_repair" for event in result.events)


def test_residual_target_builder_accepts_ul_promoted_objects() -> None:
    tok = _CoordLiteralTokenizer()
    clean_prefix = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    parsed = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="cat",
        ),
    ]
    result = _build_residual_set_correction_events(
        tokenizer=tok,
        response_token_ids=list(tok.encode(clean_prefix, add_special_tokens=False)),
        parsed_bbox_objects_raw=parsed,
        compact_full_object_spans=[],
        gts=[
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="cat",
            ),
        ],
        accepted_objects_clean=parsed,
        match=MatchResult(
            matched_pairs=[(0, 0)],
            fn_gt_indices=[],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
        ul_promoted_objects=[
            {
                "object": GTObject(
                    index=7,
                    geom_type="bbox_2d",
                    points_norm1000=[50, 60, 70, 80],
                    desc="dog",
                ),
                "loss_weight": 0.25,
            }
        ],
        assignment_iou_threshold=0.5,
        sample_id="sample-ignore-ul",
        rollout_index=0,
        lambda_ul_promoted=0.5,
    )

    assert result.metrics["residual_object_count"] == pytest.approx(2.0)
    assert result.metrics["ul_promoted_object_count"] == pytest.approx(1.0)
    assert result.events
    assert [
        atom.metadata["selected_object_id"] for atom in result.events[0].atom_drafts
    ] == ["ul:0"] * len(result.events[0].atom_drafts)
    assert result.events[0].atom_drafts[0].selected_action is not None
    assert result.events[0].atom_drafts[0].selected_action.metadata[
        "support_provenance"
    ] == ("ul",)
    assert result.events[0].atom_drafts[0].selected_action.metadata[
        "loss_weight"
    ] == pytest.approx(0.25)


@pytest.mark.parametrize("separator", ["\n", ""])
def test_compact_full_span_extractor_handles_adjacent_objects_with_optional_newlines(
    separator: str,
) -> None:
    tok = _CoordLiteralTokenizer()
    text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
        f"{separator}"
        f"{OBJECT_REF_START_TOKEN}dog{BOX_START_TOKEN}"
        "<|coord_50|><|coord_60|><|coord_70|><|coord_80|>"
    )
    token_ids = list(tok.encode(text, add_special_tokens=False))

    spans = extract_compact_full_object_token_spans(
        tokenizer=tok,
        response_token_ids=token_ids,
    )

    assert len(spans) == 2
    assert token_ids[spans[0].desc_start : spans[0].desc_end] == [
        tok._id_for("c"),
        tok._id_for("a"),
        tok._id_for("t"),
    ]
    assert token_ids[spans[1].desc_start : spans[1].desc_end] == [
        tok._id_for("d"),
        tok._id_for("o"),
        tok._id_for("g"),
    ]


def test_rollout_correction_producer_residual_ir_rebases_when_packed() -> None:
    meta_1, _drop_1 = _build_rollout_correction_meta_entry(
        **_minimal_rollout_correction_meta_entry_kwargs(
            enc_ids_list=[99, 10],
            prompt_ids=[99],
            prefix_len_eff=1,
            prefix_desc_pos=[0],
            prefix_desc_weights=[1.0],
            y_train_ids=[10],
            residual_set_selected=True,
            residual_set_target_ir=_make_stage2_residual_target_ir(),
            residual_set_rollin_policy="random_valid_branch",
            residual_set_base_seed=17,
        )
    )
    meta_2, _drop_2 = _build_rollout_correction_meta_entry(
        **_minimal_rollout_correction_meta_entry_kwargs(
            enc_ids_list=[88, 11],
            prompt_ids=[88],
            prefix_len_eff=1,
            prefix_desc_pos=[0],
            prefix_desc_weights=[1.0],
            y_train_ids=[11],
            residual_set_selected=True,
            residual_set_target_ir=_make_stage2_residual_target_ir(selected_token_id=11, valid_token_ids=frozenset({11})),
            residual_set_rollin_policy="random_valid_branch",
            residual_set_base_seed=17,
        )
    )

    for meta in (meta_1, meta_2):
        target_ir = meta["residual_set_target_ir"]
        assert target_ir.metadata["position_space"] == "segment_local"
        assert target_ir.atoms[0].target_position == 1

    input_ids = torch.tensor([[99, 10, 88, 11]], dtype=torch.long)
    logits = torch.full((1, 4, 50), -20.0, dtype=torch.float32)
    logits[0, 0, 10] = 20.0
    logits[0, 2, 11] = 20.0

    result = run_stage2_objective_pipelines(
        channel="rollout_correction",
        objective_specs=[
            {
                "name": "residual_set_correction",
                "config": {},
            }
        ],
        diagnostic_specs=[],
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits.clone(),
        meta=[meta_1, meta_2],
        coord_token_ids=(30,),
        temperature=1.0,
        token_type_masks={},
        rollout_subset_masks={},
        run_a_text=False,
        warn_once_cache=set(),
        role_vocab=_make_stage2_residual_role_vocab(),
    )

    assert result.pipeline_metrics_ctx[
        "stage2_rollout_correction/residual_set/atom_count"
    ] == pytest.approx(2.0)
    assert result.pipeline_metrics_ctx[
        "stage2_rollout_correction/residual_set/sequence_loss"
    ] == pytest.approx(0.0, abs=1.0e-6)


def test_rollout_correction_compact_full_sorted_fn_desc_reaches_prefix_meta(
    monkeypatch,
) -> None:
    t = _make_compact_rollout_correction_trainer(
        rollout_text="",
        fallback_loss_weight=0.25,
        num_rollouts=1,
    )
    t.stage2_rollout_correction_cfg["correction"]["insertion_order"] = "sorted"
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: pytest.fail("legacy CoordJSON parser must not run"),
    )

    segments, _metrics = t._prepare_rollout_correction_inputs(
        [_single_bbox_sample()],
        _segments_only=True,
    )

    assert len(segments) == 1
    _encoded, meta, _length = segments[0]
    assert meta["rollout_template_family"] == "compact_full"
    assert meta["rollout_context"] == "fallback_gt_fn_append_only"
    assert meta["prefix_len"] > 0
    assert meta["bbox_groups_fn"] == []
    assert meta["prefix_desc_pos"]
    assert [float(w) for w in meta["prefix_desc_weights"]] == [
        pytest.approx(0.25)
    ] * len(meta["prefix_desc_pos"])
    assert meta["tail_desc_pos"] == []


@pytest.mark.parametrize(
    ("rollout_text", "expected_invalid_count", "expected_empty_rate", "reason"),
    [
        ("not compact output", 4.0, 0.0, "malformed_compact_full"),
        ("", 0.0, 1.0, "empty_valid_object_set"),
        (
            f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
            "<|coord_30|><|coord_20|><|coord_10|><|coord_40|>",
            4.0,
            0.0,
            "malformed_compact_full",
        ),
    ],
)
def test_rollout_correction_compact_full_invalid_or_empty_rollout_falls_back_with_metrics(
    monkeypatch,
    rollout_text: str,
    expected_invalid_count: float,
    expected_empty_rate: float,
    reason: str,
) -> None:
    t = _make_compact_rollout_correction_trainer(
        rollout_text=rollout_text,
        fallback_loss_weight=0.25,
        pseudo_positive_enabled=True,
        num_rollouts=4,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: pytest.fail("legacy CoordJSON parser must not run"),
    )

    segments, metrics = t._prepare_rollout_correction_inputs(
        [_single_bbox_sample()],
        _segments_only=True,
    )

    assert len(segments) == 4
    _encoded, meta, _length = segments[0]
    assert meta["rollout_context"] == "fallback_gt_fn_append_only"
    assert meta["rollout_fallback_reason"] == reason
    assert meta["rollout_fallback_loss_weight"] == pytest.approx(0.25)
    assert meta["rollout_counts_as_valid_rollout"] is False
    assert meta["fn_count"] == 1
    assert meta["prefix_len"] == 0
    assert meta["tail_closure_pos"] == []
    assert meta["stop_rel_pos"] is None
    assert meta["bbox_groups_fn"][0]["gt_bins"] == [10, 20, 30, 40]
    assert meta["bbox_groups_fn"][0]["weight"] == pytest.approx(0.25)
    assert metrics["rollout/invalid_fallback_gt_fn_count"] == pytest.approx(
        expected_invalid_count
    )
    assert metrics["rollout/invalid_fallback_gt_fn_rate"] == pytest.approx(
        expected_invalid_count / 4.0
    )
    assert metrics["rollout/empty_valid_object_rate"] == pytest.approx(
        expected_empty_rate
    )
    assert metrics["rollout/fallback_loss_share"] == pytest.approx(1.0)


def test_rollout_correction_compact_full_template_mismatch_raises_before_legacy_parser(
    monkeypatch,
) -> None:
    t = _make_compact_rollout_correction_trainer(rollout_text='{"objects": []}')
    parser_called = False

    def _fail_if_legacy_parser_called(**kwargs):
        nonlocal parser_called
        parser_called = True
        raise AssertionError("legacy CoordJSON parser must not run for compact_full")

    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        _fail_if_legacy_parser_called,
    )

    with pytest.raises(Stage2RolloutTemplateMismatchError, match="compact_full"):
        t._prepare_rollout_correction_inputs([_single_bbox_sample()], _segments_only=True)

    assert parser_called is False


def test_rollout_correction_compact_full_invalid_explorer_rollouts_block_full_consensus(
    monkeypatch,
) -> None:
    anchor_text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_100|><|coord_100|><|coord_200|><|coord_200|>"
    )
    valid_explorer_text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_101|><|coord_101|><|coord_201|><|coord_201|>"
    )
    invalid_explorer_text = "not compact output"
    t = _make_compact_rollout_correction_trainer(
        rollout_text=anchor_text,
        rollout_texts_by_call=[
            anchor_text,
            valid_explorer_text,
            invalid_explorer_text,
            invalid_explorer_text,
        ],
        pseudo_positive_enabled=True,
        num_rollouts=4,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: pytest.fail("legacy CoordJSON parser must not run"),
    )

    segments, metrics = t._prepare_rollout_correction_inputs(
        [_single_bbox_sample()],
        _segments_only=True,
    )

    assert len(segments) == 4
    _encoded, meta, _length = segments[0]
    assert meta["valid_explorer_count"] == 1
    assert meta["anchor_support_counts"] == [1]
    assert meta["anchor_support_rates"] == pytest.approx([1.0 / 3.0])
    assert meta["shielded_anchor_indices"] == [0]
    assert meta["pseudo_positive_anchor_indices"] == []
    assert metrics["stage2/raw_rollouts"] == pytest.approx(4.0)
    assert metrics["rollout/invalid_fallback_gt_fn_count"] == pytest.approx(2.0)
    assert metrics["rollout/invalid_fallback_gt_fn_rate"] == pytest.approx(0.5)
    assert metrics["rollout/fallback_gt_fn_append_only_count"] == pytest.approx(2.0)
    assert metrics["rollout/peer/valid_pred_objects"] == pytest.approx(0.5)
    assert metrics["rollout/explorer/valid_pred_objects"] == pytest.approx(
        metrics["rollout/peer/valid_pred_objects"]
    )


def test_rollout_correction_live_non_residual_skips_legacy_stage2_trie_sidecar(
    monkeypatch,
) -> None:
    anchor_text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    explorer_text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_41|>"
    )
    invalid_explorer_text = "not compact output"
    t = _make_compact_rollout_correction_trainer(
        rollout_text=anchor_text,
        rollout_texts_by_call=[anchor_text, invalid_explorer_text, explorer_text],
        num_rollouts=3,
    )
    tok = t.template.tokenizer

    class _PromptTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            content = data["messages"][-1]["content"]
            assistant_ids = (
                [int(x) for x in content]
                if isinstance(content, list)
                else [
                    int(x)
                    for x in tok.encode(str(content), add_special_tokens=False)
                ]
            )
            input_ids = [12345] + list(assistant_ids)
            return {
                "input_ids": input_ids,
                "labels": [-100] + list(assistant_ids),
                "length": len(input_ids),
            }

    t.template = _PromptTemplate()
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: pytest.fail("legacy CoordJSON parser must not run"),
    )

    segments, _metrics = t._prepare_rollout_correction_inputs(
        [_single_bbox_sample()],
        _segments_only=True,
    )

    assert len(segments) == 3
    _encoded, meta, _length = segments[0]

    assert meta["valid_explorer_count"] == 1
    for _encoded, meta, _length in segments:
        assert "stage2_trie_targets" not in meta
        assert "stage2_trie_candidate_summary" not in meta


def test_rollout_correction_compact_full_non_residual_has_no_legacy_stage2_trie_roles(
    monkeypatch,
) -> None:
    rollout_text = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    t = _make_compact_rollout_correction_trainer(rollout_text=rollout_text)
    tok = t.template.tokenizer
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))

    class _PromptTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            content = data["messages"][-1]["content"]
            assistant_ids = (
                [int(x) for x in content]
                if isinstance(content, list)
                else [
                    int(x)
                    for x in tok.encode(str(content), add_special_tokens=False)
                ]
            )
            input_ids = [12345] + list(assistant_ids) + [im_end_id]
            return {
                "input_ids": input_ids,
                "labels": [-100] + list(assistant_ids) + [im_end_id],
                "length": len(input_ids),
            }

    t.template = _PromptTemplate()
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: pytest.fail("legacy CoordJSON parser must not run"),
    )

    segments, _metrics = t._prepare_rollout_correction_inputs(
        [_single_bbox_sample()],
        _segments_only=True,
    )

    _encoded, meta, _length = segments[0]

    assert "stage2_trie_targets" not in meta
    assert "stage2_trie_candidate_summary" not in meta
    assert meta["tail_closure_pos"] == [0]
    assert meta["stop_rel_pos"] == 0
    assert meta["stop_token_id"] == im_end_id
    assert meta["continue_token_id"] is None


def test_rollout_correction_matching_uses_greedy_assignment_threshold(monkeypatch):
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {}
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t.state = types.SimpleNamespace(global_step=0)

    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 7,
        "maskiou_resolution": 256,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    def _rollout_correction_get(key, default=None):
        if key == "assignment.iou_threshold":
            return 0.75
        return default

    t._rollout_correction_cfg_get = _rollout_correction_get

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 16
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        response_token_ids=[],
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )

    captured = {}

    class _StopAfterMatch(RuntimeError):
        pass

    def _fake_match(*, strategy, preds, gts, **kwargs):
        captured["strategy_id"] = str(getattr(strategy, "strategy_id", ""))
        captured["iou_threshold"] = float(getattr(strategy, "iou_threshold", -1.0))
        captured["n_pred"] = int(len(preds))
        captured["n_gt"] = int(len(gts))
        raise _StopAfterMatch("stop once assignment receives greedy strategy")

    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "x"}],
        },
    }
    with pytest.raises(_StopAfterMatch):
        t._prepare_rollout_correction_inputs([sample], _segments_only=True)

    assert captured["strategy_id"] == "greedy_iou"
    assert captured["iou_threshold"] == pytest.approx(0.75)
    assert captured["n_pred"] == 0
    assert captured["n_gt"] == 1


def test_rollout_correction_invalid_rollout_keeps_sample_via_empty_prefix_fallback(monkeypatch):
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: default

    tok = _CoordLiteralTokenizer()

    class _PromptTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            content = data["messages"][-1]["content"]
            assistant_ids = (
                [int(x) for x in content]
                if isinstance(content, list)
                else [
                    int(x)
                    for x in tok.encode(str(content), add_special_tokens=False)
                ]
            )
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _PromptTemplate()
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=list(tok.encode('{"objects": [', add_special_tokens=False)),
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=True,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )

    captured = {}

    class _StopAfterMatch(RuntimeError):
        pass

    def _fake_match(*, strategy, preds, gts, **kwargs):
        captured["n_pred"] = int(len(preds))
        captured["n_gt"] = int(len(gts))
        raise _StopAfterMatch("stop after invalid-rollout fallback reaches matcher")

    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "x"}],
        },
    }
    with pytest.raises(_StopAfterMatch):
        t._prepare_rollout_correction_inputs([sample], _segments_only=True)

    assert captured == {"n_pred": 0, "n_gt": 1}


def test_rollout_correction_enabled_pseudo_positive_drops_invalid_anchor_sample(
    monkeypatch,
    tmp_path,
):
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    ab_cfg = {
        "pseudo_positive.enabled": True,
        "triage_posterior.num_rollouts": 4,
        "invalid_rollout_policy": "dump_and_continue",
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: ab_cfg.get(key, default)

    tok = _CoordLiteralTokenizer()

    class _PromptTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            content = data["messages"][-1]["content"]
            assistant_ids = (
                [int(x) for x in content]
                if isinstance(content, list)
                else [
                    int(x)
                    for x in tok.encode(str(content), add_special_tokens=False)
                ]
            )
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _PromptTemplate()
    t.args = types.SimpleNamespace(output_dir=str(tmp_path))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1

    rollout_calls = 0

    def _fake_rollout_many(chunk, decode_override=None, request_index_offset=0):
        nonlocal rollout_calls
        rollout_calls += 1
        marker = 101 + rollout_calls
        return [([marker], "", "sampling", []) for _ in chunk]

    t._rollout_many = _fake_rollout_many
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    def _fake_parse(**kwargs):
        marker = int(kwargs["response_token_ids"][0])
        is_invalid = marker == 102
        return types.SimpleNamespace(
            prefix_token_ids=list(
                tok.encode('{"objects": [', add_special_tokens=False)
            ),
            prefix_text='{"objects": [',
            response_token_ids=list(kwargs["response_token_ids"]),
            response_text='{"objects": [{"desc": "broken", "bbox_2d": [1, 2',
            valid_objects=(
                []
                if is_invalid
                else [
                    types.SimpleNamespace(
                        index=0,
                        geom_type="bbox_2d",
                        coord_token_indices=[0, 1, 2, 3],
                        desc="x",
                    )
                ]
            ),
            dropped_invalid_by_reason={},
            dropped_invalid=0,
            dropped_ambiguous=0,
            truncated=False,
            invalid_rollout=is_invalid,
        )

    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        _fake_parse,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.points_from_coord_tokens",
        lambda **kwargs: [0, 0, 1, 1],
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "x"}],
        },
    }
    segments, batch_metrics = t._prepare_rollout_correction_inputs([sample], _segments_only=True)
    assert len(segments) == 3
    assert batch_metrics["stage2/invalid_rollout"] == pytest.approx(1.0)
    assert batch_metrics[
        "stage2_rollout_correction/invalid_rollout_sample_dropped"
    ] == pytest.approx(1.0)
    dump_dir = tmp_path / "monitor_dumps" / "prepare_failures"
    dump_paths = sorted(dump_dir.glob("*.json"))
    assert len(dump_paths) == 1
    dump_payload = json.loads(dump_paths[0].read_text())
    invalid_rollout = dump_payload["invalid_rollouts"][0]
    assert (
        invalid_rollout["response_text"]
        == '{"objects": [{"desc": "broken", "bbox_2d": [1, 2'
    )
    assert invalid_rollout["prefix_text"] == '{"objects": ['
    assert invalid_rollout["response_text_char_len"] == len(
        invalid_rollout["response_text"]
    )
    assert invalid_rollout["response_token_count"] == len(
        invalid_rollout["response_token_ids"]
    )
    assert invalid_rollout["prefix_token_count"] == len(
        invalid_rollout["prefix_token_ids"]
    )
    assert invalid_rollout["response_text_head"] == invalid_rollout["response_text"]
    assert invalid_rollout["response_text_tail"] == invalid_rollout["response_text"]


def test_rollout_correction_closure_resolution_failure_falls_back_without_dropping_sample(
    monkeypatch,
):
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: default

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction._stage2_ab_tail_closure_positions",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("synthetic closure ambiguity")
        ),
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [9, 9, 10, 10], "desc": "fn"}],
        },
    }

    segments, batch_metrics = t._prepare_rollout_correction_inputs([sample], _segments_only=True)

    assert len(segments) == 2
    meta = segments[0][1]
    assert meta["tail_closure_pos"] == []
    assert batch_metrics[
        "stage2_rollout_correction/closure_supervision/N_drop"
    ] == pytest.approx(2.0)
    assert batch_metrics["stage2_rollout_correction/invalid_rollout"] == pytest.approx(0.0)
    assert batch_metrics["rollout/template_family_coordjson"] == pytest.approx(1.0)
    assert batch_metrics["rollout/template_family_compact_full"] == pytest.approx(0.0)
    assert batch_metrics["rollout/parser_coordjson_legacy"] == pytest.approx(1.0)
    assert batch_metrics["rollout/decode_policy_legacy_coordjson"] == pytest.approx(1.0)
    assert batch_metrics["rollout/parser_template_mismatch_rate"] == pytest.approx(0.0)
    assert batch_metrics["rollout/fallback_loss_weight"] == pytest.approx(1.0)


def test_rollout_correction_duplicate_iou_threshold_zero_propagates_to_dedup(monkeypatch):
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = (
        lambda key, default=None: 0.0
        if key == "duplicate_control.iou_threshold"
        else default
    )

    tok = _DummyTokenizer()
    t.template = types.SimpleNamespace(tokenizer=tok)
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )

    captured = {}

    class _StopAfterDedup(RuntimeError):
        pass

    def _fake_duplicate_control(*, duplicate_iou_threshold, **kwargs):
        captured["threshold"] = float(duplicate_iou_threshold)
        raise _StopAfterDedup("stop after dedup threshold capture")

    monkeypatch.setattr(
        "src.trainers.rollout_correction.target_builder._apply_rollout_correction_duplicate_control",
        _fake_duplicate_control,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "x"}],
        },
    }
    with pytest.raises(_StopAfterDedup):
        t._prepare_rollout_correction_inputs([sample], _segments_only=True)

    assert captured["threshold"] == pytest.approx(0.0)


def test_rollout_correction_suspicious_monitor_dump_buffers_full_eval_style_payload(
    monkeypatch,
):
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "train_monitor_dump": {
            "enabled": True,
            "max_samples": 1,
            "max_text_chars": 32,
            "write_markdown": True,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: default

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t.state = types.SimpleNamespace(global_step=7, epoch=1.5)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    full_rollout_text = "rollout:" + (" very-duplicated" * 128)
    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=list(range(12)),
        response_text=full_rollout_text,
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="person",
            ),
            types.SimpleNamespace(
                index=1,
                geom_type="bbox_2d",
                coord_token_indices=[4, 5, 6, 7],
                desc="person",
            ),
            types.SimpleNamespace(
                index=2,
                geom_type="bbox_2d",
                coord_token_indices=[8, 9, 10, 11],
                desc="book",
            ),
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )

    coord_lookup = {
        (0, 1, 2, 3): [10, 10, 20, 20],
        (4, 5, 6, 7): [10, 10, 20, 20],
        (8, 9, 10, 11): [40, 40, 60, 60],
    }
    monkeypatch.setattr(
        "src.trainers.rollout_correction.points_from_coord_tokens",
        lambda **kwargs: list(coord_lookup[tuple(kwargs["coord_token_indices"])]),
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[(0, 0)],
            fn_gt_indices=[],
            fp_pred_indices=[1],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
    )

    sample = {
        "sample_id": "sample-314",
        "image_id": 314,
        "images": ["scene_314.jpg"],
        "width": 1000,
        "height": 1000,
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [10, 10, 20, 20], "desc": "person"}],
        },
    }

    segments, batch_metrics = t._prepare_rollout_correction_inputs([sample], _segments_only=True)
    assert len(segments) == 2
    assert batch_metrics[
        "stage2_rollout_correction/correction/dup/N_objects_suppressed"
    ] == pytest.approx(0.0)
    assert batch_metrics["stage2_rollout_correction/correction/dup/N_clusters_exempt"] == pytest.approx(
        2.0
    )
    assert len(t._stage2_train_monitor_candidates) == 2

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )
    t._stage2_flush_train_monitor_dump(global_step=8)

    dumped = captured["payload"]["samples"][0]
    assert captured["global_step"] == 8
    assert captured["payload"]["kind"] == "train_monitor_dump"
    assert dumped["image_id"] == 314
    assert dumped["gt"] == [{"desc": "person", "bbox_2d": [10, 10, 20, 20]}]
    assert dumped["pred"] == [
        {"desc": "person", "bbox_2d": [10, 10, 20, 20]},
        {"desc": "person", "bbox_2d": [10, 10, 20, 20]},
        {"desc": "book", "bbox_2d": [40, 40, 60, 60]},
    ]
    assert dumped["duplication"]["objects_suppressed"] == 0
    assert dumped["duplication"]["clusters_exempt"] == 1
    assert dumped["duplication"]["clean_accepted"] == 3
    assert dumped["match"]["fp_pred_indices"] == [1]
    assert dumped["rollout_text"] == full_rollout_text
    assert "...<truncated>" not in dumped["rollout_text"]


def test_stage2_train_monitor_dump_prefers_most_duplicate_candidate():
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "decode_mode": "greedy",
        "train_monitor_dump": {
            "enabled": True,
            "max_samples": 1,
            "write_markdown": False,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_backend = lambda: "hf"
    t.state = types.SimpleNamespace(global_step=11, epoch=0.0)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )

    t._stage2_reset_train_monitor_dump(global_step=11)
    t._stage2_note_train_monitor_candidate(
        global_step=11,
        sample={
            "sample_id": "low",
            "duplication": {"duplicates": 1, "duplicate_bursts": 1},
            "stats": {"fp_count": 0, "raw_valid_pred_objects": 2},
        },
    )
    t._stage2_note_train_monitor_candidate(
        global_step=11,
        sample={
            "sample_id": "high",
            "duplication": {"duplicates": 3, "duplicate_bursts": 2},
            "stats": {"fp_count": 1, "raw_valid_pred_objects": 5},
        },
    )

    t._stage2_flush_train_monitor_dump(global_step=11)

    assert captured["global_step"] == 11
    assert captured["payload"]["samples"][0]["sample_id"] == "high"


def test_stage2_train_monitor_dump_uses_logged_step_not_preincrement_step() -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "decode_mode": "greedy",
        "train_monitor_dump": {
            "enabled": True,
            "every_steps": 40,
            "max_samples": 1,
            "write_markdown": False,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_backend = lambda: "hf"
    t.args = types.SimpleNamespace(logging_steps=10, logging_first_step=True)
    t.state = types.SimpleNamespace(global_step=39, epoch=0.0)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []
    t._stage2_train_monitor_dump_count = 0
    t._stage2_train_monitor_dump_last_step = None

    assert t._stage2_train_monitor_step_allowed(global_step=40) is True
    assert t._stage2_train_monitor_step_allowed(global_step=39) is False

    t._stage2_reset_train_monitor_dump(global_step=40)
    t._stage2_note_train_monitor_candidate(
        global_step=40,
        sample={
            "sample_id": "dup-heavy",
            "duplication": {"duplicates": 2, "duplicate_bursts": 1},
            "stats": {"fp_count": 1, "raw_valid_pred_objects": 3},
        },
    )

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )
    t._stage2_flush_train_monitor_dump(global_step=40)

    assert captured["global_step"] == 40
    assert captured["payload"]["samples"][0]["sample_id"] == "dup-heavy"


def test_stage2_train_monitor_dump_every_rollout_steps_ignores_global_step_aliasing() -> (
    None
):
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "decode_mode": "greedy",
        "train_monitor_dump": {
            "enabled": True,
            "every_steps": 40,
            "every_rollout_steps": 3,
            "max_samples": 1,
            "write_markdown": False,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_backend = lambda: "hf"
    t.args = types.SimpleNamespace(logging_steps=10, logging_first_step=False)
    t.state = types.SimpleNamespace(global_step=11, epoch=0.0)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []
    t._stage2_train_monitor_b_step_count = 0
    t._stage2_train_monitor_dump_count = 0
    t._stage2_train_monitor_dump_last_step = None

    t._stage2_reset_train_monitor_dump(global_step=4)
    assert t._stage2_train_monitor_b_step_count == 1
    assert t._stage2_train_monitor_step_allowed(global_step=4) is False

    t._stage2_reset_train_monitor_dump(global_step=8)
    assert t._stage2_train_monitor_b_step_count == 2
    assert t._stage2_train_monitor_step_allowed(global_step=8) is False

    t._stage2_reset_train_monitor_dump(global_step=12)
    assert t._stage2_train_monitor_b_step_count == 3
    assert t._stage2_train_monitor_step_allowed(global_step=12) is True

    t._stage2_note_train_monitor_candidate(
        global_step=12,
        sample={
            "sample_id": "third-b-step",
            "duplication": {"duplicates": 2, "duplicate_bursts": 1},
            "stats": {"fp_count": 1, "raw_valid_pred_objects": 3},
        },
    )

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )
    t._stage2_flush_train_monitor_dump(global_step=12)

    assert captured["global_step"] == 12
    assert captured["payload"]["samples"][0]["sample_id"] == "third-b-step"


def test_stage2_train_monitor_cadence_advances_without_suspicious_candidate() -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "decode_mode": "greedy",
        "train_monitor_dump": {
            "enabled": True,
            "every_rollout_steps": 3,
            "max_samples": 1,
            "write_markdown": False,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t.args = types.SimpleNamespace(logging_steps=10, logging_first_step=False)
    t.is_world_process_zero = True
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []
    t._stage2_train_monitor_b_step_count = 0
    t._stage2_train_monitor_dump_count = 0
    t._stage2_train_monitor_dump_last_step = None

    t._stage2_advance_train_monitor_b_step(global_step=4)
    assert t._stage2_train_monitor_b_step_count == 1
    assert t._stage2_train_monitor_step_allowed(global_step=4) is False

    t._stage2_advance_train_monitor_b_step(global_step=8)
    assert t._stage2_train_monitor_b_step_count == 2
    assert t._stage2_train_monitor_step_allowed(global_step=8) is False

    t._stage2_advance_train_monitor_b_step(global_step=12)
    assert t._stage2_train_monitor_b_step_count == 3
    assert t._stage2_train_monitor_step_allowed(global_step=12) is True


def test_stage2_train_monitor_dump_keeps_eval_budget_and_same_step_eligibility():
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "decode_mode": "greedy",
        "train_monitor_dump": {
            "enabled": True,
            "max_events": 1,
            "max_samples": 1,
            "write_markdown": False,
        },
        "eval_monitor_dump": {"enabled": True, "every_evals": 1},
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_backend = lambda: "hf"
    t.args = types.SimpleNamespace(logging_steps=1, logging_first_step=True)
    t.state = types.SimpleNamespace(global_step=11, epoch=0.0)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []
    t._stage2_train_monitor_dump_count = 0
    t._stage2_train_monitor_dump_last_step = None

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )

    t._stage2_reset_train_monitor_dump(global_step=11)
    t._stage2_note_train_monitor_candidate(
        global_step=11,
        sample={
            "sample_id": "dup-heavy",
            "duplication": {"duplicates": 4, "duplicate_bursts": 2},
            "stats": {"fp_count": 1, "raw_valid_pred_objects": 5},
        },
    )
    t._stage2_flush_train_monitor_dump(global_step=11)

    assert captured["global_step"] == 11
    assert t._stage2_train_monitor_dump_count == 1
    assert t._stage2_train_monitor_dump_last_step == 11
    assert t._monitor_dump_count == 0
    assert t._monitor_dump_last_step is None
    assert t._should_eval_monitor_dump(global_step=11, eval_index=1) is True


def test_stage2_trie_span_score_dump_writes_jsonl(tmp_path) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    dump_root = tmp_path / "monitor_dumps"
    cfg = {
        "train_monitor_dump": {
            "enabled": True,
            "out_dir": str(dump_root),
            "async_write": False,
            "every_rollout_steps": 1,
            "max_events": 2,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t.args = types.SimpleNamespace(logging_steps=10, logging_first_step=False)
    t.state = types.SimpleNamespace(global_step=6, epoch=0.0)
    t.is_world_process_zero = True
    t._stage2_train_monitor_b_step_count = 1
    t._stage2_trie_span_score_dump_last_step = None
    t._stage2_trie_span_score_dump_count = 0
    meta = [
        {
            "stage2_trie_span_scores": [
                {
                    "sample_id": "sample-1",
                    "rollout_index": 2,
                    "candidate_source": "valid_rollout",
                    "span_role": "weak_positive_fp",
                    "object_role": "weak_positive_fp",
                    "token_start": 5,
                    "token_end": 6,
                    "mean_token_logprob": None,
                    "min_token_logprob": None,
                    "object_iou": None,
                    "support_count": 1,
                    "loss_weight": 0.05,
                }
            ]
        }
    ]

    records = t._collect_stage2_trie_span_score_records(
        meta=meta,
        global_step=7,
    )
    wrote = t._write_stage2_trie_span_score_dump(
        global_step=7,
        records=records,
    )

    assert wrote is True
    assert t._stage2_trie_span_score_dump_count == 1
    dump_path = dump_root / "stage2_trie_span_scores" / "step_000007.jsonl"
    payloads = [
        json.loads(line)
        for line in dump_path.read_text(encoding="utf-8").splitlines()
    ]
    assert payloads == [
        {
            "sample_id": "sample-1",
            "rollout_index": 2,
            "candidate_source": "valid_rollout",
            "span_role": "weak_positive_fp",
            "object_role": "weak_positive_fp",
            "token_start": 5,
            "token_end": 6,
            "mean_token_logprob": None,
            "min_token_logprob": None,
            "object_iou": None,
            "support_count": 1,
            "loss_weight": 0.05,
            "global_step": 7,
            "segment_index": 0,
            "record_index": 0,
        }
    ]


def test_stage2_trie_span_score_dump_appends_same_step_without_double_count(
    tmp_path,
) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    dump_root = tmp_path / "monitor_dumps"
    cfg = {
        "train_monitor_dump": {
            "enabled": True,
            "out_dir": str(dump_root),
            "async_write": False,
            "every_rollout_steps": 1,
            "max_events": 1,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t.args = types.SimpleNamespace(logging_steps=10, logging_first_step=False)
    t.is_world_process_zero = True
    t._stage2_train_monitor_b_step_count = 1
    t._stage2_trie_span_score_dump_last_step = None
    t._stage2_trie_span_score_dump_count = 0

    first = t._write_stage2_trie_span_score_dump(
        global_step=7,
        records=[{"sample_id": "sample-1"}],
    )
    second = t._write_stage2_trie_span_score_dump(
        global_step=7,
        records=[{"sample_id": "sample-2"}],
    )

    assert first is True
    assert second is True
    assert t._stage2_trie_span_score_dump_count == 1
    dump_path = dump_root / "stage2_trie_span_scores" / "step_000007.jsonl"
    payloads = [
        json.loads(line)
        for line in dump_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [payload["sample_id"] for payload in payloads] == [
        "sample-1",
        "sample-2",
    ]


def test_stage2_trie_span_score_dump_respects_disabled_train_monitor_dump(
    tmp_path,
) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    dump_root = tmp_path / "monitor_dumps"
    cfg = {
        "train_monitor_dump": {
            "enabled": False,
            "out_dir": str(dump_root),
            "async_write": False,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t.args = types.SimpleNamespace(logging_steps=1, logging_first_step=True)
    t.is_world_process_zero = True
    t._stage2_train_monitor_b_step_count = 1
    t._stage2_trie_span_score_dump_last_step = None
    t._stage2_trie_span_score_dump_count = 0

    wrote = t._write_stage2_trie_span_score_dump(
        global_step=1,
        records=[{"sample_id": "sample-1"}],
    )

    assert wrote is False
    assert not dump_root.exists()


def test_rollout_correction_fn_bbox_groups_anchor_to_clean_prefix_not_raw_prefix(monkeypatch):
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: default

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[2000 + i for i in range(64)],
        prefix_text="unused-raw-prefix",
        response_token_ids=[10, 10, 20, 20, 10, 10, 20, 20],
        response_text="",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="cat",
            ),
            types.SimpleNamespace(
                index=1,
                geom_type="bbox_2d",
                coord_token_indices=[4, 5, 6, 7],
                desc="cat",
            ),
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[0],
            fp_pred_indices=[0],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [9, 9, 10, 10], "desc": "fn"}],
        },
    }

    segments, _batch_metrics = t._prepare_rollout_correction_inputs([sample], _segments_only=True)
    meta = segments[0][1]

    clean_prefix = _build_canonical_prefix_data(
        tokenizer=tok,
        objects=[
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 10, 20, 20],
                desc="cat",
            )
        ],
        object_field_order="desc_first",
    )
    fn_obj = GTObject(
        index=0,
        geom_type="bbox_2d",
        points_norm1000=[9, 9, 10, 10],
        desc="fn",
    )
    append_text = _serialize_append_fragment(
        fn_objects=[fn_obj],
        prefix_text=clean_prefix.prefix_text,
    )
    rel_groups = _bbox_groups_from_token_ids(
        token_ids=list(tok.encode(append_text)),
        coord_id_set=set(range(1000)),
        gt_objs=[fn_obj],
    )
    expected_pos = [
        int(meta["prompt_len"] + meta["prefix_len"] + p) for p in rel_groups[0]
    ]

    assert int(meta["prefix_len"]) > 0
    assert int(meta["prefix_len"]) != int(len(fake_parse.prefix_token_ids))
    assert meta["bbox_groups_fn"][0]["pos"] == expected_pos


def test_rollout_correction_dual_rollout_triage_emits_recovered_ground_truth_weight_multipliers(
    monkeypatch,
) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: {
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.rollout_temperatures": [0.0, 0.7],
        "triage_posterior.explorer_top_p": 0.9,
        "triage_posterior.explorer_top_k": 12,
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
        "triage_posterior.recovered_ground_truth_weight_multiplier": 2.5,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls: list[dict[str, object]] = []

    def _fake_rollout_many(chunk, decode_override=None):
        rollout_calls.append(dict(decode_override or {}))
        temp = float((decode_override or {}).get("temperature", 0.0) or 0.0)
        marker = 101 if temp <= 0.0 else 202
        mode = "greedy" if temp <= 0.0 else "sampling"
        return [([marker], "", mode, []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    anchor_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="anchor",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="bad",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    explorer_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[202],
        response_text="explorer",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[4, 5, 6, 7],
                desc="explore",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )

    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: (
            anchor_parse
            if int(kwargs["response_token_ids"][0]) == 101
            else explorer_parse
        ),
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )

    def _fake_match(*, preds, **kwargs):
        descs = [str(obj.desc) for obj in preds]
        if descs == ["explore"]:
            return types.SimpleNamespace(
                matched_pairs=[(0, 0)],
                fn_gt_indices=[],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=1.0,
                matched_maskiou_count=1,
            )
        return types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[0],
            fp_pred_indices=list(range(len(preds))),
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )

    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [10, 10, 20, 20], "desc": "gt"}],
        },
    }

    segments, batch_metrics = t._prepare_rollout_correction_inputs([sample], _segments_only=True)

    assert len(rollout_calls) == 2
    assert rollout_calls[0]["temperature"] == pytest.approx(0.0)
    assert rollout_calls[1]["temperature"] == pytest.approx(0.7)
    assert rollout_calls[1]["top_p"] == pytest.approx(0.9)
    assert rollout_calls[1]["top_k"] == 12

    meta = segments[0][1]
    assert meta["anchor_gt_backed_indices"] == []
    assert meta["shielded_anchor_indices"] == []
    assert meta["dead_anchor_indices"] == [0]
    assert meta["recovered_gt_indices"] == [0]
    assert meta["fn_object_weights"] == [pytest.approx(2.5)]
    assert meta["bbox_groups_fn"][0]["weight"] == pytest.approx(2.5)
    assert meta["tail_desc_weights"]
    assert set(float(x) for x in meta["tail_desc_weights"]) == {2.5}
    assert meta["duplicate_control_first_divergence_diagnostics"] == []

    assert batch_metrics["train/triage/dead_anchor_count"] == pytest.approx(1.0)
    assert batch_metrics["train/triage/recovered_ground_truth_count"] == pytest.approx(
        1.0
    )
    assert batch_metrics[
        "train/triage/recovered_ground_truth_rate_den"
    ] == pytest.approx(1.0)
    assert batch_metrics["train/triage/recovered_ground_truth_rate"] == pytest.approx(
        1.0
    )
    assert batch_metrics["train/triage/dead_anchor_rate"] == pytest.approx(0.5)
    assert batch_metrics["train/triage/explorer_only_dead_rate"] == pytest.approx(0.5)
    assert batch_metrics["rollout/current/pred_objects"] == pytest.approx(2.0)
    assert batch_metrics["rollout/current/valid_pred_objects"] == pytest.approx(2.0)
    assert batch_metrics["rollout/current/gen_new_tokens_mean"] == pytest.approx(1.0)
    assert batch_metrics["rollout/current/gen_new_tokens_p90"] == pytest.approx(1.0)
    assert batch_metrics["rollout/peer/pred_objects"] == pytest.approx(1.0)
    assert batch_metrics["rollout/peer/valid_pred_objects"] == pytest.approx(1.0)
    assert batch_metrics["rollout/peer/gen_new_tokens_mean"] == pytest.approx(1.0)
    assert batch_metrics["rollout/peer/gen_new_tokens_p90"] == pytest.approx(1.0)
    assert batch_metrics["rollout/peer/temperature"] == pytest.approx(0.7)
    assert batch_metrics["rollout/peer/do_sample"] == pytest.approx(1.0)
    assert batch_metrics["rollout/peer/top_p"] == pytest.approx(0.9)
    assert batch_metrics["rollout/peer/top_k"] == pytest.approx(12.0)
    assert batch_metrics["rollout/anchor/pred_objects"] == pytest.approx(
        batch_metrics["rollout/current/pred_objects"]
    )
    assert batch_metrics["rollout/explorer/pred_objects"] == pytest.approx(
        batch_metrics["rollout/peer/pred_objects"]
    )
    assert batch_metrics[
        "rollout/matched_for_supervision_over_valid_pred"
    ] == pytest.approx(0.5)


def test_rollout_correction_dual_rollout_chunking_is_policy_symmetric(monkeypatch) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: default

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 2
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls: list[tuple[int, float]] = []

    def _fake_rollout_many(chunk, decode_override=None):
        rollout_calls.append(
            (
                int(len(chunk)),
                float((decode_override or {}).get("temperature", 0.0) or 0.0),
            )
        )
        marker = (
            101
            if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0
            else 202
        )
        return [([marker], "", "sampling", []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    with monkeypatch.context() as mp:
        mp.setattr(
            "src.trainers.rollout_correction.parse_rollout_for_matching",
            lambda **kwargs: fake_parse,
        )
        mp.setattr(
            "src.trainers.rollout_correction._extract_gt_bboxonly",
            lambda _sample: [
                GTObject(
                    index=0,
                    geom_type="bbox_2d",
                    points_norm1000=[0, 0, 1, 1],
                    desc="gt",
                )
            ],
        )
        mp.setattr(
            "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
            lambda **kwargs: types.SimpleNamespace(
                matched_pairs=[],
                fn_gt_indices=[],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=0.0,
                matched_maskiou_count=0,
            ),
        )
        mp.setattr(
            "src.trainers.rollout_correction._bbox_groups_from_token_ids",
            lambda **kwargs: [[0, 1, 2, 3] for _ in kwargs["gt_objs"]],
        )

        samples = [
            {
                "messages": [],
                "assistant_payload": {
                    "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "gt"}]
                },
            }
            for _ in range(3)
        ]
        _segments, _batch_metrics = t._prepare_rollout_correction_inputs(
            samples,
            _segments_only=True,
        )

    assert rollout_calls == [
        (2, 0.7),
        (1, 0.7),
        (2, 0.7),
        (1, 0.7),
    ]


def test_rollout_correction_enabled_pseudo_positive_uses_k4_rollouts_and_keeps_zero_object_explorer(
    monkeypatch,
) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    ab_cfg = {
        "pseudo_positive.enabled": True,
        "triage_posterior.num_rollouts": 4,
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.explorer_top_p": 0.95,
        "triage_posterior.explorer_top_k": -1,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: ab_cfg.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls: list[dict[str, float]] = []

    def _fake_rollout_many(chunk, decode_override=None, request_index_offset=0):
        rollout_calls.append(
            {
                "temperature": float(
                    (decode_override or {}).get("temperature", 0.0) or 0.0
                ),
                "request_index_offset": float(request_index_offset),
            }
        )
        marker = 101 + len(rollout_calls)
        return [([marker], "", "sampling", []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    def _parse_with_optional_empty_view(**kwargs):
        marker = int(kwargs["response_token_ids"][0])
        valid_objects = []
        if marker in {102, 103, 105}:
            valid_objects = [
                types.SimpleNamespace(
                    index=0,
                    geom_type="bbox_2d",
                    coord_token_indices=[0, 1, 2, 3],
                    desc="obj",
                )
            ]
        return types.SimpleNamespace(
            prefix_token_ids=[],
            prefix_text='{"objects": [',
            response_token_ids=list(kwargs["response_token_ids"]),
            response_text="",
            valid_objects=valid_objects,
            dropped_invalid_by_reason={},
            dropped_invalid=0,
            dropped_ambiguous=0,
            truncated=False,
            invalid_rollout=False,
        )

    with monkeypatch.context() as mp:
        mp.setattr(
            "src.trainers.rollout_correction.parse_rollout_for_matching",
            _parse_with_optional_empty_view,
        )
        mp.setattr(
            "src.trainers.rollout_correction.points_from_coord_tokens",
            lambda **kwargs: [10, 10, 20, 20],
        )
        mp.setattr(
            "src.trainers.rollout_correction._extract_gt_bboxonly",
            lambda _sample: [],
        )
        mp.setattr(
            "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
            lambda **kwargs: types.SimpleNamespace(
                matched_pairs=[],
                fn_gt_indices=[],
                fp_pred_indices=list(range(len(kwargs["preds"]))),
                gating_rejections=0,
                matched_maskiou_sum=0.0,
                matched_maskiou_count=0,
            ),
        )

        sample = {"messages": [], "assistant_payload": {"objects": []}}
        segments, batch_metrics = t._prepare_rollout_correction_inputs(
            [sample],
            _segments_only=True,
        )

    assert [call["temperature"] for call in rollout_calls] == [
        pytest.approx(0.7),
        pytest.approx(0.7),
        pytest.approx(0.7),
        pytest.approx(0.7),
    ]
    assert [call["request_index_offset"] for call in rollout_calls] == [
        pytest.approx(0.0),
        pytest.approx(1.0),
        pytest.approx(2.0),
        pytest.approx(3.0),
    ]

    meta = segments[0][1]
    assert meta["shielded_anchor_indices"] == [0]
    assert meta["dead_anchor_indices"] == []
    assert meta["pseudo_positive_anchor_indices"] == []
    assert meta["valid_explorer_count"] == 3
    assert meta["anchor_support_counts"] == [2]
    assert meta["anchor_support_rates"] == pytest.approx([2.0 / 3.0])
    assert batch_metrics["stage2/raw_rollouts"] == pytest.approx(4.0)
    assert batch_metrics["rollout/peer/pred_objects"] == pytest.approx(0.75)
    assert batch_metrics["rollout/peer/valid_pred_objects"] == pytest.approx(
        0.75
    )
    assert batch_metrics["rollout/peer/parse_truncated_rate"] == pytest.approx(0.0)
    assert batch_metrics["rollout/explorer/pred_objects"] == pytest.approx(
        batch_metrics["rollout/peer/pred_objects"]
    )


def test_rollout_correction_enabled_pseudo_positive_aborts_on_invalid_explorer(
    monkeypatch,
) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    ab_cfg = {
        "pseudo_positive.enabled": True,
        "triage_posterior.num_rollouts": 4,
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.explorer_top_p": 0.95,
        "triage_posterior.explorer_top_k": -1,
        "invalid_rollout_policy": "abort",
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: ab_cfg.get(key, default)

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls = 0

    def _fake_rollout_many(chunk, decode_override=None, request_index_offset=0):
        nonlocal rollout_calls
        rollout_calls += 1
        marker = 101 + rollout_calls
        return [([marker], "", "sampling", []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    def _parse_with_invalid_middle_explorer(**kwargs):
        marker = int(kwargs["response_token_ids"][0])
        valid_objects = [
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="obj",
            )
        ]
        invalid_rollout = marker == 104
        return types.SimpleNamespace(
            prefix_token_ids=[],
            prefix_text='{"objects": [',
            response_token_ids=list(kwargs["response_token_ids"]),
            response_text="",
            valid_objects=[] if invalid_rollout else valid_objects,
            dropped_invalid_by_reason={},
            dropped_invalid=0,
            dropped_ambiguous=0,
            truncated=False,
            invalid_rollout=invalid_rollout,
        )

    with monkeypatch.context() as mp:
        mp.setattr(
            "src.trainers.rollout_correction.parse_rollout_for_matching",
            _parse_with_invalid_middle_explorer,
        )
        mp.setattr(
            "src.trainers.rollout_correction.points_from_coord_tokens",
            lambda **kwargs: [10, 10, 20, 20],
        )
        mp.setattr(
            "src.trainers.rollout_correction._extract_gt_bboxonly",
            lambda _sample: [],
        )
        mp.setattr(
            "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
            lambda **kwargs: types.SimpleNamespace(
                matched_pairs=[],
                fn_gt_indices=[],
                fp_pred_indices=list(range(len(kwargs["preds"]))),
                gating_rejections=0,
                matched_maskiou_sum=0.0,
                matched_maskiou_count=0,
            ),
        )

        sample = {
            "messages": [],
            "assistant_payload": {"objects": []},
            "sample_id": "sample-0",
            "image_id": "image-0",
        }
        with pytest.raises(
            ValueError,
            match=(
                r"invalid_labels=\['attempt_2'\].*"
                r"sample_id=sample-0.*image_id=image-0.*manual_analysis_required=true"
            ),
        ):
            t._prepare_rollout_correction_inputs([sample], _segments_only=True)


def test_rollout_correction_triage_enabled_k2_remains_no_promotion_control() -> None:
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 10, 20, 20],
            desc="obj",
        )
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 10, 20, 20],
                desc="obj",
            )
        ]
    ]

    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=anchor_objects,
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    assert triage.valid_explorer_count == 1
    assert triage.anchor_support_counts == [1]
    assert triage.anchor_support_rates == pytest.approx([1.0])
    assert triage.pseudo_positive_candidate_indices == []
    assert triage.pseudo_positive_anchor_indices == []
    assert triage.shielded_anchor_indices == [0]
    assert triage.dead_anchor_indices == []


def test_rollout_correction_triage_pseudo_positive_requires_expected_peer_consensus() -> None:
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 10, 20, 20],
            desc="obj",
        )
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=view_i,
                geom_type="bbox_2d",
                points_norm1000=[10, 10, 20, 20],
                desc="obj",
            )
        ]
        for view_i in range(3)
    ]

    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=anchor_objects,
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
        expected_peer_count=4,
    )

    assert triage.valid_explorer_count == 3
    assert triage.anchor_support_counts == [3]
    assert triage.anchor_support_rates == pytest.approx([0.75])
    assert triage.pseudo_positive_candidate_indices == []
    assert triage.pseudo_positive_anchor_indices == []
    assert triage.shielded_anchor_indices == [0]
    assert triage.dead_anchor_indices == []


def test_rollout_correction_triage_clusters_pseudo_positive_candidates_by_support_rate() -> None:
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[0, 0, 100, 100],
            desc="obj-a",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[0, 0, 100, 90],
            desc="obj-b",
        ),
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="obj-a",
            ),
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 90],
                desc="obj-b",
            ),
        ],
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="obj-a",
            ),
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 90],
                desc="obj-b",
            ),
        ],
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="obj-a",
            ),
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 90],
                desc="obj-b",
            ),
        ],
    ]

    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=anchor_objects,
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    assert triage.valid_explorer_count == 3
    assert triage.anchor_support_counts == [3, 3]
    assert triage.anchor_support_rates == pytest.approx([1.0, 1.0])
    assert triage.pseudo_positive_candidate_indices == [0, 1]
    assert triage.pseudo_positive_anchor_indices == [0]
    assert triage.pseudo_positive_cluster_demoted_indices == [1]
    assert triage.shielded_anchor_indices == [1]
    assert triage.dead_anchor_indices == []


def test_rollout_correction_triage_lvis_policy_forces_verified_dead_and_only_shields_ambiguous() -> (
    None
):
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[0, 0, 20, 20],
            desc="dog",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[30, 0, 50, 20],
            desc="bicycle",
        ),
        GTObject(
            index=2,
            geom_type="bbox_2d",
            points_norm1000=[0, 30, 20, 50],
            desc="zebra",
        ),
        GTObject(
            index=3,
            geom_type="bbox_2d",
            points_norm1000=[30, 30, 50, 50],
            desc="cat",
        ),
    ]
    explorer_objects_by_view = [list(anchor_objects)]

    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=anchor_objects,
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}],
        anchor_policy_statuses=[
            "verified_negative",
            "not_exhaustive",
            None,
            "verified_positive",
        ],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=False,
    )

    assert triage.anchor_support_counts == [0, 1, 1, 0]
    assert triage.lvis_verified_negative_dead_anchor_indices == [0]
    assert triage.lvis_not_exhaustive_anchor_indices == [1]
    assert triage.lvis_unevaluable_anchor_indices == [2]
    assert triage.lvis_verified_positive_dead_anchor_indices == [3]
    assert triage.shielded_anchor_indices == [1, 2]
    assert triage.dead_anchor_indices == [0, 3]


def _build_fp_context_correction_targets(
    *,
    support_count: int,
    fp_policy_mode: str,
    fp_policy_min_support_count: int = 1,
    fp_policy_require_token_score: bool = False,
    tokenizer: object | None = None,
    rollout_template_family: str = "compact_full",
):
    tok = tokenizer if tokenizer is not None else _CoordLiteralTokenizer()
    anchor = GTObject(
        index=0,
        geom_type="bbox_2d",
        points_norm1000=[10, 20, 30, 40],
        desc="unmatched-anchor",
    )
    explorer_objects_by_view = [[anchor] for _ in range(int(support_count))]
    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=[anchor],
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{} for _ in explorer_objects_by_view],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=False,
    )

    return _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=False,
        pseudo_positive_coord_weight=0.4,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
        fp_policy_mode=fp_policy_mode,
        fp_policy_weak_positive_weight=0.05,
        fp_policy_min_support_count=int(fp_policy_min_support_count),
        fp_policy_require_token_score=bool(fp_policy_require_token_score),
        rollout_template_policy=resolve_stage2_rollout_template_policy(
            str(rollout_template_family)
        ),
    )


def test_rollout_correction_zero_loss_fp_context_records_neutral_span_and_shields_trie_targets() -> None:
    targets = _build_fp_context_correction_targets(
        support_count=1,
        fp_policy_mode="zero_loss_context",
    )
    span = targets.stage2_trie_object_spans[0]
    terminal_token_id = 1234
    meta = {
        "rollout_context": "rollout_valid_with_fn_append",
        "rollout_fallback_loss_weight": 0.25,
    }

    _attach_stage2_trie_sidecar_to_meta(
        meta_entry=meta,
        y_train_ids=targets.y_train_ids,
        prompt_len=3,
        sample_id="fp-context-neutral",
        rollout_index=0,
        stage2_trie_object_spans=targets.stage2_trie_object_spans,
        assistant_span_ids=list(targets.y_train_ids) + [terminal_token_id],
    )

    covered_positions = set(range(3 + span.token_start, 3 + span.token_end))
    emitted_positions = {
        target.position for target in meta["stage2_trie_targets"].token_targets
    }
    eos_targets = [
        target
        for target in meta["stage2_trie_targets"].token_targets
        if target.semantic_role == "eos"
    ]
    assert span.role == "neutral_fp"
    assert span.loss_weight == pytest.approx(0.0)
    assert covered_positions
    assert emitted_positions.isdisjoint(covered_positions)
    assert len(eos_targets) == 1
    assert eos_targets[0].position == 3 + len(targets.y_train_ids)
    assert eos_targets[0].positive_token_ids == (terminal_token_id,)
    assert meta["stage2_trie_candidate_summary"]["weak_positive_fp_count"] == 0


def test_rollout_correction_weak_positive_fp_context_records_json_safe_span_and_fallback_flag() -> None:
    tok = _CoordLiteralTokenizer()
    targets = _build_fp_context_correction_targets(
        support_count=1,
        fp_policy_mode="weak_positive_context",
        tokenizer=tok,
    )
    meta = {
        "rollout_context": "rollout_valid_with_fn_append",
        "rollout_fallback_loss_weight": 0.25,
    }

    _attach_stage2_trie_sidecar_to_meta(
        meta_entry=meta,
        y_train_ids=targets.y_train_ids,
        prompt_len=3,
        sample_id="fp-context-weak",
        rollout_index=0,
        stage2_trie_object_spans=targets.stage2_trie_object_spans,
        stage2_trie_weak_fp_span_level_fallback=(
            targets.stage2_trie_weak_fp_span_level_fallback
        ),
    )

    neutral_span = next(
        span
        for span in targets.stage2_trie_object_spans
        if span.role == "neutral_fp"
    )
    weak_span = next(
        span
        for span in targets.stage2_trie_object_spans
        if span.role == "weak_positive_fp"
    )
    desc_start = len(OBJECT_REF_START_TOKEN)
    desc_end = int(desc_start + len("unmatched-anchor"))
    desc_positions = set(range(int(desc_start), int(desc_end)))
    coord_positions = {
        int(idx)
        for idx, token_id in enumerate(targets.y_train_ids)
        if 0 <= int(token_id) <= 999
    }
    weak_positions = set(range(int(weak_span.token_start), int(weak_span.token_end)))
    weak_span_text = tok.decode(
        targets.y_train_ids[weak_span.token_start : weak_span.token_end]
    )
    emitted_positions = {
        int(target.position) - 3
        for target in meta["stage2_trie_targets"].token_targets
    }

    assert neutral_span.loss_weight == pytest.approx(0.0)
    assert weak_span.loss_weight == pytest.approx(0.05)
    assert weak_span_text == OBJECT_REF_START_TOKEN
    assert weak_positions
    assert weak_positions.isdisjoint(desc_positions)
    assert weak_positions.isdisjoint(coord_positions)
    assert emitted_positions.isdisjoint(desc_positions)
    assert emitted_positions.isdisjoint(coord_positions)
    assert targets.stage2_trie_weak_fp_span_level_fallback is True
    assert meta["stage2_trie_candidate_summary"]["weak_positive_fp_count"] == 1
    assert meta["stage2_trie_weak_fp_span_level_fallback"] is True
    weak_score = next(
        item
        for item in meta["stage2_trie_span_scores"]
        if item["span_role"] == "weak_positive_fp"
    )
    assert weak_score["loss_weight"] == pytest.approx(0.05)
    json.dumps(meta["stage2_trie_span_scores"], allow_nan=False)


def test_rollout_correction_weak_positive_fp_context_with_insufficient_support_stays_neutral() -> None:
    targets = _build_fp_context_correction_targets(
        support_count=1,
        fp_policy_mode="weak_positive_context",
        fp_policy_min_support_count=2,
    )

    assert len(targets.stage2_trie_object_spans) == 1
    assert targets.stage2_trie_object_spans[0].role == "neutral_fp"
    assert targets.stage2_trie_object_spans[0].support_count == 1
    assert targets.stage2_trie_object_spans[0].loss_weight == pytest.approx(0.0)
    assert targets.stage2_trie_weak_fp_span_level_fallback is False


def test_rollout_correction_weak_positive_fp_context_rejects_non_compact_v0() -> None:
    with pytest.raises(
        ValueError,
        match="weak_positive_context.*compact_full.*v0",
    ):
        _build_fp_context_correction_targets(
            support_count=1,
            fp_policy_mode="weak_positive_context",
            rollout_template_family="coordjson",
        )


def test_rollout_correction_weak_positive_fp_context_raises_when_marker_span_not_isolated() -> None:
    with pytest.raises(
        ValueError,
        match="object marker token span",
    ):
        _build_fp_context_correction_targets(
            support_count=1,
            fp_policy_mode="weak_positive_context",
            tokenizer=_CompactMarkerDescMergingTokenizer(),
        )


def test_rollout_correction_weak_positive_fp_context_require_token_score_raises_until_supported() -> None:
    with pytest.raises(
        ValueError,
        match="token-score-gated weak-positive false positives.*not implemented",
    ):
        _build_fp_context_correction_targets(
            support_count=1,
            fp_policy_mode="weak_positive_context",
            fp_policy_require_token_score=True,
        )


def test_rollout_correction_unknown_fp_context_policy_raises_helper_style_error() -> None:
    with pytest.raises(ValueError, match="zero_loss_context|weak_positive_context"):
        _build_fp_context_correction_targets(
            support_count=1,
            fp_policy_mode="typo_policy",
        )


def test_rollout_correction_supervision_targets_make_pseudo_positive_coord_only_and_anchor_owned() -> (
    None
):
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="obj",
        )
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="obj",
            )
        ],
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="obj",
            )
        ],
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="obj",
            )
        ],
    ]
    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=anchor_objects,
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert triage.pseudo_positive_anchor_indices == [0]
    assert targets.prefix_struct_pos == []
    assert targets.tail_desc_pos == []
    assert targets.fn_bbox_groups == []
    assert len(targets.prefix_bbox_groups) == 1
    assert targets.prefix_bbox_groups[0]["gt_bins"] == [10, 20, 30, 40]
    assert targets.prefix_bbox_groups[0]["weight"] == pytest.approx(0.4)
    assert targets.prefix_bins == [10, 20, 30, 40]


def test_rollout_correction_supervision_targets_allow_partial_pseudo_positive_coord_for_shielded_anchor() -> (
    None
):
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="obj",
        )
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="obj",
            )
        ],
        [],
        [],
    ]
    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=anchor_objects,
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.6,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert triage.pseudo_positive_anchor_indices == []
    assert triage.shielded_anchor_indices == [0]
    assert targets.prefix_struct_pos == []
    assert targets.tail_desc_pos == []
    assert targets.fn_bbox_groups == []
    assert len(targets.prefix_bbox_groups) == 1
    assert targets.prefix_bbox_groups[0]["gt_bins"] == [10, 20, 30, 40]
    assert targets.prefix_bbox_groups[0]["weight"] == pytest.approx(0.6 * (1.0 / 3.0))
    assert targets.prefix_bins == [10, 20, 30, 40]


def test_rollout_correction_supervision_targets_skip_duplicate_burst_unlikelihood_for_non_duplicate_dead_anchor() -> (
    None
):
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="kept",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[100, 200, 300, 400],
            desc="far-away",
        ),
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="kept",
            )
        ],
        [],
        [],
    ]
    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=anchor_objects,
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert [obj.index for obj in accepted_clean] == [0, 1]
    assert triage.dead_anchor_indices == [1]
    assert targets.duplicate_control_first_divergence_diagnostics == []
    assert targets.duplicate_control_first_divergence_boundary_count == 0


def test_rollout_correction_supervision_targets_keep_duplicate_control_diagnostics_when_duplicate_survivor_is_kept() -> (
    None
):
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="dup",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="dup",
        ),
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="dup",
            )
        ],
        [],
        [],
    ]
    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=anchor_objects,
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert [obj.index for obj in accepted_clean] == [0]
    assert triage.dead_anchor_indices == []
    assert targets.duplicate_control_first_divergence_diagnostics
    assert targets.duplicate_control_first_divergence_boundary_count == 1


def test_rollout_correction_supervision_targets_keep_duplicate_control_diagnostics_when_all_cluster_members_die() -> (
    None
):
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    raw_anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="dup",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="dup",
        ),
    ]
    accepted_clean, suppressed_duplicate_objects_by_boundary = (
        _apply_test_duplicate_control(
            parsed_bbox_objects_raw=raw_anchor_objects,
            duplicate_iou_threshold=0.9,
        )
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        explorer_objects_raw_by_view=[[], [], []],
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert [obj.index for obj in accepted_clean] == [0]
    assert triage.dead_anchor_indices == [0]
    assert triage.kept_anchor_objects == []
    assert sorted(triage.suppressed_duplicate_objects_by_boundary.keys()) == [0]
    assert targets.duplicate_control_first_divergence_diagnostics
    assert targets.duplicate_control_first_divergence_boundary_count == 1


def test_rollout_correction_supervision_targets_sorted_insertion_reorders_final_sequence() -> (
    None
):
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[400, 500, 450, 560],
            desc="anchor",
        )
    ]
    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=[[], [], []],
        anchor_match_by_pred={0: 0},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=False,
    )
    gts = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[400, 500, 450, 560],
            desc="matched",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="fn",
        ),
    ]

    tail_targets = _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=gts,
        match=types.SimpleNamespace(matched_pairs=[(0, 0)]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=False,
        pseudo_positive_coord_weight=0.4,
        duplicate_iou_threshold=0.9,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )
    sorted_targets = _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=gts,
        match=types.SimpleNamespace(matched_pairs=[(0, 0)]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=False,
        pseudo_positive_coord_weight=0.4,
        duplicate_iou_threshold=0.9,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
        insertion_order="sorted",
    )

    assert tail_targets.clean_target_text.find(
        '"desc": "anchor"'
    ) < tail_targets.clean_target_text.find('"desc": "fn"')
    assert sorted_targets.clean_target_text.find(
        '"desc": "fn"'
    ) < sorted_targets.clean_target_text.find('"desc": "anchor"')
    assert len(tail_targets.fn_bbox_groups) == 1
    assert sorted_targets.fn_bbox_groups == []
    assert any(
        group["gt_bins"] == [10, 20, 30, 40]
        for group in sorted_targets.prefix_bbox_groups
    )
    assert sorted_targets.append_text == "]}"
    assert sorted_targets.tail_desc_pos == []


def test_rollout_correction_compact_full_sorted_insertion_marks_fn_prefix_desc_positions() -> (
    None
):
    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[400, 500, 450, 560],
            desc="anchor",
        )
    ]
    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=[[], [], []],
        anchor_match_by_pred={0: 0},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=False,
    )
    gts = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[400, 500, 450, 560],
            desc="matched",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="fn",
        ),
    ]

    sorted_targets = _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=gts,
        match=types.SimpleNamespace(matched_pairs=[(0, 0)]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=False,
        pseudo_positive_coord_weight=0.4,
        duplicate_iou_threshold=0.9,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
        insertion_order="sorted",
        rollout_template_policy=resolve_stage2_rollout_template_policy("compact_full"),
    )

    assert sorted_targets.clean_target_text.find(
        f"{OBJECT_REF_START_TOKEN}fn"
    ) < sorted_targets.clean_target_text.find(f"{OBJECT_REF_START_TOKEN}anchor")
    assert sorted_targets.append_text == ""
    assert sorted_targets.fn_bbox_groups == []
    assert sorted_targets.tail_desc_pos == []
    assert sorted_targets.prefix_desc_pos
    assert [float(w) for w in sorted_targets.prefix_desc_weights] == [
        pytest.approx(1.0)
    ] * len(sorted_targets.prefix_desc_pos)
    desc_ids = [
        sorted_targets.clean_prefix.prefix_token_ids[int(p)]
        for p in sorted_targets.prefix_desc_pos
    ]
    assert tok.decode(desc_ids) == "fn"
    assert any(
        group["gt_bins"] == [10, 20, 30, 40]
        for group in sorted_targets.prefix_bbox_groups
    )


def test_rollout_correction_compact_full_fn_slot_shuffle_injects_fn_objects_deterministically() -> (
    None
):
    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[400, 500, 450, 560],
            desc="anchor-left",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[600, 500, 650, 560],
            desc="anchor-right",
        ),
    ]
    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=[[], [], []],
        anchor_match_by_pred={0: 0, 1: 1},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=False,
    )
    gts = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[400, 500, 450, 560],
            desc="matched-left",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[600, 500, 650, 560],
            desc="matched-right",
        ),
        GTObject(
            index=2,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="fn-a",
        ),
        GTObject(
            index=3,
            geom_type="bbox_2d",
            points_norm1000=[80, 90, 110, 140],
            desc="fn-b",
        ),
    ]

    def build(seed: int):
        return _build_rollout_correction_supervision_targets(
            tokenizer=tok,
            prompt_ids=[],
            coord_id_set=set(range(1000)),
            gts=gts,
            match=types.SimpleNamespace(matched_pairs=[(0, 0), (1, 1)]),
            triage=triage,
            recovered_ground_truth_weight_multiplier=2.0,
            pseudo_positive_enabled=False,
            pseudo_positive_coord_weight=0.4,
            duplicate_iou_threshold=0.9,
            object_field_order="desc_first",
            bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
            matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
            serialize_append_fragment_fn=_serialize_append_fragment,
            insertion_order="fn_slot_shuffle",
            rollout_template_policy=resolve_stage2_rollout_template_policy(
                "compact_full"
            ),
            shuffle_seed=seed,
        )

    first_targets = build(17)
    replay_targets = build(17)
    different_seed_targets = build(18)

    assert first_targets.clean_target_text == replay_targets.clean_target_text
    assert first_targets.clean_target_text != different_seed_targets.clean_target_text
    assert first_targets.append_text == ""
    assert first_targets.fn_bbox_groups == []
    assert first_targets.tail_desc_pos == []
    assert first_targets.clean_target_text.find(
        f"{OBJECT_REF_START_TOKEN}anchor-left"
    ) < first_targets.clean_target_text.find(f"{OBJECT_REF_START_TOKEN}anchor-right")
    assert all(
        any(group["gt_bins"] == list(obj.points_norm1000) for group in first_targets.prefix_bbox_groups)
        for obj in gts[2:]
    )
    assert first_targets.prefix_desc_pos


@pytest.mark.parametrize("insertion_order", ["sorted", "fn_slot_shuffle"])
def test_rollout_correction_reordered_prefix_fp_spans_remap_to_unmatched_kept_anchor(
    insertion_order: str,
) -> None:
    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[400, 410, 460, 470],
            desc="unmatched-anchor",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[800, 810, 860, 870],
            desc="gt-backed-anchor",
        ),
    ]
    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_rollout_correction_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=[[anchor_objects[0]]],
        anchor_match_by_pred={1: 0},
        explorer_match_by_pred_by_view=[{}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=False,
    )
    gts = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[800, 810, 860, 870],
            desc="matched-gt",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="fn-object",
        ),
    ]

    targets = _build_rollout_correction_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=gts,
        match=types.SimpleNamespace(matched_pairs=[(1, 0)]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=False,
        pseudo_positive_coord_weight=0.4,
        duplicate_iou_threshold=0.9,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
        insertion_order=insertion_order,
        fp_policy_mode="weak_positive_context",
        fp_policy_weak_positive_weight=0.05,
        fp_policy_min_support_count=1,
        rollout_template_policy=resolve_stage2_rollout_template_policy("compact_full"),
        shuffle_seed=1,
    )

    assert targets.clean_target_text.find("fn-object") < targets.clean_target_text.find(
        "unmatched-anchor"
    )
    assert len(targets.stage2_trie_object_spans) == 2
    neutral_span = next(
        span for span in targets.stage2_trie_object_spans if span.role == "neutral_fp"
    )
    weak_span = next(
        span
        for span in targets.stage2_trie_object_spans
        if span.role == "weak_positive_fp"
    )
    neutral_span_text = tok.decode(
        targets.y_train_ids[neutral_span.token_start : neutral_span.token_end]
    )
    weak_span_text = tok.decode(
        targets.y_train_ids[weak_span.token_start : weak_span.token_end]
    )
    assert neutral_span.loss_weight == pytest.approx(0.0)
    assert weak_span.loss_weight == pytest.approx(0.05)
    assert weak_span.support_count == 1
    assert "unmatched-anchor" in neutral_span_text
    assert "gt-backed-anchor" not in neutral_span_text
    assert "fn-object" not in neutral_span_text
    assert weak_span_text == OBJECT_REF_START_TOKEN


def test_rollout_correction_triage_posterior_nested_config_reaches_live_accessor_and_vllm_offsets(
    monkeypatch,
) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)

    t.stage2_rollout_correction_cfg = {
        "correction": {
            "duplicate_control": {
                "iou_threshold": 0.90,
                "center_radius_scale": 0.8,
            },
            "pseudo_positive": {
                "enabled": False,
                "coord_weight": 0.5,
            },
            "triage_posterior": {
                "num_rollouts": 2,
                "explorer_temperature": 0.7,
                "explorer_top_p": 1.0,
                "explorer_top_k": -1,
                "unlabeled_consistent_iou_threshold": 0.85,
                "recovered_ground_truth_weight_multiplier": 2.0,
            },
        }
    }
    t.stage2_rollout_correction_cfg["correction"]["triage_posterior"]["explorer_temperature"] = 0.55
    t.stage2_rollout_correction_cfg["correction"]["triage_posterior"]["rollout_temperatures"] = [
        0.0,
        0.55,
    ]
    t.stage2_rollout_correction_cfg["correction"]["triage_posterior"]["explorer_top_p"] = 0.91
    t.stage2_rollout_correction_cfg["correction"]["triage_posterior"]["explorer_top_k"] = 7
    t.stage2_rollout_correction_cfg["correction"]["triage_posterior"][
        "unlabeled_consistent_iou_threshold"
    ] = 0.82
    t.stage2_rollout_correction_cfg["correction"]["triage_posterior"][
        "recovered_ground_truth_weight_multiplier"
    ] = 3.0

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "vllm"
    t._rollout_decode_batch_size_per_rank = lambda: 2
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls: list[tuple[int, float, int]] = []

    def _fake_rollout_many(chunk, decode_override=None, request_index_offset=0):
        rollout_calls.append(
            (
                int(len(chunk)),
                float((decode_override or {}).get("temperature", 0.0) or 0.0),
                int(request_index_offset),
            )
        )
        marker = (
            101
            if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0
            else 202
        )
        return [([marker], "", "sampling", []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    with monkeypatch.context() as mp:
        mp.setattr(
            "src.trainers.rollout_correction.parse_rollout_for_matching",
            lambda **kwargs: fake_parse,
        )
        mp.setattr(
            "src.trainers.rollout_correction._extract_gt_bboxonly",
            lambda _sample: [
                GTObject(
                    index=0,
                    geom_type="bbox_2d",
                    points_norm1000=[0, 0, 1, 1],
                    desc="gt",
                )
            ],
        )
        mp.setattr(
            "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
            lambda **kwargs: types.SimpleNamespace(
                matched_pairs=[],
                fn_gt_indices=[],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=0.0,
                matched_maskiou_count=0,
            ),
        )
        mp.setattr(
            "src.trainers.rollout_correction._bbox_groups_from_token_ids",
            lambda **kwargs: [[0, 1, 2, 3] for _ in kwargs["gt_objs"]],
        )

        samples = [
            {
                "messages": [],
                "assistant_payload": {
                    "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "gt"}]
                },
            }
            for _ in range(5)
        ]
        _segments, _batch_metrics = t._prepare_rollout_correction_inputs(
            samples,
            _segments_only=True,
        )

    assert t._rollout_correction_cfg_get(
        "triage_posterior.explorer_temperature", None
    ) == pytest.approx(0.55)
    assert t._rollout_correction_cfg_get(
        "triage_posterior.recovered_ground_truth_weight_multiplier", None
    ) == pytest.approx(3.0)
    assert rollout_calls == [
        (2, 0.0, 0),
        (2, 0.0, 2),
        (1, 0.0, 4),
        (2, 0.55, 5),
        (2, 0.55, 7),
        (1, 0.55, 9),
    ]


def test_rollout_correction_anchor_only_gt_hit_projects_anchor_gt_backed(
    monkeypatch,
) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: {
        "triage_posterior.num_rollouts": 1,
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()
    t._rollout_many = lambda chunk, decode_override=None: [
        (
            [
                101
                if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0
                else 202
            ],
            "",
            "sampling",
            [],
        )
        for _ in chunk
    ]

    shared_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="anchor",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="anchor",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: shared_parse,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )

    call_idx = {"n": 0}

    def _fake_match(*, preds, **kwargs):
        idx = int(call_idx["n"])
        call_idx["n"] += 1
        if idx == 0:
            return types.SimpleNamespace(
                matched_pairs=[(0, 0)],
                fn_gt_indices=[],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=1.0,
                matched_maskiou_count=1,
            )
        return types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[0],
            fp_pred_indices=[0],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )

    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [10, 10, 20, 20], "desc": "gt"}],
        },
    }

    segments, batch_metrics = t._prepare_rollout_correction_inputs([sample], _segments_only=True)
    meta = segments[0][1]

    assert meta["anchor_gt_backed_indices"] == [0]
    assert meta["recovered_gt_indices"] == []
    assert meta["dead_anchor_indices"] == []
    assert meta["fn_object_weights"] == []
    assert meta["bbox_groups_prefix"]
    assert batch_metrics["train/triage/gt_backed_count"] == pytest.approx(1.0)


def test_rollout_correction_shielded_anchor_stays_neutral_context(monkeypatch) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: {
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.rollout_temperatures": [0.0, 0.7],
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()
    t._rollout_many = lambda chunk, decode_override=None: [
        (
            [
                101
                if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0
                else 202
            ],
            "",
            "sampling",
            [],
        )
        for _ in chunk
    ]

    shared_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="shield",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="shield",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: shared_parse,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction._extract_gt_bboxonly",
        lambda _sample: [],
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[],
            fp_pred_indices=[0],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
    )

    sample = {"messages": [], "assistant_payload": {"objects": []}}
    segments, batch_metrics = t._prepare_rollout_correction_inputs([sample], _segments_only=True)
    meta = segments[0][1]

    assert meta["shielded_anchor_indices"] == [0]
    assert meta["anchor_gt_backed_indices"] == []
    assert meta["dead_anchor_indices"] == []
    assert meta["prefix_struct_pos"] == []
    assert meta["bbox_groups_prefix"] == []
    assert meta["bbox_groups_fn"] == []
    assert batch_metrics["train/triage/unlabeled_consistent_count"] == pytest.approx(
        2.0
    )


def test_rollout_correction_explorer_only_dead_emits_no_explore_branch(monkeypatch) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: {
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.rollout_temperatures": [0.0, 0.7],
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()
    t._rollout_many = lambda chunk, decode_override=None: [
        (
            [
                101
                if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0
                else 202
            ],
            "",
            "sampling",
            [],
        )
        for _ in chunk
    ]

    anchor_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="anchor",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    explorer_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[202],
        response_text="explorer",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="explorer-only",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: (
            anchor_parse
            if int(kwargs["response_token_ids"][0]) == 101
            else explorer_parse
        ),
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction._extract_gt_bboxonly",
        lambda _sample: [],
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[],
            fp_pred_indices=list(range(len(kwargs["preds"]))),
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
    )

    sample = {"messages": [], "assistant_payload": {"objects": []}}
    segments, batch_metrics = t._prepare_rollout_correction_inputs([sample], _segments_only=True)
    meta = segments[0][1]

    assert meta["dead_explorer_indices_by_view"] == [[0]]
    assert meta["anchor_gt_backed_indices"] == []
    assert meta["shielded_anchor_indices"] == []
    assert meta["dead_anchor_indices"] == []
    assert meta["bbox_groups_prefix"] == []
    assert meta["bbox_groups_fn"] == []
    assert meta["duplicate_control_first_divergence_diagnostics"] == []
    assert batch_metrics["train/triage/explorer_only_dead_count"] == pytest.approx(1.0)


def test_rollout_correction_recovered_ground_truth_weight_multipliers_only_apply_to_recovered_tail_objects(
    monkeypatch,
) -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_correction_cfg_get = lambda key, default=None: {
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.rollout_temperatures": [0.0, 0.7],
        "triage_posterior.recovered_ground_truth_weight_multiplier": 2.5,
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(
                            super().encode(s[i : j + 2], add_special_tokens=False)
                        )
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()
    t._rollout_many = lambda chunk, decode_override=None: [
        (
            [
                101
                if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0
                else 202
            ],
            "",
            "sampling",
            [],
        )
        for _ in chunk
    ]

    anchor_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="anchor",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    explorer_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[202],
        response_text="explorer",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="recovered",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.parse_rollout_for_matching",
        lambda **kwargs: (
            anchor_parse
            if int(kwargs["response_token_ids"][0]) == 101
            else explorer_parse
        ),
    )
    monkeypatch.setattr(
        "src.trainers.rollout_correction.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )

    def _fake_match(*, preds, gts, **kwargs):
        if preds and str(preds[0].desc) == "recovered":
            return types.SimpleNamespace(
                matched_pairs=[(0, 0)],
                fn_gt_indices=[1],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=1.0,
                matched_maskiou_count=1,
            )
        return types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[0, 1],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )

    monkeypatch.setattr(
        "src.trainers.rollout_correction._assign_stage2_rollout_correction_objects",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [
                {"bbox_2d": [10, 10, 20, 20], "desc": "recovered-gt"},
                {"bbox_2d": [30, 30, 40, 40], "desc": "ordinary-fn"},
            ]
        },
    }

    segments, _batch_metrics = t._prepare_rollout_correction_inputs([sample], _segments_only=True)
    meta = segments[0][1]

    assert meta["recovered_gt_indices"] == [0]
    assert meta["fn_object_weights"] == [pytest.approx(2.5), pytest.approx(1.0)]
    assert [group["weight"] for group in meta["bbox_groups_fn"]] == [
        pytest.approx(2.5),
        pytest.approx(1.0),
    ]
    assert {round(float(weight), 2) for weight in meta["tail_desc_weights"]} == {
        1.0,
        2.5,
    }


def test_rollout_correction_tail_desc_weights_scale_desc_ce() -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {
        "correction": {},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        rollout_fn_desc_weight=1.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor([[1100, 1100, 1101, 1101, 1100, 1100]], dtype=torch.long)

    meta_base = {
        "prompt_len": 0,
        "prefix_len": 2,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "prefix_struct_pos": [0],
        "tail_desc_pos": [0, 1],
        "tail_ignore_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    loss_default = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(meta_base)],
            "input_ids": input_ids,
        },
    )

    meta_weighted = dict(meta_base)
    meta_weighted["tail_desc_weights"] = [2.0, 2.0]
    loss_weighted = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [meta_weighted],
            "input_ids": input_ids,
        },
    )

    assert float(loss_weighted.detach().cpu().item()) > float(
        loss_default.detach().cpu().item()
    )


def test_derive_rollout_seed_base_is_deterministic_and_matches_formula():
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.args = types.SimpleNamespace(seed=123)

    out1 = t._derive_rollout_seed_base(global_step=7)
    out2 = t._derive_rollout_seed_base(global_step=7)

    expected = int((123 + 7 * 1000003) & 0x7FFFFFFF)
    assert out1 == expected
    assert out2 == expected
    assert 0 <= out1 <= 0x7FFFFFFF


def test_hf_sampling_seeding_calls_seed_everything(monkeypatch):
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)

    # Verify we call transformers.trainer_utils.set_seed(...) during HF sampling seeding.
    called = {}

    def _fake_set_seed(seed: int):
        called["seed"] = int(seed)

    tu = pytest.importorskip("transformers.trainer_utils")
    monkeypatch.setattr(tu, "set_seed", _fake_set_seed, raising=True)

    with t._hf_sampling_seed_context(
        seed_base=123, backend="hf", do_sample=True
    ) as seeded:
        assert seeded is True
    assert called["seed"] == 123


def test_hf_sampling_seeding_restores_python_rng_state():
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)

    import random

    orig = random.getstate()
    try:
        random.seed(0)
        saved = random.getstate()

        with t._hf_sampling_seed_context(
            seed_base=123, backend="hf", do_sample=True
        ) as seeded:
            assert seeded is True
            _ = random.random()  # mutate RNG under the seeded context

        # After exit, RNG state should be restored to `saved`.
        assert random.getstate() == saved
    finally:
        random.setstate(orig)


def test_packing_enabled_requires_qwen_packing_metadata():
    trainer = _make_min_trainer()
    trainer.rollout_matching_cfg = {"packing_enabled": True}
    model = _DummyModel()

    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [
                {"pos": [2, 3, 4, 5], "gt_bins": [0, 1, 2, 3]},
            ],
        }
    ]

    with pytest.raises(ValueError, match="packing enabled"):
        trainer.compute_loss(
            model,
            {
                "_stage2_rollout_correction_phase": "rollout_correction",
                "_rollout_matching_meta": meta,
                "input_ids": input_ids,
            },
        )


def test_post_rollout_packing_selector_is_remainder_aware():
    pytest.importorskip("binpacking")

    from src.trainers.stage2_rollout_runtime import Stage2RolloutRuntime

    packing_length = 10
    min_fill_ratio = 0.5
    target_len = int(math.ceil(float(min_fill_ratio) * float(packing_length)))

    # Construct a small pool where FIFO-greedy produces a tiny remainder pack,
    # but a stage-1-like "smart" plan can avoid underfill by leaving a small
    # segment to pair with the leftover medium segment.
    lens = [9, 6, 4, 4, 1]

    def _legacy_fifo(buf_lens: Sequence[int]) -> List[int]:
        used = int(buf_lens[0])
        sel = [0]
        for i in range(1, len(buf_lens)):
            sl = int(buf_lens[i])
            if sl <= 0:
                continue
            if used + sl <= int(packing_length):
                sel.append(int(i))
                used += sl
        return sel

    def _simulate(buf_lens: Sequence[int], *, selector) -> int:
        buf = [int(x) for x in buf_lens]
        underfilled = 0
        while buf:
            idx = selector(buf)
            assert idx
            assert idx[0] == 0

            total = int(sum(int(buf[i]) for i in idx))
            if total < int(target_len):
                underfilled += 1

            for i in reversed(idx):
                buf.pop(int(i))
        return int(underfilled)

    legacy_underfilled = _simulate(lens, selector=_legacy_fifo)

    def _smart(buf_lens: Sequence[int]) -> List[int]:
        return Stage2RolloutRuntime._select_post_rollout_segment_indices(
            buf_lens,
            packing_length,
            min_fill_ratio=min_fill_ratio,
        )

    smart_underfilled = _simulate(lens, selector=_smart)

    assert legacy_underfilled == 1
    assert smart_underfilled == 0


def test_extract_gt_bboxonly_rejects_poly_geometry():
    sample = {"assistant_payload": {"objects": [{"desc": "x", "poly": [0, 1, 2, 3]}]}}
    with pytest.raises(ValueError, match="bbox-only v1"):
        _extract_gt_bboxonly(sample)


def test_extract_gt_bboxonly_rejects_other_geometry_key_even_with_bbox():
    sample = {
        "assistant_payload": {
            "objects": [
                {
                    "desc": "x",
                    "bbox_2d": [0, 0, 10, 10],
                    "mask_rle": {"counts": "abc", "size": [1, 1]},
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="bbox-only v1"):
        _extract_gt_bboxonly(sample)


def test_extract_gt_bboxonly_rejects_invalid_bbox_order():
    sample = {
        "assistant_payload": {"objects": [{"desc": "x", "bbox_2d": [10, 10, 0, 0]}]},
    }
    with pytest.raises(ValueError, match="invalid bbox_2d"):
        _extract_gt_bboxonly(sample)


def test_build_teacher_forced_payload_honors_object_field_order():
    gt_objects = [
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[1, 2, 3, 4],
            desc="cat",
        )
    ]

    desc_first = _build_teacher_forced_payload(
        gt_objects=gt_objects, object_field_order="desc_first"
    )
    geometry_first = _build_teacher_forced_payload(
        gt_objects=gt_objects, object_field_order="geometry_first"
    )

    assert list(desc_first["objects"][0].keys()) == ["desc", "bbox_2d"]
    assert list(geometry_first["objects"][0].keys()) == ["bbox_2d", "desc"]


def test_extract_gt_bboxonly_preserves_assistant_payload_order() -> None:
    sample = {
        "assistant_payload": {
            "objects": [
                {"desc": "later", "bbox_2d": [10, 10, 20, 20]},
                {"desc": "earlier", "bbox_2d": [0, 0, 5, 5]},
            ]
        }
    }

    gt_objects = _extract_gt_bboxonly(sample)

    assert [obj.desc for obj in gt_objects] == ["later", "earlier"]
    assert [obj.index for obj in gt_objects] == [0, 1]


def test_build_canonical_prefix_text_data_preserves_object_sequence() -> None:
    gt_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 10, 20, 20],
            desc="later",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[0, 0, 5, 5],
            desc="earlier",
        ),
    ]

    prefix_text, boundary_prefix_texts, _spans = _build_canonical_prefix_text_data(
        objects=gt_objects,
        object_field_order="desc_first",
    )

    assert prefix_text.index("later") < prefix_text.index("earlier")
    assert "later" in boundary_prefix_texts[1]
    assert "earlier" not in boundary_prefix_texts[1]


def test_stage2_rollout_correction_fragment_supports_geometry_first_order():
    frag = _serialize_append_fragment(
        fn_objects=[
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[5, 6, 7, 8],
                desc="fn",
            )
        ],
        prefix_text='{"objects": [',
        object_field_order="geometry_first",
    )
    assert frag.index('"bbox_2d"') < frag.index('"desc"')


def test_compute_loss_raises_on_sliced_logits():
    trainer = _make_min_trainer()
    model = _DummySlicedModel()

    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    position_ids = torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long)
    text_position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [
                {"pos": [2, 3, 4, 5], "gt_bins": [0, 1, 2, 3]},
            ],
        }
    ]

    with pytest.raises(ValueError, match="sliced logits"):
        trainer.compute_loss(
            model,
            {
                "_stage2_rollout_correction_phase": "rollout_correction",
                "_rollout_matching_meta": meta,
                "input_ids": input_ids,
                "position_ids": position_ids,
                "text_position_ids": text_position_ids,
            },
        )


def test_rollout_correction_unused_meta_flag_does_not_change_supervision_semantics() -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {
        "correction": {},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor([[1100, 1100, 1101, 1101, 1100, 1100]], dtype=torch.long)

    base_meta = {
        "prompt_len": 0,
        "prefix_len": 0,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "tail_ignore_pos": [],
        "tail_desc_pos": [2, 3],
        "bbox_groups_prefix": [{"pos": [1, 2, 3, 4], "gt_bins": [1, 1, 2, 2]}],
        "bbox_groups_fn": [{"pos": [1, 2, 3, 4], "gt_bins": [1, 1, 2, 2]}],
    }

    loss_no_repeat = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(base_meta, legacy_unused_flag=0)],
            "input_ids": input_ids,
        },
    )
    loss_with_repeat = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(base_meta, legacy_unused_flag=1)],
            "input_ids": input_ids,
        },
    )

    assert float(loss_no_repeat.detach().cpu().item()) == pytest.approx(
        float(loss_with_repeat.detach().cpu().item())
    )


def test_rollout_correction_tail_ignore_pos_masks_ce_tokens():
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {
        "correction": {},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor([[1100, 1100, 1101, 1101, 1100, 1100]], dtype=torch.long)

    meta_base = {
        "prompt_len": 0,
        "prefix_len": 0,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "tail_ignore_pos": [],
        "tail_desc_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    loss_full = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(meta_base)],
            "input_ids": input_ids,
        },
    )

    # Mask the two wrong tokens from CE.
    meta_mask = dict(meta_base)
    meta_mask["tail_ignore_pos"] = [2, 3]
    loss_masked = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [meta_mask],
            "input_ids": input_ids,
        },
    )

    assert float(loss_masked.detach().cpu().item()) < float(
        loss_full.detach().cpu().item()
    )


def test_matched_prefix_structure_positions_excludes_desc_and_fp_tokens():
    tok = _DummyTokenizer()

    prefix_text = (
        '{"objects":[{"desc":"matched","bbox_2d":[0,0,1,1]},'
        '{"desc":"fp","bbox_2d":[2,2,3,3]}'
    )
    prefix_token_ids = list(tok.encode(prefix_text))

    first_obj_anchor = int(prefix_text.find('{"desc":"matched"'))
    value_start = int(first_obj_anchor)
    value_end = int(prefix_text.find('},{"desc":"fp"')) + 1
    matched_obj = types.SimpleNamespace(
        value_span=(value_start, value_end),
    )

    rel = _matched_prefix_structure_positions(
        tokenizer=tok,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        matched_pred_objects=[matched_obj],
    )

    struct_pos_matched = int(prefix_text.find('"bbox_2d"'))
    struct_pos_fp = int(prefix_text.find('{"desc":"fp"'))
    desc_pos_matched = int(prefix_text.find("matched"))

    assert struct_pos_matched >= 0 and struct_pos_matched in rel
    assert struct_pos_fp >= 0 and struct_pos_fp not in rel
    assert desc_pos_matched >= 0 and desc_pos_matched not in rel


def test_matched_prefix_structure_positions_uses_parser_char_frame_for_span_checks():
    tok = _PieceFrameMismatchTokenizer()

    source_text = '{"objects":[{"desc":"~","bbox_2d":[0,0,1,1]}'
    prefix_token_ids = list(tok.encode(source_text))

    prefix_text = tok.decode(
        prefix_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    piece_text = "".join(
        tok.decode(
            [int(t)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for t in prefix_token_ids
    )
    assert int(len(piece_text)) > int(len(prefix_text))

    value_start = int(piece_text.find('{"desc":'))
    value_end = int(piece_text.rfind("}")) + 1
    matched_obj = types.SimpleNamespace(
        value_span=(value_start, value_end),
    )

    rel = _matched_prefix_structure_positions(
        tokenizer=tok,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        matched_pred_objects=[matched_obj],
    )

    key_pos = int(piece_text.find('"bbox_2d"'))
    desc_tok_idx = next(
        i for i, t in enumerate(prefix_token_ids) if int(t) == int(tok._mismatch_id)
    )

    assert key_pos >= 0 and key_pos in rel
    assert desc_tok_idx not in rel


def test_rollout_correction_prefix_structure_supervision_uses_global_prefix_knob():
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {
        "correction": {},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)

    # With the global prefix knob enabled, all non-coordinate prefix tokens
    # contribute structure CE regardless of prefix_struct_pos sparsity.
    input_ids = torch.tensor(
        [[1100, 1101, 1101, 1101, 1100, 1101, 1100, 1101]], dtype=torch.long
    )

    base_meta = {
        "prompt_len": 0,
        "prefix_len": 4,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "tail_ignore_pos": [],
        "tail_desc_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    meta_matched_only = dict(base_meta)
    meta_matched_only["prefix_struct_pos"] = [1]

    meta_oversupervised = dict(base_meta)
    meta_oversupervised["prefix_struct_pos"] = [1, 2, 3]

    loss_matched_only = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [meta_matched_only],
            "input_ids": input_ids,
        },
    )

    loss_oversupervised = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [meta_oversupervised],
            "input_ids": input_ids,
        },
    )

    assert float(loss_matched_only.detach().cpu().item()) == pytest.approx(
        float(loss_oversupervised.detach().cpu().item()),
        rel=1e-6,
        abs=1e-6,
    )


def test_rollout_correction_fn_desc_default_on_and_can_be_disabled_via_pipeline() -> None:
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {
        "correction": {},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)

    # Prefix len=4 (only rel=1 is supervised struct); Tail len=4 includes
    # one FN desc slot (rel=0) and two closure/EOS-like tail tokens (rel=2,3).
    input_ids = torch.tensor(
        [[1100, 1100, 1100, 1100, 1101, 1100, 1101, 1101]],
        dtype=torch.long,
    )

    meta_base = {
        "prompt_len": 0,
        "prefix_len": 4,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "prefix_struct_pos": [1],
        "tail_desc_pos": [0],
        "tail_ignore_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    # Default: FN desc is supervised.
    loss_default = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(meta_base)],
            "input_ids": input_ids,
        },
    )

    # Disable FN desc supervision via token_ce module config.
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        rollout_fn_desc_weight=0.0,
    )
    loss_fn_desc_off = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(meta_base)],
            "input_ids": input_ids,
        },
    )

    # Mask closure/EOS-like tail positions explicitly; this should also reduce loss,
    # confirming those positions are supervised when not masked.
    meta_mask_tail = dict(meta_base)
    meta_mask_tail["tail_ignore_pos"] = [2, 3]
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
    )
    loss_tail_masked = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [meta_mask_tail],
            "input_ids": input_ids,
        },
    )

    assert float(loss_fn_desc_off.detach().cpu().item()) < float(
        loss_default.detach().cpu().item()
    )
    assert float(loss_tail_masked.detach().cpu().item()) < float(
        loss_default.detach().cpu().item()
    )


def test_stage2_pipeline_default_parity_rollout_correction_desc_weighting_unpacked() -> None:
    desc_w = 0.35

    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {
        "correction": {},
        "desc_ce_weight": float(desc_w),
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "coord_ce_weight": 0.0,
        "coord_gate_weight": 0.0,
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=float(desc_w),
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor(
        [[1100, 1101, 1101, 1101, 1101, 1101, 1101, 1101]], dtype=torch.long
    )
    meta = {
        "prompt_len": 0,
        "prefix_len": 4,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "prefix_struct_pos": [1],
        "tail_desc_pos": [0, 1],
        "tail_ignore_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    loss_from_desc_ce = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(meta)],
            "input_ids": input_ids,
        },
    )

    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        rollout_fn_desc_weight=float(desc_w),
    )
    loss_from_rollout_fn_desc = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(meta)],
            "input_ids": input_ids,
        },
    )

    assert float(loss_from_desc_ce.detach().cpu().item()) == pytest.approx(
        float(loss_from_rollout_fn_desc.detach().cpu().item()), rel=1e-6, abs=1e-6
    )


def test_stage2_pipeline_default_parity_rollout_correction_desc_weighting_packed() -> None:
    desc_w = 0.4

    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {
        "correction": {},
        "desc_ce_weight": float(desc_w),
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "coord_ce_weight": 0.0,
        "coord_gate_weight": 0.0,
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=float(desc_w),
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)

    seg_len = 8
    input_ids = torch.tensor(
        [
            [
                1100,
                1101,
                1101,
                1101,
                1101,
                1101,
                1101,
                1101,
                1100,
                1101,
                1101,
                1101,
                1101,
                1101,
                1101,
                1101,
            ]
        ],
        dtype=torch.long,
    )
    meta = [
        {
            "prompt_len": 0,
            "prefix_len": 4,
            "train_len": seg_len,
            "encoded_len": seg_len,
            "prefix_struct_pos": [1],
            "tail_desc_pos": [0],
            "tail_ignore_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
        },
        {
            "prompt_len": 0,
            "prefix_len": 4,
            "train_len": seg_len,
            "encoded_len": seg_len,
            "prefix_struct_pos": [0, 2],
            "tail_desc_pos": [1],
            "tail_ignore_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
        },
    ]

    loss_from_desc_ce = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(meta[0]), dict(meta[1])],
            "input_ids": input_ids,
        },
    )

    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        rollout_fn_desc_weight=float(desc_w),
    )
    loss_from_rollout_fn_desc = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": [dict(meta[0]), dict(meta[1])],
            "input_ids": input_ids,
        },
    )

    assert float(loss_from_desc_ce.detach().cpu().item()) == pytest.approx(
        float(loss_from_rollout_fn_desc.detach().cpu().item()), rel=1e-6, abs=1e-6
    )


def test_tail_closure_positions_match_same_brace_used_for_fn_injection():
    tok = _DummyTokenizer()

    rollout_text = '{"objects":[{"desc":"p","bbox_2d":[<|coord_0|>,<|coord_0|>,<|coord_1|>,<|coord_1|>]}]}'
    parsed = parse_rollout_for_matching(
        tokenizer=tok,
        response_token_ids=list(tok.encode(rollout_text)),
    )

    fn_fragment = _serialize_append_fragment(
        fn_objects=[
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[5, 5, 6, 6],
                desc="fn",
            )
        ],
        prefix_text=parsed.prefix_text,
    )

    assistant_text = parsed.prefix_text + fn_fragment
    assistant_ids = list(tok.encode(assistant_text))
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))

    ignore_rel = _stage2_ab_tail_closure_positions(
        tokenizer=tok,
        assistant_span_ids=assistant_ids + [im_end_id],
        prefix_len=int(len(parsed.prefix_token_ids)),
    )

    tail_text = assistant_text[int(len(parsed.prefix_token_ids)) :]
    close_rel = int(tail_text.rfind("}"))

    assert close_rel >= 0
    assert ignore_rel == [close_rel, len(tail_text)]


def test_tail_closure_positions_ignore_braces_inside_quoted_desc():
    tok = _DummyTokenizer()

    # Include a literal '}' inside a quoted desc string; closure-marker parsing must
    # ignore it and select the brace that closes the *outermost* JSON object.
    json_text = '{"objects":[{"bbox_2d":[0,0,1,1],"desc":"a } b"}]}'
    ids = list(tok.encode(json_text))
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))

    assistant_span_ids = ids + [im_end_id]
    ignore_rel = _stage2_ab_tail_closure_positions(
        tokenizer=tok,
        assistant_span_ids=assistant_span_ids,
        prefix_len=0,
    )

    assert ignore_rel == [len(json_text) - 1, len(json_text)]


def test_tail_closure_positions_prefer_turn_end_after_json_close():
    tok = _DummyTokenizer()

    # Include a literal '<|im_end|>' substring inside a quoted desc string; closure-marker
    # parsing must select the *turn-end* token that occurs after the outermost JSON close brace.
    json_text = '{"objects":[{"bbox_2d":[0,0,1,1],"desc":"a <|im_end|> b"}]}'
    ids = list(tok.encode(json_text))
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))

    assistant_span_ids = ids + [im_end_id]
    ignore_rel = _stage2_ab_tail_closure_positions(
        tokenizer=tok,
        assistant_span_ids=assistant_span_ids,
        prefix_len=0,
    )

    assert ignore_rel == [len(json_text) - 1, len(json_text)]


def test_rollout_correction_sequential_dedup_attaches_duplicates_to_clean_boundaries() -> None:
    raw = [
        GTObject(
            index=0, geom_type="bbox_2d", points_norm1000=[10, 10, 20, 20], desc="cat"
        ),
        GTObject(
            index=1, geom_type="bbox_2d", points_norm1000=[10, 10, 20, 20], desc="cat"
        ),
        GTObject(
            index=2,
            geom_type="bbox_2d",
            points_norm1000=[100, 100, 150, 150],
            desc="dog",
        ),
        GTObject(
            index=3, geom_type="bbox_2d", points_norm1000=[10, 10, 20, 20], desc="cat"
        ),
    ]

    accepted, bursts = _apply_test_duplicate_control(
        parsed_bbox_objects_raw=raw,
        duplicate_iou_threshold=0.90,
    )

    assert [obj.desc for obj in accepted] == ["cat", "dog"]
    assert sorted(bursts.keys()) == [1, 2]
    assert [obj.index for obj in bursts[1]] == [1]
    assert [obj.index for obj in bursts[2]] == [3]

    diag = _compute_duplicate_diagnostics(raw)
    assert diag["dup/raw/max_desc_count"] == pytest.approx(3.0)
    assert diag["dup/raw/near_iou90_pairs_same_desc_count"] == pytest.approx(3.0)
    assert diag["dup/raw/near_iou90_pairs_any_desc_count"] == pytest.approx(3.0)
    assert diag["dup/raw/saturation_rate"] == pytest.approx(0.0)


def test_rollout_correction_sequential_dedup_with_zero_center_radius_still_suppresses_iou_duplicates() -> (
    None
):
    raw = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[100, 100, 200, 200],
            desc="cat",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[110, 110, 210, 210],
            desc="cat",
        ),
    ]

    accepted, bursts = _apply_test_duplicate_control(
        parsed_bbox_objects_raw=raw,
        duplicate_iou_threshold=0.50,
    )

    assert [obj.index for obj in accepted] == [0]
    assert sorted(bursts.keys()) == [1]
    assert [obj.index for obj in bursts[1]] == [1]


def test_duplicate_control_diagnostics_use_lcp_divergence_and_collapse_same_boundary_token() -> (
    None
):
    tok = _DummyTokenizer()
    accepted_clean = [
        GTObject(
            index=0, geom_type="bbox_2d", points_norm1000=[1, 1, 2, 2], desc="cat"
        ),
        GTObject(
            index=1, geom_type="bbox_2d", points_norm1000=[5, 5, 6, 6], desc="book"
        ),
    ]
    duplicate_bursts = {
        1: [
            GTObject(
                index=2, geom_type="bbox_2d", points_norm1000=[1, 1, 2, 2], desc="book"
            ),
            GTObject(
                index=3, geom_type="bbox_2d", points_norm1000=[1, 1, 2, 2], desc="book"
            ),
        ]
    }

    clean_prefix = _build_canonical_prefix_data(
        tokenizer=tok,
        objects=accepted_clean,
        object_field_order="desc_first",
    )
    y_train_ids = list(clean_prefix.prefix_token_ids) + list(tok.encode("]}"))
    diagnostics, diagnostic_boundaries, skipped = (
        _build_duplicate_control_divergence_diagnostics(
        tokenizer=tok,
        y_train_ids=y_train_ids,
        clean_target_text=clean_prefix.prefix_text + "]}",
        accepted_objects_clean=accepted_clean,
        fn_objects=[],
        suppressed_duplicate_objects_by_boundary=duplicate_bursts,
        boundary_prefix_texts=clean_prefix.boundary_prefix_texts,
        object_field_order="desc_first",
        )
    )

    assert len(diagnostics) == 1
    assert diagnostic_boundaries == 1
    assert skipped == 0

    diagnostic = diagnostics[0]
    assert diagnostic["boundary"] == 1
    assert tok.decode([diagnostic["duplicate_token_id"]]) == "1"
    assert tok.decode([y_train_ids[diagnostic["clean_rel_pos"]]]) == "5"


def test_duplicate_control_diagnostics_skip_when_no_safe_divergence_exists() -> (
    None
):
    tok = _DummyTokenizer()
    accepted_clean = [
        GTObject(
            index=0, geom_type="bbox_2d", points_norm1000=[5, 5, 6, 6], desc="book"
        ),
    ]
    duplicate_bursts = {
        0: [
            GTObject(
                index=1, geom_type="bbox_2d", points_norm1000=[1, 1, 2, 2], desc="book"
            ),
        ]
    }

    clean_prefix = _build_canonical_prefix_data(
        tokenizer=tok,
        objects=accepted_clean,
        object_field_order="desc_first",
    )
    y_train_ids = []
    diagnostics, diagnostic_boundaries, skipped = (
        _build_duplicate_control_divergence_diagnostics(
        tokenizer=tok,
        y_train_ids=y_train_ids,
        clean_target_text=clean_prefix.prefix_text + "]}",
        accepted_objects_clean=accepted_clean,
        fn_objects=[],
        suppressed_duplicate_objects_by_boundary=duplicate_bursts,
        boundary_prefix_texts=clean_prefix.boundary_prefix_texts,
        object_field_order="desc_first",
        )
    )

    assert diagnostics == []
    assert diagnostic_boundaries == 0
    assert skipped == 1


def test_duplicate_control_diagnostics_resolve_boundary_crossing_tokenization() -> (
    None
):
    tok = _BoundaryMergingTokenizer()
    accepted_clean = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[5, 5, 6, 6],
            desc="book",
        ),
    ]
    duplicate_bursts = {
        0: [
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[1, 1, 2, 2],
                desc="book",
            ),
        ]
    }

    clean_prefix = _build_canonical_prefix_data(
        tokenizer=tok,
        objects=accepted_clean,
        object_field_order="desc_first",
    )
    y_train_ids = list(clean_prefix.prefix_token_ids) + list(tok.encode("]}"))
    diagnostics, diagnostic_boundaries, skipped = (
        _build_duplicate_control_divergence_diagnostics(
        tokenizer=tok,
        y_train_ids=y_train_ids,
        clean_target_text=clean_prefix.prefix_text + "]}",
        accepted_objects_clean=accepted_clean,
        fn_objects=[],
        suppressed_duplicate_objects_by_boundary=duplicate_bursts,
        boundary_prefix_texts=clean_prefix.boundary_prefix_texts,
        object_field_order="desc_first",
        )
    )

    assert len(diagnostics) == 1
    assert diagnostic_boundaries == 1
    assert skipped == 0

    diagnostic = diagnostics[0]
    assert diagnostic["boundary"] == 0
    assert tok.decode([diagnostic["duplicate_token_id"]]) == "1"
    assert tok.decode([y_train_ids[diagnostic["clean_rel_pos"]]]) == "5"


def test_stage2_rollout_correction_removed_duplicate_burst_unlikelihood_does_not_log_live_loss() -> (
    None
):
    t = _make_min_trainer()
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=True,
        token_ce_weight=1.0,
    )
    model = _DummyCallIndexedTokenModel(pred_ids=[7], vocab=1200)
    input_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    meta = [
        {
            "stage2_surface": "rollout_correction",
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 2,
            "encoded_len": 4,
            "prefix_struct_pos": [],
            "prefix_coord_pos": [],
            "tail_desc_pos": [],
            "tail_ignore_pos": [],
            "tail_closure_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
            "duplicate_control_first_divergence_diagnostics": [
                {"boundary": 0, "clean_rel_pos": 0, "duplicate_token_id": 7},
            ],
        }
    ]

    loss = t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
        },
    )

    assert torch.isfinite(loss.detach()).item()
    pending = t._stage2_pending_train_logs[1].finalize()
    assert "train/optimization/loss_duplicate_burst_unlikelihood" not in pending
    assert "loss/stage2_rollout_correction/duplicate_burst_unlikelihood" not in pending


def test_stage2_rollout_correction_compute_loss_copies_triage_and_split_rollout_telemetry() -> (
    None
):
    t = _make_min_trainer()
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
    )
    model = _DummyAlwaysTokenModel(pred_id=7)
    input_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    meta = [
        {
            "stage2_surface": "rollout_correction",
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 2,
            "encoded_len": 4,
            "prefix_struct_pos": [],
            "prefix_coord_pos": [],
            "tail_desc_pos": [],
            "tail_ignore_pos": [],
            "tail_closure_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
            "duplicate_control_first_divergence_diagnostics": [
                {"boundary": 0, "clean_rel_pos": 0, "duplicate_token_id": 7},
            ],
        }
    ]

    t.compute_loss(
        model,
        {
            "_stage2_rollout_correction_phase": "rollout_correction",
            "_rollout_matching_meta": meta,
            "_rollout_matching_batch_metrics": {
                "train/triage/gt_backed_count": 2.0,
                "train/triage/recovered_ground_truth_count": 1.0,
                "train/triage/recovered_ground_truth_rate_num": 1.0,
                "train/triage/recovered_ground_truth_rate_den": 4.0,
                "train/triage/recovered_ground_truth_rate": 0.25,
                "rollout/current/pred_objects": 5.0,
                "rollout/anchor/pred_objects": 5.0,
                "rollout/peer/pred_objects": 7.0,
                "rollout/explorer/pred_objects": 7.0,
                "rollout/peer/temperature": 0.7,
                "rollout/explorer/temperature": 0.7,
                "rollout/peer/do_sample": 1.0,
                "rollout/explorer/do_sample": 1.0,
                "rollout/matched_for_supervision_over_valid_pred": 0.5,
                "rollout/matched_for_supervision_count": 3.0,
                "rollout/valid_pred_objects_total": 6.0,
            },
            "input_ids": input_ids,
        },
    )

    pending = t._stage2_pending_train_logs[1].finalize()
    assert pending["train/triage/gt_backed_count"] == pytest.approx(2.0)
    assert pending["train/triage/recovered_ground_truth_rate"] == pytest.approx(0.25)
    assert pending["rollout/current/pred_objects"] == pytest.approx(5.0)
    assert pending["rollout/peer/pred_objects"] == pytest.approx(7.0)
    assert pending["rollout/peer/temperature"] == pytest.approx(0.7)
    assert pending["rollout/peer/do_sample"] == pytest.approx(1.0)
    assert pending["rollout/anchor/pred_objects"] == pytest.approx(
        pending["rollout/current/pred_objects"]
    )
    assert pending["rollout/explorer/pred_objects"] == pytest.approx(
        pending["rollout/peer/pred_objects"]
    )
    assert pending["rollout/matched_for_supervision_over_valid_pred"] == pytest.approx(
        0.5
    )
    assert "train/optimization/loss_duplicate_burst_unlikelihood" not in pending
    assert "loss/stage2_rollout_correction/duplicate_burst_unlikelihood" not in pending
    stale_duplicate_diag_prefix = "diag/" + "duplicate_burst" + "/"
    assert all(
        not key.startswith(stale_duplicate_diag_prefix) for key in pending
    )


def test_pending_stage2_log_aggregates_closure_and_invalid_rollout_metrics() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2/raw_rollouts": 3.0,
            "stage2_rollout_correction/invalid_rollout": 1.0,
            "stage2_rollout_correction/closure_supervision/N_drop": 1.0,
            "rollout/_parse_truncated_num": 1.0,
            "rollout/_parse_truncated_den": 3.0,
        }
    )
    pending.add(
        {
            "stage2/raw_rollouts": 7.0,
            "stage2_rollout_correction/invalid_rollout": 2.0,
            "stage2_rollout_correction/closure_supervision/N_drop": 4.0,
            "rollout/_parse_truncated_num": 4.0,
            "rollout/_parse_truncated_den": 7.0,
        }
    )

    out = pending.finalize()

    assert out["stage2/raw_rollouts"] == pytest.approx(10.0)
    assert out["stage2_rollout_correction/invalid_rollout"] == pytest.approx(3.0)
    assert out["stage2_rollout_correction/closure_supervision/N_drop"] == pytest.approx(5.0)
    assert out["rollout/parse_truncated_rate"] == pytest.approx(0.5)
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out


@pytest.mark.parametrize(
    "metric_key",
    [
        "rollout/template_family_compact_full",
        "rollout/decode_policy_unconstrained",
        "rollout/parser_template_mismatch_rate",
        "rollout/invalid_fallback_gt_fn_count",
        "rollout/invalid_fallback_gt_fn_rate",
        "rollout/empty_valid_object_rate",
        "rollout/fallback_loss_share",
        "rollout/fallback_dominance_warning",
        "rollout/fallback_gt_fn_append_only_count",
        "rollout/fallback_loss_weight",
        "rollout/current/pred_objects",
        "rollout/peer/valid_pred_objects",
    ],
)
def test_rollout_correction_direct_batch_metric_filter_keeps_compact_fallback_metrics(
    metric_key: str,
) -> None:
    assert _is_rollout_correction_direct_batch_metric_key(metric_key) is True


def test_rollout_correction_direct_batch_metric_filter_rejects_unscoped_rollout_metric() -> None:
    assert _is_rollout_correction_direct_batch_metric_key("rollout/debug_blob") is False


def test_pending_stage2_log_aggregates_compact_fallback_metrics() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2/_log_weight": 2.0,
            "rollout/invalid_fallback_gt_fn_count": 1.0,
            "rollout/fallback_gt_fn_append_only_count": 1.0,
            "rollout/fallback_loss_share": 0.5,
            "rollout/empty_valid_object_rate": 0.0,
            "rollout/fallback_loss_weight": 1.0,
        }
    )
    pending.add(
        {
            "stage2/_log_weight": 1.0,
            "rollout/invalid_fallback_gt_fn_count": 2.0,
            "rollout/fallback_gt_fn_append_only_count": 2.0,
            "rollout/fallback_loss_share": 1.0,
            "rollout/empty_valid_object_rate": 0.5,
            "rollout/fallback_loss_weight": 0.25,
        }
    )

    out = pending.finalize()

    assert out["rollout/invalid_fallback_gt_fn_count"] == pytest.approx(3.0)
    assert out["rollout/fallback_gt_fn_append_only_count"] == pytest.approx(3.0)
    assert out["rollout/fallback_loss_share"] == pytest.approx(2.0 / 3.0)
    assert out["rollout/empty_valid_object_rate"] == pytest.approx(1.0 / 6.0)
    assert out["rollout/fallback_loss_weight"] == pytest.approx(0.75)


def test_pending_stage2_log_aggregates_strict_drop_metrics_and_reasons() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2_rollout_correction/strict_drop/N_valid_pred": 3.0,
            "stage2_rollout_correction/strict_drop/N_drop_invalid": 2.0,
            "stage2_rollout_correction/strict_drop/reason/order_violation": 1.0,
            "stage2_rollout_correction/strict_drop/reason/wrong_arity": 1.0,
        }
    )
    pending.add(
        {
            "stage2_rollout_correction/strict_drop/N_valid_pred": 4.0,
            "stage2_rollout_correction/strict_drop/N_drop_invalid": 3.0,
            "stage2_rollout_correction/strict_drop/reason/order_violation": 2.0,
            "stage2_rollout_correction/strict_drop/reason/missing_desc": 1.0,
        }
    )

    out = pending.finalize()

    assert out["stage2_rollout_correction/strict_drop/N_valid_pred"] == pytest.approx(7.0)
    assert out["stage2_rollout_correction/strict_drop/N_drop_invalid"] == pytest.approx(5.0)
    assert out[
        "stage2_rollout_correction/strict_drop/reason/order_violation"
    ] == pytest.approx(3.0)
    assert out["stage2_rollout_correction/strict_drop/reason/wrong_arity"] == pytest.approx(
        1.0
    )
    assert out["stage2_rollout_correction/strict_drop/reason/missing_desc"] == pytest.approx(
        1.0
    )


def test_pending_stage2_log_aggregates_residual_set_counts_and_weighted_scalars() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2/_log_weight": 1.0,
            "stage2_rollout_correction/residual_set/atom_count": 2.0,
            "stage2_rollout_correction/residual_set/sequence_count": 1.0,
            "stage2_rollout_correction/residual_set/eos_targets": 1.0,
            "stage2_rollout_correction/residual_set/sequence_loss": 10.0,
            "stage2_rollout_correction/residual_set/type_loss": 1.0,
            "stage2_rollout_correction/residual_set/valid_set_mass": 0.2,
        }
    )
    pending.add(
        {
            "stage2/_log_weight": 3.0,
            "stage2_rollout_correction/residual_set/atom_count": 5.0,
            "stage2_rollout_correction/residual_set/sequence_count": 2.0,
            "stage2_rollout_correction/residual_set/eos_targets": 2.0,
            "stage2_rollout_correction/residual_set/sequence_loss": 20.0,
            "stage2_rollout_correction/residual_set/type_loss": 3.0,
            "stage2_rollout_correction/residual_set/valid_set_mass": 0.8,
        }
    )

    out = pending.finalize()

    assert out["stage2_rollout_correction/residual_set/atom_count"] == pytest.approx(7.0)
    assert out["stage2_rollout_correction/residual_set/sequence_count"] == pytest.approx(3.0)
    assert out["stage2_rollout_correction/residual_set/eos_targets"] == pytest.approx(3.0)
    assert out["stage2_rollout_correction/residual_set/sequence_loss"] == pytest.approx(
        (10.0 * 1.0 + 20.0 * 3.0) / 4.0
    )
    assert out["stage2_rollout_correction/residual_set/type_loss"] == pytest.approx(
        (1.0 * 1.0 + 3.0 * 3.0) / 4.0
    )
    assert out["stage2_rollout_correction/residual_set/valid_set_mass"] == pytest.approx(
        (0.2 * 1.0 + 0.8 * 3.0) / 4.0
    )


def test_pending_stage2_log_omits_rollout_correction_keys_when_not_provided() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "loss/stage2_rollout_correction/stage2_trie_ce": 0.25,
        }
    )

    out = pending.finalize()

    assert out["loss/stage2_rollout_correction/stage2_trie_ce"] == pytest.approx(0.25)
    assert "stage2/channel_a" not in out
    assert "stage2/channel_b" not in out
    assert "stage2_rollout_correction/invalid_rollout" not in out
    assert "stage2_rollout_correction/strict_drop/N_valid_pred" not in out
    assert "stage2_rollout_correction/strict_drop/N_drop_invalid" not in out


def test_reduce_stage2_pending_metrics_global_recomputes_ratio_and_sums_invalid_rollout() -> (
    None
):
    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    class _FakeDist:
        ReduceOp = _FakeReduceOp

        def all_gather_object(self, gathered: list[object], obj: object) -> None:
            for i in range(len(gathered)):
                gathered[i] = list(obj) if isinstance(obj, list) else obj

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            if op == self.ReduceOp.SUM:
                tensor.add_(
                    torch.tensor(
                        [3.0, 6.0, 6.0, 1.0, 2.0],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )

    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer._dist_info = lambda: (0, 2, _FakeDist())

    out = trainer._reduce_stage2_pending_metrics_global(
        {
            "rollout/_parse_truncated_num": 1.0,
            "rollout/_parse_truncated_den": 4.0,
            "stage2/raw_rollouts": 4.0,
            "stage2_rollout_correction/invalid_rollout": 1.0,
            "rollout/parse_truncated": 1.0,
        }
    )

    assert out["rollout/parse_truncated_rate"] == pytest.approx(0.4)
    assert out["stage2_rollout_correction/invalid_rollout"] == pytest.approx(2.0)
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out


def test_reduce_stage2_pending_metrics_global_uses_weight_total_for_means() -> None:
    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    class _FakeDist:
        ReduceOp = _FakeReduceOp

        def all_gather_object(self, gathered: list[object], obj: object) -> None:
            # Mirror the local keys across all ranks (key union should be stable).
            for i in range(len(gathered)):
                gathered[i] = list(obj) if isinstance(obj, list) else obj

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            if op == self.ReduceOp.SUM:
                # Simulate rank1 having: weight_total=3, loss_mean=20 => numerator=60.
                tensor.add_(
                    torch.tensor(
                        [3.0, 60.0],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )

    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer._dist_info = lambda: (0, 2, _FakeDist())

    out = trainer._reduce_stage2_pending_metrics_global(
        {
            "stage2/_log_weight_total": 1.0,
            "loss/stage2_rollout_correction/stage2_trie_ce": 10.0,
        }
    )

    assert out["loss/stage2_rollout_correction/stage2_trie_ce"] == pytest.approx(
        (10.0 * 1.0 + 20.0 * 3.0) / 4.0
    )
    assert "stage2/_log_weight_total" not in out


def test_reduce_stage2_pending_metrics_global_treats_train_optimization_losses_as_weighted_means() -> (
    None
):
    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    class _FakeDist:
        ReduceOp = _FakeReduceOp

        def all_gather_object(self, gathered: list[object], obj: object) -> None:
            for i in range(len(gathered)):
                gathered[i] = list(obj) if isinstance(obj, list) else obj

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            if op == self.ReduceOp.SUM:
                tensor.add_(
                    torch.tensor(
                        [3.0, 60.0],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )

    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer._dist_info = lambda: (0, 2, _FakeDist())

    out = trainer._reduce_stage2_pending_metrics_global(
        {
            "stage2/_log_weight_total": 1.0,
            "loss/stage2_rollout_correction/stage2_trie_ce": 10.0,
        }
    )

    assert out["loss/stage2_rollout_correction/stage2_trie_ce"] == pytest.approx((10.0 * 1.0 + 20.0 * 3.0) / 4.0)
    assert "stage2/_log_weight_total" not in out


def test_reduce_stage2_pending_metrics_global_handles_residual_set_metric_specs() -> None:
    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    class _FakeDist:
        ReduceOp = _FakeReduceOp

        def all_gather_object(self, gathered: list[object], obj: object) -> None:
            for i in range(len(gathered)):
                gathered[i] = list(obj) if isinstance(obj, list) else obj

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            if op == self.ReduceOp.SUM:
                tensor.add_(
                    torch.tensor(
                        [3.0, 5.0, 60.0, 2.4],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )

    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer._dist_info = lambda: (0, 2, _FakeDist())

    out = trainer._reduce_stage2_pending_metrics_global(
        {
            "stage2/_log_weight_total": 1.0,
            "stage2_rollout_correction/residual_set/atom_count": 2.0,
            "stage2_rollout_correction/residual_set/sequence_loss": 10.0,
            "stage2_rollout_correction/residual_set/valid_set_mass": 0.2,
        }
    )

    assert out["stage2_rollout_correction/residual_set/atom_count"] == pytest.approx(7.0)
    assert out["stage2_rollout_correction/residual_set/sequence_loss"] == pytest.approx(
        (10.0 * 1.0 + 20.0 * 3.0) / 4.0
    )
    assert out["stage2_rollout_correction/residual_set/valid_set_mass"] == pytest.approx(
        (0.2 * 1.0 + 0.8 * 3.0) / 4.0
    )
    assert "stage2/_log_weight_total" not in out


def test_stage2_core_loss_logs_preserves_rollout_correction_trie_metrics_only() -> None:
    out = build_stage2_core_loss_logs(
        channel="rollout_correction",
        pipeline_metrics_ctx={
            "stage2_trie/target_positions": 3.0,
            "stage2_trie/branch_points": 2.0,
            "stage2_trie/candidate_count_mean": 4.0,
            "loss/stage2_rollout_correction/stage2_trie_ce": 1.25,
            "loss/stage2_trie_ce": 88.0,
            "trie/target_positions": 99.0,
            "diagnostic/debug_only": 66.0,
        },
        token_ce_module_w=0.0,
        run_a_text=False,
        token_desc_ce_weight=1.0,
        fn_desc_ce_weight=1.0,
    )

    assert out["stage2_trie/target_positions"] == pytest.approx(3.0)
    assert out["stage2_trie/branch_points"] == pytest.approx(2.0)
    assert out["stage2_trie/candidate_count_mean"] == pytest.approx(4.0)
    assert out["loss/stage2_rollout_correction/stage2_trie_ce"] == pytest.approx(1.25)
    assert "trie/target_positions" not in out
    assert "loss/stage2_trie_ce" not in out
    assert "diagnostic/debug_only" not in out


def test_stage2_core_loss_logs_passes_residual_set_metrics_only_under_stable_prefix() -> None:
    out = build_stage2_core_loss_logs(
        channel="rollout_correction",
        pipeline_metrics_ctx={
            "stage2_rollout_correction/residual_set/atom_count": 1.0,
            "stage2_rollout_correction/residual_set/type_loss": 0.5,
            "stage2_rollout_correction/residual_set/valid_set_mass": 0.75,
            "residual_set/atom_count": 99.0,
            "stage2_rollout_correction/residual/atom_count": 88.0,
            "diagnostic/debug_only": 99.0,
        },
        token_ce_module_w=0.0,
        run_a_text=False,
        token_desc_ce_weight=1.0,
        fn_desc_ce_weight=1.0,
    )

    assert out["stage2_rollout_correction/residual_set/atom_count"] == pytest.approx(1.0)
    assert out["stage2_rollout_correction/residual_set/type_loss"] == pytest.approx(0.5)
    assert out["stage2_rollout_correction/residual_set/valid_set_mass"] == pytest.approx(0.75)
    assert "residual_set/atom_count" not in out
    assert "stage2_rollout_correction/residual/atom_count" not in out
    assert "diagnostic/debug_only" not in out


def _make_stage2_residual_role_vocab() -> RoleVocab:
    return RoleVocab(
        schema_token_ids=frozenset({20}),
        text_token_ids=frozenset({10, 11}),
        coord_token_ids=frozenset({30}),
        stop_token_id=40,
    )


def _make_stage2_residual_target_ir(
    *,
    selected_token_id: int = 10,
    valid_token_ids: frozenset[int] = frozenset({10}),
) -> TeacherForcingTargetIR:
    atom = SupervisionAtom(
        batch_index=0,
        logit_position=0,
        target_position=1,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=valid_token_ids,
        selected_token_id=int(selected_token_id),
        latent_valid_token_ids=valid_token_ids,
        coverage_target_weights=None,
        loss_tags=frozenset({"residual_set"}),
        loss_weight=1.0,
        coord_role=None,
        provenance={
            "support_provenance": ("labeled",),
            "correction_kind": "selected_path_singleton",
            "source_position_kind": "unit_test",
        },
    )
    return TeacherForcingTargetIR(
        schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
        atoms=(atom,),
        metadata={
            "objective": "residual_set_correction",
            "position_space": "segment_local",
        },
    )


def test_stage2_objective_pipelines_threads_role_vocab_for_residual_set_sidecar() -> None:
    logits = torch.zeros((1, 2, 50), dtype=torch.float32)
    result = run_stage2_objective_pipelines(
        channel="rollout_correction",
        objective_specs=[
            {
                "name": "residual_set_correction",
                "config": {},
            }
        ],
        diagnostic_specs=[],
        input_ids=torch.tensor([[99, 10]], dtype=torch.long),
        logits=logits,
        logits_ce=logits.clone(),
        meta=[
            {
                "encoded_len": 2,
                "residual_set_target_ir": _make_stage2_residual_target_ir(),
            }
        ],
        coord_token_ids=(30,),
        temperature=1.0,
        token_type_masks={},
        rollout_subset_masks={},
        run_a_text=False,
        warn_once_cache=set(),
        role_vocab=_make_stage2_residual_role_vocab(),
    )

    assert result.pipeline_metrics_ctx[
        "stage2_rollout_correction/residual_set/atom_count"
    ] == pytest.approx(1.0)
    assert "residual_set_correction_contrib" in result.pipeline_ctx_result.state


def test_stage2_objective_pipelines_rebase_residual_sidecars_for_unpacked_rows() -> None:
    input_ids = torch.tensor([[99, 10], [88, 10]], dtype=torch.long)
    logits = torch.full((2, 2, 50), -20.0, dtype=torch.float32)
    logits[:, 0, 10] = 20.0

    result = run_stage2_objective_pipelines(
        channel="rollout_correction",
        objective_specs=[
            {
                "name": "residual_set_correction",
                "config": {},
            }
        ],
        diagnostic_specs=[],
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits.clone(),
        meta=[
            {
                "encoded_len": 2,
                # Stage-2 stores residual sidecars in segment-local coordinates.
                "residual_set_target_ir": _make_stage2_residual_target_ir(),
            },
            {
                "encoded_len": 2,
                "residual_set_target_ir": _make_stage2_residual_target_ir(),
            },
        ],
        coord_token_ids=(30,),
        temperature=1.0,
        token_type_masks={},
        rollout_subset_masks={},
        run_a_text=False,
        warn_once_cache=set(),
        role_vocab=_make_stage2_residual_role_vocab(),
    )

    assert result.pipeline_metrics_ctx[
        "stage2_rollout_correction/residual_set/atom_count"
    ] == pytest.approx(2.0)
    assert result.pipeline_metrics_ctx[
        "stage2_rollout_correction/residual_set/sequence_loss"
    ] == pytest.approx(0.0, abs=1.0e-6)


def test_stage2_objective_pipelines_fail_closed_without_role_vocab_for_residual_set_sidecar() -> None:
    logits = torch.zeros((1, 2, 50), dtype=torch.float32)

    with pytest.raises(ValueError, match="role_vocab"):
        run_stage2_objective_pipelines(
            channel="rollout_correction",
            objective_specs=[
                {
                    "name": "residual_set_correction",
                    "config": {},
                }
            ],
            diagnostic_specs=[],
            input_ids=torch.tensor([[99, 10]], dtype=torch.long),
            logits=logits,
            logits_ce=logits.clone(),
            meta=[
                {
                    "encoded_len": 2,
                    "residual_set_target_ir": _make_stage2_residual_target_ir(),
                }
            ],
            coord_token_ids=(30,),
            temperature=1.0,
            token_type_masks={},
            rollout_subset_masks={},
            run_a_text=False,
            warn_once_cache=set(),
        )


def test_stage2_objective_pipelines_raise_when_residual_sidecar_missing() -> None:
    logits = torch.zeros((1, 2, 50), dtype=torch.float32)

    with pytest.raises(ValueError, match="residual_set_target_ir"):
        run_stage2_objective_pipelines(
            channel="rollout_correction",
            objective_specs=[
                {
                    "name": "residual_set_correction",
                    "config": {},
                }
            ],
            diagnostic_specs=[],
            input_ids=torch.tensor([[99, 10]], dtype=torch.long),
            logits=logits,
            logits_ce=logits.clone(),
            meta=[{"encoded_len": 2}],
            coord_token_ids=(30,),
            temperature=1.0,
            token_type_masks={},
            rollout_subset_masks={},
            run_a_text=False,
            warn_once_cache=set(),
            role_vocab=_make_stage2_residual_role_vocab(),
        )


def test_reduce_stage2_pending_metrics_global_strips_internal_underscore_keys() -> None:
    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer._dist_info = lambda: (0, 1, None)

    out = trainer._reduce_stage2_pending_metrics_global(
        {
            "stage2/_log_weight_total": 2.0,
            "rollout/_parse_truncated_num": 1.0,
            "rollout/_parse_truncated_den": 4.0,
            "rollout/parse_truncated": 1.0,
            "stage2/raw_rollouts": 4.0,
            "loss/stage2_rollout_correction/stage2_trie_ce": 1.0,
        }
    )

    assert "stage2/_log_weight_total" not in out
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out
    assert all(not str(k).startswith("rollout/_") for k in out)


def test_rollout_correction_step_budgeted_path_is_supported_under_ddp_mock(monkeypatch):
    # Stage-2 rollout-correction standardizes training to a single step-budgeted pathway.
    # This should not be rejected just because torch.distributed is initialized.
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(
        torch.distributed, "is_initialized", lambda: True, raising=False
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)

    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t.stage2_rollout_correction_cfg = {"correction": {"mode": "async"}}

    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t.args = types.SimpleNamespace(seed=123)
    t.state = types.SimpleNamespace(global_step=0)

    # Minimal executor shim state.
    t._stage2_rollout_correction_step_gs = None
    t._stage2_rollout_correction_step_micro = 0
    t._stage2_rollout_correction_step_raw = []

    # Avoid heavy rollout/packing work: just confirm the call path is allowed.
    t._stage2_training_step_b_step_mode = (
        lambda model, inputs, global_step: torch.tensor(1.0)
    )

    class _M:
        training = True

    out = t.training_step(_M(), [{"messages": []}])
    assert isinstance(out, torch.Tensor)


def test_rollout_correction_realized_tracks_optimizer_steps_once():
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    t._stage2_rollout_correction_realized_last_gs = None

    t._stage2_record_realized_step(global_step=0, executed_b=False)
    t._stage2_record_realized_step(global_step=0, executed_b=True)  # same step, ignored
    t._stage2_record_realized_step(global_step=1, executed_b=True)

    assert pytest.approx(t._stage2_rollout_correction_realized(), rel=1e-6) == 0.5


def test_merge_rollout_matching_batch_metrics_preserves_existing_keys():
    t = _make_min_trainer()
    batch = {"_rollout_matching_batch_metrics": {"rollout/backend_vllm": 1.0}}
    t._merge_rollout_matching_batch_metrics(
        batch,
        {
            "stage2_rollout_correction/active": 1.0,
            "rollout/backend_vllm": 2.0,
        },
    )
    bm = batch.get("_rollout_matching_batch_metrics")
    assert isinstance(bm, dict)
    assert bm["stage2_rollout_correction/active"] == 1.0
    assert bm["rollout/backend_vllm"] == 2.0


def _make_eval_ready_stage2_rollout_correction_trainer() -> Stage2RolloutCorrectionTrainer:
    trainer = object.__new__(Stage2RolloutCorrectionTrainer)

    class _EvalModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

    trainer.model = _EvalModel()
    trainer.args = types.SimpleNamespace()
    trainer.state = types.SimpleNamespace(global_step=11)
    trainer.control = types.SimpleNamespace(tag="ctrl")
    trainer.template = types.SimpleNamespace(tokenizer=object())
    trainer._cfg = lambda _k, default=None: default
    trainer._desc_monitor_cfg = lambda: {"enabled": False}
    trainer._coord_id_map = lambda: {i: i for i in range(1000)}
    trainer.get_eval_dataloader = lambda _eval_dataset=None: [[{"sample_id": 0}]]
    trainer._rollout_many = lambda batch, prompt_variant_override=None, **_kwargs: [
        ([101], "{}", "greedy", []) for _ in batch
    ]
    trainer._maybe_eval_vllm_colocate_window = lambda **_kwargs: nullcontext()
    trainer.callback_handler = types.SimpleNamespace(
        on_evaluate=lambda args, state, control, metrics: control
    )
    trainer.log = lambda _metrics: None
    return trainer


def test_stage2_rollout_correction_eval_emits_rollout_map_and_coco_contract(
    monkeypatch,
) -> None:
    trainer = _make_eval_ready_stage2_rollout_correction_trainer()

    parse_obj = types.SimpleNamespace(
        response_token_ids=[101],
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="cat",
            )
        ],
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
    )

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime.parse_rollout_for_matching",
        lambda **_kwargs: parse_obj,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime._points_from_coord_tokens",
        lambda **_kwargs: [0, 0, 10, 10],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime._extract_gt_objects",
        lambda _sample: [
            GTObject(
                index=0,
                geom_type="bbox",
                points_norm1000=[0, 0, 10, 10],
                desc="cat",
            )
        ],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime.greedy_match_iou",
        lambda **_kwargs: types.SimpleNamespace(
            matched_pairs=[(0, 0)],
            fp_pred_indices=[],
            fn_gt_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime._compute_eval_detection_coco_metrics",
        lambda **_kwargs: (
            {"bbox_AP": 0.25, "bbox_AP50": 0.5, "segm_AP": 0.75},
            {"empty_pred": 0},
        ),
    )

    logged_metrics: dict[str, float] = {}
    trainer.log = lambda metrics: logged_metrics.update(dict(metrics))

    metrics = trainer.evaluate(metric_key_prefix="eval")

    assert trainer.model.training is True
    assert logged_metrics == metrics
    assert metrics["eval/detection/mAP"] == pytest.approx(0.25)
    assert metrics["eval/runtime/coco_eval_ok"] == pytest.approx(1.0)
    assert all(not k.startswith("eval/detection/bbox_") for k in metrics)
    assert all(not k.startswith("eval/detection/segm_") for k in metrics)


def test_stage2_rollout_correction_eval_raises_when_coco_eval_fails(monkeypatch) -> None:
    trainer = _make_eval_ready_stage2_rollout_correction_trainer()

    parse_obj = types.SimpleNamespace(
        response_token_ids=[101],
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="cat",
            )
        ],
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
    )

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime.parse_rollout_for_matching",
        lambda **_kwargs: parse_obj,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime._points_from_coord_tokens",
        lambda **_kwargs: [0, 0, 10, 10],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime._extract_gt_objects",
        lambda _sample: [
            GTObject(
                index=0,
                geom_type="bbox",
                points_norm1000=[0, 0, 10, 10],
                desc="cat",
            )
        ],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime.greedy_match_iou",
        lambda **_kwargs: types.SimpleNamespace(
            matched_pairs=[(0, 0)],
            fp_pred_indices=[],
            fn_gt_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
    )

    def _raise_coco_eval(**_kwargs):
        raise ValueError("synthetic coco eval failure")

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime._compute_eval_detection_coco_metrics",
        _raise_coco_eval,
    )

    with pytest.raises(RuntimeError, match=r"Eval-step COCO/mAP failed"):
        trainer.evaluate(metric_key_prefix="eval")
