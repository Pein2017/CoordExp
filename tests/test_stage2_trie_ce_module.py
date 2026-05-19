from __future__ import annotations

import types

import pytest
import torch

from src.trainers.stage2_two_channel.trie_supervision import (
    Stage2TrieSummary,
    Stage2TrieTargets,
    Stage2TrieTokenTarget,
)
from src.trainers.teacher_forcing.contracts import (
    PipelineModuleSpec,
    TeacherForcingContext,
)
from src.trainers.teacher_forcing.module_registry import OBJECTIVE_MODULE_CATALOG
from src.trainers.teacher_forcing.modules.stage2_trie_ce import (
    Stage2TrieCEConfig,
    build_stage2_trie_ce_config,
    run_stage2_trie_ce_module,
)
from src.trainers.teacher_forcing.objective_pipeline import (
    run_teacher_forcing_pipeline,
)


def _make_context() -> TeacherForcingContext:
    input_ids = torch.tensor([[7, 11, 12, 13]], dtype=torch.long)
    logits = torch.randn(1, input_ids.shape[1], 32, dtype=torch.float32)

    return TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits,
        meta=[
            {
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
                "tail_ignore_pos": [],
                "tail_desc_pos": [],
                "tail_closure_pos": [],
                "drop_invalid_total": 0,
            }
        ],
        coord_token_ids=[],
        temperature=1.0,
    )


def _make_context_with_logits(
    logits: torch.Tensor,
    *,
    input_ids: torch.Tensor | None = None,
    meta: list[dict[str, object]] | None = None,
    channel: str = "B",
) -> TeacherForcingContext:
    if input_ids is None:
        input_ids = torch.arange(logits.shape[1], dtype=torch.long).unsqueeze(0)

    if meta is None:
        meta = [
            {
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": int(logits.shape[1] - 1),
                "tail_ignore_pos": [],
                "tail_desc_pos": [],
                "tail_closure_pos": [],
                "drop_invalid_total": 0,
            }
        ]

    return TeacherForcingContext(
        channel=channel,
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits,
        meta=meta,
        coord_token_ids=[],
        temperature=1.0,
    )


def _make_spec() -> PipelineModuleSpec:
    return PipelineModuleSpec(
        name="stage2_trie_ce",
        enabled=True,
        weight=1.0,
        channels=("B",),
        application={"preset": "rollout_trie_hard_ce"},
        config={
            "support_weight": 1.0,
            "balance_weight": 1.0,
            "struct_weight": 1.0,
            "desc_weight": 1.0,
            "coord_hard_ce_weight": 1.0,
            "eos_weight": 1.0,
            "normalization": "token_mean",
        },
    )


def _make_targets(
    *token_targets: Stage2TrieTokenTarget,
    candidate_count: int = 1,
    fallback_candidate_count: int = 0,
    weak_positive_fp_count: int = 0,
) -> Stage2TrieTargets:
    branching_factors = [len(target.positive_token_ids) for target in token_targets]

    return Stage2TrieTargets(
        token_targets=tuple(token_targets),
        span_score_records=(),
        summary=Stage2TrieSummary(
            candidate_count=candidate_count,
            fallback_candidate_count=fallback_candidate_count,
            fallback_loss_weight_sum=float(fallback_candidate_count),
            weak_positive_fp_count=weak_positive_fp_count,
            target_positions=len(token_targets),
            branch_points=sum(1 for factor in branching_factors if factor > 1),
            max_branching_factor=max(branching_factors, default=0),
        ),
    )


def test_stage2_trie_ce_catalog_entry_exists_with_expected_contract() -> None:
    definition = OBJECTIVE_MODULE_CATALOG["stage2_trie_ce"]

    assert definition.family == "text"
    assert definition.semantic_role == "stage2_trie_ce"
    assert definition.emission_group == "text"
    assert definition.config_keys == frozenset(
        {
            "support_weight",
            "balance_weight",
            "struct_weight",
            "desc_weight",
            "coord_hard_ce_weight",
            "eos_weight",
            "normalization",
        }
    )
    assert definition.application_presets == frozenset({"rollout_trie_hard_ce"})
    assert tuple(
        (atom.atom_name, atom.state_key) for atom in definition.projected_atoms
    ) == (("trie_ce", "stage2_trie_ce_contrib"),)
    assert Stage2TrieCEConfig().normalization == "token_mean"


def test_stage2_trie_ce_config_rejects_semantic_image_bucket_balanced_v0() -> None:
    with pytest.raises(ValueError, match=r"pure hard CE v0.*token_mean"):
        build_stage2_trie_ce_config(
            {
                "support_weight": 1.0,
                "balance_weight": 1.0,
                "struct_weight": 1.0,
                "desc_weight": 1.0,
                "coord_hard_ce_weight": 1.0,
                "eos_weight": 1.0,
                "normalization": "semantic_image_bucket_balanced",
            }
        )


def test_stage2_trie_ce_config_accepts_semantic_token_weights() -> None:
    config = build_stage2_trie_ce_config(
        {
            "support_weight": 1.0,
            "balance_weight": 1.0,
            "struct_weight": 1.5,
            "desc_weight": 0.75,
            "coord_hard_ce_weight": 2.0,
            "eos_weight": 3.0,
            "normalization": "token_mean",
        }
    )

    assert config.struct_weight == pytest.approx(1.5)
    assert config.desc_weight == pytest.approx(0.75)
    assert config.coord_hard_ce_weight == pytest.approx(2.0)
    assert config.eos_weight == pytest.approx(3.0)


def test_stage2_trie_ce_module_returns_zero_shell_without_sidecars() -> None:
    context = _make_context()
    spec = _make_spec()

    out = run_stage2_trie_ce_module(context=context, spec=spec)

    assert out.loss.dtype == torch.float32
    assert float(out.loss.detach().cpu().item()) == pytest.approx(0.0)
    assert out.metrics["stage2_trie/target_positions"] == pytest.approx(0.0)
    assert "stage2_trie_ce" in out.state
    assert "stage2_trie_ce_contrib" in out.state
    assert torch.equal(out.state["stage2_trie_ce"], out.loss)
    assert torch.equal(out.state["stage2_trie_ce_contrib"], out.loss)


def test_stage2_trie_ce_multi_positive_logsum_uses_float32_math() -> None:
    logits = torch.full((1, 4, 8), -20.0, dtype=torch.bfloat16)
    logits[0, 1, 2] = 30.0
    logits[0, 1, 5] = 30.0
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(2, 5),
            source_weights=(1.0, 1.0),
            semantic_role="text",
        )
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )

    out = run_stage2_trie_ce_module(context=context, spec=_make_spec())

    assert out.loss.dtype == torch.float32
    assert float(out.loss.detach().cpu().item()) == pytest.approx(0.0, abs=1.0e-4)


def test_stage2_trie_ce_any_active_positive_child_is_acceptable() -> None:
    logits = torch.full((1, 4, 8), -20.0, dtype=torch.float32)
    logits[0, 1, 5] = 30.0
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(2, 5),
            source_weights=(1.0, 1.0),
            semantic_role="text",
        )
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )

    out = run_stage2_trie_ce_module(context=context, spec=_make_spec())

    assert float(out.loss.detach().cpu().item()) == pytest.approx(0.0, abs=1.0e-4)


@pytest.mark.parametrize("predicted_token_id", [2, 5])
def test_stage2_trie_ce_equal_weight_positives_remain_hard_union(
    predicted_token_id: int,
) -> None:
    logits = torch.full((1, 4, 8), -20.0, dtype=torch.float32)
    logits[0, 1, int(predicted_token_id)] = 30.0
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(2, 5),
            source_weights=(1.0, 1.0),
            semantic_role="text",
        )
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )

    out = run_stage2_trie_ce_module(context=context, spec=_make_spec())

    assert float(out.loss.detach().cpu().item()) == pytest.approx(0.0, abs=1.0e-4)


def test_stage2_trie_ce_mixed_source_weights_do_not_make_weak_branch_free() -> None:
    logits = torch.full((1, 4, 8), -20.0, dtype=torch.float32)
    logits[0, 1, 5] = 30.0
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(2, 5),
            source_weights=(1.0, 0.05),
            semantic_role="text",
        )
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )

    out = run_stage2_trie_ce_module(context=context, spec=_make_spec())

    log_probs = torch.log_softmax(logits.float()[0, 1], dim=-1)
    expected = -log_probs[2] + 0.05 * -log_probs[5]

    assert float(out.loss.detach().cpu().item()) == pytest.approx(
        float(expected.detach().cpu().item())
    )


def test_stage2_trie_ce_hard_singleton_matches_log_softmax() -> None:
    logits = torch.tensor(
        [[[0.0, 0.5, -0.25, 1.0], [1.5, -0.5, 2.25, 0.75], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(2,),
            source_weights=(1.0,),
            semantic_role="text",
        )
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 2,
            }
        ],
    )
    expected = -torch.log_softmax(logits.float()[0, 1], dim=-1)[2]

    out = run_stage2_trie_ce_module(context=context, spec=_make_spec())

    assert float(out.loss.detach().cpu().item()) == pytest.approx(
        float(expected.detach().cpu().item())
    )


def test_stage2_trie_ce_eos_weight_changes_mixed_role_loss() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [2.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=1,
            positive_token_ids=(0,),
            source_weights=(1.0,),
            semantic_role="text",
        ),
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(1,),
            source_weights=(1.0,),
            semantic_role="eos",
        ),
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 2,
            }
        ],
    )
    spec = _make_spec()
    spec.config["eos_weight"] = 3.0

    out = run_stage2_trie_ce_module(context=context, spec=spec)

    log_probs_0 = torch.log_softmax(logits.float()[0, 0], dim=-1)
    log_probs_1 = torch.log_softmax(logits.float()[0, 1], dim=-1)
    text_ce = -log_probs_0[0]
    eos_ce = -log_probs_1[1]
    expected = (text_ce + 3.0 * eos_ce) / 4.0
    assert float(out.loss.detach().cpu().item()) == pytest.approx(
        float(expected.detach().cpu().item())
    )


def test_stage2_trie_ce_emits_role_target_count_metrics() -> None:
    logits = torch.zeros((1, 7, 8), dtype=torch.float32)
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=1,
            positive_token_ids=(1,),
            source_weights=(1.0,),
            semantic_role="text",
        ),
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(2,),
            source_weights=(1.0,),
            semantic_role="struct",
        ),
        Stage2TrieTokenTarget(
            position=3,
            positive_token_ids=(3,),
            source_weights=(1.0,),
            semantic_role="desc",
        ),
        Stage2TrieTokenTarget(
            position=4,
            positive_token_ids=(4,),
            source_weights=(1.0,),
            semantic_role="coord",
        ),
        Stage2TrieTokenTarget(
            position=5,
            positive_token_ids=(5,),
            source_weights=(1.0,),
            semantic_role="eos",
        ),
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 5,
            }
        ],
    )

    out = run_stage2_trie_ce_module(context=context, spec=_make_spec())

    assert out.metrics["stage2_trie/role_text_targets"] == pytest.approx(1.0)
    assert out.metrics["stage2_trie/role_struct_targets"] == pytest.approx(1.0)
    assert out.metrics["stage2_trie/role_desc_targets"] == pytest.approx(1.0)
    assert out.metrics["stage2_trie/role_coord_targets"] == pytest.approx(1.0)
    assert out.metrics["stage2_trie/role_eos_targets"] == pytest.approx(1.0)


def test_stage2_trie_ce_source_weight_scales_numerator_not_denominator() -> None:
    logits = torch.tensor(
        [
            [
                [0.25, -0.75, 1.25],
                [-0.5, 0.5, 1.5],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=1,
            positive_token_ids=(2,),
            source_weights=(1.0,),
            semantic_role="text",
        ),
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(2,),
            source_weights=(0.25,),
            semantic_role="text",
        ),
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 2,
            }
        ],
    )
    ce1 = -torch.log_softmax(logits.float()[0, 0], dim=-1)[2]
    ce2 = -torch.log_softmax(logits.float()[0, 1], dim=-1)[2]
    expected = (ce1 + 0.25 * ce2) / 2.0

    out = run_stage2_trie_ce_module(context=context, spec=_make_spec())

    assert float(out.loss.detach().cpu().item()) == pytest.approx(
        float(expected.detach().cpu().item())
    )


def test_stage2_trie_ce_ignores_non_channel_b_and_skip_loss_metadata() -> None:
    logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(2,),
            source_weights=(1.0,),
            semantic_role="text",
        )
    )

    context_a = _make_context_with_logits(
        logits,
        channel="A",
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )
    out_a = run_stage2_trie_ce_module(context=context_a, spec=_make_spec())

    context_skipped = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_skip_loss": True,
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )
    out_skipped = run_stage2_trie_ce_module(context=context_skipped, spec=_make_spec())

    assert float(out_a.loss.detach().cpu().item()) == pytest.approx(0.0)
    assert out_a.metrics["stage2_trie/target_positions"] == pytest.approx(0.0)
    assert float(out_skipped.loss.detach().cpu().item()) == pytest.approx(0.0)
    assert out_skipped.metrics["stage2_trie/target_positions"] == pytest.approx(0.0)


def test_stage2_trie_ce_raises_clear_error_for_position_outside_logits() -> None:
    logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=5,
            positive_token_ids=(2,),
            source_weights=(1.0,),
            semantic_role="text",
        )
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )

    with pytest.raises(ValueError, match="outside logits sequence length"):
        run_stage2_trie_ce_module(context=context, spec=_make_spec())


def test_stage2_trie_ce_projects_packed_segment_local_positions_to_row_logits() -> None:
    logits = torch.full((1, 6, 8), -10.0, dtype=torch.float32)
    logits[0, 4, 6] = 10.0
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(6,),
            source_weights=(1.0,),
            semantic_role="text",
        )
    )
    context = _make_context_with_logits(
        logits,
        input_ids=torch.arange(6, dtype=torch.long).unsqueeze(0),
        meta=[
            {
                "encoded_len": 3,
                "stage2_channel": "B",
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 2,
            },
            {
                "encoded_len": 3,
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 2,
            },
        ],
    )

    out = run_stage2_trie_ce_module(context=context, spec=_make_spec())

    assert float(out.loss.detach().cpu().item()) == pytest.approx(0.0, abs=1.0e-4)
    assert out.metrics["stage2_trie/target_positions"] == pytest.approx(1.0)


def test_stage2_trie_ce_rejects_packed_target_at_segment_start() -> None:
    logits = torch.zeros((1, 6, 8), dtype=torch.float32)
    targets = _make_targets(
        types.SimpleNamespace(
            position=0,
            positive_token_ids=(6,),
            source_weights=(1.0,),
            semantic_role="text",
        )
    )
    context = _make_context_with_logits(
        logits,
        input_ids=torch.arange(6, dtype=torch.long).unsqueeze(0),
        meta=[
            {
                "encoded_len": 3,
                "stage2_channel": "B",
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 2,
            },
            {
                "encoded_len": 3,
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 2,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"token_targets\[0\].position.*int > 0",
    ):
        run_stage2_trie_ce_module(context=context, spec=_make_spec())


def test_stage2_trie_ce_rejects_malformed_sidecar_shape() -> None:
    logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    malformed_targets = object()
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": malformed_targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )

    with pytest.raises(TypeError, match="stage2_trie_targets.*token_targets"):
        run_stage2_trie_ce_module(context=context, spec=_make_spec())


@pytest.mark.parametrize(
    ("target_kwargs", "error_type", "match"),
    [
        (
            {
                "position": "2",
                "positive_token_ids": (2,),
                "source_weights": (1.0,),
                "semantic_role": "text",
            },
            TypeError,
            r"token_targets\[0\].position.*expected int",
        ),
        (
            {
                "position": 2,
                "positive_token_ids": (True,),
                "source_weights": (1.0,),
                "semantic_role": "text",
            },
            TypeError,
            r"positive_token_ids\[0\].*expected int",
        ),
        (
            {
                "position": 2,
                "positive_token_ids": (2,),
                "source_weights": (float("nan"),),
                "semantic_role": "text",
            },
            ValueError,
            r"source_weights\[0\].*finite",
        ),
        (
            {
                "position": 2,
                "positive_token_ids": (2,),
                "source_weights": (1.0,),
                "semantic_role": object(),
            },
            TypeError,
            r"semantic_role.*expected str",
        ),
    ],
)
def test_stage2_trie_ce_rejects_malformed_nested_sidecar_fields(
    target_kwargs: dict[str, object],
    error_type: type[Exception],
    match: str,
) -> None:
    logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    malformed_targets = types.SimpleNamespace(
        token_targets=(types.SimpleNamespace(**target_kwargs),),
        summary=types.SimpleNamespace(
            candidate_count=1,
            fallback_candidate_count=0,
            weak_positive_fp_count=0,
            target_positions=1,
            branch_points=0,
            max_branching_factor=1,
        ),
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": malformed_targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )

    with pytest.raises(error_type, match=match):
        run_stage2_trie_ce_module(context=context, spec=_make_spec())


def test_stage2_trie_ce_rejects_malformed_summary_fields() -> None:
    logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    targets = types.SimpleNamespace(
        token_targets=(
            types.SimpleNamespace(
                position=2,
                positive_token_ids=(2,),
                source_weights=(1.0,),
                semantic_role="text",
            ),
        ),
        summary=types.SimpleNamespace(
            candidate_count=True,
            fallback_candidate_count=0,
            weak_positive_fp_count=0,
            target_positions=1,
            branch_points=0,
            max_branching_factor=1,
        ),
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )

    with pytest.raises(TypeError, match=r"summary.candidate_count.*expected int"):
        run_stage2_trie_ce_module(context=context, spec=_make_spec())


def test_stage2_trie_ce_emits_task4_metrics_and_pipeline_loss_metric() -> None:
    logits = torch.full((1, 4, 8), -5.0, dtype=torch.float32)
    logits[0, 1, 2] = 5.0
    targets = _make_targets(
        Stage2TrieTokenTarget(
            position=2,
            positive_token_ids=(2, 5),
            source_weights=(2.0, 1.0),
            semantic_role="struct",
        ),
        candidate_count=4,
        fallback_candidate_count=2,
        weak_positive_fp_count=3,
    )
    context = _make_context_with_logits(
        logits,
        meta=[
            {
                "stage2_channel": "B",
                "stage2_trie_targets": targets,
                "prompt_len": 1,
                "prefix_len": 1,
                "train_len": 3,
            }
        ],
    )

    module_out = run_stage2_trie_ce_module(context=context, spec=_make_spec())
    pipeline_out = run_teacher_forcing_pipeline(
        context=context,
        objective_specs=[
            {
                "name": "stage2_trie_ce",
                "enabled": True,
                "weight": 1.0,
                "channels": ("B",),
                "application": {"preset": "rollout_trie_hard_ce"},
                "config": _make_spec().config,
            }
        ],
        diagnostics_specs=[],
    )

    assert module_out.metrics["stage2_trie/target_positions"] == pytest.approx(1.0)
    assert module_out.metrics["stage2_trie/branch_points"] == pytest.approx(1.0)
    assert module_out.metrics["stage2_trie/max_branching_factor"] == pytest.approx(2.0)
    assert module_out.metrics["stage2_trie/candidate_count_mean"] == pytest.approx(4.0)
    assert module_out.metrics["stage2_trie/fallback_candidate_share"] == pytest.approx(
        0.5
    )
    assert module_out.metrics["stage2_trie/fallback_loss_share"] == pytest.approx(0.5)
    assert module_out.metrics["stage2_trie/fallback_dominance_warning"] == pytest.approx(1.0)
    assert module_out.metrics["stage2_trie/fp_policy_weak_positive_count"] == pytest.approx(
        3.0
    )
    assert "loss/B/stage2_trie_ce" in module_out.metrics
    assert pipeline_out.metrics["loss/stage2_trie_ce"] == pytest.approx(
        module_out.metrics["loss/B/stage2_trie_ce"]
    )


def test_teacher_forcing_pipeline_executes_stage2_trie_ce_spec() -> None:
    context = _make_context()
    spec = _make_spec()

    out = run_teacher_forcing_pipeline(
        context=context,
        objective_specs=[
            {
                "name": spec.name,
                "enabled": spec.enabled,
                "weight": spec.weight,
                "channels": spec.channels,
                "application": spec.application,
                "config": spec.config,
            }
        ],
        diagnostics_specs=[],
    )

    assert float(out.total_loss.detach().cpu().item()) == pytest.approx(0.0)
    assert float(out.module_losses["stage2_trie_ce"].detach().cpu().item()) == pytest.approx(
        0.0
    )
    assert out.metrics["stage2_trie/target_positions"] == pytest.approx(0.0)
    assert out.metrics["loss/stage2_trie_ce"] == pytest.approx(0.0)
    assert "stage2_trie_ce_contrib" in out.state
