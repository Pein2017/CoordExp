from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from probes.human13.output_qp import (
    MARGIN,
    STAGES,
    HoldError,
    SelectedOutputRowsHook,
    aggregate_results,
    canonical_route_text,
    immutable_json,
    recover_fixed_max,
    select_trainable_rows,
    solve_minimum_frobenius,
)


def test_stage_registry_matches_preregistered_nested_panels() -> None:
    assert {
        name: (
            stage.image_ids,
            stage.owner_count,
            stage.decision_state_count,
            stage.unique_target_token_count,
            dict(stage.route_token_counts),
        )
        for name, stage in STAGES.items()
    } == {
        "N2": ((6040, 16228), 65, 592, 237, {6040: 136, 16228: 456}),
        "N4": (
            (4134, 6040, 13923, 16228),
            123,
            1147,
            401,
            {4134: 345, 6040: 136, 13923: 210, 16228: 456},
        ),
        "N13": (
            (1584, 2299, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228),
            392,
            3637,
            832,
            {
                1584: 172,
                2299: 415,
                2685: 282,
                4134: 345,
                5001: 212,
                6040: 136,
                7511: 400,
                10707: 184,
                13348: 138,
                13923: 210,
                14038: 438,
                14439: 249,
                16228: 456,
            },
        ),
    }


def test_canonical_route_assembly_span_count_and_hash() -> None:
    row = {
        "image_id": 7,
        "objects": [
            {
                "desc": "cat",
                "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
            },
            {
                "desc": "dog",
                "bbox_2d": ["<|coord_5|>", "<|coord_6|>", "<|coord_7|>", "<|coord_8|>"],
            },
        ],
    }
    text, rows = canonical_route_text(row)
    assert len(rows) == 2
    assert "".join(rows) + "<|im_end|>" == text
    assert rows[0] == (
        "<|object_ref_start|>cat<|object_ref_end|><|box_start|>"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
    )
    assert hashlib.sha256(text.encode()).hexdigest() == "c3da1dd13d31c0748ead6fc9af84a3633eab83760805f90171ed03f807a08a5f"


def test_fixed_max_recovery_and_selected_rows_exclude_target() -> None:
    target_ids = np.array([10, 11])
    top_ids = np.array([[11, 99, 10], [11, 10, 99]])
    top_logits = np.array([[1.0, 0.5, 0.0], [2.0, 1.0, 0.5]])
    assert select_trainable_rows(target_ids, top_ids, top_logits) == (10,)
    # The target may be outside top-K; its separately captured exact logit owns
    # selection while top-K still owns the full-vocabulary competitor maximum.
    assert select_trainable_rows(
        [10], [[11, 99, 98]], [[1.0, 0.5, 0.25]], target_logits=[0.0]
    ) == (10,)
    # 11 is selected and 10 is the target, so 99 is the exact fixed maximum.
    assert recover_fixed_max([11, 10, 99], [2.0, 1.0, 0.5], {11}, target_id=10) == (99, 0.5)


def test_small_active_set_qp_has_exhaustive_fp32_certificate() -> None:
    hidden = np.eye(2, dtype=np.float64)
    targets = np.array([10, 11], dtype=np.int64)
    route_ids = np.array([10, 11], dtype=np.int64)
    base_route = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    top_ids = np.array([[11, 99, 10], [10, 99, 11]], dtype=np.int64)
    top_logits = np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32)

    # Sensitivity/RED witness: the zero solution violates both target margins.
    assert base_route[0, 0] - base_route[0, 1] < MARGIN
    assert base_route[1, 1] - base_route[1, 0] < MARGIN

    solved = solve_minimum_frobenius(
        hidden_states=hidden,
        target_ids=targets,
        route_token_ids=route_ids,
        base_route_logits=base_route,
        top_ids=top_ids,
        top_logits=top_logits,
    )
    selected = solved["selected_token_ids"].tolist()
    assert selected == [10, 11]
    rows = solved["residual_rows"].astype(np.float32)
    delta = hidden.astype(np.float32) @ rows.T
    index = {token: i for i, token in enumerate(selected)}
    for pos, target in enumerate(targets.tolist()):
        competitor = 11 if target == 10 else 10
        target_logit = base_route[pos, route_ids.tolist().index(target)] + delta[pos, index[target]]
        competitor_logit = (
            base_route[pos, route_ids.tolist().index(competitor)] + delta[pos, index[competitor]]
        )
        assert target_logit - competitor_logit >= MARGIN - 2e-5
        assert target_logit - 0.5 >= MARGIN - 2e-5
    assert solved["receipt"]["max_fp64_violation"] <= 2e-5
    assert solved["receipt"]["minimum_fp32_hook_margin"] >= MARGIN - 2e-5
    assert solved["receipt"]["outer_solve_count"] >= 1


def test_solver_continues_once_from_exact_maxiter_boundary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scipy.optimize

    real_minimize = scipy.optimize.minimize
    calls = []
    successful_nit = []
    first_terminal = np.array([0.25, 0.5], dtype=np.float64)

    def maxiter_then_succeed(fun, x0, **kwargs):
        calls.append((fun, np.asarray(x0).copy(), kwargs))
        if len(calls) == 1:
            value, _ = fun(first_terminal)
            return SimpleNamespace(
                x=first_terminal.copy(),
                fun=value,
                success=False,
                status=1,
                message="TOTAL NO. OF ITERATIONS REACHED LIMIT",
                nit=4000,
                nfev=4001,
                njev=4001,
            )
        result = real_minimize(fun, x0, **kwargs)
        successful_nit.append(int(result.nit))
        return result

    monkeypatch.setattr(scipy.optimize, "minimize", maxiter_then_succeed)
    solved = solve_minimum_frobenius(
        hidden_states=np.eye(2, dtype=np.float64),
        target_ids=np.array([10, 11], dtype=np.int64),
        route_token_ids=np.array([10, 11], dtype=np.int64),
        base_route_logits=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        top_ids=np.array([[11, 99, 10], [10, 99, 11]], dtype=np.int64),
        top_logits=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )

    assert len(calls) == 2
    assert calls[1][0] is calls[0][0]
    np.testing.assert_array_equal(calls[1][1], first_terminal)
    assert calls[1][2] == calls[0][2] == {
        "jac": True,
        "method": "L-BFGS-B",
        "bounds": [(0.0, None), (0.0, None)],
        "options": {"ftol": 1e-14, "gtol": 1e-10, "maxiter": 4000, "maxls": 50},
    }
    assert solved["receipt"]["outer_solve_count"] == 1
    assert solved["receipt"]["optimizer_iteration_count"] == 4000 + successful_nit[0]
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["diagnostic_segment_index"] for item in progress] == [1, 2]
    assert [item["diagnostic_segment_limit"] for item in progress] == [3, 3]
    assert [item["diagnostic_outer_solve_index"] for item in progress] == [1, 1]


def test_solver_polishes_successful_noncertificate_without_new_cuts_three_times(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scipy.optimize

    calls = []
    terminals = [
        np.array([0.25, 0.50], dtype=np.float64),
        np.array([0.30, 0.50], dtype=np.float64),
        np.array([0.35, 0.50], dtype=np.float64),
    ]

    def successful_noncertificate(fun, x0, **kwargs):
        terminal = terminals[len(calls)]
        calls.append((np.asarray(x0).copy(), kwargs))
        value, _ = fun(terminal)
        return SimpleNamespace(
            x=terminal.copy(),
            fun=value,
            success=True,
            status=0,
            message="CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH",
            nit=7,
            nfev=8,
            njev=8,
        )

    monkeypatch.setattr(scipy.optimize, "minimize", successful_noncertificate)
    with pytest.raises(
        HoldError,
        match="violated constraints remain but no cutting plane was added",
    ):
        solve_minimum_frobenius(
            hidden_states=np.eye(2, dtype=np.float64),
            target_ids=np.array([10, 11], dtype=np.int64),
            route_token_ids=np.array([10, 11], dtype=np.int64),
            base_route_logits=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            top_ids=np.array([[11, 99, 10], [10, 99, 11]], dtype=np.int64),
            top_logits=np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32),
        )

    assert len(calls) == 3
    np.testing.assert_array_equal(calls[1][0], terminals[0])
    np.testing.assert_array_equal(calls[2][0], terminals[1])
    assert [call[1]["options"]["ftol"] for call in calls] == [1e-14, 0.0, 0.0]
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["diagnostic_polish"] for item in progress] == [False, True, True]
    assert [item["diagnostic_optimizer_ftol"] for item in progress] == [1e-14, 0.0, 0.0]
    assert [item["diagnostic_segment_index"] for item in progress] == [1, 2, 3]
    assert all(
        item["diagnostic_full_registered_worst_primal_violation"] > 2e-5
        for item in progress
    )


def test_solver_polish_can_reach_fp32_certified_solution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scipy.optimize

    real_minimize = scipy.optimize.minimize
    calls = []
    first_terminal = np.array([0.25, 0.50], dtype=np.float64)

    def noncertificate_then_polish(fun, x0, **kwargs):
        calls.append((np.asarray(x0).copy(), kwargs))
        if len(calls) == 1:
            value, _ = fun(first_terminal)
            return SimpleNamespace(
                x=first_terminal.copy(),
                fun=value,
                success=True,
                status=0,
                message="CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH",
                nit=7,
                nfev=8,
                njev=8,
            )
        return real_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", noncertificate_then_polish)
    solved = solve_minimum_frobenius(
        hidden_states=np.eye(2, dtype=np.float64),
        target_ids=np.array([10, 11], dtype=np.int64),
        route_token_ids=np.array([10, 11], dtype=np.int64),
        base_route_logits=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        top_ids=np.array([[11, 99, 10], [10, 99, 11]], dtype=np.int64),
        top_logits=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1][0], first_terminal)
    assert [call[1]["options"]["ftol"] for call in calls] == [1e-14, 0.0]
    assert solved["receipt"]["outer_solve_count"] == 1
    assert solved["receipt"]["max_fp64_violation"] <= 2e-5
    assert solved["receipt"]["minimum_fp32_hook_margin"] >= MARGIN - 2e-5
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["diagnostic_polish"] for item in progress] == [False, True]
    assert progress[0]["diagnostic_full_registered_worst_primal_violation"] > 2e-5
    assert progress[1]["diagnostic_full_registered_worst_primal_violation"] <= 2e-5


def test_solver_failure_prints_terminal_diagnostics_and_still_holds_after_three_segments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scipy.optimize

    calls = []

    def fail_at_maxiter(fun, x0, **kwargs):
        calls.append(kwargs)
        value, _ = fun(x0)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64),
            fun=value,
            success=False,
            status=1,
            message="TOTAL NO. OF ITERATIONS REACHED LIMIT",
            nit=4000,
            nfev=4001,
            njev=4001,
        )

    monkeypatch.setattr(scipy.optimize, "minimize", fail_at_maxiter)
    with pytest.raises(
        HoldError,
        match="QP dual optimizer failed: TOTAL NO. OF ITERATIONS REACHED LIMIT",
    ):
        solve_minimum_frobenius(
            hidden_states=np.eye(2, dtype=np.float64),
            target_ids=np.array([10, 11], dtype=np.int64),
            route_token_ids=np.array([10, 11], dtype=np.int64),
            base_route_logits=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            top_ids=np.array([[11, 99, 10], [10, 99, 11]], dtype=np.int64),
            top_logits=np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32),
        )

    assert calls == [
        {
            "jac": True,
            "method": "L-BFGS-B",
            "bounds": [(0.0, None), (0.0, None)],
            "options": {"ftol": 1e-14, "gtol": 1e-10, "maxiter": 4000, "maxls": 50},
        }
    ] * 3
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    progress = json.loads(lines[-1])
    assert progress["schema"] == "human13_output_qp_solver_progress.v1"
    assert progress["diagnostic_outer_solve_index"] == 1
    assert progress["diagnostic_segment_index"] == 3
    assert progress["diagnostic_segment_limit"] == 3
    assert progress["diagnostic_active_constraint_count"] == 2
    assert (
        progress["diagnostic_active_set_sha256"]
        == "145f130c386db5b90dc94fa4162164c4c2a2b1aa279c53172da4fb78a79e3c49"
    )
    assert progress["diagnostic_result_success"] is False
    assert progress["diagnostic_result_status"] == 1
    assert progress["diagnostic_result_message"] == "TOTAL NO. OF ITERATIONS REACHED LIMIT"
    assert progress["diagnostic_result_nit"] == 4000
    assert progress["diagnostic_result_nfev"] == 4001
    assert progress["diagnostic_result_njev"] == 4001
    assert progress["diagnostic_result_fun"] == pytest.approx(0.0)
    assert progress["diagnostic_lambda_l2_norm"] == pytest.approx(0.0)
    assert progress["diagnostic_lambda_max"] == pytest.approx(0.0)
    assert progress["diagnostic_projected_gradient_kkt_inf_norm"] > 0.0
    assert progress["diagnostic_full_registered_worst_primal_violation"] > 0.0
    assert progress["diagnostic_primal_half_frobenius_squared"] == pytest.approx(0.0)
    assert progress["diagnostic_dual_objective"] == pytest.approx(0.0)
    assert progress["diagnostic_candidate_gap"] == pytest.approx(0.0)
    assert progress["diagnostic_active_complementarity_max"] == pytest.approx(0.0)
    assert all("fp32" not in key for key in progress)


def test_solver_never_retries_non_maxiter_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import scipy.optimize

    calls = 0

    def fail_immediately(fun, x0, **kwargs):
        nonlocal calls
        calls += 1
        value, _ = fun(x0)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64),
            fun=value,
            success=False,
            status=2,
            message="ABNORMAL_TERMINATION_IN_LNSRCH",
            nit=7,
            nfev=8,
        )

    monkeypatch.setattr(scipy.optimize, "minimize", fail_immediately)
    with pytest.raises(HoldError, match="QP dual optimizer failed: ABNORMAL_TERMINATION"):
        solve_minimum_frobenius(
            hidden_states=np.eye(2, dtype=np.float64),
            target_ids=np.array([10, 11], dtype=np.int64),
            route_token_ids=np.array([10, 11], dtype=np.int64),
            base_route_logits=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            top_ids=np.array([[11, 99, 10], [10, 99, 11]], dtype=np.int64),
            top_logits=np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32),
        )
    assert calls == 1


def test_output_hook_changes_only_selected_logits_and_zero_is_exact() -> None:
    head = nn.Linear(3, 5, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10)
    hidden = torch.tensor([[[1.0, 2.0, -1.0]]])
    baseline = head(hidden)

    with SelectedOutputRowsHook(head, [], torch.empty((0, 3), dtype=torch.float64)):
        zero = head(hidden)
    assert torch.equal(zero, baseline)

    rows = torch.tensor([[0.5, -0.25, 1.0], [-1.0, 0.0, 0.25]], dtype=torch.float64)
    with SelectedOutputRowsHook(head, [1, 4], rows):
        changed = head(hidden)
    assert torch.equal(changed[..., [0, 2, 3]], baseline[..., [0, 2, 3]])
    expected_delta = hidden @ rows.to(torch.float32).T
    assert torch.equal(changed[..., [1, 4]], baseline[..., [1, 4]] + expected_delta)


def test_immutable_json_never_overwrites(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    immutable_json(path, {"writer": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        immutable_json(path, {"writer": 2})
    assert json.loads(path.read_text()) == {"writer": 1}


def test_aggregate_rejects_rp110_candidates(tmp_path) -> None:
    candidates = []
    restored = []
    for image_id, owners in ((6040, 15), (16228, 50)):
        identity = {
            "generated_token_ids_sha256": f"tokens-{image_id}",
            "parser_text_sha256": f"parser-{image_id}",
            "stop_reason": "eos",
        }
        candidate = tmp_path / f"candidate-{image_id}.json"
        candidate.write_text(
            json.dumps(
                {
                    "stage": "N2",
                    "image_ids": list(STAGES["N2"].image_ids),
                    "image_id": image_id,
                    "mode": "candidate",
                    "repetition_penalty": 1.1,
                    "payload_sha256": "a" * 64,
                    "pre_source_rp1_0": identity,
                    "evaluation": {
                        "matched_owner_count": {"50": owners, "60": owners, "80": owners},
                        "duplicate_count": 0,
                        "unmatched_prediction_count": 0,
                        "malformed_count": 0,
                        "cap_debt": False,
                        "natural_eos": True,
                        "exact_route": True,
                    },
                }
            )
        )
        source = tmp_path / f"source-{image_id}.json"
        source.write_text(
            json.dumps(
                {
                    "stage": "N2",
                    "image_ids": list(STAGES["N2"].image_ids),
                    "image_id": image_id,
                    "mode": "source",
                    "repetition_penalty": 1.0,
                    "evaluation": identity,
                }
            )
        )
        candidates.append(candidate)
        restored.append(source)
    with pytest.raises(HoldError, match="RP1.0 candidate"):
        aggregate_results(results=candidates, post_source=restored)


def test_aggregate_enforces_selected_stage_order(tmp_path) -> None:
    wrong_order = (6040, 4134, 13923, 16228)
    candidates = []
    post_source = []
    for kind, paths in (("candidate", candidates), ("source", post_source)):
        for index, image_id in enumerate(wrong_order):
            path = tmp_path / f"{kind}-{index}.json"
            path.write_text(json.dumps({"image_id": image_id}))
            paths.append(path)
    with pytest.raises(HoldError, match="frozen N4 order"):
        aggregate_results(results=candidates, post_source=post_source, stage="N4")
