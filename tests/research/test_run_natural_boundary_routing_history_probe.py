from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.research.run_natural_boundary_routing_history_probe import (
    MAX_ROW_TOKENS,
    NativeRowContract,
    TechnicalInvalid,
    build_event_context,
    build_residual_request,
    persist_verbatim_failure_log,
    release_natural_event,
    run_natural_residual_flow,
    score_natural_row_segments,
    validate_admission_receipt,
)


OPEN = 1
REF_END = 2
BOX_START = 3
BOX_END = 9
COORD = 10
STOP = 0


def _contract() -> NativeRowContract:
    return NativeRowContract.closed(
        opener_token_id=OPEN,
        object_ref_end_token_id=REF_END,
        box_start_token_id=BOX_START,
        box_end_token_id=BOX_END,
        coordinate_token_start_id=COORD,
        coordinate_bin_count=100,
        stop_token_id=STOP,
    )


def _row(description: int = 20) -> tuple[int, ...]:
    return (OPEN, description, REF_END, BOX_START, COORD, COORD + 1, COORD + 2, COORD + 3, BOX_END)


class PrefixScriptModel:
    """Tiny deterministic model keyed by the exact generated prefix."""

    def __init__(self, script: dict[tuple[int, ...], int], *, vocab: int = 64) -> None:
        self.script = dict(script)
        self.vocab = vocab
        self.calls: list[dict[str, object]] = []

    def __call__(self, *, input_ids: torch.Tensor, use_cache: bool = False, **kwargs: object) -> SimpleNamespace:
        values = tuple(int(value) for value in input_ids[0].tolist())
        self.calls.append({"input_ids": values, "use_cache": use_cache, **kwargs})
        token = self.script.get(values, STOP)
        logits = torch.full((1, input_ids.shape[1], self.vocab), -50.0)
        logits[:, :, token] = 20.0
        return SimpleNamespace(logits=logits)


class StrictPrefixScriptModel(PrefixScriptModel):
    def __call__(self, *, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        return super().__call__(input_ids=input_ids, use_cache=use_cache)


def _single_row_script(prefix: tuple[int, ...] = (50,)) -> dict[tuple[int, ...], int]:
    row = _row()
    script: dict[tuple[int, ...], int] = {}
    current = prefix
    for token in row:
        script[current] = token
        current = current + (token,)
    return script


def _context(
    *,
    prefix: tuple[int, ...] = (50,),
    history: tuple[int, ...] = (),
    max_rows: int = 3,
    max_row_tokens: int = MAX_ROW_TOKENS,
    max_new_tokens: int | None = None,
    latest_history_row: tuple[int, ...] = (),
) -> object:
    return build_event_context(
        event_id="fake-event",
        prompt_token_ids=prefix,
        exact_history_token_ids=history,
        row_contract=_contract(),
        max_rows=max_rows,
        max_row_tokens=max_row_tokens,
        max_new_tokens=max_new_tokens,
        latest_history_row_token_ids=latest_history_row,
    )


def test_natural_release_uses_exact_boundary_and_scores_model_opener() -> None:
    prefix = (50,)
    script = _single_row_script(prefix)
    model = PrefixScriptModel(script)
    result = release_natural_event(model, _context(prefix=prefix, max_rows=1))

    row = result["rows"][0]
    assert row["status"] == "closure"
    assert row["token_ids"] == list(_row())
    assert row["admission_mode"] == "pre_opener_natural"
    assert row["initial_prefix_last_token_id"] == prefix[-1]
    assert row["opener_token_id"] == OPEN
    assert row["opener_injected"] is False
    assert row["first_generated_token_id"] == OPEN
    assert row["opener_generated_by_model"] is True
    assert result["synthetic_opener_injections"] == 0
    assert row["segment_scores"]["segments"]["row_entry"]["token_count"] == 1
    assert row["segment_scores"]["segments"]["full_row"]["token_count"] == len(_row())
    assert row["opener_logits"]["token_id"] == OPEN
    assert row["first_token_logits"]["token_id"] == OPEN
    assert row["terminal_logits"]["token_id"] == BOX_END
    assert all(call["use_cache"] is False for call in model.calls)
    assert [len(call["input_ids"]) for call in model.calls] == list(range(1, len(_row()) + 1))


def test_natural_rows_never_seed_an_opener_between_rows() -> None:
    prefix = (50,)
    first = _row()
    second = _row(description=21)
    script = _single_row_script(prefix)
    current = prefix + first
    for token in second:
        script[current] = token
        current = current + (token,)
    script[current] = STOP
    model = PrefixScriptModel(script)
    result = release_natural_event(model, _context(prefix=prefix, max_rows=3))

    assert [row["token_ids"] for row in result["rows"]] == [list(first), list(second)]
    assert result["rows"][1]["prefix_before_row_token_ids"] == list(prefix + first)
    assert result["rows"][1]["prefix_before_row_token_ids"][-1] == BOX_END
    assert result["rows"][1]["opener_generated_by_model"] is True
    assert result["terminal_reason"] == "native_stop"
    # The first token of row two is the boundary lookahead, not a caller seed.
    assert model.calls[len(first)]["input_ids"] == prefix + first


def test_three_row_cap_stops_after_three_complete_natural_rows() -> None:
    prefix = (50,)
    rows = [_row(description=20 + index) for index in range(3)]
    script: dict[tuple[int, ...], int] = {}
    current = prefix
    for row in rows:
        for token in row:
            script[current] = token
            current = current + (token,)
    model = PrefixScriptModel(script)
    result = release_natural_event(model, _context(prefix=prefix, max_rows=3))
    assert len(result["rows"]) == 3
    assert all(row["status"] == "closure" for row in result["rows"])
    assert result["terminal_reason"] == "closure"
    assert result["row_limit_reached"] is True
    assert len(result["generated_token_ids"]) == sum(len(row) for row in rows)


@pytest.mark.parametrize(
    ("token", "terminal", "row_status"),
    ((STOP, "native_stop", "native_stop"), (7, "invalid", "invalid")),
)
def test_native_stop_and_invalid_first_token_are_distinct(token: int, terminal: str, row_status: str) -> None:
    model = PrefixScriptModel({(50,): token})
    result = release_natural_event(model, _context(prefix=(50,), max_rows=1))
    assert result["terminal_reason"] == terminal
    assert result["rows"][0]["status"] == row_status
    assert result["rows"][0]["opener_generated_by_model"] is False


def test_over_continuation_and_row_budget_are_explicit() -> None:
    prefix = (50,)
    row = _row()
    script = _single_row_script(prefix)
    # Boundary token after a complete row is not an opener: classify it as
    # over-continuation rather than pretending that row admission was invalid.
    script[prefix + row] = 7
    model = PrefixScriptModel(script)
    result = release_natural_event(model, _context(prefix=prefix, max_rows=2))
    assert result["terminal_reason"] == "over_continuation"
    assert result["rows"][0]["status"] == "over_continuation"

    # A one-token cap cannot reach closure and is a budget endpoint.
    budget_model = PrefixScriptModel(_single_row_script(prefix))
    budget_result = release_natural_event(
        budget_model,
        _context(prefix=prefix, max_rows=1, max_row_tokens=3),
    )
    assert budget_result["terminal_reason"] == "max_budget"
    assert budget_result["rows"][0]["status"] == "max_budget"
    assert budget_result["rows"][0]["row_token_count"] == 3


def test_segment_alignment_mismatch_is_technical_invalidity() -> None:
    with pytest.raises(TechnicalInvalid, match="alignment mismatch"):
        score_natural_row_segments(
            _row(),
            [0.0] * (len(_row()) - 1),
            [1] * len(_row()),
            contract=_contract(),
        )
    with pytest.raises(TechnicalInvalid, match="alignment mismatch"):
        score_natural_row_segments(
            _row(),
            [0.0] * len(_row()),
            [1] * (len(_row()) - 1),
            contract=_contract(),
        )


def test_admission_cross_fields_fail_closed() -> None:
    with pytest.raises(TechnicalInvalid, match="disagrees"):
        validate_admission_receipt(
            {
                "admission_mode": "pre_opener_natural",
                "opener_generated_by_model": False,
                "synthetic_opener_injections": 0,
                "opener_injected": False,
            },
            initial_prefix_token_ids=(50,),
            opener_token_id=OPEN,
            generated_token_ids=(OPEN,),
        )
    with pytest.raises(TechnicalInvalid, match="ends with"):
        validate_admission_receipt(
            {
                "admission_mode": "pre_opener_natural",
                "opener_generated_by_model": True,
                "synthetic_opener_injections": 0,
                "opener_injected": False,
            },
            initial_prefix_token_ids=(50, OPEN),
            opener_token_id=OPEN,
            generated_token_ids=(OPEN,),
        )
    assert validate_admission_receipt(
        {
            "admission_mode": "post_opener_seeded",
            "opener_generated_by_model": False,
            "seed_provenance": "caller_supplied_opener",
            "supplied_opener_token_id": OPEN,
        },
        initial_prefix_token_ids=(50, OPEN),
        opener_token_id=OPEN,
        generated_token_ids=(),
    )["admission_mode"] == "post_opener_seeded"


def test_n01_complete_trajectory_parity_and_attention_callback() -> None:
    history = _row(description=19)
    prefix = (50,)
    script = _single_row_script(prefix + history)
    model = PrefixScriptModel(script)
    context = _context(prefix=prefix, history=history, latest_history_row=history, max_rows=1)
    residual_calls: list[str] = []
    attention_calls: list[tuple[int, tuple[int, ...]]] = []

    def residual(_model: object, *, request: object, **_kwargs: object):
        residual_calls.append(request.arm_id)
        return nullcontext()

    def attention(_context: object, *, input_ids: torch.Tensor, step: int, **_kwargs: object):
        attention_calls.append((step, tuple(int(value) for value in input_ids[0].tolist())))
        return {"attention_mask": torch.ones((1, 1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.bool), "receipt": {"kind": "fake"}}

    flow = run_natural_residual_flow(
        model,
        context,
        residual_actuator=residual,
        attention_mask_actuator=attention,
        residual_requests={
            arm: build_residual_request(arm, positions=(1,))
            for arm in ("N01", "N10")
        }
        | {"N20": build_residual_request("N20", positions=(1, 2, 3))},
    )
    assert set(flow["arms"]) == {"N00", "N01", "N10", "N20"}
    assert flow["n01_parity"]["complete_trajectory"] is True
    assert flow["n01_parity"]["per_step_max_abs_delta"] <= 1e-4
    assert flow["n01_parity"]["full_logit_parity"]["comparison_scope"] == "every_full_vocabulary_logit_per_scalar_step"
    assert flow["n01_parity"]["full_logit_parity"]["passed"] is True
    assert flow["n01_parity"]["full_logit_max_abs_delta"] == 0.0
    assert flow["n01_parity"]["passed"] is True
    assert {"N01", "N10", "N20"} <= set(residual_calls)
    assert attention_calls
    assert all(call[0] >= 0 for call in attention_calls)
    assert all(item["use_cache"] is False for item in model.calls)


def test_verbatim_failure_log_helper(tmp_path: Path) -> None:
    stderr = b"line 1\n\xff\n"
    receipt = persist_verbatim_failure_log(tmp_path / "run", stderr)
    path = Path(receipt["path"])
    assert path.read_bytes() == stderr
    assert receipt["size_bytes"] == len(stderr)
    assert receipt["status"] == "persisted"


def test_attention_actuator_callback_enters_natural_runner_without_seeding_opener() -> None:
    # This is the real pure helper callback, not a local stand-in.  A one-token
    # sequence keeps its prebuilt K01 mask shape exact while the fake model
    # emits native STOP at the pre-opener boundary.
    from scripts.research.natural_boundary_attention_actuators import (
        build_k_arm,
        make_natural_runner_callback,
    )

    actuator = build_k_arm("K01", sequence_length=1)
    callback = make_natural_runner_callback(actuator)
    model = PrefixScriptModel({(50,): STOP})
    result = release_natural_event(
        model,
        _context(prefix=(50,), max_rows=1),
        attention_mask_actuator=callback,
    )
    assert result["synthetic_opener_injections"] == 0
    assert result["rows"][0]["status"] == "native_stop"
    assert result["rows"][0]["opener_generated_by_model"] is False
    assert model.calls[0]["attention_mask"] is not None
    receipt = result["scalar_receipts"][0]["attention_mask"]["receipt"]
    assert receipt["arm_id"] == "K01"
    assert receipt["status"] == "ready"


def test_attention_mask_is_never_silently_filtered_from_model_payload() -> None:
    from scripts.research.natural_boundary_attention_actuators import (
        build_k_arm,
        make_natural_runner_callback,
    )

    callback = make_natural_runner_callback(build_k_arm("K01", sequence_length=1))
    with pytest.raises(TechnicalInvalid, match="no field may be filtered"):
        release_natural_event(
            StrictPrefixScriptModel({(50,): STOP}),
            _context(prefix=(50,), max_rows=1),
            attention_mask_actuator=callback,
        )


def test_native_stop_accepts_explicit_token_set() -> None:
    alternate_stop = 63
    contract = NativeRowContract.closed(
        opener_token_id=OPEN,
        object_ref_end_token_id=REF_END,
        box_start_token_id=BOX_START,
        box_end_token_id=BOX_END,
        coordinate_token_start_id=COORD,
        coordinate_bin_count=100,
        stop_token_ids=(STOP, alternate_stop),
    )
    context = build_event_context(
        event_id="multi-stop",
        prompt_token_ids=(50,),
        exact_history_token_ids=(),
        row_contract=contract,
        max_rows=1,
    )
    result = release_natural_event(PrefixScriptModel({(50,): alternate_stop}), context)
    assert result["terminal_reason"] == "native_stop"
    assert result["rows"][0]["first_generated_token_id"] == alternate_stop
    assert result["prefix"]["row_contract"]["stop_token_ids"] == [STOP, alternate_stop]


def test_release_row_can_flow_through_endpoint_finalizer() -> None:
    from scripts.research.finalize_natural_boundary_routing_history_evidence import classify_endpoint

    result = release_natural_event(
        PrefixScriptModel(_single_row_script((50,))),
        _context(prefix=(50,), max_rows=1),
    )
    endpoint = {
        **result["rows"][0],
        "native_parse": {"valid": True, "parse_status": "accepted"},
        "generation_status": "complete",
        "complete_row": True,
        "owner_match": {
            "status": "unique",
            "owner_id": "gt:fake:1",
            "source_specific": True,
            "physical_match": True,
        },
        "owner_bookkeeping": {
            "G": ["gt:fake:1"],
            "K": [],
            "L": [],
            "net": 1,
            "parse": {"unmatched_rows": 0, "duplicate_rows": 0},
        },
    }
    classified = classify_endpoint(endpoint)
    assert classified["mechanically_valid"] is True
    assert classified["source_specific_match"] is True
