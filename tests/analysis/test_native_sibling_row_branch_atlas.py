from __future__ import annotations

import json
from pathlib import Path

from scripts.research.run_native_sibling_branch_atlas import build_atlas


def _row(category: str, bins: tuple[int, int, int, int]) -> tuple[list[int], list[str]]:
    token_ids = [151646, 9000 if category == "cup" else 9001, 151647, 151648]
    token_texts = [
        "<|object_ref_start|>",
        category,
        "<|object_ref_end|>",
        "<|box_start|>",
    ]
    for value in bins:
        token_ids.append(151770 + value)
        token_texts.append(f"<|coord_{value}|>")
    token_ids.append(151649)
    token_texts.append("<|box_end|>")
    return token_ids, token_texts


def _bundle(path: Path, image_id: str, rows: list[tuple[str, tuple[int, int, int, int]]], seed: int) -> None:
    generated: list[int] = []
    trace: list[dict[str, object]] = []
    text: list[str] = []
    for category, bins in rows:
        ids, names = _row(category, bins)
        generated.extend(ids)
        trace.extend({"token_text": name} for name in names)
        text.extend(names)
    generated.append(151645)
    trace.append({"token_text": "<|im_end|>"})
    payload = {
        "attempt_status": "completed",
        "request_id": f"call:{image_id}:{seed}",
        "stop_reason": "im_end",
        "scheduled_request": {
            "arm": {"arm_code": "FULL_BAG_K"},
            "call_label": f"seed-{seed}",
            "sampling_seed": seed,
            "image_id": image_id,
        },
        "execution_evidence": {
            "image_id": image_id,
            "source_width": 1000,
            "source_height": 1000,
            "sampling_seed": seed,
        },
        "decode_result": {
            "prompt_token_ids": [1, 2, 3],
            "generated_token_ids": generated,
            "raw_generated_text": "".join(text) + "<|im_end|>",
            "token_trace": trace,
        },
    }
    path.write_text(json.dumps(payload))


def test_atlas_is_deterministic_and_preserves_exact_parent_prefix(tmp_path: Path) -> None:
    root = tmp_path / "calls"
    root.mkdir()
    call_a = root / "a"
    call_b = root / "b"
    call_a.mkdir()
    call_b.mkdir()
    _bundle(call_a / "terminal-output-bundle.json", "42", [("cup", (10, 20, 30, 40)), ("fork", (50, 60, 70, 80))], 1)
    _bundle(call_b / "terminal-output-bundle.json", "42", [("spoon", (100, 120, 140, 160)), ("fork", (170, 180, 190, 200))], 2)

    first = build_atlas((root,), ledger_path=None, image_ids=["42"])
    second = build_atlas((root,), ledger_path=None, image_ids=["42"])

    assert first == second
    assert first["call_count"] == 2
    assert first["calls_per_image"] == {"42": 2}
    assert all(row["parse_status"] == "accepted" for call in first["calls"] for row in call["rows"])
    assert all(len(row["generated_token_span"]) == 2 for call in first["calls"] for row in call["rows"])

    groups = first["recurrent_sibling_groups"]
    assert len(groups) == 3
    # The second row from the cup and spoon prefixes must not be mixed merely
    # because both have row_index=1 and category="fork".
    second_row_groups = [group for group in groups if group["support_count"] == 1]
    assert len(second_row_groups) == 2
    assert {tuple(group["category_counts"]) for group in second_row_groups} == {("fork",)}
    assert len({group["parent_prefix_sha256"] for group in second_row_groups}) == 2


def test_non_full_bag_calls_are_not_admitted(tmp_path: Path) -> None:
    root = tmp_path / "calls"
    root.mkdir()
    full_path = root / "full" / "terminal-output-bundle.json"
    masked_path = root / "masked" / "terminal-output-bundle.json"
    full_path.parent.mkdir()
    masked_path.parent.mkdir()
    _bundle(full_path, "42", [("cup", (10, 20, 30, 40))], 1)
    payload = json.loads(full_path.read_text())
    payload["scheduled_request"]["arm"]["arm_code"] = "MASK_RESET"
    masked_path.write_text(json.dumps(payload))

    atlas = build_atlas((root,), ledger_path=None, image_ids=["42"])
    assert atlas["call_count"] == 1
    assert atlas["skipped"]["non_full_bag_arm"] == 1
