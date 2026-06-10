from __future__ import annotations

from src.analysis.prefix_state_transition_tomography.gallery import select_gallery_rows


def test_select_gallery_rows_prioritizes_disagreements_and_key_quadrants() -> None:
    rows = [
        _row("a", "et_rmp_ce", "boundary_good_x1_good", emitted=0.0),
        _row("a", "pure_ce", "boundary_good_x1_good", emitted=0.0),
        _row("b", "et_rmp_ce", "boundary_good_x1_bad", emitted=0.7),
        _row("b", "pure_ce", "boundary_bad_x1_good", emitted=0.2),
        _row("c", "et_rmp_ce", "boundary_bad_x1_bad", emitted=0.1),
        _row("c", "pure_ce", "boundary_bad_x1_bad", emitted=0.1),
    ]

    selected = select_gallery_rows(rows, max_rows=2)

    assert [row["paired_key"] for row in selected] == ["b", "c"]
    assert selected[0]["quadrant_et_rmp_ce"] == "boundary_good_x1_bad"
    assert selected[0]["quadrant_pure_ce"] == "boundary_bad_x1_good"
    assert selected[0]["manual_label"] == ""


def _row(paired_key: str, role: str, quadrant: str, *, emitted: float) -> dict[str, object]:
    return {
        "paired_key": paired_key,
        "checkpoint_role": role,
        "quadrant": quadrant,
        "split": "train",
        "transition_type": "same_desc_transition",
        "prefix_depth": "shallow_1",
        "emitted_attraction_rate": emitted,
    }
