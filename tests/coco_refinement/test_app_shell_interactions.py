from __future__ import annotations

from pathlib import Path
import re


STATIC_ROOT = Path(__file__).parents[2] / "src" / "coco_refinement" / "static"


def test_app_shell_exposes_sticky_class_and_object_inventory() -> None:
    index = (STATIC_ROOT / "index.html").read_text()
    css = (STATIC_ROOT / "app.css").read_text()

    assert 'id="drawing-category"' in index
    assert 'id="object-count"' in index
    assert 'id="object-list"' in index
    assert 'aria-label="Objects in training order"' in index
    assert ".drawing-category" in css
    assert ".object-list" in css
    assert '.object-list button[aria-current="true"]' in css
    guide_rule = re.search(r"\.editor-crosshair-guide\s*\{([^}]*)\}", css)
    assert guide_rule
    assert "stroke: rgba(255, 255, 255, .78)" in guide_rule.group(1)
    assert "vector-effect: non-scaling-stroke" in guide_rule.group(1)
    assert "pointer-events: none" in guide_rule.group(1)


def test_app_wires_guarded_mode_shortcuts_and_split_class_reset() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text()

    assert "function modeShortcut" in app_source
    assert "event.metaKey || event.ctrlKey" in app_source
    assert "['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)" in app_source
    assert "!state.taskOpen || semanticActionLocked()" in app_source
    assert "$('category-search').focus" in app_source
    assert re.search(
        r"async function switchSplit\([\s\S]*?setActiveCategory\(null\)",
        app_source,
    )


def test_object_inventory_uses_stable_training_order_and_bidirectional_selection() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text()

    assert "function objectsInTrainingOrder" in app_source
    assert "left.object.bbox_2d[1] - right.object.bbox_2d[1]" in app_source
    assert "left.object.bbox_2d[0] - right.object.bbox_2d[0]" in app_source
    assert "left.index - right.index" in app_source
    assert "editor.setSelected(state.selectedRegion)" in app_source
    assert "selectObject(object.region_key, object)" in app_source
    assert "scrollIntoView({ block: 'nearest' })" in app_source
    assert re.search(
        r"function renderDraftState\(snapshot\)[\s\S]*?renderObjectInventory\(snapshot.objects\)",
        app_source,
    )
