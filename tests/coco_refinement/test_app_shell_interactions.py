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
    handle_rule = re.search(r"\.editor-handle\s*\{([^}]*)\}", css)
    assert handle_rule
    assert "pointer-events: none" in handle_rule.group(1)
    assert ".editor-handle.is-hovered" in css
    for cursor in ("ns", "ew", "nesw", "nwse"):
        assert f'#bbox-overlay[data-resize-handle="{cursor}"]' in css


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
    assert "activateInventoryObject(object.region_key, object)" in app_source
    assert "scrollIntoView({ block: 'nearest' })" in app_source
    assert re.search(
        r"function renderDraftState\(snapshot\)[\s\S]*?renderObjectInventory\(snapshot.objects\)",
        app_source,
    )


def test_object_inventory_activation_enters_select_and_supports_keyboard_delete() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text()

    assert re.search(
        r"function activateInventoryObject\(regionKey, object\)\s*\{"
        r"\s*setMode\('select'\);"
        r"\s*selectObject\(regionKey, object\);",
        app_source,
    )
    assert re.search(
        r"function isSelectedObjectKeyboardContext\(target\)[\s\S]*?"
        r"\$\('bbox-overlay'\)\.contains\(target\)[\s\S]*?"
        r"\$\('object-list'\)\.contains\(target\)[\s\S]*?"
        r"dataset\.regionKey === state\.selectedRegion",
        app_source,
    )
    assert re.search(
        r"event\.key === 'Delete' \|\| event\.key === 'Backspace'[\s\S]*?"
        r"isSelectedObjectKeyboardContext\(event\.target\)[\s\S]*?"
        r"event\.preventDefault\(\);[\s\S]*?deleteSelectedRegion\(\)",
        app_source,
    )


def test_local_magnification_shell_exposes_zoom_focus_and_canvas_focus() -> None:
    index = (STATIC_ROOT / "index.html").read_text()
    css = (STATIC_ROOT / "app.css").read_text()
    app_source = (STATIC_ROOT / "app.js").read_text()

    assert 'id="zoom-level"' in index
    assert 'id="zoom-selection"' in index
    assert 'id="canvas-focus"' in index
    assert "onViewChange: renderViewState" in app_source
    assert "editor.focusSelected()" in app_source
    assert "editor.setTemporaryPan(true)" in app_source
    assert "event.code === 'Space'" in app_source
    assert "event.key.toLowerCase() === 'f'" in app_source
    assert "event.key === '0'" in app_source
    assert ".workspace.is-canvas-focused" in css


def test_focus_queue_shell_separates_navigation_commit_and_publication() -> None:
    index = (STATIC_ROOT / "index.html").read_text()
    css = (STATIC_ROOT / "app.css").read_text()
    app_source = (STATIC_ROOT / "app.js").read_text()

    assert 'id="view-full-button"' in index
    assert 'id="view-focus-button"' in index
    assert 'id="focus-commit-button"' in index
    assert 'id="focus-batch-status"' in index
    assert 'id="focus-publication-status"' in index
    assert ".focus-panel[hidden]" in css
    assert "await api.getJson('/api/focus')" in app_source
    assert "await api.postJson('/api/focus/commits', { batch_id: focusBatchId() })" in app_source
    assert "await api.postJson('/api/focus/publication/retry', {})" in app_source
    assert "await api.deleteJson('/api/focus')" in app_source
    assert "tasks: members" in app_source
    assert "if (state.navigationView === 'focus')" in app_source
    assert "state.page.tasks[index].task_id" in app_source
    assert "Batch: ${focusState(queue.batch)}" in app_source
    assert "Publish: ${focusState(queue.publication)}" in app_source


def test_failed_commit_rebind_is_latched_instead_of_retried_by_every_poll() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text()

    assert app_source.count("if (state.commitRebindError) return;") == 2
    assert "if (state.commitRebindError) return Promise.reject(state.commitRebindError);" in app_source
    assert ": state.commitRebindError ? 'Authority refresh failed'" in app_source


def test_api_client_exposes_csrf_protected_delete() -> None:
    source = (STATIC_ROOT / "api-client.js").read_text()

    assert "deleteJson(path)" in source
    assert "method: 'DELETE'" in source
    assert "mutation: true" in source
