from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


MODULE = Path(__file__).parents[2] / "src/coco_refinement/static/editor-geometry.js"


def _run_node(expression: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is optional and unavailable")
    script = f"""
import * as geometry from {json.dumps(MODULE.resolve().as_uri())};
const value = ({expression});
console.log(JSON.stringify(value));
"""
    completed = subprocess.run(
        [node, "--experimental-default-type=module", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_natural_extent_and_norm1000_projection_for_landscape_and_portrait() -> None:
    value = _run_node("""({
      landscape: geometry.norm1000ToNaturalRect([0, 0, 999, 999], 1920, 1080),
      portrait: geometry.norm1000ToNaturalRect([333, 111, 666, 888], 900, 1600),
      extent: geometry.validateNaturalExtent(640, 480),
    })""")

    assert value["landscape"] == {"x": 0, "y": 0, "width": 1920, "height": 1080}
    assert value["portrait"]["x"] == pytest.approx(300)
    assert value["portrait"]["y"] == pytest.approx(177.7777777778)
    assert value["portrait"]["width"] == pytest.approx(300)
    assert value["portrait"]["height"] == pytest.approx(1244.4444444444)
    assert value["extent"] == {"width": 640, "height": 480}


def test_drag_is_ordered_and_clipped_from_each_corner() -> None:
    value = _run_node("""[
      geometry.orderedClippedDragRect({x: -20, y: -10}, {x: 120, y: 90}, 100, 80),
      geometry.orderedClippedDragRect({x: 120, y: -10}, {x: -20, y: 90}, 100, 80),
      geometry.orderedClippedDragRect({x: 120, y: 90}, {x: -20, y: -10}, 100, 80),
      geometry.orderedClippedDragRect({x: -20, y: 90}, {x: 120, y: -10}, 100, 80),
    ]""")

    assert value == [{"x": 0, "y": 0, "width": 100, "height": 80}] * 4


def test_move_preserves_size_and_clamps_at_all_boundaries() -> None:
    value = _run_node("""({
      northwest: geometry.moveRect({x: 20, y: 30, width: 40, height: 20}, -100, -100, 100, 80),
      southeast: geometry.moveRect({x: 20, y: 30, width: 40, height: 20}, 100, 100, 100, 80),
    })""")

    assert value["northwest"] == {"x": 0, "y": 0, "width": 40, "height": 20}
    assert value["southeast"] == {"x": 60, "y": 60, "width": 40, "height": 20}


def test_all_eight_resize_handles_and_minimum_size() -> None:
    value = _run_node("""Object.fromEntries(
      ['n','ne','e','se','s','sw','w','nw'].map(handle => [
        handle,
        geometry.resizeRect({x: 20, y: 20, width: 40, height: 30}, handle, 5, 7, 100, 80, 4),
      ])
    )""")

    assert value["n"] == {"x": 20, "y": 27, "width": 40, "height": 23}
    assert value["ne"] == {"x": 20, "y": 27, "width": 45, "height": 23}
    assert value["e"] == {"x": 20, "y": 20, "width": 45, "height": 30}
    assert value["se"] == {"x": 20, "y": 20, "width": 45, "height": 37}
    assert value["s"] == {"x": 20, "y": 20, "width": 40, "height": 37}
    assert value["sw"] == {"x": 25, "y": 20, "width": 35, "height": 37}
    assert value["w"] == {"x": 25, "y": 20, "width": 35, "height": 30}
    assert value["nw"] == {"x": 25, "y": 27, "width": 35, "height": 23}

    minimum = _run_node(
        "geometry.resizeRect({x: 20, y: 20, width: 40, height: 30}, 'nw', 999, 999, 100, 80, 6)"
    )
    assert minimum == {"x": 54, "y": 44, "width": 6, "height": 6}


def test_zoom_about_anchor_pan_clamp_and_reset_vectors() -> None:
    value = _run_node("""(() => {
      const reset = geometry.resetViewBox(1200, 800);
      const zoomed = geometry.zoomViewBox(reset, 2, {x: 300, y: 200}, 1200, 800);
      return {
        reset,
        zoomed,
        panNorthwest: geometry.panViewBox(zoomed, -9999, -9999, 1200, 800),
        panSoutheast: geometry.panViewBox(zoomed, 9999, 9999, 1200, 800),
        zoomOut: geometry.zoomViewBox(zoomed, 0.01, {x: 300, y: 200}, 1200, 800),
      };
    })()""")

    assert value["reset"] == {"x": 0, "y": 0, "width": 1200, "height": 800}
    assert value["zoomed"] == {"x": 150, "y": 100, "width": 600, "height": 400}
    assert value["panNorthwest"] == {"x": 0, "y": 0, "width": 600, "height": 400}
    assert value["panSoutheast"] == {"x": 600, "y": 400, "width": 600, "height": 400}
    assert value["zoomOut"] == {"x": 0, "y": 0, "width": 1200, "height": 800}


def test_client_natural_contain_transform_round_trips_through_letterbox() -> None:
    value = _run_node("""(() => {
      const bounds = {left: 10, top: 20, width: 200, height: 200};
      const viewBox = {x: 100, y: 50, width: 1200, height: 800};
      const natural = {x: 400, y: 250};
      const client = geometry.naturalToClientPoint(natural, bounds, viewBox);
      return {
        client,
        roundTrip: geometry.clientToNaturalPoint(client, bounds, viewBox),
        topLetterbox: geometry.clientToNaturalPoint({x: 110, y: 20}, bounds, viewBox),
      };
    })()""")

    assert value["client"] == pytest.approx({"x": 60, "y": 86.6666666667})
    assert value["roundTrip"] == pytest.approx({"x": 400, "y": 250})
    assert value["topLetterbox"] == pytest.approx({"x": 700, "y": -150})


def test_focus_viewbox_preserves_image_aspect_padding_and_edge_clamp() -> None:
    value = _run_node("""({
      centered: geometry.focusViewBox(
        {x: 100, y: 100, width: 100, height: 50}, 1200, 800, 0.15, 24),
      atEdge: geometry.focusViewBox(
        {x: 0, y: 0, width: 20, height: 20}, 1200, 800, 0.15, 24),
      full: geometry.focusViewBox(
        {x: 0, y: 0, width: 1200, height: 800}, 1200, 800, 0.15, 24),
    })""")

    centered = value["centered"]
    assert centered == pytest.approx(
        {"x": 85, "y": 81.6666666667, "width": 130, "height": 86.6666666667}
    )
    assert centered["width"] / centered["height"] == pytest.approx(1200 / 800)
    assert value["atEdge"]["x"] == 0
    assert value["atEdge"]["y"] == 0
    assert value["full"] == {"x": 0, "y": 0, "width": 1200, "height": 800}


def test_invalid_geometry_fails_closed() -> None:
    value = _run_node("""(() => {
      const cases = [
        () => geometry.validateNaturalExtent(0, 10),
        () => geometry.norm1000ToNaturalRect([0, 0, 1000, 2], 10, 10),
        () => geometry.resizeRect({x: 1, y: 1, width: 2, height: 2}, 'center', 1, 1, 10, 10),
        () => geometry.clientToNaturalPoint(
          {x: 1, y: 1}, {left: 0, top: 0, width: 0, height: 10},
          {x: 0, y: 0, width: 10, height: 10}),
        () => geometry.focusViewBox(
          {x: 1, y: 1, width: 2, height: 2}, 10, 10, -0.1),
      ];
      return cases.map(run => { try { run(); return false; } catch { return true; } });
    })()""")

    assert value == [True, True, True, True, True]
