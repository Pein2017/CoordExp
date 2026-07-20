from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


MODULE = Path(__file__).parents[2] / "src/coco_refinement/static/svg-editor.js"


def _run_node(body: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is optional and unavailable")
    harness = r"""
import assert from 'node:assert/strict';

class FakeClassList {
  constructor(owner) {
    this.owner = owner;
    this.values = new Set();
  }

  add(...values) {
    values.forEach(value => this.values.add(value));
    this.sync();
  }

  remove(...values) {
    values.forEach(value => this.values.delete(value));
    this.sync();
  }

  toggle(value, force) {
    const enabled = force === undefined ? !this.values.has(value) : Boolean(force);
    if (enabled) this.values.add(value);
    else this.values.delete(value);
    this.sync();
    return enabled;
  }

  contains(value) {
    return this.values.has(value);
  }

  replaceFromAttribute(value) {
    this.values = new Set(String(value).split(/\s+/).filter(Boolean));
  }

  sync() {
    this.owner.attributes.set('class', [...this.values].join(' '));
  }
}

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.attributes = new Map();
    this.children = [];
    this.dataset = {};
    this.listeners = new Map();
    this.classList = new FakeClassList(this);
    this.textContent = '';
    this.parentElement = null;
  }

  setAttribute(name, value) {
    const text = String(value);
    this.attributes.set(name, text);
    if (name === 'class') this.classList.replaceFromAttribute(text);
  }

  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }

  hasAttribute(name) {
    return this.attributes.has(name);
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  append(...children) {
    for (const child of children) {
      child.parentElement = this;
      this.children.push(child);
    }
  }

  replaceChildren(...children) {
    for (const child of this.children) child.parentElement = null;
    this.children = [];
    this.append(...children);
  }

  addEventListener(type, callback) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(callback);
  }

  dispatch(type, properties = {}) {
    const event = {
      type,
      target: this,
      button: 0,
      pointerId: 1,
      clientX: 0,
      clientY: 0,
      preventDefault() { this.defaultPrevented = true; },
      ...properties,
    };
    for (const callback of this.listeners.get(type) || []) callback(event);
    return event;
  }

  closest(selector) {
    if (selector !== '[data-region-key]') return null;
    let current = this;
    while (current) {
      if (current.hasAttribute('data-region-key')) return current;
      current = current.parentElement;
    }
    return null;
  }
}

class FakeSVGSVGElement extends FakeElement {
  constructor() {
    super('svg');
    this.capturedPointers = new Set();
    this.bounds = { left: 10, top: 20, width: 200, height: 100 };
  }

  getBoundingClientRect() {
    return this.bounds;
  }

  getScreenCTM() {
    return null;
  }

  focus() {}

  setPointerCapture(pointerId) {
    this.capturedPointers.add(pointerId);
  }

  hasPointerCapture(pointerId) {
    return this.capturedPointers.has(pointerId);
  }

  releasePointerCapture(pointerId) {
    this.capturedPointers.delete(pointerId);
  }
}

globalThis.Element = FakeElement;
globalThis.SVGSVGElement = FakeSVGSVGElement;
globalThis.DOMPoint = class DOMPoint {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  matrixTransform() {
    return { x: this.x, y: this.y };
  }
};
globalThis.document = {
  createElementNS(_namespace, tagName) {
    return new FakeElement(tagName);
  },
};

function descendants(root) {
  return root.children.flatMap(child => [child, ...descendants(child)]);
}

function guides(svg) {
  return descendants(svg).filter(node => node.classList.contains('editor-crosshair-guide'));
}

function guide(svg, axis) {
  const match = guides(svg).find(node => node.getAttribute('data-guide-axis') === axis);
  assert.ok(match, `missing ${axis} guide`);
  return match;
}

function coordinates(node) {
  return Object.fromEntries(['x1', 'y1', 'x2', 'y2'].map(name => [name, Number(node.getAttribute(name))]));
}

function editorPart(svg, regionKey, part, handle = null) {
  const match = descendants(svg).find(node =>
    node.getAttribute('data-region-key') === regionKey &&
    node.getAttribute('data-editor-part') === part &&
    (handle === null || node.getAttribute('data-handle') === handle));
  assert.ok(match, `missing ${regionKey} ${part} ${handle || ''}`);
  return match;
}

function rectCenter(node) {
  return {
    x: Number(node.getAttribute('x')) + Number(node.getAttribute('width')) / 2,
    y: Number(node.getAttribute('y')) + Number(node.getAttribute('height')) / 2,
  };
}

function naturalToClient(svg, point) {
  const [x, y, width, height] = svg.getAttribute('viewBox').split(/\s+/).map(Number);
  const bounds = svg.getBoundingClientRect();
  const scale = Math.min(bounds.width / width, bounds.height / height);
  const offsetX = (bounds.width - width * scale) / 2;
  const offsetY = (bounds.height - height * scale) / 2;
  return {
    clientX: bounds.left + offsetX + (point.x - x) * scale,
    clientY: bounds.top + offsetY + (point.y - y) * scale,
  };
}

const { createSvgEditor } = await import(MODULE_URL);
"""
    script = harness.replace("MODULE_URL", json.dumps(MODULE.resolve().as_uri())) + body
    subprocess.run(
        [node, "--experimental-default-type=module", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_pointer_guides_follow_draw_eligibility_without_semantic_callbacks() -> None:
    _run_node(
        r"""
const svg = new FakeSVGSVGElement();
const callbacks = { gestures: [], selections: [], messages: [] };
const editor = createSvgEditor({
  svg,
  onGesture: detail => callbacks.gestures.push(detail),
  onSelection: detail => callbacks.selections.push(detail),
  onMessage: detail => callbacks.messages.push(detail),
});
assert.equal(guides(svg).length, 2);

editor.setTask({ image_width: 1000, image_height: 500, image_url: '/api/images/one', objects: [] });
svg.dispatch('pointermove', { clientX: 60, clientY: 45 });
for (const node of guides(svg)) assert.ok(node.hasAttribute('hidden'));

editor.setMode('draw');
for (const node of guides(svg)) assert.ok(!node.hasAttribute('hidden'));
assert.deepEqual(coordinates(guide(svg, 'horizontal')), { x1: 0, y1: 125, x2: 1000, y2: 125 });
assert.deepEqual(coordinates(guide(svg, 'vertical')), { x1: 250, y1: 0, x2: 250, y2: 500 });

svg.dispatch('pointermove', { clientX: 5, clientY: 45 });
for (const node of guides(svg)) assert.ok(node.hasAttribute('hidden'));
svg.dispatch('pointermove', { clientX: 60, clientY: 45 });
for (const node of guides(svg)) assert.ok(!node.hasAttribute('hidden'));

svg.dispatch('pointerleave', { clientX: 60, clientY: 45 });
for (const node of guides(svg)) assert.ok(node.hasAttribute('hidden'));
editor.setMode('select');
svg.dispatch('pointermove', { clientX: 60, clientY: 45 });
for (const node of guides(svg)) assert.ok(node.hasAttribute('hidden'));

editor.setMode('draw');
editor.setDisabled(true);
for (const node of guides(svg)) assert.ok(node.hasAttribute('hidden'));
editor.setDisabled(false);
svg.dispatch('pointermove', { clientX: 60, clientY: 45 });
for (const node of guides(svg)) assert.ok(!node.hasAttribute('hidden'));
editor.setTask(null);
for (const node of guides(svg)) assert.ok(node.hasAttribute('hidden'));

assert.deepEqual(callbacks, { gestures: [], selections: [], messages: [] });
"""
    )


def test_pointer_guides_use_current_viewbox_and_stay_bounded_during_capture_drag() -> None:
    _run_node(
        r"""
const svg = new FakeSVGSVGElement();
const gestures = [];
const editor = createSvgEditor({ svg, onGesture: detail => gestures.push(detail) });
editor.setTask({ image_width: 1000, image_height: 500, image_url: '/api/images/one', objects: [] });
editor.setMode('draw');
editor.setCategory({ id: 1, name: 'person' });

editor.zoomIn();
svg.dispatch('pointermove', { pointerId: 7, clientX: 60, clientY: 45 });
assert.deepEqual(coordinates(guide(svg, 'horizontal')), { x1: 0, y1: 150, x2: 1000, y2: 150 });
assert.deepEqual(coordinates(guide(svg, 'vertical')), { x1: 300, y1: 0, x2: 300, y2: 500 });

editor.setMode('pan');
svg.dispatch('pointerdown', { pointerId: 8, clientX: 110, clientY: 70 });
svg.dispatch('pointermove', { pointerId: 8, clientX: 90, clientY: 70 });
svg.dispatch('pointerup', { pointerId: 8, clientX: 90, clientY: 70 });
editor.setMode('draw');
svg.dispatch('pointermove', { pointerId: 7, clientX: 60, clientY: 45 });
assert.deepEqual(coordinates(guide(svg, 'vertical')), { x1: 380, y1: 0, x2: 380, y2: 500 });

svg.dispatch('pointerdown', { pointerId: 7, clientX: 60, clientY: 45 });
assert.equal(editor.hasActiveGesture(), true);
svg.dispatch('pointerleave', { pointerId: 7, clientX: 220, clientY: 200 });
assert.deepEqual(coordinates(guide(svg, 'horizontal')), { x1: 0, y1: 500, x2: 1000, y2: 500 });
assert.deepEqual(coordinates(guide(svg, 'vertical')), { x1: 1000, y1: 0, x2: 1000, y2: 500 });
for (const node of guides(svg)) assert.ok(!node.hasAttribute('hidden'));
svg.dispatch('pointermove', { pointerId: 7, clientX: 260, clientY: 200 });
assert.deepEqual(coordinates(guide(svg, 'horizontal')), { x1: 0, y1: 500, x2: 1000, y2: 500 });
assert.deepEqual(coordinates(guide(svg, 'vertical')), { x1: 1000, y1: 0, x2: 1000, y2: 500 });
for (const node of guides(svg)) assert.ok(!node.hasAttribute('hidden'));

svg.dispatch('pointerup', { pointerId: 7, clientX: 260, clientY: 200 });
assert.equal(editor.hasActiveGesture(), false);
for (const node of guides(svg)) assert.ok(node.hasAttribute('hidden'));
assert.deepEqual(gestures, [{ operation: 'create', pixelXYXY: [380, 150, 1000, 500] }]);
"""
    )


def test_selected_handle_wins_over_overlapping_body_without_raising_selected_body() -> None:
    _run_node(
        r"""
const svg = new FakeSVGSVGElement();
const gestures = [];
const selections = [];
const editor = createSvgEditor({
  svg,
  onGesture: detail => gestures.push(detail),
  onSelection: detail => selections.push(detail),
});
editor.setTask({
  image_width: 1000,
  image_height: 500,
  image_url: '/api/images/overlap',
  objects: [
    { region_key: 'a', bbox_2d: [100, 100, 500, 500], category_id: 1, category_name: 'person' },
    { region_key: 'b', bbox_2d: [450, 450, 800, 800], category_id: 1, category_name: 'person' },
  ],
});
editor.setSelected('a');

const regionsLayer = svg.children.find(node => node.classList.contains('editor-regions'));
const selectionLayer = svg.children.find(node => node.classList.contains('editor-selection-layer'));
assert.ok(svg.children.indexOf(selectionLayer) > svg.children.indexOf(regionsLayer));
const bodyB = editorPart(svg, 'b', 'body');
const se = editorPart(svg, 'a', 'handle', 'se');
assert.equal(se.getAttribute('pointer-events'), 'none');
const corner = naturalToClient(svg, rectCenter(se));

svg.dispatch('pointermove', { ...corner, target: bodyB });
assert.equal(svg.getAttribute('data-resize-handle'), 'nwse');
assert.ok(editorPart(svg, 'a', 'handle', 'se').classList.contains('is-hovered'));
svg.dispatch('pointerdown', { pointerId: 4, ...corner, target: bodyB });
assert.equal(editor.hasActiveGesture(), true);
assert.equal(svg.capturedPointers.has(4), true);
svg.dispatch('pointermove', {
  pointerId: 4,
  clientX: corner.clientX + 10,
  clientY: corner.clientY + 6,
  target: bodyB,
});
svg.dispatch('pointerup', {
  pointerId: 4,
  clientX: corner.clientX + 10,
  clientY: corner.clientY + 6,
  target: bodyB,
});
assert.equal(gestures.length, 1);
assert.equal(gestures[0].operation, 'update');
assert.equal(gestures[0].regionKey, 'a');
assert.ok(gestures[0].pixelXYXY[2] > rectCenter(se).x);
assert.deepEqual(selections, []);

const bodyBCenter = naturalToClient(svg, rectCenter(bodyB));
svg.dispatch('pointermove', { ...bodyBCenter, target: bodyB });
assert.equal(svg.getAttribute('data-resize-handle'), null);
svg.dispatch('pointerdown', { pointerId: 5, ...bodyBCenter, target: bodyB });
svg.dispatch('pointerup', { pointerId: 5, ...bodyBCenter, target: bodyB });
assert.equal(selections.at(-1).regionKey, 'b');
assert.equal(gestures.length, 1);
"""
    )


def test_screen_space_handle_radius_uses_nearest_with_corner_tie_priority() -> None:
    _run_node(
        r"""
const svg = new FakeSVGSVGElement();
const editor = createSvgEditor({ svg });
editor.setTask({
  image_width: 1000,
  image_height: 500,
  image_url: '/api/images/tiny',
  objects: [
    { region_key: 'tiny', bbox_2d: [100, 100, 120, 120], category_id: 1, category_name: 'person' },
  ],
});
editor.setSelected('tiny');

let nw = naturalToClient(svg, rectCenter(editorPart(svg, 'tiny', 'handle', 'nw')));
let n = naturalToClient(svg, rectCenter(editorPart(svg, 'tiny', 'handle', 'n')));
const exactTie = {
  clientX: (nw.clientX + n.clientX) / 2,
  clientY: (nw.clientY + n.clientY) / 2,
};
svg.dispatch('pointermove', exactTie);
assert.equal(svg.getAttribute('data-resize-handle'), 'nwse');
assert.ok(editorPart(svg, 'tiny', 'handle', 'nw').classList.contains('is-hovered'));

editor.zoomIn();
nw = naturalToClient(svg, rectCenter(editorPart(svg, 'tiny', 'handle', 'nw')));
const outwardUnit = 1 / Math.sqrt(2);
svg.dispatch('pointermove', {
  clientX: nw.clientX - 13 * outwardUnit,
  clientY: nw.clientY - 13 * outwardUnit,
});
assert.equal(svg.getAttribute('data-resize-handle'), 'nwse');
svg.dispatch('pointermove', {
  clientX: nw.clientX - 15 * outwardUnit,
  clientY: nw.clientY - 15 * outwardUnit,
});
assert.equal(svg.getAttribute('data-resize-handle'), null);
"""
    )


def test_local_magnification_anchors_pointer_and_keeps_pan_presentation_only() -> None:
    _run_node(
        r"""
const svg = new FakeSVGSVGElement();
const callbacks = { gestures: [], selections: [], views: [] };
const editor = createSvgEditor({
  svg,
  onGesture: detail => callbacks.gestures.push(detail),
  onSelection: detail => callbacks.selections.push(detail),
  onViewChange: detail => callbacks.views.push(detail),
});
editor.setTask({
  image_width: 1000,
  image_height: 500,
  image_url: '/api/images/zoom',
  objects: [
    { region_key: 'tiny', bbox_2d: [100, 100, 150, 150], category_id: 1, category_name: 'person' },
  ],
});
editor.setMode('draw');

const anchorClient = { clientX: 60, clientY: 45 };
const anchorBefore = { x: 250, y: 125 };
const wheel = svg.dispatch('wheel', { ...anchorClient, deltaY: -100 });
assert.equal(wheel.defaultPrevented, true);
const [x, y, width, height] = svg.getAttribute('viewBox').split(/\s+/).map(Number);
const anchorAfter = {
  x: x + (anchorClient.clientX - svg.bounds.left) * width / svg.bounds.width,
  y: y + (anchorClient.clientY - svg.bounds.top) * height / svg.bounds.height,
};
assert.deepEqual(anchorAfter, anchorBefore);
assert.ok(callbacks.views.at(-1).zoom > 1);
assert.equal(callbacks.views.at(-1).fit, false);

const beforePan = svg.getAttribute('viewBox');
editor.setTemporaryPan(true);
svg.dispatch('pointerdown', { pointerId: 8, ...anchorClient });
svg.dispatch('pointermove', { pointerId: 8, clientX: 40, clientY: 45 });
svg.dispatch('pointerup', { pointerId: 8, clientX: 40, clientY: 45 });
editor.setTemporaryPan(false);
assert.notEqual(svg.getAttribute('viewBox'), beforePan);
assert.equal(svg.dataset.mode, 'draw');

editor.setSelected('tiny');
assert.equal(editor.focusSelected(), true);
const focused = editor.getViewState();
assert.ok(focused.viewBox.width < 1000);
assert.equal(focused.viewBox.width / focused.viewBox.height, 2);
assert.equal(focused.zoom, 1000 / focused.viewBox.width);

const beforeMiddlePan = svg.getAttribute('viewBox');
svg.dispatch('pointerdown', { pointerId: 9, button: 1, clientX: 110, clientY: 70 });
svg.dispatch('pointermove', { pointerId: 9, button: 1, clientX: 90, clientY: 70 });
svg.dispatch('pointerup', { pointerId: 9, button: 1, clientX: 90, clientY: 70 });
assert.notEqual(svg.getAttribute('viewBox'), beforeMiddlePan);
assert.equal(svg.dataset.mode, 'draw');
assert.deepEqual(callbacks.gestures, []);
assert.deepEqual(callbacks.selections, []);

editor.setCategory({ id: 1, name: 'person' });
svg.bounds = { left: 10, top: 20, width: 200, height: 200 };
editor.refreshViewport();
const naturalStart = { x: 120, y: 60 };
const naturalEnd = { x: 140, y: 70 };
const clientStart = naturalToClient(svg, naturalStart);
const clientEnd = naturalToClient(svg, naturalEnd);
svg.dispatch('pointerdown', { pointerId: 10, ...clientStart });
svg.dispatch('pointermove', { pointerId: 10, ...clientEnd });
svg.dispatch('pointerup', { pointerId: 10, ...clientEnd });
assert.equal(callbacks.gestures.length, 1);
assert.equal(callbacks.gestures[0].operation, 'create');
const expected = [naturalStart.x, naturalStart.y, naturalEnd.x, naturalEnd.y];
callbacks.gestures[0].pixelXYXY.forEach((value, index) => {
  assert.ok(Math.abs(value - expected[index]) < 1e-9);
});
"""
    )
