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
