import {
  moveRect,
  norm1000ToNaturalRect,
  orderedClippedDragRect,
  panViewBox,
  resetViewBox,
  resizeRect,
  zoomViewBox,
} from './editor-geometry.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const MODES = new Set(['select', 'draw', 'pan']);
const VISIBILITY_MODES = new Set(['all', 'dim-nonselected', 'hide-nonselected']);
const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
const MINIMUM_BOX_SIZE = 1;

function element(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

function copyRect(rect) {
  return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
}

function rectToXYXY(rect) {
  return [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height];
}

function rectanglesEqual(left, right) {
  return left.x === right.x && left.y === right.y &&
    left.width === right.width && left.height === right.height;
}

function colorFor(value) {
  let hash = 2166136261;
  for (const char of String(value)) {
    hash ^= char.codePointAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return `hsl(${Math.abs(hash) % 360} 82% 44%)`;
}

function handleCenter(rect, handle) {
  const left = rect.x;
  const centerX = rect.x + rect.width / 2;
  const right = rect.x + rect.width;
  const top = rect.y;
  const centerY = rect.y + rect.height / 2;
  const bottom = rect.y + rect.height;
  return {
    x: handle.includes('w') ? left : handle.includes('e') ? right : centerX,
    y: handle.includes('n') ? top : handle.includes('s') ? bottom : centerY,
  };
}

function validCategory(category) {
  return Boolean(
    category && typeof category === 'object' &&
    Number.isInteger(category.id) && typeof category.name === 'string' && category.name,
  );
}

/**
 * Own the image and bbox DOM inside one natural-coordinate SVG.
 *
 * `onGesture` receives one event only at pointer-up:
 * `{ operation: 'create'|'update', regionKey?, pixelXYXY }`.
 * Persistence and canonicalization stay with the application controller/server.
 */
export function createSvgEditor({
  svg,
  onGesture = () => {},
  onSelection = () => {},
  onMessage = () => {},
}) {
  if (!(svg instanceof SVGSVGElement)) throw new TypeError('svg must be an SVGSVGElement');
  if (![onGesture, onSelection, onMessage].every(callback => typeof callback === 'function')) {
    throw new TypeError('editor callbacks must be functions');
  }

  const image = element('image', { preserveAspectRatio: 'none' });
  image.classList.add('editor-image');
  const regionsLayer = element('g', { 'aria-label': 'Bounding box regions' });
  regionsLayer.classList.add('editor-regions');
  const previewLayer = element('g', { 'aria-hidden': 'true' });
  previewLayer.classList.add('editor-preview-layer');
  const horizontalGuide = element('line', {
    class: 'editor-crosshair-guide',
    'data-guide-axis': 'horizontal',
    'aria-hidden': 'true',
    'pointer-events': 'none',
    visibility: 'hidden',
    hidden: '',
  });
  const verticalGuide = element('line', {
    class: 'editor-crosshair-guide',
    'data-guide-axis': 'vertical',
    'aria-hidden': 'true',
    'pointer-events': 'none',
    visibility: 'hidden',
    hidden: '',
  });
  svg.replaceChildren(image, regionsLayer, previewLayer, horizontalGuide, verticalGuide);
  svg.setAttribute('tabindex', '0');
  svg.setAttribute('role', 'application');
  svg.setAttribute('aria-label', 'COCO bounding box editor');
  svg.dataset.mode = 'select';

  const state = {
    width: 0,
    height: 0,
    objects: [],
    rectangles: new Map(),
    selected: null,
    category: null,
    mode: 'select',
    visibilityMode: 'all',
    hiddenRegionKeys: new Set(),
    disabled: false,
    viewBox: null,
    gesture: null,
    pointerClient: null,
  };

  function message(code, text, tone = 'info') {
    onMessage({ code, message: text, tone });
  }

  function hasTask() {
    return state.width > 0 && state.height > 0 && state.viewBox;
  }

  function applyViewBox() {
    if (!state.viewBox) return;
    const { x, y, width, height } = state.viewBox;
    svg.setAttribute('viewBox', `${x} ${y} ${width} ${height}`);
    renderRegions();
    renderPointerGuides();
  }

  function naturalUnitsForPixels(pixels) {
    const bounds = svg.getBoundingClientRect();
    if (!state.viewBox || bounds.width <= 0 || bounds.height <= 0) return pixels;
    return Math.max(
      pixels * state.viewBox.width / bounds.width,
      pixels * state.viewBox.height / bounds.height,
    );
  }

  function renderPreview(rect) {
    previewLayer.replaceChildren();
    if (!rect || rect.width <= 0 || rect.height <= 0) return;
    previewLayer.append(element('rect', {
      class: 'editor-preview',
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    }));
  }

  function renderRegions() {
    regionsLayer.replaceChildren();
    if (!hasTask()) return;
    const handleSize = naturalUnitsForPixels(10);
    const labelSize = naturalUnitsForPixels(13);
    for (const object of state.objects) {
      const regionKey = object.region_key;
      const rect = state.rectangles.get(regionKey);
      if (!rect) continue;
      const selected = regionKey === state.selected;
      const focusHidden = state.visibilityMode === 'hide-nonselected' && !selected;
      const hidden = state.hiddenRegionKeys.has(regionKey) || focusHidden;
      const dimmed = state.visibilityMode === 'dim-nonselected' && !selected;
      const color = colorFor(regionKey);
      const group = element('g', {
        class: [
          'editor-region',
          selected ? 'is-selected' : '',
          dimmed ? 'is-dimmed' : '',
          hidden ? 'is-presentation-hidden' : '',
        ].filter(Boolean).join(' '),
        'data-region-key': regionKey,
        role: 'button',
        tabindex: '0',
        'aria-label': `${object.category_name || object.category_id}, bbox ${object.bbox_2d.join(', ')}`,
        'aria-hidden': String(hidden),
      });
      if (hidden) group.setAttribute('hidden', '');
      const title = element('title');
      title.textContent = `${object.category_name || object.category_id} · ${regionKey}`;
      const visual = element('rect', {
        class: 'editor-box',
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        fill: color,
        stroke: color,
      });
      const hitTarget = element('rect', {
        class: 'editor-box-hit',
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        'data-region-key': regionKey,
        'data-editor-part': 'body',
      });
      const label = element('text', {
        class: 'editor-label',
        x: rect.x + naturalUnitsForPixels(4),
        y: Math.max(labelSize, rect.y + labelSize),
        'font-size': labelSize,
        fill: color,
      });
      label.textContent = object.category_name || `COCO ${object.category_id}`;
      group.append(title, visual, hitTarget, label);
      if (selected) {
        for (const handle of HANDLES) {
          const center = handleCenter(rect, handle);
          group.append(element('rect', {
            class: 'editor-handle',
            x: center.x - handleSize / 2,
            y: center.y - handleSize / 2,
            width: handleSize,
            height: handleSize,
            'data-region-key': regionKey,
            'data-editor-part': 'handle',
            'data-handle': handle,
          }));
        }
      }
      regionsLayer.append(group);
    }
  }

  function select(regionKey, notify = true) {
    const next = state.rectangles.has(regionKey) ? regionKey : null;
    if (next === state.selected) return;
    if (next) state.hiddenRegionKeys.delete(next);
    state.selected = next;
    if (next === null && state.visibilityMode === 'hide-nonselected') {
      state.visibilityMode = 'all';
    }
    renderRegions();
    if (notify) {
      const object = state.objects.find(candidate => candidate.region_key === next) || null;
      onSelection({ regionKey: next, object });
    }
  }

  function updateObjects(objects) {
    if (!Array.isArray(objects)) throw new TypeError('objects must be an array');
    const rectangles = new Map();
    const accepted = [];
    for (const object of objects) {
      const regionKey = object?.region_key;
      if (typeof regionKey !== 'string' || !regionKey || rectangles.has(regionKey)) {
        message('invalid_region', 'Skipped an object with a missing or duplicate stable key.', 'warning');
        continue;
      }
      try {
        rectangles.set(
          regionKey,
          norm1000ToNaturalRect(object.bbox_2d, state.width, state.height),
        );
        accepted.push(object);
      } catch (_error) {
        message('invalid_region_geometry', `Skipped invalid bbox ${regionKey}.`, 'warning');
      }
    }
    const selectedWasRemoved = state.selected !== null && !rectangles.has(state.selected);
    state.objects = accepted;
    state.rectangles = rectangles;
    state.hiddenRegionKeys = new Set(
      [...state.hiddenRegionKeys].filter(regionKey => rectangles.has(regionKey)),
    );
    if (selectedWasRemoved) {
      state.selected = null;
      if (state.visibilityMode === 'hide-nonselected') state.visibilityMode = 'all';
    }
    renderRegions();
    if (selectedWasRemoved) onSelection({ regionKey: null, object: null });
  }

  function restoreVisibility() {
    state.visibilityMode = 'all';
    state.hiddenRegionKeys.clear();
    renderRegions();
    return getPresentationState();
  }

  function getPresentationState() {
    return {
      visibilityMode: state.visibilityMode,
      hiddenRegionKeys: [...state.hiddenRegionKeys].sort(),
      selectedRegionKey: state.selected,
    };
  }

  function clientToNatural(event) {
    const matrix = svg.getScreenCTM();
    if (matrix) {
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
      return { x: point.x, y: point.y };
    }
    const bounds = svg.getBoundingClientRect();
    if (!state.viewBox || bounds.width <= 0 || bounds.height <= 0) return { x: 0, y: 0 };
    return {
      x: state.viewBox.x + (event.clientX - bounds.left) * state.viewBox.width / bounds.width,
      y: state.viewBox.y + (event.clientY - bounds.top) * state.viewBox.height / bounds.height,
    };
  }

  function hidePointerGuides() {
    horizontalGuide.setAttribute('visibility', 'hidden');
    verticalGuide.setAttribute('visibility', 'hidden');
    horizontalGuide.setAttribute('hidden', '');
    verticalGuide.setAttribute('hidden', '');
  }

  function renderPointerGuides() {
    if (
      state.mode !== 'draw' || state.disabled || !hasTask() || !state.pointerClient
    ) {
      hidePointerGuides();
      return;
    }
    const point = clientToNatural(state.pointerClient);
    const inside = point.x >= 0 && point.x <= state.width &&
      point.y >= 0 && point.y <= state.height;
    const capturedDraw = state.gesture?.type === 'draw' &&
      state.gesture.pointerId === state.pointerClient.pointerId;
    if (!inside && !capturedDraw) {
      hidePointerGuides();
      return;
    }
    const x = Math.min(state.width, Math.max(0, point.x));
    const y = Math.min(state.height, Math.max(0, point.y));
    for (const [node, attributes] of [
      [horizontalGuide, { x1: 0, y1: y, x2: state.width, y2: y }],
      [verticalGuide, { x1: x, y1: 0, x2: x, y2: state.height }],
    ]) {
      for (const [name, value] of Object.entries(attributes)) {
        node.setAttribute(name, value);
      }
      node.setAttribute('visibility', 'visible');
      node.removeAttribute('hidden');
    }
  }

  function trackPointer(event) {
    state.pointerClient = {
      clientX: event.clientX,
      clientY: event.clientY,
      pointerId: event.pointerId,
    };
    renderPointerGuides();
  }

  function leavePointer(event = null) {
    const capturedDraw = event && state.gesture?.type === 'draw' &&
      state.gesture.pointerId === event.pointerId;
    if (capturedDraw) {
      trackPointer(event);
      return;
    }
    state.pointerClient = null;
    hidePointerGuides();
  }

  function beginGesture(event) {
    if (state.disabled || !hasTask() || event.button !== 0) return;
    svg.focus({ preventScroll: true });
    const target = event.target instanceof Element ? event.target : null;
    const regionKey = target?.getAttribute('data-region-key');
    const part = target?.getAttribute('data-editor-part');
    const handle = target?.getAttribute('data-handle');
    const point = clientToNatural(event);

    if (state.mode === 'draw') {
      if (!validCategory(state.category)) {
        message('category_required', 'Choose one canonical COCO-80 category before drawing.', 'warning');
        return;
      }
      state.gesture = { type: 'draw', pointerId: event.pointerId, start: point, preview: null };
    } else if (state.mode === 'pan') {
      const bounds = svg.getBoundingClientRect();
      const screenMatrix = svg.getScreenCTM();
      state.gesture = {
        type: 'pan',
        pointerId: event.pointerId,
        start: point,
        inverseMatrix: screenMatrix ? screenMatrix.inverse() : null,
        bounds,
        viewBox: copyRect(state.viewBox),
      };
    } else if (regionKey && state.rectangles.has(regionKey)) {
      select(regionKey);
      const original = copyRect(state.rectangles.get(regionKey));
      state.gesture = {
        type: part === 'handle' && HANDLES.includes(handle) ? 'resize' : 'move',
        pointerId: event.pointerId,
        regionKey,
        handle,
        start: point,
        original,
        preview: original,
      };
    } else {
      select(null);
      return;
    }
    event.preventDefault();
    svg.setPointerCapture(event.pointerId);
    svg.classList.add('is-gesturing');
  }

  function moveGesture(event) {
    const gesture = state.gesture;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    if (gesture.type === 'pan') {
      let point;
      if (gesture.inverseMatrix) {
        point = new DOMPoint(event.clientX, event.clientY).matrixTransform(gesture.inverseMatrix);
      } else {
        const scale = Math.min(
          gesture.bounds.width / gesture.viewBox.width,
          gesture.bounds.height / gesture.viewBox.height,
        );
        const offsetX = (gesture.bounds.width - gesture.viewBox.width * scale) / 2;
        const offsetY = (gesture.bounds.height - gesture.viewBox.height * scale) / 2;
        point = {
          x: gesture.viewBox.x + (event.clientX - gesture.bounds.left - offsetX) / Math.max(scale, Number.EPSILON),
          y: gesture.viewBox.y + (event.clientY - gesture.bounds.top - offsetY) / Math.max(scale, Number.EPSILON),
        };
      }
      const deltaX = gesture.start.x - point.x;
      const deltaY = gesture.start.y - point.y;
      state.viewBox = panViewBox(gesture.viewBox, deltaX, deltaY, state.width, state.height);
      applyViewBox();
      return;
    }
    const point = clientToNatural(event);
    if (gesture.type === 'draw') {
      gesture.preview = orderedClippedDragRect(gesture.start, point, state.width, state.height);
    } else if (gesture.type === 'move') {
      gesture.preview = moveRect(
        gesture.original,
        point.x - gesture.start.x,
        point.y - gesture.start.y,
        state.width,
        state.height,
      );
    } else {
      gesture.preview = resizeRect(
        gesture.original,
        gesture.handle,
        point.x - gesture.start.x,
        point.y - gesture.start.y,
        state.width,
        state.height,
        MINIMUM_BOX_SIZE,
      );
    }
    renderPreview(gesture.preview);
  }

  function finishGesture(event) {
    const gesture = state.gesture;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    trackPointer(event);
    // Pointer-up can be the first or newest position delivered for a fast drag.
    // Recompute before clearing state so the emitted geometry uses that endpoint.
    moveGesture(event);
    state.gesture = null;
    svg.classList.remove('is-gesturing');
    if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    renderPreview(null);
    renderPointerGuides();
    if (gesture.type === 'pan') return;
    const rect = gesture.preview;
    if (!rect || rect.width < MINIMUM_BOX_SIZE || rect.height < MINIMUM_BOX_SIZE) {
      if (gesture.type === 'draw') {
        message('bbox_too_small', 'Draw a non-degenerate bbox before releasing.', 'warning');
      }
      return;
    }
    if (gesture.type !== 'draw' && rectanglesEqual(rect, gesture.original)) return;
    const detail = {
      operation: gesture.type === 'draw' ? 'create' : 'update',
      pixelXYXY: rectToXYXY(rect),
    };
    if (gesture.type !== 'draw') detail.regionKey = gesture.regionKey;
    onGesture(detail);
  }

  function cancelGesture() {
    const gesture = state.gesture;
    if (!gesture) return;
    state.gesture = null;
    svg.classList.remove('is-gesturing');
    if (svg.hasPointerCapture(gesture.pointerId)) svg.releasePointerCapture(gesture.pointerId);
    if (gesture.type === 'pan') state.viewBox = gesture.viewBox;
    renderPreview(null);
    applyViewBox();
    message('gesture_cancelled', 'Gesture cancelled.', 'info');
  }

  svg.addEventListener('pointerdown', event => {
    trackPointer(event);
    beginGesture(event);
    renderPointerGuides();
  });
  svg.addEventListener('pointermove', event => {
    trackPointer(event);
    moveGesture(event);
  });
  svg.addEventListener('pointerleave', leavePointer);
  svg.addEventListener('pointerup', finishGesture);
  svg.addEventListener('pointercancel', () => {
    leavePointer();
    cancelGesture();
  });
  svg.addEventListener('keydown', event => {
    if (event.key === 'Escape' && state.gesture) {
      event.preventDefault();
      cancelGesture();
      return;
    }
    if ((event.key === 'Enter' || event.key === ' ') && event.target instanceof Element) {
      const region = event.target.closest('[data-region-key]');
      const regionKey = region?.getAttribute('data-region-key');
      if (regionKey && state.rectangles.has(regionKey)) {
        event.preventDefault();
        select(regionKey);
      }
    }
  });

  return {
    setTask(task) {
      cancelGesture();
      state.pointerClient = null;
      hidePointerGuides();
      state.visibilityMode = 'all';
      state.hiddenRegionKeys.clear();
      if (!task) {
        state.width = 0;
        state.height = 0;
        state.viewBox = null;
        state.objects = [];
        state.rectangles.clear();
        state.selected = null;
        image.removeAttribute('href');
        regionsLayer.replaceChildren();
        previewLayer.replaceChildren();
        svg.setAttribute('viewBox', '0 0 1000 1000');
        return;
      }
      const width = Number(task.image_width ?? task.width);
      const height = Number(task.image_height ?? task.height);
      if (!(width > 0 && height > 0)) throw new RangeError('task image dimensions must be positive');
      const imageUrl = task.image_url ?? task.imageUrl;
      if (typeof imageUrl !== 'string' || !imageUrl.startsWith('/api/')) {
        throw new TypeError('task image URL must be a server-owned API path');
      }
      state.width = width;
      state.height = height;
      state.viewBox = resetViewBox(width, height);
      state.selected = null;
      image.setAttribute('href', imageUrl);
      image.setAttribute('x', '0');
      image.setAttribute('y', '0');
      image.setAttribute('width', String(width));
      image.setAttribute('height', String(height));
      updateObjects(task.objects || []);
      applyViewBox();
    },

    updateObjects,

    setMode(mode) {
      if (!MODES.has(mode)) throw new RangeError('mode must be select, draw, or pan');
      cancelGesture();
      state.mode = mode;
      svg.dataset.mode = mode;
      svg.setAttribute('aria-label', `COCO bounding box editor, ${mode} mode`);
      renderPointerGuides();
    },

    setCategory(category) {
      state.category = validCategory(category) ? { id: category.id, name: category.name } : null;
    },

    setDisabled(disabled) {
      state.disabled = Boolean(disabled);
      if (state.disabled) cancelGesture();
      svg.classList.toggle('is-disabled', state.disabled);
      svg.setAttribute('aria-disabled', String(state.disabled));
      renderPointerGuides();
    },

    setSelected(regionKey) {
      select(regionKey, false);
    },

    setVisibilityMode(mode) {
      if (!VISIBILITY_MODES.has(mode)) {
        throw new RangeError('visibility mode must be all, dim-nonselected, or hide-nonselected');
      }
      if (mode === 'hide-nonselected' && state.selected === null) {
        message(
          'visibility_selection_required',
          'Select one bbox before hiding non-selected objects.',
          'warning',
        );
        return getPresentationState();
      }
      state.visibilityMode = mode;
      renderRegions();
      return getPresentationState();
    },

    toggleRegionHidden(regionKey) {
      if (typeof regionKey !== 'string' || !state.rectangles.has(regionKey)) {
        throw new RangeError('regionKey must identify a current bbox');
      }
      let selectionCleared = false;
      if (state.hiddenRegionKeys.has(regionKey)) {
        state.hiddenRegionKeys.delete(regionKey);
      } else {
        state.hiddenRegionKeys.add(regionKey);
        if (state.selected === regionKey) {
          state.selected = null;
          state.visibilityMode = 'all';
          selectionCleared = true;
        }
      }
      renderRegions();
      if (selectionCleared) onSelection({ regionKey: null, object: null });
      return getPresentationState();
    },

    restoreVisibility,

    getPresentationState,

    hasActiveGesture() {
      return Boolean(state.gesture);
    },

    zoomIn() {
      if (!hasTask()) return;
      const anchor = {
        x: state.viewBox.x + state.viewBox.width / 2,
        y: state.viewBox.y + state.viewBox.height / 2,
      };
      state.viewBox = zoomViewBox(state.viewBox, 1.25, anchor, state.width, state.height, 24);
      applyViewBox();
    },

    zoomOut() {
      if (!hasTask()) return;
      const anchor = {
        x: state.viewBox.x + state.viewBox.width / 2,
        y: state.viewBox.y + state.viewBox.height / 2,
      };
      state.viewBox = zoomViewBox(state.viewBox, 0.8, anchor, state.width, state.height, 24);
      applyViewBox();
    },

    reset() {
      if (!hasTask()) return;
      cancelGesture();
      restoreVisibility();
      state.viewBox = resetViewBox(state.width, state.height);
      applyViewBox();
    },
  };
}
