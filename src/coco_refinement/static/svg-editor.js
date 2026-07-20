import {
  clientToNaturalPoint,
  focusViewBox,
  moveRect,
  naturalToClientPoint,
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
const CORNER_HANDLES = new Set(['nw', 'ne', 'se', 'sw']);
const HANDLE_HIT_RADIUS = { corner: 14, edge: 12 };
const HANDLE_TIE_EPSILON = 1e-6;
const MINIMUM_BOX_SIZE = 1;
const MINIMUM_VIEW_SIZE = 24;
const WHEEL_ZOOM_SENSITIVITY = 0.002;

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

function handleCursor(handle) {
  if (handle === 'n' || handle === 's') return 'ns';
  if (handle === 'e' || handle === 'w') return 'ew';
  if (handle === 'ne' || handle === 'sw') return 'nesw';
  return 'nwse';
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
  onViewChange = () => {},
}) {
  if (!(svg instanceof SVGSVGElement)) throw new TypeError('svg must be an SVGSVGElement');
  if (![onGesture, onSelection, onMessage, onViewChange].every(callback => typeof callback === 'function')) {
    throw new TypeError('editor callbacks must be functions');
  }

  const image = element('image', { preserveAspectRatio: 'none' });
  image.classList.add('editor-image');
  const regionsLayer = element('g', { 'aria-label': 'Bounding box regions' });
  regionsLayer.classList.add('editor-regions');
  const previewLayer = element('g', { 'aria-hidden': 'true' });
  previewLayer.classList.add('editor-preview-layer');
  const selectionLayer = element('g', {
    'aria-hidden': 'true',
    'pointer-events': 'none',
  });
  selectionLayer.classList.add('editor-selection-layer');
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
  svg.replaceChildren(
    image,
    regionsLayer,
    previewLayer,
    selectionLayer,
    horizontalGuide,
    verticalGuide,
  );
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
    hoverHandle: null,
    temporaryPan: false,
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
    syncResizeAffordance(state.pointerClient);
    onViewChange(getViewState());
  }

  function getViewState() {
    if (!hasTask()) return null;
    const zoom = state.width / state.viewBox.width;
    return {
      viewBox: copyRect(state.viewBox),
      zoom,
      fit: Math.abs(zoom - 1) <= 1e-9,
    };
  }

  function naturalUnitsForPixels(pixels) {
    const bounds = svg.getBoundingClientRect();
    if (!state.viewBox || bounds.width <= 0 || bounds.height <= 0) return pixels;
    return Math.max(
      pixels * state.viewBox.width / bounds.width,
      pixels * state.viewBox.height / bounds.height,
    );
  }

  function naturalToClient(point) {
    const matrix = svg.getScreenCTM();
    if (matrix) {
      return new DOMPoint(point.x, point.y).matrixTransform(matrix);
    }
    const bounds = svg.getBoundingClientRect();
    if (!state.viewBox || bounds.width <= 0 || bounds.height <= 0) return point;
    return naturalToClientPoint(point, bounds, state.viewBox);
  }

  function selectedHandleNear(event) {
    if (
      state.mode !== 'select' || state.disabled || !hasTask() ||
      state.selected === null || state.gesture
    ) return null;
    const rect = state.rectangles.get(state.selected);
    if (!rect || state.hiddenRegionKeys.has(state.selected)) return null;
    let best = null;
    for (const handle of HANDLES) {
      const center = naturalToClient(handleCenter(rect, handle));
      const distanceSquared =
        (event.clientX - center.x) ** 2 + (event.clientY - center.y) ** 2;
      const corner = CORNER_HANDLES.has(handle);
      const radius = corner ? HANDLE_HIT_RADIUS.corner : HANDLE_HIT_RADIUS.edge;
      if (distanceSquared > radius ** 2) continue;
      if (
        best === null ||
        distanceSquared < best.distanceSquared - HANDLE_TIE_EPSILON ||
        (
          Math.abs(distanceSquared - best.distanceSquared) <= HANDLE_TIE_EPSILON &&
          corner && !best.corner
        )
      ) {
        best = { handle, distanceSquared, corner };
      }
    }
    return best?.handle || null;
  }

  function renderSelectionAffordance() {
    selectionLayer.replaceChildren();
    if (!hasTask() || state.selected === null) return;
    const rect = state.rectangles.get(state.selected);
    if (!rect || state.hiddenRegionKeys.has(state.selected)) return;
    const handleSize = naturalUnitsForPixels(10);
    const outline = element('rect', {
      class: 'editor-selection-outline',
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    });
    selectionLayer.append(outline);
    for (const handle of HANDLES) {
      const center = handleCenter(rect, handle);
      selectionLayer.append(element('rect', {
        class: `editor-handle${state.hoverHandle === handle ? ' is-hovered' : ''}`,
        x: center.x - handleSize / 2,
        y: center.y - handleSize / 2,
        width: handleSize,
        height: handleSize,
        'data-region-key': state.selected,
        'data-editor-part': 'handle',
        'data-handle': handle,
        'pointer-events': 'none',
      }));
    }
  }

  function syncResizeAffordance(event = null) {
    const handle = state.gesture?.type === 'resize'
      ? state.gesture.handle
      : event ? selectedHandleNear(event) : null;
    if (handle) svg.setAttribute('data-resize-handle', handleCursor(handle));
    else svg.removeAttribute('data-resize-handle');
    if (handle === state.hoverHandle) return;
    state.hoverHandle = handle;
    renderSelectionAffordance();
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
    if (!hasTask()) {
      renderSelectionAffordance();
      return;
    }
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
      group.append(visual, hitTarget, label);
      regionsLayer.append(group);
    }
    renderSelectionAffordance();
  }

  function select(regionKey, notify = true) {
    const next = state.rectangles.has(regionKey) ? regionKey : null;
    if (next === state.selected) return;
    state.hoverHandle = null;
    svg.removeAttribute('data-resize-handle');
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
    state.hoverHandle = null;
    svg.removeAttribute('data-resize-handle');
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
    return clientToNaturalPoint({ x: event.clientX, y: event.clientY }, bounds, state.viewBox);
  }

  function pointerInsideViewBox(event) {
    const point = clientToNatural(event);
    return point.x >= state.viewBox.x && point.x <= state.viewBox.x + state.viewBox.width &&
      point.y >= state.viewBox.y && point.y <= state.viewBox.y + state.viewBox.height;
  }

  function wheelDeltaPixels(event, delta, axis) {
    if (event.deltaMode === 1) return delta * 16;
    if (event.deltaMode === 2) {
      const bounds = svg.getBoundingClientRect();
      return delta * (axis === 'x' ? bounds.width : bounds.height);
    }
    return delta;
  }

  function naturalUnitsForSignedPixels(pixels) {
    return Math.sign(pixels) * naturalUnitsForPixels(Math.abs(pixels));
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
    syncResizeAffordance(event);
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
    syncResizeAffordance(null);
  }

  function beginGesture(event) {
    const temporaryPan = state.temporaryPan || event.button === 1;
    if (state.disabled || !hasTask() || (!temporaryPan && event.button !== 0)) return;
    svg.focus({ preventScroll: true });
    const target = event.target instanceof Element ? event.target : null;
    const regionKey = target?.getAttribute('data-region-key');
    const point = clientToNatural(event);
    const selectedHandle = selectedHandleNear(event);

    if (temporaryPan || state.mode === 'pan') {
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
    } else if (state.mode === 'draw') {
      if (!validCategory(state.category)) {
        message('category_required', 'Choose one canonical COCO-80 category before drawing.', 'warning');
        return;
      }
      state.gesture = { type: 'draw', pointerId: event.pointerId, start: point, preview: null };
    } else if (selectedHandle && state.selected !== null) {
      const original = copyRect(state.rectangles.get(state.selected));
      state.gesture = {
        type: 'resize',
        pointerId: event.pointerId,
        regionKey: state.selected,
        handle: selectedHandle,
        start: point,
        original,
        preview: original,
      };
    } else if (regionKey && state.rectangles.has(regionKey)) {
      select(regionKey);
      const original = copyRect(state.rectangles.get(regionKey));
      state.gesture = {
        type: 'move',
        pointerId: event.pointerId,
        regionKey,
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
    syncResizeAffordance(event);
  }

  function moveGesture(event) {
    const gesture = state.gesture;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    if (gesture.type === 'pan') {
      let point;
      if (gesture.inverseMatrix) {
        point = new DOMPoint(event.clientX, event.clientY).matrixTransform(gesture.inverseMatrix);
      } else {
        point = clientToNaturalPoint(
          { x: event.clientX, y: event.clientY },
          gesture.bounds,
          gesture.viewBox,
        );
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
    syncResizeAffordance(event);
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
    syncResizeAffordance(state.pointerClient);
    message('gesture_cancelled', 'Gesture cancelled.', 'info');
  }

  svg.addEventListener('pointerdown', event => {
    trackPointer(event);
    beginGesture(event);
    renderPointerGuides();
  });
  svg.addEventListener('wheel', event => {
    if (state.disabled || !hasTask() || !pointerInsideViewBox(event)) return;
    if (!event.metaKey && (event.ctrlKey || event.altKey)) return;

    const previous = state.viewBox;
    let next;
    if (event.metaKey) {
      if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return;
      const deltaPixels = wheelDeltaPixels(event, event.deltaY, 'y');
      const factor = Math.min(1.5, Math.max(
        2 / 3,
        Math.exp(-deltaPixels * WHEEL_ZOOM_SENSITIVITY),
      ));
      next = zoomViewBox(
        previous,
        factor,
        clientToNatural(event),
        state.width,
        state.height,
        MINIMUM_VIEW_SIZE,
      );
    } else if (event.shiftKey) {
      const delta = Number.isFinite(event.deltaX) && event.deltaX !== 0
        ? event.deltaX
        : event.deltaY;
      if (!Number.isFinite(delta) || delta === 0) return;
      const deltaNatural = naturalUnitsForSignedPixels(wheelDeltaPixels(event, delta, 'x'));
      next = panViewBox(previous, deltaNatural, 0, state.width, state.height);
    } else {
      if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return;
      const deltaNatural = naturalUnitsForSignedPixels(
        wheelDeltaPixels(event, event.deltaY, 'y'),
      );
      next = panViewBox(previous, 0, deltaNatural, state.width, state.height);
    }

    if (rectanglesEqual(previous, next)) return;
    event.preventDefault();
    state.viewBox = next;
    applyViewBox();
  }, { passive: false });
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
      state.temporaryPan = false;
      svg.classList.remove('is-temporary-pan');
      svg.removeAttribute('aria-keyshortcuts');
      state.pointerClient = null;
      state.hoverHandle = null;
      svg.removeAttribute('data-resize-handle');
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
        selectionLayer.replaceChildren();
        svg.setAttribute('viewBox', '0 0 1000 1000');
        onViewChange(null);
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
      syncResizeAffordance(state.pointerClient);
    },

    setCategory(category) {
      state.category = validCategory(category) ? { id: category.id, name: category.name } : null;
    },

    setDisabled(disabled) {
      state.disabled = Boolean(disabled);
      if (state.disabled) {
        cancelGesture();
        state.temporaryPan = false;
        svg.classList.remove('is-temporary-pan');
        svg.removeAttribute('aria-keyshortcuts');
      }
      svg.classList.toggle('is-disabled', state.disabled);
      svg.setAttribute('aria-disabled', String(state.disabled));
      renderPointerGuides();
      syncResizeAffordance(state.pointerClient);
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

    getViewState,

    hasActiveGesture() {
      return Boolean(state.gesture);
    },

    setTemporaryPan(active) {
      state.temporaryPan = Boolean(active);
      svg.classList.toggle('is-temporary-pan', state.temporaryPan);
      if (state.temporaryPan) svg.setAttribute('aria-keyshortcuts', 'Space');
      else svg.removeAttribute('aria-keyshortcuts');
    },

    focusSelected(paddingRatio = 0.15) {
      if (!hasTask() || state.selected === null) return false;
      const rect = state.rectangles.get(state.selected);
      if (!rect) return false;
      state.viewBox = focusViewBox(
        rect,
        state.width,
        state.height,
        paddingRatio,
        MINIMUM_VIEW_SIZE,
      );
      applyViewBox();
      return true;
    },

    refreshViewport() {
      if (hasTask()) applyViewBox();
    },

    zoomIn() {
      if (!hasTask()) return;
      const anchor = {
        x: state.viewBox.x + state.viewBox.width / 2,
        y: state.viewBox.y + state.viewBox.height / 2,
      };
      state.viewBox = zoomViewBox(
        state.viewBox, 1.25, anchor, state.width, state.height, MINIMUM_VIEW_SIZE,
      );
      applyViewBox();
    },

    zoomOut() {
      if (!hasTask()) return;
      const anchor = {
        x: state.viewBox.x + state.viewBox.width / 2,
        y: state.viewBox.y + state.viewBox.height / 2,
      };
      state.viewBox = zoomViewBox(
        state.viewBox, 0.8, anchor, state.width, state.height, MINIMUM_VIEW_SIZE,
      );
      applyViewBox();
    },

    reset() {
      if (!hasTask()) return;
      cancelGesture();
      state.viewBox = resetViewBox(state.width, state.height);
      applyViewBox();
    },
  };
}
