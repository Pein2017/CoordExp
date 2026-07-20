/**
 * Pure presentation geometry for the standalone bbox editor.
 *
 * These helpers deliberately know nothing about Drafts, persistence, DOM state,
 * or annotation identity. Natural-image coordinates use the half-open extent
 * [0, width] x [0, height]. Persisted norm1000 bins use inclusive endpoints
 * 0..999, hence the explicit /999 projection.
 */

const HANDLES = new Set(['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']);

function finiteNumber(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
  return value;
}

function positiveNumber(value, name) {
  finiteNumber(value, name);
  if (value <= 0) throw new RangeError(`${name} must be greater than zero`);
  return value;
}

function clamp(value, lower, upper) {
  return Math.min(upper, Math.max(lower, value));
}

function point(value, name) {
  if (!value || typeof value !== 'object') throw new TypeError(`${name} must be a point`);
  return {
    x: finiteNumber(value.x, `${name}.x`),
    y: finiteNumber(value.y, `${name}.y`),
  };
}

function rectangle(value, name) {
  if (!value || typeof value !== 'object') throw new TypeError(`${name} must be a rectangle`);
  return {
    x: finiteNumber(value.x, `${name}.x`),
    y: finiteNumber(value.y, `${name}.y`),
    width: positiveNumber(value.width, `${name}.width`),
    height: positiveNumber(value.height, `${name}.height`),
  };
}

function viewport(value, name = 'bounds') {
  if (!value || typeof value !== 'object') throw new TypeError(`${name} must be viewport bounds`);
  return {
    left: finiteNumber(value.left, `${name}.left`),
    top: finiteNumber(value.top, `${name}.top`),
    width: positiveNumber(value.width, `${name}.width`),
    height: positiveNumber(value.height, `${name}.height`),
  };
}

export function validateNaturalExtent(width, height) {
  return {
    width: positiveNumber(width, 'width'),
    height: positiveNumber(height, 'height'),
  };
}

function validateRect(rect, extent, name = 'rect') {
  const value = rectangle(rect, name);
  if (
    value.x < 0 || value.y < 0 ||
    value.x + value.width > extent.width ||
    value.y + value.height > extent.height
  ) {
    throw new RangeError(`${name} must fit inside the natural image extent`);
  }
  return value;
}

export function norm1000ToNaturalRect(bbox, naturalWidth, naturalHeight) {
  const extent = validateNaturalExtent(naturalWidth, naturalHeight);
  if (!Array.isArray(bbox) || bbox.length !== 4) {
    throw new TypeError('bbox must be [x1, y1, x2, y2]');
  }
  const [x1, y1, x2, y2] = bbox.map((value, index) => finiteNumber(value, `bbox[${index}]`));
  if (x1 < 0 || y1 < 0 || x2 > 999 || y2 > 999 || x2 <= x1 || y2 <= y1) {
    throw new RangeError('bbox must be ordered, nondegenerate, and within 0..999');
  }
  const left = x1 * extent.width / 999;
  const top = y1 * extent.height / 999;
  const right = x2 * extent.width / 999;
  const bottom = y2 * extent.height / 999;
  return { x: left, y: top, width: right - left, height: bottom - top };
}

export function orderedClippedDragRect(start, end, naturalWidth, naturalHeight) {
  const extent = validateNaturalExtent(naturalWidth, naturalHeight);
  const a = point(start, 'start');
  const b = point(end, 'end');
  const ax = clamp(a.x, 0, extent.width);
  const ay = clamp(a.y, 0, extent.height);
  const bx = clamp(b.x, 0, extent.width);
  const by = clamp(b.y, 0, extent.height);
  const x = Math.min(ax, bx);
  const y = Math.min(ay, by);
  return { x, y, width: Math.abs(bx - ax), height: Math.abs(by - ay) };
}

export function moveRect(rect, deltaX, deltaY, naturalWidth, naturalHeight) {
  const extent = validateNaturalExtent(naturalWidth, naturalHeight);
  const value = validateRect(rect, extent);
  finiteNumber(deltaX, 'deltaX');
  finiteNumber(deltaY, 'deltaY');
  return {
    x: clamp(value.x + deltaX, 0, extent.width - value.width),
    y: clamp(value.y + deltaY, 0, extent.height - value.height),
    width: value.width,
    height: value.height,
  };
}

export function resizeRect(
  rect,
  handle,
  deltaX,
  deltaY,
  naturalWidth,
  naturalHeight,
  minimumSize = 1,
) {
  const extent = validateNaturalExtent(naturalWidth, naturalHeight);
  const value = validateRect(rect, extent);
  if (!HANDLES.has(handle)) throw new RangeError('handle must be one of n, ne, e, se, s, sw, w, nw');
  finiteNumber(deltaX, 'deltaX');
  finiteNumber(deltaY, 'deltaY');
  positiveNumber(minimumSize, 'minimumSize');
  if (minimumSize > extent.width || minimumSize > extent.height) {
    throw new RangeError('minimumSize must fit inside the natural image extent');
  }

  let left = value.x;
  let top = value.y;
  let right = value.x + value.width;
  let bottom = value.y + value.height;
  if (handle.includes('w')) left = clamp(left + deltaX, 0, right - minimumSize);
  if (handle.includes('e')) right = clamp(right + deltaX, left + minimumSize, extent.width);
  if (handle.includes('n')) top = clamp(top + deltaY, 0, bottom - minimumSize);
  if (handle.includes('s')) bottom = clamp(bottom + deltaY, top + minimumSize, extent.height);
  return { x: left, y: top, width: right - left, height: bottom - top };
}

export function resetViewBox(naturalWidth, naturalHeight) {
  const extent = validateNaturalExtent(naturalWidth, naturalHeight);
  return { x: 0, y: 0, width: extent.width, height: extent.height };
}

function containTransform(bounds, viewBox) {
  const frame = viewport(bounds);
  const view = rectangle(viewBox, 'viewBox');
  const scale = Math.min(frame.width / view.width, frame.height / view.height);
  return {
    frame,
    view,
    scale,
    offsetX: (frame.width - view.width * scale) / 2,
    offsetY: (frame.height - view.height * scale) / 2,
  };
}

export function clientToNaturalPoint(clientPoint, bounds, viewBox) {
  const client = point(clientPoint, 'clientPoint');
  const transform = containTransform(bounds, viewBox);
  return {
    x: transform.view.x +
      (client.x - transform.frame.left - transform.offsetX) / transform.scale,
    y: transform.view.y +
      (client.y - transform.frame.top - transform.offsetY) / transform.scale,
  };
}

export function naturalToClientPoint(naturalPoint, bounds, viewBox) {
  const natural = point(naturalPoint, 'naturalPoint');
  const transform = containTransform(bounds, viewBox);
  return {
    x: transform.frame.left + transform.offsetX +
      (natural.x - transform.view.x) * transform.scale,
    y: transform.frame.top + transform.offsetY +
      (natural.y - transform.view.y) * transform.scale,
  };
}

export function focusViewBox(
  rect,
  naturalWidth,
  naturalHeight,
  paddingRatio = 0.15,
  minimumViewSize = 1,
) {
  const extent = validateNaturalExtent(naturalWidth, naturalHeight);
  const target = validateRect(rect, extent);
  finiteNumber(paddingRatio, 'paddingRatio');
  if (paddingRatio < 0) throw new RangeError('paddingRatio must be zero or greater');
  positiveNumber(minimumViewSize, 'minimumViewSize');
  const paddedWidth = target.width * (1 + 2 * paddingRatio);
  const paddedHeight = target.height * (1 + 2 * paddingRatio);
  const scale = Math.min(1, Math.max(
    paddedWidth / extent.width,
    paddedHeight / extent.height,
    minimumViewSize / extent.width,
    minimumViewSize / extent.height,
  ));
  const width = extent.width * scale;
  const height = extent.height * scale;
  const centerX = target.x + target.width / 2;
  const centerY = target.y + target.height / 2;
  return {
    x: clamp(centerX - width / 2, 0, extent.width - width),
    y: clamp(centerY - height / 2, 0, extent.height - height),
    width,
    height,
  };
}

export function panViewBox(viewBox, deltaX, deltaY, naturalWidth, naturalHeight) {
  const extent = validateNaturalExtent(naturalWidth, naturalHeight);
  const value = validateRect(viewBox, extent, 'viewBox');
  finiteNumber(deltaX, 'deltaX');
  finiteNumber(deltaY, 'deltaY');
  return {
    x: clamp(value.x + deltaX, 0, extent.width - value.width),
    y: clamp(value.y + deltaY, 0, extent.height - value.height),
    width: value.width,
    height: value.height,
  };
}

export function zoomViewBox(
  viewBox,
  zoomFactor,
  anchor,
  naturalWidth,
  naturalHeight,
  minimumViewSize = 1,
) {
  const extent = validateNaturalExtent(naturalWidth, naturalHeight);
  const value = validateRect(viewBox, extent, 'viewBox');
  positiveNumber(zoomFactor, 'zoomFactor');
  positiveNumber(minimumViewSize, 'minimumViewSize');
  const focus = point(anchor, 'anchor');
  const anchorX = clamp(focus.x, 0, extent.width);
  const anchorY = clamp(focus.y, 0, extent.height);
  const minimumScale = Math.min(
    1,
    Math.max(minimumViewSize / value.width, minimumViewSize / value.height),
  );
  const maximumScale = Math.min(
    extent.width / value.width,
    extent.height / value.height,
  );
  const scale = clamp(1 / zoomFactor, minimumScale, maximumScale);
  const width = value.width * scale;
  const height = value.height * scale;
  const relativeX = (anchorX - value.x) / value.width;
  const relativeY = (anchorY - value.y) / value.height;
  return {
    x: clamp(anchorX - relativeX * width, 0, extent.width - width),
    y: clamp(anchorY - relativeY * height, 0, extent.height - height),
    width,
    height,
  };
}
