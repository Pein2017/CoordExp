import { ApiError } from './api-client.js';

const MAX_UNDO_SNAPSHOTS = 50;

export function createDraftController({ api, onChange, randomUUID } = {}) {
  if (!api || typeof api.postJson !== 'function' || typeof api.putJson !== 'function') {
    throw new TypeError('api must provide postJson and putJson');
  }
  const uuid = randomUUID || globalThis.crypto?.randomUUID?.bind(globalThis.crypto);
  if (typeof uuid !== 'function') throw new TypeError('randomUUID is required');
  const listeners = new Set();
  if (onChange !== undefined) listeners.add(requireListener(onChange));

  let task = null;
  let objects = [];
  let binding = null;
  let authority = 'committed';
  let version = 0;
  let durableVersion = 0;
  let pendingAttempt = null;
  let pumpPromise = null;
  let projectionPromise = null;
  let error = null;
  let conflict = null;
  let phase = 'Closed';
  let undoHistory = [];

  const emit = () => {
    const snapshot = getState();
    for (const listener of listeners) listener(snapshot);
  };

  const getState = () => ({
    phase,
    task: task ? clone(task) : null,
    binding: binding ? clone(binding) : null,
    objects: clone(objects),
    authority,
    dirty: version > durableVersion,
    error,
    conflict: conflict ? clone(conflict) : null,
    canUndo: canUndo(),
  });

  const canUndo = () => Boolean(
    undoHistory.length > 0
    && !pendingAttempt
    && !pumpPromise
    && !projectionPromise
    && !error
    && !conflict
    && version <= durableVersion,
  );

  const hasUnsavedLocal = () => Boolean(
    version > durableVersion || pendingAttempt || pumpPromise || projectionPromise || error || conflict,
  );

  const open = (authoritative, { discardUnsaved = false } = {}) => {
    if (pumpPromise || projectionPromise) {
      throw new Error('cannot replace a task while an operation is in flight');
    }
    if (hasUnsavedLocal() && !discardUnsaved) {
      throw new Error('cannot open another task while local state is unresolved');
    }
    const normalized = normalizeTask(authoritative);
    task = normalized.task;
    objects = normalized.objects;
    binding = normalized.binding;
    authority = normalized.authority;
    version = 0;
    durableVersion = 0;
    pendingAttempt = null;
    error = null;
    conflict = null;
    undoHistory = [];
    phase = authority === 'draft' ? 'Draft' : 'Committed';
    emit();
    return getState();
  };

  const rebind = (authoritative) => {
    requireOpen();
    if (pumpPromise || projectionPromise || hasUnsavedLocal()) {
      throw new Error('cannot rebind a task while local state is unresolved');
    }
    const normalized = normalizeTask(authoritative);
    if (normalized.task.split !== task.split || normalized.task.task_id !== task.task_id) {
      throw new Error('authority rebind must preserve task identity');
    }
    if (!sameSemanticsWithAuthorityIdentityEnrichment(objects, normalized.objects)) {
      throw new Error('authority rebind changed task semantics');
    }
    const authoritativeIds = new Map(normalized.objects
      .filter(object => Number.isInteger(object.coco_ann_id) && object.coco_ann_id !== 0)
      .map(object => [object.region_key, object.coco_ann_id]));
    undoHistory = undoHistory.map(snapshot => snapshot.map(object => {
      const cocoAnnId = authoritativeIds.get(object.region_key);
      return cocoAnnId === undefined ? clone(object) : { ...clone(object), coco_ann_id: cocoAnnId };
    }));
    task = normalized.task;
    objects = normalized.objects;
    binding = normalized.binding;
    authority = normalized.authority;
    version = 0;
    durableVersion = 0;
    pendingAttempt = null;
    error = null;
    conflict = null;
    phase = authority === 'draft' ? 'Draft' : 'Committed';
    emit();
    return getState();
  };

  const replaceObjects = (nextObjects, { recordUndo = true } = {}) => {
    requireOpen();
    if (conflict) throw new Error('resolve the authoritative conflict before editing');
    if (!Array.isArray(nextObjects)) throw new TypeError('objects must be an array');
    const next = clone(nextObjects);
    if (recordUndo) pushUndoSnapshot(objects);
    objects = next;
    version += 1;
    phase = error ? 'Error' : 'Saving';
    emit();
    schedulePump();
    return getState();
  };

  const undo = async () => {
    requireOpen();
    if (!canUndo()) throw new Error('Undo is unavailable while local state is unresolved');
    const previous = undoHistory.pop();
    replaceObjects(previous, { recordUndo: false });
    await flush();
    return getState();
  };

  const replaceRegion = (regionKey, replacement) => {
    requireOpen();
    const index = objects.findIndex((item) => item.region_key === regionKey);
    if (index < 0) throw new Error('region key is not present in local objects');
    const next = clone(objects);
    const value = typeof replacement === 'function'
      ? replacement(clone(next[index]))
      : replacement;
    if (!value || value.region_key !== regionKey) {
      throw new Error('replacement must preserve the stable region key');
    }
    next[index] = clone(value);
    return replaceObjects(next);
  };

  const projectAndApply = async (request) => {
    requireOpen();
    if (projectionPromise) throw new Error('an object projection is already running');
    projectionPromise = (async () => {
      await flush();
      const operation = request?.operation;
      if (operation !== 'create' && operation !== 'update') {
        throw new TypeError('projection operation must be create or update');
      }
      const body = {
        operation,
        pixel_xyxy: clone(request.pixel_xyxy),
        category_name: request.category_name,
        expected_revision: binding.revision,
        expected_generation: binding.generation,
        expected_base_row_hash: binding.base_row_hash,
        ...(operation === 'create'
          ? { request_id: `projection:${uuid()}` }
          : { region_key: request.region_key }),
      };
      let projection;
      try {
        projection = await api.postJson(objectProjectionPath(task), body);
      } catch (caught) {
        recordProjectionConflict(caught);
        throw caught;
      }
      assertProjectionBinding(projection?.binding, binding);
      const next = clone(objects);
      if (operation === 'create') {
        next.push(clone(projection.object));
      } else {
        const index = next.findIndex((item) => item.region_key === request.region_key);
        if (index < 0 || projection.object?.region_key !== request.region_key) {
          throw new Error('projection response changed update identity');
        }
        next[index] = clone(projection.object);
      }
      replaceObjects(next);
      await flush();
      return clone(projection.object);
    })();
    try {
      return await projectionPromise;
    } finally {
      projectionPromise = null;
      emit();
    }
  };

  const flush = async () => {
    requireOpen();
    while (true) {
      if (conflict) throw new ApiError('Draft authority is in conflict', { status: 409, body: conflict });
      if (error) throw error;
      if (version <= durableVersion && !pendingAttempt && !pumpPromise) return getState();
      ensurePump();
      if (pumpPromise) await pumpPromise;
    }
  };

  const retry = async () => {
    requireOpen();
    if (conflict) throw new ApiError('Draft authority is in conflict', { status: 409, body: conflict });
    if (!error || !pendingAttempt) return flush();
    error = null;
    phase = 'Saving';
    emit();
    ensurePump();
    return flush();
  };

  const ensurePump = () => {
    if (pumpPromise || error || conflict || (!pendingAttempt && version <= durableVersion)) return;
    pumpPromise = runPump().finally(() => {
      pumpPromise = null;
      emit();
      if (!error && !conflict && version > durableVersion) schedulePump();
    });
  };

  const schedulePump = () => queueMicrotask(ensurePump);

  const runPump = async () => {
    while (!error && !conflict && (pendingAttempt || version > durableVersion)) {
      if (!pendingAttempt) pendingAttempt = captureAttempt();
      phase = 'Saving';
      emit();
      try {
        const response = await api.putJson(draftPath(task), pendingAttempt.body);
        applySaveResponse(response, pendingAttempt.version);
        pendingAttempt = null;
      } catch (caught) {
        if (caught instanceof ApiError && caught.status === 409) {
          conflict = caught.body || { error: { message: caught.message } };
          error = caught;
          pendingAttempt = null;
          phase = 'Conflict';
          emit();
          return;
        }
        if (isAmbiguous(caught) && pendingAttempt.replays < 1) {
          pendingAttempt.replays += 1;
          continue;
        }
        error = caught instanceof Error ? caught : new Error(String(caught));
        phase = 'Error';
        emit();
        return;
      }
    }
  };

  const captureAttempt = () => ({
    version,
    replays: 0,
    body: {
      mutation_id: `draft:${uuid()}`,
      expected_revision: binding.revision,
      expected_generation: binding.generation,
      expected_base_row_hash: binding.base_row_hash,
      objects: clone(objects),
    },
  });

  const applySaveResponse = (response, savedVersion) => {
    if (response?.status !== 'applied' || !Array.isArray(response.objects)) {
      throw new Error('Draft save response is not authoritative');
    }
    binding = bindingFrom(response);
    authority = response.authority;
    durableVersion = Math.max(durableVersion, savedVersion);
    if (version === savedVersion) objects = clone(response.objects);
    phase = version > durableVersion ? 'Saving' : authority === 'draft' ? 'Draft' : 'Committed';
    error = null;
    emit();
  };

  const recordProjectionConflict = (caught) => {
    if (caught instanceof ApiError && caught.status === 409) {
      conflict = caught.body || { error: { message: caught.message } };
      error = caught;
      phase = 'Conflict';
      emit();
    }
  };

  const requireOpen = () => {
    if (!task || !binding) throw new Error('no authoritative task is open');
  };

  const pushUndoSnapshot = (snapshot) => {
    undoHistory.push(clone(snapshot));
    if (undoHistory.length > MAX_UNDO_SNAPSHOTS) undoHistory.shift();
  };

  return Object.freeze({
    open,
    rebind,
    getState,
    onChange(listener) {
      listeners.add(requireListener(listener));
      return () => listeners.delete(listener);
    },
    replaceObjects,
    replaceRegion,
    projectAndApply,
    undo,
    flush,
    retry,
    hasUnsavedLocal,
  });
}

function normalizeTask(value) {
  if (!value || typeof value !== 'object' || !Array.isArray(value.objects)) {
    throw new TypeError('authoritative task response is invalid');
  }
  if (typeof value.split !== 'string' || typeof value.task_id !== 'string') {
    throw new TypeError('authoritative task identity is invalid');
  }
  return {
    task: { split: value.split, task_id: value.task_id },
    objects: clone(value.objects),
    binding: bindingFrom(value),
    authority: value.authority === 'draft' ? 'draft' : 'committed',
  };
}

function bindingFrom(value) {
  const binding = {
    revision: value.revision,
    generation: value.generation,
    base_row_hash: value.base_row_hash,
  };
  if (!Number.isInteger(binding.revision) || !Number.isInteger(binding.generation)
      || typeof binding.base_row_hash !== 'string') {
    throw new TypeError('authoritative task binding is invalid');
  }
  return binding;
}

function assertProjectionBinding(actual, expected) {
  if (!actual || actual.revision !== expected.revision
      || actual.generation !== expected.generation
      || actual.base_row_hash !== expected.base_row_hash) {
    throw new Error('object projection response binding is stale');
  }
}

function draftPath(task) {
  return `/api/splits/${encodeURIComponent(task.split)}/tasks/${encodeURIComponent(task.task_id)}/draft`;
}

function objectProjectionPath(task) {
  return `/api/splits/${encodeURIComponent(task.split)}/tasks/${encodeURIComponent(task.task_id)}/objects/canonicalize`;
}

function isAmbiguous(error) {
  return !(error instanceof ApiError) || error.status === 0 || error.status >= 500;
}

function requireListener(value) {
  if (typeof value !== 'function') throw new TypeError('listener must be a function');
  return value;
}

function clone(value) {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

function sameSemanticsWithAuthorityIdentityEnrichment(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
  return left.every((value, index) => {
    const current = clone(value);
    const authoritative = clone(right[index]);
    const currentId = Number.isInteger(current.coco_ann_id) && current.coco_ann_id !== 0
      ? current.coco_ann_id : null;
    const authoritativeId = Number.isInteger(authoritative.coco_ann_id)
      && authoritative.coco_ann_id !== 0 ? authoritative.coco_ann_id : null;
    delete current.coco_ann_id;
    delete authoritative.coco_ann_id;
    return JSON.stringify(current) === JSON.stringify(authoritative)
      && (currentId === null || currentId === authoritativeId);
  });
}
