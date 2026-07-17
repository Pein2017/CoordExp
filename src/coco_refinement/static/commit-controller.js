const ACTIVE_BATCH_KEY = 'coco-refinement.commit.active.v1';
const ACTIVE_STATUSES = new Set(['queued', 'running', 'reconciling']);
const RECEIPT_STATUSES = new Set([...ACTIVE_STATUSES, 'succeeded', 'failed']);
const BATCH_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$/;
const GENERIC_FAILURE = 'Commit failed; inspect the local durable receipt before retrying.';

export function createCommitController({
  api,
  onChange,
  scheduler = defaultScheduler(),
  storage = globalThis.localStorage,
  randomUUID = globalThis.crypto?.randomUUID?.bind(globalThis.crypto),
  pollIntervalMs = 1000,
} = {}) {
  if (!api || typeof api.getJson !== 'function' || typeof api.postJson !== 'function') {
    throw new TypeError('api must provide getJson and postJson');
  }
  if (!storage || typeof storage.getItem !== 'function'
      || typeof storage.setItem !== 'function' || typeof storage.removeItem !== 'function') {
    throw new TypeError('storage must implement the localStorage interface');
  }
  if (!scheduler || typeof scheduler.setTimeout !== 'function'
      || typeof scheduler.clearTimeout !== 'function') {
    throw new TypeError('scheduler must provide setTimeout and clearTimeout');
  }
  if (typeof randomUUID !== 'function') throw new TypeError('randomUUID is required');
  if (!Number.isFinite(pollIntervalMs) || pollIntervalMs < 0) {
    throw new TypeError('pollIntervalMs must be non-negative');
  }

  const listeners = new Set();
  if (onChange !== undefined) listeners.add(requireListener(onChange));

  let split = null;
  let status = 'idle';
  let batch = null;
  let projectState = null;
  let detail = null;
  let projectError = null;
  let busy = false;
  let pollTimer = null;
  let epoch = 0;

  const getState = () => ({
    split,
    status,
    batch: batch ? clone(batch) : null,
    projectState: projectState ? clone(projectState) : null,
    detail,
    projectError,
    busy,
    polling: pollTimer !== null,
    canCommit: Boolean(split) && !busy && !ACTIVE_STATUSES.has(status) && status !== 'failed',
    canRetry: Boolean(split && batch) && status === 'failed' && !busy,
  });

  const emit = () => {
    for (const listener of listeners) listener(getState());
  };

  const setSplit = async (nextSplit) => {
    requireSplit(nextSplit);
    const operationEpoch = resetLifecycle();
    split = nextSplit;
    status = 'idle';
    batch = null;
    projectState = null;
    detail = null;
    projectError = null;
    busy = true;
    emit();
    try {
      await refreshProjectStateFor(nextSplit, operationEpoch);
      const activeBatchId = readActiveBatch(nextSplit);
      if (activeBatchId && isCurrent(nextSplit, operationEpoch)) {
        await recoverPersistedBatch(nextSplit, activeBatchId, operationEpoch);
      }
      return getState();
    } finally {
      if (isCurrent(nextSplit, operationEpoch)) {
        busy = false;
        emit();
      }
    }
  };

  const requireOpenSplit = () => {
    if (!split) throw new Error('no Commit split is open');
    return split;
  };

  const commit = async () => {
    const selected = requireOpenSplit();
    if (busy) throw new Error('a Commit controller operation is already running');
    if (ACTIVE_STATUSES.has(status)) throw new Error('a Commit batch is already active');
    if (status === 'failed' && batch) throw new Error('retry the failed Commit with its existing batch id');
    const operationEpoch = epoch;
    const batchId = newBatchId(randomUUID());
    persistActiveBatch(selected, batchId);
    batch = emptyBatch(batchId, selected);
    status = 'queued';
    detail = null;
    busy = true;
    emit();
    try {
      const receipt = await postWithLostResponseRecovery(
        selected, batchId, operationEpoch, { statusFirst: false },
      );
      if (!isCurrent(selected, operationEpoch)) return getState();
      applyReceipt(receipt, selected, batchId, operationEpoch);
      return getState();
    } catch (error) {
      if (isCurrent(selected, operationEpoch)) failLocally(error);
      throw error;
    } finally {
      if (isCurrent(selected, operationEpoch)) {
        busy = false;
        emit();
      }
    }
  };

  const retry = async () => {
    const selected = requireOpenSplit();
    if (busy) throw new Error('a Commit controller operation is already running');
    if (status !== 'failed' || !batch) throw new Error('no failed Commit is available to retry');
    const operationEpoch = epoch;
    const failedBatchId = batch.batch_id;
    busy = true;
    detail = null;
    emit();
    try {
      let batchId = failedBatchId;
      let existing = null;
      try {
        existing = await readStatus(selected, failedBatchId);
      } catch (error) {
        if (!isNotFound(error)) throw error;
      }
      if (existing && existing.status !== 'failed') {
        if (isCurrent(selected, operationEpoch)) {
          applyReceipt(existing, selected, failedBatchId, operationEpoch);
        }
        return getState();
      }
      if (existing?.status === 'failed') {
        batchId = newBatchId(randomUUID());
        if (batchId === failedBatchId) {
          throw new Error('retry requires a fresh batch identity');
        }
      }
      persistActiveBatch(selected, batchId);
      batch = emptyBatch(batchId, selected);
      status = 'queued';
      emit();
      const receipt = await postWithLostResponseRecovery(
        selected, batchId, operationEpoch, { statusFirst: false },
      );
      if (!isCurrent(selected, operationEpoch)) return getState();
      applyReceipt(receipt, selected, batchId, operationEpoch);
      return getState();
    } catch (error) {
      if (isCurrent(selected, operationEpoch)) failLocally(error);
      throw error;
    } finally {
      if (isCurrent(selected, operationEpoch)) {
        busy = false;
        emit();
      }
    }
  };

  const refreshProjectState = async () => {
    const selected = requireOpenSplit();
    await refreshProjectStateFor(selected, epoch);
    return getState();
  };

  const stop = () => {
    resetLifecycle();
    split = null;
    status = 'idle';
    batch = null;
    projectState = null;
    detail = null;
    projectError = null;
    busy = false;
    emit();
  };

  const postWithLostResponseRecovery = async (
    selected, batchId, operationEpoch, { statusFirst },
  ) => {
    if (statusFirst) {
      try {
        const existing = await readStatus(selected, batchId);
        if (existing.status !== 'failed') return existing;
      } catch (error) {
        if (!isNotFound(error)) throw error;
      }
    }
    try {
      return validateReceipt(
        await api.postJson(commitPath(selected), { batch_id: batchId }), selected, batchId,
      );
    } catch (error) {
      if (!isAmbiguous(error)) throw error;
      let recovered;
      try {
        recovered = await readStatus(selected, batchId);
      } catch (statusError) {
        if (!isNotFound(statusError)) throw statusError;
        try {
          return validateReceipt(
            await api.postJson(commitPath(selected), { batch_id: batchId }),
            selected,
            batchId,
          );
        } catch (replayError) {
          if (!isAmbiguous(replayError)) throw replayError;
          return readStatus(selected, batchId);
        }
      }
      if (!isCurrent(selected, operationEpoch)) return recovered;
      return recovered;
    }
  };

  const readStatus = async (selected, batchId) => validateReceipt(
    await api.getJson(statusPath(selected, batchId)), selected, batchId,
  );

  const recoverPersistedBatch = async (selected, batchId, operationEpoch) => {
    batch = emptyBatch(batchId, selected);
    status = 'reconciling';
    detail = 'Recovering the durable Commit status.';
    emit();
    try {
      let receipt;
      try {
        receipt = await readStatus(selected, batchId);
      } catch (error) {
        if (!isNotFound(error)) throw error;
        receipt = await replayOnceAfterConfirmedMissing(selected, batchId);
      }
      if (isCurrent(selected, operationEpoch)) {
        applyReceipt(receipt, selected, batchId, operationEpoch);
      }
    } catch (error) {
      if (isCurrent(selected, operationEpoch)) failLocally(error);
    }
  };

  const replayOnceAfterConfirmedMissing = async (selected, batchId) => {
    try {
      return validateReceipt(
        await api.postJson(commitPath(selected), { batch_id: batchId }), selected, batchId,
      );
    } catch (error) {
      if (!isAmbiguous(error)) throw error;
      return readStatus(selected, batchId);
    }
  };

  const applyReceipt = (receipt, selected, batchId, operationEpoch) => {
    const safe = validateReceipt(receipt, selected, batchId);
    batch = safe;
    status = safe.status;
    detail = safe.status === 'failed' ? GENERIC_FAILURE : null;
    if (safe.status === 'succeeded') removeActiveBatch(selected);
    if (ACTIVE_STATUSES.has(safe.status)) schedulePoll(selected, batchId, operationEpoch);
    else clearPollTimer();
    emit();
  };

  const schedulePoll = (selected, batchId, operationEpoch) => {
    clearPollTimer();
    if (!isCurrent(selected, operationEpoch) || !ACTIVE_STATUSES.has(status)) return;
    pollTimer = scheduler.setTimeout(() => {
      pollTimer = null;
      return poll(selected, batchId, operationEpoch);
    }, pollIntervalMs);
    emit();
  };

  const poll = async (selected, batchId, operationEpoch) => {
    if (!isCurrent(selected, operationEpoch)) return;
    try {
      const receipt = await readStatus(selected, batchId);
      if (!isCurrent(selected, operationEpoch)) return;
      applyReceipt(receipt, selected, batchId, operationEpoch);
      if (!ACTIVE_STATUSES.has(receipt.status)) {
        await refreshProjectStateFor(selected, operationEpoch).catch(() => undefined);
      }
    } catch (_error) {
      if (!isCurrent(selected, operationEpoch)) return;
      detail = 'Commit status is temporarily unavailable; retrying.';
      schedulePoll(selected, batchId, operationEpoch);
    }
  };

  const refreshProjectStateFor = async (selected, operationEpoch) => {
    try {
      const next = validateProjectState(await api.getJson(projectStatePath(selected)), selected);
      if (!isCurrent(selected, operationEpoch)) return;
      projectState = next;
      projectError = null;
      emit();
    } catch (error) {
      if (!isCurrent(selected, operationEpoch)) return;
      projectError = 'Project state is temporarily unavailable.';
      emit();
      throw error;
    }
  };

  const failLocally = (_error) => {
    status = 'failed';
    detail = GENERIC_FAILURE;
    clearPollTimer();
    emit();
  };

  const resetLifecycle = () => {
    epoch += 1;
    clearPollTimer();
    busy = false;
    return epoch;
  };

  const clearPollTimer = () => {
    if (pollTimer !== null) scheduler.clearTimeout(pollTimer);
    pollTimer = null;
  };

  const isCurrent = (selected, operationEpoch) => (
    split === selected && epoch === operationEpoch
  );

  const persistActiveBatch = (selected, batchId) => {
    storage.setItem(storageKey(selected), JSON.stringify({ batch_id: batchId }));
  };

  const readActiveBatch = (selected) => {
    const raw = storage.getItem(storageKey(selected));
    if (raw === null) return null;
    try {
      const value = JSON.parse(raw);
      if (!value || !isBatchId(value.batch_id)) throw new Error('invalid stored batch');
      return value.batch_id;
    } catch (_error) {
      storage.removeItem(storageKey(selected));
      return null;
    }
  };

  const removeActiveBatch = (selected) => {
    storage.removeItem(storageKey(selected));
  };

  return Object.freeze({
    setSplit,
    commit,
    retry,
    refreshProjectState,
    stop,
    getState,
    onChange(listener) {
      listeners.add(requireListener(listener));
      return () => listeners.delete(listener);
    },
  });
}

function validateReceipt(value, expectedSplit, expectedBatchId) {
  if (!value || typeof value !== 'object'
      || value.split !== expectedSplit || value.batch_id !== expectedBatchId
      || !RECEIPT_STATUSES.has(value.status)
      || !Number.isInteger(value.member_count) || value.member_count <= 0
      || !Number.isInteger(value.base_generation) || value.base_generation < 0
      || typeof value.payload_hash !== 'string' || !/^[0-9a-f]{64}$/.test(value.payload_hash)) {
    throw new Error('Commit receipt is not authoritative');
  }
  if ((value.status === 'queued' || value.status === 'running') && value.generation !== null) {
    throw new Error('Commit receipt has an invalid generation');
  }
  if (value.status === 'succeeded'
      && (!Number.isInteger(value.generation) || value.generation <= value.base_generation)) {
    throw new Error('Commit receipt has an invalid terminal generation');
  }
  if ((value.status === 'reconciling' || value.status === 'failed')
      && value.generation !== null
      && (!Number.isInteger(value.generation) || value.generation < value.base_generation)) {
    throw new Error('Commit receipt has an invalid reconciliation generation');
  }
  return {
    base_generation: value.base_generation,
    batch_id: value.batch_id,
    detail: value.status === 'failed' ? GENERIC_FAILURE : null,
    generation: value.generation,
    member_count: value.member_count,
    payload_hash: value.payload_hash,
    split: value.split,
    status: value.status,
  };
}

function validateProjectState(value, expectedSplit) {
  if (!value || typeof value !== 'object' || value.split !== expectedSplit
      || !Number.isInteger(value.generation) || value.generation < 0
      || !Number.isInteger(value.pending_draft_count) || value.pending_draft_count < 0
      || typeof value.accepting_writes !== 'boolean' || !value.worker
      || typeof value.worker !== 'object') {
    throw new Error('Project state response is not authoritative');
  }
  return clone(value);
}

function emptyBatch(batchId, split) {
  return {
    batch_id: batchId,
    split,
    status: 'queued',
    member_count: 0,
    base_generation: null,
    generation: null,
    payload_hash: null,
    detail: null,
  };
}

function newBatchId(uuid) {
  const batchId = `web:${uuid}`;
  if (!isBatchId(batchId)) throw new Error('randomUUID returned an unsafe batch identity');
  return batchId;
}

function isBatchId(value) {
  return typeof value === 'string' && BATCH_ID_PATTERN.test(value);
}

function isAmbiguous(error) {
  const status = Number(error?.status || 0);
  return status === 0 || status >= 500;
}

function isNotFound(error) {
  return Number(error?.status) === 404;
}

function requireSplit(value) {
  if (value !== 'train' && value !== 'val') throw new TypeError('split must be train or val');
  return value;
}

function requireListener(value) {
  if (typeof value !== 'function') throw new TypeError('listener must be a function');
  return value;
}

function storageKey(split) {
  return `${ACTIVE_BATCH_KEY}:${split}`;
}

function commitPath(split) {
  return `/api/splits/${encodeURIComponent(split)}/commits`;
}

function statusPath(split, batchId) {
  return `${commitPath(split)}/${encodeURIComponent(batchId)}`;
}

function projectStatePath(split) {
  return `/api/splits/${encodeURIComponent(split)}/state`;
}

function defaultScheduler() {
  return {
    setTimeout: globalThis.setTimeout.bind(globalThis),
    clearTimeout: globalThis.clearTimeout.bind(globalThis),
  };
}

function clone(value) {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}
