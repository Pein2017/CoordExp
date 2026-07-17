import { createApiClient } from '/api-client.js';
import { installCategorySearch } from '/class-search.js';
import { createCommitController } from '/commit-controller.js';
import { createDraftController } from '/draft-controller.js';
import { norm1000ToNaturalRect } from '/editor-geometry.js';
import { createSvgEditor } from '/svg-editor.js';

const state = {
  split: 'train', limit: 50, page: null, selected: -1, task: null,
  taskOpen: false, categories: [], category: null, selectedRegion: null,
  pageRequest: 0, taskRequest: 0, interactionBusy: false,
  interactionPromise: null, navigationReminder: false,
  commitActionPromise: null,
  commitRebindTarget: null, commitRebindPromise: null, commitRebindError: null,
};
const $ = id => document.getElementById(id);
const api = createApiClient();
const status = $('connection-status');
const notice = $('notice');
const editorButtons = [
  $('mode-select'), $('mode-draw'), $('mode-pan'),
  $('zoom-out'), $('zoom-in'), $('zoom-reset'),
];
const visibilityControls = [
  $('visibility-mode'), $('hide-selected'), $('restore-visibility'),
];
let categoryPicker = null;
const setNotice = (message, tone = '') => { notice.textContent = message; notice.dataset.tone = tone; };
const setStatus = (message, tone = 'pending') => { status.textContent = message; status.dataset.tone = tone; };

let controller;
let commitController;
const editor = createSvgEditor({
  svg: $('bbox-overlay'),
  onGesture: detail => { void applyEditorGesture(detail); },
  onSelection: ({ regionKey, object }) => {
    state.selectedRegion = regionKey;
    $('visibility-mode').value = editor.getPresentationState().visibilityMode;
    $('hide-selected').disabled = !regionKey || !state.taskOpen;
    $('delete-button').disabled = !regionKey || !state.taskOpen;
    if (object) {
      state.category = { id: object.category_id, name: object.category_name };
      editor.setCategory(state.category);
      categoryPicker?.choose(state.category, { notify: false });
    }
  },
  onMessage: ({ message, tone }) => setNotice(message, tone),
});
controller = createDraftController({ api, onChange: snapshot => renderDraftState(snapshot) });
commitController = createCommitController({
  api,
  onChange: snapshot => {
    renderCommitState(snapshot);
    scheduleCommitRebind(snapshot);
  },
});

function commitRebindRequired() {
  const currentGeneration = controller.getState().binding?.generation;
  return state.taskOpen && Number.isInteger(state.commitRebindTarget)
    && Number.isInteger(currentGeneration) && currentGeneration < state.commitRebindTarget;
}

const navigationBusy = () => Boolean(
  state.interactionBusy || state.commitActionPromise || state.commitRebindPromise,
);

function semanticActionLocked(snapshot = controller.getState()) {
  return state.interactionBusy || commitRebindRequired()
    || ['Saving', 'Error', 'Conflict'].includes(snapshot.phase);
}

function renderPage() {
  const page = state.page;
  $('page-summary').textContent = page && page.total ? `${page.cursor + 1}–${Math.min(page.cursor + page.tasks.length, page.total)} / ${page.total}` : page ? '0 / 0' : '—';
  $('task-list').replaceChildren(...(page?.tasks || []).map((task, index) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button'; button.disabled = navigationBusy(); button.setAttribute('aria-current', String(index === state.selected));
    button.textContent = `#${page.cursor + index + 1} · ${task.image_id}`;
    const meta = document.createElement('span'); meta.className = 'task-meta'; meta.textContent = `${task.image_width} × ${task.image_height}`;
    button.append(meta); button.addEventListener('click', () => navigate(() => selectIndex(index))); li.append(button); return li;
  }));
  $('previous-button').disabled = navigationBusy() || !page || page.cursor + state.selected <= 0;
  $('next-button').disabled = navigationBusy() || !page || page.cursor + state.selected + 1 >= page.total;
  $('split-select').disabled = navigationBusy();
  $('sample-number').disabled = navigationBusy();
  $('jump-button').disabled = navigationBusy();
  $('sample-number').value = page && state.selected >= 0 ? String(page.cursor + state.selected + 1) : '';
}

function renderTaskShell() {
  if (!state.task) {
    $('task-heading').textContent = 'No sample selected';
    $('task-counter').textContent = '—';
    $('image-placeholder').hidden = false;
    setEditorControls(false);
    return;
  }
  $('task-heading').textContent = `${state.task.split} · ${state.task.image_id}`;
  $('task-counter').textContent = `${pagePosition()} / ${state.page?.total ?? '—'}`;
  $('image-placeholder').hidden = true;
  setEditorControls(true);
}

function renderDraftState(snapshot) {
  if (state.taskOpen) editor.updateObjects(snapshot.objects);
  const visiblePhase = state.interactionBusy ? 'Saving'
    : commitRebindRequired() ? 'Refreshing authority' : snapshot.phase;
  const tone = { Committed: 'ok', Draft: 'pending', Saving: 'pending', Conflict: 'error', Error: 'error' }[visiblePhase] || 'pending';
  $('save-status').textContent = visiblePhase === 'Draft' ? 'Draft saved · Commit pending' : visiblePhase;
  $('save-status').dataset.tone = tone;
  $('save-retry').hidden = snapshot.phase !== 'Error';
  $('task-reload').hidden = !['Error', 'Conflict'].includes(snapshot.phase) && !state.commitRebindError;
  const locked = semanticActionLocked(snapshot);
  editor.setDisabled(locked);
  $('category-search').disabled = locked || !state.taskOpen;
  for (const button of editorButtons) button.disabled = locked || !state.taskOpen;
  $('undo-button').disabled = locked || !snapshot.canUndo;
  $('delete-button').disabled = locked || !state.selectedRegion;
  for (const control of visibilityControls) control.disabled = !state.taskOpen;
  $('hide-selected').disabled = !state.taskOpen || !state.selectedRegion;
  for (const button of $('task-list').querySelectorAll('button')) button.disabled = navigationBusy();
  $('split-select').disabled = navigationBusy();
  $('sample-number').disabled = navigationBusy();
  $('jump-button').disabled = navigationBusy();
  $('previous-button').disabled = navigationBusy() || !state.page || pagePosition() <= 1;
  $('next-button').disabled = navigationBusy() || !state.page || pagePosition() >= state.page.total;
  const details = $('task-details').querySelectorAll('dd');
  details[0].textContent = snapshot.authority || '—';
  details[1].textContent = snapshot.phase;
  details[2].textContent = snapshot.binding?.revision ?? '—';
  details[3].textContent = snapshot.binding?.generation ?? '—';
  details[4].textContent = state.task ? `${state.task.image_width} × ${state.task.image_height}` : '—';
}

function renderCommitState(snapshot) {
  const pending = snapshot.projectState?.pending_draft_count;
  const active = ['queued', 'running', 'reconciling'].includes(snapshot.status);
  const tone = snapshot.projectError || snapshot.status === 'failed' ? 'error'
    : snapshot.status === 'succeeded' ? 'ok' : 'pending';
  let text = snapshot.projectError || 'Commit unavailable';
  if (snapshot.batch) {
    const members = snapshot.batch.member_count ? ` · ${snapshot.batch.member_count} task${snapshot.batch.member_count === 1 ? '' : 's'}` : '';
    const generation = Number.isInteger(snapshot.batch.generation) ? ` · generation ${snapshot.batch.generation}` : '';
    text = `${snapshot.status} · ${snapshot.batch.batch_id}${members}${generation}`;
  } else if (Number.isInteger(pending)) {
    text = `${pending} pending Draft${pending === 1 ? '' : 's'}`;
  }
  if (snapshot.projectError || snapshot.detail) {
    text += ` · ${snapshot.projectError || snapshot.detail}`;
  }
  $('commit-status').textContent = text;
  $('commit-status').dataset.tone = tone;
  $('commit-button').textContent = snapshot.canRetry ? 'Retry Commit' : 'Commit Drafts';
  $('commit-button').disabled = Boolean(state.commitActionPromise) || snapshot.busy || active || (!snapshot.canRetry && (
    !snapshot.projectState || !snapshot.projectState.accepting_writes || pending === 0
  ));
}

function scheduleCommitRebind(snapshot) {
  const generation = snapshot.batch?.generation;
  if (snapshot.split !== state.split || state.task?.split !== snapshot.split
      || !state.taskOpen || !Number.isInteger(generation)) return;
  if (!['reconciling', 'succeeded'].includes(snapshot.status)) return;
  const current = controller.getState().binding?.generation;
  if (!Number.isInteger(current) || generation <= current) return;
  state.commitRebindTarget = Math.max(state.commitRebindTarget ?? 0, generation);
  void ensureCurrentCommitBinding().catch(error => {
    setNotice(error.message || 'Commit succeeded, but task authority refresh failed.', 'error');
  });
}

function ensureCurrentCommitBinding() {
  if (!commitRebindRequired()) return Promise.resolve(controller.getState());
  if (state.commitRebindPromise) return state.commitRebindPromise;
  const split = state.split;
  const taskId = state.task.task_id;
  state.commitRebindError = null;
  state.commitRebindPromise = (async () => {
    if (state.interactionPromise) await state.interactionPromise;
    await controller.flush();
    if (!state.taskOpen || state.split !== split || state.task.task_id !== taskId) return;
    renderDraftState(controller.getState());
    const task = await api.getJson(`/api/splits/${split}/tasks/${encodeURIComponent(taskId)}`);
    if (!state.taskOpen || state.split !== split || state.task.task_id !== taskId) return;
    controller.rebind(task);
    state.task = task;
    state.commitRebindTarget = null;
    state.commitRebindError = null;
    setNotice('Commit advanced the dataset; this task was rebound without interrupting the batch.', 'ok');
  })().catch(error => {
    state.commitRebindError = error;
    throw error;
  }).finally(() => {
    state.commitRebindPromise = null;
    renderDraftState(controller.getState());
  });
  renderDraftState(controller.getState());
  return state.commitRebindPromise;
}

function setEditorControls(enabled) {
  for (const button of editorButtons) button.disabled = !enabled || state.interactionBusy;
  $('category-search').disabled = !enabled || state.interactionBusy;
  $('undo-button').disabled = !enabled || !controller.getState().canUndo;
  $('delete-button').disabled = !enabled || !state.selectedRegion;
  for (const control of visibilityControls) control.disabled = !enabled;
  $('hide-selected').disabled = !enabled || !state.selectedRegion;
}

function clearVisibleTask() {
  state.task = null;
  state.taskOpen = false;
  state.selectedRegion = null;
  state.commitRebindTarget = null;
  state.commitRebindError = null;
  editor.setTask(null);
  $('visibility-mode').value = 'all';
  renderTaskShell();
}
const pagePosition = () => state.page ? state.page.cursor + state.selected + 1 : 0;

async function beforeNavigation() {
  if (state.interactionPromise) await state.interactionPromise;
  if (state.commitActionPromise) await state.commitActionPromise;
  await ensureCurrentCommitBinding();
  if (!state.taskOpen) return;
  await controller.flush();
  if (controller.getState().authority === 'draft') state.navigationReminder = true;
}

async function loadPage(cursor = 0, selected = 0, { flush = true } = {}) {
  if (flush) await beforeNavigation();
  clearVisibleTask();
  setNotice('Loading task list…');
  const split = state.split;
  const request = ++state.pageRequest;
  state.taskRequest += 1;
  const page = await api.getJson(`/api/splits/${split}/tasks?cursor=${cursor}&limit=${state.limit}`);
  if (request !== state.pageRequest || split !== state.split) return;
  state.page = page;
  state.selected = page.tasks.length ? Math.min(selected, page.tasks.length - 1) : -1;
  renderPage();
  if (state.selected >= 0) await loadTask(page.tasks[state.selected].task_id, { flush: false });
  else setNotice('This split contains no indexed samples.');
}

async function loadTask(taskId, { flush = true, selectedIndex = null } = {}) {
  if (flush) await beforeNavigation();
  clearVisibleTask();
  setNotice('Loading sample…');
  const split = state.split;
  const request = ++state.taskRequest;
  const task = await api.getJson(`/api/splits/${split}/tasks/${encodeURIComponent(taskId)}`);
  if (request !== state.taskRequest || split !== state.split || task.task_id !== taskId) return;
  if (selectedIndex !== null) state.selected = selectedIndex;
  controller.open(task);
  state.task = task;
  state.taskOpen = true;
  state.selectedRegion = null;
  state.commitRebindTarget = null;
  state.commitRebindError = null;
  editor.setTask(task);
  $('visibility-mode').value = 'all';
  editor.setCategory(state.category);
  renderPage();
  renderTaskShell();
  setStatus('Connected', 'ok');
  if (state.navigationReminder) {
    setNotice('Previous sample saved as Draft. Remember to Commit the batch when ready.', 'warning');
    state.navigationReminder = false;
  } else setNotice('Ready · edits autosave after each completed gesture.');
}

async function selectIndex(index) {
  if (index < 0 || !state.page?.tasks[index] || index === state.selected) return;
  await loadTask(state.page.tasks[index].task_id, { selectedIndex: index });
}

async function goTo(position) {
  const target = Math.max(1, Math.min(Number(position) || 1, state.page?.total || 1));
  const cursor = Math.floor((target - 1) / state.limit) * state.limit;
  await loadPage(cursor, target - cursor - 1);
}

async function applyEditorGesture(detail) {
  if (!state.taskOpen || semanticActionLocked()) return;
  const snapshot = controller.getState();
  const existing = detail.regionKey
    ? snapshot.objects.find(object => object.region_key === detail.regionKey)
    : null;
  const category = detail.operation === 'create'
    ? state.category
    : existing ? { id: existing.category_id, name: existing.category_name } : null;
  if (!category) {
    setNotice('Choose one canonical COCO-80 category before drawing.', 'warning');
    return;
  }
  await runEditorMutation(async () => {
    const projected = await controller.projectAndApply({
      operation: detail.operation,
      ...(detail.regionKey ? { region_key: detail.regionKey } : {}),
      pixel_xyxy: detail.pixelXYXY,
      category_name: category.name,
    });
    state.selectedRegion = projected.region_key;
    editor.setSelected(projected.region_key);
    setNotice('Draft saved. Continue editing or navigate to another sample.', 'ok');
  });
}

async function chooseCategory(category) {
  if (semanticActionLocked()) {
    if (state.category) categoryPicker?.choose(state.category, { notify: false });
    else categoryPicker?.clearSelection({ notify: false });
    return;
  }
  state.category = category;
  editor.setCategory(category);
  if (!state.taskOpen || !state.selectedRegion) return;
  const object = controller.getState().objects.find(candidate => candidate.region_key === state.selectedRegion);
  if (!object || object.category_name === category.name) return;
  const rect = norm1000ToNaturalRect(object.bbox_2d, state.task.image_width, state.task.image_height);
  await runEditorMutation(async () => {
    const projected = await controller.projectAndApply({
      operation: 'update',
      region_key: object.region_key,
      pixel_xyxy: [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height],
      category_name: category.name,
    });
    editor.setSelected(projected.region_key);
    setNotice(`Relabeled as ${projected.category_name}; Draft saved.`, 'ok');
  });
}

async function runEditorMutation(action) {
  if (semanticActionLocked()) return undefined;
  if (state.interactionPromise) return state.interactionPromise;
  try { await ensureCurrentCommitBinding(); }
  catch (error) {
    setNotice(error.message || 'Refresh task authority before editing.', 'error');
    return undefined;
  }
  if (state.interactionPromise) return state.interactionPromise;
  state.interactionBusy = true;
  renderDraftState(controller.getState());
  state.interactionPromise = (async () => {
    try {
      await action();
      void commitController.refreshProjectState().catch(() => undefined);
    }
    catch (error) { setNotice(error.message || 'Edit could not be saved.', 'error'); }
    finally {
      state.interactionBusy = false;
      state.interactionPromise = null;
      renderDraftState(controller.getState());
    }
  })();
  return state.interactionPromise;
}

async function undoLastEdit() {
  if (!state.taskOpen || semanticActionLocked() || !controller.getState().canUndo) return;
  await runEditorMutation(async () => {
    await controller.undo();
    setNotice('Last semantic edit undone; the restored Draft is saved.', 'ok');
  });
}

async function deleteSelectedRegion() {
  if (!state.taskOpen || semanticActionLocked() || !state.selectedRegion) return;
  if (controller.getState().objects.length <= 1) {
    setNotice('The last bbox was not deleted: final-bbox policy still requires operator confirmation.', 'warning');
    return;
  }
  const regionKey = state.selectedRegion;
  await runEditorMutation(async () => {
    await controller.deleteRegion(regionKey);
    setNotice('BBox deleted; Undo can restore it.', 'ok');
  });
}

function startCommit() {
  if (state.commitActionPromise) return state.commitActionPromise;
  const selectedSplit = state.split;
  state.commitActionPromise = (async () => {
    if (state.interactionPromise) await state.interactionPromise;
    await ensureCurrentCommitBinding();
    if (state.taskOpen) await controller.flush();
    await commitController.refreshProjectState();
    const snapshot = commitController.getState();
    if (state.split !== selectedSplit || snapshot.split !== selectedSplit) {
      throw new Error('Commit action was cancelled because the split changed.');
    }
    if (snapshot.canRetry) {
      const retried = await commitController.retry();
      setNotice(retried.status === 'failed'
        ? 'The new Commit attempt failed; Drafts remain available.'
        : 'A new Commit attempt was accepted; annotation remains available while it runs.',
      retried.status === 'failed' ? 'error' : 'ok');
    } else {
      await commitController.commit();
      setNotice('Commit queued in the background. You can keep annotating.', 'ok');
    }
  })().finally(() => {
    state.commitActionPromise = null;
    renderPage();
    renderCommitState(commitController.getState());
  });
  renderPage();
  renderCommitState(commitController.getState());
  return state.commitActionPromise;
}

async function retrySave() {
  await controller.retry();
  setNotice('Draft retry succeeded.', 'ok');
}

async function reloadCurrentTask() {
  if (!state.taskOpen) return;
  const discard = controller.hasUnsavedLocal();
  if (discard && !window.confirm('Discard unresolved local edits and reload the authoritative task?')) return;
  const task = await api.getJson(`/api/splits/${state.split}/tasks/${encodeURIComponent(state.task.task_id)}`);
  controller.open(task, { discardUnsaved: discard });
  state.task = task;
  state.selectedRegion = null;
  state.commitRebindTarget = null;
  state.commitRebindError = null;
  editor.setTask(task);
  $('visibility-mode').value = 'all';
  editor.setCategory(state.category);
  setNotice(discard
    ? 'Authoritative task reloaded; unresolved local edits were discarded.'
    : 'Authoritative task refreshed.', discard ? 'warning' : 'ok');
}

async function navigate(action) {
  try { await action(); }
  catch (error) { setNotice(error.message || 'Navigation failed.', 'error'); }
}

async function switchSplit(nextSplit) {
  const prior = state.split;
  try {
    await beforeNavigation();
    state.split = nextSplit;
    await commitController.setSplit(nextSplit).catch(() => undefined);
    await loadPage(0, 0, { flush: false });
  } catch (error) {
    state.split = prior;
    $('split-select').value = prior;
    await commitController.setSplit(prior).catch(() => undefined);
    throw error;
  }
}

function setMode(mode) {
  editor.setMode(mode);
  for (const name of ['select', 'draw', 'pan']) {
    $(`mode-${name}`).setAttribute('aria-pressed', String(name === mode));
  }
}

function setVisibilityMode(mode) {
  const presentation = editor.setVisibilityMode(mode);
  $('visibility-mode').value = presentation.visibilityMode;
}

function hideSelectedRegion() {
  if (!state.selectedRegion) return;
  const presentation = editor.toggleRegionHidden(state.selectedRegion);
  $('visibility-mode').value = presentation.visibilityMode;
}

function restoreVisibility() {
  const presentation = editor.restoreVisibility();
  $('visibility-mode').value = presentation.visibilityMode;
  setNotice('All boxes are visible. Visibility changes never alter the Draft.', 'ok');
}

async function start() {
  try {
    setStatus('Starting…');
    await api.bootstrapSession();
    const categoryData = await api.getJson('/api/categories');
    if (!Array.isArray(categoryData.categories) || categoryData.categories.length !== 80) {
      throw new Error('Server COCO-80 registry is unavailable.');
    }
    state.categories = categoryData.categories;
    categoryPicker = installCategorySearch({
      input: $('category-search'),
      results: $('category-results'),
      selection: $('category-selection'),
      categories: state.categories,
      onChoose: category => { void chooseCategory(category); },
      onClear: () => {
        state.category = null;
        editor.setCategory(null);
      },
    });
    setMode('select');
    await commitController.setSplit(state.split).catch(() => undefined);
    await loadPage(0, 0, { flush: false });
  } catch (error) {
    setStatus('Unavailable', 'error');
    setNotice(error.message || 'Application startup failed.', 'error');
  }
}

$('split-select').addEventListener('change', event => { navigate(() => switchSplit(event.target.value)); });
$('previous-button').addEventListener('click', () => navigate(() => goTo(pagePosition() - 1)));
$('next-button').addEventListener('click', () => navigate(() => goTo(pagePosition() + 1)));
$('jump-button').addEventListener('click', () => navigate(() => goTo($('sample-number').value)));
$('sample-number').addEventListener('keydown', event => { if (event.key === 'Enter') navigate(() => goTo(event.target.value)); });
for (const mode of ['select', 'draw', 'pan']) {
  $(`mode-${mode}`).addEventListener('click', () => setMode(mode));
}
$('zoom-in').addEventListener('click', () => editor.zoomIn());
$('zoom-out').addEventListener('click', () => editor.zoomOut());
$('zoom-reset').addEventListener('click', () => editor.reset());
$('undo-button').addEventListener('click', () => { void undoLastEdit(); });
$('delete-button').addEventListener('click', () => { void deleteSelectedRegion(); });
$('visibility-mode').addEventListener('change', event => setVisibilityMode(event.target.value));
$('hide-selected').addEventListener('click', hideSelectedRegion);
$('restore-visibility').addEventListener('click', restoreVisibility);
$('commit-button').addEventListener('click', () => navigate(startCommit));
$('save-retry').addEventListener('click', () => navigate(retrySave));
$('task-reload').addEventListener('click', () => navigate(reloadCurrentTask));
document.addEventListener('keydown', event => {
  const tag = event.target instanceof Element ? event.target.tagName : '';
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z'
      && !['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)
      && !semanticActionLocked() && controller.getState().canUndo) {
    event.preventDefault();
    void undoLastEdit();
  }
  if ((event.key === 'Delete' || event.key === 'Backspace')
      && $('bbox-overlay').contains(event.target)
      && !semanticActionLocked() && state.selectedRegion) {
    event.preventDefault();
    void deleteSelectedRegion();
  }
});
window.addEventListener('beforeunload', event => {
  if (!controller.hasUnsavedLocal() && !state.interactionPromise && !editor.hasActiveGesture()) return;
  event.preventDefault();
  event.returnValue = '';
});
start();
