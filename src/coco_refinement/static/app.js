import { createApiClient } from '/api-client.js';
import { installCategorySearch } from '/class-search.js';
import { createDraftController } from '/draft-controller.js';
import { norm1000ToNaturalRect } from '/editor-geometry.js';
import { createSvgEditor } from '/svg-editor.js';

const state = {
  split: 'train', limit: 50, page: null, selected: -1, task: null,
  taskOpen: false, categories: [], category: null, selectedRegion: null,
  pageRequest: 0, taskRequest: 0, interactionBusy: false,
  interactionPromise: null, navigationReminder: false,
};
const $ = id => document.getElementById(id);
const api = createApiClient();
const status = $('connection-status');
const notice = $('notice');
const editorButtons = [
  $('mode-select'), $('mode-draw'), $('mode-pan'),
  $('zoom-out'), $('zoom-in'), $('zoom-reset'),
];
let categoryPicker = null;
const setNotice = (message, tone = '') => { notice.textContent = message; notice.dataset.tone = tone; };
const setStatus = (message, tone = 'pending') => { status.textContent = message; status.dataset.tone = tone; };

let controller;
const editor = createSvgEditor({
  svg: $('bbox-overlay'),
  onGesture: detail => { void applyEditorGesture(detail); },
  onSelection: ({ regionKey, object }) => {
    state.selectedRegion = regionKey;
    if (object) {
      state.category = { id: object.category_id, name: object.category_name };
      editor.setCategory(state.category);
      categoryPicker?.choose(state.category, { notify: false });
    }
  },
  onMessage: ({ message, tone }) => setNotice(message, tone),
});
controller = createDraftController({ api, onChange: snapshot => renderDraftState(snapshot) });

function renderPage() {
  const page = state.page;
  $('page-summary').textContent = page && page.total ? `${page.cursor + 1}–${Math.min(page.cursor + page.tasks.length, page.total)} / ${page.total}` : page ? '0 / 0' : '—';
  $('task-list').replaceChildren(...(page?.tasks || []).map((task, index) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button'; button.disabled = state.interactionBusy; button.setAttribute('aria-current', String(index === state.selected));
    button.textContent = `#${page.cursor + index + 1} · ${task.image_id}`;
    const meta = document.createElement('span'); meta.className = 'task-meta'; meta.textContent = `${task.image_width} × ${task.image_height}`;
    button.append(meta); button.addEventListener('click', () => navigate(() => selectIndex(index))); li.append(button); return li;
  }));
  $('previous-button').disabled = state.interactionBusy || !page || page.cursor + state.selected <= 0;
  $('next-button').disabled = state.interactionBusy || !page || page.cursor + state.selected + 1 >= page.total;
  $('split-select').disabled = state.interactionBusy;
  $('sample-number').disabled = state.interactionBusy;
  $('jump-button').disabled = state.interactionBusy;
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
  const visiblePhase = state.interactionBusy ? 'Saving' : snapshot.phase;
  const tone = { Committed: 'ok', Draft: 'pending', Saving: 'pending', Conflict: 'error', Error: 'error' }[visiblePhase] || 'pending';
  $('save-status').textContent = visiblePhase === 'Draft' ? 'Draft saved · Commit pending' : visiblePhase;
  $('save-status').dataset.tone = tone;
  $('save-retry').hidden = snapshot.phase !== 'Error';
  $('task-reload').hidden = !['Error', 'Conflict'].includes(snapshot.phase);
  const locked = state.interactionBusy || ['Saving', 'Error', 'Conflict'].includes(snapshot.phase);
  editor.setDisabled(locked);
  $('category-search').disabled = locked || !state.taskOpen;
  for (const button of editorButtons) button.disabled = locked || !state.taskOpen;
  for (const button of $('task-list').querySelectorAll('button')) button.disabled = state.interactionBusy;
  $('split-select').disabled = state.interactionBusy;
  $('sample-number').disabled = state.interactionBusy;
  $('jump-button').disabled = state.interactionBusy;
  $('previous-button').disabled = state.interactionBusy || !state.page || pagePosition() <= 1;
  $('next-button').disabled = state.interactionBusy || !state.page || pagePosition() >= state.page.total;
  const details = $('task-details').querySelectorAll('dd');
  details[0].textContent = snapshot.authority || '—';
  details[1].textContent = snapshot.phase;
  details[2].textContent = snapshot.binding?.revision ?? '—';
  details[3].textContent = snapshot.binding?.generation ?? '—';
  details[4].textContent = state.task ? `${state.task.image_width} × ${state.task.image_height}` : '—';
}

function setEditorControls(enabled) {
  for (const button of editorButtons) button.disabled = !enabled || state.interactionBusy;
  $('category-search').disabled = !enabled || state.interactionBusy;
}

function clearVisibleTask() {
  state.task = null;
  state.taskOpen = false;
  state.selectedRegion = null;
  editor.setTask(null);
  renderTaskShell();
}
const pagePosition = () => state.page ? state.page.cursor + state.selected + 1 : 0;

async function beforeNavigation() {
  if (state.interactionPromise) await state.interactionPromise;
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
  editor.setTask(task);
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
  if (!state.taskOpen || state.interactionBusy) return;
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
  if (state.interactionBusy) {
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
  if (state.interactionPromise) return state.interactionPromise;
  state.interactionBusy = true;
  renderDraftState(controller.getState());
  state.interactionPromise = (async () => {
    try { await action(); }
    catch (error) { setNotice(error.message || 'Edit could not be saved.', 'error'); }
    finally {
      state.interactionBusy = false;
      state.interactionPromise = null;
      renderDraftState(controller.getState());
    }
  })();
  return state.interactionPromise;
}

async function retrySave() {
  await controller.retry();
  setNotice('Draft retry succeeded.', 'ok');
}

async function reloadCurrentTask() {
  if (!state.taskOpen) return;
  if (!window.confirm('Discard unresolved local edits and reload the authoritative task?')) return;
  const task = await api.getJson(`/api/splits/${state.split}/tasks/${encodeURIComponent(state.task.task_id)}`);
  controller.open(task, { discardUnsaved: true });
  state.task = task;
  state.selectedRegion = null;
  editor.setTask(task);
  editor.setCategory(state.category);
  setNotice('Authoritative task reloaded; unresolved local edits were discarded.', 'warning');
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
    await loadPage(0, 0, { flush: false });
  } catch (error) {
    state.split = prior;
    $('split-select').value = prior;
    throw error;
  }
}

function setMode(mode) {
  editor.setMode(mode);
  for (const name of ['select', 'draw', 'pan']) {
    $(`mode-${name}`).setAttribute('aria-pressed', String(name === mode));
  }
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
$('save-retry').addEventListener('click', () => navigate(retrySave));
$('task-reload').addEventListener('click', () => navigate(reloadCurrentTask));
window.addEventListener('beforeunload', event => {
  if (!controller.hasUnsavedLocal() && !state.interactionPromise && !editor.hasActiveGesture()) return;
  event.preventDefault();
  event.returnValue = '';
});
start();
