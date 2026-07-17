import { installCategorySearch } from '/class-search.js';

const state = { csrf: '', split: 'train', limit: 50, pageCursor: 0, page: null, selected: -1, task: null, categories: [], pageRequest: 0, taskRequest: 0 };
const $ = (id) => document.getElementById(id);
const status = $('connection-status');
const notice = $('notice');
const setNotice = (message, tone = '') => { notice.textContent = message; notice.dataset.tone = tone; };
const setStatus = (message, tone = 'pending') => { status.textContent = message; status.dataset.tone = tone; };

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: 'same-origin', ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body?.error?.message || `Request failed (${response.status})`);
  return body;
}

function renderPage() {
  const page = state.page;
  $('page-summary').textContent = page && page.total ? `${page.cursor + 1}–${Math.min(page.cursor + page.tasks.length, page.total)} / ${page.total}` : page ? '0 / 0' : '—';
  $('task-list').replaceChildren(...(page?.tasks || []).map((task, index) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button'; button.setAttribute('aria-current', String(index === state.selected));
    button.textContent = `#${page.cursor + index + 1} · ${task.image_id}`;
    const meta = document.createElement('span'); meta.className = 'task-meta'; meta.textContent = `${task.image_width} × ${task.image_height}`;
    button.append(meta); button.addEventListener('click', () => selectIndex(index)); li.append(button); return li;
  }));
  $('previous-button').disabled = !page || page.cursor + state.selected <= 0;
  $('next-button').disabled = !page || page.cursor + state.selected + 1 >= page.total;
  $('sample-number').value = page && state.selected >= 0 ? String(page.cursor + state.selected + 1) : '';
}

function renderTask() {
  const task = state.task;
  const image = $('task-image'); const placeholder = $('image-placeholder'); const svg = $('bbox-overlay');
  svg.replaceChildren();
  if (!task) { image.hidden = true; placeholder.hidden = false; $('task-heading').textContent = 'No sample selected'; $('task-counter').textContent = '—'; return; }
  image.src = task.image_url; image.hidden = false; placeholder.hidden = true;
  svg.setAttribute('viewBox', `0 0 ${task.image_width} ${task.image_height}`);
  $('task-heading').textContent = `${task.split} · ${task.image_id}`;
  $('task-counter').textContent = `${pagePosition()} / ${state.page.total}`;
  for (const object of task.objects || []) {
    const [x1, y1, x2, y2] = object.bbox_2d || [];
    if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
    const left = x1 * task.image_width / 999;
    const top = y1 * task.image_height / 999;
    const right = x2 * task.image_width / 999;
    const bottom = y2 * task.image_height / 999;
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect'); rect.setAttribute('x', left); rect.setAttribute('y', top); rect.setAttribute('width', Math.max(0, right - left)); rect.setAttribute('height', Math.max(0, bottom - top)); svg.append(rect);
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text'); label.setAttribute('x', left + 4); label.setAttribute('y', Math.max(16, top + 16)); label.textContent = object.category_name || String(object.category_id); svg.append(label);
  }
  const details = $('task-details').querySelectorAll('dd'); details[0].textContent = task.authority || '—'; details[1].textContent = 'loaded (read-only)'; details[2].textContent = task.revision ?? '—'; details[3].textContent = task.generation ?? '—'; details[4].textContent = `${task.image_width} × ${task.image_height}`;
}
const pagePosition = () => state.page ? state.page.cursor + state.selected + 1 : 0;

async function loadPage(cursor = 0, selected = 0) {
  setNotice('Loading task list…');
  const split = state.split;
  const request = ++state.pageRequest;
  state.taskRequest += 1;
  let page;
  try {
    page = await api(`/api/splits/${split}/tasks?cursor=${cursor}&limit=${state.limit}`);
  } catch (error) {
    if (request !== state.pageRequest || split !== state.split) return;
    throw error;
  }
  if (request !== state.pageRequest || split !== state.split) return;
  state.pageCursor = cursor; state.page = page; state.task = null;
  state.selected = state.page.tasks.length ? Math.min(selected, state.page.tasks.length - 1) : -1; renderPage();
  renderTask();
  if (state.selected >= 0) await loadTask(state.page.tasks[state.selected].task_id); else renderTask();
}
async function loadTask(taskId) {
  setNotice('Loading sample…');
  const split = state.split;
  const request = ++state.taskRequest;
  let task;
  try {
    task = await api(`/api/splits/${split}/tasks/${encodeURIComponent(taskId)}`);
  } catch (error) {
    if (request !== state.taskRequest || split !== state.split) return;
    throw error;
  }
  if (request !== state.taskRequest || split !== state.split || task.task_id !== taskId) return;
  state.task = task; renderTask(); setNotice('Ready · browsing only'); setStatus('Connected', 'ok');
}
async function selectIndex(index) { if (index < 0 || !state.page?.tasks[index]) return; state.selected = index; state.task = null; renderPage(); renderTask(); await loadTask(state.page.tasks[index].task_id); }
async function goTo(position) { const target = Math.max(1, Math.min(Number(position) || 1, state.page?.total || 1)); const cursor = Math.floor((target - 1) / state.limit) * state.limit; await loadPage(cursor, target - cursor - 1); }
async function navigate(action) { try { await action(); } catch (error) { setNotice(error.message, 'error'); } }

async function start() {
  try {
    setStatus('Starting…'); const session = await api('/api/session'); state.csrf = session.csrf_token || '';
    const categoryData = await api('/api/categories'); state.categories = categoryData.categories || [];
    installCategorySearch({ input: $('category-search'), results: $('category-results'), selection: $('category-selection'), categories: state.categories });
    await loadPage(0, 0);
  } catch (error) { setStatus('Unavailable', 'error'); setNotice(error.message, 'error'); }
}

$('split-select').addEventListener('change', (event) => { state.split = event.target.value; navigate(() => loadPage(0, 0)); });
$('previous-button').addEventListener('click', () => navigate(() => goTo(pagePosition() - 1)));
$('next-button').addEventListener('click', () => navigate(() => goTo(pagePosition() + 1)));
$('jump-button').addEventListener('click', () => navigate(() => goTo($('sample-number').value)));
$('sample-number').addEventListener('keydown', (event) => { if (event.key === 'Enter') navigate(() => goTo(event.target.value)); });
start();
