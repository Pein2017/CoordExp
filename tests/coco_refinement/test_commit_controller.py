from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


MODULE = Path(__file__).parents[2] / "src" / "coco_refinement" / "static" / "commit-controller.js"


def _run_node(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional test tooling")
    completed = subprocess.run(
        [
            node,
            "--experimental-default-type=module",
            "--input-type=module",
            "--eval",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_commit_controller_background_lifecycle_and_recovery() -> None:
    module_uri = json.dumps(MODULE.resolve().as_uri())
    _run_node(
        f"""
import assert from 'node:assert/strict';
import {{ createCommitController }} from {module_uri};

const hash = 'a'.repeat(64);
const project = split => ({{
  split, project_id: `project:${{split}}`, task_count: 10, generation: 0,
  pending_draft_count: 2, accepting_writes: true,
  worker: {{ split, state: 'idle', healthy: true }},
}});
const receipt = (batch_id, status, overrides = {{}}) => ({{
  batch_id, split: 'train', status, member_count: 2, base_generation: 0,
  generation: status === 'succeeded' || status === 'failed' ? 1 : null,
  payload_hash: hash, detail: null, ...overrides,
}});
const apiError = status => Object.assign(new Error(`HTTP ${{status}}`), {{ status }});
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

function memoryStorage(seed = {{}}) {{
  const values = new Map(Object.entries(seed));
  return {{
    getItem(key) {{ return values.has(key) ? values.get(key) : null; }},
    setItem(key, value) {{ values.set(key, String(value)); }},
    removeItem(key) {{ values.delete(key); }},
    dump() {{ return Object.fromEntries(values); }},
  }};
}}

function manualScheduler() {{
  let nextId = 1;
  const callbacks = new Map();
  return {{
    setTimeout(callback) {{ const id = nextId++; callbacks.set(id, callback); return id; }},
    clearTimeout(id) {{ callbacks.delete(id); }},
    size() {{ return callbacks.size; }},
    async runNext() {{
      const entry = callbacks.entries().next().value;
      assert.ok(entry, 'a scheduled poll is required');
      callbacks.delete(entry[0]);
      await entry[1]();
      await tick();
    }},
  }};
}}

// Durable enqueue returns before any status poll and polling reaches every exact state.
{{
  const scheduler = manualScheduler();
  const storage = memoryStorage();
  const events = [];
  const statuses = ['running', 'reconciling', 'succeeded'];
  const calls = [];
  const api = {{
    async getJson(path) {{
      calls.push(['GET', path]);
      if (path.endsWith('/state')) return project('train');
      return receipt('web:uuid-1', statuses.shift());
    }},
    async postJson(path, body) {{
      calls.push(['POST', path, structuredClone(body)]);
      return receipt(body.batch_id, 'queued');
    }},
  }};
  const controller = createCommitController({{
    api, scheduler, storage, randomUUID: () => 'uuid-1', pollIntervalMs: 1,
    onChange: state => events.push(state),
  }});
  await controller.setSplit('train');
  const committed = await controller.commit();
  assert.equal(committed.status, 'queued');
  assert.equal(scheduler.size(), 1);
  assert.deepEqual(calls.filter(call => call[0] === 'POST')[0][2], {{ batch_id: 'web:uuid-1' }});
  assert.ok(Object.keys(storage.dump()).some(key => key.endsWith(':train')));
  assert.equal(calls.filter(call => call[1].includes('/commits/web')).length, 0);
  assert.equal(calls.filter(call => call[1].endsWith('/state')).length, 1);
  await scheduler.runNext();
  assert.equal(controller.getState().status, 'running');
  await scheduler.runNext();
  assert.equal(controller.getState().status, 'reconciling');
  await scheduler.runNext();
  assert.equal(controller.getState().status, 'succeeded');
  assert.equal(scheduler.size(), 0);
  assert.deepEqual(storage.dump(), {{}});
  assert.ok(events.some(state => state.status === 'queued' && state.polling));
  events.at(-1).status = 'corrupted';
  assert.equal(controller.getState().status, 'succeeded');
}}

// A lost POST response resolves status first and never recaptures the batch.
{{
  const scheduler = manualScheduler();
  const storage = memoryStorage();
  const calls = [];
  const api = {{
    async getJson(path) {{
      calls.push(['GET', path]);
      if (path.endsWith('/state')) return project('train');
      return receipt('web:lost-id', 'queued');
    }},
    async postJson(path, body) {{
      calls.push(['POST', path, structuredClone(body)]);
      throw apiError(0);
    }},
  }};
  const controller = createCommitController({{
    api, scheduler, storage, randomUUID: () => 'lost-id', pollIntervalMs: 1,
  }});
  await controller.setSplit('train');
  await controller.commit();
  assert.equal(calls.filter(call => call[0] === 'POST').length, 1);
  assert.equal(calls.filter(call => call[1].endsWith('/web%3Alost-id')).length, 1);
  controller.stop();
  assert.equal(scheduler.size(), 0);
}}

// Only a 404 status permits one idempotent replay, with the identical batch body.
{{
  const scheduler = manualScheduler();
  const bodies = [];
  let statusReads = 0;
  const api = {{
    async getJson(path) {{
      if (path.endsWith('/state')) return project('train');
      statusReads += 1;
      if (statusReads === 1) throw apiError(404);
      return receipt('web:replay-id', 'queued');
    }},
    async postJson(_path, body) {{
      bodies.push(structuredClone(body));
      if (bodies.length === 1) throw apiError(503);
      return receipt(body.batch_id, 'queued');
    }},
  }};
  const controller = createCommitController({{
    api, scheduler, storage: memoryStorage(), randomUUID: () => 'replay-id',
  }});
  await controller.setSplit('train');
  await controller.commit();
  assert.equal(statusReads, 1);
  assert.deepEqual(bodies, [{{ batch_id: 'web:replay-id' }}, {{ batch_id: 'web:replay-id' }}]);
  controller.stop();
}}

// A failed receipt is generic, and explicit retry starts a fresh batch after confirming failure.
{{
  const scheduler = manualScheduler();
  const storage = memoryStorage();
  let postCount = 0;
  const bodies = [];
  const api = {{
    async getJson(path) {{
      if (path.endsWith('/state')) return project('train');
      return receipt('web:retry-id', 'failed', {{
        detail: {{ message: 'secret /tmp/private/receipt' }},
      }});
    }},
    async postJson(_path, body) {{
      bodies.push(structuredClone(body));
      postCount += 1;
      return receipt(body.batch_id, postCount === 1 ? 'failed' : 'queued');
    }},
  }};
  const controller = createCommitController({{
    api, scheduler, storage, randomUUID: (() => {{
      const values = ['retry-id', 'retry-new'];
      return () => values.shift();
    }})(),
  }});
  await controller.setSplit('train');
  await controller.commit();
  assert.equal(controller.getState().status, 'failed');
  assert.doesNotMatch(JSON.stringify(controller.getState()), /private|secret/);
  assert.equal(controller.getState().canCommit, false);
  await assert.rejects(controller.commit(), /retry the failed Commit/);
  await controller.retry();
  assert.equal(controller.getState().status, 'queued');
  assert.deepEqual(bodies, [{{ batch_id: 'web:retry-id' }}, {{ batch_id: 'web:retry-new' }}]);
  controller.stop();
}}

// Reload resumes a persisted batch; switching split cancels its timer and stale callbacks.
{{
  const key = 'coco-refinement.commit.active.v1:train';
  const storage = memoryStorage({{ [key]: JSON.stringify({{ batch_id: 'web:resume-id' }}) }});
  const scheduler = manualScheduler();
  const api = {{
    async getJson(path) {{
      if (path.endsWith('/state')) return project(path.includes('/val/') ? 'val' : 'train');
      return receipt('web:resume-id', 'running');
    }},
    async postJson() {{ throw new Error('resume must status-check first'); }},
  }};
  const controller = createCommitController({{
    api, scheduler, storage, randomUUID: () => 'unused',
  }});
  await controller.setSplit('train');
  assert.equal(controller.getState().status, 'running');
  assert.equal(scheduler.size(), 1);
  await controller.setSplit('val');
  assert.equal(controller.getState().split, 'val');
  assert.equal(controller.getState().status, 'idle');
  assert.equal(scheduler.size(), 0);
}}

// Stored intent whose original POST never arrived is replayed only after status 404.
{{
  const key = 'coco-refinement.commit.active.v1:train';
  const storage = memoryStorage({{ [key]: JSON.stringify({{ batch_id: 'web:resume-replay' }}) }});
  const order = [];
  const api = {{
    async getJson(path) {{
      order.push(['GET', path]);
      if (path.endsWith('/state')) return project('train');
      throw apiError(404);
    }},
    async postJson(path, body) {{
      order.push(['POST', path, structuredClone(body)]);
      return receipt(body.batch_id, 'queued');
    }},
  }};
  const controller = createCommitController({{
    api, scheduler: manualScheduler(), storage, randomUUID: () => 'unused',
  }});
  await controller.setSplit('train');
  assert.equal(controller.getState().status, 'queued');
  const statusIndex = order.findIndex(call => call[0] === 'GET' && call[1].includes('/commits/'));
  const postIndex = order.findIndex(call => call[0] === 'POST');
  assert.ok(statusIndex >= 0 && postIndex > statusIndex);
  assert.deepEqual(order[postIndex][2], {{ batch_id: 'web:resume-replay' }});
}}

// Reload restores an already-failed batch without silently retrying it.
{{
  const key = 'coco-refinement.commit.active.v1:train';
  const storage = memoryStorage({{ [key]: JSON.stringify({{ batch_id: 'web:failed-reload' }}) }});
  let posted = false;
  const api = {{
    async getJson(path) {{
      if (path.endsWith('/state')) return project('train');
      return receipt('web:failed-reload', 'failed', {{ detail: {{ message: '/tmp/secret' }} }});
    }},
    async postJson() {{ posted = true; throw new Error('must not retry during reload'); }},
  }};
  const controller = createCommitController({{
    api, scheduler: manualScheduler(), storage, randomUUID: () => 'unused',
  }});
  await controller.setSplit('train');
  assert.equal(controller.getState().status, 'failed');
  assert.equal(posted, false);
  assert.equal(JSON.stringify(controller.getState()).includes('secret'), false);
  assert.equal(JSON.stringify(controller.getState()).includes('/tmp/'), false);
}}

// Unsafe UUID output never reaches storage or the API.
{{
  const storage = memoryStorage();
  let posted = false;
  const api = {{
    async getJson() {{ return project('train'); }},
    async postJson() {{ posted = true; }},
  }};
  const controller = createCommitController({{
    api, scheduler: manualScheduler(), storage, randomUUID: () => '../unsafe id',
  }});
  await controller.setSplit('train');
  await assert.rejects(controller.commit(), /unsafe batch identity/);
  assert.equal(posted, false);
  assert.deepEqual(storage.dump(), {{}});
}}
"""
    )
