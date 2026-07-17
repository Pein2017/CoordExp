from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


STATIC = Path(__file__).parents[2] / "src" / "coco_refinement" / "static"


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


def test_api_client_and_draft_controller_state_machine() -> None:
    api_uri = json.dumps((STATIC / "api-client.js").as_uri())
    controller_uri = json.dumps((STATIC / "draft-controller.js").as_uri())
    _run_node(
        f"""
import assert from 'node:assert/strict';
import {{ ApiError, createApiClient }} from {api_uri};
import {{ createDraftController }} from {controller_uri};

const hash = 'a'.repeat(64);
const source = {{
  region_key: 'train:coco:101', bbox_2d: [10, 20, 300, 400],
  category_name: 'person', category_id: 1, coco_ann_id: 101,
}};
const task = (objects = [source], revision = 0) => ({{
  split: 'train', task_id: 'train:7', authority: revision ? 'draft' : 'committed',
  revision, generation: 0, base_row_hash: hash, objects,
}});
const applied = (body, revision) => ({{
  status: 'applied', mutation_id: body.mutation_id, authority: 'draft',
  revision, generation: 0, base_row_hash: hash, objects: structuredClone(body.objects),
}});
const ids = (...values) => {{
  let index = 0;
  return () => values[index++] || `uuid-${{index}}`;
}};
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

// The HTTP client bootstraps CSRF, keeps structured 409 bodies, and marks mutations.
{{
  const calls = [];
  const fetchImpl = async (path, options) => {{
    calls.push([path, structuredClone(options)]);
    if (path === '/api/session') return {{ ok: true, status: 200, json: async () => ({{ csrf_token: 'csrf' }}) }};
    return {{ ok: false, status: 409, json: async () => ({{ error: {{ message: 'stale' }}, binding: {{ revision: 2 }} }}) }};
  }};
  const client = createApiClient({{ fetchImpl, origin: 'http://127.0.0.1:9144' }});
  await client.bootstrapSession();
  await assert.rejects(
    client.putJson('/api/splits/train/tasks/train:7/draft', {{ value: 1 }}),
    error => error instanceof ApiError && error.status === 409 && error.body.binding.revision === 2,
  );
  assert.equal(calls[1][1].headers['x-csrf-token'], 'csrf');
  assert.equal(calls[1][1].credentials, 'same-origin');
  assert.equal(calls[1][1].body, '{{"value":1}}');
}}

// Projection create/update is followed by full-Draft PUT and preserves stable order.
{{
  const calls = [];
  let revision = 0;
  const api = {{
    async postJson(path, body) {{
      calls.push(['POST', path, structuredClone(body)]);
      const object = body.operation === 'create'
        ? {{ region_key: 'local:created', bbox_2d: [1, 2, 30, 40], category_name: body.category_name, category_id: 2 }}
        : {{ ...source, bbox_2d: [2, 3, 31, 41], category_name: body.category_name, category_id: 18 }};
      return {{ object, binding: {{ revision, generation: 0, base_row_hash: hash }} }};
    }},
    async putJson(path, body) {{
      calls.push(['PUT', path, structuredClone(body)]);
      revision += 1;
      return applied(body, revision);
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids('request-1', 'save-1', 'request-2', 'save-2') }});
  controller.open(task());
  await controller.projectAndApply({{ operation: 'create', pixel_xyxy: [1, 2, 3, 4], category_name: 'bicycle' }});
  assert.deepEqual(calls.map(call => call[0]), ['POST', 'PUT']);
  assert.equal(calls[0][2].request_id, 'projection:request-1');
  assert.equal(calls[1][2].mutation_id, 'draft:save-1');
  assert.deepEqual(calls[1][2].objects.map(item => item.region_key), ['train:coco:101', 'local:created']);
  await controller.projectAndApply({{ operation: 'update', region_key: 'train:coco:101', pixel_xyxy: [2, 3, 4, 5], category_name: 'dog' }});
  assert.equal(calls[2][2].region_key, 'train:coco:101');
  assert.deepEqual(calls[3][2].objects.map(item => item.region_key), ['train:coco:101', 'local:created']);
  assert.equal(controller.getState().objects[0].coco_ann_id, 101);
  assert.equal(controller.hasUnsavedLocal(), false);
}}

// Project waits for a prior local save; a non-409 projection failure leaves durable state usable.
{{
  const order = [];
  let revision = 0;
  const api = {{
    async putJson(_path, body) {{ order.push('PUT'); revision += 1; return applied(body, revision); }},
    async postJson() {{ order.push('POST'); throw new ApiError('offline', {{ status: 503 }}); }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids('save-before', 'projection-fail') }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{ ...source, bbox_2d: [11, 20, 300, 400] }});
  await assert.rejects(controller.projectAndApply({{ operation: 'create', pixel_xyxy: [1, 2, 3, 4], category_name: 'person' }}));
  assert.deepEqual(order, ['PUT', 'POST']);
  assert.equal(controller.getState().phase, 'Draft');
  assert.equal(controller.hasUnsavedLocal(), false);
  controller.open(task([], 0));
}}

// An in-flight projection is unsaved work even before it produces local objects.
{{
  let releaseProjection;
  const projection = new Promise(resolve => {{ releaseProjection = resolve; }});
  const api = {{
    async postJson() {{ return projection; }},
    async putJson(_path, body) {{ return applied(body, 1); }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids('projection-pending', 'save-after') }});
  controller.open(task());
  const pending = controller.projectAndApply({{ operation: 'create', pixel_xyxy: [1, 2, 3, 4], category_name: 'person' }});
  await tick();
  assert.equal(controller.hasUnsavedLocal(), true);
  assert.throws(() => controller.open(task([])));
  releaseProjection({{
    object: {{ ...source, region_key: 'local:pending' }},
    binding: {{ revision: 0, generation: 0, base_row_hash: hash }},
  }});
  await pending;
  assert.equal(controller.hasUnsavedLocal(), false);
}}

// A lost save response is automatically replayed once with byte-equivalent body.
{{
  const bodies = [];
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson(_path, body) {{
      bodies.push(structuredClone(body));
      if (bodies.length === 1) throw new ApiError('lost', {{ status: 0 }});
      return applied(body, 1);
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids('lost-save') }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{ ...source, bbox_2d: [12, 20, 300, 400] }});
  await controller.flush();
  assert.equal(bodies.length, 2);
  assert.deepEqual(bodies[0], bodies[1]);
  assert.equal(controller.hasUnsavedLocal(), false);
  controller.open(task([], 0));
}}

// An edit made during a save is serialized against the returned newer revision.
{{
  const bodies = [];
  let release;
  const first = new Promise(resolve => {{ release = resolve; }});
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson(_path, body) {{
      bodies.push(structuredClone(body));
      if (bodies.length === 1) return first;
      return applied(body, 2);
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids('save-a', 'save-b') }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{ ...source, bbox_2d: [12, 20, 300, 400] }});
  await tick();
  const flushing = controller.flush();
  controller.replaceRegion('train:coco:101', {{ ...source, bbox_2d: [13, 20, 300, 400] }});
  release(applied(bodies[0], 1));
  await flushing;
  assert.equal(bodies.length, 2);
  assert.equal(bodies[1].expected_revision, 1);
  assert.deepEqual(bodies[1].objects[0].bbox_2d, [13, 20, 300, 400]);
  assert.deepEqual(controller.getState().objects[0].bbox_2d, [13, 20, 300, 400]);
}}

// Two ambiguous failures enter Error; manual retry reuses the original request.
{{
  const bodies = [];
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson(_path, body) {{
      bodies.push(structuredClone(body));
      if (bodies.length < 3) throw new ApiError('lost', {{ status: 503 }});
      return applied(body, 1);
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids('retry-save') }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{ ...source, bbox_2d: [14, 20, 300, 400] }});
  await assert.rejects(controller.flush());
  assert.equal(controller.getState().phase, 'Error');
  assert.equal(controller.hasUnsavedLocal(), true);
  assert.throws(() => controller.open(task([])));
  await controller.retry();
  assert.deepEqual(bodies[0], bodies[1]);
  assert.deepEqual(bodies[1], bodies[2]);
  assert.equal(controller.hasUnsavedLocal(), false);
  controller.open(task([]));
}}

// 409 preserves local objects and blocks flush/navigation replacement.
{{
  const conflictBody = {{ error: {{ message: 'stale' }}, binding: {{ revision: 2 }} }};
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson() {{ throw new ApiError('stale', {{ status: 409, body: conflictBody }}); }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids('conflict-save') }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{ ...source, bbox_2d: [15, 20, 300, 400] }});
  await assert.rejects(controller.flush(), error => error.status === 409);
  assert.equal(controller.getState().phase, 'Conflict');
  assert.deepEqual(controller.getState().objects[0].bbox_2d, [15, 20, 300, 400]);
  assert.equal(controller.getState().conflict.binding.revision, 2);
  assert.throws(() => controller.open(task([])));
}}
"""
    )
