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


def test_draft_controller_bounded_undo_state_machine() -> None:
    api_uri = json.dumps((STATIC / "api-client.js").as_uri())
    controller_uri = json.dumps((STATIC / "draft-controller.js").as_uri())
    _run_node(
        f"""
import assert from 'node:assert/strict';
import {{ ApiError }} from {api_uri};
import {{ createDraftController }} from {controller_uri};

const hash = 'b'.repeat(64);
const first = {{
  region_key: 'train:coco:101', bbox_2d: [10, 20, 300, 400],
  category_name: 'person', category_id: 1, coco_ann_id: 101,
  metadata: {{ source: {{ row: 7 }}, note: 'preserve me' }},
}};
const second = {{
  region_key: 'train:coco:202', bbox_2d: [30, 40, 500, 600],
  category_name: 'dog', category_id: 18, coco_ann_id: 202,
  metadata: {{ nested: {{ values: [1, 2, 3] }} }},
}};
const baseline = [first, second];
const task = (objects = baseline, revision = 0) => ({{
  split: 'train', task_id: 'train:undo', authority: revision ? 'draft' : 'committed',
  revision, generation: 0, base_row_hash: hash, objects: structuredClone(objects),
}});
const applied = (body, revision) => ({{
  status: 'applied', mutation_id: body.mutation_id, authority: 'draft',
  revision, generation: 0, base_row_hash: hash, objects: structuredClone(body.objects),
}});
const ids = () => {{
  let index = 0;
  return () => `undo-uuid-${{++index}}`;
}};
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

// A successful canonical projection records exactly one pre-action snapshot.
// Lost PUT responses replay the same request without duplicating Undo history.
{{
  const putBodies = [];
  let revision = 0;
  const api = {{
    async postJson(_path, body) {{
      return {{
        object: {{
          region_key: 'local:created', bbox_2d: [1, 2, 30, 40],
          category_name: body.category_name, category_id: 2,
          metadata: {{ origin: 'human' }},
        }},
        binding: {{ revision, generation: 0, base_row_hash: hash }},
      }};
    }},
    async putJson(_path, body) {{
      putBodies.push(structuredClone(body));
      if (putBodies.length === 1) throw new ApiError('response lost', {{ status: 0 }});
      revision += 1;
      return applied(body, revision);
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  assert.equal(controller.getState().canUndo, false);
  await controller.projectAndApply({{
    operation: 'create', pixel_xyxy: [1, 2, 30, 40], category_name: 'bicycle',
  }});
  assert.deepEqual(putBodies[0], putBodies[1]);
  assert.equal(controller.getState().canUndo, true);
  assert.deepEqual(
    controller.getState().objects.map(object => object.region_key),
    ['train:coco:101', 'train:coco:202', 'local:created'],
  );

  await controller.undo();
  assert.deepEqual(controller.getState().objects, baseline);
  assert.deepEqual(putBodies.at(-1).objects, baseline);
  assert.equal(controller.getState().canUndo, false);
  assert.equal(controller.hasUnsavedLocal(), false);

  controller.replaceRegion('train:coco:101', {{ ...first, bbox_2d: [11, 20, 300, 400] }});
  await controller.flush();
  assert.equal(controller.getState().canUndo, true);
  controller.open(task([], revision));
  assert.equal(controller.getState().canUndo, false);
}}

// A same-task Commit authority rebind preserves Undo and rebases allocated negative IDs.
{{
  let revision = 0;
  const bodies = [];
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson(_path, body) {{
      bodies.push(structuredClone(body));
      revision += 1;
      return bodies.length === 1 ? applied(body, revision) : {{
        ...applied(body, revision), generation: 1, base_row_hash: 'c'.repeat(64),
      }};
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  const local = {{
    region_key: 'local:new', bbox_2d: [50, 60, 200, 240],
    category_name: 'cat', category_id: 17, metadata: {{ origin: 'human' }},
  }};
  controller.open(task([...baseline, local]));
  controller.replaceRegion('local:new', {{ ...local, bbox_2d: [55, 60, 200, 240] }});
  await controller.flush();
  const rebasedCurrent = controller.getState().objects.map(object => (
    object.region_key === 'local:new' ? {{ ...object, coco_ann_id: -9101 }} : object
  ));
  controller.rebind({{
    ...task(rebasedCurrent, revision), generation: 1, base_row_hash: 'c'.repeat(64),
  }});
  assert.equal(controller.getState().canUndo, true);
  assert.equal(controller.getState().objects[2].coco_ann_id, -9101);
  await controller.undo();
  assert.deepEqual(controller.getState().objects[2].bbox_2d, local.bbox_2d);
  assert.equal(controller.getState().objects[2].coco_ann_id, -9101);
  assert.equal(bodies.at(-1).expected_generation, 1);
  assert.equal(bodies.at(-1).expected_base_row_hash, 'c'.repeat(64));
  const changedNegativeId = controller.getState().objects.map(object => (
    object.region_key === 'local:new' ? {{ ...object, coco_ann_id: -9102 }} : object
  ));
  assert.throws(() => controller.rebind({{
    ...task(changedNegativeId, revision), generation: 1, base_row_hash: 'c'.repeat(64),
  }}), /changed task semantics/);
  const changedPositiveId = controller.getState().objects.map(object => (
    object.region_key === 'train:coco:101' ? {{ ...object, coco_ann_id: 102 }} : object
  ));
  assert.throws(() => controller.rebind({{
    ...task(changedPositiveId, revision), generation: 1, base_row_hash: 'c'.repeat(64),
  }}), /changed task semantics/);
}}

// A failed projection never creates an Undo step.
{{
  const api = {{
    async postJson() {{ throw new ApiError('projection failed', {{ status: 503 }}); }},
    async putJson() {{ throw new Error('unused'); }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  await assert.rejects(controller.projectAndApply({{
    operation: 'create', pixel_xyxy: [1, 2, 30, 40], category_name: 'person',
  }}));
  assert.equal(controller.getState().canUndo, false);
}}

// Only the newest 50 complete snapshots are retained, and Undo does not add
// redo/history entries of its own.
{{
  let revision = 0;
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson(_path, body) {{ revision += 1; return applied(body, revision); }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  for (let value = 1; value <= 51; value += 1) {{
    controller.replaceRegion('train:coco:101', {{ ...first, bbox_2d: [10 + value, 20, 300, 400] }});
    await controller.flush();
  }}
  for (let count = 0; count < 50; count += 1) await controller.undo();
  assert.deepEqual(controller.getState().objects[0].bbox_2d, [11, 20, 300, 400]);
  assert.deepEqual(controller.getState().objects[1], second);
  assert.equal(controller.getState().canUndo, false);
}}

// Saving, Error, Conflict, and projection-in-flight states reject Undo.
{{
  let releaseSave;
  const save = new Promise(resolve => {{ releaseSave = resolve; }});
  let revision = 0;
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson(_path, body) {{ await save; revision += 1; return applied(body, revision); }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{ ...first, bbox_2d: [12, 20, 300, 400] }});
  assert.equal(controller.getState().canUndo, false);
  await assert.rejects(controller.undo(), /Undo is unavailable/);
  releaseSave();
  await controller.flush();
  assert.equal(controller.getState().canUndo, true);
}}

for (const status of [422, 409]) {{
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson() {{ throw new ApiError('save rejected', {{ status, body: {{ status }} }}); }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{ ...first, bbox_2d: [13, 20, 300, 400] }});
  await assert.rejects(controller.flush());
  assert.equal(controller.getState().canUndo, false);
  await assert.rejects(controller.undo(), /Undo is unavailable/);
}}

{{
  let revision = 0;
  let releaseProjection;
  const projection = new Promise(resolve => {{ releaseProjection = resolve; }});
  const api = {{
    async postJson() {{ return projection; }},
    async putJson(_path, body) {{ revision += 1; return applied(body, revision); }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{ ...first, bbox_2d: [14, 20, 300, 400] }});
  await controller.flush();
  const pending = controller.projectAndApply({{
    operation: 'update', region_key: 'train:coco:101',
    pixel_xyxy: [15, 20, 300, 400], category_name: 'person',
  }});
  await tick();
  assert.equal(controller.getState().canUndo, false);
  await assert.rejects(controller.undo(), /Undo is unavailable/);
  releaseProjection({{
    object: {{ ...first, bbox_2d: [15, 20, 300, 400] }},
    binding: {{ revision, generation: 0, base_row_hash: hash }},
  }});
  await pending;
  assert.equal(controller.getState().canUndo, true);
}}
"""
    )


def test_draft_controller_delete_region_policy_and_undo() -> None:
    api_uri = json.dumps((STATIC / "api-client.js").as_uri())
    controller_uri = json.dumps((STATIC / "draft-controller.js").as_uri())
    _run_node(
        f"""
import assert from 'node:assert/strict';
import {{ ApiError }} from {api_uri};
import {{ createDraftController, FinalBboxPolicyError }} from {controller_uri};

const hash = 'd'.repeat(64);
const objects = [
  {{
    region_key: 'train:coco:101', bbox_2d: [10, 20, 300, 400],
    category_name: 'person', category_id: 1, coco_ann_id: 101,
    metadata: {{ source: {{ row: 7 }}, note: 'first' }},
  }},
  {{
    region_key: 'train:coco:202', bbox_2d: [30, 40, 500, 600],
    category_name: 'dog', category_id: 18, coco_ann_id: 202,
    metadata: {{ nested: {{ values: [1, 2, 3] }}, note: 'middle' }},
  }},
  {{
    region_key: 'local:new', bbox_2d: [50, 60, 700, 800],
    category_name: 'cat', category_id: 17, coco_ann_id: -9101,
    metadata: {{ origin: 'human', note: 'last' }},
  }},
];
const task = (taskObjects = objects, revision = 0) => ({{
  split: 'train', task_id: 'train:delete', authority: revision ? 'draft' : 'committed',
  revision, generation: 0, base_row_hash: hash, objects: structuredClone(taskObjects),
}});
const applied = (body, revision) => ({{
  status: 'applied', mutation_id: body.mutation_id, authority: 'draft',
  revision, generation: 0, base_row_hash: hash, objects: structuredClone(body.objects),
}});
const ids = () => {{
  let index = 0;
  return () => `delete-uuid-${{++index}}`;
}};
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

// Deleting the middle object is one full-Draft PUT and Undo restores the exact
// original order, metadata, stable keys, and COCO identities with one more PUT.
{{
  const calls = [];
  let revision = 0;
  const api = {{
    async postJson(path, body) {{ calls.push(['POST', path, structuredClone(body)]); }},
    async putJson(path, body) {{
      calls.push(['PUT', path, structuredClone(body)]);
      revision += 1;
      return applied(body, revision);
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  await controller.deleteRegion('train:coco:202');
  assert.deepEqual(calls.map(call => call[0]), ['PUT']);
  assert.deepEqual(
    calls[0][2].objects.map(object => object.region_key),
    ['train:coco:101', 'local:new'],
  );
  assert.equal(controller.getState().canUndo, true);
  await controller.undo();
  assert.deepEqual(calls.map(call => call[0]), ['PUT', 'PUT']);
  assert.deepEqual(calls[1][2].objects, objects);
  assert.deepEqual(controller.getState().objects, objects);
  assert.equal(controller.getState().canUndo, false);
}}

// Refusing the final bbox and an unknown key performs no request and changes
// neither the authoritative binding nor local Undo/semantic state.
{{
  const calls = [];
  const api = {{
    async postJson(...args) {{ calls.push(['POST', ...args]); }},
    async putJson(...args) {{ calls.push(['PUT', ...args]); }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task([objects[0]], 3));
  const before = controller.getState();
  await assert.rejects(
    controller.deleteRegion('train:coco:101'),
    error => error instanceof FinalBboxPolicyError
      && error.message === 'final bbox policy requires operator confirmation',
  );
  assert.deepEqual(controller.getState(), before);
  assert.equal(calls.length, 0);
  await assert.rejects(
    controller.deleteRegion('missing:key'),
    /region key is not present in local objects/,
  );
  assert.deepEqual(controller.getState(), before);
  assert.equal(calls.length, 0);
}}

// A response-loss replay reuses the same delete body and does not duplicate
// its Undo snapshot.
{{
  const bodies = [];
  let revision = 0;
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson(_path, body) {{
      bodies.push(structuredClone(body));
      if (bodies.length === 1) throw new ApiError('response lost', {{ status: 0 }});
      revision += 1;
      return applied(body, revision);
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  await controller.deleteRegion('train:coco:202');
  assert.equal(bodies.length, 2);
  assert.deepEqual(bodies[0], bodies[1]);
  assert.equal(controller.getState().canUndo, true);
  await controller.undo();
  assert.equal(bodies.length, 3);
  assert.deepEqual(controller.getState().objects, objects);
  assert.equal(controller.getState().canUndo, false);
}}

// Delete follows the controller's existing unresolved-operation boundary.
{{
  let releaseSave;
  const delayedSave = new Promise(resolve => {{ releaseSave = resolve; }});
  let revision = 0;
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson(_path, body) {{
      await delayedSave;
      revision += 1;
      return applied(body, revision);
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{
    ...objects[0], bbox_2d: [11, 20, 300, 400],
  }});
  await tick();
  await assert.rejects(
    controller.deleteRegion('train:coco:202'),
    /Delete is unavailable while local state is unresolved/,
  );
  releaseSave();
  await controller.flush();
}}

for (const status of [422, 409]) {{
  const api = {{
    async postJson() {{ throw new Error('unused'); }},
    async putJson() {{
      throw new ApiError('save rejected', {{ status, body: {{ status }} }});
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  controller.replaceRegion('train:coco:101', {{
    ...objects[0], bbox_2d: [12, 20, 300, 400],
  }});
  await assert.rejects(controller.flush());
  await assert.rejects(
    controller.deleteRegion('train:coco:202'),
    /Delete is unavailable while local state is unresolved/,
  );
}}

{{
  let releaseProjection;
  const projection = new Promise(resolve => {{ releaseProjection = resolve; }});
  let revision = 0;
  const api = {{
    async postJson() {{ return projection; }},
    async putJson(_path, body) {{
      revision += 1;
      return applied(body, revision);
    }},
  }};
  const controller = createDraftController({{ api, randomUUID: ids() }});
  controller.open(task());
  const pending = controller.projectAndApply({{
    operation: 'update', region_key: 'train:coco:101',
    pixel_xyxy: [11, 20, 300, 400], category_name: 'person',
  }});
  await tick();
  await assert.rejects(
    controller.deleteRegion('train:coco:202'),
    /Delete is unavailable while local state is unresolved/,
  );
  releaseProjection({{
    object: {{ ...objects[0], bbox_2d: [11, 20, 300, 400] }},
    binding: {{ revision, generation: 0, base_row_hash: hash }},
  }});
  await pending;
}}
"""
    )
