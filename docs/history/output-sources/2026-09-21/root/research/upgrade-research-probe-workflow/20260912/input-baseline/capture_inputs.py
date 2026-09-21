"""Task-local old/new CPU input evidence; never loads executable model weights."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=('old', 'new'), default='old')
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
source = args.source.resolve()
sys.path.insert(0, str(source))
args.output.mkdir(parents=True, exist_ok=False)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

import_start = time.perf_counter()
import torch
from src.config.inference import load_research_infer_config
from src.data import load_raw_examples
from src.inference.backend import DecodeRequest, GenerationPolicy
from src.inference.image_plan import plan_image_batch
from src.inference.prompt import build_prompt_record
from src.inference.runtime import assemble_frontend
from src.qwen.native import prepare_native_inputs
from src.qwen.native import NativeRequest
from src.qwen.encoding import EncodedExample
from src.inference.image_plan import ImagePlanRow
from src.inference.prompt import PromptRecord
import src.qwen.runtime_loading as loading
from probes.dora_owner_learning import runtime as dora
from probes.dora_owner_learning.prepare import _native_ce_group
from probes.logit_lens import runtime as logit
import_seconds = time.perf_counter() - import_start

def forbidden_weights(*a, **kw):
    raise AssertionError('baseline forbids model weight loading')
loading._load_model_from_options = forbidden_weights

fixture = source / 'tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.jsonl'
rows = load_raw_examples(fixture)
assert len(rows) == 2
profiles = {
    'source256': source / 'probes/dora_owner_learning/configs/source256.yaml',
    'logit_lens': source / 'probes/logit_lens/configs/qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml',
}

class Observe:
    names = {
        ('src/templates/renderer.py', 'render_example'): 'row_render',
        ('src/qwen/images.py', 'plan_qwen_image'): 'image_plan',
        ('src/qwen/native.py', 'prepare_native_inputs'): 'native_materialization',
        ('src/qwen/runtime_loading.py', 'load_qwen_components_from_options'): 'components_load',
    }
    def __init__(self):
        self.counts = Counter()
        self.encoded = {}
    def __call__(self, frame, event, value):
        path = frame.f_code.co_filename
        name = frame.f_code.co_name
        if event == 'call':
            for (suffix, symbol), counter in self.names.items():
                if name == symbol and path.endswith('/' + suffix):
                    self.counts[counter] += 1
        if event == 'return' and name == 'encode_rendered_example' and path.endswith('/src/qwen/encoding.py') and value is not None:
            self.encoded[value.example_id] = value

def old_plan(profile, config, frontend):
    result = []
    for index, raw in enumerate(rows):
        # This is the unchanged Source256 target-preparation consumer. For Logit
        # Lens, it is an explicit combined-inspection comparator, not a claim
        # that historical Logit Lens runs prepared annotated CE targets.
        group = _native_ce_group(raw, index=index, config=config, frontend=frontend)
        if profile == 'source256':
            request, image, prompt = dora.build_request(raw, config=config, qwen=frontend.qwen, row_index=index)
        else:
            image = plan_image_batch([raw], components=frontend.qwen,
                processor_config=logit._processor_config(config), row_indices=[index]).rows[0]
            prompt = build_prompt_record(raw, logit._template_config(config),
                processor=frontend.qwen.processor, row_index=index,
                merged_visual_tokens=image.merged_visual_tokens,
                object_order_seed=config.template.object_order_seed)
            request = DecodeRequest(request_id=str(raw.example_id), chat_text=prompt.chat_text,
                input_prompt_token_ids=tuple(prompt.input_prompt_token_ids),
                expected_executed_prompt_token_ids=tuple(prompt.expected_executed_prompt_token_ids),
                image_path=image.image_path, declared_image_width=image.declared_width,
                declared_image_height=image.declared_height, decoded_image_width=image.decoded_width,
                decoded_image_height=image.decoded_height, image_sha256=image.image_content_sha256,
                expected_image_grid_thw=tuple(image.expected_image_grid_thw),
                logical_transform_id=image.logical_transform_id,
                generation_policy=GenerationPolicy(max_new_tokens=3084, repetition_penalty=1.0))
        result.append((raw, group, request, image, prompt))
    return result

def materialize(profile, frontend, planned):
    result = []
    for raw, group, request, image, prompt in planned:
        if profile == 'source256' or args.mode == 'new':
            batch = prepare_native_inputs(frontend.qwen.processor, [request], device='cpu', record_media_identity=True)
            native = (batch.inputs, batch.prompt_token_ids, batch.image_grids, batch.media_sha256)
        else:
            native = logit.materialize_request(frontend.qwen, request)
        result.append((raw, group, image, prompt, native))
    return result

new_encoded = {}

def new_plan(profile, config, frontend):
    from src.inference.inputs import plan_examples
    planned = plan_examples(rows, config=config, components=frontend.qwen, target_max_length=12000)
    result = []
    for raw, entry in zip(rows, planned, strict=True):
        # The accepted API groups these existing typed values. Select by their
        # concrete types so this task-local harness does not dictate field names.
        values = [getattr(entry, field.name) for field in fields(entry)]
        def one(kind):
            matches = [value for value in values if isinstance(value, kind)]
            assert len(matches) == 1, (kind, len(matches))
            return matches[0]
        target, prompt, image, request = (one(kind) for kind in (EncodedExample, PromptRecord, ImagePlanRow, NativeRequest))
        new_encoded[raw.example_id] = target
        start = target.supervised_token_spans[0].physical_token_start
        end = target.supervised_token_spans[-1].physical_token_end
        group = {'prompt_token_ids': list(prompt.expected_executed_prompt_token_ids),
            'actions': [{'action_token_ids': list(target.input_ids[start:end])}]}
        result.append((raw, group, request, image, prompt))
    return result

def tensor_identity(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'sha256': hashlib.sha256(raw).hexdigest()}

def summarize(sample, encoded):
    raw, group, image, prompt, native = sample
    inputs, token_rows, grids, media = native
    target = encoded[raw.example_id]
    supervised = [s.to_artifact_dict() for s in target.supervised_token_spans]
    ignored = [s.to_artifact_dict() for s in target.ignored_token_spans]
    prefix = list(token_rows[0])
    assert prefix == list(target.input_ids[:target.supervised_token_spans[0].physical_token_start])
    assert prefix == group['prompt_token_ids']
    record = {'example_id': raw.example_id, 'chat_text': prompt.chat_text,
        'prompt_token_ids': prefix, 'grid': list(grids[0]), 'executed_rgb_sha256': media[0],
        'image_file_sha256': image.image_content_sha256,
        'logical_transform': image.logical_transform_id,
        'realized_object_order': prompt.realized_object_order,
        'target_chat_text': target.chat_text, 'full_target_token_ids': list(target.input_ids),
        'action_token_ids': group['actions'][0]['action_token_ids'],
        'supervised_spans': supervised, 'ignored_spans': ignored,
        'native_tensors': {key: tensor_identity(value) for key, value in inputs.items() if isinstance(value, torch.Tensor)},
        'terminal_eos_id': group['actions'][0]['action_token_ids'][-1],
        'generation_prefix_length': len(prefix)}
    record['content_sha256'] = digest(record)
    return record

plan = old_plan if args.mode == 'old' else new_plan
receipt = {'mode': args.mode, 'source_root': str(source), 'script_sha256': file_digest(__file__),
    'fixture_sha256': file_digest(fixture), 'ordered_ids': [row.example_id for row in rows],
    'model_weights_loaded': False, 'python_import_seconds': import_seconds, 'profiles': {},
    'comparison_scope': 'Old Source256 target helper plus each profile generation helper; batch-one native materialization per row. Logit target path is composed inspection, not a historical Logit execution claim.'}
for profile, config_path in profiles.items():
    resolved = load_research_infer_config(config_path)
    config = resolved.config
    start = time.perf_counter()
    frontend = assemble_frontend(config, generation_config_fingerprint=digest(config.generation.model_dump(mode='json')))
    load_seconds = time.perf_counter() - start
    assert frontend.qwen.model is None
    observer = Observe()
    sys.setprofile(observer)
    try:
        planned = plan(profile, config, frontend)
        samples = materialize(profile, frontend, planned)
    finally:
        sys.setprofile(None)
    outputs = [summarize(sample, observer.encoded if args.mode == 'old' else new_encoded) for sample in samples]
    timings = []
    for repeat in range(args.repeats):
        start = time.perf_counter()
        planned = plan(profile, config, frontend)
        planned_at = time.perf_counter()
        materialize(profile, frontend, planned)
        end = time.perf_counter()
        timings.append({'planning_seconds': planned_at-start, 'materialization_seconds': end-planned_at, 'total_seconds': end-start})
    record = {'config_path': str(config_path), 'config_file_sha256': file_digest(config_path),
        'resolved_config_fingerprint': resolved.fingerprint,
        'model_base_path': str(frontend.qwen.base_model_path),
        'frontend_load_seconds': load_seconds, 'operation_counts_two_rows': dict(observer.counts),
        'timings_warm_uninstrumented': timings,
        'timing_summary': {key: {'median': statistics.median(t[key] for t in timings),
            'min': min(t[key] for t in timings), 'max': max(t[key] for t in timings)} for key in timings[0]},
        'rows': outputs, 'output_content_sha256': digest(outputs)}
    (args.output / f'{profile}.json').write_text(json.dumps(record, indent=2)+'\n')
    receipt['profiles'][profile] = {key: record[key] for key in ('output_content_sha256', 'operation_counts_two_rows', 'timing_summary', 'frontend_load_seconds')}
    print(json.dumps({'profile': profile, **receipt['profiles'][profile]}), flush=True)
(args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
