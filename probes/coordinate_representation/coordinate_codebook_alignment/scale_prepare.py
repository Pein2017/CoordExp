"""Bind the authorized scale launch without allocating a GPU or rebuilding data."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path

from src.artifacts.utf8_json import binding
from src.artifacts.source_provenance import preserve_source
from src.config.loader import load_train_config
from src.config.paths import resolve_run_directory
from src.qwen import load_qwen_components
from src.training.pack_cache import build_packing_cache_fingerprint, load_cache_manifest
from src.training.schedule import resolve_planned_step_schedule

REPO = Path(__file__).resolve().parents[3]
PARENT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
V3 = PARENT / '2026-09-22-coordinate-codebook-scale-preparation/v3'
ROOT = PARENT / '2026-09-22-coordinate-codebook-scale'


def prepare() -> Path:
    path = ROOT / 'launch-v1.json'
    if path.exists():
        raise ValueError('launch already frozen')
    packet = json.loads((V3 / 'manifest.json').read_text())
    if binding(V3 / 'manifest.json')['sha256'] != '2e9d064e5248d30668cbe324ece04102afb574af478b87509907c2b7cc4ccdea':
        raise ValueError('preparation packet changed')
    records = packet['bindings'] + [packet[k] for k in ('unchanged_split', 'closed_package', 'authority', 'report')]
    records += json.loads((V3 / 'launch-routes.json').read_text())['source_bindings']
    old = PARENT / '2026-09-22-coordinate-codebook-alignment'
    records += json.loads((old / 'selection-v4/admission.json').read_text())['source_bindings']
    records += json.loads((old / 'launches/production-v1/manifest.json').read_text())['base_weights']
    for item in records:
        if binding(item['path'])['sha256'] != item['sha256']:
            raise ValueError(f"input/source binding changed: {item['path']}")
    resolved = load_train_config(V3 / 'training-config.json')
    config = resolved.config
    run_dir = resolve_run_directory(config, cwd=REPO).run_dir
    if config.run.collision_policy != 'fail' or run_dir.exists():
        raise ValueError('new run collision')
    components = load_qwen_components(config, load_model=False)
    assert components.model is None
    fingerprint = build_packing_cache_fingerprint(config, components, dataset=config.data.train, split='train')
    cache = V3 / 'packing-cache' / fingerprint
    manifest = load_cache_manifest(cache, expected_fingerprint=fingerprint)
    if manifest['micro_step_count'] != 492:
        raise ValueError('wrong admitted cache size')
    schedule = resolve_planned_step_schedule(config, packs_per_epoch=492, world_size=4)
    assert schedule.resolved_max_steps == 1968 and schedule.tail_fill_pack_count == 0
    assert [x.planned_step_id for x in schedule.events['checkpoint']] == [62, 123, 246, 984, 1968]
    evaluation = {}
    for condition, count, step in [('source', 1248, None), ('epoch4', 96, 246), ('epoch16', 96, 984), ('epoch32', 1280, 1968)]:
        queue = ROOT / 'queues' / condition / 'queue.json'
        original = V3 / f'queue-{condition}.json'
        if queue.read_bytes() != original.read_bytes() or queue.with_name('queue-state.json').exists():
            raise ValueError('queue copy or empty claim state mismatch')
        assert len(json.loads(queue.read_text())['specs']) == count
        records.append(binding(queue))
        evaluation[condition] = {'queue': str(queue), 'checkpoint': 'source' if step is None else str(run_dir / 'checkpoints' / f'step-{step}'), 'output': str(ROOT / 'production' / condition), 'cells': count}
    assert len({Path(x['queue']).parent for x in evaluation.values()}) == 4
    unit = REPO / 'research/experiments/2026-09-22-coordinate-codebook-scale/unit.md'
    sources = list((REPO / 'src').rglob('*.py'))
    sources += [p for p in Path(__file__).parent.glob('*.py') if p.name not in ('scale_reduce.py', 'test_scale_reduce.py')]
    sources += [unit, Path(__file__).resolve().parents[3] / 'src/artifacts/utf8_json.py']
    captures = []
    for source in sorted(set(sources)):
        capture = preserve_source(source, run_root=path, relative_name=source.relative_to(REPO))
        captures.append({'current': binding(source), 'capture': binding(capture)})
    for name in ('transformers.models.qwen3_vl.modeling_qwen3_vl', 'transformers.models.qwen3_vl.processing_qwen3_vl', 'peft.tuners.lora.layer', 'peft.tuners.lora.dora'):
        source = Path(importlib.import_module(name).__file__)
        capture = preserve_source(source, run_root=path, relative_name=Path('runtime') / (name + '.py'))
        captures.append({'current': binding(source), 'capture': binding(capture)})
    records += [binding(unit), binding(V3 / 'manifest.json')]
    records += [x['current'] for x in captures]
    argv = ['python', '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', '-m', 'probes.coordinate_representation.coordinate_codebook_alignment.scale_train', '--config', str(V3 / 'training-config.json'), '--output', str(ROOT / 'first-production.json'), '--packing-plan', str(V3 / 'packing-exposure.json')]
    result = {'status': 'mechanical_launch_checks_passed', 'root': str(ROOT), 'config': binding(V3 / 'training-config.json'), 'run_dir': str(run_dir), 'cache_root': str(V3 / 'packing-cache'), 'cache_fingerprint': fingerprint, 'cache_rebuilt': False, 'schedule': schedule.to_artifact_dict(), 'training_argv': argv, 'training_environment': {'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'coordexp_infras_PACK_CACHE_ROOT': str(V3 / 'packing-cache'), 'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}, 'evaluation': evaluation, 'evaluation_admission': str(V3 / 'evaluation-admission.json'), 'bindings': records, 'source_captures': captures, 'runtime': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'accelerate', 'flash-attn')}, 'limits': {'wall_seconds': 28800, 'allocated_gpu_seconds': 230400, 'stop_admission_after_seconds': 27900, 'reserve_seconds': 900, 'clock_boundary': 'first new model-entry process launch including loading'}, 'model_calls': 0, 'reuse_qualification': str(old / 'qualification/complete-v1/manifest.json'), 'implementation_allowlist': ['execute.py', 'scale_execute.py', 'scale_prepare.py', 'scale_train.py', 'scale_reduce.py', 'test_scale_execute.py', 'test_scale_train.py', 'test_scale_reduce.py']}
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return path


if __name__ == '__main__':
    print(json.dumps(binding(prepare())))
