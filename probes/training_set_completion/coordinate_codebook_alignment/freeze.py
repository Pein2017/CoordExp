"""Freeze an execution packet; captures are evidence, never import paths."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path

from probes.training_set_completion.artifacts import binding
from src.artifacts.source_provenance import preserve_source
from src.config.loader import load_train_config

REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-alignment')
ADMISSION_SHA = 'ff5a7db03d75dca035776c6cd95cb417e5292e96a0b091c612b6fa2d05f28b65'


def freeze(name: str, configs: list[Path]) -> Path:
    destination = ROOT / 'launches' / name
    destination.mkdir(parents=True, exist_ok=False)
    admission_path = ROOT / 'selection-v4/admission.json'
    admitted = binding(admission_path)
    if admitted['sha256'] != ADMISSION_SHA:
        raise ValueError('frozen admission changed')
    admission = json.loads(admission_path.read_text())
    for item in admission['source_bindings']:
        if binding(item['path'])['sha256'] != item['sha256']:
            raise ValueError(f"admission source changed: {item['path']}")
    sources = list((REPO / 'src').rglob('*.py'))
    sources += list(Path(__file__).parent.rglob('*.py'))
    sources += [REPO / 'probes/training_set_completion/artifacts.py']
    sources += list((REPO / 'configs/research/coordinate_codebook_alignment').glob('*.yaml'))
    sources += [REPO / 'research/experiments/2026-09-22-coordinate-codebook-alignment/unit.md']
    captures = []
    for source in sorted(set(sources)):
        copy = preserve_source(source, run_root=destination, relative_name=source.relative_to(REPO))
        captures.append({'current': binding(source), 'capture': binding(copy)})
    for name in ('transformers.models.qwen3_vl.modeling_qwen3_vl',
                 'transformers.models.qwen3_vl.processing_qwen3_vl',
                 'peft.tuners.lora.layer', 'peft.tuners.lora.dora'):
        module = importlib.import_module(name)
        source = Path(module.__file__)
        copy = preserve_source(source, run_root=destination,
                               relative_name=Path('runtime') / (name + '.py'))
        captures.append({'current': binding(source), 'capture': binding(copy)})
    config_records = []
    for path in configs:
        resolved = load_train_config(path)
        config_records.append({'binding': binding(path), 'fingerprint': resolved.fingerprint,
                               'resolved': resolved.config_dict})
    base = Path(admission['source_config']['model']['base_model'])
    index = base / 'model.safetensors.index.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    manifest = {
        'status': 'frozen', 'package': '2026-09-22-coordinate-codebook-alignment',
        'admission': admitted, 'configs': config_records, 'sources': captures,
        'base_weights': [binding(index), *[binding(base / p) for p in shards]],
        'inputs': [binding(ROOT / 'selection-v4' / p) for p in
                   ('fit.coord.jsonl', 'monitor.coord.jsonl', 'qualification.coord.jsonl')]
                  + [binding(p) for p in sorted((ROOT / 'runtime-inputs-v2').glob('*'))],
        'unit': binding(REPO / 'research/experiments/2026-09-22-coordinate-codebook-alignment/unit.md'),
        'runtime': {p: importlib.metadata.version(p) for p in
                    ('torch', 'transformers', 'peft', 'accelerate', 'flash-attn')},
        'qualification_ids': [2299, 13004, 417044],
        'limits': {'model_wall_seconds': 28800, 'allocated_gpu_seconds': 230400,
                   'gpus': list(range(8)), 'stop_reserve_seconds': 120},
        'native_policy': {'empty_prefix': True, 'max_new_tokens': 3084,
                          'greedy': True, 'repetition_penalty': 1.0},
    }
    path = destination / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(binding(path)))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--config', type=Path, action='append', required=True)
    args = parser.parse_args()
    freeze(args.name, args.config)
