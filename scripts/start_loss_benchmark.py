"""Prepare and run the fixed GT-prefix fourth-loss comparison (no rollout training)."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/CoordExp/outputs/infra_base/start-loss-benchmark-20260928')
SOURCE_RUN = Path('/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-illegal-mass001-ebs24-4epoch')
SOURCE = SOURCE_RUN / 'checkpoints/step-2444'
VAL = Path('/data/CoordExp/outputs/infra_base/untied-axis-val200-20260918/val200.coord.jsonl')
ORDERS = (17, 29)
ARMS = ('control', 'ce', 'local_mass', 'instance_margin')
DEADLINE = '2026-09-28T19:21:00+00:00'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def write_yaml(name, config):
    path = ROOT / 'configs' / f'{name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def infer_config(name, checkpoint, *, smoke=False):
    from src.config.inference import load_infer_config
    config = load_infer_config('/data/CoordExp/outputs/infra_base/untied-illegal-mass-val200-20260927/norm.yaml').config.model_dump(mode='json')
    source_config = json.loads((SOURCE_RUN / 'resolved_config.json').read_text())['config']
    config['model']['base_model'] = str(Path(source_config['model']['base_model']).resolve())
    config['template'] = copy.deepcopy(source_config['template'])
    config['run'] = {'name': name, 'artifact_root': str(ROOT / 'infer'), 'collision_policy': 'fail'}
    config['data']['input_jsonl'] = str(ROOT / 'data/smoke_eval.jsonl') if smoke else str(VAL)
    config['adapter']['path'] = str(checkpoint / 'adapter')
    config['embedding_delta']['path'] = str(checkpoint / 'special_token_embeddings')
    config['generation'].update(batch_size=4, max_new_tokens=3084, repetition_penalty=1.0, temperature=0.0, top_p=1.0, n=1)
    config['backend']['hf']['coordinate_output_norm'] = 'median'
    return config


def prepare():
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Protocol already exists; do not overwrite a frozen experiment')
    base = json.loads((SOURCE_RUN / 'resolved_config.json').read_text())['config']
    source_data = Path(base['data']['train']['path'])
    rng = random.Random(1729)
    reservoir = []
    digest = hashlib.sha256()
    count = 0
    with source_data.open('rb') as f:
        for count, line in enumerate(f, 1):
            digest.update(line)
            if count <= 8192: reservoir.append(line)
            else:
                slot = rng.randrange(count)
                if slot < 8192: reservoir[slot] = line
    rows = [json.loads(line) for line in reservoir]
    rows.sort(key=lambda row: row['image_id'])
    for row in rows:
        row['images'] = [os.path.relpath((source_data.parent / image).resolve(), ROOT / 'data') for image in row['images']]
    assert len({row['image_id'] for row in rows}) == 8192
    eval_rows = [json.loads(line) for line in VAL.read_text().splitlines()]
    assert not {row['image_id'] for row in rows} & {row['image_id'] for row in eval_rows}
    (ROOT / 'data').mkdir(parents=True, exist_ok=True)
    data_paths = {}
    for seed in ORDERS:
        ordered = rows.copy()
        random.Random(seed).shuffle(ordered)
        path = ROOT / 'data' / f'train8192-order{seed}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in ordered))
        data_paths[seed] = path
    smoke_train = ROOT / 'data/smoke_train.jsonl'
    smoke_train.write_text(''.join(data_paths[17].read_text().splitlines(keepends=True)[:256]))
    for row in eval_rows[:8]:
        row['images'] = [os.path.relpath((VAL.parent / image).resolve(), ROOT / 'data') for image in row['images']]
    (ROOT / 'data/smoke_eval.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in eval_rows[:8]))

    base['model']['base_model'] = str(Path(base['model']['base_model']).resolve())
    base['adapter'].update(seed_mode='warm_start_expand_dora', path=None,
                           source_adapter_path=str(SOURCE / 'adapter'),
                           repaired_embedding_payload_path=str(SOURCE / 'special_token_embeddings'))
    base['resume'] = {'mode': 'disabled', 'checkpoint_dir': None}
    base['training'].update(epochs=8, max_steps=256, effective_batch_size=24)
    base['optimizer']['groups']['adapters']['language']['lr'] = 2e-5
    base['optimizer']['groups']['token_embeddings']['lr'] = 1e-5
    base['checkpoint'] = {'steps': [64, 256], 'save_final': True}
    base['eval'] = {'forward': {'steps': []}, 'inference': {'enabled': False}}
    base['data']['eval'] = {'path': str(ROOT / 'data/smoke_eval.jsonl'), 'sample_limit': None}
    base['observability']['steps'] = 1
    for seed in ORDERS:
        for arm in ARMS:
            name = f'{arm}-order{seed}'
            config = copy.deepcopy(base)
            config['run'] = {'name': name, 'artifact_root': str(ROOT / 'train'), 'collision_policy': 'fail'}
            config['runtime']['seed'] = seed
            config['data']['train'] = {'path': str(data_paths[seed]), 'sample_limit': None}
            if arm != 'control':
                config['losses']['auxiliary']['start_coordinate'] = {
                    'mode': arm, 'weight': .1, 'margin': .2,
                    'radius_fraction': .02, 'radius_cap': 4, 'calibrate': False,
                }
            write_yaml(name, config)
            for step in (64, 256):
                checkpoint = ROOT / 'train' / name / 'checkpoints' / f'step-{step}'
                write_yaml(f'eval-{name}-step{step}', infer_config(f'{name}-step{step}', checkpoint))
    smoke = copy.deepcopy(base)
    smoke['run'] = {'name': 'calibration-smoke', 'artifact_root': str(ROOT / 'train'), 'collision_policy': 'fail'}
    smoke['data']['train'] = {'path': str(smoke_train), 'sample_limit': None}
    smoke['training'].update(epochs=1, max_steps=1, effective_batch_size=24)
    smoke['checkpoint'] = {'steps': [], 'save_final': True}
    smoke['losses']['auxiliary']['start_coordinate'] = {'mode': 'ce', 'weight': .1, 'calibrate': True}
    write_yaml('calibration-smoke', smoke)
    write_yaml('eval-calibration-smoke', infer_config('calibration-smoke', ROOT / 'train/calibration-smoke/checkpoints/step-1', smoke=True))
    write_yaml('eval-source', infer_config('source', SOURCE))
    protocol = {
        'status': 'prepared', 'anchor': str(SOURCE), 'anchor_step': 2444,
        'anchor_payload_manifest_sha256': sha(SOURCE / 'inference_payload_manifest.json'),
        'anchor_config_sha256': sha(SOURCE_RUN / 'resolved_config.json'),
        'question': 'Which fourth onset loss improves mature three-loss checkpoint greedy detection after matched short continuation?',
        'claim_limit': 'Fixed checkpoint and budgets; not from-pretrained SFT superiority or physical false-positive truth.',
        'arms': list(ARMS), 'training_orders': list(ORDERS), 'train_rows': 8192,
        'source_train_rows': count, 'source_train_sha256': digest.hexdigest(),
        'data_sha256': {str(k): sha(v) for k, v in data_paths.items()}, 'val_sha256': sha(VAL),
        'steps': [64, 256], 'primary_endpoint': 256, 'primary_metric': 'COCO mAP 0.50:0.95',
        'secondary': ['FN50 class-aware IoU=.5 maxDets100', 'AP50', 'AP75', 'invalid_geometry', 'strict_repeats', 'length_caps'],
        'decode': {'greedy': True, 'repetition_penalty': 1.0, 'max_new_tokens': 3084, 'batch_size': 4, 'coordinate_output_norm': 'median'},
        'training': {'ranks_per_job': 4, 'concurrent_jobs': 2, 'global_batch': 24, 'packing_length': 12000,
                     'language_lr': 2e-5, 'embedding_lr': 1e-5, 'fresh_optimizer': True, 'untied': True},
        'auxiliary_weight_rule': '0.1 * initial CE coordinate-logit gradient norm / candidate norm, calibrated on train-only smoke before model updates; fixed thereafter. This does not equalize parameter gradients.',
        'budget': {'max_concurrent_gpus': 8, 'wall_hours': 16, 'gpu_hours_ceiling': 128, 'deadline_utc': DEADLINE},
        'stop_rule': 'Stop on deadline, non-finite update, failed load/save/evaluation identity, or incomplete arm. No automatic budget, cohort, dose or weight expansion.',
    }
    write_json(ROOT / 'protocol.json', protocol)
    print(json.dumps({'status': 'prepared', 'root': str(ROOT), 'train_count': len(rows)}, sort_keys=True))


def calibrate():
    records = []
    for line in (ROOT / 'logs/calibration-smoke.log').read_text().splitlines():
        if 'START_LOSS_CALIBRATION ' in line:
            records.append(json.loads(line.split('START_LOSS_CALIBRATION ', 1)[1]))
    if {r['rank'] for r in records} != set(range(4)):
        raise RuntimeError('Calibration needs all four ranks')
    sums = {mode: sum(r['grad_sq'][mode] for r in records) for mode in ARMS if mode != 'control'}
    if min(sums.values()) <= 0 or not all(math.isfinite(x) for x in sums.values()):
        raise RuntimeError('Missing finite, nonzero calibration support; do not invent a weight')
    weights = {mode: .1 * math.sqrt(sums['ce'] / value) for mode, value in sums.items()}
    for seed in ORDERS:
        for mode, weight in weights.items():
            name = f'{mode}-order{seed}'
            path = ROOT / 'configs' / f'{name}.yaml'
            c = yaml.safe_load(path.read_text())
            c['losses']['auxiliary']['start_coordinate']['weight'] = weight
            write_yaml(name, c)
    write_json(ROOT / 'calibration.json', {'records': records, 'gradient_squared_sums': sums, 'weights': weights})
    print(json.dumps({'weights': weights, 'records': len(records)}, sort_keys=True))


def command(args, log, env):
    remaining = datetime.fromisoformat(DEADLINE).timestamp() - time.time()
    if remaining <= 0: raise TimeoutError('Benchmark budget exhausted')
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open('x') as out:
        p = subprocess.Popen(args, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=remaining)
        except BaseException:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
    write_json(log.with_suffix('.receipt.json'), {'argv': args, 'returncode': code, 'elapsed_seconds': time.time() - started,
                                               'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES')})
    if code: raise RuntimeError(f'Command failed ({code}): {log}')


def evaluate(name, env):
    # Match the source evaluation's inference environment; strict replay is a training policy.
    env = dict(env)
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('FLASH_ATTENTION_DETERMINISTIC', None)
    command([sys.executable, '-m', 'src.infer', '--config', str(ROOT / 'configs' / f'eval-{name}.yaml')],
            ROOT / 'logs' / f'eval-{name}.log', env)
    command([sys.executable, '-m', 'scripts.evaluate_detection', '--artifact-dir', str(ROOT / 'infer' / name),
             '--out-dir', str(ROOT / 'infer' / name / 'eval')], ROOT / 'logs' / f'score-{name}.log', env)


def training_env(devices):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=devices, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', FLASH_ATTENTION_DETERMINISTIC='1')


def run_group(seed, devices):
    if not (ROOT / 'calibration.json').exists(): raise RuntimeError('Weights must be calibrated first')
    env = training_env(devices)
    jobs = list(ARMS)
    if seed == 29: jobs.reverse()  # Counterbalance order without changing paired within-seed data.
    state = {'status': 'running', 'seed': seed, 'devices': devices, 'completed': []}
    path = ROOT / f'group-{seed}.json'
    write_json(path, state)
    try:
        for arm in jobs:
            name = f'{arm}-order{seed}'
            state['active'] = name
            write_json(path, state)
            print(f'{datetime.now(timezone.utc).isoformat()} train {name}', flush=True)
            command([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                     '-m', 'src.train', '--config', str(ROOT / 'configs' / f'{name}.yaml')], ROOT / 'logs' / f'{name}.log', env)
            run = json.loads((ROOT / 'train' / name / 'run.json').read_text())
            if run['status'] != 'completed' or run['completed_steps'] != 256 or run['final_finite_status'] != 'finite':
                raise RuntimeError(f'Training terminal contract failed: {name}')
            for step in (64, 256): evaluate(f'{name}-step{step}', env)
            state['completed'].append(name)
            write_json(path, state)
        state.update(status='completed', active=None)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally: write_json(path, state)


def main():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'calibrate', 'group', 'source', 'smoke'])
    parser.add_argument('--seed', type=int, choices=ORDERS)
    parser.add_argument('--devices')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'calibrate': calibrate()
    elif args.action == 'source': evaluate('source', dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices))
    elif args.action == 'smoke':
        env = training_env(args.devices)
        command([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                 '-m', 'src.train', '--config', str(ROOT / 'configs/calibration-smoke.yaml')], ROOT / 'logs/calibration-smoke.log', env)
        evaluate('calibration-smoke', env)
    else: run_group(args.seed, args.devices)


if __name__ == '__main__': main()
