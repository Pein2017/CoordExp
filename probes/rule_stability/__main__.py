"""Thin commands for CPU preparation and exact-packet native execution."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from . import artifacts as a


def runtime_identity():
    return dict(python=sys.version, packages={name: version(name) for name in
                ('torch', 'transformers', 'peft', 'safetensors', 'flash-attn')})


def input_binding():
    from .data import FULL_LABEL_PATH, INPUT_MANIFEST_PATH, load_inputs
    images, manifest = load_inputs()
    return dict(labels_path=str(FULL_LABEL_PATH.relative_to(a.ROOT)), labels_sha256=a.digest(FULL_LABEL_PATH),
                manifest_path=str(INPUT_MANIFEST_PATH.relative_to(a.ROOT)), manifest_sha256=a.digest(INPUT_MANIFEST_PATH)), images


def config_for(output, *, engine, arm, updates, source):
    inputs, _ = input_binding()
    return dict(schema=a.SCHEMA, output=str(output.resolve()), engine=engine, arm=arm, updates=updates,
        world_size=8, checkpoint_versions=[v for v in (0, 1, 4, 8, 16) if v <= updates],
        source=source, inputs=inputs, runtime=runtime_identity(), anchor=a.verify_anchor(hash_payload=False),
        protocol=dict(path=str((a.UNIT / 'unit.md').relative_to(a.ROOT)), sha256=a.digest(a.UNIT / 'unit.md')),
        frozen=dict(horizon=3084, coordinate_norm='median', images=18, labels=570,
                    sample_temperature=1, duplicate_comparator='>', duplicate_iou=.9))


def launch(config_path, output, *, native_released, payload=None):
    config = a.load(config_path)
    output.mkdir(parents=True, exist_ok=False)
    a.write(output / 'invocation.json', dict(config_path=str(config_path.resolve()), config_sha256=a.digest(config_path),
        native_released=native_released, pid=os.getpid(), started=time.time(), runtime=runtime_identity(),
        payload_qualification=payload))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
               '-m', 'probes.rule_stability', 'rank-entry', '--config', str(config_path.resolve()), '--output', str(output.resolve())]
    env = dict(os.environ, OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    if not native_released:
        env['CUDA_VISIBLE_DEVICES'] = ''
    begin = time.monotonic()
    with (output / 'ranks.log').open('x') as stream:
        result = subprocess.run(command, cwd=a.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    a.write(output / 'process-exit.json', dict(argv=command, exit_code=result.returncode,
        seconds=time.monotonic() - begin, process_owner='torchrun parent; no engine child or background runner',
        retry='no_automatic_relaunch'))
    if result.returncode:
        raise RuntimeError(f'rank job exit {result.returncode}; preserve {output}/ranks.log')
    from .runner import readback
    readback(config_path, output)
    if native_released and config['mode'] == 'qualification':
        reload_output = output / 'fresh-reload'
        reload_output.mkdir(exist_ok=False)
        reload_command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            '-m', 'probes.rule_stability', 'reload-rank', '--config', str(config_path.resolve()),
            '--output', str(reload_output.resolve())]
        with (reload_output / 'ranks.log').open('x') as stream:
            reload_result = subprocess.run(reload_command, cwd=a.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        a.write(reload_output / 'process-exit.json', dict(argv=reload_command, exit_code=reload_result.returncode,
            retry='no_automatic_relaunch', owner='same qualification invocation'))
        if reload_result.returncode:
            raise RuntimeError(f'fresh reload exit {reload_result.returncode}; preserve logs')
        from .runner import reload_readback
        reload_readback(config_path, reload_output)
    if native_released:
        # Full base bytes were hashed once at entry; terminal stat check detects live mutation.
        for name, identity in payload.items():
            stat = Path(name).stat()
            if stat.st_size != identity['size_bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
                raise ValueError(f'qualified payload changed during invocation: {name}')
        from src.artifacts.git_identity import verify_source_identity
        verify_source_identity(config['source'], required_paths=a.source_paths(), root=a.ROOT)
    else:
        a.verify_candidate_source(config['source'])
    a.write(output / 'terminal.json', dict(status='complete', job_exit_code=result.returncode,
        seconds=time.monotonic() - begin, readback_sha256=a.digest(output / 'readback.json'),
        cleanup='all torchrun rank processes settled before readback'))


def qualify_payload_once():
    from probes import iterative_positive as p
    policy = p.load(p.POLICY)
    checkpoint = Path(policy['checkpoint'])
    files = {}
    for raw, expected in policy['payload_sha256'].items():
        path = Path(raw)
        if path.is_relative_to(checkpoint):
            path = a.ANCHOR / path.relative_to(checkpoint)
        if a.digest(path) != expected:
            raise ValueError(f'base/anchor payload differs: {path}')
        stat = path.stat()
        files[str(path)] = dict(sha256=expected, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return files


def prepare(output):
    from src.artifacts.git_identity import capture_source_identity
    from .data import request_records, balanced_layout
    inputs, images = input_binding()
    _, manifest = __import__('probes.rule_stability.data', fromlist=['load_inputs']).load_inputs()
    source = capture_source_identity(a.source_paths(), root=a.ROOT)
    directory = output
    directory.mkdir(parents=True, exist_ok=False)
    layout = balanced_layout(request_records(images, manifest))
    a.write(directory / 'candidate.json', dict(schema=a.SCHEMA, source=source, inputs=inputs,
        anchor=a.verify_anchor(hash_payload=False), runtime=runtime_identity(), rank_layout=layout,
        status='CPU_CANDIDATE_NATIVE_UNRELEASED'))
    for mode, arms in [('qualification', ('B',)), ('primary', ('A', 'B'))]:
        for arm in arms:
            target = a.OUTPUT / f'native-{mode}-{arm}-01'
            config = config_for(target, engine='native', arm=arm, updates=1 if mode == 'qualification' else 16, source=source)
            config.update(released=False, lead_thread=a.LEAD, mode=mode, retry='no_automatic_relaunch',
                operational_observation_seconds=3600 if mode == 'qualification' else None,
                cleanup_owner='persistent worker owns torchrun invocation and checks exact process settlement',
                normal_completion='declared updates then final greedy and complete fresh readback; no quality gate')
            config['technical_diagnostic'] = dict(a.TECHNICAL_DIAGNOSTIC) if mode == 'qualification' else None
            a.write(directory / f'{mode}-{arm}-proposal.json', config)
    a.write(directory / 'work-bounds.json', dict(
        native_qualification=dict(arms=1, updates=1, greedy_requests=36, sampled_requests=18,
            additional_fresh_checkpoint_reload_greedy_requests=18, positive_replays=18,
            geometry_replays_max=18, duplicate_replays=18,
            technical_generation_requests=1, technical_generated_actions=6,
            technical_replay_requests=1, total_generation_requests=73,
            acquisition_actions_max=222054),
        primary=dict(arms=2, updates_per_arm=16, greedy_requests=612, sampled_requests=576,
            acquisition_actions_max=3663792, positive_replays=576, geometry_replays_max=576,
            duplicate_replays=288, checkpoint_exports=10),
        rank_layout=layout['ranks'], per_rank_images_max=3, horizon=3084,
        prompt_tokens_max=1372, replay_context_max=4456,
        acquisition_forward_calls_max=3663792,
        processor_materializations_per_arm=36, native_input_materializations_per_arm=18,
        positive_encodings_per_arm=18,
        policy_score_slabs='each FP32 [3084,152670] slab <=1883337120 bytes; HF raw and processed histories retained until trace conversion',
        sample_factor_graph='1000x2048 effective FP32 rows and FP64 norm inputs; current-weight factors refreshed each replay',
        checkpoint_disk_estimate_bytes=3000000000,
        raw_trace_and_diagnostic_disk_estimate_bytes=1500000000,
        timing_and_peak_memory='unmeasured native; qualify startup/acquisition/replay/save/readback and max rank skew before primary release',
        baseline_refresh_exports=0, resident_engine_refresh=0,
        reason='resident native HF model itself is current policy; no hardcoded-greedy vLLM reuse',
        active_jobs=[]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('cpu-smoke', 'rank-entry', 'reload-rank', 'readback', 'prepare', 'native-run', 'compare'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('A', 'B'), default='B')
    parser.add_argument('--a-run', type=Path)
    parser.add_argument('--a-config', type=Path)
    parser.add_argument('--b-run', type=Path)
    parser.add_argument('--b-config', type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != a.ROOT:
        raise ValueError('select the canonical research checkout')
    if args.command == 'cpu-smoke':
        # Generated config is evidence; it is not executable source or native release.
        config_path = args.output.parent / (args.output.name + '-config.json')
        config = config_for(args.output, engine='cpu_fixture', arm=args.arm, updates=1, source=a.candidate_source())
        a.write(config_path, config)
        launch(config_path, args.output, native_released=False)
    elif args.command == 'rank-entry':
        from .runner import run_rank
        run_rank(args.config, args.output)
    elif args.command == 'readback':
        from .runner import readback
        readback(args.config, args.output)
    elif args.command == 'reload-rank':
        from .runner import reload_rank
        reload_rank(args.config, args.output)
    elif args.command == 'prepare':
        prepare(args.output)
    elif args.command == 'compare':
        from .consumer import compare_arms
        left, right = a.load(args.a_config), a.load(args.b_config)
        if left['arm'] != 'A' or right['arm'] != 'B':
            raise ValueError('comparison requires declared A and B arms')
        for key in ('source', 'inputs', 'protocol', 'anchor', 'runtime', 'engine', 'updates', 'frozen'):
            if left[key] != right[key]:
                raise ValueError(f'comparison identity differs: {key}')
        metrics = []
        for config, root in [(left, args.a_run), (right, args.b_run)]:
            receipt = a.load(root / 'readback.json')
            if (receipt['status'] != 'complete' or receipt['arm'] != config['arm'] or
                    receipt['updates'] != config['updates'] or receipt['engine'] != config['engine'] or
                    receipt['metrics_sha256'] != a.digest(root / 'metrics.json')):
                raise ValueError('comparison requires unchanged completed readback')
            metrics.append(a.load(root / 'metrics.json'))
        result = compare_arms(*metrics)
        result.update(source=left['source'], engine=left['engine'], scientific_evidence=left['engine'] == 'native',
            artifacts={name: dict(run=str(root.resolve()), metrics_sha256=a.digest(root / 'metrics.json'))
                       for name, root in [('A', args.a_run), ('B', args.b_run)]})
        a.write(args.output, result)
    else:
        packet = a.load(args.config)
        a.validate_packet(packet, args.output, mode=packet.get('mode'))
        if packet['runtime'] != runtime_identity():
            raise ValueError('runtime differs from release')
        payload = qualify_payload_once()
        a.verify_anchor(hash_payload=False)
        # Runtime binding is immutable and fresh for this invocation.
        launch(args.config, args.output, native_released=True, payload=payload)


if __name__ == '__main__':
    main()
