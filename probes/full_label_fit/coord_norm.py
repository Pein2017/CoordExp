"""One fixed-checkpoint, matched resident-vLLM coordinate readout comparison."""
from __future__ import annotations

import argparse
import json
import os
import math
import resource
import time
from importlib.metadata import version
from pathlib import Path

from probes import online_row_credit as online
from probes.full_label_fit.experiment import evaluate_versions


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def validate_contract(contract):
    assert contract['schema'] == 'full-label-coordinate-norm-inference-v1'
    assert contract['conditions'] == ['off', 'median']
    assert contract['generation'] == dict(temperature=0, top_p=1, top_k=-1,
        repetition_penalty=1, min_tokens=0, cap=3084, seed=92711)
    assert contract['bounds'] == dict(ranks=8, requests=36, generated_tokens=111024,
        requests_per_rank=3, context=5032, visual=1024, whole_wall_seconds=900, cleanup_seconds=30)
    ids = contract['image_ids']
    assert len(ids) == len(set(ids)) == 18 and ids == sorted(ids)
    groups = contract['groups']
    assert len(groups) == 8 and all(1 <= len(group) <= 3 for group in groups)
    assert sorted(i for group in groups for i in group) == ids
    coordinates = contract['coordinate_ids']
    assert len(coordinates) == len(set(coordinates)) == 1000
    assert all(type(i) is int and i >= 0 for i in coordinates)
    for path, expected in contract['input_sha256'].items():
        assert online.p.digest(path) == expected, path
    assert load(Path(contract['checkpoint']) / 'identity.json') == contract['checkpoint_files']
    assert all(version(package) == expected for package, expected in contract['runtime'].items())
    return {row['image_id']: row for row in load(contract['input_path'])}


def validate_records(records, inputs, producer):
    assert len(records) == 18 and {r['image_id'] for r in records} == set(inputs)
    assert len({r['request_id'] for r in records}) == 18
    for record in records:
        assert record['producer'] == producer
        assert record['request_id'] == inputs[record['image_id']]['request_id']
        assert record['generated_tokens'] == len(record['token_ids']) <= 3084
        for key, value in inputs[record['image_id']].items():
            assert record[key] == value, key
        assert record['raw_identity'] == online.seal(record, producer)['raw_identity']


def validate_policy(policy, condition, contract):
    assert policy['mode'] == condition and policy['identity'] == contract['weight_identity']
    assert policy['calls'] > 0 and policy['coordinate_tokens'] == 1000
    assert policy['coordinate_ids'] == contract['coordinate_ids']
    first = policy['first_call']
    assert first['scaling_active'] == (condition == 'median')
    assert first['non_coordinate_unchanged'] is True
    if condition == 'median':
        assert all(math.isfinite(policy[k]) and policy[k] > 0 for k in
                   ('norm_min', 'norm_max', 'median_norm', 'factor_min', 'factor_max'))
        assert first['changed_coordinates'] > 0


def run(contract_path, output):
    import torch
    import torch.distributed as dist
    from probes.rollout_row_credit import frontend
    from src.qwen.vllm_rollout import VllmDoraRollout, validate_device_assignments

    contract = load(contract_path)
    inputs = validate_contract(contract)
    rank = int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE']) == 8 and int(os.environ['LOCAL_RANK']) == rank
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    directory = output / f'rank-{rank}'
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    q = frontend()
    assert q.model is None and str(q.base_model_path) == contract['base_model']
    assert q.base_config_sha256 == contract['base_config_sha256']
    assert q.tokenizer_sha256 == contract['tokenizer_sha256']
    ids = [q.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]
    assert ids == contract['coordinate_ids']
    group = contract['groups'][rank]
    requests = online.vllm_requests(q, inputs, group)
    identity = contract['weight_identity']
    try:
        with VllmDoraRollout(base_model=q.base_model_path, checkpoint=contract['checkpoint'],
                identity=identity, log_path=directory / 'vllm.log', device=rank, trainer_rank=rank,
                max_model_len=5032, max_num_seqs=3, seed=92711, timeout=870) as engine:
            device = dict(rank=rank, request=engine.device_request, startup=engine.startup)
            devices = [None] * 8
            dist.all_gather_object(devices, device)
            validate_device_assignments(devices, list(range(8)))
            write(directory / 'devices.json', devices)
            for condition in contract['conditions']:
                configured = engine.configure_coordinate_output_norm(condition, ids, identity=identity)
                assert configured['mode'] == condition and configured['identity'] == identity
                dist.barrier()
                begin = time.monotonic()
                results = engine.generate(requests, budgets=[3084] * len(group),
                    eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),
                    pad_token_id=q.tokenizer.pad_token_id, identity=identity, trace=True)
                from probes.full_label_fit.rollout import align_results
                aligned = align_results(requests, results)
                evidence = engine.receipts[-1]['coordinate_output_norm']
                validate_policy(evidence, condition, contract)
                producer = dict(kind='fixed_checkpoint_coord_norm_inference', condition=condition,
                    weight_identity=identity, source=contract['source'],
                    contract_sha256=online.p.digest(contract_path))
                rows = []
                for index, image_id in enumerate(group):
                    result = aligned[inputs[image_id]['request_id']]
                    rows.append(online.seal(dict(inputs[image_id], token_ids=list(result.token_ids),
                        text=q.tokenizer.decode(result.token_ids, skip_special_tokens=False),
                        generated_tokens=len(result.token_ids), stop_reason=result.stop_reason,
                        raw_logprobs=list(result.raw_logprobs), generation_rank=rank,
                        generation_batch_index=index, generation_batch_image_ids=group,
                        generation_batch_seconds=time.monotonic()-begin,
                        score_semantics=('native_head_scores' if condition == 'off' else
                            'native_head_scores_after_coordinate_median_norm'), seed=92711), producer))
                raw = output / condition / f'rank-{rank}'
                raw.mkdir(parents=True, exist_ok=False)
                for row in rows:
                    write(raw / f"{row['image_id']}.json", row)
                write(raw / 'complete.json', dict(status='complete', producer=producer,
                    image_ids=group, policy=evidence, artifacts={f"{r['image_id']}.json":
                        online.p.digest(raw / f"{r['image_id']}.json") for r in rows}))
                gathered = [None] * 8
                dist.all_gather_object(gathered, rows)
                validate_records([row for part in gathered for row in part], inputs, producer)
                dist.barrier()
            operations = list(engine.receipts)
        write(directory / 'complete.json', dict(status='complete', rank=rank,
            weights_unchanged_identity=identity, optimizer_steps=0, HF_forwards=0,
            seconds=time.monotonic()-started, operations=operations,
            rss_max_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            frontend=q.to_artifact_dict()))
    finally:
        dist.destroy_process_group()


def score(contract_path, output):
    contract = load(contract_path)
    inputs = validate_contract(contract)
    frozen = {}
    for condition, key in [('off', 'zero'), ('median', 'median')]:
        rows = online.frozen_records(output / condition, contract['image_ids'], freeze=True)
        expected_producer = dict(kind='fixed_checkpoint_coord_norm_inference', condition=condition,
            weight_identity=contract['weight_identity'], source=contract['source'],
            contract_sha256=online.p.digest(contract_path))
        validate_records(rows, inputs, expected_producer)
        for rank, group in enumerate(contract['groups']):
            complete = load(output / condition / f'rank-{rank}/complete.json')
            assert complete['image_ids'] == group and complete['producer'] == expected_producer
            policy = complete['policy']
            validate_policy(policy, condition, contract)
            local = [row for row in rows if row['generation_rank'] == rank]
            assert [row['image_id'] for row in sorted(local, key=lambda x:x['generation_batch_index'])] == group
        # The existing whole-image scorer requires arm metadata; raw acquisition stays immutable.
        assert all('arm' not in row or row['arm'] == 'greedy' for row in rows)
        frozen[key] = [dict(row, arm='greedy') for row in rows]
    for rank in range(8):
        receipt = load(output / f'rank-{rank}/complete.json')
        assert receipt['status'] == 'complete' and receipt['rank'] == rank
        assert receipt['weights_unchanged_identity'] == contract['weight_identity']
        assert receipt['optimizer_steps'] == receipt['HF_forwards'] == 0
        operations = receipt['operations']
        assert len(operations) == 4
        assert [op['coordinate_output_norm']['mode'] for op in operations] == ['off', 'off', 'median', 'median']
        assert all(op['identity'] == contract['weight_identity'] for op in operations)
    assert sum(r['generated_tokens'] for rows in frozen.values() for r in rows) <= 111024
    labels = load(contract['label_path'])
    paired = evaluate_versions(labels, frozen)
    old = online.frozen_records(Path(contract['reference_run']) / 'rollout-16', contract['image_ids'])
    drift = evaluate_versions(labels, {'zero': old, 'fresh_off': frozen['zero'], 'median': frozen['median']})
    write(output / 'metrics.json', dict(new_disabled_control=paired,
        old_endpoint16_reference=drift, pooling=False,
        disabled_token_drift_image_ids=[a['image_id'] for a,b in zip(sorted(old,key=lambda r:r['image_id']),
            sorted(frozen['zero'],key=lambda r:r['image_id'])) if a['token_ids'] != b['token_ids']],
        matcher='class-agnostic cardinality-first IoU>=.5 one-to-one, then exact-description agreement',
        scorer_projection='In-memory arm=greedy for existing whole-image parser; raw/producer/policy bytes unchanged',
        scope='One fixed-checkpoint policy contrast; no training or norm-origin causal claim'))
    write(output / 'readback.json', dict(status='complete', requests=36,
        tokens=sum(r['generated_tokens'] for rows in frozen.values() for r in rows),
        raw_frozen_sha256={condition:online.p.digest(output / condition / 'frozen.json')
                          for condition in contract['conditions']},
        metrics_sha256=online.p.digest(output / 'metrics.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'score'))
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == 'run' else score)(args.contract, args.output)


if __name__ == '__main__':
    main()
