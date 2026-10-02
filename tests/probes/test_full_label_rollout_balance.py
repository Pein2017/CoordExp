from copy import deepcopy
from dataclasses import dataclass

import pytest

from probes.full_label_fit.rollout import (
    MAX_REQUESTS_PER_RANK,
    POLICY,
    WORLD_SIZE,
    align_results,
    build_assignment,
    rank_image_ids,
    route_records,
    verify_assignment,
)


def records_for(version=0):
    producer = dict(kind='live_online', update=version, rollout_policy=POLICY)
    return [dict(image_id=i, producer=producer, token_ids=[i] * (10 + i),
                 generated_tokens=10 + i, raw_identity=f'{i:064x}',
                 prompt_token_ids=[i] * (100 + (i % 4))) for i in range(18)]


def test_assignment_is_deterministic_bounded_and_uses_prior_rollout_length():
    ids = list(range(18))
    prompt_lengths = {i: 100 + i % 4 for i in ids}
    previous = records_for()
    first = build_assignment(ids, prompt_lengths, None, 0)
    second = build_assignment(ids, prompt_lengths, previous, 1)

    assert first == build_assignment(reversed(ids), prompt_lengths, None, 0)
    assert second == build_assignment(ids, prompt_lengths, previous, 1)
    assert second['policy'] == POLICY and second['previous_version'] == 0
    assert len(second['items']) == 18
    assert all(1 <= len(row['image_ids']) <= MAX_REQUESTS_PER_RANK for row in second['ranks'])
    assert {i for rank in second['ranks'] for i in rank['image_ids']} == set(ids)
    assert all(second['items'][i]['previous_generated_tokens'] == previous[i]['generated_tokens']
               for i in ids)
    assert all(row['estimated_tokens'] == row['prompt_tokens'] + row['previous_generated_tokens']
               for row in second['items'])
    assert all(len(rank_image_ids(second, rank)) == len(second['ranks'][rank]['image_ids'])
               for rank in range(WORLD_SIZE))


def test_assignment_reduces_the_known_old_stride_rank_7_skew():
    ids = list(range(18))
    prompt_lengths = {i: 100 + i % 4 for i in ids}
    previous = records_for()
    for image_id, generated in ((7, 1900), (15, 1700)):
        previous[image_id]['token_ids'] = [image_id] * generated
        previous[image_id]['generated_tokens'] = generated
    # The existing learner stride assigns both long trajectories (7 and 15) to rank 7.
    old_stride_loads = [sum(prompt_lengths[i] + previous[i]['generated_tokens']
                            for i in ids[rank::WORLD_SIZE]) for rank in range(WORLD_SIZE)]
    balanced = build_assignment(ids, prompt_lengths, previous, 1)
    self_reported_max = max(row['estimated_tokens'] for row in balanced['ranks'])
    assert 7 in ids[7::WORLD_SIZE] and 15 in ids[7::WORLD_SIZE]
    assert balanced['predicted_max_tokens'] == self_reported_max
    assert self_reported_max < max(old_stride_loads)


@pytest.mark.parametrize('mutate,version', [
    (lambda rows: rows[:-1], 1),
    (lambda rows: rows + [deepcopy(rows[0])], 1),
    (lambda rows: [dict(row, producer=dict(row['producer'], update=7)) for row in rows], 1),
    (lambda rows: [dict(row, generated_tokens=3085) if row['image_id'] == 0 else row for row in rows], 1),
])
def test_assignment_fails_closed_on_invalid_previous_rollout(mutate, version):
    rows = mutate(records_for())
    with pytest.raises(ValueError):
        build_assignment(list(range(18)), {i: 100 for i in range(18)}, rows, version)


def test_assignment_rejects_boolean_version_and_prompt_length():
    with pytest.raises(ValueError):
        build_assignment(list(range(18)), {i: 100 for i in range(18)}, None, True)
    with pytest.raises(ValueError):
        build_assignment(list(range(18)), {**{i: 100 for i in range(17)}, 17: True}, None, 0)


@dataclass
class Request:
    request_id: str


@dataclass
class Result:
    request_id: str


def test_results_align_by_unique_request_id_not_return_order():
    requests = [Request('a'), Request('b'), Request('c')]
    results = [Result('c'), Result('a'), Result('b')]
    assert [align_results(requests, results)[request.request_id].request_id for request in requests] == [
        'a', 'b', 'c']
    with pytest.raises(ValueError):
        align_results(requests, [Result('a'), Result('b'), Result('extra')])
    with pytest.raises(ValueError):
        align_results(requests, [Result('a'), Result('a'), Result('c')])
    with pytest.raises(ValueError):
        align_results(requests, [Result('a'), Result('b')])
    with pytest.raises(ValueError):
        align_results(requests, [Result('a'), Result('b'), object()])


def test_records_route_back_to_fixed_learner_order_and_reject_coverage_drift():
    rows = [dict(image_id=i) for i in (3, 1, 2)]
    assert route_records(rows, [1, 2, 3]) == [dict(image_id=i) for i in (1, 2, 3)]
    with pytest.raises(ValueError):
        route_records(rows, [1, 2, 4])


def test_saved_assignment_and_each_record_dispatch_are_recomputed():
    ids = list(range(18))
    prompt_lengths = {i: 100 + i % 4 for i in ids}
    producer = dict(kind='live_online', update=1, rollout_policy=POLICY)
    previous = records_for()
    assignment = build_assignment(ids, prompt_lengths, previous, 1)
    records = []
    for item in assignment['items']:
        rank = item['generation_rank']
        batch_ids = assignment['ranks'][rank]['image_ids']
        records.append(dict(image_id=item['image_id'], producer=producer,
                            generation_rank=rank,
                            generation_batch_index=item['generation_batch_index'],
                            generation_batch_size=len(batch_ids),
                            generation_batch_image_ids=batch_ids))
    assert verify_assignment(assignment, ids, prompt_lengths, previous, records, producer) == assignment

    bad_schedule = deepcopy(assignment)
    bad_schedule['ranks'][0]['estimated_tokens'] += 1
    with pytest.raises(ValueError):
        verify_assignment(bad_schedule, ids, prompt_lengths, previous, records, producer)
    bad_record = deepcopy(records)
    bad_record[0]['generation_rank'] = (bad_record[0]['generation_rank'] + 1) % WORLD_SIZE
    with pytest.raises(ValueError):
        verify_assignment(assignment, ids, prompt_lengths, previous, bad_record, producer)


def test_persisted_readback_and_offline_accept_then_reject_resigned_rollout_drift(tmp_path):
    from test_online_full_label_region import FullLabelRegionTest
    from probes import online_row_credit as online

    FullLabelRegionTest.setUpClass()
    fixture = FullLabelRegionTest()
    output, root, training, images, inputs, qual = fixture.persisted_full_label_fixture(
        tmp_path, rollout_policy=POLICY)
    original_load = online.p.load

    def load(path):
        path = __import__('pathlib').Path(path)
        if path == training:
            return [images[image_id] for image_id in sorted(images)]
        if path == online.INPUTS:
            return inputs
        return original_load(path)

    kwargs = dict(geometry_weight=.1, start_checkpoint=root / 'anchor',
                  recipe_sha256=online.identity(qual['correction']), correction_arm='treatment',
                  duplicate_weight=1, rollout_backend='vllm', schema_geometry=True,
                  activation_checkpointing=False, full_label_region=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(online.p, 'load', load)
        monkeypatch.setattr(online.r, 'frontend', lambda: fixture.q)
        monkeypatch.setattr('probes.full_label_fit.experiment.verify_inputs', lambda *_: None)
        monkeypatch.setattr('src.artifacts.git_identity.verify_source_identity', lambda *_args, **_kwargs: None)
        monkeypatch.setattr(online, 'verify_start_export', lambda *_: None)
        monkeypatch.setattr('probes.full_label_fit.experiment.evaluate_versions', lambda *_: {'accepted': True})

        # Both actual consumer entrypoints accept the fresh persisted schedule.
        online.readback(output, root, 2, **kwargs)
        assert original_load(output / 'readback.json')[0]['producer']['rollout_policy'] == POLICY
        online.offline(output, root, 2, **kwargs)
        assert original_load(output / 'offline-results.json') == {'accepted': True}

        # Change per-record ownership, then resign the raw shard receipt and frozen
        # inventory so readback must detect the semantic dispatch mismatch itself.
        record_path = output / 'rollout-0' / 'rank-0' / '0.json'
        raw_complete_path = record_path.parent / 'complete.json'
        frozen_path = record_path.parents[1] / 'frozen.json'
        record_bytes, raw_complete_bytes, frozen_bytes = (
            record_path.read_bytes(), raw_complete_path.read_bytes(), frozen_path.read_bytes())
        record = original_load(record_path)
        record['generation_rank'] = (record['generation_rank'] + 1) % WORLD_SIZE
        record_path.write_text(online.p.canonical(record) + '\n')
        raw_complete = original_load(raw_complete_path)
        raw_complete['artifacts'][record_path.name] = online.p.digest(record_path)
        raw_complete_path.write_text(online.p.canonical(raw_complete) + '\n')
        rollout_root = record_path.parents[1]
        frozen = {str(path.relative_to(rollout_root)): online.p.digest(path)
                  for path in sorted(rollout_root.rglob('*.json')) if path.name != 'frozen.json'}
        frozen_path.write_text(online.p.canonical(frozen) + '\n')
        (output / 'readback.json').unlink()
        with pytest.raises(ValueError, match='dispatch metadata'):
            online.readback(output, root, 2, **kwargs)
        assert not (output / 'readback.json').exists()
        record_path.write_bytes(record_bytes)
        raw_complete_path.write_bytes(raw_complete_bytes)
        frozen_path.write_bytes(frozen_bytes)
        online.readback(output, root, 2, **kwargs)

        # Change the same saved assignment on every rank and resign each root
        # completion map. Offline must recompute the schedule from frozen rollouts.
        name = 'rollout-assignment-1.json'
        for rank in range(WORLD_SIZE):
            directory = output / f'rank-{rank}'
            assignment_path = directory / name
            assignment = original_load(assignment_path)
            assignment['predicted_max_tokens'] += 1
            assignment_path.write_text(online.p.canonical(assignment) + '\n')
            receipt_path = directory / 'complete.json'
            receipt = original_load(receipt_path)
            receipt['artifacts'][name] = online.p.digest(assignment_path)
            receipt_path.write_text(online.p.canonical(receipt) + '\n')
        with pytest.raises(ValueError, match='saved rollout assignment'):
            online.offline(output, root, 2, **kwargs)


def test_global_records_route_to_ordered_learner_subsets_and_reject_corruption():
    rows = [dict(image_id=i) for i in reversed(range(18))]
    for rank in (0, 7):
        local = list(range(18))[rank::8]
        assert [row['image_id'] for row in route_records(rows, local)] == local
    assert [row['image_id'] for row in route_records(rows, [16, 0, 8])] == [16, 0, 8]
    for corrupted in (rows[:-1], rows + [rows[-1]], rows + [rows[0]], rows + [None], rows + [{}], rows + [{'image_id': True}]):
        with pytest.raises(ValueError):
            route_records(corrupted, [0, 8, 16])
    for local in ([0, 0], [True], [19]):
        with pytest.raises(ValueError):
            route_records(rows, local)
