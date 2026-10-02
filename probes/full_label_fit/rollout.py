"""Deterministic, rollout-only work assignment for full-label fitting."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


POLICY = 'previous_rollout_tokens_lpt_v1'
SCHEMA = 'coordexp-full-label-rollout-assignment-v1'
WORLD_SIZE = 8
MAX_REQUESTS_PER_RANK = 3
MAX_GENERATED_TOKENS = 3084


def _ids(image_ids):
    ids = list(image_ids)
    if len(ids) != 18 or any(type(value) is not int for value in ids) or len(set(ids)) != 18:
        raise ValueError('rollout assignment requires 18 unique integer image IDs')
    return sorted(ids)


def build_assignment(image_ids, prompt_lengths, previous_records, version):
    """Build a pure LPT dispatch plan from prompt lengths and the prior rollout only."""
    if type(version) is not int or version < 0:
        raise ValueError('rollout version must be a nonnegative integer')
    ids = _ids(image_ids)
    if not isinstance(prompt_lengths, Mapping) or set(prompt_lengths) != set(ids):
        raise ValueError('prompt lengths must cover exactly the rollout image IDs')
    for image_id, length in prompt_lengths.items():
        if type(length) is not int or length <= 0:
            raise ValueError(f'invalid prompt length for image {image_id}')

    prior = {}
    if version == 0:
        if previous_records not in (None, (), []):
            raise ValueError('version zero has no previous rollout')
    else:
        if not isinstance(previous_records, Sequence) or isinstance(previous_records, (str, bytes)):
            raise ValueError('later rollout versions require the preceding same-run records')
        if len(previous_records) != len(ids):
            raise ValueError('previous rollout must cover all images exactly once')
        producer = None
        for row in previous_records:
            if not isinstance(row, Mapping):
                raise ValueError('previous rollout records must be mappings')
            image_id = row.get('image_id')
            if type(image_id) is not int or image_id not in prompt_lengths or image_id in prior:
                raise ValueError('previous rollout image IDs are missing, duplicated, or unexpected')
            current_producer = row.get('producer')
            if (not isinstance(current_producer, Mapping) or current_producer.get('kind') != 'live_online'
                    or current_producer.get('update') != version - 1
                    or current_producer.get('rollout_policy') != POLICY):
                raise ValueError('previous rollout is not the immediately preceding balanced version')
            if producer is None:
                producer = current_producer
            elif current_producer != producer:
                raise ValueError('previous rollout producer is mixed')
            token_ids = row.get('token_ids')
            generated = row.get('generated_tokens')
            raw_identity = row.get('raw_identity')
            prompt = row.get('prompt_token_ids')
            if (not isinstance(token_ids, (list, tuple)) or type(generated) is not int
                    or generated != len(token_ids) or not 0 <= generated <= MAX_GENERATED_TOKENS):
                raise ValueError(f'invalid previous generated-token length for image {image_id}')
            if not isinstance(prompt, (list, tuple)) or len(prompt) != prompt_lengths[image_id]:
                raise ValueError(f'previous prompt length drift for image {image_id}')
            if (not isinstance(raw_identity, str) or len(raw_identity) != 64
                    or any(char not in '0123456789abcdef' for char in raw_identity)):
                raise ValueError(f'invalid previous raw identity for image {image_id}')
            prior[image_id] = (generated, raw_identity)
        if set(prior) != set(ids):
            raise ValueError('previous rollout image coverage differs')

    items = []
    for image_id in ids:
        generated, raw_identity = prior.get(image_id, (0, None))
        prompt = prompt_lengths[image_id]
        items.append(dict(image_id=image_id, prompt_tokens=prompt,
                          previous_generated_tokens=generated,
                          previous_raw_identity=raw_identity,
                          estimated_tokens=prompt + generated))

    ranked = sorted(items, key=lambda row: (-row['estimated_tokens'], row['image_id']))
    rank_items = [[] for _ in range(WORLD_SIZE)]
    loads = [0] * WORLD_SIZE
    for item in ranked:
        candidates = [rank for rank in range(WORLD_SIZE)
                      if len(rank_items[rank]) < MAX_REQUESTS_PER_RANK]
        if not candidates:
            raise ValueError('rollout request capacity exceeded')
        rank = min(candidates, key=lambda value: (loads[value], len(rank_items[value]), value))
        rank_items[rank].append(item['image_id'])
        loads[rank] += item['estimated_tokens']
        item['generation_rank'] = rank
        item['generation_batch_index'] = len(rank_items[rank]) - 1

    if any(not rows for rows in rank_items):
        raise ValueError('rollout assignment must leave every rank nonempty')
    by_id = {row['image_id']: row for row in items}
    return dict(schema=SCHEMA, policy=POLICY, version=version,
                previous_version=None if version == 0 else version - 1,
                world_size=WORLD_SIZE, max_requests_per_rank=MAX_REQUESTS_PER_RANK,
                items=[by_id[image_id] for image_id in ids],
                ranks=[dict(rank=rank, image_ids=list(rank_items[rank]),
                            estimated_tokens=loads[rank]) for rank in range(WORLD_SIZE)],
                predicted_max_tokens=max(loads))


def rank_image_ids(assignment, rank):
    if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
        raise ValueError('generation rank is outside the rollout world')
    if assignment.get('schema') != SCHEMA or assignment.get('policy') != POLICY:
        raise ValueError('unsupported rollout assignment')
    row = assignment['ranks'][rank]
    if row.get('rank') != rank or not 1 <= len(row.get('image_ids', ())) <= MAX_REQUESTS_PER_RANK:
        raise ValueError('invalid per-rank rollout request batch')
    return list(row['image_ids'])


def align_results(requests, results):
    """Match vLLM results to dispatched NativeRequest IDs; never rely on list order."""
    expected = [getattr(request, 'request_id', None) for request in requests]
    actual = [getattr(result, 'request_id', None) for result in results]
    if not expected or len(expected) > MAX_REQUESTS_PER_RANK:
        raise ValueError('balanced rollout batch must contain one to three requests')
    if any(not isinstance(value, str) or not value for value in expected + actual):
        raise ValueError('rollout request/result IDs must be nonempty strings')
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(expected) != set(actual):
        raise ValueError('vLLM result request IDs differ from dispatched requests')
    return {result.request_id: result for result in results}


def route_records(records, learner_image_ids):
    """Select the learner partition, in its existing order, from global results."""
    expected = list(learner_image_ids)
    if any(type(image_id) is not int for image_id in expected) or len(expected) != len(set(expected)):
        raise ValueError('learner image IDs must be unique integers')
    by_id = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError('generated rollout records must be mappings')
        image_id = record.get('image_id')
        if type(image_id) is not int or image_id in by_id:
            raise ValueError('generated rollout records have duplicate or invalid image IDs')
        by_id[image_id] = record
    if not set(expected).issubset(by_id):
        raise ValueError('generated rollout coverage differs from learner partition')
    return [by_id[image_id] for image_id in expected]


def verify_assignment(saved, image_ids, prompt_lengths, previous_records, records, producer):
    """Recompute and validate a saved schedule and all per-record dispatch metadata."""
    expected = build_assignment(image_ids, prompt_lengths, previous_records, saved.get('version'))
    if saved != expected:
        raise ValueError('saved rollout assignment differs from recomputed prior-version schedule')
    if (producer.get('kind') != 'live_online' or producer.get('update') != expected['version']
            or producer.get('rollout_policy') != POLICY):
        raise ValueError('current producer differs from balanced rollout assignment')
    ids = _ids(image_ids)
    if len(records) != len(ids) or {row.get('image_id') for row in records} != set(ids):
        raise ValueError('generated records do not cover the balanced assignment exactly once')
    if any(row.get('producer') != producer for row in records):
        raise ValueError('generated rollout producer is mixed')
    entries = {row['image_id']: row for row in expected['items']}
    rank_batches = {row['rank']: row['image_ids'] for row in expected['ranks']}
    for row in records:
        item = entries[row['image_id']]
        rank = item['generation_rank']
        batch_ids = rank_batches[rank]
        if (row.get('generation_rank'), row.get('generation_batch_index'),
                row.get('generation_batch_size'), row.get('generation_batch_image_ids')) != (
                rank, item['generation_batch_index'], len(batch_ids), batch_ids):
            raise ValueError(f'generation dispatch metadata differs for image {row["image_id"]}')
    return expected
