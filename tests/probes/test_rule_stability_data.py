"""Frozen full570, positive atoms, and offline category-credit contracts."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from probes.rule_stability import data, consumer
from probes import rollout_row_credit as retained
from src.losses.vocab import build_token_vocabulary_groups


@pytest.fixture(scope='module')
def inputs():
    return data.load_inputs()


@pytest.fixture(scope='module')
def frontend():
    torch.set_num_threads(2)
    return retained.frontend()


def record(image, objects=()):
    text = ''.join(
        f"<|object_ref_start|>{obj['desc']}<|object_ref_end|><|box_start|>"
        + ''.join(f'<|coord_{v}|>' for v in obj['bbox_2d']) + '<|box_end|>'
        for obj in objects) + '<|im_end|>'
    return dict(image_id=image['image_id'], request_id=f"fixture:{image['image_id']}",
                width=image['width'], height=image['height'],
                crop=[0, 0, image['width'], image['height']], text=text,
                token_ids=[151645], generated_tokens=1, stop_reason='im_end')


def test_frozen_inputs_include_all570_and_formerly_missing_owners(inputs):
    images, manifest = inputs
    assert len(images) == 18 and sum(len(i['objects']) for i in images) == 570
    full = {(i['image_id'], o['coco_ann_id']) for i in images for o in i['objects']}
    old = {(i['image_id'], a) for i in manifest['images'] for a in i['retained_object_ids']}
    assert len(full - old) == 57
    requests = data.request_records(images, manifest)
    assert len(requests) == 18 and [r['image_id'] for r in requests] == [i['image_id'] for i in images]
    for image, request, binding in zip(images, requests, manifest['images'], strict=True):
        assert request['crop'] == [0, 0, image['width'], image['height']]
        assert request['prompt_token_ids'] == binding['prompt_token_ids']
        assert request['media_sha256'] == binding['media_sha256']
        assert request['image_grid_thw'] == binding['image_grid_thw']
    assert data.INPUT_MANIFEST_SHA256 == '51ad01b2cf599087e4d93b269a0abb9d5c6be83625e9fe9c5f8541731bd962ad'


def test_load_fails_closed_for_changed_labels_and_original_manifest(tmp_path, inputs):
    labels = Path(data.FULL_LABEL_PATH).read_bytes()
    manifest = Path(data.INPUT_MANIFEST_PATH).read_bytes()
    label_path, manifest_path = tmp_path / 'labels.json', tmp_path / 'manifest.json'
    label_path.write_bytes(labels)
    manifest_path.write_bytes(manifest)
    assert data.load_inputs(label_path, manifest_path)[0] == inputs[0]
    label_path.write_bytes(labels + b' ')
    with pytest.raises(ValueError, match='full-label'):
        data.load_inputs(label_path, manifest_path)
    label_path.write_bytes(labels)
    manifest_path.write_bytes(manifest + b' ')
    with pytest.raises(ValueError, match='manifest'):
        data.load_inputs(label_path, manifest_path)


@pytest.mark.parametrize('mutation', [
    lambda images, manifest: images[0]['objects'].pop(),
    lambda images, manifest: images[0]['objects'][0].update(coco_ann_id=999999999),
    lambda images, manifest: images[0]['objects'][0].update(bbox_2d=[10, 10, 10, 20]),
    lambda images, manifest: manifest['images'][0].update(image_id=True),
    lambda images, manifest: images[0].update(hidden_objects=[]),
])
def test_full_label_schema_rejects_denominator_and_identity_drift(inputs, mutation):
    images, manifest = deepcopy(inputs)
    mutation(images, manifest)
    with pytest.raises(ValueError):
        data.validate_inputs(images, manifest)


def test_real570_atoms_include_each_owner_and_exclude_terminal_eos(inputs, frontend):
    images, manifest = inputs
    bindings = {row['image_id']: row for row in manifest['images']}
    owners = []
    for image in images:
        encoded, sequence, row_ids = data.full_label_sequence(image, frontend)
        binding = bindings[image['image_id']]
        owners.extend((image['image_id'], int(oid.split(':', 1)[1])) for oid in row_ids)
        assert {int(oid.split(':', 1)[1]) for oid in row_ids} == {o['coco_ann_id'] for o in image['objects']}
        assert tuple(dict.fromkeys(a.object_id for a in sequence.atoms)) == row_ids
        assert all(a.object_id in row_ids and a.token_type != 'eos' for a in sequence.atoms)
        eos_id = frontend.tokenizer.convert_tokens_to_ids('<|im_end|>')
        eos_position = max(j for j, token in enumerate(sequence.input_ids) if token == eos_id)
        assert all(a.target_position < eos_position for a in sequence.atoms)
        assert list(sequence.input_ids[:len(binding['prompt_token_ids'])]) == binding['prompt_token_ids']
        assert list(encoded.image_encoding.image_grid_thw) == binding['image_grid_thw']
        assert all(sequence.input_ids[a.target_position] == a.token_id and
                   a.causal_logits_position == a.target_position - 1 for a in sequence.atoms)
    assert len(owners) == len(set(owners)) == 570


def test_request_records_bind_real_processor_prompt_media_and_grid(inputs, frontend):
    request = data.request_records(*inputs)[0]
    batch = retained.p.native_request(request, retained.p.load(retained.p.POLICY), frontend.processor)
    assert list(batch.prompt_token_ids[0]) == request['prompt_token_ids']
    assert list(batch.image_grids[0]) == request['image_grid_thw']
    assert batch.media_sha256[0] == request['media_sha256']


def test_normalized_positive_loss_uses_row_mean_and_has_no_eos_gradient(inputs, frontend):
    image = deepcopy(inputs[0][0])
    image['objects'] = [dict(image['objects'][0], desc='person'), dict(image['objects'][1], desc='traffic light')]
    _, sequence, row_ids = data.full_label_sequence(image, frontend)
    vocab = build_token_vocabulary_groups(frontend.token_identity, tokenizer=frontend.tokenizer)
    eos_id = frontend.tokenizer.convert_tokens_to_ids('<|im_end|>')
    eos_position = max(j for j, token in enumerate(sequence.input_ids) if token == eos_id)
    positions = tuple(a.causal_logits_position for a in sequence.atoms) + (eos_position - 1,)
    logits = torch.zeros(1, len(positions), vocab.vocab_size, requires_grad=True)
    # Unequal row token lengths and row logits distinguish row averaging from token averaging.
    with torch.no_grad():
        for j, atom in enumerate(sequence.atoms):
            logits[0, j, atom.token_id] = 2 if atom.object_id == row_ids[0] else -2
    loss, details = data.normalized_positive_objective(logits, sequence, row_ids, vocab, positions)
    expected, expected_rows = retained.retained_objective(logits, sequence, row_ids, vocab, positions)
    assert torch.equal(loss, expected) and details == expected_rows
    assert len(details) == 2 and loss.item() == pytest.approx(sum(d['loss'] for d in details) / 2)
    loss.backward()
    assert all(logits.grad[0, j].abs().sum() > 0 for j in range(len(sequence.atoms)))
    assert logits.grad[0, -1].abs().sum() == 0
    bad = replace(sequence, atoms=sequence.atoms[:-1])
    with pytest.raises(AssertionError):
        data.normalized_positive_objective(logits, bad, row_ids, vocab, positions)


def test_balanced_eighteen_layout_and_unequal_rank_gradient_weighting(inputs):
    records = data.request_records(*inputs)
    plan = data.balanced_layout(records)
    assert plan['world_size'] == 8 and plan['max_requests_per_rank'] == 3
    ranks = plan['ranks']
    flat = [i for rank in ranks for i in rank['image_ids']]
    assert len(flat) == len(set(flat)) == 18
    assert sorted(flat) == sorted(r['image_id'] for r in records)
    assert min(len(r['image_ids']) for r in ranks) >= 1
    assert max(len(r['image_ids']) for r in ranks) == 3
    assert plan == data.balanced_layout(list(reversed(records)))
    parameter = torch.tensor(1.0, requires_grad=True)
    weights = {i: 100 * len(rank['image_ids']) for rank in ranks for i in rank['image_ids']}
    ddp_average = sum(sum(parameter * weights[i] for i in rank['image_ids'])
                      * data.image_backward_scale() for rank in ranks) / 8
    ddp_average.backward()
    assert parameter.grad.item() == pytest.approx(sum(weights.values()) / 18)
    wrong = sum(sum(weights[i] for i in r['image_ids']) / len(r['image_ids']) for r in ranks) / 8
    assert wrong != pytest.approx(sum(weights.values()) / 18)


def test_evaluate_projects_arm_without_mutation_and_tracks_own_baseline(inputs):
    images, _ = inputs
    target_image = images[0]
    first, second = target_image['objects'][:2]
    versions = {
        0: [record(i, [first] if i == target_image else []) for i in images],
        '1': [record(i, [second] if i == target_image else []) for i in images],
    }
    raw = deepcopy(versions)
    result = consumer.evaluate(images, versions)
    assert versions == raw and all('arm' not in r for rows in versions.values() for r in rows)
    assert set(result['scored']) == {'0', '1'}
    assert result['scored']['0']['totals']['denominator'] == 570
    assert result['transitions']['1']['gained'] == [[target_image['image_id'], second['coco_ann_id']]]
    assert result['transitions']['1']['lost'] == [[target_image['image_id'], first['coco_ann_id']]]
    assert result['scored']['0']['images'][target_image['image_id']]['category_correct_ids'] == [first['coco_ann_id']]


def test_category_credit_follows_geometry_assignment(inputs):
    images, _ = inputs
    image = images[0]
    gt = image['objects'][0]
    wrong_category = dict(gt, desc='elephant' if gt['desc'] != 'elephant' else 'person')
    versions = {0: [record(i, [wrong_category] if i == image else []) for i in images]}
    result = consumer.evaluate(images, versions)['scored']['0']
    row = result['images'][image['image_id']]
    assert gt['coco_ann_id'] in row['geometric_match_ids']
    assert row['category_correct_ids'] == [] and row['tp'] == 0
    assert row['category_disagreements'] == 1 and result['totals']['fn'] == 570


def test_category_cannot_select_a_less_overlapping_competing_prediction(inputs):
    images, _ = inputs
    image, gt = images[0], images[0]['objects'][0]
    wrong = dict(gt, desc='elephant')
    correct = deepcopy(gt)
    correct['bbox_2d'][2] -= 1
    versions = {0: [record(i, [wrong, correct] if i == image else []) for i in images]}
    scored = consumer.evaluate(images, versions)['scored']['0']
    row = scored['images'][image['image_id']]
    assert row['geometric_match_ids'] == [gt['coco_ann_id']]
    assert row['category_correct_ids'] == []
    assert row['category_disagreements'] == 1 and scored['totals']['tp'] == 0


def test_consumer_records_per_image_and_pooled_rule_burdens(inputs):
    images, _ = inputs
    image, gt = images[0], images[0]['objects'][0]
    changed_category = dict(gt, desc='elephant')
    versions = {0: [record(i, [gt, gt, changed_category] if i == image else []) for i in images]}
    scored = consumer.evaluate(images, versions)['scored']['0']
    row = scored['images'][image['image_id']]
    assert row['rule_burdens']['duplicate_events'] == 2
    assert row['rule_burdens']['longest_event_burst'] == 2
    assert row['overlap_distribution']['pair_count'] == 3
    assert row['overlap_distribution']['strict_gt_09_pairs'] == 3
    assert scored['totals']['rule_burdens']['duplicate_events'] == 2
    assert scored['totals']['overlap_distribution']['mean'] == 1
    assert scored['totals']['rule_burdens']['empty'] == 17


@pytest.mark.parametrize('versions', [{1: []}, {0: [], '0': []}, {True: []}])
def test_consumer_rejects_missing_or_ambiguous_baseline(inputs, versions):
    with pytest.raises(ValueError):
        consumer.evaluate(inputs[0], versions)


@pytest.fixture(scope='module')
def comparison_metrics(inputs):
    images, _ = inputs
    image, first, second = images[0], images[0]['objects'][0], images[0]['objects'][1]
    # ArmA loses its one baseline owner; armB gains from its own empty baseline.
    arm_a = consumer.evaluate(images, {
        0: [record(i, [first] if i == image else []) for i in images],
        1: [record(i) for i in images],
    })
    arm_b = consumer.evaluate(images, {
        0: [record(i) for i in images],
        1: [record(i, [second, second] if i == image else []) for i in images],
    })
    return arm_a, arm_b


def test_compare_arms_reports_each_version_endpoint_and_own_baselines(comparison_metrics, monkeypatch):
    arm_a, arm_b = deepcopy(comparison_metrics)
    raw = deepcopy((arm_a, arm_b))
    monkeypatch.setattr(consumer, 'evaluate_versions', lambda *_: pytest.fail('comparison must not rescore'))
    monkeypatch.setattr(consumer, 'trajectory_diagnostics', lambda *_: pytest.fail('comparison must not reparse'))
    result = consumer.compare_arms(arm_a, arm_b)
    assert (arm_a, arm_b) == raw
    assert result['versions'] == ['0', '1'] and result['endpoint_version'] == '1'
    assert result['endpoint'] == result['per_version']['1']
    zero, end = result['per_version']['0'], result['per_version']['1']
    assert zero['metric_differences']['tp'] == -1
    assert end['metric_differences']['tp'] == 1
    assert end['metric_differences']['fn'] == -1
    assert end['rule_burden_differences']['duplicate_events'] == 1
    assert end['overlap_distribution_differences']['mean'] is None
    transitions = end['own_baseline_transitions']
    assert transitions['A'] == arm_a['transitions']['1']
    assert transitions['B'] == arm_b['transitions']['1']
    assert transitions['count_differences']['gained_count'] == 1
    assert transitions['count_differences']['lost_count'] == -1
    target = str(arm_a['denominator_annotation_ids'][0][0])
    assert target in end['images']
    # Returned transition ledgers are independent copies, not mutable raw-artifact aliases.
    transitions['A']['lost'].clear()
    assert (arm_a, arm_b) == raw


def test_compare_arms_keeps_all17_primary_versions(comparison_metrics):
    arms = []
    for arm in comparison_metrics:
        expanded = deepcopy(arm)
        for version in range(2, 17):
            expanded['scored'][str(version)] = deepcopy(expanded['scored']['1'])
            expanded['transitions'][str(version)] = deepcopy(expanded['transitions']['1'])
        expanded['greedy_versions'] = [str(v) for v in range(17)]
        arms.append(expanded)
    result = consumer.compare_arms(*arms)
    assert result['versions'] == [str(v) for v in range(17)]
    assert len(result['per_version']) == 17 and result['endpoint_version'] == '16'
    assert result['endpoint'] == result['per_version']['16']


@pytest.mark.parametrize('mutation', [
    lambda a, b: b['scored']['1']['totals'].update(denominator=569),
    lambda a, b: b['scored'].pop('1'),
    lambda a, b: b['scored'].update({'2': b['scored'].pop('1')}),
    lambda a, b: b.update(greedy_versions=['0']),
    lambda a, b: b.update(baseline_version='1'),
    lambda a, b: b['transitions'].pop('1'),
    lambda a, b: b['denominator_annotation_ids'][0].__setitem__(1, 999999999),
    lambda a, b: b['scored']['1']['totals'].update(f1=float('nan')),
    lambda a, b: b['scored']['1']['totals']['rule_burdens'].pop('duplicate_events'),
])
def test_compare_arms_fails_closed_on_incompatible_metrics(comparison_metrics, mutation):
    a, b = deepcopy(comparison_metrics)
    mutation(a, b)
    with pytest.raises(ValueError):
        consumer.compare_arms(a, b)
