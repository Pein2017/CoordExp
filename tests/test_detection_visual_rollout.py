from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from src.common.errors import ArtifactContractError
from src.inference.parsing import parse_compact_object_box_closed
from src.vis.matching import match_row
from src.vis.normalization import load_visual_rows


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _span(box, desc='person'):
    return '<|object_ref_start|>' + desc + '<|object_ref_end|><|box_start|>' + ''.join(f'<|coord_{v}|>' for v in box) + '<|box_end|>'


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return path


def _inputs(tmp_path):
    image_path = tmp_path / 'image.png'
    Image.new('RGB', (1248, 832), 'white').save(image_path)
    image = dict(image_id=7, image_path=str(image_path), image_sha256=_sha(image_path.read_bytes()), width=1248, height=832,
                 objects=[dict(desc='person', bbox_2d=[318, 19, 746, 527], coco_ann_id=91)])
    text = _span([318, 19, 746, 527]) + _span([700, 50, 300, 200]) + '<|object_ref_start|>broken' + _span([10, 10, 20, 20]) + '<|im_end|>'
    record = {k:v for k,v in image.items() if k != 'objects'}
    record.update(crop=[0, 0, 1248, 832], view_scale=1, request_id='rule-stability:B:16:greedy:7',
                  token_ids=list(range(50)), text=text, stop_reason='im_end', prompt_token_ids=[1, 2], media_sha256='a'*64,
                  image_grid_thw=[1, 2, 3], generated_tokens=50, raw_logprobs=[-1.0]*50, policy_logprobs=[-1.0]*50,
                  generation_rank=0, producer=dict(kind='rule_stability', engine='native', arm='B', channel='greedy'))
    identity_keys = ('producer', 'request_id', 'image_id', 'token_ids', 'text', 'prompt_token_ids', 'media_sha256', 'image_grid_thw', 'stop_reason')
    canonical = json.dumps({k:record[k] for k in identity_keys}, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    record['raw_identity'] = _sha(canonical.encode())
    labels = _write(tmp_path / 'labels.json', [image])
    path = tmp_path / 'rollout.jsonl'
    path.write_text(json.dumps(record) + '\n')
    return record, image, path, labels


def _refresh_identity(record):
    keys = ('producer', 'request_id', 'image_id', 'token_ids', 'text', 'prompt_token_ids', 'media_sha256', 'image_grid_thw', 'stop_reason')
    record['raw_identity'] = _sha(json.dumps({k:record[k] for k in keys}, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def test_explicit_native_rollout_preserves_sparse_raw_order_and_invalid_geometry(tmp_path):
    _, _, path, labels = _inputs(tmp_path)
    rows = load_visual_rows(path, input_format='rollout', labels_json=labels).rows
    row = rows[0]
    assert row.row_id == '7'
    assert [obj.index for obj in row.pred] == [0, 1, 3]
    assert [obj.geometry_valid for obj in row.pred] == [True, False, True]
    assert row.gt[0].bbox_pixel_xyxy == row.pred[0].bbox_pixel_xyxy == (397, 16, 931, 438)
    assert row.pred[1].source_bbox == (700, 50, 300, 200)
    assert row.pred[1].bbox_pixel_xyxy == (874, 42, 374, 166)
    assert row.pred[0].to_manifest()['source_metadata']['prediction_id'] == 'rule-stability:B:16:greedy:7:p0'
    assert row.gt[0].to_manifest()['source_metadata']['owner_id'] == '91'
    assert 'positions' not in row.pred[0].source_metadata
    assert row.source_metadata['malformed_outputs'][0]['generated_order'] == 2
    result = match_row(row)
    assert result.matched_pred_indices == {0}
    assert result.fp_pred_indices == (3,)
    assert result.invalid_pred_indices == (1,)
    assert result.stats()['invalid_pred'] == 1
    assert result.to_manifest()['invalid_pred_indices'] == [1]
    assert not result.duplicate_candidates


@pytest.mark.parametrize('field,value,code', [
    ('image_id', 99, 'vis.rollout_image_join'),
    ('width', 200, 'vis.rollout_image_join'),
    ('crop', [1, 0, 1248, 832], 'vis.rollout_view'),
    ('crop', [False, 0, 1248, 832], 'vis.rollout_view'),
    ('view_scale', 2, 'vis.rollout_view'),
    ('text', None, 'vis.rollout_field'),
    ('raw_logprobs', [float('nan')]*50, 'vis.rollout_nonfinite'),
    ('image_sha256', 'b'*64, 'vis.rollout_image_join'),
])
def test_rollout_rejects_contract_drift(tmp_path, field, value, code):
    record, _, path, labels = _inputs(tmp_path)
    record[field] = value
    path.write_text(json.dumps(record) + '\n')
    with pytest.raises(ArtifactContractError) as exc:
        load_visual_rows(path, input_format='rollout', labels_json=labels)
    assert exc.value.code == code


def test_explicit_format_does_not_relax_scored_reader(tmp_path):
    _, _, path, labels = _inputs(tmp_path)
    with pytest.raises(ArtifactContractError, match='gt_vs_pred_scored'):
        load_visual_rows(path)
    with pytest.raises(ArtifactContractError) as exc:
        load_visual_rows(path, input_format='rollout')
    assert exc.value.code == 'vis.rollout_labels_required'
    with pytest.raises(ArtifactContractError) as exc:
        load_visual_rows(path, input_format='guess', labels_json=labels)
    assert exc.value.code == 'vis.input_format'


def _bind_analysis(tmp_path, record, path):
    parsed = parse_compact_object_box_closed(record['text'], row_id=record['request_id'], row_index=0,
                                             image_width=record['width'], image_height=record['height'])
    complete = [dict(row, valid=True, order=row['generated_order'], bbox=row['coord_bins'], raw_text=row['raw_span_text']) for row in parsed.predictions]
    malformed = []
    for drop in parsed.dropped_predictions:
        if drop['reason'] == 'geometry_invalid':
            box = [int(s['text'][8:-2]) for s in drop['coord_token_spans']]
            complete.append(dict(drop, bbox=box, valid=False, order=drop['generated_order'], description='person'))
        else:
            malformed.append(dict(drop, censored=False))
    complete.sort(key=lambda x:x['order'])
    for i, row in enumerate(complete):
        row.update(positions=list(range(i*9, i*9+9)), completion_position=i*9+8, coordinate_positions=list(range(i*9+4, i*9+8)))
    analysis = dict(rows=complete, malformed=malformed, duplicate_events=[], burdens=dict(valid_rows=2, invalid_rows=1))
    raw_path = _write(tmp_path / 'source.json', record)
    analysis_path = _write(tmp_path / 'source-analysis.json', analysis)
    record = dict(record, visualization_provenance=dict(format='rule-stability-native-rollout-v1', coordinate_space='norm1000',
                  raw_source=dict(path=str(raw_path), sha256=_sha(raw_path.read_bytes())),
                  analysis_source=dict(path=str(analysis_path), sha256=_sha(analysis_path.read_bytes()))))
    path.write_text(json.dumps(record)+'\n')
    return record, analysis_path


def test_bound_saved_analysis_preserves_original_action_positions(tmp_path):
    record, _, path, labels = _inputs(tmp_path)
    _, analysis_path = _bind_analysis(tmp_path, record, path)
    row = load_visual_rows(path, input_format='rollout', labels_json=labels).rows[0]
    assert row.pred[1].source_metadata['positions'] == list(range(9, 18))
    assert row.pred[1].source_metadata['completion_position'] == 17
    assert row.source_metadata['provenance']['analysis_source']['path'] == str(analysis_path)
    assert row.source_metadata['original_research_diagnostics']['duplicate_events'] == []


@pytest.mark.parametrize('mutation', ['sha', 'span', 'positions', 'missing_raw', 'coordinate_space'])
def test_analysis_binding_fails_closed(tmp_path, mutation):
    record, _, path, labels = _inputs(tmp_path)
    bound, analysis_path = _bind_analysis(tmp_path, record, path)
    if mutation == 'sha':
        bound['visualization_provenance']['analysis_source']['sha256'] = '0'*64
    elif mutation == 'missing_raw':
        del bound['visualization_provenance']['raw_source']
    elif mutation == 'coordinate_space':
        bound['visualization_provenance']['coordinate_space'] = 'pixel'
    else:
        analysis = json.loads(analysis_path.read_text())
        if mutation == 'span':
            analysis['rows'][0]['char_end'] -= 1
        else:
            analysis['rows'][0]['positions'] = [9999]
        _write(analysis_path, analysis)
        bound['visualization_provenance']['analysis_source']['sha256'] = _sha(analysis_path.read_bytes())
    path.write_text(json.dumps(bound)+'\n')
    with pytest.raises(ArtifactContractError) as exc:
        load_visual_rows(path, input_format='rollout', labels_json=labels)
    assert exc.value.code.startswith('vis.rollout_')


def test_scored_invalid_box_remains_rejected_and_pixels_stay_pixels(tmp_path):
    _, image, _, _ = _inputs(tmp_path)
    run = tmp_path / 'scored'; run.mkdir()
    raw = dict(row_id='7', row_index=0, image_path=image['image_path'], image_width=1248, image_height=832,
               gt=[dict(description='person', bbox=[318, 19, 746, 527], owner_id='91')])
    (run / 'gt_vs_pred.jsonl').write_text(json.dumps(raw)+'\n')
    pred = dict(description='person', bbox=[397, 16, 931, 438], coord_bins=[1, 2, 3, 4])
    (run / 'gt_vs_pred_scored.jsonl').write_text(json.dumps(dict(raw, pred=[pred]))+'\n')
    row = load_visual_rows(run).rows[0]
    assert row.pred[0].bbox_pixel_xyxy == (397, 16, 931, 438)
    assert row.gt[0].source_metadata['owner_id'] == '91'
    pred['bbox'] = [700, 50, 300, 200]
    (run / 'gt_vs_pred_scored.jsonl').write_text(json.dumps(dict(raw, pred=[pred]))+'\n')
    with pytest.raises(ArtifactContractError) as exc:
        load_visual_rows(run)
    assert exc.value.code == 'vis.invalid_pred_bbox'


def test_assembler_retains_records_selected_order_and_refuses_collision(tmp_path):
    from probes.rule_stability.visualization import assemble_rollouts
    record, _, _, _ = _inputs(tmp_path)
    run = tmp_path / 'native'
    _write(run / 'rank-0/version-16/greedy-7.json', record)
    parsed = parse_compact_object_box_closed(record['text'], row_id=record['request_id'], row_index=0, image_width=1248, image_height=832)
    _write(run / 'rank-0/version-16/greedy-7-analysis.json', dict(rows=[], malformed=[]))
    output = tmp_path / 'assembled.jsonl'
    assemble_rollouts(run, version=16, image_ids=[7], output=output)
    assembled = json.loads(output.read_text())
    assert {k:v for k,v in assembled.items() if k!='visualization_provenance'} == record
    assert assembled['visualization_provenance']['raw_source']['path'] == str((run / 'rank-0/version-16/greedy-7.json').resolve())
    with pytest.raises(ArtifactContractError) as exc:
        assemble_rollouts(run, version=16, image_ids=[7], output=output)
    assert exc.value.code == 'vis.rollout_output_exists'
