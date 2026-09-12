import copy
import json

import pytest

from probes.dora_owner_learning import selective_preservation_eval as evaluator


@pytest.fixture
def packet(tmp_path,monkeypatch):
    original=dict(records=[dict(example_id=f'e{i}',split='train' if i<2 else 'guard',baseline={'gt':[]}) for i in range(18)],
        configs={},source_identity={},source_model={},selection={})
    path=tmp_path/'manifest.json';path.write_text(json.dumps(original))
    monkeypatch.setattr(evaluator,'POINT_EVAL',tmp_path);monkeypatch.setattr(evaluator,'ORIGINAL_SHA',evaluator.file_hash(path))
    p=copy.deepcopy(original);p.update(schema='selective_preservation.eval.v1',arm='soft-preservation-10',
        point_ce23=[dict(example_id=r['example_id'],split=r['split'],parsed={'gt':[]}) for r in original['records']])
    return p


def test_unchanged18_packet_rejects_selection_source_split_and_ids(packet):
    evaluator.validate_manifest(packet)
    for mutate in [lambda p:p['records'].pop(),lambda p:p['source_model'].update(wrong=True),
                   lambda p:p['point_ce23'][0].update(split='guard'),lambda p:p['point_ce23'][0].update(example_id='e1')]:
        bad=copy.deepcopy(packet);mutate(bad)
        with pytest.raises(ValueError):evaluator.validate_manifest(bad)


def test_new_fixed23_receipt_not_old_margin_stop(packet,monkeypatch,tmp_path):
    monkeypatch.setattr(evaluator,'ARM_ROOT',tmp_path)
    monkeypatch.setattr(evaluator,'validate_source_receipt',lambda r,p:r['adapter'])
    inputs=tmp_path/'training/inputs.json';inputs.parent.mkdir();inputs.write_text(json.dumps(dict(schema_version='selective_preservation.inputs.v1',lambda_kl=10.,updates=23)))
    receipt=dict(schema_version='selective_preservation.training.v1',status='completed',stop_reason='fixed_steps',updates=23,
        adapter={'root':str(tmp_path/'training/adapter')},lambda_kl=10.,inputs_sha256=evaluator.file_hash(inputs),final_scores={'example':{'A_vs_best_other_margin':-10.}})
    assert evaluator.validate_receipt(receipt,packet)==receipt['adapter']
    for change in [dict(schema_version='entrance_ce.training.v1'),dict(status='running'),dict(stop_reason='joint_margin'),
                   dict(updates=22),dict(adapter={'root':str(tmp_path/'wrong')}),dict(lambda_kl=1.),dict(inputs_sha256='wrong')]:
        with pytest.raises(ValueError):evaluator.validate_receipt(dict(receipt,**change),packet)


def test_target_box_keeps_native_index_when_projection_drops_invalid_category():
    parsed={'pred':[dict(description='',bbox=[0,0,1,1],coord_bins=[0,0,1,1]),
                    dict(description='person',bbox=[10,20,30,40],coord_bins=[11,22,33,44])]}
    score={t:{'matches':[dict(owner='owner',pred_index=0,iou=.9)]} for t in ('50','60','80')}
    target=evaluator.target_boxes(parsed,score,'owner')[0]
    assert target['projected_pred_index']==0 and target['native_pred_index']==1
    assert target['pixel_bbox']==[10,20,30,40] and target['coord_bins']==[11,22,33,44]


def test_occupied_evaluation_output_is_preserved(tmp_path):
    path=tmp_path/'existing';path.write_text('keep')
    with pytest.raises(ValueError,match='occupied'):evaluator.prepare(tmp_path)
    assert path.read_text()=='keep'
