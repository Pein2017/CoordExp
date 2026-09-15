"""Decision-bearing CPU checks on the exact admitted row/endpoint surface."""
from copy import deepcopy

from probes.parallel_owner_research import data_flywheel as lane


def fixture():
    sidecar=lane.read(lane.ROOT/'candidate-sidecar.json')
    source=lane.read(lane.ENDPOINT_SOURCE)
    frozen=next(r for r in source['eval_records'] if r['example_id']==lane.EXAMPLE)
    consumer=lane.read(lane.ROOT.parent/'instance-state/panel-v1/consumer.json')
    parsed=next(r['parsed'] for r in consumer['records'] if (r['case_id'],r['arm'],r['carrier'])==('417044','owner_a','coordinates'))
    return sidecar,source,frozen,parsed


def test_source_34_rows_have_25_reviewed_not_34_owners():
    sidecar,_,_,parsed=fixture()
    result=lane.reviewed_score(parsed,sidecar)
    assert result['tp']==25
    assert result['prediction_count']==34
    assert {k:v['denominator'] for k,v in result['partitions'].items()}=={'annotated':11,'supplied_source_unlabeled':1,'free_discovery_unlabeled':13}


def test_one_to_one_duplicate_cannot_create_a_second_owner():
    sidecar,_,_,parsed=fixture()
    copy=deepcopy(parsed)
    copy['pred']=[parsed['pred'][1],parsed['pred'][3]]
    result=lane.reviewed_score(copy,sidecar)
    assert result['tp']==1 and result['owners']==['P1']


def test_clean_trajectory_is_literal_compaction_not_masked_original():
    sidecar,_,_,_=fixture()
    rows=lane.read(lane.ROOT/'proposed-literal-positive-records.json')
    original={r['row_id']:r for r in sidecar['rows']}
    accumulated=[];unchanged=[]
    for row in rows:
        assert row['prefix_token_ids']==accumulated
        assert row['target_token_ids']==original[row['source_row_id']]['row_token_ids']
        assert lane.EOS not in row['target_token_ids']
        unchanged.append(row['prefix_token_ids']==original[row['source_row_id']]['source_prefix_token_ids'])
        accumulated.extend(row['target_token_ids'])
    assert len(rows)==25 and len(accumulated)==249
    assert unchanged==[True]*3+[False]*22


def test_literal_ids_decode_to_exact_admitted_rows():
    from transformers import AutoTokenizer
    sidecar,source,_,_=fixture()
    tok=AutoTokenizer.from_pretrained(source['model']['base_model_path'],local_files_only=True)
    for r in sidecar['rows']:
        assert tok.decode(r['row_token_ids'],skip_special_tokens=False)==r['row_text']


def test_fixed_endpoint_population_is_native_384_plus_one_baseline():
    _,source,_,_=fixture()
    jobs=[lane.endpoint_jobs(source,s) for s in range(8)]
    assert [len(x) for x in jobs]==[49,48,48,48,48,48,48,48]
    all_jobs=[r for shard in jobs for r in shard]
    assert sum(r['arm']=='Stable50' for r in all_jobs)==1
    trained=[r['example_id'] for r in all_jobs if r['arm']==lane.ARM]
    assert len(set(trained))==len(trained)==384
    assert jobs[0][0]=={'arm':'Stable50','example_id':lane.EXAMPLE}


def test_reviewed_coverage_counter_has_teeth():
    sidecar,_,_,parsed=fixture()
    copy=deepcopy(parsed)
    copy['pred']=[parsed['pred'][int(name[1:])] for name in lane.KEEPS]
    assert lane.reviewed_score(copy,sidecar)['tp']==25
    copy['pred'][0]['bbox']=[1100,800,1150,860]
    changed=lane.reviewed_score(copy,sidecar)
    assert changed['tp']==24 and changed['unrecovered']==['P0']
