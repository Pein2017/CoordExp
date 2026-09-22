import copy
import json

import pytest

from probes.source_rweak_row_cross.owner_row_robustness import (
    CAP, branch, digest, native_rows, publish, validate_admission, variant,
    consume, native_record,
    reduce_records,
)


class Tokens:
    def token_to_id(self, token):
        fixed={'<|object_ref_start|>':1,'<|object_ref_end|>':2,
               '<|box_start|>':3,'<|box_end|>':4}
        return fixed[token] if token in fixed else 100+int(token[8:-2])

    def decode(self, ids, skip_special_tokens=False):
        fixed={1:'<|object_ref_start|>',2:'<|object_ref_end|>',3:'<|box_start|>',
               4:'<|box_end|>',42:'person',151645:'<|im_end|>'}
        return ''.join(fixed[i] if i in fixed else f'<|coord_{i-100}|>' for i in ids)

    convert_tokens_to_ids=token_to_id


def fixture():
    tok=Tokens()
    a=native_rows([1,42,2,3,110,120,130,140,4],tok)[0]
    b=native_rows([1,42,2,3,115,125,135,145,4],tok)[0]
    return tok,dict(case_id='c',prefix_ids=[8]*18,sample_prefix_ids=[8]*9,A=a,B=b,Aprime=variant(a,tok))


def test_native_one_bin_and_frozen_coordinates():
    tok,c=fixture();a=c['A'];v=c['Aprime']
    assert v['coords']==[10,20,31,40]
    assert [i for i,(x,y) in enumerate(zip(a['ids'],v['ids'])) if x!=y]==[6]
    assert native_rows(v['ids'],tok)[0]['coords']==v['coords']
    a['coords'][2]=999;a['ids'][6]=1099
    assert variant(a,tok)['coords']==[10,20,998,40]
    a['coords'][0]=998
    with pytest.raises(ValueError,match='no legal'):variant(a,tok)


def test_exact_prefix_and_whole_action_budget():
    _,c=fixture()
    for arm in ('B','A','Aprime','partial_A','sample_A'):
        job=branch(c,arm)
        assert len(job['prefix'])+len(job['forced'])+job['remaining']==CAP
    assert len(branch(c,'partial_A')['forced'])==5
    assert branch(c,'A')['remaining']==branch(c,'Aprime')['remaining']==branch(c,'B')['remaining']
    assert branch(c,'sample_A')['prefix']==c['sample_prefix_ids']
    c['prefix_ids']=[8]*CAP
    with pytest.raises(ValueError,match='nonpositive'):branch(c,'B')


def test_admission_missing_duplicate_and_manifest_identity():
    _,c=fixture();manifest={'selected':[c]}
    admission={'manifest_sha256':digest(manifest),'admitted_case_ids':['c']}
    assert validate_admission(manifest,admission)==[c]
    for ids,error in [(['c','c'],'duplicate'),(['absent'],'missing')]:
        with pytest.raises(ValueError,match=error):validate_admission(manifest,dict(admission,admitted_case_ids=ids))
    with pytest.raises(ValueError,match='manifest mismatch'):
        validate_admission(manifest,dict(admission,manifest_sha256='wrong'))
    duplicated={'selected':[c,c]}
    with pytest.raises(ValueError,match='duplicate'):
        validate_admission(duplicated,dict(admission,manifest_sha256=digest(duplicated)))


def test_occupied_artifact_is_not_overwritten(tmp_path):
    p=tmp_path/'out.json';publish(p,{'original':True})
    with pytest.raises(FileExistsError):publish(p,{'replacement':True})
    assert json.loads(p.read_text())=={'original':True}


def test_malformed_native_rows_fail_closed():
    tok,c=fixture()
    with pytest.raises(ValueError):native_rows([7]+c['A']['ids'],tok)
    with pytest.raises(ValueError):native_rows(c['A']['ids']+[151645,7],tok)
    bad=copy.deepcopy(c['A']['ids']);bad[6]=105
    with pytest.raises(ValueError,match='invalid box'):native_rows(bad,tok)


def test_disk_consumer_has_teeth_for_missing_duplicate_and_token_corruption(tmp_path):
    tok,c=fixture();c.update(example_id='image',owner='a',B_owner='b',prefix_ids=[])
    c['golden']=dict(example_id='image',row_id='image',row_index=0,image_width=999,image_height=999,
        image_path='fixture.png',gt=[dict(object_id='a',description='person',bbox=[10,20,30,40]),
                                    dict(object_id='b',description='person',bbox=[15,25,35,45])])
    manifest={'selected':[c]};ids=c['B']['ids']+[151645];text=tok.decode(ids)
    row=dict(case_id='c',arm='B',manifest_sha256=digest(manifest),suffix_ids=[151645],
             action_ids=ids,text=text,stop_reason='im_end',
             parsed=native_record(text,{'row_id':'image'},c['golden'],'im_end'))
    path=tmp_path/'rows.jsonl';path.write_text(json.dumps(row)+'\n')
    result=consume(path,manifest,tok,[('c','B')]);assert result[0]['score']['50']['owners']==['b']
    with pytest.raises(ValueError,match='missing output'):consume(path,manifest,tok,[('c','B'),('c','A')])
    path.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError,match='duplicate output'):consume(path,manifest,tok,[('c','B')])
    row['action_ids']=row['action_ids'][:-1];path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='budget/identity'):consume(path,manifest,tok,[('c','B')])


def test_sample_history_control_rejects_incompatible_or_empty_workload():
    manifest={'selected':[{'case_id':'c','boundary':0}]}
    records=[]
    for arm in ('B','A','Aprime','partial_A','sample_A'):
        incidence={'a':[],'b':[]}
        records.append(dict(case_id='c',arm=arm,parsed={'pred':[]},
            score={t:dict(owners=[],matches=[]) for t in ('50','60','80')},
            conditional=dict(remaining_obligations=['a','b'] if arm=='sample_A' else ['a'],
                free_row_count=0,free_suffix_incidence=incidence,direct_action_incidence=incidence,
                any_category_incidence=incidence,direct_incidence_by_threshold={t:incidence for t in ('50','60','80')})))
    result=reduce_records(records,manifest)[0]
    assert not result['sampled_history_control_discriminating']
    assert result['sampled_only_remaining_obligations']==['b']
    records[-1]['conditional']['remaining_obligations']=['a']
    assert reduce_records(records,manifest)[0]['sampled_history_control_discriminating']
    records[-1]['conditional']['remaining_obligations']=[]
    assert not reduce_records(records,manifest)[0]['sampled_history_control_discriminating']
