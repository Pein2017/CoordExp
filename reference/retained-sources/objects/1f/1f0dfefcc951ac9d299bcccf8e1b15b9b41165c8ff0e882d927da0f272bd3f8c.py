import copy
import pytest
from tokenizers import Tokenizer

from probes.dora_owner_learning.candidate_opportunity import score
from probes.dora_owner_learning.geometric_dedup_eval import consume,overlap_counts,reduce_records,CAP
from probes.source_rweak_row_cross.run import native_record


def _score(owner, *, cap=0, repeats=0):
    value={t:dict(tp=1,fp=1,fn=1,f1=.5,owners=[owner],matches=[]) for t in ('50','60','80')}
    value.update(prediction_count=2,parsed_prediction_count=2,invalid_predictions=0,parser_drops=0,
                 complete_token_length=CAP if cap else 19,strict_repeats=repeats,cap=cap)
    return value


def test_reduction_uses_current_stable_owner_sets_and_retains_cap_failure():
    old=dict(example_id='a',image_id=1,split='online8',stable_score=_score('current-owner',repeats=1),
        stable_overlap_counts={'80':1,'90':1,'95':1},golden={'legacy_source_owner':'not-the-baseline'})
    new=dict(example_id='a',image_id=1,split='online8',score=_score('new-owner',cap=1),
        overlap_counts={'80':0,'90':0,'95':0})
    result=reduce_records([new],dict(eval_records=[old],protected_targets={'1':'current-owner'}))
    assert result['panels']['union384']['owner_changes']['50']==dict(gained=1,lost=1,retained=0)
    assert result['per_image'][0]['owner_changes']['50']['lost']==['current-owner']
    assert result['new_cap_images']==[1]
    assert not result['protected_targets']['1']['50']
    assert result['panels']['union384']['candidate']['cap']==1
    assert result['panels']['union384']['images']==1


def test_overlap_diagnostics_are_strict_pixel_geometry_and_class_blind():
    parsed={'pred':[{'bbox':[0,0,100,100],'description':'cat'},
                    {'bbox':[0,0,100,95],'description':'dog'},
                    {'bbox':[0,0,100,96],'description':'bowl'},
                    {'bbox':[1,1,1,5],'description':'person'}]}
    assert overlap_counts(parsed)=={'80':2,'90':2,'95':1}


def test_consumer_rejects_forced_output_and_preserves_native_eos():
    tok=Tokenizer.from_file('/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent/tokenizer.json')
    text='<|object_ref_start|>cat<|object_ref_end|><|box_start|><|coord_100|><|coord_100|><|coord_400|><|coord_400|><|box_end|><|im_end|>'
    ids=tok.encode(text,add_special_tokens=False).ids
    golden=dict(example_id='a',row_id='a',row_index=0,image_path='/tmp/synthetic.jpg',image_width=640,image_height=480,
        gt=[dict(object_id='gt',description='cat',bbox=[100,100,400,400])])
    frozen=dict(example_id='a',case={'row_id':'a'},golden=golden)
    raw=dict(example_id='a',action_ids=ids,prefix_ids=[],forced_ids=[],remaining_budget=CAP,text=text,stop_reason='im_end',
        parsed=native_record(text,frozen['case'],golden,'im_end'))
    result=consume(raw,frozen,tok)
    assert result['score']['50']['owners']==['gt']
    bad=copy.deepcopy(raw);bad['forced_ids']=ids[:1]
    with pytest.raises(ValueError,match='unforced original-input'):
        consume(bad,frozen,tok)
    bad=copy.deepcopy(raw);bad['action_ids']=ids[:-1]
    with pytest.raises(ValueError,match='terminal corruption'):
        consume(bad,frozen,tok)
