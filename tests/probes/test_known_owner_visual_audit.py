"""Adapter preserves parser gaps and rejects stale arm/image/row projections."""
import copy
import tempfile
from pathlib import Path

from probes import known_owner_visual_audit as audit
from probes.hidden_human_recovery import canonical


def test_saved_projection_and_mismatch_falsifiers():
    image=dict(image_id=7,image_path='/fixture/7.jpg',image_sha256='fixture',width=1248,height=832,
               objects=[dict(coco_ann_id=-8107737316336679,desc='person',bbox_2d=[100,100,300,300])])
    text=''.join('<|object_ref_start|>person<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{b}|>' for b in box)+'<|box_end|>'
                 for box in ([100,100,100,300],[100,100,300,300],[100,100,300,300]))
    raw={k:v for k,v in image.items() if k!='objects'}
    raw.update(request_id='7:greedy:0',arm='greedy',crop=[0,0,1248,832],text=text)
    pool,_=audit.candidates([raw])
    value=audit.projection(image,raw,pool,0)
    assert len(value['pred'])==2 and [p['raw_parser_order'] for p in value['pred']]==[1,2]
    assert value['pred'][0]['bbox']==[125,83,374,250]
    assert value['gt'][0]['owner_id']=='-8107737316336679'
    def reject(fn):
        try: fn()
        except ValueError: return
        raise AssertionError('mismatch accepted')
    stale=copy.deepcopy(raw);stale['image_id']=8
    reject(lambda:audit.projection(image,stale,pool,0))
    stale_pool=copy.deepcopy(pool);stale_pool[0]['prediction_id']='7:greedy:0:p0'
    reject(lambda:audit.projection(image,raw,stale_pool,0))
    reject(lambda:audit.projection(image,raw,pool[:1],0))
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        for name in ('gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl'):
            (folder/name).write_text(canonical(value)+'\n')
        audit.check_projection(folder,[image],[{'raw':raw}],[pool])
        stale=copy.deepcopy(value);stale['pred'][0]['bbox'][0]+=1
        (folder/'gt_vs_pred_scored.jsonl').write_text(canonical(stale)+'\n')
        reject(lambda:audit.check_projection(folder,[image],[{'raw':raw}],[pool]))
