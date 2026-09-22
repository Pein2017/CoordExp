"""Post-acquisition strata; does not alter frozen admission or denominators."""
import json
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer
from probes.source_rweak_row_cross.owner_row_robustness import native_rows
from probes.dora_owner_learning.route_access import publish
from probes.dora_owner_learning.candidate_opportunity import file_hash
from probes.dora_owner_learning.entrance_ce_eval import aggregate_scores

OUT=Path(__file__).parent


def summarize():
    m=json.loads((OUT/'inputs.json').read_text()); r=json.loads((OUT/'reduction.json').read_text())
    tok=Tokenizer.from_file(m['source_model']['base_model_path']+'/tokenizer.json')
    shard={eid:i for i,group in enumerate(m['shards']) for eid in group}; candidates={c['image_id']:c for c in m['candidates']}
    strata=Counter(); downstream=[]
    for x in r['results']:
        if x['status']!='two_correction_closure': continue
        a,b=x['first_insertion_at_EOS'],x['second_insertion_at_EOS']
        strata['both_at_EOS' if a and b else 'second_at_EOS' if b else 'both_before_existing_rows']+=1
        raw=json.loads((OUT/f"shard-{shard[x['example_id']]}/{x['image_id']}-second.json").read_text())
        rows=native_rows(raw['action_ids'],tok); free_start=len(raw['prefix_ids'])+len(raw['forced_ids'])
        old=set(x['stages'][0]['score']['50']['owners']); matches=x['stages'][-1]['score']['50']['matches']
        free_old=[v['owner'] for v in matches if v['owner'] in old and rows[v['pred_index']]['start']>=free_start]
        if free_old: downstream.append(dict(image_id=x['image_id'],selection_rank=candidates[x['image_id']]['selection_rank'],free_old_owners_after_second=free_old))
    before=[x['stages'][0]['score'] for x in r['results']]; after=[x['stages'][-1]['score'] for x in r['results']]
    pairs=[(set(a['50']['owners']),set(b['50']['owners'])) for a,b in zip(before,after)]
    return dict(schema='recursive_owner_admission.interpretation.v1',reduction_sha256=file_hash(OUT/'reduction.json'),
        two_correction_strata=dict(strata),old_owner_free_continuation_after_second_count=len(downstream),
        old_owner_free_continuation_after_second=sorted(downstream,key=lambda x:x['selection_rank']),
        natural=aggregate_scores(before),final_assisted=aggregate_scores(after),
        gained=sum(len(b-a) for a,b in pairs),lost=sum(len(a-b) for a,b in pairs),retained=sum(len(a&b) for a,b in pairs),
        note='Post-acquisition diagnostic strata, not an altered primary gate. Forced targets are not autonomous gains.')


if __name__=='__main__':
    result=summarize(); path=OUT/'interpretation.json'
    if path.exists(): assert result==json.loads(path.read_text())
    else: publish(path,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['natural','final_assisted','old_owner_free_continuation_after_second']}))
