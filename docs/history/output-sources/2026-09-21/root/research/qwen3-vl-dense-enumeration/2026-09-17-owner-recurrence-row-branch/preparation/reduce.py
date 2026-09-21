"""Saved-token consumer. Known-bank matching is distinct from physical review."""
import argparse, importlib.util, json
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
SCORE=ROOT.parent/'2026-09-16-endpoint-loop-natural-readout-norm/reduce.py'
spec=importlib.util.spec_from_file_location('accepted_score',SCORE)
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)

def consume(panel, runtime_root):
 tok=AutoTokenizer.from_pretrained(panel['config']['model']['base_model'],trust_remote_code=True)
 results={};receipts=[]
 for binding in panel['sources']:
  assert score.bind(binding['path'])==binding, binding['path']
 for case in panel['cases']:
  iid=str(case['image_id']);bank=panel['banks'][iid];input_case=case['group']['cases'][case['target_position']]
  excluded=set(case['supplied_known_owner_union']);arms={}
  saved=json.loads(Path(case['saved_raw']['path']).read_text())['rows'][case['target_position']]
  for arm in case['arms']:
   folder=runtime_root/iid/arm
   if not (folder/'raw.json').exists():continue
   raw=json.loads((folder/'raw.json').read_text());rec=json.loads((folder/'receipt.json').read_text())
   assert rec['status']=='candidate_complete',folder
   assert rec['panel']==score.bind(ROOT/'panel.json')
   assert rec['raw']==score.bind(folder/'raw.json')
   row=raw['rows'][case['target_position']];ids=row['token_ids'];start=case['start_offset'];end=case['end_offset']
   assert ids[start:end]==case['arms'][arm]['token_ids']
   assert ids[:start]==saved['token_ids'][:start]
   if arm=='native':assert ids==saved['token_ids'] and row['stop']==saved['stop']
   def view(tokens,stop):
    return score.score(dict(token_ids=tokens,text=tok.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False),stop=stop),input_case,bank)
   full=view(ids,row['stop']);free=view(ids[end:],row['stop']);history=view(ids[:start],'supplied_history');supplied=view(ids[start:end],'supplied_row')
   owners=set(free['matches']['covered_owner_ids']);h=set(history['matches']['covered_owner_ids'])
   arms[arm]=dict(full=full,free=free,common_history=history,supplied=supplied,eligible_free_owner_ids=sorted(owners-excluded),new_free_relative_history=sorted(owners-h-excluded),supplied_owner_exclusion=sorted(excluded),raw=score.bind(folder/'raw.json'),receipt=score.bind(folder/'receipt.json'))
   receipts.append(rec)
  native=set(arms['native']['eligible_free_owner_ids'])
  for arm,v in arms.items():
   current=set(v['eligible_free_owner_ids']);v['known_free_vs_native']=dict(gained=sorted(current-native),lost=sorted(native-current),retained=sorted(current&native))
   assert not(current&excluded)
  results[iid]=arms
 return dict(panel=score.bind(ROOT/'panel.json'),status='candidate',consumer=score.bind(__file__),accepted_parser_scorer=score.bind(SCORE),images=results,physical_review='separate sidecar; known losses are not physical disappearance',cost=dict(batch_executions=len(receipts),model_forwards=sum(x.get('counts',{}).get('model_forwards',x.get('model_forwards',0)) for x in receipts)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 panel=json.loads((ROOT/'panel.json').read_text());result=consume(panel,ROOT/'runtime')
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({i:{a:v['known_free_vs_native'] for a,v in arms.items()} for i,arms in result['images'].items()},indent=2))
