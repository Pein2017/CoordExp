import json,collections
from pathlib import Path
from probes.training_set_completion.recurrence_mass.reduce import reduce_state
root=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass')
checks=[]
for line in (root/'draws.jsonl').open():
 s=json.loads(line); assert len(s['draws'])==256
 assert [d['draw_index'] for d in s['draws']]==list(range(256))
 count=collections.Counter();exact_invalid=0
 for d in s['draws']:
  ts=d['token_ids']; box=[t-151670 for t in ts[:4]]
  if len(ts)==5 and all(0<=x<1000 for x in box) and ts[-1]==151649:
   legal=box[0]<box[2] and box[1]<box[3]
   match=any(max(abs(x-y) for x,y in zip(box,b))<=8 for b in s['repeat_union_bins'])
   invmatch=any(max(abs(x-y) for x,y in zip(box,b))<=8 for b in s['literal_repeat_union_bins'])
   if legal:count['legal_repeat' if match else 'legal_nonrepeat']+=1
   else:
    count['invalid_geometry_near_repeat' if invmatch else 'invalid_extent']+=1
    exact_invalid+=box in s['literal_repeat_union_bins']
  elif 151645 in ts and ts.index(151645)<4:count['early_eos']+=1
  elif len(ts)<5:count['early_eos' if 151645 in ts else 'short_or_length']+=1
  else:count['grammar_escape']+=1
 r=reduce_state(s)
 for k in set(count)|set(r['outcome_counts']):assert count[k]==r['outcome_counts'].get(k,0),(s['state_id'],k)
 assert exact_invalid==r['exact_invalid_recurrence_count']
 checks.append({'state':s['state_id'],'draws':256,'successes':count['legal_repeat'],'counts':dict(count),'exact_invalid':exact_invalid})
assert len(checks)==45, len(checks)
(Path(__file__).parent/'mass-events-check.json').write_text(json.dumps({'status':'pass','states':45,'draws':11520,'checks':checks},indent=2)+'\n')
print('PASS45states/11520 independent token-event checks')
