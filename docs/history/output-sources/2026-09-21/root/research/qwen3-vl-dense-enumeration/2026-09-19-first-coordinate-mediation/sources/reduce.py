"""Separate wholly free successor rows from the partially controlled first row."""
import argparse,json,hashlib
from pathlib import Path
from probes.training_set_completion.numerical_feedback.metrics import release_metrics
from probes.training_set_completion.numerical_feedback.select import rows

def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def reduce(root):
 plan=json.loads((root/'execution-plan.json').read_text());bs={b['id']:b for b in json.loads(Path(plan['selection']).read_text())['boundaries']};cells=[];bindings=[bind(root/'execution-plan.json')]
 for c in plan['cells']:
  ps=list((root/'runtime').glob('*/'+c['id']+'/release.json'));assert len(ps)==1,(c['id'],ps)
  p=ps[0];r=json.loads(p.read_text());tokens=r['target']['token_ids'];assert tokens[:5]==plan['expected_prelude'] and tokens[5]==c['forced_token'];rr=rows(tokens);whole=release_metrics(tokens,bs[c['boundary_id']]['source_row']);first=rr[0] if rr else None;free=tokens[first['end']:] if first else [];primary=release_metrics(free,bs[c['boundary_id']]['source_row']);assert len(tokens)<=512 and whole['complete_rows']<=32
  equal=None
  if c['concordant']:
   cp=Path(c['control_source']['path']);assert bind(cp)['sha256']==c['control_source']['sha256'];equal=tokens==json.loads(cp.read_text())['target']['token_ids'];assert equal,c['id']
  cells.append(dict(id=c['id'],boundary_id=c['boundary_id'],model=c['model'],policy=c['policy'],forced_choice=c['forced_choice'],concordant=c['concordant'],control_equal=equal,tokens=tokens,stop=r['target']['stop']['reason'],first_partially_controlled_row=first,primary_start_offset=first['end'] if first else None,primary_status='free_rows_after_first_complete_row' if first else 'HOLD_no_completed_controlled_row',primary=primary,whole_including_controlled_row=whole,release=bind(p)));bindings.append(bind(p))
 contrasts=[]
 for model in ['tied','untied']:
  for policy in plan['policies']:
   cc={c['forced_choice']:c for c in cells if c['model']==model and c['policy']==policy};a,b=cc[0],cc[1];contrasts.append(dict(model=model,policy=policy,choice1_minus_choice0={k:b['primary'][k]-a['primary'][k] for k in ['longest_exact_run','longest_near_run','complete_rows','invalid_rows','malformed_openers']},choice0_stop=a['stop'],choice1_stop=b['stop']))
 return dict(status='candidate',cells=cells,contrasts=contrasts,bindings=bindings,limits='Primary excludes entirefirstcompletedpartiallycontrolledrow. Numerical recurrence is not physical-owner identity/recovery. Outcome-selected two-model donutfollowup; all8signs retained, no bin0 coefficient uniqueness inference.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=reduce(a.root);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(cells=len(d['cells']),contrasts=len(d['contrasts']))))
