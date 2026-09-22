"""CPU-only fixed crossover outcomes, crossing-cut recurrence and paired contrasts."""
import argparse,json,hashlib,itertools
from pathlib import Path
from probes.training_set_completion.numerical_feedback.metrics import release_metrics
from probes.training_set_completion.numerical_feedback.select import rows,same

def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def reduce(root):
 plan=json.loads((root/'execution-plan.json').read_text());histories={h['id']:h for h in plan['histories']};out=[];traces={};bindings=[bind(root/'execution-plan.json')]
 for c in plan['cells']:
  ps=list((root/'runtime').glob('*/'+c['id']+'/release.json'));assert len(ps)==1,(c['id'],ps)
  p=ps[0];d=json.loads(p.read_text());traces[c['id']]=d['trace']['steps'];h=histories[c['history_id']];t=d['target']['token_ids'];m=release_metrics(t,h['boundary']['source_row']);rr=rows(t);assert len(t)<=512 and len(rr)<=32
  stop='eos' if m['eos'] else 'rows' if len(rr)==32 else 'cap';assert stop==d['target']['stop']['reason'];assert stop!='cap' or len(t)==512
  tail=h['supplied_rows'][-2:];combined=tail+rr;events=[]
  for i in range(2,len(combined)):
   triple=combined[i-2:i+1]
   if all(same(a,b,8) for a,b in itertools.combinations(triple,2)):
    events.append(dict(completed_free_row=i-len(tail)+1,start_free_row=i-2-len(tail)+1,crosses_cut=i-2<len(tail),values=[x['values'] for x in triple]))
  overlap=h['saved_overlap_tokens'];n=min(len(t),len(overlap));control=c['policy']==h['history_policy'];equal=t[:n]==overlap[:n] if control else None
  if control:assert equal,(c['id'],'unchanged_policy_mismatch')
  out.append(dict(id=c['id'],boundary_id=h['boundary']['id'],model=c['model'],image_id=h['boundary']['image_id'],cut=h['cut'],history_policy=h['history_policy'],future_policy=c['policy'],tokens=t,stop=stop,metrics=m,first_recurrence_with_supplied_tail=events[0] if events else None,crossing_cut_events=events,unchanged_control=control,saved_overlap_equal=equal,overlap_tokens=n if control else None,release=bind(p)))
  bindings.append(bind(p))
 contrasts=[]
 for h in plan['histories']:
  cells={c['future_policy']:c for c in out if c['boundary_id']==h['boundary']['id'] and c['cut']==h['cut'] and c['history_policy']==h['history_policy']};a,b=cells['original'],cells['full'];fork=next((i for i,(x,y) in enumerate(zip(a['tokens'],b['tokens'])) if x!=y),None)
  if fork is None and len(a['tokens'])!=len(b['tokens']):fork=min(len(a['tokens']),len(b['tokens']))
  shadow=traces[a['id']][fork]['operators'] if fork is not None and fork<len(traces[a['id']]) else None
  if shadow:assert shadow['full']['top2'][0]['token_id']==b['tokens'][fork]
  contrasts.append(dict(same_state_first_fork=shadow,history_id=h['id'],boundary_id=h['boundary']['id'],cut=h['cut'],history_policy=h['history_policy'],contrast='withdrawal_vs_continued_full' if h['history_policy']=='full' else 'late_switch_vs_original',first_divergence=fork,exact_tokens_equal=a['tokens']==b['tokens'],full_minus_original={k:b['metrics'][k]-a['metrics'][k] for k in ['longest_exact_run','longest_near_run','invalid_rows','malformed_openers','complete_rows']},original_stop=a['stop'],full_stop=b['stop']))
 history_comparisons=[]
 for bid in dict.fromkeys(h['boundary']['id'] for h in plan['histories']):
  for cut in [4,8]:
   hh={h['history_policy']:h for h in plan['histories'] if h['boundary']['id']==bid and h['cut']==cut}
   history_comparisons.append(dict(boundary_id=bid,cut=cut,identical_prefix=hh['original']['prefix_tokens']==hh['full']['prefix_tokens'],original_prefix_length=len(hh['original']['prefix_tokens']),full_prefix_length=len(hh['full']['prefix_tokens'])))
 return dict(history_comparisons=history_comparisons,status='candidate',cells=out,contrasts=contrasts,unsupported=plan['unsupported'],bindings=bindings,limits='Suffix metrics exclude supplied rows; crossing-cut events explicitly include up to2 supplied rows. Numerical patterns are not physical-owner recurrence/recovery. Cuts/models not independent samples.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=reduce(a.root);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(cells=len(d['cells']),contrasts=len(d['contrasts']))))
