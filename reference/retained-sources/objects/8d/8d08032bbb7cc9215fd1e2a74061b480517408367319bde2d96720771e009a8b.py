"""Saved-target recurrence comparison; no model calls or physical-owner claims."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from probes.training_set_completion.numerical_feedback.metrics import release_metrics


def binding(p):
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def reduce(root):
    plan=json.loads((root/'execution-plan.json').read_text())
    boundaries={b['id']:b for b in json.loads(Path(plan['selection']).read_text())['boundaries']}
    cells=[]; traces={}; bindings=[binding(root/'execution-plan.json'),binding(Path(plan['selection']))]
    for c in plan['cells']:
        matches=list((root/'runtime').glob('*/'+c['id']+'/release.json')); assert len(matches)==1, (c['id'],matches)
        p=matches[0]
        d=json.loads(p.read_text()); traces[c['id']]=d['trace']['steps']; tokens=d['target']['token_ids']; b=boundaries[c['boundary_id']]
        assert len(tokens)<=512
        m=release_metrics(tokens,b['source_row'])
        assert m['complete_rows']<=32
        assert not m['eos'] or tokens[-1]==151645
        reason='eos' if m['eos'] else 'rows' if m['complete_rows']==32 else 'cap'
        assert reason==d['target']['stop']['reason']
        assert reason!='cap' or len(tokens)==512
        cells.append(dict(id=c['id'],boundary_id=c['boundary_id'],model=b['model'],image_id=b['image_id'],kind=b['kind'],episode_stratum=b['episode_stratum'],policy=c['policy'],tokens=tokens,token_count=len(tokens),stop=reason,metrics=m))
        bindings.append(binding(p))
    pairs=[]
    for bid in boundaries:
        arms={c['policy']:c for c in cells if c['boundary_id']==bid}
        for a,b in itertools.combinations(plan['policies'],2):
            x,y=arms[a],arms[b]; tx,ty=x['tokens'],y['tokens']; fork=next((i for i,(u,v) in enumerate(zip(tx,ty)) if u!=v),None)
            if fork is None and len(tx)!=len(ty): fork=min(len(tx),len(ty))
            pairs.append(dict(boundary_id=bid,left=a,right=b,exact_tokens_equal=tx==ty,first_divergence=fork,left_token=tx[fork] if fork is not None and fork<len(tx) else None,right_token=ty[fork] if fork is not None and fork<len(ty) else None,delta_right_minus_left={k:y['metrics'][k]-x['metrics'][k] for k in ['complete_rows','invalid_rows','malformed_openers','longest_exact_run','longest_near_run']}))
    forks=[]
    for bid in boundaries:
        original=next(c for c in cells if c['boundary_id']==bid and c['policy']=='original')
        for policy in ['full','shared','centered']:
            pair=next(x for x in pairs if x['boundary_id']==bid and x['left']=='original' and x['right']==policy)
            i=pair['first_divergence']
            if i is not None and i<len(traces[original['id']]):
                step=traces[original['id']][i]
                forks.append(dict(boundary_id=bid,policy=policy,offset=i,absolute_action_offset=boundaries[bid]['source_row']['end']+i,raw_winner=step['raw_winner_token'],actual_other_token=pair['right_token'],same_state_shadows=step['operators']))
    summary=[]
    for model,kind,policy in itertools.product(['tied','untied'],['failure','healthy'],plan['policies']):
        cc=[c for c in cells if (c['model'],c['kind'],c['policy'])==(model,kind,policy)]
        summary.append(dict(model=model,kind=kind,policy=policy,n=len(cc),eos=sum(c['stop']=='eos' for c in cc),row_stop=sum(c['stop']=='rows' for c in cc),token_cap=sum(c['stop']=='cap' for c in cc),native_return=sum(bool(c['metrics']['native_near_return_rows']) for c in cc),alternate_repeat=sum(bool(c['metrics']['alternate_repeat_starts']) for c in cc),invalid_rows=sum(c['metrics']['invalid_rows'] for c in cc),malformed_openers=sum(c['metrics']['malformed_openers'] for c in cc),longest_near_runs=[c['metrics']['longest_near_run'] for c in cc]))
    return dict(status='candidate_saved_output_reduction',cells=cells,pairs=pairs,first_forks=forks,summary=summary,bindings=bindings,limits='Target-only numerical recurrence; correlated boundaries and seven images, not population rates. Proxy healthy may later loop. EOS/shorter/alternate repetition is not physical recovery.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    d=reduce(a.root);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(cells=len(d['cells']),pairs=len(d['pairs']),output=str(a.out))))
