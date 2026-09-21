"""Independent CPU formula, token-choice and source replay checks; no forwards."""
import argparse,json,hashlib
from pathlib import Path
import torch

POLICIES=['original','full','shared','centered']
def verify(root):
    torch.set_num_threads(2); reports=[]
    plan=json.loads((root/'execution-plan.json').read_text())
    histories={h['id']:h for h in plan['histories']};cells={c['id']:c for c in plan['cells']}
    for shard in sorted((root/'runtime').iterdir()):
        wp=shard/'effective-readout.pt'
        if not shard.is_dir() or not wp.exists():continue
        w=torch.load(wp,map_location='cpu',weights_only=False);W=w['output_rows'].double();norm=W.norm(dim=1);alpha=norm.median()/norm;mu=W.mean(0)
        assert torch.allclose(alpha,w['factors'],atol=1e-12,rtol=0)
        assert torch.allclose(mu,w['mu'],atol=1e-12,rtol=0)
        assert w['coordinate_ids'].tolist()==list(range(151670,152670))
        for rp in sorted(shard.glob('*/release.json')):
            r=json.loads(rp.read_text());t=torch.load(rp.parent/'trajectory.pt',map_location='cpu',weights_only=False)
            z=t['raw_coordinate_logits'].double();h=t['head_inputs'].double();b=h@mu;full=z*alpha;shared=z+(alpha-1)*b[:,None];centered=z+(alpha-1)*(z-b[:,None]);expected=torch.stack([z,full,shared,centered],1).float()
            error=float((expected-t['operator_coordinate_logits']).abs().max());assert error<=0.0002,(rp,error)
            decomposition=float(((shared-z)+(centered-z)-(full-z)).abs().max());assert decomposition<1e-12
            tokens=r['target']['token_ids'];steps=r['trace']['steps'];assert len(steps)==len(tokens)==len(h)
            assert t['target_tokens'].tolist()==tokens
            for i,s in enumerate(steps):
                assert s['offset']==i and s['chosen_token']==s['emitted_token']==tokens[i]
                assert s['operators'][r['policy']]['top2'][0]['token_id']==tokens[i]
                for j,p in enumerate(POLICIES):
                    winner=s['operators'][p]['top2'][0]['token_id']
                    if 151670<=winner<152670:
                        assert int(t['operator_coordinate_logits'][i,j].argmax())==winner-151670
            history=histories[cells[r['job_id']]['history_id']];native=history['saved_overlap_tokens'];n=min(len(tokens),len(native));source_equal=tokens[:n]==native[:n];control=r['policy']==history['history_policy'];assert r['history']['history_prefix_tokens']==history['prefix_tokens']
            if control:assert source_equal, r['job_id']
            q=r['qualification']
            if q:
                assert q['identity_operator_bitwise'] and all(q['noncoordinate_bitwise_unchanged'].values())
                assert q['full_formula_max_abs_error']==0 and q['shared_plus_centered_increment_max_abs_error']<1e-12
                if control:assert source_equal
            reports.append(dict(cell=r['job_id'],steps=len(tokens),formula_max_abs=error,decomposition_max_abs=decomposition,original_source_overlap_equal=source_equal if control else None,source_overlap_tokens=n if control else None,trajectory_sha256=hashlib.sha256((rp.parent/'trajectory.pt').read_bytes()).hexdigest()))
    return dict(status='candidate_CPU_verified',cells=len(reports),steps=sum(x['steps'] for x in reports),reports=reports)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=verify(a.root);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(cells=d['cells'],steps=d['steps'])))
