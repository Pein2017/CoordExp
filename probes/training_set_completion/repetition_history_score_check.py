"""Independent CPU reconstruction of the saved history-exposure scores."""
import argparse
import json
from pathlib import Path
import torch
from probes.training_set_completion.readout_norm_fresh import _binding, _write


def check(folder):
    receipt=json.loads((folder/'receipt.json').read_text())
    assert receipt['status']=='candidate_complete'
    assert receipt['scores']==_binding(folder/'scores.json')
    scores=json.loads((folder/'scores.json').read_text())
    maximum_error=0.; count=0
    for name,condition in scores.items():
        path=Path(condition['tensors']['path'])
        assert _binding(path)==condition['tensors']
        tensors=torch.load(path,map_location='cpu',weights_only=True)
        for candidate,row in condition['candidates'].items():
            logits=tensors[candidate]['logits'].float()
            lp=torch.log_softmax(logits,-1)
            selected=lp[torch.arange(len(row['token_ids'])),torch.tensor(row['token_ids'])]
            error=abs(float(selected.sum())-row['sum_logprob'])
            maximum_error=max(maximum_error,error)
            assert error < 1e-4,(name,candidate,error)
            assert logits.argmax(-1).tolist()==row['winners']
            assert torch.isfinite(tensors[candidate]['head_input']).all()
            count+=1
        for fork in condition['forks']:
            a,b=fork['left'],fork['right'];j=fork['row_offset']
            ta=condition['candidates'][a]['token_ids'];tb=condition['candidates'][b]['token_ids']
            assert ta[:j]==tb[:j] and ta[j]!=tb[j]
            v=tensors[a]['logits'][j]
            assert float(v[ta[j]]-v[tb[j]])==fork['left_minus_right_margin']
    factorial={}
    if all(k in scores for k in ('AA','AB','BA','BB')):
        for candidate in scores['AA']['candidates']:
            z={k:scores[k]['candidates'][candidate]['sum_logprob'] for k in ('AA','AB','BA','BB')}
            factorial[candidate]=dict(scores=z,first_position_A_minus_B=((z['AA']+z['AB'])-(z['BA']+z['BB']))/2,
                second_position_A_minus_B=((z['AA']+z['BA'])-(z['AB']+z['BB']))/2,
                interaction=z['AA']-z['AB']-z['BA']+z['BB'],AB_minus_BA=z['AB']-z['BA'])
    return dict(status='candidate_cpu_recomputed',receipt=_binding(folder/'receipt.json'),
        consumer=_binding(Path(__file__)),candidate_rows=count,max_logprob_recompute_error=maximum_error,
        factorial=factorial,interpretation='relative exposure and earlier placement, not isolated count or facilitation')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args()
    result=check(a.folder);_write(a.folder/'cpu-check.json',result)
    print(json.dumps(result))
