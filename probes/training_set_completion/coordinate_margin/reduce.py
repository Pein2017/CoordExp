"""FP64 saved-tensor accounting; no model calls or fitted directions."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

TOL = 2e-4

def binding(p):
    p = Path(p)
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def account(t, competitors):
    d = lambda k: t[k].double()
    x, a, m = d('input_residual'), d('attention'), d('mlp')
    before, after, post = d('layer_inputs'), d('layer_outputs'), d('post_attention')
    final, gamma, head, W = d('pre_final'), d('norm_weight'), d('head_input'), d('effective_W')
    scale = float(t['norm_scale'])
    # These are measured FP32 addition residuals, not extra learned branches.
    attn_round = post - before - a
    mlp_round = after - post - m
    gaps = torch.cat((before[:1] - x, before[1:] - after[:-1], (final - after[-1])[None]), dim=0)
    summed = x + a.sum(0) + m.sum(0)
    residual = final - summed
    norm_reconstructed = final * scale * gamma
    norms = W.norm(dim=1)
    units = W / norms[:, None]
    z = W @ head
    equal = norms.median() * (units @ head)
    native = t['logits'].double()[t['coordinate_ids'].long()]
    maxabs = lambda v: float(v.abs().max())
    checks = {
        'residual_sum_max_abs': maxabs(residual),
        'interlayer_extra_max_abs': maxabs(gaps),
        'normalizer_max_abs': maxabs(norm_reconstructed-head),
        'effective_head_logits_max_abs': maxabs(z-native),
        'raw_coordinate_winner_exact': int(z.argmax()) == int(native.argmax()),
    }
    pairs = []
    for j in competitors:
        if j == 0: continue
        direction = (W[0]-W[j]) * gamma * scale
        eqdirection = norms.median() * (units[0]-units[j]) * gamma * scale
        raw = float(native[0]-native[j])
        parts = {'input':float(x@direction),'attention':(a@direction).tolist(),'mlp':(m@direction).tolist()}
        eqparts = {'input':float(x@eqdirection),'attention':(a@eqdirection).tolist(),'mlp':(m@eqdirection).tolist()}
        total = parts['input']+sum(parts['attention'])+sum(parts['mlp'])
        eqtotal = eqparts['input']+sum(eqparts['attention'])+sum(eqparts['mlp'])
        q0,qj = float(units[0]@head),float(units[j]@head)
        angular = float((norms[0]+norms[j])/2)*(q0-qj)
        length = float((norms[0]-norms[j])/2)*(q0+qj)
        pair = {'competitor':j,'raw_margin':raw,'effective_margin':float(z[0]-z[j]),
                'equal_norm_margin':float(equal[0]-equal[j]),'norm0':float(norms[0]),'normj':float(norms[j]),
                'symmetric_direction_term':angular,'symmetric_length_term':length,
                'contributions':parts,'equal_norm_contributions':eqparts,
                'reconstructed_raw_margin':total,'raw_reconstruction_residual':raw-total,
                'equal_reconstruction_residual':float(equal[0]-equal[j])-eqtotal,
                'attention_addition_rounding':(attn_round@direction).tolist(),
                'mlp_addition_rounding':(mlp_round@direction).tolist(),
                'interlayer_extra_projection':(gaps@direction).tolist(),
                'normalization_rounding_projection':float((head-norm_reconstructed)@(W[0]-W[j])),
                'readout_rounding_residual':raw-float(z[0]-z[j])}
        pairs.append(pair)
    checks['pair_margin_max_abs'] = max(abs(p['raw_reconstruction_residual']) for p in pairs)
    checks['equal_margin_max_abs'] = max(abs(p['equal_reconstruction_residual']) for p in pairs)
    checks['pass'] = all(v if isinstance(v,bool) else v <= TOL for v in checks.values())
    # Corruption follows the actual strongest pair direction, guaranteeing a
    # meaningful margin defect rather than a perturbation orthogonal to readout.
    j=competitors[1]; direction=(W[0]-W[j])*gamma*scale
    corrupted=float((x+direction/direction.square().sum()*0.01)@direction+a.sum(0)@direction+m.sum(0)@direction)
    corruption_detected=abs(float(native[0]-native[j])-corrupted)>TOL
    assert corruption_detected, 'contribution corruption escaped reconstruction gate'
    return {'checks':checks,'corruption_detected':corruption_detected,
            'raw_winner_bin':int(native.argmax()),'equal_norm_winner_bin':int(equal.argmax()),
            'coordinate_ranking_raw':torch.argsort(native,descending=True,stable=True).tolist(),
            'coordinate_ranking_equal':torch.argsort(equal,descending=True,stable=True).tolist(),
            'norm_scale':scale,'norm_scale_fp64':float(torch.rsqrt(final.square().mean()+float(t['norm_eps']))),
            'pairs':pairs}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    selected=json.loads((args.root/'coordination/selected-states.json').read_text())
    results=[]
    for state in selected['states']:
        paths=list((args.root/'runtime').rglob(state['id']+'/capture.pt'))
        if not paths: continue
        assert len(paths)==1, paths
        tensor=torch.load(paths[0],map_location='cpu',weights_only=False)
        row=account(tensor,state['fixed_competitor_bins']);row.update(id=state['id'],model=state['model'],source_policy=state['source_policy'],offset=state['offset'],capture=binding(paths[0]))
        results.append(row)
    assert results, 'no captures'
    result={'schema':'coordinate_margin.cpu_accounting.v1','tolerance':TOL,'state_count':len(results),'all_pass':all(x['checks']['pass'] for x in results),'states':results}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'states':len(results),'all_pass':result['all_pass']}))

if __name__=='__main__':main()
