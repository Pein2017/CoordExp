"""Independent saved-state gates, using only captured tensors and source bytes."""
import argparse,json,hashlib,os
from pathlib import Path
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();states=json.loads((a.root/'coordination/selected-states.json').read_text())['states'];reports=[];bindings={}
 for s in states:
  for key in ('source_release','source_trajectory','effective_readout'):
   b=s[key];assert sha(b['path'])==b['sha256'];bindings[b['path']]=b['sha256']
  cell=a.root/'runtime'/s['id'];r=json.loads((cell/'receipt.json').read_text());assert sha(r['capture']['path'])==r['capture']['sha256'];t=torch.load(cell/'capture.pt',map_location='cpu',weights_only=False)
  assert r['model_forwards']==2*(s['offset']+1)
  assert not Path('/proc',str(r['pid'])).exists(),f"producer still live {r['pid']}"
  assert torch.equal(t['logits'].view(torch.int32),t['no_hook_logits'].view(torch.int32))
  assert torch.equal(t['head_input'],t['no_hook_head_input'])
  release=json.loads(Path(s['source_release']['path']).read_text());step=release['trace']['steps'][s['offset']]
  assert release['target']['token_ids'][:s['offset']]==s['actual_prefix_token_ids']
  source=torch.load(s['source_trajectory']['path'],map_location='cpu',weights_only=False)
  headerr=float((source['head_inputs'][s['offset']]-t['head_input']).abs().max());coorderr=float((source['raw_coordinate_logits'][s['offset']]-t['logits'][t['coordinate_ids']]).abs().max())
  errors=[abs(float(t['logits'][v['token_id']])-v['logit']) for v in step['raw_top2']['top2']]
  assert max([headerr,coorderr]+errors)<=2e-4
  assert int(t['logits'].argmax())==s['original_winner']
  accum=t['input_residual'].double().clone()
  for av,mv in zip(t['attention'],t['mlp']):accum+=av.double();accum+=mv.double()
  err=float((accum-t['pre_final'].double()).abs().max());assert err<=2e-4
  head=accum*float(t['norm_scale'])*t['norm_weight'].double();scores=t['effective_W'].double()@head
  errscore=float((scores-t['logits'][t['coordinate_ids']].double()).abs().max());assert errscore<=2e-4
  assert int(scores.argmax())==int(t['logits'][t['coordinate_ids']].argmax())
  reports.append({'id':s['id'],'head_source_error':headerr,'coordinate_source_error':coorderr,'source_top2_error':max(errors),'residual_error':err,'all_coordinate_reconstruction_error':errscore,'status':'pass'})
 out={'status':'pass','states':len(reports),'reports':reports,'source_bindings':bindings};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'states':len(reports),'status':'pass'}))
if __name__=='__main__':main()
