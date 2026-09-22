from pathlib import Path
import argparse,json,torch
from probes.training_set_completion.training import raw_axis_validity_hinge
p=argparse.ArgumentParser();p.add_argument('--original',action='store_true');a=p.parse_args()
if a.original:
 source=(Path(__file__).parent/'producer-before.py').read_text();start=source.index('                axis=raw_axis_validity_hinge');end=source.index("                return {'ce':ce",start);body=source[start:end];body='\n'.join(line[16:] for line in body.splitlines());code="def axis_term(logits,boxes,coords):\n    route={'trusted_boxes':boxes}\n"+'\n'.join('    '+line for line in body.splitlines())+'\n    return axis\n';scope={'raw_axis_validity_hinge':raw_axis_validity_hinge};exec(code,scope);fn=scope['axis_term']
else:
 from probes.training_set_completion.untied_gradient import axis_term as fn
x=torch.zeros(4,1000);x[0,999]=15;x[1,999]=15;x[2,0]=15;x[3,0]=15;x.requires_grad_();boxes=[dict(x1_position=0,y1_position=1,x2_position=2,y2_position=3)];coords=list(range(1000))
v=fn(x,boxes,coords)
with torch.no_grad():u=fn(x,boxes,coords)
assert float(v)>0.5
assert torch.equal(v.detach(),u),('nonempty no_grad value changed',float(v),float(u))
z=fn(x,[],coords);g=torch.autograd.grad(z,x)[0];assert float(z)==0 and torch.count_nonzero(g)==0
with torch.no_grad():assert float(fn(x,[],coords))==0
print(json.dumps(dict(status='passed',nonempty_value=float(v.detach()),no_grad_value=float(u),zero_box_zero_gradient=True,model_calls=0)))
