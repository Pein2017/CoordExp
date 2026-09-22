import hashlib,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
torch.set_num_threads(2)
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
paths={k:R/f'stage2/runtime/native-{k}/states.pt' for k in ['S','F']}
paths['S_components']=R/'stage2/component-capture/components.pt'
paths['F_components']=R/'stage2/runtime/mlp-S-to-F-layer20/components.pt'
a={k:torch.load(p,map_location='cpu',weights_only=False) for k,p in paths.items()}
out={k:a[k]['captures']['67']['layers']['20']['before'][1].double() for k in ['S','F']}
attn={k:a[k+'_components']['states']['attention'][1].double() for k in ['S','F']}
mlp={k:a[k+'_components']['states']['mlp'][1].double() for k in ['S','F']}
# Actual decoder equation: output = incoming residual + attention output + MLP output.
# Incoming was not directly archived: infer it algebraically, with FP32 rounding caveat.
incoming={k:out[k]-attn[k]-mlp[k] for k in ['S','F']}
delta=out['S']-out['F'];parts={'incoming_residual_inferred':incoming['S']-incoming['F'],'attention_output':attn['S']-attn['F'],'mlp_output':mlp['S']-mlp['F']}
assert torch.allclose(sum(parts.values()),delta,atol=1e-10,rtol=0)
result=dict(status='candidate_offline_accounting',sources={k:bind(p) for k,p in paths.items()},site='zero-based layer20, action67, target1',output_difference_norm=float(delta.norm()),parts={k:dict(difference_norm=float(v.norm()),signed_projection_on_output_difference=float(v.dot(delta)/delta.dot(delta))) for k,v in parts.items()},limits=['Incoming residual is algebraically inferred, not a directly captured layer input.','Equation has FP32 residual-addition rounding; projections are descriptive vector accounting, not causal attribution percentages.','Native F attention/MLP before-values come from the MLP-only run before its sole intervention; its literal prefix and all earlier computation are unchanged.','Neither component-alone failure nor these vectors identifies a unique information origin or neural circuit.'])
(R/'layer20-decomposition.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['sources','limits']}))
