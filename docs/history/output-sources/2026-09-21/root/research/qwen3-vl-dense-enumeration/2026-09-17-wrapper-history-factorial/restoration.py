import json,torch,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'2026-09-17-history-rereading-mechanism';Q=R.parent/'2026-09-17-successful-row-mechanism'
torch.set_num_threads(2)
load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
a=load(R/'runtime/RHW/capture.pt')['tensors'];s=load(P/'runtime/extract-S/capture.pt')['tensors'];f=load(R/'runtime/native-F/capture.pt')['tensors'];self=load(R/'runtime/self-F/capture.pt')['tensors']
old=Q/'stage2/runtime/residual-S-to-F-layer20';head=load(old/'states.pt')['captures']['67']['head_before'];log=load(old/'logits.pt')['raw_logits'][67]
def diff(x,y):return dict(exact=torch.equal(x,y),max_abs=float((x-y).abs().max()),l2=float((x-y).float().norm()))
checks=dict(layer20_to_native_S=diff(a['residual'][67]['20']['after'][1],s['residual'][67]['20']['before'][1]),head_to_R20=diff(a['head'][67][1],head[1]),logits_to_R20=diff(a['logits'][67][1],log[1]),self_logit_bound=diff(self['logits'][67],f['logits'][67]))
# The unchanged FP32 native/self floor determines acceptance, not an invented tolerance.
assert all(checks[k]['exact'] for k in ['layer20_to_native_S','head_to_R20','logits_to_R20','self_logit_bound']),checks
out=dict(status='passed',checks=checks,bindings=[bind(R/'runtime/RHW/capture.pt'),bind(P/'runtime/extract-S/capture.pt'),bind(old/'states.pt'),bind(old/'logits.pt')],interpretation='Expected all-source restoration, not evidence of a selective wrapper mechanism')
(R/'restoration.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(checks))
