"""Freeze actual saved four/eight-row histories; no outcome-based selection."""
import json,hashlib,datetime
from pathlib import Path
from probes.training_set_completion.numerical_feedback.select import rows
from probes.training_set_completion.numerical_feedback.metrics import release_metrics
BASE=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
PRE=BASE/'2026-09-18-readout-common-component';OUT=BASE/'2026-09-18-history-readout-crossover'
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def th(t):return hashlib.sha256(json.dumps(t,separators=(',',':')).encode()).hexdigest()
def main():
 plan=json.loads((PRE/'execution-plan.json').read_text());selection=json.loads(Path(plan['selection']).read_text());bs=[b for b in selection['boundaries'] if b['kind']=='failure'];assert len(bs)==11
 histories=[];cells=[];unsupported=[]
 for b in bs:
  for policy in ['original','full']:
   paths=list((PRE/'runtime').glob('*/'+b['id']+'--'+policy+'/release.json'));assert len(paths)==1
   p=paths[0];d=json.loads(p.read_text());t=d['target']['token_ids'];rr=rows(t)
   for cut in [4,8]:
    hid=f"{b['id']}--cut{cut}--history-{policy}"
    if len(rr)<cut or 151645 in t[:rr[cut-1]['end']]:unsupported.append(dict(id=hid,reason='source_terminated_before_cut',source=bind(p)));continue
    added=t[:rr[cut-1]['end']];prefix=b['native_tokens'][:b['source_row']['end']]+added
    h=dict(id=hid,boundary=b,history_policy=policy,cut=cut,prefix_tokens=prefix,prefix_sha256=th(prefix),added_tokens=added,added_sha256=th(added),source=bind(p),source_receipt=bind(p.parent/'receipt.json'),saved_overlap_tokens=t[len(added):],supplied_rows=rr[:cut],history_metrics=release_metrics(added,b['source_row']))
    histories.append(h)
    for future in ['original','full']:cells.append(dict(id=hid+'--future-'+future,history_id=hid,model=b['model'],policy=future,qualification=b['id']=='untied-885-failure' and cut==4))
 for model in ['tied','untied']:
  for i,c in enumerate(c for c in cells if c['model']==model and not c['qualification']):c['gpu']=i%4+(4 if model=='untied' else 0)
 for c in cells:
  if c['qualification']:c['gpu']=0
 OUT.mkdir(exist_ok=True);p=OUT/'execution-plan.json';assert not p.exists()
 result=dict(status='frozen_before_gpu',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),panel=plan['panel'],selection=plan['selection'],bindings=[bind(PRE/n) for n in ['lead-acceptance.json','execution-plan.json','integrated-terminal.json']]+[bind(Path(plan['selection'])),bind(Path(plan['panel']))],histories=histories,cells=cells,unsupported=unsupported,bounds=dict(target_continuations=88,model_forwards=100000,gpu_seconds=28800,tensor_bytes=16*1024**3,release_tokens=512,release_rows=32),expected_max_release_forwards=len(cells)*512,qualification='untied-885-failure cut4, fourcells counted and reused',conditioning='Exact actual target histories. Native prompt masks/media/batch membership unchanged. Companion outputs uninterpretable; source-order companion prefixes rebuilt at matching width; all44 unchanged-policy source-overlap checks required.')
 p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(histories=len(histories),cells=len(cells),unsupported=unsupported)))
if __name__=='__main__':main()
