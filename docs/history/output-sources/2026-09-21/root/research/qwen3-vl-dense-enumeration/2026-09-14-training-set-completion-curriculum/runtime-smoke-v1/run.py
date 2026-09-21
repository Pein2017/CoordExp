import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT.parent/'runtime-preparation-v1/manifest.json'
started=time.monotonic(); phase='initial'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name,args):
 global phase
 phase=name
 with (ROOT/(name+'.log')).open('x') as log:
  result=subprocess.run([sys.executable,'-m','probes.training_set_completion.training',*args],stdout=log,stderr=subprocess.STDOUT,timeout=660)
 if result.returncode:raise RuntimeError(f'{name} exit {result.returncode}')
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
receipt={'schema':'training_set_completion.runtime_smoke.v1','pid':os.getpid(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'manifest':str(MANIFEST),'manifest_sha256':h(MANIFEST),'scope':'single-image2-update technical gradient/export/resume/readback only; no scientific11image-stagepass'}
try:
 run('full',['run','--manifest',str(MANIFEST),'--output',str(ROOT/'full'),'--device','cuda:0'])
 run('resumed',['run','--manifest',str(MANIFEST),'--output',str(ROOT/'resumed'),'--resume',str(ROOT/'full/checkpoints/step-00001'),'--device','cuda:0'])
 full=ROOT/'full/checkpoints/step-00002'; resumed=ROOT/'resumed/checkpoints/step-00002'
 a=torch.load(full/'state.pt',map_location='cpu',weights_only=False);b=torch.load(resumed/'state.pt',map_location='cpu',weights_only=False)
 assert a['step']==b['step']==2 and equal(a['optimizer_state_dict'],b['optimizer_state_dict']), 'optimizer resume mismatch'
 assert h(full/'adapter/adapter_model.safetensors')==h(resumed/'adapter/adapter_model.safetensors'),'adapter resume mismatch'
 for name,checkpoint in [('full-readback',full),('resumed-readback',resumed)]:
  run(name,['readback','--manifest',str(MANIFEST),'--adapter',str(checkpoint/'adapter'),'--output',str(ROOT/(name+'.json')),'--device','cuda:0'])
 c=json.loads((ROOT/'full-readback.json').read_text());d=json.loads((ROOT/'resumed-readback.json').read_text())
 assert [r['generated_token_ids'] for r in c['rows']]==[r['generated_token_ids'] for r in d['rows']], 'cold natural readback mismatch'
 receipt.update(status='passed',resume_optimizer_exact=True,resume_adapter_bytes_exact=True,cold_natural_tokens_exact=True,readback_generated_tokens=[len(r['generated_token_ids']) for r in c['rows']])
except BaseException as exc:
 receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
finally:
 receipt.update(last_phase=phase,elapsed_seconds=time.monotonic()-started)
 (ROOT/'terminal.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('GPU_SMOKE_SUCCESS' if receipt['status']=='passed' else 'GPU_SMOKE_FAILED',flush=True)
sys.exit(0 if receipt['status']=='passed' else 1)
