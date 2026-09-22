"""Freeze all temperature/seed cells, never choose by sampled outcomes."""
import json,hashlib,copy
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
m=read(R/'runtime-manifest.json');base=read(R/'panels/B/309264-established_one.json');producer=R/'sampling-producer.py';assert producer.exists()
entries=[]
for temp in [.1,.3,.7]:
 for seed in range(19,27):
  name=f'T{temp:.1f}-seed{seed}';p=copy.deepcopy(base);c=p['cases'][0];c.update(sampling_temperature=temp,sampling_seed=seed,max_sample_tokens=32);p['condition']=name;p['schema']='repetition_history.sampling.v1';p['sources'].append(bind(producer));p['sources'].append(bind(Path('research/experiments/2026-09-17-repetition-history-mechanism/stage-c.md').resolve()));path=R/'panels/C'/f'309264-{name}.json';path.parent.mkdir(exist_ok=True);assert not path.exists();path.write_text(json.dumps(p,indent=2)+'\n');entries.append(dict(stage='C',image_id=309264,condition=name,mode='pulse1',panel=bind(path),output_root=str(R/'runtime/C'/name),temperature=temp,seed=seed,producer=bind(producer)))
m['cells'].extend(entries);(R/'runtime-manifest.json').write_text(json.dumps(m,indent=2)+'\n');(R/'sampling-manifest.json').write_text(json.dumps(dict(status='frozen',cells=entries,seeds=list(range(19,27)),temperatures=[.1,.3,.7],bounds=dict(executions=24,forwards=74016,max_sampled_tokens_per_cell=32)),indent=2)+'\n');print(json.dumps(dict(cells=len(entries),producer=bind(producer))))
