"""Saved-evidence integration only. No model execution or global research writes."""
import json,hashlib,collections,datetime
from pathlib import Path
A=Path(__file__).parent;ROOT=A.parent;B=ROOT/'2026-09-18-untied-active-readout-geometry';C=ROOT/'2026-09-18-untied-gradient-path-accounting';D=ROOT/'2026-09-18-untied-recurrence-feedback';D.mkdir(exist_ok=True)
REPO=Path('/data/CoordExp/.worktrees/research-probes');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
cache={}
def sha(p):
 p=Path(p)
 if str(p) not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  cache[str(p)]=h.hexdigest()
 return cache[str(p)]
def bind(p):p=Path(p);return dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size)
panel=read(A/'panel.json');d=read(A/'reduction.json');assert d==read(A/'reduction-recheck.json')
# Historical producer names can now refer to a corrected current file; resolve exact executed snapshots.
snapshots={sha(p):str(p) for root in [A/'sources',C/'cpu-axis-correction'] for p in root.glob('*.py')}
verified={};relocations=[]
def verify(x):
 if isinstance(x,dict):
  if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):
   f=Path(x['path']);expected=x['sha256'];actual=sha(f) if f.is_file() else None
   if actual!=expected:
    assert expected in snapshots,(str(f),expected,actual)
    relocations.append(dict(original_path=str(f),sha256=expected,executed_snapshot=snapshots[expected]));f=Path(snapshots[expected])
   verified[(str(f),expected)]=bind(f)
  for v in x.values():verify(v)
 elif isinstance(x,list):
  for v in x:verify(v)
verify(panel['sources']);verify(read(A/'shared-gate.json'))
receipts=[];pids=[]
for cond in panel['conditions']:
 for group in panel['groups']:
  f=A/'runtime'/cond/group['key']/'receipt.json';r=read(f);assert r['status']=='candidate_complete' and r['no_parameter_mutation'];verify(r);receipts.append(r);pids.append(r['pid'])
assert len(receipts)==148 and sum(len(v) for v in d['cells'].values())==580
exits={str(p):p.read_text().strip() for p in A.glob('main-*.exit')};assert len(exits)==6 and set(exits.values())=={'0'}
for model in ['tied','untied']:
 r=read(B/'native-captures'/f'{model}-original'/'receipt.json');verify(r);pids.append(r['pid']);assert all(c['replay_parity'] for c in r['captures'])
 assert (B/'native-captures'/f'{model}.exit').read_text().strip()=='0'
pids.append(read(C/'job-closure.json')['pid']);live=[p for p in sorted(set(pids)) if Path(f'/proc/{p}').exists()];assert not live,live
review=read(A/'physical-review/review-phase1.json');events=read(A/'physical-review/events.json')['events'];keys={f"{e['image_id']}:{e['model']}:{e['kind']}" for e in events};assert keys=={e['key'] for e in review['events']};assert len(keys)==30
review.update(final_selection_verified=True,source_selection=bind(A/'physical-review/events.json'),note='Full 18-image selection adds no new bird event. Reuse the single completed bounded review, not a second census.');write(A/'physical-review/final-review.json',review)
verification=dict(status='passed',natural_reduction_json_exact=True,groups=148,rows=580,verified_unique_bindings=len(verified),bindings=list(verified.values()),historical_source_resolutions=list({(x['original_path'],x['sha256']):x for x in relocations}.values()),natural_exits=exits,owned_pids=sorted(set(pids)),live_jobs=[],physical_events=30,readout_reconstruction=bind(B/'summary.json'),checked_at=now)
write(A/'verification.json',verification)
# D stops at missing physical/control admission; selected native source remains available.
cell=d['cells']['untied-original']['417044'];raw=read(cell['raw']['path'])['rows'][cell['batch_index']];tokens=raw['token_ids'];view=cell['views']['refined'];rep=view['first_strict_repeat'];row=next(x for x in view['valid_predictions'] if x['prediction_id']==rep['prediction_id'])
# Bind the actual raw trajectory without inventing an unverified slot mapping.
Dresult=dict(status='admission_HOLD',model_calls=0,model='untied-plus-axis2444',native_candidate=dict(image_id=417044,group=cell['group'],batch_index=cell['batch_index'],raw=cell['raw'],token_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),first_repeat_proxy=rep,row={k:row[k] for k in ['prediction_id','generated_order','description','coord_bins_1000']}),physical_evidence=[e for e in review['events'] if e['image_id']==417044 and e['model']=='untied'],missing_gate='Covered physical A, credible unvisited B and matched healthy native boundary are not jointly admitted. Donut partial-box recurrence remains identity/extent HOLD; proxy overlap is insufficient.',unfrozen=['exact failing-prefix coordinate slots','physical A/B score contrast','healthy matched native prefix','directions','epsilon','numerical gate','model budget'],decision='No feedback inference, Jacobian or finite-difference measurement performed; root release required.',sources=[bind(A/'reduction.json'),bind(A/'physical-review/final-review.json'),bind(B/'events.json')])
write(D/'admission.json',Dresult)
# Decision-bearing aggregates retain original strata and per-owner ledgers in reduction.json.
Aresult=dict(status='candidate',technical='148/148 complete; 580/580 output rows; independent saved-output reduction JSON-exact',summaries=d['summaries'],comparisons=d['comparisons'],shadow=d['shadow'],cost=d['cost'],physical_review=bind(A/'physical-review/final-review.json'),sources=[bind(A/'panel.json'),bind(A/'shared-gate.json'),bind(A/'reduction.json'),bind(A/'verification.json')],limits=['untied+axis is a complete training package, not an untie-only intervention','sentinel128 is reused, not fresh held-out','unmatched is UNKNOWN, not physical FP','known-match turnover is not automatically physical gain/loss','bootstrap intervals are paired image descriptive intervals'],live_jobs=[])
write(A/'result.json',Aresult)
br=read(B/'summary.json');write(B/'result.json',dict(status='candidate',models=br['models'],runtime=read(B/'runtime.json'),sources=[bind(B/'events.json'),bind(B/'summary.json'),bind(B/'reconstruction-recheck.json')],limits=br['limits'],live_jobs=[]))
Cresult=dict(status='technical_invalid',scientific='unanswered',failure=read(C/'job-closure.json'),cpu_correction=bind(C/'cpu-axis-correction/receipt.json'),root_cpu_only_verification=bind(C/'cpu-axis-correction/lead-verification.json'),model_rerun=False,live_jobs=[]);write(C/'result.json',Cresult)
write(D/'result.json',Dresult)
for root,ev,disp in [(A,'unreviewed','Natural candidate: mixed package-specific benefit and damage; no promotion'),(B,'unreviewed','Active readout candidate: norm-sensitive subset and substantial surviving alignment; observational layers'),(C,'invalid','Gradient technical-invalid; CPU axis-context correction verified, numerical question unanswered'),(D,'unreviewed','CPU admission-HOLD; no released or executed model work')]:
 unit=REPO/'research/experiments'/root.name;state=read(unit/'state.json');state.update(lifecycle='closed',evidence=ev,disposition=disp,state_as_of=now,result=f'research/experiments/{root.name}/results.md',state_source=f'research/experiments/{root.name}/results.md',next_action='Root independent acceptance; no automatic continuation');write(unit/'state.json',state)
 cost= d['cost'] if root==A else read(B/'runtime.json') if root==B else read(C/'job-closure.json') if root==C else {'model_calls':0}
 write(root/'terminal.json',dict(status='candidate_complete' if root in [A,B] else 'technical_invalid' if root==C else 'admission_HOLD',acceptance='unreviewed_by_root',result=bind(root/'result.json'),cost=cost,live_jobs=[],closed_at=now))
storage={root.name:sum(p.stat().st_size for p in root.rglob('*.pt')) for root in [A,B,C,D]}
write(A/'storage-accounting.json',dict(actual_pt_file_bytes=storage,note='Actual file sizes, including backing storage. B static tensor logical bytes73736192 differ from serialized file1316221045; actual total remains under16GiB. No original receipts changed.'))
agg=dict(model_forwards=d['cost']['forwards']+read(B/'runtime.json')['model_forwards']+7,backwards=2,allocated_gpu_seconds=d['cost']['allocated_gpu_seconds_completed_workers']+read(B/'runtime.json')['allocated_gpu_seconds']+read(C/'job-closure.json')['allocated_gpu_seconds'],additional_model_calls_during_closeout=0,actual_pt_file_bytes=storage)
write(A/'integrated-terminal.json',dict(status='PACKAGE_CANDIDATE',acceptance='unreviewed_by_root',units={root.name:bind(root/'terminal.json') for root in [A,B,C,D]},cost=agg,live_jobs=[],closed_at=now))
print(json.dumps(dict(verified_bindings=len(verified),source_resolutions=len(relocations),cost=agg,status='candidate'),indent=2))
