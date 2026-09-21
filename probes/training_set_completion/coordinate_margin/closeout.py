"""Seal the one frozen eight-state accounting candidate, without root acceptance."""
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-margin-provenance')
UNIT=Path('research/experiments/2026-09-19-coordinate-margin-provenance')
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def bind(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 d=json.loads((ROOT/'reduction.json').read_text());v=json.loads((ROOT/'verification.json').read_text());assert d['all_pass'] and d['state_count']==8 and v['status']=='pass';assert d==json.loads((ROOT/'reduction-recheck.json').read_text())
 receipts=[json.loads((ROOT/'runtime'/s['id']/'receipt.json').read_text()) for s in d['states']]
 assert all(r['status']=='candidate_complete' and not Path('/proc',str(r['pid'])).exists() for r in receipts)
 settled=json.loads((ROOT/'launch/scaleout-settled.json').read_text());assert all(x['exit_code']==0 for x in settled['jobs']);assert (ROOT/'launch/pilot.exit').read_text().strip()=='0'
 cost={'states':8,'prefix_replays':16,'model_forwards':sum(r['model_forwards'] for r in receipts),'allocated_gpu_seconds':sum(r['gpu_seconds'] for r in receipts),'retained_tensor_bytes':sum(p.stat().st_size for p in (ROOT/'runtime').rglob('*.pt')),'gpus_used':list(range(8)),'failed_model_attempts':0,'free_rollouts':0}
 assert cost['model_forwards']<=10000 and cost['allocated_gpu_seconds']<=7200 and cost['retained_tensor_bytes']<=4*1024**3
 write(ROOT/'cost.json',cost);write(ROOT/'job-closure.json',{'status':'all_ended','jobs':[{'pid':r['pid'],'state_id':r['state_id'],'exit_code':0,'live':False} for r in receipts],'children':[{'name':'feedback_luna','model':'gpt-5.6-luna','effort':'max','status':'completed','contribution':'unexecuted draft; parent corrected and executed','live':False}]})
 with (ROOT/'contributions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['state','competitor','component','layer','raw_projection','equal_norm_projection'])
  for s in d['states']:
   for p in s['pairs']:
    c,e=p['contributions'],p['equal_norm_contributions'];w.writerow([s['id'],p['competitor'],'input','',c['input'],e['input']])
    for key in ('attention','mlp'):
     for i,(a,b) in enumerate(zip(c[key],e[key])):w.writerow([s['id'],p['competitor'],key,i,a,b])
 differences=[]
 by={s['id']:s for s in d['states']}
 for model,policy in [('tied','pair01'),('untied','pair01'),('untied','only0')]:
  donor=by[f'{model}-{policy}-offset15'];native=by[f'{model}-original-offset15']
  for q in donor['pairs']:
   n=next((p for p in native['pairs'] if p['competitor']==q['competitor']),None)
   if n is None:continue
   a,b=q['contributions'],n['contributions']
   differences.append({'model':model,'policy':policy,'competitor':q['competitor'],'raw_margin_difference':q['raw_margin']-n['raw_margin'],'input_difference':a['input']-b['input'],'attention_difference':[x-y for x,y in zip(a['attention'],b['attention'])],'mlp_difference':[x-y for x,y in zip(a['mlp'],b['mlp'])],'scale_note':'Each state retains its own actual final RMS scale; this is an accounting difference, not a fixed-state causal intervention.'})
 write(ROOT/'history-differences.json',differences)
 summary=[]
 for s in d['states']:
  p1=next(p for p in s['pairs'] if p['competitor']==1);p30=next(p for p in s['pairs'] if p['competitor']==30)
  summary.append({'id':s['id'],'raw_winner':s['raw_winner_bin'],'equal_winner':s['equal_norm_winner_bin'],'margin_0_1':p1['raw_margin'],'equal_margin_0_1':p1['equal_norm_margin'],'margin_0_30':p30['raw_margin'],'equal_margin_0_30':p30['equal_norm_margin'],'direction_0_30':p30['symmetric_direction_term'],'length_0_30':p30['symmetric_length_term'],'input_0_30':p30['contributions']['input'],'attention_0_30':sum(p30['contributions']['attention']),'mlp_0_30':sum(p30['contributions']['mlp'])})
 result={'status':'INTEGRATED_CANDIDATE','acceptance':'unreviewed','technical':'all eight states pass','scientific':'Exact distributed score accounting with margin-specific readout-length effects; no causal localization or physical-recovery claim.','summary':summary,'cost':cost,'checks':{'max_residual':max(s['checks']['residual_sum_max_abs'] for s in d['states']),'max_pair_reconstruction':max(s['checks']['pair_margin_max_abs'] for s in d['states']),'source_head_error':max(x['head_source_error'] for x in v['reports']),'source_coordinate_error':max(x['coordinate_source_error'] for x in v['reports']),'corruption_detected_all':all(s['corruption_detected'] for s in d['states'])},'failed':[],'HOLD':[],'limitations':['Outcome-selected eight x1 states on one image, mature package comparison not untie-only.','Observed histories differ in whole earlier rows, not isolated x1.','Signed projections are exact accounting, not causal responsibility or early-exit predictions.','No physical review or free rollout.','No-hook means no new residual instrumentation; existing score/head observation remains identical to qualified native capture.','Position identity follows unchanged native builder, masks and incremental generate; no extra position or KV tensor archive.'],'reduction':bind(ROOT/'reduction.json'),'verification':bind(ROOT/'verification.json')}
 write(ROOT/'result.json',result)
 lines=['# Coordinate-margin provenance — candidate','', 'Eight frozen states passed the forward and CPU gates. Evidence remains **unreviewed** pending root acceptance. No new free continuation was generated.','', '| State | Raw winner | Equal-norm winner | Raw 0−1 | Equal 0−1 | Raw 0−30 | Equal 0−30 |','|---|---:|---:|---:|---:|---:|---:|']
 for s in summary:lines.append(f"| {s['id']} | {s['raw_winner']} | {s['equal_winner']} | {s['margin_0_1']:.6f} | {s['equal_margin_0_1']:.6f} | {s['margin_0_30']:.6f} | {s['equal_margin_0_30']:.6f} |")
 lines+=['','## What the accounting establishes','', 'At tied pair01 offset15, the raw 0−30 margin is +0.332750. Its symmetric direction term is −0.589133 and row-length term +0.921884: length reverses that selected-pair ordering. Equal norms give −0.582937, but the global equal-norm winner is 23, not 30. The 0−1 margin remains positive after equalization in all eight states. These are distinct competitions.','', 'For 0−1, the final MLP (zero-based layer27) is the largest positive component in all eight states; its contribution ranges approximately +0.4315 to +0.6734. Other attention/MLP contributions oppose or reinforce it. This does not establish that layer27 originates or causes recurrence.','', 'For tied 0−30, native offset15 has attention sum +0.64978 and MLP sum +0.64548; pair01 offset15 has +0.51103 and −0.17804. The margin change is distributed, with several late MLP updates changing sign or magnitude. At tied pair01 offset25 the raw margin becomes negative and raw/equal winner is52. Untied pair01/only0 offset15 select23/46 while 0−1 still favors0, showing why a selected pair is not a full-winner explanation.','', 'The strongest surviving account is ordinary history-dependent distributed computation combined with static readout-length advantages. This accounting supplies candidate components, not a unique faulty circuit, training-origin explanation or owner ledger. No patch or successor is authorized here.','', '## Exact convention and evidence','', 'Let n_i=||W_i|| and q_i=(W_i/n_i)·h. Raw margin is decomposed as ((n0+nj)/2)(q0−qj)+((n0−nj)/2)(q0+qj). Equal-norm scores use the accepted lower median coordinate norm. There is no centering or bias. Effective W includes base plus the actual output delta. Each residual contribution is projected through that state’s actual final RMS scale and gain; cumulative values are accounting views, not logit-lens predictions. FP32 addition, normalization and head-rounding residuals are retained separately. Target inter-layer extra contribution is zero at every measured seam.','', f"Maximum residual-vector reconstruction error: {result['checks']['max_residual']:.9g}; maximum selected-pair error: {result['checks']['max_pair_reconstruction']:.9g}. Both are below frozen2e-4. All source head/coordinate comparisons are exact; hooks preserve full-vocabulary scores bitwise. Independent saved-tensor reconstruction and +0.01-margin corruption sensitivity pass. No model reruns or failed model attempts.",'', f"Cost: {cost['model_forwards']} batch forwards,16 prefix replays, {cost['allocated_gpu_seconds']:.3f} allocated GPU-seconds across GPUs0–7; {cost['retained_tensor_bytes']} retained tensor bytes. All8 producers exited0; no live child or owned job.",'', 'The supplied prefixes retain native heterogeneous companions, image/masks and incremental positions. Companion outputs are not a scientific outcome. No attention/KV archive, gradient, physical review or new generation was acquired.','', f"Artifacts: `{ROOT}/artifact-map.json (summary)`; result, reconstruction, verification, costs and closure are under the same root. Exact checkpoint/config/input identities are in every `runtime/<state>/receipt.json` and its bound native receipt."]
 (UNIT/'results.md').write_text('\n'.join(lines)+'\n')
 state=json.loads((UNIT/'state.json').read_text());state.update(lifecycle='closed',evidence='unreviewed',state_as_of=datetime.now(timezone.utc).isoformat(),result=str(UNIT/'results.md'),state_source=str(UNIT/'results.md'),disposition=result['scientific'],next_action='Root independently accepts or rejects this stable candidate; no successor authorized.');write(UNIT/'state.json',state)
 write(ROOT/'artifact-map.json', {'summary': '''# Reconstruction map
- `coordination/selected-states.json`: frozen8 source states, hashes, offsets and tokens.
- `execution-plan.json`, `sources/`: prelaunch producer/reducer snapshots, actual installed forward source, unexecuted child draft.
- `runtime/<state>/capture.pt`: actual residual/component/norm/head/logit tensors, no-hook comparison, effective rows and source states.
- `runtime/<state>/receipt.json` and `native/`: exact model/config/media/input/group provenance and per-state counters.
- `reduction.json`, `reduction-recheck.json`: JSON-exact FP64 accounting, full rankings and signed per-layer contributions including rounding residuals.
- `verification.json`: independent source/no-hook/all-coordinate reconstruction checks.
- `contributions.csv`, `history-differences.json`: full signed tables and conditional-history differences.
- `launch/`: commands, logs, PIDs and observed exits; `cost.json`, `job-closure.json`.
- `result.json`, `integrated-terminal.json`: candidate scientific/technical result; root acceptance is separate.
- `direct-delivery.json`: one direct App Server completion delivery; no wake monitor.

Run from /data/CoordExp/.worktrees/research-probes:
```sh
python probes/training_set_completion/coordinate_margin/reduce.py --root '''+str(ROOT)+''' --output /tmp/coordinate-margin-recomputed.json
python probes/training_set_completion/coordinate_margin/verify.py --root '''+str(ROOT)+''' --output /tmp/coordinate-margin-verification.json
```
No full KV/attention maps or extra physical labels are captured. Positions are bound by native source/replay identity, not an additional all-position tensor archive.
'''})
 files=[p for p in ROOT.rglob('*') if p.is_file() and 'coordination' not in p.relative_to(ROOT).parts and p.name not in ('integrated-terminal.json','direct-delivery.json')]
 files += [UNIT/'unit.md',UNIT/'results.md',UNIT/'state.json']+list(Path('probes/training_set_completion/coordinate_margin').glob('*.py'))
 write(ROOT/'integrated-terminal.json',{'status':'INTEGRATED_CANDIDATE','acceptance':'unreviewed','result':bind(ROOT/'result.json'),'bindings':[bind(p) for p in sorted(set(files))],'cost':cost,'live_jobs':0,'live_children':0,'cleanup':[]})
 with (ROOT/'coordination/events.log').open('a') as f:f.write(json.dumps({'event':'INTEGRATED_CANDIDATE','result_ref':str(ROOT/'integrated-terminal.json'),'time':datetime.now(timezone.utc).isoformat()})+'\n')
 print(json.dumps({'cost':cost,'checks':result['checks'],'terminal':bind(ROOT/'integrated-terminal.json')},indent=2))
if __name__=='__main__':main()
