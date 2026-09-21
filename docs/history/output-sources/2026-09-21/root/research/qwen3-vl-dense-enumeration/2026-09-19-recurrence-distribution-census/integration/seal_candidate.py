"""Seal the four completed candidates; no scientific/model work."""
import datetime,hashlib,json,os,shutil
from pathlib import Path
repo=Path('/data/CoordExp/.worktrees/research-probes');base=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration');out=Path(__file__).parent
names=['recurrence-distribution-census','recurrence-spatial-source','recurrence-conditional-mass','coordinate-input-continuity']
roots={k:base/('2026-09-19-'+n) for k,n in zip('ABCD',names)}
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
def save(name,obj):
 p=out/name;assert not p.exists(),p;p.write_text(json.dumps(obj,indent=2)+'\n');return p
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
# Fresh executable process check, without touching shared stress or unrelated work.
needles=['recurrence_census','recurrence_spatial','recurrence_mass','coordinate_continuity'];live=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=(proc/'cmdline').read_bytes().split(b'\0');decoded=[x.decode(errors='replace') for x in args if x]
  if decoded and 'python' in Path(decoded[0]).name and any(any(n in arg for n in needles) for arg in decoded[1:]):live.append({'pid':int(proc.name),'args':decoded})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not live,live
closures=[roots['A']/'finalization-receipt.json',roots['B']/'final/broad-v1/job-closure.json',roots['B']/'final/technical-gate-v3/closure.json',roots['D']/'runtime/closure.json',roots['C']/'run-manifest.json']
job=save('job-closure.json',{'status':'closed','checked_utc':now,'owned_live_model_processes':live,'children':{'recurrence_census':'completed','recurrence_spatial':'completed','recurrence_mass':'completed','feedback_luna':'completed'},'routing':'four disjoint gpt-5.6-luna max children, no descendants','child_status_basis':'native agents.list_agents and delivered final records; census completed earlier and no longer listed','receipts':[bind(p) for p in closures],'C_closure_scope':'all45 states/11520 draws present; run completed and fresh matching process scan empty; no fabricated OS exit receipt'})
files=[];changed=[]
for name in names:
 src=repo/'research/experiments'/('2026-09-19-'+name)
 for p in sorted(src.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:
   q=out/'candidate-records'/p.relative_to(repo);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);files.append(q);changed.append(str(p.relative_to(repo)))
for dirname in needles:
 src=repo/'probes/training_set_completion'/dirname
 for p in sorted(src.rglob('*.py')):
  q=out/'source-snapshot'/p.relative_to(repo);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);files.append(q);changed.append(str(p.relative_to(repo)))
save('changed-paths.json',{'repository_owned_files':changed,'routing_only_edits':['research/index.md','research/experiments/catalog.jsonl'],'routing_note':'only current unreviewed package entries appended; unrelated prior edits remain','artifact_roots':[str(p) for p in roots.values()],'production_or_checkpoint_edits':[],'cleanup_deleted_bytes':0})
selected={
 'A':['result.json','scientific-synthesis.json','mature-census.json','new-census.json','shared-panel.json','shared-sources.json','selection-rule.json','identity-audit.json','exclusions.json','archival/archival-binding-reconciliation.json'],
 'B':['final/broad-v1/result.json','final/broad-v1/artifact-map.json','final/broad-v1/cost-receipt.json','final/broad-v1/job-closure.json','final/attempt-ledger.json','final/cost-receipt.json','final/correction/correction-receipt.json','final/technical-gate-v3/source.pt','final/technical-gate-v3/target.pt','final/technical-gate-v3/trace.json','final/technical-gate-v3/compare.json'],
 'C':['draws.jsonl','reduction.json','state-bindings.json','sampler-entry.json','qualification-attempts.json','run-manifest.json','launch-failures.json'],
 'D':['result.json','reduction.json','verification.json','cpu-geometry.json','selection.json','runtime/closure.json']}
for lane,rels in selected.items():files.extend(roots[lane]/p for p in rels)
checks=['cohort-check.json','mature-check.json','new-census-check.json','continuity-cpu-check.json','continuity-capture-check.json','continuity-reduction-check.json','mass-events-check.json','mass-reduction-check.json','mass-conditioning-check.json','spatial-cpu-recomputed/parent-check.json','spatial-input-check.json','spatial-pilot-check.json','parity-position-red-green.json','spatial-parity-v3-check.json','spatial-broad-check.json','spatial-aggregate-check.json','spatial-bindings-check.json','knowledge-check-final.log']
assert json.loads((out/'knowledge-check-final.log').read_text())['ok']
files.extend(out/p for p in checks);files.extend([out/'cost.json',out/'results.md',out/'ARTIFACTS.md',job,out/'changed-paths.json',out/'spatial-effect-summary.json'])
files.extend(p for p in out.glob('*.py') if p.name!='seal_candidate.py')
state=json.loads((out/'integration-state.json').read_text());state['status']='INTEGRATED_CANDIDATE';state['lanes']['B']['scientific']='candidate_with_local_admission_HOLD';state['next']='root independent acceptance; no automatic successor';state['all_children_settled']=True;state['all_model_jobs_ended']=True;(out/'integration-state.json').write_text(json.dumps(state,indent=2)+'\n');files.append(out/'integration-state.json')
result=save('result.json',{'schema':'recurrence_source_distribution.integrated_candidate.v1','status':'INTEGRATED_CANDIDATE','lead_acceptance':'pending','completed_utc':now,'panel':bind(roots['A']/'shared-panel.json'),'lanes':{'A':{'status':'candidate','mature_saved_outputs':580,'prospective_images':128,'prospective_outputs':256,'result':bind(roots['A']/'result.json')},'B':{'status':'corrected_candidate_with_local_HOLD','states_attempted':45,'corrected_cells':267,'primary_admitted_states':35,'local_HOLD_states':10,'unexecuted_signed_cells':48,'pilot_signed_diagnostic_only_cells':12,'invalid_prior_science_excluded':True,'gate_lineage_calls':10,'result':bind(roots['B']/'final/broad-v1/result.json')},'C':{'status':'candidate','states':45,'draws':11520,'legal_repeat_draws':441,'structural_empty_union_proxy_states':7,'result':bind(roots['C']/'reduction.json')},'D':{'status':'candidate','states':16,'paired_contrasts':32,'sites':64,'immediate_winner_changes':1,'later_winner_changes':0,'role_scope':'y2 only','result':bind(roots['D']/'result.json')}},'cost':bind(out/'cost.json'),'job_closure':bind(job),'interpretation':bind(out/'results.md'),'artifact_map':bind(out/'ARTIFACTS.md'),'archival_limits':['A original overwritten prelaunch/shared-source JSON bytes unavailable; raw/cohort/runtime/current snapshots independently checked','D one original shard receipt overwritten and reconstructed; original bytes unavailable'],'source_qualification_scope':'B tied417044 final119-action position only plus already-passed forced-x1 and untied C qualification; not blanket native replay for every transformed state','prohibited_claims':['physical recovery or false positives from unmatched','untie-only causal effect','unique recurrence circuit','population prevalence from selected45-state panel','lead acceptance'],'cleanup':{'deleted_paths':[],'deleted_bytes':0}});files.append(result)
manifest=save('manifest.json',{'schema':'recurrence_source_distribution.manifest.v1','status':'candidate_bindings','files':[bind(p) for p in sorted(set(files))],'scope':'Selected integrated evidence and frozen code/research snapshots; nested raw evidence remains in each lane. Explicit archival gaps are not claimed restored.'})
terminal=save('integrated-terminal.json',{'status':'INTEGRATED_CANDIDATE','timestamp':now,'result':bind(result),'manifest':bind(manifest),'artifact_map':bind(out/'ARTIFACTS.md'),'cost':bind(out/'cost.json'),'jobs':bind(job),'all_required_child_work_settled':True,'live_owned_model_jobs':[],'scientific_acceptance':'pending_root','next_action':'direct delivery to root then stop','no_successor_launched':True})
print(json.dumps({'terminal':str(terminal),'manifest_files':len(set(files)),'total_model_calls':json.loads((out/'cost.json').read_text())['total_model_forward_invocations']}))
