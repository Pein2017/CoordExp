import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;U=Path('research/experiments')/R.name
read=lambda p:json.loads(p.read_text())
def bind(p):
 p=p.resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
d=read(R/'result.json');e=read(R/'edit-receipt.json');c=d['cell'];assert e['free_suffix_exact']=={'C00':True,'Y998':True}
assert c['joint_match_excluding_supplied137']['matched_count']==2 and not c['new_covered_owner_ids'] and not c['lost_history_owner_ids'] and not c['history_assignment_changes']
assert len(c['supplied137_rows_excluded'])==1 and c['supplied137_rows_excluded'][0]['status']=='parsed_valid'
text='''# Final paired coordinate-substitution result

**Candidate; execution-owner checks passed, independent lead acceptance pending.** One Y1_998 cell completed. This closes the authorized local substitution series; no successor is running or authorized by this record.

## Paired result and accumulated causal evidence

- **Both tested one-bin edits are insufficient:** Y1_998 chair [0,998,999,999] produces exactly the same1851 free tokens as accepted Y998 and C00, with coverage2/27 and no new free owners. The only full-target difference from C00 is supplied action offset1229,152669→152668; Y998 instead changes1231. Neither recovers any of the16 owners recovered by the successful factorial controls.
- **No useful excursion:** immediate next row is chair [0,999,999,999]; all205 complete free rows repeat it with invalid geometry. There is no nonzero-x1 row or all999 alternate cycle. One malformed tail, no EOS, total3084-token cap. Free-valid, strict valid-repeat and UNKNOWN counts are0 because there are no valid free predictions, not because enumeration improved.
- **Credit and retention:** common-history owners remain matched, G/L=0/0, no assignment swaps; free-only owner IDs are empty. The actual native parser accepts supplied137 as a positive-height [0,830,1247,831] pixel box, but it receives ZERO owner/error credit and does not imply a physical object. The unchanged supplied-credit falsification gives1 match when admitted and0 when excluded.
- **The edit was consumed:** next-opener margin C00/Y998/Y1_998 is4.397127/4.495892/4.582087; next chair-category margin0.184498/0.140511/0.215508; next x1=0 margin1.216604/1.207817/1.226772. Logits change while all greedy suffix tokens remain identical. These margins do not measure embedding distance or a full-distribution effect.
- **Transient causal recovery remains established:** accepted C10 category-only, C01 full-extent and C11 joint recover the same16 owners, coverage2→18/27, then relapse from free row46 and cap. Category-only success disproves universal necessity of validity repair. Y1_998 failure shows this particular validity-crossing thin box is not sufficient; it does not show validity changes never help.
- **Strongest supported explanation:** escape depends on the specific recent-history edit, while the original repeated trajectory tolerates both tested999→998 edits at the greedy level. Slot, direction, extent, category/history representation and validity effects remain unresolved; negative controls do not establish useful slot-dependent recovery. Numerical one-bin distance is not small embedding distance. Neither outcome identifies embedding defects, KV malfunction or a universal mechanism.
- **Series STOP:** further work would need a qualitatively different, separately authorized mechanism question—such as why a successful bridge restores coverage only transiently and later relapses—rather than more local coordinate substitutions. No additional discriminator or intervention is executed here.

## Matched saved-output metrics

| Cell | Coverage /27 | Free-only matched | G/L vs common136 | Invalid complete free | Malformed | Strict valid repeats | UNKNOWN free | Free tokens / stop |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C00 reused |2|0|0/0|205|1|0|0|1851 / cap|
| Y998 reused |2|0|0/0|205|1|0|0|1851 / cap|
| Y1_998 new |2|0|0/0|205|1|0|0|1851 / cap|
| C01 reused |18|16|16/0|160|1|15|29|1851 / cap|

Strict-repeat counts with supplied-history references and free-only references are both0 for Y1_998. All205 invalid complete rows are nevertheless literal repetitions, retained in raw accounting. The malformed fragment is `<|object_ref_start|>chair<|object_ref_end|><|box_start|><|coord_0|><|coord_999|>`. No free/history owner is lost or reassigned; the16 comparison identities and complete ledgers are in result.json. UNKNOWN stays UNKNOWN.

## Technical acceptance and reproducibility

The producer derivative changes only root, CLI cell and description; explicit diff is retained. All predecessor panel/config/source/control hashes were verified without modifying predecessor bytes. Existing accepted pure consume is imported normally with no global rebinding. Raw saved-token reduction is a separate CPU invocation. Original first1224 action tokens, all3 companion sequences/stops, loaded model identity, original embeddings, actual heterogeneous bs4/order/left padding, FP32/SDPA/RP1, native MRoPE/prompt positions and generation/cache settings match C00. Supplied137 mask is sample1/action[1224,1233), with the only literal delta at1229; subsequent logits/EOS are unrestricted. Pre-edit argmax checks passed. No original control was rerun on GPU.

The panel binds immutable output sources, not mutable research records. All natural prefix, row137 token, model/media/token identity and parser/matcher evidence remains in panel/runtime/result/edit receipts. Bound predecessor receipt sources are preserved; their research candidate records were already archived by lead and are not edited.

'''
text+=f"Cost: {d['technical']['counts']['model_forwards']} native bs4 forwards (12336 batch slots including finished-companion padding),1 vision invocation, {d['technical']['elapsed_gpu_seconds']:.6f} allocated GPU-seconds ({d['technical']['elapsed_gpu_seconds']/3600:.6f} GPU-hours), within5000 forwards and intended1800 seconds. One successful model attempt, exit0; no live owned jobs. Runtime wall time includes loading/checks, not kernel-only time. A host event-wait initially lacked os.pidfd_open and was replaced with the Linux pidfd syscall; no producer/model rerun or execution change resulted.\n\n"
for name in ['panel.json','producer.py','producer.diff','reduce.py','verify.py','result.json','edit-receipt.json','runtime/Y1_998/raw.json','runtime/Y1_998/receipt.json','run.sh','launch.json','terminal.json']:
 text+=f'- [{name}]({R/name})\n'
text+='\nPredecessors: [accepted factorial](../2026-09-16-corner-loop-bridge-factorial/results.md), [accepted y2 control](../2026-09-16-corner-loop-one-bin-control/results.md), [accepted execution/slot phase](../2026-09-16-corner-loop-mechanism/results.md). These link onward to older coordinate-history/KV controls; none was repeated.\n'
(U/'results.md').write_text(text)
s=read(U/'state.json');s.update(lifecycle='closed',evidence='unreviewed',disposition='candidate_both_tested_one_bin_edits_insufficient_exact_C00_suffix',result=str(U/'results.md'),state_source=str(U/'results.md'),next_action='STOP. Final local coordinate-substitution series complete; lead acceptance pending. Further work requires a qualitatively different separately authorized question.');(U/'state.json').write_text(json.dumps(s,indent=2)+'\n')
p=Path('research/experiments/catalog.jsonl');rows=p.read_text().splitlines()
for i,line in enumerate(rows):
 x=json.loads(line)
 if x['id']==R.name:x.update(result_records=[str(U/'results.md')],reading_entry=str(U/'results.md'));rows[i]=json.dumps(x)
p.write_text('\n'.join(rows)+'\n')
p=Path('research/index.md');s=p.read_text();old='Final local substitution control: [Y1_998 unit](experiments/'+R.name+'/unit.md), [state](experiments/'+R.name+'/state.json). One frozen cell; stop after reduction, no automatic successor.';new='Final local substitution [Y1_998 candidate](experiments/'+R.name+'/results.md) is complete: exact C00/Y998 free suffix, no recovery. Its [state](experiments/'+R.name+'/state.json) is closed/unreviewed pending lead acceptance. Local substitution series stopped; further work requires a qualitatively different authorized question.';assert old in s;p.write_text(s.replace(old,new))
files=[R/f for f in ['panel.json','producer.py','producer.diff','prepare.py','reduce.py','verify.py','closeout.py','result.json','edit-receipt.json','runtime/Y1_998/raw.json','runtime/Y1_998/receipt.json','run.sh','run.exit','launch.json']]+[U/f for f in ['unit.md','state.json','results.md']]
t=dict(schema='corner_loop.y1_one_bin_control.terminal.v1',status='candidate_for_lead_acceptance',verdict='Both tested one-bin edits insufficient at this boundary; exact C00/Y998 free suffix; local series stopped.',exit_code=0,live_owned_jobs=[],next_phase_executed=False,technical=d['technical'],edit=bind(R/'edit-receipt.json'),bindings=[bind(f) for f in files]);(R/'terminal.json').write_text(json.dumps(t,indent=2)+'\n');print('Candidate closed; predecessor bytes verified and untouched.')
