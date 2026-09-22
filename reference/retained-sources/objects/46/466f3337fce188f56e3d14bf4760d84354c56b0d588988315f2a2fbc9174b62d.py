import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
W=Path('/data/CoordExp/.worktrees/research-probes');U=W/'research/experiments'/R.name
P=R.parent/'2026-09-17-owner-recurrence-row-branch'
def bind(p):
 p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
check=json.loads((R/'admission-check.json').read_text())
reasons={67116:'Fruit-region recurrence is not yet attributed to one apple; no supported comparable covered/uncovered apple pair retained.',99184:'Original route stays in ambiguous border-crowd regions; saved distinct foreground people have substantially different position/extent. No confirmed comparable pair admitted.',181260:'Worker proposed O87 as uncovered at O89, but O87 precedes that boundary. N76 differs in direction/size and has unresolved physical identity relative to O85/O88. GT owner IDs do not repair this.',253212:'Selected O16 shelf recurrence: covered O9 is leftward, future O20 rightward with smaller extent and only two changed fields. Individual spine identity/coverage remains uncertain. Later metadata recurrence does not establish a better physical comparator.',309264:'Confirmed early recurrence exists, but common prefix contains only left-border bird extents. It lacks a covered bird with movement comparable to the far-right uncovered bird.',356238:'Saved border shelf extents have unresolved group/single-owner designation; no comparable physically supported covered/uncovered pair established.',386313:'Confirmed teal spine recurrence exists. Supported covered books are mostly left/upper-left; future shelf rows advance right. No reviewed missed same-class owner with comparable motion and defensible uncovered status established.',522489:'Cross-category oven/microwave recurrence, no retained same-class covered/uncovered appliance pair.'}
side={'status':'HOLD','scope':'Selected admission hypotheses only; no new physical truth census or annotation edits','reviewed_this_phase':[{'image_id':386313,'views':['original','predecessor selected full/detail'],'finding':reasons[386313]},{'image_id':181260,'views':['original','selected-full'],'finding':'Real vegetable material is visible; individual strip continuity and previous coverage are unresolved. Opposite horizontal direction and 77x73 versus 33x47 extent confound comparison.'},{'image_id':253212,'views':['original','selected-full'],'finding':'Real shelf books are visible; selected boxes do not establish distinct covered/uncovered individual spines with matched advancement.'}],'could_reverse_admission':'Yes: clarified physical identity could change individual eligibility, but does not remove the measured movement mismatches. No statement of impossibility across all fresh128.','color_legend':{'red':'native recurrence candidate','cyan':'covered candidate','yellow':'alternative candidate; not verified uncovered'},'cases':[{'image_id':i,'status':'HOLD','reason':v} for i,v in reasons.items()]}
(R/'physical-admission.json').write_text(json.dumps(side,indent=2)+'\n')
(R/'team-observations.md').write_text('''# Bounded team correction
One Luna high worker was assigned CPU metadata screening of the existing eight images. It returned no artifact and proposed an O87/O89 local triangle. Root rejected that interpretation: uncovered means absent from the exact common prefix, and O87 is already present before O89. Root stopped further search, reproduced literal rows and movements, and retained the correction here. No worker/model jobs remain. Future briefs should require the explicit prefix-coverage predicate before ranking spatial candidates; no extra audit or model-capability claim follows.
''')
model=json.loads((P/'source-manifest.json').read_text())['frozen_model_config']
result={'schema_version':1,'unit_id':R.name,'status':'candidate','scientific_status':'admission-HOLD; novelty-versus-displacement question unanswered','technical_status':'CPU source/literal-prefix checks passed; native intervention execution not attempted','denominators':{'source_cohort':128,'existing_screen':8,'newly_plotted_admission_cases':2,'admitted':0,'executed':0,'analyzable_intervention_cases':0},'missing_contrast':'A physically supported uncovered same-class row and an already-covered same-class row with comparable movement from a confirmed recurrent row at the exact common-prefix boundary.','case_findings':side['cases'],'frozen_model_config_inherited_not_loaded':model,'checks':{'source_bindings_verified':len(check['bindings']),'prefix_falsification':'O87 is strictly before O89; rejected as uncovered','saved_output_intervention_reduction':'not applicable: no new output','native_replay':'not rerun: admission failed before launch'},'cost':{'batch_executions':0,'model_forwards':0,'allocated_gpu_seconds':0},'owned_live_jobs':[],'evidence':[bind(R/'admission-check.json'),bind(R/'physical-admission.json'),bind(R/'check_admission.py')]+check['bindings'],'absent':['new suffixes','candidate row scores','sparse logits/readout captures','free-owner G/L: no admitted execution'],'next_action':'Stop for cross-study lead acceptance. Do not run a jitter, opposite-direction, cross-category or already-covered-as-uncovered surrogate. Reopening needs a specific supported contrast, not a larger visual census.'}
(R/'result.json').write_text(json.dumps(result,indent=2)+'\n')
(U/'results.md').write_text('''# Displacement control: candidate admission-HOLD

**No scientific panel admitted; no GPU execution.** This is an eligibility result, not a negative model result. The accepted predecessor is unchanged. The comparison still needs a confirmed recurrent owner plus covered/uncovered same-class alternatives that make comparable spatial moves from the same literal prefix.

The bounded metadata pass used the existing eight-image fresh128 recurrence screen. Root inspected selected full-image evidence for181260/253212 and reused386313 admission views. No fresh rollout bank, broad image review, labels or training were created.

| Selected counterexample | Why it does not isolate owner novelty |
|---|---|
|181260 at original row89|The proposed uncovered row87 is already in the common prefix. A nearby normalized row76 moves (+25,−43) bins versus covered row87 (−33,−46), with33×47 versus77×73 extent. Physical strip identity and coverage relative to earlier broad rows remain HOLD.|
|253212 at original row16|Covered row9 moves (−25.5,+9); future row20 moves (+11,0), with33×92 versus16×80 extent and four versus two coordinate fields changed. Opposite scan direction and individual-spine uncertainty remain.|
|309264 /386313 accepted recurrence cases|Bird prefix lacks a displaced covered counterpart to the far-right candidate. Book covered/future routes retain left-versus-right advancement and coverage ambiguity; the old book-to-clock arm is not reused as a same-class control.|

All row numbers above are zero-based complete literal rows. A future/normalized row is not automatically a new physical owner. Overlap, imperfect GT agreement and crowd context are not automatic rejection rules; the specific unresolved issue is identity/coverage and comparability. The remaining screen cases and their missing contrasts are recorded in the receipt. This pass does not establish that no eligible contrast exists anywhere in fresh128.

[Authoritative candidate receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-owner-recurrence-displacement-control/result.json) · [Artifact index](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-owner-recurrence-displacement-control/ARTIFACTS.md)

CPU checks verify26 source bindings, exact saved row extraction and the prefix-membership counterexample. Model calls, GPU seconds and intervention outputs are zero; free-owner gains/losses and likelihood captures are therefore absent, not zeros or failed outcomes. All delegated work is settled. Physical uncertainty could change admission, but current evidence cannot answer novelty versus scan advancement. Stop without a substitute arm or automatic review expansion. Cross-study lead owns acceptance and global frontier/coordination integration; Notion, memory and installed skills remain untouched.
''')
s=json.loads((U/'state.json').read_text());s.update(lifecycle='closed',evidence='unreviewed',disposition='candidate admission-HOLD; no model execution or mechanism conclusion',result=str((U/'results.md').relative_to(W)),state_source=str((U/'results.md').relative_to(W)),next_action='Cross-study lead acceptance of bounded admission-HOLD; no automatic expansion.')
(U/'state.json').write_text(json.dumps(s,indent=2)+'\n')
p=W/'research/experiments/catalog.jsonl';lines=p.read_text().splitlines()
for i,line in enumerate(lines):
 d=json.loads(line)
 if d.get('id')==R.name:d['result_records']=[str((U/'results.md').relative_to(W))];d['reading_entry']=d['result_records'][0];lines[i]=json.dumps(d)
p.write_text('\n'.join(lines)+'\n')
(R/'ARTIFACTS.md').write_text('''# Admission evidence index

- `result.json`: authoritative candidate status, source bindings, missing contrast and zero model cost.
- `admission-check.json`, `check_admission.py`: exact saved-row extraction, displacement measurements,26 source identities and prefix-coverage falsification. Reproduce with `python check_admission.py` from this directory; no model load.
- `physical-admission.json`: bounded physical judgments, unresolved coverage and case dispositions.
- `181260-selected-full.png`, `253212-selected-full.png`: full-image selected hypothesis overlays. Red=native candidate, cyan=covered candidate, yellow=alternative hypothesis, not GT/match status.
- `team-observations.md`: decision-relevant worker correction.
- `terminal.json`: terminal candidate and settled-job receipt.

No new suffix/token bank, score/logit tensors or free-owner ledgers exist because no panel passed admission. Original saved rows and all prior outputs remain at their bound source paths.
''')
terminal={'status':'candidate_admission_HOLD','scientific_status':'unanswered','producer_jobs':[],'workers_settled':True,'batch_executions':0,'model_forwards':0,'bindings':[bind(R/'result.json'),bind(U/'results.md'),bind(U/'state.json'),bind(R/'ARTIFACTS.md')]}
(R/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
print(json.dumps({'status':terminal['status'],'result':bind(R/'result.json'),'terminal':bind(R/'terminal.json')}))
