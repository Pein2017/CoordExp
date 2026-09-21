import json,hashlib,difflib,copy
from pathlib import Path
B=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration');OLD=B/'2026-09-16-corner-loop-bridge-factorial';R=B/'2026-09-16-endpoint-loop-image-history';p=json.loads((R/'panel.json').read_text());original=(OLD/'producer.py').read_text();s=original.replace(str(OLD),str(R)).replace('Four frozen recent-row interventions through the original heterogeneous native batch.','Fixed image x literal-history contrast; full target prefix supplied at native processor.')
s=s.replace("receipt={'schema'", "panel['cases']=panel['original_cases'] if cell=='I00' else panel['donor_cases']\n if cell!='I00':\n  panel['batch_group'][1]={**panel['batch_group'][1], 'image_id':panel['donor']['image_id'], 'executed_media_sha256':panel['donor']['generation']['executed_media_sha256']}\n receipt={'schema'")
s=s.replace("if cell!='C00':\n   control=json.loads((ROOT/'runtime/C00/receipt.json').read_text());assert control['status']=='candidate_complete' and control['full_original_batch_identity']\n   receipt['control_gate']=bind(ROOT/'runtime/C00/receipt.json')", "if cell!='I00':\n   control=json.loads((ROOT/'runtime/I00/receipt.json').read_text());assert control['status']=='candidate_complete' and control['full_original_batch_identity']\n   receipt['control_gate']=bind(ROOT/'runtime/I00/receipt.json')")
s=s.replace("must_match=offset<1224 or cell=='C00'", "must_match=cell=='I00'")
s=s.replace("if cell!='C00' and 1224<=offset<1233:","if 0<=offset<1233:")
s=s.replace("if b!=1 or cell=='C00':", "if b!=1 or cell=='I00':").replace("full_original_batch_identity=cell=='C00'", "full_original_batch_identity=cell=='I00'")
s=s.replace("choices=['C00','C10','C01','C11']", "choices=['I00','D00','D10']")
s=s.replace("'common136_token_ids':panel['common_history_ids']", "'common136_token_ids':panel['common_history_ids'], 'target_image_id':group[1]['image_id'], 'history_supply_interval':[0,1233]")
s=s.replace("common136_identity=True", "common136_identity=True,common136_natural=cell=='I00',history_mask={'batch_position':1,'start':0,'end_exclusive':1233,'free_logits_unrestricted':True}")
assert "if cell!='C00'" not in s
(R/'producer.py').write_text(s);(R/'producer.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(OLD/'producer.py'),tofile=str(R/'producer.py'))))
p['cells']={'I00':copy.deepcopy(p['cells']['C00']),'D00':copy.deepcopy(p['cells']['C00']),'D10':copy.deepcopy(p['cells']['C10'])};p['producer_sha256']=hashlib.sha256((R/'producer.py').read_bytes()).hexdigest();p['processor_mask']={'target_sample':1,'supplied_action_start':0,'supplied_action_end_exclusive':1233,'all_companions_and_free_steps_unmodified':True};(R/'panel.json').write_text(json.dumps(p,indent=2)+'\n');(R/'prepare.py').write_text(Path(__file__).read_text())
U=Path('research/experiments')/R.name;U.mkdir();(U/'unit.md').write_text('''# Endpoint loop current visual dependence

From fixed R16 and accepted C00/C10 completed-row137 histories, does substituting ONE qualified natural image alter free endpoint-loop maintenance and image-specific known-owner coverage over the same1851 free-token horizon?

## Frozen donor and contrast

Selected image7116 deterministically as the smallest eligible ID among58 qualified of256 Source256 train records. Exclude477415 and all3 original companions. Filters: saved empty-prefix R16 EOS, zero parser/geometry debt and strict repeats,≥2 known matches; same executed1248x832 geometry/grid[1,52,78], exact1362 prompt IDs under unchanged processor. CPU real processor preparation verified geometry/media/prompt; detailed eligibility and baseline4/6 matches in qualification.json/panel.json. No class/inventory preference, no intervention outcomes or new dataset used for selection. Donor classes boat/person; its fixed6-owner bank remains unchanged.

Cross original477415 vs7116 with literal C00 chair[0,999,999,999] vsC10 person[0,999,999,999]. Reuse bound original C00/C10 free continuations. TWO donor continuations only. Original R16 adapter, original paired embeddings/readout, FP32/SDPA/RP1 native greedy, heterogeneous bs4 companions/order/left padding/prompt/attention/MRoPE, common1224 plus9 row tokens, totalcap3084 remain fixed. All1233 target tokens are supplied at the native per-step logits processor, not prefilling a different path. All later logits/opener/EOS and all companion logits remain native. Supplied donor history is deliberately off-policy and receives zero credit.

## Admission

ONE original-image C00 full-prefix-supply identity continuation I00 must exactly reproduce accepted full target and allcompanions, runtime/prompt position identities. Only after it passes run donor D00/D10 concurrently. C10 reuse relies on identical literal prefix/model/processor equivalence: the new processor supplies the exact same common prefix and9-token C10 row already consumed by original C10; free steps are untouched. If runtime equivalence fails, block rather than launch a qualification matrix.

## Primary scoring and limits

All primary matches/FN are FREE-SUFFIX ONLY using each image’s own unchanged frozen known-owner bank and existing native parser/class-agnostic one-to-one matcher. No supplied history/row137 matching, even accidental donor matches. Recompute free-only original controls from saved outputs. Both-bank cross-scoring and class correctness remain explicitly diagnostic, never replace primary matching or establish physical truth. Report identities, FN,first token divergence/early and full behavior, endpoint occupancy, old/alternate cycles, valid/invalid literal repeats, malformed/strict repeats/UNKNOWN,EOS/cap. No images reviewed or labels added.

C00 release supports a current visual maintenance contribution. C00 trapped with donor-responsive C10 prioritizes history-dependent visual access. Both similarly changed is not selective trap evidence; both unchanged is this donor’s null only. Natural-image substitution still mismatches supplied history; shorter text or an early changed token is not repair or initial-entry evidence.

## Budget and stop

Two scientific cells plus at most one identity,each≤3084 native bs4 forwards,intended9252 total,hard10000;≤7200 allocated GPU-seconds and≤7200 elapsed seconds. No training/loss/labels/checkpoint/architecture/embedding/KV change,donor/layer/coordinate/seed sweep,agents or visual expansion. Bounded concrete mechanical repair only within same ceiling,otherwise technical-invalid. Stop after saved-output reduction and candidate closeout; no automatic successor.

Immutable bindings,exact tokens/banks/media/source diffs: '''+str(R/'panel.json')+'\n')
state=dict(schema_version=1,unit_id=R.name,lifecycle='ready',evidence='none',disposition='frozen_single_donor_two_histories',state_as_of='2026-09-16',protocol=str(U/'unit.md'),result=None,state_source=str(U/'unit.md'),boundary='One qualified donor x C00/C10; one mechanical identity; primary free-only own-bank matching',not_authorized=['additional donors or cells','training','KV/embedding changes','agents','automatic successor'],next_action='Identity admission then two donor suffixes,CPU reduce and stop');(U/'state.json').write_text(json.dumps(state,indent=2)+'\n')
cat=Path('research/experiments/catalog.jsonl');entry=dict(id=R.name,title='Endpoint maintenance: one natural image crossed with two histories',kind='experiment',topics=['history-repetition-stopping'],record_root=str(U),protocols=[str(U/'unit.md')],result_records=[],reading_entry=str(U/'unit.md'),tracking='current',state=str(U/'state.json'));cat.write_text(cat.read_text()+json.dumps(entry)+'\n');idx=Path('research/index.md');idx.write_text(idx.read_text()+f'\nVisual-maintenance [unit](experiments/{R.name}/unit.md), [state](experiments/{R.name}/state.json): frozen donor7116 x C00/C10; identity admission then two suffixes only.\n')
print('Frozen donor7116; only full-history processor mask and target image selection change')
