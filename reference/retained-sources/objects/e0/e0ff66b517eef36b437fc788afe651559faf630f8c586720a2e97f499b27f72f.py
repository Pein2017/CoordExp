"""CPU-only lead replay of the one-bin control and unchanged native controls."""
import hashlib
import importlib.util
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / '2026-09-16-corner-loop-bridge-factorial'
UNIT = Path('/data/CoordExp/.worktrees/research-probes/research/experiments') / ROOT.name
read = lambda p: json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def bind(p):
    return {'path':str(p), 'sha256':sha(p)}

panel, result, terminal = (read(ROOT/n) for n in ['panel.json','result.json','terminal.json'])
for b in panel['sources'] + terminal['bindings'] + result['bindings']:
    path = Path(b['path'])
    if path.parent == UNIT:
        path = ROOT / 'candidate-records' / path.name
    assert sha(path) == b['sha256'], str(path)
prior = read(OLD/'panel.json')
for key in ['config','config_sha256','batch_group','cases','common_history_ids','common_history_sha256','owner_targets','owner_bank_sha256','parse_context','original_action_cap','supplied_tokens','free_budget','eos_id']:
    assert panel[key] == prior[key], key
raw = read(ROOT/'runtime/Y998/raw.json'); baseline = read(OLD/'runtime/C00/raw.json')
receipt = read(ROOT/'runtime/Y998/receipt.json'); control = read(OLD/'runtime/C00/receipt.json')
for key in ['loaded_identity','batch_shape','generate_settings','prompt_positions_sha256','positions']:
    assert receipt[key] == control[key], key
assert receipt['status'] == 'candidate_complete' and not receipt['mechanical_forks']
assert receipt['counts'] == {'model_forwards':3084,'processor_calls':3084,'vision_forwards':1}
assert panel['active_cells'] == ['Y998']
for i,(a,b,g) in enumerate(zip(raw['rows'],baseline['rows'],panel['batch_group'])):
    assert a['image_id'] == b['image_id'] == g['image_id']
    assert a['stop'] == b['stop'] == g['decode_stop_reason']
    if i != 1:
        assert a['token_ids'] == b['token_ids'] == g['generated_token_ids']
target = raw['rows'][1]['token_ids']; native = baseline['rows'][1]['token_ids']
assert len(target) == len(native) == 3084
delta = [{'offset':i,'old':a,'new':b} for i,(a,b) in enumerate(zip(native,target)) if a!=b]
assert delta == [{'offset':1231,'old':152669,'new':152668}]
assert target[:1224] == panel['common_history_ids'] == raw['common136_token_ids']
assert target[1224:1233] == panel['cells']['Y998']['tokens'] == raw['supplied137_token_ids']
assert target[1233:] == native[1233:] == raw['free_token_ids'] == baseline['free_token_ids']
spec = importlib.util.spec_from_file_location('accepted_consumer',OLD/'reduce.py')
consumer = importlib.util.module_from_spec(spec); spec.loader.exec_module(consumer)
tok = AutoTokenizer.from_pretrained(panel['config']['model']['base_model'],local_files_only=True)
assert tok.convert_tokens_to_ids('<|coord_998|>') == 152668
assert tok.convert_tokens_to_ids('<|coord_999|>') == 152669
got = consumer.consume(panel,raw,tok)
assert got == result['cell']
owners = set(got['joint_match_excluding_supplied137']['covered_owner_ids'])
assert len(owners) == 2 and not got['new_covered_owner_ids'] and not got['lost_history_owner_ids']
assert not got['history_assignment_changes']
controls = read(OLD/'result.json')['cells']
for name,c in controls.items():
    other = set(c['joint_match_excluding_supplied137']['covered_owner_ids'])
    assert result['comparisons'][name] == {'gained_owner_ids':sorted(owners-other),'lost_owner_ids':sorted(other-owners),'matched_count_delta':len(owners)-len(other)}
fixture = tok.decode(panel['cells']['C11']['history_ids'],skip_special_tokens=False)
_,valid,_ = consumer.parse(fixture,panel['parse_context'])
synthetic = [consumer.metrics._target(image_id=477415,owner_id='synthetic-credit-test',description='person',coord_bins=[0,0,999,999])]
included = consumer.match._ledger_image(synthetic,valid,threshold=.5)['matched_count']
excluded = consumer.match._ledger_image(synthetic,[v for v in valid if v['generated_order']!=136],threshold=.5)['matched_count']
assert (included,excluded) == (1,0)
seams = []
for offset in [1234,1237]:
    pair = [next(x for x in d['seam_logits'] if x['action_offset']==offset) for d in [control,receipt]]
    assert pair[0]['original_token'] == pair[1]['original_token']
    margins = [x['top5'][0]['logit']-x['top5'][1]['logit'] for x in pair]
    assert margins[0] != margins[1]
    seams.append({'action_offset':offset,'C00_margin':margins[0],'Y998_margin':margins[1]})
live = []
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try: cmd=p.read_bytes().split(b'\0')
    except (OSError,ProcessLookupError): continue
    if any(arg==str(ROOT/'producer.py').encode() or arg==str(ROOT/'run.sh').encode() for arg in cmd):
        live.append(int(p.parent.name))
assert not live
assert (ROOT/'run.exit').read_text().strip() == '0'
out = {'status':'lead-accepted','scope':'477415/R16 single y2:999->998 history intervention; exact C00 suffix, no rescue',
       'result':bind(ROOT/'result.json'),'panel':bind(ROOT/'panel.json'),'terminal':bind(ROOT/'terminal.json'),'verifier':bind(Path(__file__)),
       'candidate_records_archive':str(ROOT/'candidate-records'),'all_saved_token_metrics_recomputed_equal':True,
       'credit_exclusion_sensitivity':[included,excluded],'target_delta':delta,'exact_free_suffix_tokens':1851,
       'exact_companion_sequences':3,'seam_margins':seams,'matched':len(owners),'new_owners':0,
       'free_invalid_complete_rows':got['burden']['free_geometry_invalid_complete_rows'],'stop':got['stop'],
       'counts':receipt['counts'],'exit_code':0,'live_owned_jobs':live,
       'limits':['Single image/checkpoint/history and edit.','Numerical distance is not embedding distance.',
                 'Zero-height to inverted-height changes geometry semantics.','No embedding-origin identification or stable repair.']}
(ROOT/'lead-acceptance.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:out[k] for k in ['status','exact_free_suffix_tokens','target_delta','seam_margins','matched','new_owners','free_invalid_complete_rows','stop','live_owned_jobs']}))
