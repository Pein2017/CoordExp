"""Lead CPU verification of frozen raw suffixes and credit exclusion."""
import hashlib
import importlib.util
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def bind(p):
    return {'path': str(p), 'sha256': sha(p)}

panel, result, terminal = (read(ROOT / n) for n in ('panel.json','result.json','terminal.json'))
for b in panel['sources'] + terminal['bindings']:
    assert sha(b['path']) == b['sha256'], b['path']
spec = importlib.util.spec_from_file_location('frozen_bridge_consumer', ROOT/'reduce.py')
consumer = importlib.util.module_from_spec(spec); spec.loader.exec_module(consumer)
tok = AutoTokenizer.from_pretrained(panel['config']['model']['base_model'], local_files_only=True)
assert sha(ROOT/'reduce.py') == result['consumer']['sha256']
common = panel['common_history_ids']; group = panel['batch_group']
assert len(common) == 1224 and panel['free_budget'] == 1851
assert common == group[1]['generated_token_ids'][:1224]
native = panel['cells']['C00']['tokens']
assert len(native) == 9
expected_edits = {'C00': [], 'C10': [1], 'C01': [5], 'C11': [1,5]}
summary, live, identity = {}, [], []
for name, cell in panel['cells'].items():
    tokens = cell['tokens']; assert len(tokens) == 9
    assert [j for j,(a,b) in enumerate(zip(native,tokens)) if a != b] == expected_edits[name]
    assert cell['history_ids'] == common + tokens
    raw = read(ROOT/f'runtime/{name}/raw.json'); receipt = read(ROOT/f'runtime/{name}/receipt.json')
    assert receipt['status'] == 'candidate_complete' and not receipt['mechanical_forks']
    assert receipt['raw']['sha256'] == sha(ROOT/f'runtime/{name}/raw.json')
    for j,row in enumerate(raw['rows']):
        assert row['image_id'] == group[j]['image_id']
        if j != 1 or name == 'C00':
            assert row['token_ids'] == group[j]['generated_token_ids']
            assert row['stop'] == group[j]['decode_stop_reason']
            identity.append((name,j))
    got = consumer.consume(panel,raw,tok)
    reference = result['cells'][name]
    for key,val in got.items():
        assert val == reference[key], (name,key)
    history = set(got['history_match']['covered_owner_ids'])
    owners = set(got['joint_match_excluding_supplied137']['covered_owner_ids'])
    assert len(history) == 2 and not got['lost_history_owner_ids'] and not got['history_assignment_changes']
    assert all(m['prediction_id'] != 'SUPPLIED137' for m in got['joint_match_excluding_supplied137']['matches'])
    assert set(got['new_covered_owner_ids']) == owners-history
    assert reference['vs_C00'] == {'gained': sorted(owners-history), 'lost': [], 'FN': 27-len(owners)}
    b = got['burden']; assert b['free_complete_rows'] == 205 and got['free_token_count'] == 1851 and got['stop'] == 'length'
    assert b['free_valid_rows'] + b['free_geometry_invalid_complete_rows'] == 205
    summary[name] = {'matched':len(owners), 'gained_ids':sorted(owners-history), 'free_valid':b['free_valid_rows'], 'free_invalid':b['free_geometry_invalid_complete_rows'], 'free_repeats':b['free_strict_repeat_rows_with_supplied_history'], 'stop':got['stop'], 'recurrence':got['later_recurrence']}
    proc = Path('/proc')/str(receipt['pid'])/'cmdline'
    if proc.exists() and b'probes.training_set_completion.corner_loop_bridge_factorial' in proc.read_bytes():
        live.append(receipt['pid'])
assert not live
assert [summary[c]['matched'] for c in expected_edits] == [2,18,18,18]
assert summary['C10']['gained_ids'] == summary['C01']['gained_ids'] == summary['C11']['gained_ids']
fixture = tok.decode(panel['cells']['C11']['history_ids'], skip_special_tokens=False)
_,valid,_ = consumer.parse(fixture,panel['parse_context'])
synthetic = [consumer.metrics._target(image_id=477415,owner_id='synthetic-credit-test',description='person',coord_bins=[0,0,999,999])]
assert consumer.match._ledger_image(synthetic,valid,threshold=.5)['matched_count'] == 1
assert consumer.match._ledger_image(synthetic,[x for x in valid if x['generated_order'] != 136],threshold=.5)['matched_count'] == 0
exits = {p.name:int(p.read_text()) for p in ROOT.glob('*.exit')}
assert len(exits) == 5 and set(exits.values()) == {0}
out = {'status':'lead-accepted','scope':'fixed477415/R16 recent-row factorial; transient free-owner recovery with endpoint relapse','result':bind(ROOT/'result.json'),'panel':bind(ROOT/'panel.json'),'terminal':bind(ROOT/'terminal.json'),'consumer':bind(ROOT/'reduce.py'),'verifier':bind(Path(__file__)),'all_saved_row_reductions_equal':True,'credit_exclusion_falsification_passed':True,'original_batch_sequences_reverified':len(identity),'summary':summary,'exit_codes':exits,'live_owned_jobs':live,'next_phase_executed':False,'limits':['Single exposed history/image/checkpoint.','Supplied row never receives matching credit.','Geometry intervention changes area, validity and extent.','All recovered branches relapse and cap; no stable repair or embedding-origin claim.']}
(ROOT/'lead-acceptance.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':out['status'],'recomputed_cells':len(summary),'exact_original_sequences':len(identity),'coverage':{k:v['matched'] for k,v in summary.items()},'credit_exclusion':True,'live_owned_jobs':live}))
