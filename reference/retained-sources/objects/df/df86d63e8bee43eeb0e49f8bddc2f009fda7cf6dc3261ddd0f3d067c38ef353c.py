"""CPU-only cross-study acceptance checks; no model execution or artifact repair."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def bind(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)

panel = read(ROOT / 'panel.json')
candidate = read(ROOT / 'terminal.json')
verification = read(ROOT / 'final-verification.json')
bindings = panel['sources'] + verification['bindings']
bindings += [candidate[k] for k in ['result', 'state', 'results_record', 'artifact_index']]
unique = {}
for binding in bindings:
    if binding in [candidate[k] for k in ['state', 'results_record']]:
        # Closure updates live routing; verify the preserved candidate bytes.
        snapshot = ROOT / 'candidate-records' / Path(binding['path']).name
        actual = bind(snapshot) if snapshot.exists() else bind(binding['path'])
        assert actual['sha256'] == binding['sha256'] and actual['size_bytes'] == binding['size_bytes'], binding['path']
    else:
        assert bind(binding['path']) == binding, binding['path']
    unique[binding['path']] = binding
assert read(ROOT / 'lead-known-replay.json') == read(ROOT / 'known-result.json')
reduced = read(ROOT / 'lead-known-replay.json')
checks = []
for case in panel['cases']:
    iid = str(case['image_id'])
    target = case['target_position']
    start, end = case['start_offset'], case['end_offset']
    saved = read(case['saved_raw']['path'])
    native = read(ROOT / 'runtime' / iid / 'native' / 'raw.json')
    native_receipt = read(ROOT / 'runtime' / iid / 'native' / 'receipt.json')
    for j, row in enumerate(native['rows']):
        for key in ['token_ids', 'text', 'stop']:
            assert row[key] == saved['rows'][j][key], (iid, j, key)
    for arm, supplied in case['arms'].items():
        folder = ROOT / 'runtime' / iid / arm
        raw = read(folder / 'raw.json')
        receipt = read(folder / 'receipt.json')
        for key in ['loaded_identity', 'generate_settings', 'input_identity', 'prefill_mrope_sha256', 'batch_shape']:
            assert receipt[key] == native_receipt[key], (iid, arm, key)
        assert receipt['batch_shape']['size'] == 4
        assert receipt['generate_settings']['repetition_penalty'] == 1.0
        assert receipt['generate_settings']['max_new_tokens'] == 3084
        assert not receipt['generate_settings']['do_sample']
        for j, row in enumerate(raw['rows']):
            if j != target:
                for key in ['token_ids', 'text', 'stop']:
                    assert row[key] == native['rows'][j][key]
        row = raw['rows'][target]
        ids = row['token_ids']
        assert ids[:start] == native['rows'][target]['token_ids'][:start]
        assert ids[start:end] == supplied['token_ids']
        assert len(ids) <= 3084
        assert (row['stop'] == 'im_end' and ids[-1] == 151645) or (row['stop'] == 'length' and len(ids) == 3084)
        v = reduced['images'][iid][arm]
        assert not set(v['eligible_free_owner_ids']) & set(case['supplied_known_owner_union'])
        checks.append(dict(image_id=iid, arm=arm, tokens=len(ids), stop=row['stop'], free_known_G_L=v['known_free_vs_native']))
    if iid == '309264':
        same = read(ROOT / 'runtime' / iid / 'same' / 'raw.json')['rows'][target]
        assert same['token_ids'][end:] == native['rows'][target]['token_ids'][end:]
        assert len(same['token_ids'][end:]) == 3030
    if iid == '386313':
        distinct = read(ROOT / 'runtime' / iid / 'distinct' / 'raw.json')['rows'][target]
        assert distinct['token_ids'][end:] == [151645]
exits = {p.name:int(p.read_text().strip()) for p in (ROOT/'logs').glob('*.exit')}
assert len(exits) == 6 and set(exits.values()) == {0}
processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,args'], text=True).splitlines()
model_jobs = [s for s in processes if str(ROOT / 'producer') in s]
assert not model_jobs, model_jobs
report = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(), source_bindings_verified=len(panel['sources']), unique_hash_bindings_verified=len(unique), independent_consumer_json_exact=True, exact_native_sequences_verified=8, execution_checks=checks, exit_codes=exits, owned_live_model_jobs=model_jobs, bindings=list(unique.values()), consumer=bind(ROOT/'reduce.py'), replay=bind(ROOT/'lead-known-replay.json'), candidate_result=bind(ROOT/'result.json'), scope='CPU saved-evidence verification; physical review separately bounded; no fresh model runs')
(ROOT/'lead-verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['bindings','execution_checks']},indent=2))
