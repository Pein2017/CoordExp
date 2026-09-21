"""CPU-only independent verification of saved phase1 evidence; no model loads."""
import hashlib
import json
import math
import subprocess
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def binding(p):
    return {'path': str(p), 'sha256': sha(p)}

torch.set_num_threads(2)
panel, result = read(ROOT / 'panel.json'), read(ROOT / 'result.json')
assert result['status'] == 'candidate_complete'
bindings = panel['sources'] + result['cells'] + result['logit_files']
for key in ('panel', 'producer', 'reducer', 'launch'):
    bindings.append(result[key])
for item in bindings:
    assert sha(item['path']) == item['sha256'], item['path']
oldroot = ROOT.parent / '2026-09-16-source256-output-ranking-repair'
bankpath = ROOT.parent / '2026-09-16-source256-fixed-prefix-completion/preparation/source256-admitted-v1/preparation.json'
bank = {r['image_id']: r for r in read(bankpath)['bank']['records']}
for case in panel['cases']:
    src = read(oldroot / f"visuals/human-review-pr16-v1/{case['image_id']}-P-source.json")['saved_generation']
    assert case['P_action_ids'] == src['generated_token_ids']
    assert case['prompt_ids'] == src['prompt_token_ids']
    assert sha(case['image_identity']['image_path']) == case['image_identity']['image_content_sha256']
    for b in case['boundaries']:
        tokens = b['observed']['tokens']
        assert case['P_action_ids'][b['offset']:b['offset']+len(tokens)] == tokens
        alt = b['alternative']
        owner = next(o for o in bank[case['image_id']]['owners'] if o['owner_id'] == alt['owner_id'])
        assert owner['coord_bins'] == alt['box'] and owner['description'] == alt['category']

slots = flips = batch_flips = 0
max_error = max_ratio = 0.0
exits = []
cells = []
for bound in result['cells']:
    path = Path(bound['path']); cell = read(path); cells.append(cell)
    assert cell['status'] == 'candidate_complete'
    assert cell['panel_sha256'] == sha(ROOT / 'panel.json')
    assert cell['producer_sha256'] == sha(ROOT / 'producer.py')
    for b in cell['boundaries']:
        for route in b['routes']:
            data = torch.load(path.parent / f"row{b['row_1based']}-{route['role']}-logits.pt", map_location='cpu', weights_only=True)
            assert route['full_noop_max_abs'] == 0
            for s in route['slots']:
                name = s['slot']; f, c, four = (data[name + '.' + suffix] for suffix in ('full', 'cache', 'bs4'))
                assert f.ndim == c.ndim == four.ndim == 1
                assert all(torch.isfinite(v).all() for v in (f, c, four))
                top = f.topk(2); margin = float(top.values[0] - top.values[1]); err = float((f-c).abs().max())
                ratio = 2 * err / margin
                assert margin > 0 and ratio < 1
                assert margin == s['full']['top_margin'] and err == s['max_abs_cache_full']
                assert math.isclose(ratio, s['twice_error_over_margin'], rel_tol=1e-12)
                for v, name2 in ((f, 'full'), (c, 'cached'), (four, 'bs4_full')):
                    assert int(v.argmax()) == s[name2]['argmax']
                for candidate in s['full']['candidates']:
                    idx = candidate['id']; assert float(f[idx]) == candidate['logit']
                    assert int((f > f[idx]).sum()) + 1 == candidate['rank']
                slots += 1; flips += int(f.argmax() != c.argmax()); batch_flips += int(f.argmax() != four.argmax())
                max_error = max(max_error, err); max_ratio = max(max_ratio, ratio)
                if cell['image_id'] == 477415 and b['row_1based'] == 138 and route['role'] == 'observed':
                    assert int(f.argmax()) == route['tokens'][s['row_token_offset']]
                    exits.append({'checkpoint': cell['checkpoint'], 'slot': name, 'top': s['full']['argmax_text'], 'margin': margin})
            del data
assert (len(cells), slots, flips, batch_flips) == (9, 363, 0, 0)
assert max_error == result['parity']['max_slot_cache_full_abs']
assert max_ratio == result['parity']['max_twice_error_over_margin']
assert len({c['coordinate_input_sha256'] for c in cells}) == 1
assert len({json.dumps(c['readout_parameter_hashes'], sort_keys=True) for c in cells}) == 1
exit_codes = {p.name: int(p.read_text()) for p in ROOT.glob('*.exit')}
assert len(exit_codes) == 9 and set(exit_codes.values()) == {0}
live = []
for cell in cells:
    p = Path('/proc') / str(cell['pid']) / 'cmdline'
    if p.exists() and 'probes.training_set_completion.corner_loop_phase1' in p.read_bytes().decode(errors='replace'):
        live.append(cell['pid'])
assert not live
old = read(oldroot / 'runtime/main-v1/result.json')
coverage = {label: next(r for r in old['scores'][label]['splits']['train']['per_image'] if r['image_id'] == 477415)['primary_class_agnostic_iou50']['matched_count'] for label in panel['checkpoints']}
assert coverage == {'Bnormalized64': 18, 'P16': 17, 'R16': 2}
receipt = {'status': 'lead-accepted', 'scope': 'phase1 fixed-history slot/parity evidence only', 'method': 'independent CPU recomputation from all57 saved full/cache/bs4 logit files; source and candidate binding checks; producer inspection', 'result': binding(ROOT/'result.json'), 'panel': binding(ROOT/'panel.json'), 'verifier': binding(Path(__file__)), 'bank': binding(bankpath), 'verified_bindings': len(bindings), 'cells': len(cells), 'slots': slots, 'cache_full_argmax_flips': flips, 'duplicate_bs4_argmax_flips': batch_flips, 'max_logit_error': max_error, 'max_twice_error_over_margin': max_ratio, 'exit_slot_preferences': exits, 'natural477415_coverage': coverage, 'exit_codes': exit_codes, 'live_owned_jobs': live, 'phase2_executed': False, 'limits': ['All tested histories are supplied P histories; B/R may be off policy.', 'Only selected admission/category/coordinate positions tested; no full-row or free-suffix rescue claim.', 'Exit history already contains P row137 person/full-canvas bridge.', 'Duplicate bs4 does not reproduce original heterogeneous batch/padding.', 'Embedding identity does not exclude an inherited coordinate prior.']}
(ROOT/'lead-acceptance.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','verified_bindings','cells','slots','cache_full_argmax_flips','duplicate_bs4_argmax_flips','max_logit_error','max_twice_error_over_margin','live_owned_jobs','phase2_executed')}))
