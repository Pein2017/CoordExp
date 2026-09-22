"""Root CPU acceptance of retained content diagnostic evidence; no model calls."""
from pathlib import Path
import hashlib
import json

from probes.row_feedback.content import classify_mechanical_outcome, verify_packet

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-row-feedback-pilot')
RUN = ROOT / 'content/run-v2'

def read(path):
    return json.loads(path.read_text())

packet = read(ROOT / 'content/packet-v2.json')
verify_packet(packet)
fit = read(ROOT / 'training/fit-F-v1/training-receipt.json')
receipt = read(RUN / 'receipt.json')
assert receipt['case_count'] == len(receipt['case_receipts']) == len(packet['cases']) == 3
assert receipt['technical_invalid_cases'] == []
assert receipt['adapter_path'] == str(ROOT / 'training/fit-F-v1/adapter')
rows = []
for case, reference in zip(packet['cases'], receipt['case_receipts']):
    path = RUN / reference['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['sha256']
    result = read(path)
    assert result['case_id'] == case['case_id']
    materialized = result['materialized_input']
    assert materialized['prepared_inputs_sha256'] == fit['materialization']['bank'][materialized['record_id']]
    arms = result['arms']
    assert arms['correct_f']['visible_token_ids'] == arms['exact_self_replay']['visible_token_ids']
    expected = {
        'boundary_index': case['recipient']['feedback_boundary']['box_end_occurrence'],
        'visible_boundary_index': case['recipient']['feedback_boundary']['visible_boundary_index'],
    }
    for name in ['exact_self_replay', 'wrong_owner']:
        assert arms[name]['override'] == {'count': 1, 'boundaries': [expected]}
        applied = [x for x in arms[name]['feedback_boundaries'] if x['override_applied']]
        assert len(applied) == 1
        source = 'correct_c' if name == 'exact_self_replay' else 'wrong_owner_w'
        assert applied[0]['source_sha256'] == result['sources'][source]['tensor']['sha256']
        assert applied[0]['native_source_sha256'] == result['sources']['correct_c']['tensor']['sha256']
    for generation in arms.values():
        assert generation['decode_contract']['max_visible_tokens'] == 3084
        assert len(generation['visible_token_ids']) == generation['visible_generated_tokens']
        assert generation['cap'] == (generation['finish_reason'] == 'length')
        assert generation['eos'] == (generation['finish_reason'] == 'eos')
    classification = classify_mechanical_outcome(
        correct_visible_ids=arms['correct_f']['visible_token_ids'],
        self_visible_ids=arms['exact_self_replay']['visible_token_ids'],
        wrong_visible_ids=arms['wrong_owner']['visible_token_ids'],
        self_override_count=1, wrong_override_count=1,
        correct_source_receipt=result['sources']['correct_c']['tensor'],
        wrong_source_receipt=result['sources']['wrong_owner_w']['tensor'],
    )
    assert classification == result['mechanical_classification']
    rows.append({
        'case_id': result['case_id'], 'prepared_input_matches_fit': True,
        'self_exact': True, 'override_site_and_source_hashes_match': True,
        'classification': classification,
        'tokens': {k: v['visible_generated_tokens'] for k, v in arms.items()},
        'case_receipt': {'path': str(path), 'sha256': reference['sha256']},
    })
value = {
    'schema': 'row_feedback.content_root_mechanical_checks.v1',
    'status': 'lead_accepted_mechanics_pending_bounded_physical_interpretation',
    'rows': rows, 'packet_verified': True,
    'counter_scope': 'Outer model/image/slot/visible totals sum nine generations only; three donor replays are separately counted but their forward totals were not persisted. Whole-run wall time includes donor work.',
}
output = ROOT / 'content/run-v2-root-mechanical-checks-v1.json'
if output.exists():
    assert read(output) == value
else:
    with output.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
print(json.dumps({'rows': rows, 'receipt_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, sort_keys=True))
