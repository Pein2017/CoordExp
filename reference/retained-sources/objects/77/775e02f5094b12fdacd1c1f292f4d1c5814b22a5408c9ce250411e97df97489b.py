"""Freeze existing IoU50 matches; only remaining rows require visual review."""
import hashlib
import json
from pathlib import Path
from probes.training_set_completion.readback_selectors import one_to_one_matches

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
IMAGES = (25274,59571,99937,210457,219546,323322,351017,388795,417044,477415,528944)
def bind(p):
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rows = []
for step, phase, directory in [(64,'parent64','parent-step-00064'),(256,'final256','fourth-step-00256')]:
    for image_id in IMAGES:
        pp = BASE/'fourth-fit-eval-preparation-v1'/directory/f'image-{image_id:012d}'/'packet.json'
        packet = json.loads(pp.read_text())
        sp = Path(packet['source']['scored_selector']['path'])
        assert bind(sp)['sha256'] == packet['source']['scored_selector']['sha256']
        scored = next(r for r in json.loads(sp.read_text())['rows'] if r['image_id']==image_id)
        matches = scored['class_agnostic_iou_matching']['0.5']['matches']
        valid = [r for r in packet['full_raw_rows'] if r['status']=='parsed_valid']
        assert matches == one_to_one_matches(packet['target_catalog_references'], valid, .5)
        matched_ids = {r['prediction_id'] for r in matches}
        v4ids = {r['owner_id'] for r in packet['target_catalog_references']}
        extra = [r for r in packet['annotation_catalog_v5_references'] if r['owner_id'] not in v4ids]
        additional = one_to_one_matches(extra, [r for r in valid if r['prediction_id'] not in matched_ids], .5)
        all_matched = matched_ids | {r['prediction_id'] for r in additional}
        rows.append({'image_id':image_id,'phase':phase,'checkpoint_step':step,'source_packet':bind(pp),
                     'scored_selector':bind(sp),'fixed_v4_matches':matches,'v5_additional_matches':additional,
                     'unmatched_valid_prediction_ids':[r['prediction_id'] for r in valid if r['prediction_id'] not in all_matched],
                     'parser_invalid_prediction_ids':[r['prediction_id'] for r in packet['full_raw_rows'] if r['status']!='parsed_valid']})
result={'schema':'fourth_fit_match_inheritance.v2','threshold':.5,
        'authority':'User clarification 2026-09-15: view_image only unmatched; already IoU matched rows need no independent physical/extent rejudgment.',
        'existing_threshold_evidence':'readback_selectors.py review packet chooses class_agnostic_iou_matching[0.5].unmatched_predictions; 0.8 remains a diagnostic.',
        'policy':'Reuse unchanged class-agnostic cardinality-first one-to-one IoU50 v4 matches. Preserve those primary matches, then match remaining predictions to v5-only owners with same implementation/threshold. Only remaining valid predictions require visual review. Matched owner and extent accepted by rule; class certainty remains separate and must not be invented. Recompute physical repeats in generated order after unmatched judgments. Parser invalid rows remain invalid.',
        'producer':bind(Path(__file__)),'rows':rows}
p=OUT/'match-inheritance-v2.json';text=json.dumps(result,indent=2)+'\n'
if p.exists():
    assert p.read_text()==text
else:
    p.write_text(text)
print(json.dumps({'file':str(p),'counts':{phase:{'v4_matched':sum(len(r['fixed_v4_matches']) for r in rows if r['phase']==phase),'v5_additional_matched':sum(len(r['v5_additional_matches']) for r in rows if r['phase']==phase),'needs_visual':sum(len(r['unmatched_valid_prediction_ids']) for r in rows if r['phase']==phase)} for phase in ['parent64','final256']}},indent=2))
