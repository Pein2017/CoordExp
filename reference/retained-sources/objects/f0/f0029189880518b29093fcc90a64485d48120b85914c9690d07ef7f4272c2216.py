"""Frozen selective rule; inference artifacts and reference labels stay separate."""
import json
from pathlib import Path

SPEC=json.loads((Path(__file__).resolve().parent/'selected-candidate-v1.json').read_text())

def canonical(category):
    value=' '.join(category.strip().lower().split())
    return SPEC['aliases'].get(value,value)

def judge(candidate, detector, observations):
    assert candidate['case_id']==detector['case_id']
    categories=[canonical(o['parsed']['category']) for o in observations if o.get('parsed') and not o.get('error') and o.get('finish_reason')=='stop']
    semantic=canonical(candidate['category']) in categories
    box=bool(detector.get('selected') and detector['selected']['score']>=SPEC['detector_score_min'] and detector['candidate_iou']>=SPEC['candidate_detector_iou_min'])
    accept=box and semantic
    return {'case_id':candidate['case_id'],'decision':'accept_candidate' if accept else 'unknown','entity_category':'supported_proxy' if accept else 'unknown','single_instance_localization':'acceptable_proxy' if accept else 'unknown','detector_geometry_agreement':box,'semantic_category_support':semantic,'candidate_iou':detector['candidate_iou'],'observed_categories':categories,'reason':'independent_category_and_context_detector_agreement' if accept else ('semantic_support_missing' if box else 'detector_support_missing'),'cross_prediction_uniqueness':'not_evaluated','reference_labels_used':False}
