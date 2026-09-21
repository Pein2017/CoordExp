#!/usr/bin/env python3
"""Exact-history CPU consumer; descriptive four-cell accounting, not causal attribution."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import math
from typing import Any

from src.data.geometry import coord_bins_to_pixel_xyxy, iou_xyxy
from src.eval.assignment import global_matches
from src.eval.detection_categories import normalize_coco_category_name
from src.inference.parsing import parse_compact_object_box_closed

MANIFEST_SHA256 = 'af3fd69e05bea3c148c293e528a2ea738ac49e45631bb0b71a94d7eb56d3fcd8'
CELLS = {'00':('source','source'), '01':('source','rweak'),
         '10':('rweak','source'), '11':('rweak','rweak')}
THRESHOLDS = (.5,.6,.8)
CAP = 3084
EOS = 151645


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def decode(tokenizer, ids):
    require(isinstance(ids,list) and all(type(i) is int and i >= 0 for i in ids), 'invalid token ID sequence')
    return tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)


def parse_raw(text, identity):
    parsed = parse_compact_object_box_closed(text, row_id=identity['row_id'], row_index=identity['row_index'],
                                            image_width=identity['image_width'], image_height=identity['image_height'])
    row = {k:identity[k] for k in ('row_id','row_index','example_id','image_path','image_width','image_height','gt')}
    row.update(raw_decode_text=text, pred=parsed.predictions, dropped_predictions=parsed.dropped_predictions,
               valid_prediction_count=parsed.valid_prediction_count,dropped_prediction_count=parsed.dropped_prediction_count,
               parse_status=parsed.parse_status,parser_id=parsed.parser_id,parser_policy=parsed.parser_policy,
               metric_bearing=parsed.metric_bearing)
    return row


def validate_parsed(saved, text, identity):
    fresh = parse_raw(text, identity)
    for key, value in fresh.items():
        require(key in saved and saved[key] == value, f'saved parsed mismatch: {key}')
    return fresh


def validate_stop(ids, suffix, action, declared):
    require(0 < len(ids) <= CAP, 'whole trajectory cap mismatch')
    if action['kind']=='eos':
        require(action['token_ids']==[EOS] and not suffix, 'forced EOS reopened')
        require(ids[-1]==EOS and EOS not in ids[:-1], 'forced EOS sequence mismatch')
        require(declared in ('im_end','forced_eos','forced_im_end'), 'forced EOS stop mismatch')
        return 'forced_eos'
    require(action['kind']=='row' and EOS not in action['token_ids'], 'invalid row action')
    if EOS in ids:
        require(ids[-1]==EOS and EOS not in ids[:-1] and declared=='im_end', 'natural EOS stop mismatch')
        return 'natural_eos'
    require(len(ids)==CAP and declared=='length', 'nonterminal or incorrectly capped trajectory')
    return 'length'


def validate_cross(record, case, tokenizer, manifest_hash):
    cell=record['cell']
    require(cell in ('01','10'), 'cross consumer accepts only off-diagonal cells')
    require((record['recipient'],record['action_source'])==CELLS[cell], 'donor/recipient/cell mismatch')
    require(record['mode'] in ('cross','qualify'), 'unsupported record mode')
    require(record['manifest_sha256']==manifest_hash, 'cross manifest hash mismatch')
    require(record['case_id']==case['row_id'], 'cross case identity mismatch')
    action=case['actions'][record['action_source']]
    prefix=case['common_prefix_token_ids']
    require(record['prefix_token_ids']==prefix and record['action_token_ids']==action['token_ids'], 'forced prefix/action token mismatch')
    ids=record['generated_token_ids']
    suffix=record['suffix_token_ids']
    require(ids==prefix+action['token_ids']+suffix, 'literal prefix/action/suffix concatenation mismatch')
    require(record['remaining_budget']==CAP-len(prefix)-len(action['token_ids']), 'remaining budget mismatch')
    require(len(suffix)<=record['remaining_budget'], 'suffix exceeds remaining budget')
    require(record['generated_token_count']==len(ids), 'generated token count mismatch')
    text=decode(tokenizer,ids)
    require(text==record['raw_decode_text'], 'token/text mismatch')
    stop=validate_stop(ids,suffix,action,record['decode_stop_reason'])
    identity=case['diagonals']['source']['raw_record']
    parsed=validate_parsed(record['parsed'],text,identity)
    require(record['parsed'].get('decode_stop_reason')==record['decode_stop_reason'], 'saved parsed stop reason mismatch')
    parsed['decode_stop_reason']=record['decode_stop_reason']
    return parsed, stop


def ratio(n,d):
    return n/d if d else 0.0


def scores(tp,npred,ngt):
    return {'tp':tp,'fp':npred-tp,'fn':ngt-tp,'precision':ratio(tp,npred),
            'recall':ratio(tp,ngt),'f1':ratio(2*tp,npred+ngt)}


def _category(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    return normalize_coco_category_name(
        obj.get("description", obj.get("desc", obj.get("label", obj.get("category", ""))))
    )


def _bbox(obj: Any) -> Any:
    return obj.get("bbox", obj.get("bbox_2d")) if isinstance(obj, dict) else None


def _pixel_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box):
        return None
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return None
    return box


def _dimensions(row: dict[str, Any]) -> tuple[int, int] | None:
    width = row.get("image_width", row.get("width"))
    height = row.get("image_height", row.get("height"))
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return None
    return width, height


def _gt_objects(row: dict[str, Any], *, row_id: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    objects = row.get("gt", [])
    if not isinstance(objects, list):
        raise ValueError(f"row {row_id!r} has malformed GT list")
    dimensions = _dimensions(row)
    if dimensions is None:
        raise ValueError(f"row {row_id!r} has invalid image dimensions")
    width, height = dimensions
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, obj in enumerate(objects):
        raw = _bbox(obj)
        try:
            box = coord_bins_to_pixel_xyxy(
                raw,
                image_width=width,
                image_height=height,
                field=f"gt[{index}].bbox",
            )
        except Exception as exc:
            raise ValueError(f"row {row_id!r} has invalid GT bbox at index {index}") from exc
        result.append((_category(obj), tuple(float(item) for item in box)))
    return result


def _pred_objects(row: dict[str, Any]) -> tuple[list[tuple[str, tuple[float, float, float, float]]], int]:
    objects = row.get("pred", [])
    if not isinstance(objects, list):
        return [], 1
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    invalid = 0
    for obj in objects:
        box = _pixel_box(_bbox(obj))
        if box is None or not _category(obj):
            invalid += 1
            continue
        result.append((_category(obj), box))
    return result, invalid


def owner_evidence(row, threshold):
    """Include all compatible edges: chosen assignment alone is not direct support."""
    gt=_gt_objects(row,row_id=row['row_id'])
    pred,invalid=_pred_objects(row)
    owners=[int(g['object_id']) for g in row['gt']]
    require(len(owners)==len(set(owners)), 'duplicate canonical GT owner')
    matches=global_matches(gt,pred,threshold)
    edges=[{'gt_id':owners[i],'prediction_index':j,'iou':iou_xyxy(box,pbox)}
           for i,(category,box) in enumerate(gt) for j,(pcategory,pbox) in enumerate(pred)
           if category==pcategory and iou_xyxy(box,pbox)>=threshold]
    return {'matched_gt_ids':sorted(owners[i] for i,_,_ in matches),
            'matches':[{'gt_id':owners[i],'prediction_index':j,'iou':overlap} for i,j,overlap in matches],
            'incident_gt_ids':sorted({edge['gt_id'] for edge in edges}),'compatible_edges':edges,
            'valid_prediction_count':len(pred),'invalid_prediction_count':invalid,
            **scores(len(matches),len(pred),len(gt))}


def strict_later_repeats(row):
    # Independent of category and GT; count later predictions, not all pairs.
    predictions=row['pred']
    result=[]
    for later in range(len(predictions)):
        later_objects,_=_pred_objects({'pred':[predictions[later]]})
        if not later_objects:
            continue
        earlier=[]
        for before in range(later):
            before_objects,_=_pred_objects({'pred':[predictions[before]]})
            if before_objects and iou_xyxy(before_objects[0][1],later_objects[0][1])>.95:
                earlier.append({'prediction_index':before,'iou':iou_xyxy(before_objects[0][1],later_objects[0][1])})
        if earlier:
            result.append({'later_prediction_index':later,'earlier':earlier})
    return result


def analyze_output(row, prefix_action_text, ids, suffix_length, stop, tokenizer):
    direct=parse_raw(prefix_action_text,row)
    repeated=strict_later_repeats(row)
    result={'generated_token_ids':ids,'raw_record':row,'prefix_action_text':prefix_action_text,
            'generated_token_count':len(ids),'suffix_token_count':suffix_length,'stop':stop,
            'valid_prediction_count':len(_pred_objects(row)[0]),'invalid_prediction_count':_pred_objects(row)[1],
            'parser_drop_count':row['dropped_prediction_count'],'parse_status':row['parse_status'],
            'strict_later_repeat_count':len(repeated),'strict_later_repeats':repeated,'thresholds':{}}
    for threshold in THRESHOLDS:
        full=owner_evidence(row,threshold)
        action=owner_evidence(direct,threshold)
        full['prefix_action'] = action
        full['direct_or_matching_ambiguous_gt_ids']=sorted(set(full['matched_gt_ids']) & set(action['incident_gt_ids']))
        full['pure_tail_gt_ids']=sorted(set(full['matched_gt_ids'])-set(action['incident_gt_ids']))
        result['thresholds'][f'iou_{threshold:.2f}']=full
    return result


def compare(left, right):
    """right minus left, never replacing complete-output primary owner accounting."""
    l,r=set(left['matched_gt_ids']),set(right['matched_gt_ids'])
    incident=set(left['prefix_action']['incident_gt_ids']) | set(right['prefix_action']['incident_gt_ids'])
    changes={'gained_gt_ids':sorted(r-l),'lost_gt_ids':sorted(l-r),'retained_gt_ids':sorted(l&r)}
    return {**changes,'union_prefix_action_incident_gt_ids':sorted(incident),
            'attribution':{name:{'direct_or_matching_ambiguous':sorted(set(ids)&incident),
                                 'pure_tail':sorted(set(ids)-incident)} for name,ids in changes.items()}}


def owner_patterns(outputs, gt, threshold):
    sets={cell:set(value['thresholds'][threshold]['matched_gt_ids']) for cell,value in outputs.items()}
    incident={cell:set(value['thresholds'][threshold]['prefix_action']['incident_gt_ids']) for cell,value in outputs.items()}
    all_incident=set().union(*incident.values())
    records=[]
    for owner in sorted(int(g['object_id']) for g in gt):
        native='lost' if owner in sets['00']-sets['11'] else 'gained' if owner in sets['11']-sets['00'] else 'retained' if owner in sets['00']&sets['11'] else 'unmatched_both'
        records.append({'gt_id':owner,'pattern':''.join('1' if owner in sets[cell] else '0' for cell in ('00','01','10','11')),
                        'present':{cell:owner in owners for cell,owners in sets.items()},'native_rweak_vs_source':native,
                        'cross_attribution':{cell:('not_realized' if owner not in sets[cell] else 'direct_or_matching_ambiguous' if owner in all_incident else 'pure_tail') for cell in ('01','10')},
                        'prefix_action_incident_cells':[cell for cell in ('00','01','10','11') if owner in incident[cell]]})
    return records


ORIGINAL_CODE_PREFIX = Path('/data/CoordExp/.worktrees/coco-gt-correction-portfolio')


def original_code_bindings(manifest, original_code_root=None):
    """Locate historical bytes for verification only; never import or execute them."""
    bindings = []
    for source in manifest['sources']['code']:
        original = Path(source['path'])
        resolved = original if original_code_root is None else Path(original_code_root) / original.relative_to(ORIGINAL_CODE_PREFIX)
        require(sha256(resolved) == source['sha256'], f"original consumer/provider code changed: {original}")
        bindings.append({'original_path': str(original), 'resolved_path': str(resolved.resolve()), 'sha256': source['sha256']})
    return bindings


def load_manifest(path, *, original_code_root=None):
    require(sha256(path)==MANIFEST_SHA256,'frozen manifest hash mismatch')
    manifest=json.loads(Path(path).read_text())
    require(manifest['schema']=='row_cross_manifest_v1' and len(manifest['cases'])==32,'frozen panel schema/count mismatch')
    require(manifest['selection']['population_images']==512 and manifest['selection']['population_gt']==3759,'frozen population denominator mismatch')
    original_code_bindings(manifest, original_code_root)
    return manifest


def reduce_files(manifest_path, cross_dirs, qualification_case_ids=None, *, original_code_root=None):
    manifest=load_manifest(manifest_path, original_code_root=original_code_root)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(manifest['sources']['source']['config']['model']['base_model'],local_files_only=True)
    cases={c['row_id']:c for c in manifest['cases']}
    selected=set(cases) if qualification_case_ids is None else set(qualification_case_ids)
    require(bool(selected) and selected <= set(cases),'unknown or empty qualification subset')
    require(qualification_case_ids is None or len(selected)==len(qualification_case_ids),'duplicate qualification case ID')
    saved={}
    input_bindings=[]
    synthetic=False
    for directory in cross_dirs:
        path=Path(directory)/'rows.jsonl'
        input_bindings.append({'path':str(path.resolve()),'sha256':sha256(path)})
        with path.open() as stream:
            for line in stream:
                record=json.loads(line)
                key=(record['case_id'],record['cell'])
                require(key[0] in selected and key[1] in ('01','10'),'foreign case/cell in cross input')
                require(key not in saved,'duplicate cross case/cell')
                validate_cross(record,cases[key[0]],tokenizer,MANIFEST_SHA256)
                synthetic = synthetic or bool(record.get('synthetic_fixture',False))
                saved[key]=record
    require(set(saved)=={(rid,cell) for rid in selected for cell in ('01','10')},'incomplete cross case/cell denominator')
    require(not synthetic or qualification_case_ids is not None,'synthetic fixture cannot be a final panel')
    results=[]
    for rid in sorted(selected):
        case=cases[rid]
        outputs={}
        for cell,(recipient,donor) in CELLS.items():
            action=case['actions'][donor]
            prefix=case['common_prefix_token_ids']
            consumed=prefix+action['token_ids']
            if cell in ('00','11'):
                diag=case['diagonals'][recipient]
                ids=diag['generated_token_ids']
                require(ids[:len(consumed)]==consumed,'diagonal forced-history mismatch')
                text=decode(tokenizer,ids)
                row=validate_parsed(diag['raw_record'],text,case['diagonals']['source']['raw_record'])
                require(diag['raw_record']['decode_stop_reason']==diag['stop_reason'],'diagonal stop identity mismatch')
                row['decode_stop_reason']=diag['stop_reason']
                suffix=ids[len(consumed):]
                stop=validate_stop(ids,suffix,action,diag['stop_reason'])
                if stop=='forced_eos':
                    stop='natural_eos'  # Stored diagonals were native-from-prompt, never forced.
            else:
                record=saved[(rid,cell)]
                row,stop=validate_cross(record,case,tokenizer,MANIFEST_SHA256)
                ids,suffix=record['generated_token_ids'],record['suffix_token_ids']
            result=analyze_output(row,decode(tokenizer,consumed),ids,len(suffix),stop,tokenizer)
            if cell in ('00','11'):
                require(all(result['thresholds'][t]['matched_gt_ids']==case['diagonals'][recipient]['matched_gt_ids'][t] for t in result['thresholds']), 'frozen diagonal owner matches changed')
            outputs[cell]=result
        comparisons={}
        patterns={}
        for threshold in ('iou_0.50','iou_0.60','iou_0.80'):
            comparisons[threshold]={f'{right}_vs_{left}':compare(outputs[left]['thresholds'][threshold],outputs[right]['thresholds'][threshold])
                                    for left,right in (('00','11'),('00','01'),('11','10'),('00','10'),('11','01'),('01','10'))}
            patterns[threshold]=owner_patterns(outputs,case['gt'],threshold)
        results.append({'row_id':rid,'stratum':case['stratum'],'gt_count':len(case['gt']),
                        'outputs':outputs,'comparisons':comparisons,'owner_patterns':patterns,
                        'baseline_changed_owner_patterns':{t:[r for r in records if r['native_rweak_vs_source'] in ('lost','gained')] for t,records in patterns.items()}})
    aggregate={}
    for cell in CELLS:
        outputs=[case['outputs'][cell] for case in results]
        summary={key:sum(r[key] for r in outputs) for key in ('generated_token_count','suffix_token_count','valid_prediction_count','invalid_prediction_count','parser_drop_count','strict_later_repeat_count')}
        summary['stop_counts']=dict(Counter(r['stop'] for r in outputs))
        summary['thresholds']={}
        for threshold in ('iou_0.50','iou_0.60','iou_0.80'):
            values=[row['thresholds'][threshold] for row in outputs]
            tp=sum(v['tp'] for v in values)
            micro=scores(tp,sum(v['tp']+v['fp'] for v in values),sum(v['tp']+v['fn'] for v in values))
            summary['thresholds'][threshold]={'micro':micro,'macro':{key:sum(v[key] for v in values)/len(values) for key in ('precision','recall','f1')},
                                              'matched_gt_ids':sorted(owner for value in values for owner in value['matched_gt_ids'])}
        aggregate[cell]=summary
    pattern_counts={}
    aggregate_comparisons={}
    for threshold in ('iou_0.50','iou_0.60','iou_0.80'):
        counts=Counter(r['pattern'] for case in results for r in case['owner_patterns'][threshold])
        pattern_counts[threshold]={f'{i:04b}':counts[f'{i:04b}'] for i in range(16)}
        aggregate_comparisons[threshold]={}
        for name in results[0]['comparisons'][threshold]:
            values=[case['comparisons'][threshold][name] for case in results]
            combined={key:sorted(owner for v in values for owner in v[key]) for key in ('gained_gt_ids','lost_gt_ids','retained_gt_ids')}
            combined['counts']={key:len(ids) for key,ids in combined.items()}
            combined['attribution']={key:{kind:sorted(owner for v in values for owner in v['attribution'][key][kind])
                                           for kind in ('direct_or_matching_ambiguous','pure_tail')}
                                     for key in ('gained_gt_ids','lost_gt_ids','retained_gt_ids')}
            aggregate_comparisons[threshold][name]=combined
    return {'schema':'row_cross_reduction_v1','status':'consumer_validated','scope':'final_panel' if qualification_case_ids is None else 'qualification_subset',
            'synthetic_fixture':synthetic,'manifest_sha256':MANIFEST_SHA256,'manifest_path':str(Path(manifest_path).resolve()),
            'execution':{'reducer_path':str(Path(__file__).resolve()),'reducer_sha256':sha256(__file__),
                         'original_code_bindings':original_code_bindings(manifest, original_code_root)},
            'cross_inputs':input_bindings,'case_count':len(results),'cross_output_count':len(saved),'complete_output_count':len(results)*4,
            'gt_count':sum(c['gt_count'] for c in results),'cell_order':['00','01','10','11'],
            'policies':{'zero_denominator_scores':0.0,'repeat_rule':'category/GT-independent pixel IoU strictly >0.95; count each later prediction once',
                        'attribution':'Pure-tail only outside union of compared prefix/action compatible-edge owner sets; incident owners are direct/matching-ambiguous, not causally assigned.',
                        'primary':'Complete-output category-consistent global one-to-one IoU matching; never remove direct-action losses',
                        'claim_scope':'Selected-panel descriptive conditional interventions only; no causal shares or native deployment claims'},
            'aggregate':aggregate,'aggregate_comparisons':aggregate_comparisons,'owner_pattern_counts':pattern_counts,'cases':results}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--cross-dir',type=Path,action='append',required=True,help='Repeat for independent rows.jsonl directories; no duplicates allowed')
    ap.add_argument('--qualification-case-ids',help='Explicit comma-separated subset; report is never marked final_panel')
    ap.add_argument('--original-code-root',type=Path,help='Preserved original COCO code root; historical bytes are verified, never imported')
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    ids=None if args.qualification_case_ids is None else args.qualification_case_ids.split(',')
    result=reduce_files(args.manifest,args.cross_dir,ids,original_code_root=args.original_code_root)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'reduction.json'
    with path.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print(json.dumps({'output':str(path),'sha256':sha256(path),'scope':result['scope'],'case_count':result['case_count'],'cross_output_count':result['cross_output_count']}))


if __name__=='__main__':
    main()
