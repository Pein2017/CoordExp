#!/usr/bin/env python3
"""Independent CPU readback for corrected pilot-v2 raw receipts."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

WORKTREE=Path('/data/CoordExp/.worktrees/research-probes')
ROOT=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source')
OUT=ROOT/'final/corrected-pilot-v2'
MODELS=('tied','untied')
CELLS=('00','10-','10+','01-','01+','11-','11+')
SIGNED=CELLS[1:]

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def bind(path:Path)->dict[str,Any]:
 path=path.resolve(strict=True);return {'path':str(path),'sha256':sha(path),'size_bytes':path.stat().st_size}
def load(path:Path):return json.loads(path.read_text())
def hash_json(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def compact_cell(cell:dict[str,Any])->dict[str,Any]:
 free=cell['free']; parsed=free.get('parse',{}); boundary=cell.get('boundary',{}); forced=boundary.get('forced_description_x1',{})
 row_limit=free.get('row_limit',{})
 return {
  'status':cell.get('status'),
  'token_count':len(free.get('token_ids',[])),
  'stop_reason':free.get('stop_reason'),
  'row_limit':row_limit,
  'parse':{
   'complete_rows':parsed.get('complete_rows'),'valid_rows':parsed.get('valid_rows'),
   'invalid_rows':parsed.get('invalid_rows'),'malformed_rows':parsed.get('malformed_rows'),
   'failure_predicate':parsed.get('failure_predicate'),
   'exact_runs':parsed.get('exact_runs',[]),'near_runs':parsed.get('near_runs',[]),
  },
  'known':free.get('known'),
  'identity':{
   'request_ids':cell.get('input_identity',{}).get('request_ids'),
   'media_sha256':cell.get('input_identity',{}).get('media_sha256'),
   'image_grids':cell.get('input_identity',{}).get('image_grids'),
   'prompt_token_sha256':hash_json(cell.get('input_identity',{}).get('prompt_token_ids',[])),
   'input_ids_sha256':cell.get('input_identity',{}).get('tensor_sha256',{}).get('input_ids'),
  },
  'boundary':{
   'opener':{k:forced.get(k) for k in ('chosen_token_id','stop_reason','input_width','eos','coordinate_family') if k in forced} if False else boundary.get('opener'),
   'forced_description_x1':{
    'chosen_token_id':forced.get('chosen_token_id'),'stop_reason':forced.get('stop_reason'),
    'input_width':forced.get('input_width'),'eos':forced.get('eos'),
    'coordinate_family':forced.get('coordinate_family'),
    'windows':forced.get('windows'),'windows_by_sign':forced.get('windows_by_sign'),
    'window_selection':forced.get('window_selection'),
    'coordinate_log_probs_count':len(forced.get('coordinate_log_probs',[])),
    'top10_count':len(forced.get('top10',[])),
   },
  },
 }

def main():
 input_receipt=OUT/'input-construction-receipt.json'; launch_receipt=OUT/'launch-receipt.json'; transform_receipt=OUT/'transform-launch-receipt.json'; snapshot=OUT/'source-snapshot.json'
 ir=load(input_receipt); lr=load(launch_receipt); tr=load(transform_receipt)
 errors=[]; models={}; identity_checks={}
 if ir.get('status')!='ready' or ir.get('constructed_cells')!=315: errors.append('all45 input construction receipt is not 45/315 ready')
 if ir.get('panel',{}).get('sha256') != '005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49': errors.append('panel hash drift')
 for model in MODELS:
  zero_path=OUT/'raw'/f'{model}-runtime.json'; signed_path=OUT/'raw'/f'{model}-transforms-runtime.json'
  zero=load(zero_path); signed=load(signed_path)
  cells={}
  if set(zero.get('cells',{})) != {'00'}: errors.append(f'{model}: 00 raw cell set drift')
  if set(signed.get('cells',{})) != set(SIGNED): errors.append(f'{model}: signed raw cell set drift')
  cells.update(zero.get('cells',{})); cells.update(signed.get('cells',{}))
  if set(cells) != set(CELLS): errors.append(f'{model}: merged cell set drift')
  manifest=load(OUT/'inputs/manifests'/f'{model}-417044-failure.json')
  if manifest['source']['image_id'] != 417044 or manifest['source']['group'] != 'refined-03': errors.append(f'{model}: pilot source identity drift')
  if manifest['prefix']['source_row_index'] not in (11,16): errors.append(f'{model}: unexpected original pilot row')
  # All cells must use the same native request/prompt shape; media identity
  # may differ only with the declared visual offset.
  prompt_hashes=[]; request_ids=[]; grids=[]; input_shas=[]; media_by_cell={}
  for key in CELLS:
   cell=cells[key]; ident=cell.get('input_identity',{})
   prompt_hashes.append(hash_json(ident.get('prompt_token_ids',[])))
   request_ids.append(tuple(ident.get('request_ids',[])))
   grids.append(tuple(ident.get('image_grids',[[]])[0]))
   input_shas.append(ident.get('tensor_sha256',{}).get('input_ids'))
   media_by_cell[key]=tuple(ident.get('media_sha256',[]))
   row_limit=cell.get('free',{}).get('row_limit')
   if not isinstance(row_limit,dict) or row_limit.get('mode')!='free_suffix_complete_row_cap': errors.append(f'{model}/{key}: row-limit provenance missing')
   else:
    complete=cell.get('free',{}).get('parse',{}).get('complete_rows')
    if row_limit.get('last_free_complete_rows') != [complete]: errors.append(f'{model}/{key}: row-limit/readback count mismatch')
    if row_limit.get('injected_eos') != (row_limit.get('injection_reason')=='row_cap'): errors.append(f'{model}/{key}: row-limit injection provenance mismatch')
   forced=cell.get('boundary',{}).get('forced_description_x1',{})
   if len(forced.get('coordinate_log_probs',[])) != 1000: errors.append(f'{model}/{key}: forced full coordinate vocabulary missing')
   if set(forced.get('windows_by_sign',{})) != {'-','+'}: errors.append(f'{model}/{key}: fixed-sign windows missing')
  if len(set(prompt_hashes)) != 1: errors.append(f'{model}: prompt identity mismatch across cells')
  if len(set(request_ids)) != 1: errors.append(f'{model}: request identity mismatch across cells')
  if len(set(grids)) != 1: errors.append(f'{model}: image grid mismatch across cells')
  if len(set(input_shas)) != 1: errors.append(f'{model}: input_ids tensor identity mismatch across cells')
  expected_same_visual=[('00','01-'),('00','01+'),('10-','11-'),('10+','11+')]
  media_mismatches=[]
  for left,right in expected_same_visual:
   if media_by_cell[left] != media_by_cell[right]: media_mismatches.append([left,right])
  if media_mismatches: errors.append(f'{model}: same-visual media identity mismatch {media_mismatches}')
  # Different visual offsets must remain distinguishable in processor media.
  if media_by_cell['00']==media_by_cell['10-'] or media_by_cell['00']==media_by_cell['10+']:
   errors.append(f'{model}: visual offset media identity collapsed')
  zero_admission=zero.get('admission',{})
  models[model]={
   '00_status':zero.get('status'),'signed_status':signed.get('status'),
   'admission':zero_admission,
   'model_forwards':int(zero.get('model_forwards',0))+int(signed.get('model_forwards',0)),
   'vision_forwards':int(zero.get('vision_forwards',0))+int(signed.get('vision_forwards',0)),
   'elapsed_seconds':float(zero.get('elapsed_seconds',0))+float(signed.get('elapsed_seconds',0)),
   'cells':{key:compact_cell(cells[key]) for key in CELLS},
  }
  identity_checks[model]={
   'prompt_identity_hashes':sorted(set(prompt_hashes)),
   'request_ids':sorted(set(request_ids)),
   'image_grids':sorted(set(grids)),
   'input_ids_tensor_hashes':sorted(set(input_shas)),
   'media_sha256_by_cell':media_by_cell,
   'same_visual_media_pairs':expected_same_visual,
   'same_visual_pair_mismatches':media_mismatches,
   'different_visual_media_distinct':media_by_cell['00']!=media_by_cell['10-'] and media_by_cell['00']!=media_by_cell['10+'],
   'native_route':zero.get('source',{}).get('native_route'),
  }
 summary={
  'schema':'recurrence_spatial_source.corrected_pilot_v2_reduction.v1',
  'unit_id':'2026-09-19-recurrence-spatial-source','attempt_id':'corrected-pilot-v2',
  'status':'bounded_hold' if not errors else 'technical_invalid',
  'decision':'stop_broad_launch' if not errors else 'do_not_interpret',
  'panel_sha256':ir.get('panel',{}).get('sha256'),
  'input_construction':{'receipt':bind(input_receipt),'declared_states':ir.get('declared_states'),'constructed_cells':ir.get('constructed_cells'),'model_calls':0,'errors':ir.get('errors',[])},
  'launch':{'initial':bind(launch_receipt),'signed_transforms':bind(transform_receipt),'device':'cuda:0','gpu_seconds':None,'wall_seconds':lr.get('wall_seconds_total',0)+tr.get('wall_seconds_total',0)},
  'denominator':{'models':2,'cells_per_model':7,'free_continuations':14,'00_cells':2,'signed_cells':12,'all45_cpu_cells':315},
  'models':models,'identity_checks':identity_checks,
  'errors':errors,
  'source_bindings':{'snapshot':bind(snapshot),'manifests':{m:bind(OUT/'inputs/manifests'/f'{m}-417044-failure.json') for m in MODELS},'raw':{p.stem:bind(p) for p in OUT.joinpath('raw').glob('*.json')}},
  'stop_rule':'00 admission HOLD is preserved; signed cells were run with skip_gate only to expose diagnostics; no tuning/replacement/broad launch',
 }
 out=OUT/'reduced/pilot-summary.json'; out.write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'status':summary['status'],'errors':len(errors),'cells':14,'models':models},indent=2))
 if errors: raise SystemExit(1)

if __name__=='__main__':main()
