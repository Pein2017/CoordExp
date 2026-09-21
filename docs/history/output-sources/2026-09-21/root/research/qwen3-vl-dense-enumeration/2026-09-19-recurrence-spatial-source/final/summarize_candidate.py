from __future__ import annotations
import collections, hashlib, json, math, pathlib, statistics
import sys
from typing import Any

BASE = pathlib.Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source')
REPO = pathlib.Path('/data/CoordExp/.worktrees/research-probes')
PANEL = pathlib.Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json')
SNAP = BASE/'final/source-snapshot.json'
IDX = BASE/'final/execution-index.json'

if '--check-only' in sys.argv:
    result_path = BASE/'final/result.json'
    result = json.loads(result_path.read_text())
    assert result.get('schema') == 'recurrence_spatial_source_candidate_result.v1'
    assert result.get('declared_denominator', {}).get('states') == 45
    assert len(result.get('state_summaries', [])) == 45
    assert len(result.get('cell_summaries', [])) == 267
    assert result.get('panel', {}).get('sha256') == '005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49'
    print(json.dumps({'status':'ok','states':45,'cells':267,'panel_sha256':result['panel']['sha256']}))
    raise SystemExit(0)


def load(path: pathlib.Path):
    return json.loads(path.read_text())

def sha(path: pathlib.Path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def bind(path: pathlib.Path):
    path=path.resolve(); return {'path':str(path),'sha256':sha(path),'size_bytes':path.stat().st_size}

def pabs(s: str|pathlib.Path) -> pathlib.Path:
    return pathlib.Path(s).expanduser().resolve()

def quant(v):
    if isinstance(v,float): return round(v,6)
    return v

def compact_boundary(raw: dict[str,Any]) -> dict[str,Any] | None:
    b=raw.get('boundary') or {}
    op=b.get('opener') or {}
    fd=b.get('forced_description_x1') or {}
    top=op.get('top10') or []
    chosen=op.get('chosen_token_id')
    chosen_ent=next((x for x in top if x.get('token_id')==chosen),None)
    def comp(e):
        if not isinstance(e,dict): return None
        return {k:e.get(k) for k in ('token_id','logit','log_prob')}
    mass=op.get('coordinate_family') or {}
    eos=op.get('eos') or {}
    fmass=fd.get('coordinate_family') or {}
    feos=fd.get('eos') or {}
    out={
        'opener': {
            'chosen_token_id':chosen,
            'chosen_logit':chosen_ent.get('logit') if chosen_ent else None,
            'chosen_log_prob':chosen_ent.get('log_prob') if chosen_ent else None,
            'eos_logit':eos.get('logit'), 'eos_log_prob':eos.get('log_prob'),
            'coordinate_family_log_mass':mass.get('log_mass'), 'coordinate_family_mass':mass.get('mass'),
            'coordinate_family_max_bin':mass.get('max_bin'), 'coordinate_family_max_log_prob':mass.get('max_log_prob'),
            'top3': [comp(x) for x in top[:3]],
        },
        'forced_description_x1': {
            'chosen_token_id':fd.get('chosen_token_id'),
            'chosen_logit':(fd.get('top10') or [{}])[0].get('logit') if fd.get('top10') else None,
            'chosen_log_prob':(fd.get('top10') or [{}])[0].get('log_prob') if fd.get('top10') else None,
            'eos_logit':feos.get('logit'), 'eos_log_prob':feos.get('log_prob'),
            'coordinate_family_log_mass':fmass.get('log_mass'), 'coordinate_family_mass':fmass.get('mass'),
            'coordinate_family_max_bin':fmass.get('max_bin'), 'coordinate_family_max_log_prob':fmass.get('max_log_prob'),
            'top3': [comp(x) for x in (fd.get('top10') or [])[:3]],
            'windows': fd.get('windows'),
        } if fd else None,
    }
    return out if b else None

def run_anchor(runs):
    if not runs: return None
    # exact is preferred by the predicate; caller supplies preferred list.
    r=max(runs,key=lambda x:(x.get('length',0),-x.get('start_row',0)))
    box=r.get('coord_bins_source')
    center=[round((box[0]+box[2])/2,3),round((box[1]+box[3])/2,3)] if box else None
    return {'start_row':r.get('start_row'),'length':r.get('length'),'description':r.get('description'),'coord_bins_source':box,'center_source':center}

def cell_metrics(raw: dict[str,Any], red: dict[str,Any], manifest_cell: dict[str,Any], center: dict[str,Any]|None):
    pa=red['parse']; rows=pa.get('rows',[])
    exact=pa.get('exact_runs',[]); near=pa.get('near_runs',[])
    preferred=exact if exact else near
    anchor=run_anchor(preferred)
    valid=[r for r in rows if r.get('source_geometry_valid')]
    centers=[[ (r['coord_bins_source'][0]+r['coord_bins_source'][2])/2, (r['coord_bins_source'][1]+r['coord_bins_source'][3])/2] for r in valid if r.get('coord_bins_source')]
    mean=[round(statistics.mean(x),3) for x in zip(*centers)] if centers else None
    cdesc=sorted({str(x.get('description')) for x in preferred})
    center_desc=[]
    if center:
        center_desc=center.get('repeat_descriptions',[])
    same_desc=bool(cdesc and center_desc and set(cdesc)&set(center_desc))
    rec=bool(exact or near)
    alt=bool(rec and center and ((not center.get('recurrence')) or not same_desc))
    op=compact_boundary(raw)
    return {
      'key': manifest_cell.get('key'),
      'visual_offset_px': manifest_cell.get('visual_offset_px'),
      'history_offset_px': manifest_cell.get('history_offset_px'),
      'history_sha256': manifest_cell.get('history_sha256'),
      'status': raw.get('status'),
      'recurrence': rec,
      'exact_recurrence': bool(exact), 'near_recurrence': bool(near),
      'exact_runs': exact, 'near_runs': near,
      'repeat_descriptions': cdesc,
      'repeat_anchor': anchor,
      'alternative_recurrence_vs_00': alt,
      'same_repeat_description_as_00': same_desc if center else None,
      'pair_edges_exact': sum(int(x.get('length',0))*(int(x.get('length',0))-1)//2 for x in exact),
      'pair_edges_near': sum(int(x.get('length',0))*(int(x.get('length',0))-1)//2 for x in near),
      'complete_rows': pa.get('complete_rows'), 'valid_rows': pa.get('valid_rows'),
      'invalid_rows': pa.get('invalid_rows'), 'malformed_rows': pa.get('malformed_rows'),
      'canvas_border_rows': sum(bool(r.get('canvas_border')) for r in rows),
      'source_out_of_bounds_rows': sum(not bool(r.get('source_in_bounds')) for r in rows if 'source_in_bounds' in r),
      'source_geometry_invalid_rows': sum(not bool(r.get('source_geometry_valid')) for r in rows),
      'mean_valid_center_source': mean,
      'output_token_count': red.get('token_count'), 'stop_reason': red.get('stop_reason'),
      'boundary': op,
      'input_identity': raw.get('input_identity'),
    }

def shift(a,b):
    if not a or not b: return None
    return [round(b[i]-a[i],3) for i in range(2)]

def source_history(manifest):
    cells=manifest.get('cells',{})
    rows=[]; changes=[]; mapped_bad={}; order_bad={}
    for key,c in cells.items():
        hb=c.get('history_boxes') or []
        if key=='00': rows=hb
        mapped_bad[key]=sum(not bool(x.get('mapped_valid')) for x in hb)
        order_bad[key]=sum(not bool(x.get('order_preserved')) for x in hb)
        changes.extend([x for x in hb if x.get('validity_changed_by_rounding')])
    return {
      'source_rows':len(rows),
      'source_invalid_rows':sum(not bool(x.get('source_valid')) for x in rows),
      'source_order_invalid_rows':sum(not bool(x.get('source_valid')) or not bool(x.get('order_preserved')) for x in rows),
      'mapped_invalid_rows_by_cell':mapped_bad,
      'order_invalid_rows_by_cell':order_bad,
      'rounding_validity_changes':[{'cell':k,'row_index':x.get('row_index'),'source_bins':x.get('source_bins'),'mapped_bins':x.get('mapped_bins'),'inverse_bins':x.get('inverse_bins')} for k,c in cells.items() for x in (c.get('history_boxes') or []) if x.get('validity_changed_by_rounding')],
    }

def relative_path(path: pathlib.Path):
    try:return str(path.relative_to(REPO))
    except ValueError:return str(path)

idx=load(IDX)
state_summaries=[]; cell_summaries=[]; source_bindings=[]
for e in idx['states']:
    sid=e['id']; mode=e.get('mode'); model=e['model']; kind=e.get('kind')
    manifest=pabs(e['manifest']['path'])
    man=load(manifest)
    if mode=='reuse_pilot':
        center_rt=pabs(e['runtime']['center']['path']); trans_rt=pabs(e['runtime']['transforms']['path'])
        center_red=pabs(e['runtime']['reduced_center']['path']); trans_red=pabs(e['runtime']['reduced_transforms']['path'])
        rt0=load(center_rt); rt1=load(trans_rt); red0=load(center_red); red1=load(trans_red)
        raw_cells={**rt0.get('cells',{}),**rt1.get('cells',{})}
        red_cells={**red0.get('cells',{}),**red1.get('cells',{})}
        state_status=red0.get('status'); admission=red0.get('admission') or {}
        counters={'model_forwards':(rt0.get('model_forwards') or 0)+(rt1.get('model_forwards') or 0),'vision_forwards':(rt0.get('vision_forwards') or 0)+(rt1.get('vision_forwards') or 0),'elapsed_seconds':(rt0.get('elapsed_seconds') or 0)+(rt1.get('elapsed_seconds') or 0),'gpu_seconds':None}
        runtime_paths=[center_rt,trans_rt]; reduced_paths=[center_red,trans_red]
    else:
        rp=pabs(e['runtime_path']); redp=pabs(e['reduced_path'])
        rt0=load(rp); raw_cells=rt0.get('cells',{}); red0=load(redp); red_cells=red0.get('cells',{})
        state_status=red0.get('status'); admission=red0.get('admission') or {}
        counters={'model_forwards':rt0.get('model_forwards') or 0,'vision_forwards':rt0.get('vision_forwards') or 0,'elapsed_seconds':rt0.get('elapsed_seconds') or 0,'gpu_seconds':None}
        runtime_paths=[rp]; reduced_paths=[redp]
    center_rec = bool(red_cells.get('00',{}).get('parse',{}).get('failure_predicate')) if red_cells.get('00') else None
    center_desc=[]
    if red_cells.get('00'):
        rr=red_cells['00']['parse']; center_desc=sorted({str(x.get('description')) for x in (rr.get('exact_runs') or rr.get('near_runs') or [])})
    center_stub={'recurrence':center_rec,'repeat_descriptions':center_desc} if center_rec is not None else None
    sm=source_history(man)
    cells=[]
    for key in man.get('cells',{}):
        if key not in raw_cells or key not in red_cells: continue
        cm=cell_metrics(raw_cells[key],red_cells[key],man['cells'][key],center_stub if key!='00' else None)
        cm['state_id']=sid; cm['model']=model; cm['kind']=kind; cm['source_group']=man.get('source',{}).get('group'); cm['mode']=mode
        cells.append(cm); cell_summaries.append(cm)
    bykey={c['key']:c for c in cells}
    c00=bykey.get('00')
    def contrast(keys):
        out=[]
        for key in keys:
            c=bykey.get(key)
            if not c: continue
            out.append({'cell':key,'recurrence':c['recurrence'],'exact_recurrence':c['exact_recurrence'],'near_recurrence':c['near_recurrence'],'valid_rows':c['valid_rows'],'invalid_rows':c['invalid_rows'],'malformed_rows':c['malformed_rows'],'longest_run':(c['repeat_anchor'] or {}).get('length'),'repeat_description':(c['repeat_anchor'] or {}).get('description'),'mean_center_shift_source':shift((c00 or {}).get('mean_valid_center_source'),c.get('mean_valid_center_source')) if c00 else None,'repeat_anchor_shift_source':shift((c00 or {}).get('repeat_anchor',{}).get('center_source') if c00 and c00.get('repeat_anchor') else None,c.get('repeat_anchor',{}).get('center_source') if c.get('repeat_anchor') else None) if c00 else None,'alternative_recurrence_vs_00':c['alternative_recurrence_vs_00']})
        return out
    primary={'coherent_11':contrast(['11-','11+']),'mismatch_10':contrast(['10-','10+']),'mismatch_01':contrast(['01-','01+'])}
    # state source identities and image/hash binding
    source=man.get('source',{})
    src_image=source.get('source_image') or {}
    state={
      'state_id':sid,'model':model,'kind':kind,'mode':mode,'status':state_status,
      'admission':admission,'source_group':source.get('group'),'image_id':source.get('image_id'),'split':source.get('split'),
      'condition':source.get('condition'),'feedback_selection_id':source.get('feedback_selection_id'),
      'mature_raw':source.get('mature_raw'),'source_image':src_image,
      'mature_panel':source.get('mature_panel'),'source_case_row_id':(source.get('case') or {}).get('row_id'),
      'source_history_geometry':sm,
      'counters':counters,'runtime_paths':[bind(x) for x in runtime_paths],'reduced_paths':[bind(x) for x in reduced_paths],
      'manifest':bind(manifest),'prefix':man.get('prefix'),'geometry':man.get('geometry'),
      'center_00':c00,'contrasts':primary,
      'cell_keys_present':list(bykey),
      'cell_recurrence':{key:bykey[key]['recurrence'] for key in bykey},
      'all_cells':cells,
    }
    state_summaries.append(state)
    source_bindings.append({'state_id':sid,'manifest':bind(manifest),'mature_raw':source.get('mature_raw'),'source_case_row_id':(source.get('case') or {}).get('row_id'),'image_id':source.get('image_id'),'model':model,'kind':kind,'mode':mode})

# counts/tables
states_by=collections.Counter((s['kind'],s['model'],s['mode']) for s in state_summaries)
cell_by=collections.defaultdict(lambda: {'present':0,'recurrence':0,'exact':0,'near':0,'valid_rows':0,'invalid_rows':0,'malformed_rows':0})
for c in cell_summaries:
 x=cell_by[c['key']]; x['present']+=1; x['recurrence']+=int(c['recurrence']); x['exact']+=int(c['exact_recurrence']); x['near']+=int(c['near_recurrence']); x['valid_rows']+=c['valid_rows'] or 0; x['invalid_rows']+=c['invalid_rows'] or 0; x['malformed_rows']+=c['malformed_rows'] or 0
admission_table=collections.Counter((s['kind'],bool(s['admission'].get('admitted')),s['status']) for s in state_summaries)
source_group=collections.defaultdict(lambda:{'states':0,'images':set(),'kinds':collections.Counter(),'models':collections.Counter()})
for s in state_summaries:
 x=source_group[s['source_group']]; x['states']+=1; x['images'].add((s['image_id'],s['model'])); x['kinds'][s['kind']]+=1; x['models'][s['model']]+=1
source_group={k:{'states':v['states'],'image_model_pairs':len(v['images']),'kinds':dict(v['kinds']),'models':dict(v['models'])} for k,v in source_group.items()}
# primary comparison counts among transformed cells in states with all cells
contrast_table={}
for name,keys in [('coherent_11',['11-','11+']),('mismatch_10',['10-','10+']),('mismatch_01',['01-','01+'])]:
 contrast_table[name]={}
 for key in keys:
  avail=[c for c in cell_summaries if c['key']==key]
  contrast_table[name][key]={'present':len(avail),'recurrence':sum(c['recurrence'] for c in avail),'exact':sum(c['exact_recurrence'] for c in avail),'near':sum(c['near_recurrence'] for c in avail),'alternative_recurrence_vs_00':sum(c['alternative_recurrence_vs_00'] for c in avail)}
# cost and artifact bytes
model_forwards=sum(s['counters']['model_forwards'] for s in state_summaries); vision_forwards=sum(s['counters']['vision_forwards'] for s in state_summaries); elapsed=sum(s['counters']['elapsed_seconds'] for s in state_summaries)
runtime_bytes=sum(p.stat().st_size for s in state_summaries for p in [pabs(x['path']) for x in s['runtime_paths']])
reduced_bytes=sum(p.stat().st_size for s in state_summaries for p in [pabs(x['path']) for x in s['reduced_paths']])
# hold/exclusion list
holds=[]
for s in state_summaries:
 if s['status']=='admission_hold' or not s['admission'].get('admitted',False):
  holds.append({'state_id':s['state_id'],'model':s['model'],'kind':s['kind'],'mode':s['mode'],'image_id':s['image_id'],'source_group':s['source_group'],'source_case_row_id':s['source_case_row_id'],'center_valid_rows':(s['center_00'] or {}).get('valid_rows'),'center_complete_rows':(s['center_00'] or {}).get('complete_rows'),'center_invalid_rows':(s['center_00'] or {}).get('invalid_rows'),'center_malformed_rows':(s['center_00'] or {}).get('malformed_rows'),'failure_predicate':(s['center_00'] or {}).get('recurrence'),'known_matches':s['admission'].get('known_matches'),'reason':'00 did not retain frozen failure recurrence predicate; no transformed cells generated'})
# panel binding and final summary
snap=load(SNAP)
result={
 'schema':'recurrence_spatial_source_candidate_result.v1',
 'unit_id':'2026-09-19-recurrence-spatial-source',
 'status':'candidate',
 'decision_boundary':{'lifecycle':'candidate','root_acceptance':'not_done','policy_promotion':'not_authorized','successor_launch':'not_authorized','physical_identity_claim':'limited_by_optional_grounding_witness','estimand':'same-scene common-canvas horizontal spatial displacement crossed with identically transformed history boxes; primary 00 vs 11 mapped back; 10/01 are mismatch diagnostics'},
 'panel':snap['panel'],'readiness':snap['readiness'],'declared_denominator':{'states':45,'failure':21,'proxy':24,'panel_sha256':snap['panel']['sha256']},
 'geometry_rule':{'operation':'lossless copy into fixed common canvas','visual_offsets_px':[0,128,256],'history_offsets_px':[0,128,256],'cells':['00','10-','10+','01-','01+','11-','11+'],'canvas_offset_signs':'negative/positive correspond to 128 and 256 px relative to unchanged 0; source bins inverse-mapped before interpretation','grid_px':32,'fill_rgb':[0,0,0],'interpolation':'none','rounding':'round-half-even','no_clipping':True,'no_scale_change':True},
 'counts':{'states_in_result':len(state_summaries),'states_with_current_runtime':sum(s['mode']!='reuse_pilot' for s in state_summaries),'pilot_reused_states':sum(s['mode']=='reuse_pilot' for s in state_summaries),'free_continuations':len(cell_summaries),'transformed_continuations':sum(c['key']!='00' for c in cell_summaries),'center_continuations':sum(c['key']=='00' for c in cell_summaries),'admission_hold_states':len(holds),'candidate_complete_states':sum(s['status']=='candidate_complete' for s in state_summaries),'state_by_kind_model_mode':{'|'.join(k):v for k,v in states_by.items()},'admission_table':{'|'.join(map(str,k)):v for k,v in admission_table.items()},'cell_table':dict(cell_by),'coherent_mismatch_table':contrast_table},
 'cost':{'model_forwards':model_forwards,'vision_forwards':vision_forwards,'elapsed_seconds_sum':round(elapsed,6),'gpu_seconds':None,'gpu_seconds_status':'producer did not instrument device time; elapsed wall time is retained and is not relabeled GPU time','runtime_json_bytes':runtime_bytes,'reduced_json_bytes':reduced_bytes,'mechanical_pre_fix_attempts':40,'mechanical_pre_fix_attempts_detail':'20 first-batch + 20 rerun-v2 new-cohort attempts failed before native forward on missing image_plan; no scientific outputs were admitted from these attempts','corrected_new_cohort_state_attempts':20,'pilot_model_forwards':sum(s['counters']['model_forwards'] for s in state_summaries if s['mode']=='reuse_pilot'),'pilot_vision_forwards':sum(s['counters']['vision_forwards'] for s in state_summaries if s['mode']=='reuse_pilot')},
 'admission_holds':holds,
 'source_group_table':source_group,
 'source_bindings':source_bindings,
 'state_summaries':state_summaries,
 'cell_summaries':cell_summaries,
 'limits':['00 is the only admission gate; proxy 00 cells are valid-row controls and can themselves show recurrence without becoming failure strata','known-bank matches are optional physical-grounding witnesses; zero or absent witnesses remain limitations, not numerical nulls','10/01 mismatch cells diagnose route sensitivity and are not detection-quality controls','coordinate probability and forced-description window values are full-vocabulary log probabilities; no window-renormalized quality is claimed','all free paths stopped at im_end in this result; no cap/EOS truncation was observed, while EOS competition is retained per cell','source boxes, including invalid source rows, were retained; mapped validity and rounding changes are reported per state/cell','mapped output rows are source-coordinate summaries after inverse affine mapping; a recurrence predicate is numerical and does not establish physical instance identity'],
 'acceptance_commands':[
  'python3 -m py_compile probes/training_set_completion/recurrence_spatial/producer.py probes/training_set_completion/recurrence_spatial/reduce.py probes/training_set_completion/recurrence_spatial/state_entry.py',
  f'python3 -m probes.training_set_completion.recurrence_spatial.reduce --model tied --manifest-path {BASE}/final/manifests/tied-train-169872-failure.json --runtime-path {BASE}/final/runtime/tied-train-169872-failure.json --reduced-path /tmp/recur-spatial-recheck.json',
  f'python3 {BASE}/final/summarize_candidate.py --check-only',
 ],
 'artifact_map':{'result':str(BASE/'final/result.json'),'cost_receipt':str(BASE/'final/cost-receipt.json'),'job_closure':str(BASE/'final/job-closure.json'),'manifest_reconciliation':str(BASE/'final/manifest-reconciliation.json'),'source_snapshot':str(SNAP),'execution_index':str(BASE/'final/execution-index.json'),'reducer':'probes/training_set_completion/recurrence_spatial/reduce.py','producer':'probes/training_set_completion/recurrence_spatial/producer.py'},
}
# JSON-friendly counter values are already ordinary in most places
out=BASE/'final/result.json'; out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
# split cost receipt and state records for easy acceptance
cost={'schema':'recurrence_spatial_source_cost_receipt.v1','unit_id':result['unit_id'],'result_binding':bind(out),'cost':result['cost'],'denominator':result['declared_denominator'],'mechanical_attempts':{'pre_fix_missing_image_plan':40,'corrected_new_cohort':20,'pilot_reuse_cells':14},'remaining_owned_processes':[]}
(BASE/'final/cost-receipt.json').write_text(json.dumps(cost,indent=2)+'\n')
closure={'schema':'recurrence_spatial_source_job_closure.v1','unit_id':result['unit_id'],'status':'closed_for_worker_candidate','process_check':'pgrep found no producer or launch script after final wave','remaining_owned_processes':[],'devices_released':[0,1,2,3,4,5,7],'active_other_lane_devices':[6],'notes':['No completed scientific state was restarted after final corrected rerun; old manifest hash mismatch is reconciled by semantic cell/source checks in manifest-reconciliation.json','A receipt image_plan fields were reused for new-cohort frontend planning and bound image tensor identity was verified in runtime input_identity']}
(BASE/'final/job-closure.json').write_text(json.dumps(closure,indent=2)+'\n')
print(json.dumps({'states':len(state_summaries),'cells':len(cell_summaries),'holds':len(holds),'model_forwards':model_forwards,'vision_forwards':vision_forwards,'elapsed':elapsed,'runtime_bytes':runtime_bytes,'reduced_bytes':reduced_bytes,'output':str(out)},indent=2))
