import json
from pathlib import Path
base=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration');out=Path(__file__).parent
def read(rel):return json.loads((base/rel).read_text())
a=read('2026-09-19-recurrence-distribution-census/result.json')['cost']
bad=read('2026-09-19-recurrence-spatial-source/final/cost-receipt.json')['cost']
b=read('2026-09-19-recurrence-spatial-source/final/broad-v1/cost-receipt.json')
ledger=read('2026-09-19-recurrence-spatial-source/final/attempt-ledger.json')
extra=next(x['model_forwards'] for x in ledger['attempts'] if x['id']=='candidate-v1-forwards-outside-final-reuse')
c=read('2026-09-19-recurrence-conditional-mass/run-manifest.json')['cost'];cq=read('2026-09-19-recurrence-conditional-mass/qualification-attempts.json')['attempts']
d=read('2026-09-19-coordinate-input-continuity/runtime/closure.json')
counts={'A':a['model_forwards'],'B_invalid':bad['model_forwards']+extra,'B_corrected_including_all_gate_attempts':b['disjoint_corrected_attempt_sum']['model_forwards'],'C_scientific':c['model_forwards'],'C_all_qualification':sum(x.get('model_forwards',0) for x in cq),'D_including_failure':d['total_model_forwards_observed']}
assert counts=={'A':8971,'B_invalid':45726,'B_corrected_including_all_gate_attempts':42606,'C_scientific':7290,'C_all_qualification':13,'D_including_failure':97}
prior=read('2026-09-19-recurrence-spatial-source/final/correction/correction-receipt.json')['historical_occupancy_bound']
corrected_bound=70*60+8*21*60
receipt={'status':'candidate_cost_accounting','model_forward_invocations':counts,'total_model_forward_invocations':sum(counts.values()),'no_training':True,'corrected_B_gate_lineage_forwards':10,'cost_scope':'batch model calls, not per-image/token or vision-call equivalents','observed_elapsed_seconds':{'A_group_sum':a['gpu_seconds'],'B_invalid_final_producer_sum':bad['elapsed_seconds_sum'],'B_corrected_broad_run':b['broad']['run_wall_seconds'],'B_corrected_broad_device_interval_sum':sum(x['wall_seconds'] for x in b['broad']['device_intervals']),'C_scientific_sum':c['elapsed_seconds'],'D_all_jobs_sum':d['cost']['total_elapsed_seconds']},'elapsed_note':'Producer elapsed/device-interval values are not measured GPU kernel utilization; A legacy gpu_seconds field is treated as elapsed only.','allocated_gpu_hours_conservative_upper_bound':(prior['prior_plus_reserved_c_gpu_seconds']+corrected_bound)/3600,'bound_components':{'prior_A_D_invalid_B_enclosing_seconds':prior['prior_upper_bound_gpu_seconds'],'C_entire_lane_upper_reserve_seconds':4*3600,'corrected_B_one_device_0950_to1100_UTC_seconds':70*60,'corrected_B_eight_devices_1100_to1121_UTC_seconds':8*21*60},'bound_basis':'Prior source-bound ownership intervals plus full C4h reserve. Corrected B actual pilot begins09:51:39; all gates use cuda:0; eight-device release11:02:46 and final exit11:20:38. Rounded outward reservation intervals include loading, idle gaps, failed gates and science. Missing timestamps are not zero.','released_total_ceiling_gpu_hours':24,'within_ceiling':True,'cleanup':{'deleted_paths':[],'deleted_bytes':0}}
assert receipt['allocated_gpu_hours_conservative_upper_bound']<24
p=out/'cost.json'
if p.exists():assert json.loads(p.read_text())==receipt
else:p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'calls':sum(counts.values()),'conservative_GPU_hours':receipt['allocated_gpu_hours_conservative_upper_bound']}))
