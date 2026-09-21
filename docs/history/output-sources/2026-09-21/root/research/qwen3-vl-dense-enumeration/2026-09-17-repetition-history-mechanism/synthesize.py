"""Build the compact candidate from the frozen saved-output reduction."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
r=read(R/'reduction.json');scores=read(R/'scores/309264-smoke/scores.json');alt=read(R/'scores/309264-alt/scores.json');cells=r['cells']
a={}
for name,s in scores.items():
 c=cells['A/309264/'+name]
 a[name]=dict(row_logprobs={k:v['sum_logprob'] for k,v in s['candidates'].items()},forks=s['forks'],free_owner_ids=c['known_accounting']['free'],known_accounting=c['known_accounting'],free_burden=c['after_release']['burden'],free_rows=c['after_release']['complete_rows'])
effects={}
for owner in ['A','B','C','A_alt','EOS']:
 x={k:a[k]['row_logprobs'][owner] for k in ['AA','AB','BA','BB']}
 effects[owner]=dict(first_slot_A_minus_B=((x['AA']-x['BA'])+(x['AB']-x['BB']))/2,second_slot_A_minus_B=((x['AA']-x['AB'])+(x['BA']-x['BB']))/2,interaction=x['AA']-x['AB']-x['BA']+x['BB'],AB_minus_BA=x['AB']-x['BA'])
b={}
for key,c in cells.items():
 if not key.startswith('B/') or key.endswith('_replay'):continue
 b[key]=dict(full_known=c['full']['matches']['covered_owner_ids'],full_burden=c['full']['burden'],post_release_burden=c['after_release']['burden'],release=c['release'],winner_changes=c['winner_changes'],known_accounting=c['known_accounting'],per_trajectory_known_accounting=c['per_trajectory_known_accounting'])
replays=[]
for key,c in cells.items():
 if not key.endswith('_replay'):continue
 original=cells[key.removesuffix('_replay')];left=read(Path(c['raw']['path']));right=read(Path(original['raw']['path']));assert left['rows']==right['rows'];replays.append(dict(replay=key,original=key.removesuffix('_replay'),all_native_rows_exact=True))
assert len(replays)==6
assert cells['A/309264/AA_alt']['post_release_token_ids']==cells['A/309264/AA']['post_release_token_ids']
sampling=read(R/'sampling-summary.json');assert sampling.get('status')=='complete'
v=read(R/'sampling-verification.json');decisions=[d for check in v['checks'] for d in check['decisions']];changed=[d for d in decisions if d['changed']]
assert all(151670<=d['raw_winner']<=152669 and 151670<=d['selected']<=152669 for d in changed)
t={x['condition']:x for x in sampling['trajectories']}
sampling['captured_sampling_decisions']=dict(total=len(decisions),changed=len(changed),all_changed_family_pairs='coordinate-to-coordinate',positive_pulse_box=t['T0.7-seed24']['pulse']['complete_rows'][0]['box'],failed_nonzero_x1_examples=[t[k]['pulse']['complete_rows'][0]['box'] for k in ['T0.7-seed19','T0.7-seed26']],scope='Descriptive supplied-row contrasts; not a slot-isolated intervention')
out=dict(schema='repetition_history.result.v1',status='candidate',acceptance='unreviewed; cross-study lead owns acceptance',
 stage_a=dict(case_image_id=309264,conditions=a,relative_replacement_effects=effects,alternate_expression_scores=alt,alternate_free_tokens_identical=True,claim='Relative exposure/placement effects, not isolated count. Native vs synthetic references also change recent route and covered history.'),
 stage_b=dict(cells=b,mechanical_replays=replays,claim='Useful continuation and output debt are distinct; same sampled or norm-produced owners are not autonomous gains.'),
 stage_c=sampling,
 physical_review=bind(R/'physical-review.json'),
 evidence={name:bind(R/name) for name in ['reduction.json','independent-reduction-check.json','runtime-verification.json','sampling-verification.json','sampling-manifest.json','runtime-manifest.json','stage-a-panel.json','scores/309264-smoke/scores.json','scores/309264-alt/scores.json','anchor-summary-correction.json','cost.json','knowledge-check.json','source-snapshots.json','ARTIFACTS.md']},
 execution=dict(native_batch_runs=47,scientific_or_control_runs_excluding_token_replay=41,exact_token_replay_runs=6,score_forwards=35,failed_model_cells=0,stage_a_anchor_hold_cases=[386313],stage_c_executed=24,stage_c_failed=0),
 scope=dict(training=False,parameters_updated=False,annotation_changes=False,visual_census=False,kv_attention_archive=False),
 hold=['Same-class A/B book history anchors not admitted; bird history contrast executed instead.','Exact physical identity/extent in right white bird and feeder region remains partial; unknown is not false positive.','Book native-future shelf preservation and matching swaps remain physical HOLD.'],
 scientific_disposition='Mixed selected-case result: history replacement changes scores/routes without isolated count credit; finite norm pulses relapse; one of eight T0.7 sampled pulses yields autonomous known/credible-owner continuation, versus zero of eight at T0.1 and T0.3. Physical debt and wide seed uncertainty prevent policy promotion or empty-prefix improvement claims.')
(R/'result.json').write_text(json.dumps(out,indent=2)+'\n')
