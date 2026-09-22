"""Deterministic reduction of saved qualification evidence; never executes a model."""
import argparse
import json
from pathlib import Path
from probes.training_set_completion.artifacts import binding,canonical


def reduce(root):
    qroot=root/'qualification'
    read=lambda p:json.loads(p.read_text())
    train=qroot/'model-attempt04'
    reload=qroot/'reload-attempt05'
    launch=read(qroot/'launch-v2.json')
    fit=read(train/'fit.json')
    gates=read(train/'qualification.json')
    restored=read(reload/'reload.json')
    throughput=read(train/'throughput.json')
    closure=read(qroot/'closure-v1.json')
    trained=read(train/'trained-generation.json')
    assert trained['token_ids']==restored['greedy']['token_ids']
    assert gates['checkpoint']==binding(train/'sidecar.pt')==restored['checkpoint']
    assert fit['frozen_before']==fit['frozen_after']
    assert gates['zero_gain_max']==gates['causal_suffix_max']==0
    assert fit['final_loss'] < fit['initial_QK_at_final_gain_loss']
    assert len(fit['losses'])==16 and all(value>0 for value in fit['parameter_deltas'].values())
    paths=[Path(ref['path']) for ref in closure['cost_receipts']]
    assert [binding(p) for p in paths] == closure['cost_receipts']
    costs=[read(p) for p in paths]
    wall=read(root/'wall-start.json')
    seconds_per_token=throughput['greedy_seconds']/throughput['greedy_tokens']
    inputs=launch['forecast_inputs']
    admission=read(Path(launch['admission']['path']))
    train_cases=[json.loads(line) for line in Path(admission['cohorts']['train']['cases_path']).read_text().splitlines()]
    cache_bytes=[row['image_plan']['merged_visual_tokens']*2048*4 + len(row['input_record']['objects'])*4*((1000+2048+1)*4+16) for row in train_cases]
    native_cells=inputs['native_conditions']*(inputs['evaluation_images']+inputs['development_max_images'])
    native_seconds=native_cells*inputs['native_cap']*seconds_per_token
    training_seconds=4*256*8*throughput['sidecar_forward_backward_seconds']
    features_seconds=128*throughput['feature_forward_seconds']
    # A cost envelope, not new scientific cases: exact development prefixes remain lead-owned.
    diagnostic_seconds=5*32*8*seconds_per_token
    prefix_seconds=5*11*64*seconds_per_token
    fixed_seconds=training_seconds+features_seconds+diagnostic_seconds+prefix_seconds
    observed=sum(c['allocated_gpu_seconds'] for c in costs)
    assert observed < 115200 and closure['as_of_epoch']-wall['started_epoch'] < 14400
    base_seconds=native_seconds+fixed_seconds
    two_x_seconds=2*native_seconds+fixed_seconds
    obj=next(o for o in launch['diagnostic_case']['input_record']['objects'] if o['desc']==launch['diagnostic_referent'])
    truth=[int(x[8:-2]) for x in obj['bbox_2d']]
    generated=restored['free_box']['parsed']['predictions'][0]['coord_bins']
    assert len(generated)==4 and len(truth)==4
    evidence_files=[qroot/'launch-v2.json',train/'qualification.json',train/'fit.json',train/'source-parity.json',
        train/'causal-alignment.json',train/'slot-isolation.json',train/'geometry.json',train/'throughput.json',
        train/'sources.json',reload/'reload.json',reload/'sources.json',qroot/'alignment-attempt03/alignment.json',
        qroot/'closure-v1.json',qroot/'checks-v1.json']
    evidence_files+=paths
    return dict(schema='address_readout_pilot.qualification_candidate.v1',status='candidate',lead_accepted=False,
        admission=launch['admission'],launch=binding(qroot/'launch-v2.json'),checkpoint=binding(train/'sidecar.pt'),
        evidence=[binding(p) for p in evidence_files],
        gates=dict(zero_gain_full_vocab_max=gates['zero_gain_max'],causal_suffix_max=gates['causal_suffix_max'],
            family_logsumexp_max=gates['family_logsumexp_max'],cached_full_vocab_max=gates['cached_replay_full_vocab_max'],
            cached_logprob_max=gates['cached_replay_logprob_max'],frozen_original_equal=True,
            exact_reload_greedy=True,free_complete_box=True,bank_isolation=True,
            compact_alignment=read(train/'causal-alignment.json'),geometry=read(train/'geometry.json')),
        learning=dict(parameter_count=gates['parameter_count'],updates=fit['updates'],coordinate_tokens_per_update=fit['tokens_per_update'],
            first_ce=fit['losses'][0],final_ce=fit['final_loss'],initial_QK_at_final_gain_ce=fit['initial_QK_at_final_gain_loss'],
            first_gradients=fit['gradients'][0],second_gradients=fit['gradients'][1],deltas=fit['parameter_deltas'],
            initial_bin_mae=fit['initial_bin_mae'],final_bin_mae=fit['final_bin_mae'],ce_accounting_max=fit['ce_accounting_max']),
        free_coordinate_diagnostic=dict(case=launch['diagnostic_case']['row_id'],description=launch['diagnostic_referent'],
            supplied_coordinate_count=0,predicted_bins=generated,target_bins=truth,
            absolute_errors_norm1000=[abs(a-b)/1000 for a,b in zip(generated,truth)],
            mean_absolute_error_norm1000=sum(abs(a-b) for a,b in zip(generated,truth))/4000,
            scope='one technical reload diagnostic; no baseline comparison or calibration-quality claim',
            cap=restored['free_box']['cap'],stop_reason=restored['free_box']['stop_reason']),
        costs=dict(allocated_gpu_seconds=observed,allocated_gpu_hours=observed/3600,
            model_forwards=sum(c['model_forwards'] for c in costs),vision_forwards=sum(c['vision_forwards'] for c in costs),
            optimizer_updates=16,supervised_coordinate_tokens_in_updates=16*fit['tokens_per_update'],
            throughput_coordinate_tokens_without_update=throughput['coordinate_tokens'],
            model_wall_start=wall['started_epoch'],last_model_job_terminal=max(c['terminal_epoch'] for c in costs),
            closure_as_of=closure['as_of_epoch'],remaining_wall_seconds_at_closure=14400-(closure['as_of_epoch']-wall['started_epoch']),
            remaining_allocated_gpu_seconds=115200-observed,package_wall_clock_continues=True,
            artifact_bytes_at_closure=closure['artifact_bytes']),
        forecast=dict(measurement=throughput,proposal_only=True,native_cells=native_cells,cap_per_cell=inputs['native_cap'],
            measured_seconds_per_token=seconds_per_token,all_cells_hit_cap_native_gpu_seconds=native_seconds,
            conservative_largest_case_feature_gpu_seconds=features_seconds,largest_case_training_gpu_seconds=training_seconds,
            unique_referent_diagnostic_gpu_seconds=diagnostic_seconds,development_prefix_cost_envelope_gpu_seconds=prefix_seconds,
            base_total_gpu_hours=base_seconds/3600,base_ideal_eight_gpu_wall_seconds=base_seconds/8,
            two_x_native_decode_gpu_hours=two_x_seconds/3600,two_x_ideal_eight_gpu_wall_seconds=two_x_seconds/8,
            two_x_exceeds_remaining_gpu_budget=two_x_seconds>115200-observed,
            projected_cache_bytes_from_admitted_shapes=sum(cache_bytes),projected_max_case_cache_bytes=max(cache_bytes),
            caveat='28 emitted tokens measured; 3084-token decoder throughput not measured. All-cap projection is a sensitivity, not a guaranteed bound. Shared budget guards must stop incomplete cells as HOLD; never silently reduce denominator.'),
        failures=[dict(receipt=binding(p),error=c['error']) for p,c in zip(paths,costs) if c['status']=='failed'],
        jobs=closure['jobs'],lane_b=dict(status='source_inapplicable_HOLD',scientific_cells=0),
        boundary='Technical Lane A candidate only. Broad training and scientific evaluation remain held; no physical-owner or burst-remedy conclusion.',
        producer=binding(Path(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=reduce(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:f.write(canonical(result))
    print(json.dumps({'path':str(a.output),'sha256':binding(a.output)['sha256'],'status':result['status']}))

if __name__=='__main__':main()
