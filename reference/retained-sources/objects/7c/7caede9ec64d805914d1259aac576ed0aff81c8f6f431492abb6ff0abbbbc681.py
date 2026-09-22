"""Bind the existing admission to a bounded Lane A qualification, without resampling."""
import argparse
import copy
import json
from pathlib import Path
from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_address_readout.runtime import ROOT, write_once
from probes.training_set_completion.coordinate_address_readout.bridge import address_permutation

ADMISSION = ROOT/'selection-v3/selection/manifest.json'
EXPECTED = 'b9a4f088466c1874ffef15380ff4fab7931b752b4208cbb2abba0de9836834d7'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert args.output.resolve().is_relative_to(ROOT.resolve())
    assert binding(ADMISSION)['sha256'] == EXPECTED
    admission=json.loads(ADMISSION.read_text())
    rows={name:[json.loads(line) for line in Path(cohort['cases_path']).read_text().splitlines()]
          for name,cohort in admission['cohorts'].items()}
    first=rows['train'][0]
    largest=max(rows['train'],key=lambda row:(len(row['input_record']['objects']),row['row_id']))
    diagnostic=rows['calibration'][0]
    referent=next(x['referent_description'] for x in admission['diagnostic_cases'] if x['row_id']==diagnostic['row_id'])
    configs=copy.deepcopy(admission['configs'])
    for config in configs.values():
        config['run'].update(artifact_root=str(ROOT),name='address-readout-pilot',output_dir=None)
    inputs=[ADMISSION,*[Path(c['cases_path']) for c in admission['cohorts'].values()]]
    inputs += [Path(x['path']) for x in admission['sources']['source_identity']['bound_files']]
    inputs += list(Path(configs['train']['model']['base_model']).glob('*safetensors*'))
    inputs += [Path(row['image_path']) for row in [first,largest,diagnostic]]
    shapes=sorted({tuple(row['image_plan']['observed_image_grid_thw'][1:]) for rr in rows.values() for row in rr})
    permutations={f'{h//2}x{w//2}':address_permutation(h//2,w//2).tolist() for h,w in shapes}
    result=dict(schema='address_readout_pilot.qualification_launch.v1',status='frozen_for_qualification_only',
        admission=binding(ADMISSION),lead_ruling=binding(Path('research/experiments/2026-09-22-address-readout-pilot/lead-ruling-01.md')),
        input_bindings=[binding(p) for p in dict.fromkeys(inputs)],model_config=configs['train'],calibration_config=configs['calibration'],
        qualification_cases=[first,largest],diagnostic_case=diagnostic,diagnostic_referent=referent,
        training=dict(status='proposal_not_lead_accepted',seeds=[1729,2718],optimizer='AdamW',learning_rate=.001,
                      betas=[.9,.999],eps=1e-8,weight_decay=0,batch_images=8,updates=256,
                      final_checkpoint='fixed final update only',data_order='seeded shuffle of admitted training IDs, cycle each epoch',
                      loss='sum positive coordinate full-vocabulary CE / global positive coordinate token count',
                      dtype='fp32',attention='sdpa',tf32=False,gradient_accumulation='single-image sequential within 8-image update'),
        qualification=dict(updates=16,training_rows=2,greedy_cap=64,free_coordinate_cap=8,
                           optimizer_seed=1729,primary_image_selection='first frozen training row',
                           throughput_image_selection='most source positive objects, row_id tie break',
                           max_model_forwards_per_process=600,max_seconds_per_process=2400,
                           tolerances=dict(causal_hidden=1e-6,causal_logits=1e-5,compact_logits=2e-5,
                                           family_logsumexp=2e-5,cached_full_logits=2e-4)),
        address_permutations=permutations,lane_b=dict(status='source_inapplicable_HOLD',scientific_cells=0),
        budget=dict(model_wall_seconds=14400,allocated_gpu_seconds=115200,devices=list(range(8))),
        forecast_inputs=dict(train_images=128,calibration_images=32,evaluation_images=32,development_max_images=6,
                             train_positive_boxes=sum(len(r['input_record']['objects']) for r in rows['train']),
                             trained_arm_seed_count=4,native_conditions=5,native_cap=3084),
        producer=binding(Path(__file__)))
    print(write_once(args.output,result))

if __name__=='__main__':
    main()
