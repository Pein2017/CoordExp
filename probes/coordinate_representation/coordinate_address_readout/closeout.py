"""Deterministic pilot candidate from closed saved producers, never model work."""
import argparse
import json
from pathlib import Path
from probes.coordinate_representation.coordinate_address_readout.runtime import ROOT, write_once
from probes.coordinate_representation.coordinate_address_readout.reduce_production import reduce_production
from src.artifacts.utf8_json import binding
from src.artifacts.source_provenance import preserve_source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); production=ROOT/'production'
    args.output.mkdir(parents=True,exist_ok=False)
    reduction=reduce_production(manifest_path=production/'manifest-v4.json',cells_root=production/'evaluation',
                                output_path=args.output/'reduction.json',cap=3084)
    jobs=[]
    for process in sorted(ROOT.rglob('process.json')):
        cost=process.with_name('cost.json')
        if not cost.exists(): raise ValueError(f'unclosed producer: {process}')
        jobs.append(dict(process=binding(process),cost=binding(cost),**json.loads(cost.read_text())))
    wall=json.loads((ROOT/'wall-start.json').read_text())
    terminal=max(j['terminal_epoch'] for j in jobs)
    allocated=sum(j['allocated_gpu_seconds'] for j in jobs)
    assert allocated<115200 and terminal-wall['started_epoch']<14400
    captures=[]
    for source in sorted(Path(__file__).parent.glob('*.py')):
        captured=preserve_source(source,run_root=args.output,relative_name=source.resolve().relative_to(Path.cwd()))
        captures.append(dict(current=binding(source),capture=binding(captured)))
    training=json.loads((production/'training-complete.json').read_text())
    cells=[]
    for path in sorted((production/'evaluation').glob('*/cells/*.json')):
        cell=json.loads(path.read_text()); free=cell.get('free',{})
        cells.append(dict(binding=binding(path),row_id=cell['row_id'],condition=cell['condition'],kind=cell['kind'],
            status=cell['status'],seconds=cell['seconds'],tokens=len(cell.get('token_ids',free.get('token_ids',[]))),
            stop_reason=cell.get('stop_reason',free.get('stop_reason'))))
    run_bindings=[binding(p) for p in sorted(production.rglob('*.json')) if
                  not p.is_relative_to(args.output) and '/cells/' not in str(p)]
    result=dict(schema='address_readout_pilot.final_candidate.v1',status='candidate_not_lead_accepted',
        production_manifest=binding(production/'manifest-v4.json'),reduction=binding(args.output/'reduction.json'),
        qualification=binding(ROOT/'qualification/candidate-v2/manifest.json'),
        source_captures=captures,run_bindings=run_bindings,cells=cells,jobs=jobs,
        planned=dict(native_cells=190,calibration_cells=160,lane_b_cells=0,training_fits=4,updates_per_fit=256),
        observed=dict(native=reduction['native']['observed'],calibration=reduction['calibration']['observed']),
        training=dict(summary=binding(production/'training-complete.json'),updates=training['updates'],coordinate_tokens=training['coordinate_tokens']),
        budget=dict(wall_start_epoch=wall['started_epoch'],last_producer_terminal_epoch=terminal,wall_seconds=terminal-wall['started_epoch'],
            allocated_gpu_seconds=allocated,model_forwards=sum(j['model_forwards'] for j in jobs),vision_forwards=sum(j['vision_forwards'] for j in jobs),
            per_job_artifact_bytes=sum(j['artifact_bytes'] for j in jobs),limits=dict(wall_seconds=14400,allocated_gpu_seconds=115200)),
        lane_b=dict(status='source_inapplicable_HOLD',cells=0,
            availability=binding(ROOT/'qualification/blocker-v1/manifest.json'),
            lead_readback=binding(ROOT/'qualification/lead-lane-b-readback-v1.json')),
        authority=[binding(Path('research/experiments/2026-09-22-address-readout-pilot')/name) for name in ('unit.md','lead-ruling-01.md','lead-ruling-02.md')],
        limitations=['All owner coverage/revisits/gains/losses are annotation IoU proxies, not new physical adjudication.',
                    'Unmatched predictions are UNKNOWN, not false objects.',
                    'Development six-image pool and fresh 32-image cohort remain separate.',
                    'Fixed roll-1 address permutation is not information-free.',
                    'No recurrence-prefix production cells and no Lane B contrast executed.'],
        replay_commands=[f'python -B -m probes.coordinate_representation.coordinate_address_readout.reduce_production --manifest {production}/manifest-v4.json --cells-root {production}/evaluation --output /ABS/FRESH-reduction.json --cap 3084',
          'python -B scripts/research/check_research_knowledge.py check',
          'python -B -m src.artifacts.output_layout --root /data/CoordExp/outputs --root /data/CoordExp/.worktrees/research-probes/outputs',
          'git diff --check'])
    print(write_once(args.output/'manifest.json',result))

if __name__=='__main__': main()
