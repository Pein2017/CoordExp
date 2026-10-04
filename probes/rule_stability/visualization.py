"""Assemble saved native greedy JSONs into an explicitly bound visualization JSONL.

This CPU-only command preserves the original generated record. Full labels are
provided separately to the visualization reader; no scored artifacts are created.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.errors import ArtifactContractError
from src.vis.rollout import FORMAT, PROVENANCE, _json_file, _require, sha256_file, validate_native_record


def assemble_rollouts(run_dir: str | Path, *, version: int, image_ids: list[int], output: str | Path) -> Path:
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    _require(isinstance(version, int) and not isinstance(version, bool) and version >= 0, 'rollout version must be a nonnegative integer')
    _require(image_ids and all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in image_ids) and len(set(image_ids)) == len(image_ids),
             'select distinct explicit image IDs for native rollout assembly')
    _require(not output.exists(), 'refuse existing native visualization JSONL output', 'vis.rollout_output_exists', path=str(output))
    records = []
    for image_id in image_ids:
        matches = list(run_dir.glob(f'rank-*/version-{version}/greedy-{image_id}.json'))
        _require(len(matches) == 1, 'native rollout assembly requires exactly one rank/version/image record', 'vis.rollout_source_ambiguous', image_id=image_id, matches=[str(p) for p in matches])
        source = matches[0]
        record = _json_file(source)
        _require(isinstance(record, dict), 'native rollout source must be an object')
        validate_native_record(record)
        producer = record['producer']
        rank_text = source.parent.parent.name.removeprefix('rank-')
        _require(rank_text.isdecimal() and record.get('generation_rank') == int(rank_text) and record['image_id'] == image_id and
                 producer.get('channel') == 'greedy' and producer.get('version', version) == version and
                 record['request_id'] == f"rule-stability:{producer.get('arm')}:{version}:greedy:{image_id}",
                 'native raw record differs from selected rank/version/image identity', 'vis.rollout_source_identity')
        _require(PROVENANCE not in record, 'native source already contains derivative provenance', 'vis.rollout_source_identity')
        analysis_path = source.with_name(f'greedy-{image_id}-analysis.json')
        analysis = _json_file(analysis_path)
        _require(isinstance(analysis, dict), 'native companion analysis must be an object', 'vis.rollout_analysis')
        records.append(dict(record, **{PROVENANCE:dict(format=FORMAT, coordinate_space='norm1000',
                       raw_source=dict(path=str(source), sha256=sha256_file(source)),
                       analysis_source=dict(path=str(analysis_path), sha256=sha256_file(analysis_path)))}))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open('x', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    except FileExistsError as exc:
        raise ArtifactContractError('refuse existing native visualization JSONL output', code='vis.rollout_output_exists', context={'path':str(output)}, cause=exc) from exc
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Saved native rule-stability run directory (read only).')
    parser.add_argument('--version', type=int, required=True, help='Exact saved greedy version.')
    parser.add_argument('--image-id', type=int, action='append', required=True, help='Selected image ID; repeat in output order.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh rollout JSONL destination; collisions fail.')
    args = parser.parse_args(argv)
    try:
        path = assemble_rollouts(args.run_dir, version=args.version, image_ids=args.image_id, output=args.output)
    except ArtifactContractError as exc:
        parser.exit(2, f'{exc.code}: {exc}\n')
    print(json.dumps(dict(output=str(path), version=args.version, image_ids=args.image_id, model_calls=0, tokenizer_calls=0)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
