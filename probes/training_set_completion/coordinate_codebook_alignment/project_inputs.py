"""Project frozen admission metadata into the strict relative-path data schema."""
import argparse
import json
import os
from pathlib import Path

from probes.training_set_completion.artifacts import binding
from src.data.examples import raw_example_from_jsonl_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name in ('fit', 'monitor', 'qualification'):
        source = args.selection / f'{name}.coord.jsonl'
        target = args.output / source.name
        rows = [json.loads(s) for s in source.read_text().splitlines()]
        projected = []
        for index, row in enumerate(rows):
            clean = {k: v for k, v in row.items() if k != '_admission'}
            original = Path(row['images'][0]).resolve(strict=True)
            clean['images'] = [os.path.relpath(original, target.parent)]
            assert (target.parent / clean['images'][0]).resolve(strict=True) == original
            assert {k: v for k, v in row.items() if k not in ('_admission', 'images')} == {
                k: v for k, v in clean.items() if k != 'images'}
            raw_example_from_jsonl_row(clean, jsonl_path=target, row_number=index + 1,
                                      raw_line=json.dumps(clean))
            projected.append(clean)
        target.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in projected))
        records.append({'source': binding(source), 'runtime': binding(target), 'count': len(rows),
                        'removed_fields': ['_admission'], 'rebased_fields': ['images'],
                        'resolved_images_identical': True, 'all_other_fields_identical': True})
    (args.output / 'projection.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
