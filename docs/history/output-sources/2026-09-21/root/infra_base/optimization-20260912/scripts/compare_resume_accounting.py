"""Task-local exact continuation accounting and native DDP bucket witness."""
import argparse
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in (path / 'logging.jsonl').read_text().splitlines()]


def project(row):
    names = {
        'split', 'step', 'finite_status', 'acc_top1', 'acc_top5',
        'accuracy_stats', 'example_count', 'pack_count', 'micro_step_count',
        'scheduler_step_count', 'non_finite_fields',
    }
    prefixes = ('loss/', 'count/', 'finite/', 'grad_norm/', 'lr/', 'optimizer_')
    return {k: v for k, v in row.items() if k in names or k.startswith(prefixes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parent', type=Path)
    parser.add_argument('child', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    with args.receipt.open('x') as output:
        parent, child = rows(args.parent), rows(args.child)
        expected = [project(r) for r in parent if r['step'] in (3, 4)]
        actual = [project(r) for r in child]
        expected_steps = [('train', 3), ('train', 4), ('eval', 4)]
        structure_equal = [(r['split'], r['step']) for r in expected] == expected_steps
        structure_equal &= [(r['split'], r['step']) for r in actual] == expected_steps
        differences = []
        for index, (a, b) in enumerate(zip(expected, actual)):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b or type(a[key]) is not type(b[key]) or a[key] != b[key]:
                    differences.append({'row': index, 'key': key, 'parent': a.get(key), 'child': b.get(key)})
        bucket_rows = [
            {'arm': arm, 'step': r['step'], 'metadata': r.get('ddp_bucket_metadata_rank0')}
            for arm, records in [('parent', parent), ('child', child)]
            for r in records if r['split'] == 'train'
        ]
        bucket_valid = len(bucket_rows) == 6 and all(
            isinstance(r['metadata'], dict)
            and r['metadata'].get('find_unused_parameters') == 1
            and (
                r['metadata'].get('has_rebuilt_buckets') == 0
                or ('has_rebuilt_buckets' not in r['metadata']
                    and (r['arm'], r['step']) in [('parent', 1), ('child', 3)])
            )
            and bool(r['metadata'].get('bucket_sizes'))
            and not r['metadata'].get('rebuilt_bucket_sizes')
            for r in bucket_rows
        )
        if bucket_valid:
            bucket_valid = len({r['metadata']['bucket_sizes'] for r in bucket_rows}) == 1
        result = {
            'status': 'equal' if structure_equal and not differences and bucket_valid else 'mismatch',
            'parent': str(args.parent), 'child': str(args.child),
            'comparison': 'Exact selected accounting values, including loss, denominators, gradient norms, LR, update and scheduler counters.',
            'expected_structure': expected_steps, 'structure_equal': structure_equal,
            'differences': differences, 'bucket_valid': bucket_valid,
            'bucket_observation_scope': 'Native rebuild status may be absent at the first completed update in each process; every subsequent update must explicitly report zero. Missing initial status remains missing in the evidence.',
            'rank0_bucket_rows': bucket_rows,
        }
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({'status': result['status'], 'difference_count': len(differences), 'bucket_valid': bucket_valid, 'receipt': str(args.receipt)}))
    return 0 if result['status'] == 'equal' else 1


if __name__ == '__main__':
    raise SystemExit(main())
