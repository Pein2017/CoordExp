#!/usr/bin/env python3
"""Validate lead-owned task receipts and summarize comparable delegation routes.

Standard library only. Inputs are read-only; reports are disposable derivatives.
No runtime mutation, model ranking, pricing inference, or automatic policy edits.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import tempfile

GROUP_FIELDS = ('origin', 'task_class', 'comparison_key', 'brief_style', 'topology', 'risk', 'verifier')
COST_FIELDS = ('worker_usd', 'lead_usd', 'runtime_usd')
ROW_FIELDS = {'schema_version', 'task_id', 'revision', 'recorded_at', 'outcome', 'evidence', 'attempts', *GROUP_FIELDS}
ATTEMPT_FIELDS = {'attempt_id', 'model', 'effort', 'fork', 'outcome', 'cost_source', 'failure_kind', *COST_FIELDS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def timestamp(value):
    require(text(value), 'recorded_at must be an ISO timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'recorded_at needs a timezone')
    return parsed


def validate(row):
    require(isinstance(row, dict) and set(row) == ROW_FIELDS, 'unexpected or missing task fields')
    require(type(row['schema_version']) is int and row['schema_version'] == 1, 'unsupported schema_version')
    require(type(row['revision']) is int and row['revision'] > 0, 'revision must be a positive integer')
    for key in ('task_id', 'task_class', 'comparison_key', 'brief_style', 'topology'):
        require(text(row[key]), f'{key} must be nonempty text')
    timestamp(row['recorded_at'])
    require(row['origin'] in ('production', 'benchmark'), 'invalid origin')
    require(row['risk'] in ('low', 'medium', 'high'), 'invalid risk')
    require(row['verifier'] in ('deterministic', 'integration', 'review'), 'invalid verifier')
    require(row['outcome'] in ('accepted', 'failed', 'pending', 'invalidated'), 'invalid task outcome')
    require(isinstance(row['evidence'], list) and row['evidence'] and all(text(v) for v in row['evidence']), 'evidence needs source paths or references')
    require(isinstance(row['attempts'], list) and row['attempts'], 'attempts must be nonempty')
    ids = set()
    for a in row['attempts']:
        require(isinstance(a, dict) and set(a) == ATTEMPT_FIELDS, 'unexpected or missing attempt fields')
        for key in ('attempt_id', 'model', 'effort', 'fork', 'cost_source'):
            require(text(a[key]), f'{key} must be nonempty text')
        require(a['attempt_id'] not in ids, 'duplicate attempt_id inside task')
        ids.add(a['attempt_id'])
        require(a['effort'] in ('low', 'medium', 'high', 'xhigh', 'max'), 'invalid effort')
        require(a['fork'] in ('none', 'all') or (a['fork'].isascii() and a['fork'].isdigit() and int(a['fork']) > 0), 'invalid fork')
        require(a['outcome'] in ('accepted', 'rework', 'failed', 'escalated'), 'invalid attempt outcome')
        require(a['failure_kind'] in ('none', 'implementation', 'reasoning', 'brief', 'environment', 'verifier', 'unknown'), 'invalid failure_kind')
        for key in COST_FIELDS:
            v = a[key]
            require(v is None or (type(v) in (int, float) and math.isfinite(v) and v >= 0), f'{key} must be finite nonnegative money or null')
    if row['outcome'] == 'accepted':
        require(row['attempts'][-1]['outcome'] == 'accepted', 'accepted task needs final accepted attempt')
        require(all(a['outcome'] != 'accepted' for a in row['attempts'][:-1]), 'earlier attempts cannot also be accepted')
    if row['outcome'] == 'failed':
        require(all(a['outcome'] != 'accepted' for a in row['attempts']), 'failed task cannot contain an accepted attempt; use invalidated')
    return row


def reject_constant(value):
    raise ValueError(f'nonfinite JSON constant {value}')


def discover(inputs):
    paths = set()
    for raw in inputs:
        path = Path(raw).resolve()
        require(path.exists(), f'input does not exist: {path}')
        if path.is_dir():
            paths.update(p.resolve() for p in path.rglob('delegation-outcomes.jsonl') if p.is_file())
        else:
            require(path.is_file(), f'not a regular input file: {path}')
            paths.add(path)
    require(paths, 'no delegation-outcomes.jsonl records found in selected roots')
    return sorted(paths)


def load(inputs):
    paths = discover(inputs)
    versions = {}
    sources = defaultdict(set)
    owners = {}
    duplicates = 0
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = validate(json.loads(line, parse_constant=reject_constant))
                key = (row['task_id'], row['revision'])
                if key in versions:
                    require(versions[key] == row, f'conflicting task revision {key}')
                    duplicates += 1
                else:
                    versions[key] = row
                sources[key].add(f'{path}:{number}')
                for a in row['attempts']:
                    owner = owners.setdefault(a['attempt_id'], row['task_id'])
                    require(owner == row['task_id'], f'attempt {a["attempt_id"]} billed under multiple tasks')
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc
    require(versions, 'inputs contain no task records')
    latest = {}
    for (task_id, revision), row in sorted(versions.items()):
        previous = latest.get(task_id)
        if previous:
            require(timestamp(row['recorded_at']) >= timestamp(previous['recorded_at']), f'{task_id}: revision timestamp goes backwards')
            old_ids = [a['attempt_id'] for a in previous['attempts']]
            require([a['attempt_id'] for a in row['attempts'][:len(old_ids)]] == old_ids, f'{task_id}: revision drops or reorders prior attempts')
        latest[task_id] = row
    return list(latest.values()), sources, {'input_files': len(paths), 'unique_revisions': len(versions), 'identical_duplicate_rows': duplicates}


def total(values):
    value = math.fsum(values)
    require(math.isfinite(value), 'cost sum overflow')
    return value


def summarize(rows, sources):
    buckets = defaultdict(list)
    for row in rows:
        route = tuple((a['model'], a['effort'], a['fork']) for a in row['attempts'])
        buckets[(tuple(row[k] for k in GROUP_FIELDS), route)].append(row)
    groups = []
    for (context, route), members in sorted(buckets.items()):
        accepted = sum(m['outcome'] == 'accepted' for m in members)
        terminal = all(m['outcome'] in ('accepted', 'failed') for m in members)
        complete = sum(all(a[k] is not None for a in m['attempts'] for k in COST_FIELDS) for m in members)
        costs = {k: total(a[k] for m in members for a in m['attempts'] if a[k] is not None) for k in COST_FIELDS}
        known = total(costs.values())
        worker_complete = all(a['worker_usd'] is not None for m in members for a in m['attempts'])
        group = dict(zip(GROUP_FIELDS, context))
        group.update(route=[dict(zip(('model', 'effort', 'fork'), step)) for step in route],
                     task_count=len(members), accepted_count=accepted,
                     first_pass_count=sum(m['outcome'] == 'accepted' and len(m['attempts']) == 1 for m in members),
                     failed_count=sum(m['outcome'] == 'failed' for m in members),
                     pending_count=sum(m['outcome'] == 'pending' for m in members),
                     invalidated_count=sum(m['outcome'] == 'invalidated' for m in members),
                     known_cost_usd=known, known_cost_components=costs, complete_cost_tasks=complete,
                     cost_per_accepted_usd=known / accepted if terminal and complete == len(members) and accepted else None,
                     worker_cost_per_accepted_usd=costs['worker_usd'] / accepted if terminal and worker_complete and accepted else None,
                     oldest_record=min(m['recorded_at'] for m in members), newest_record=max(m['recorded_at'] for m in members),
                     tasks=[{'task_id': m['task_id'], 'outcome': m['outcome'], 'evidence': m['evidence'],
                             'record_sources': sorted(sources[(m['task_id'], m['revision'])]),
                             'failure_kinds': [a['failure_kind'] for a in m['attempts'] if a['failure_kind'] != 'none']}
                            for m in sorted(members, key=lambda m: m['task_id'])])
        groups.append(group)
    return groups


def markdown(report):
    def escape(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    def money(value):
        return 'unknown' if value is None else f'${value:.6f}'
    lines = ['# Delegation evidence', '',
             f"Selected latest task records: {report['task_count']}. Filters: {escape(json.dumps(report['filters'], sort_keys=True))}.", '',
             'Descriptive evidence, not a model ranking. Compare within the same task conditions; route sequences include repairs and switches. Missing costs are not zero. Pending/invalidated tasks prevent complete cost-per-acceptance claims. No automatic policy changes.', '']
    for g in report['groups'][:30]:
        context = ' / '.join(escape(g[k]) for k in GROUP_FIELDS)
        route = ' → '.join(escape(f"{a['model']}:{a['effort']} fork={a['fork']}") for a in g['route'])
        lines += [f'## {context}', '', route, '',
                  f"Tasks {g['task_count']}; accepted {g['accepted_count']}; first-pass {g['first_pass_count']}; failed {g['failed_count']}; pending {g['pending_count']}; invalidated {g['invalidated_count']}.",
                  f"Known spend {money(g['known_cost_usd'])}; complete-cost records {g['complete_cost_tasks']}/{g['task_count']}; total cost/accepted {money(g['cost_per_accepted_usd'])}; worker-only cost/accepted {money(g['worker_cost_per_accepted_usd'])}.",
                  f"Record dates {g['oldest_record']} — {g['newest_record']}. {'Single observation; insufficient to set a default.' if g['task_count'] == 1 else 'Nonrandom task selection; inspect comparability before changing a preference.'}", '']
        for t in g['tasks'][:3]:
            lines.append(f"- {escape(t['task_id'])}: {t['outcome']}; failures={escape(', '.join(t['failure_kinds']) or 'none')}; evidence: {escape('; '.join(t['evidence']))}; record: {escape('; '.join(t['record_sources']))}")
        if len(g['tasks']) > 3:
            lines.append(f"- {len(g['tasks']) - 3} further task records omitted; see summary.json.")
        lines.append('')
    if len(report['groups']) > 30:
        lines.append(f"{len(report['groups']) - 30} further groups omitted; narrow the filters or inspect summary.json.")
    return '\n'.join(lines)


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(value)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('validate', 'summarize'):
        p = sub.add_parser(command)
        p.add_argument('inputs', nargs='+')
        if command == 'summarize':
            p.add_argument('--output-dir', required=True)
            p.add_argument('--origin', choices=('production', 'benchmark', 'all'), default='production')
            p.add_argument('--task-class')
            p.add_argument('--comparison-key')
            p.add_argument('--since', help='Inclusive recorded date YYYY-MM-DD, after resolving latest revisions')
    args = parser.parse_args()
    try:
        rows, sources, scan = load(args.inputs)
        if args.command == 'validate':
            print(json.dumps({'valid': True, 'task_count': len(rows), **scan}))
            return 0
        if args.since:
            datetime.strptime(args.since, '%Y-%m-%d')
        rows = [r for r in rows if (args.origin == 'all' or r['origin'] == args.origin)
                and (not args.task_class or r['task_class'] == args.task_class)
                and (not args.comparison_key or r['comparison_key'] == args.comparison_key)
                and (not args.since or timestamp(r['recorded_at']).date().isoformat() >= args.since)]
        report = {'schema_version': 1, 'task_count': len(rows), 'scan': scan,
                  'filters': {k: getattr(args, k) for k in ('origin', 'task_class', 'comparison_key', 'since')},
                  'groups': summarize(rows, sources)}
        out = Path(args.output_dir)
        input_paths = set(discover(args.inputs))
        require(not any((out / n).resolve() in input_paths for n in ('summary.json', 'summary.md')), 'output would overwrite an input')
        atomic_write(out / 'summary.json', json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        atomic_write(out / 'summary.md', markdown(report))
        print(json.dumps({'task_count': len(rows), 'groups': len(report['groups']), 'summary': str(out / 'summary.md')}))
        return 0
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
