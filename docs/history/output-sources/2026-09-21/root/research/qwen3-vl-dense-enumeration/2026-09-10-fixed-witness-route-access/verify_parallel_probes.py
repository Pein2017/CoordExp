"""Task-local CPU acceptance and supplemental readout; never runs a model."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import statistics

from transformers import AutoTokenizer
from probes.dora_owner_learning.route_access import reduce, credit_summary
from probes.source_rweak_row_cross.owner_row_robustness import consume, reduce_records

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
P1 = ROOT / '2026-09-10-fixed-witness-route-access'
P2 = ROOT / '2026-09-10-owner-row-continuation-robustness'
CWD = Path('/data/CoordExp/.worktrees/research-probes')

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, value):
    if p.exists():
        assert read(p) == value, f'occupied nonidentical result: {p}'
    else:
        with p.open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')

packet = read(P1 / 'inputs.json')
reduction = reduce(P1)
credit = credit_summary(P1)
assert reduction == read(P1 / 'reduction.json')
assert credit == read(P1 / 'credit.json')
assert reduction['samples'] == 43 and reduction['images'] == 32
assert all(v['passed'] and not v['mismatch_positions'] for v in reduction['parity'].values())
witnesses = {w['witness_id']: w for w in packet['witnesses']}
greedy = {v['example_id']: v for v in credit['greedy_comparisons'] if 'source_greedy' in v['roles']}
labels = Counter()
details = []
for c in reduction['comparisons']:
    f = c['first_forks']['source_greedy']
    labels[witnesses[c['witness_id']]['sections']['partition'][f['position']]] += 1
    details.append(dict(witness_id=c['witness_id'], example_id=c['example_id'],
        advantage_sign=c['advantage_sign'], delta_logprob=c['sum_delta_logprob'],
        delta_logodds_vs_fixed_source_greedy=c['sum_delta_logprob'] - greedy[c['example_id']]['delta_logprob'],
        source_fork_margin=f['source']['target_best_other_margin'],
        post_fork_margin=f['post']['target_best_other_margin'],
        fork_margin_delta=f['post']['target_best_other_margin'] - f['source']['target_best_other_margin']))
stats = {}
for key in ('delta_logprob', 'delta_logodds_vs_fixed_source_greedy', 'source_fork_margin', 'post_fork_margin', 'fork_margin_delta'):
    values = [x[key] for x in details]
    stats[key] = dict(min=min(values), median=statistics.median(values), mean=statistics.mean(values),
                      max=max(values), positive=sum(v > 0 for v in values))
supplement = dict(schema='lead_fixed_witness_supplement.v1', fixed_witnesses=43, images=32,
    first_fork_row_partition=dict(labels), statistics=stats, details=details,
    positive_advantage_logodds_improved=sum(x['delta_logodds_vs_fixed_source_greedy'] > 0 for x in details if x['advantage_sign'] == 'positive'),
    mean_image_delta_logodds=reduction['mean_image_delta_logprob'] - statistics.mean(x['delta_logprob'] for x in greedy.values()),
    scope='Descriptive fixed-sequence log-odds movement, not expected reward, a selector, or a necessary owner-access route. Do not pool repeated images as independent trials.')
save(P1 / 'lead-supplement.json', supplement)

manifest = read(P2 / 'manifest-v2.json')
admission = read(P2 / 'admission-v1.json')
cases = {c['case_id']: c for c in manifest['selected']}
assert sha(P2 / 'manifest-v2.json') == admission['manifest_file_sha256']
tokenizer = AutoTokenizer.from_pretrained(packet['plan']['model']['base_model_path'], local_files_only=True)
expected = [(cid, arm) for cid in admission['admitted_case_ids'] for arm in ('B', 'partial_A', 'A', 'Aprime', 'sample_A')]
records = consume(P2 / 'execution/rows.jsonl', manifest, tokenizer, expected)
assert records == read(P2 / 'execution/consumer.json')
assert reduce_records(records, manifest) == read(P2 / 'execution/reduction-v2.json')
case_checks = []
for cid in admission['admitted_case_ids']:
    arms = {r['arm']: r for r in records if r['case_id'] == cid}
    case = cases[cid]
    assert arms['B']['action_ids'] == case['source_ids']
    assert arms['B']['parsed'] == case['golden']
    assert arms['A']['suffix_ids'] == arms['Aprime']['suffix_ids']
    assert arms['B']['parsed']['dropped_prediction_count'] == 0
    for arm in ('partial_A', 'A', 'Aprime'):
        r = arms[arm]
        assert case['owner'] not in r['conditional']['prefix_covered']
        assert case['owner'] in r['conditional']['release_covered']
        assert r['conditional']['B_free_suffix_globally_assigned']
        assert set(arms['B']['score']['50']['owners']) < set(r['score']['50']['owners'])
    case_checks.append(dict(case_id=cid, B_exact=True, A_Aprime_suffix_exact=True,
        arms=[dict(arm=a, tp=[r['score'][t]['tp'] for t in ('50','60','80')],
                   fp50=r['score']['50']['fp'], fn50=r['score']['50']['fn'],
                   f1_50=r['score']['50']['f1'], repeats=r['score']['strict_repeats'],
                   drops=r['score']['parser_drops'], stop=r['stop_reason'],
                   complete_tokens=len(r['action_ids']), generated_tokens=len(r['suffix_ids'])) for a,r in arms.items()]))
source_term = read(P1 / 'terminal.json')
post_term = read(P1 / 'terminal_post.json')
row_term = read(P2 / 'execution/terminal.json')
assert source_term['status'] == 'failed' and 'frozen_instance' in source_term['error']
assert post_term['status'] == row_term['status'] == 'completed'
assert post_term['actual_score_forwards'] == 170 and post_term['model_loads'] == 2
assert row_term['continuations'] == 10 and row_term['model_loads'] == 1
assert post_term['cumulative_model_execution_seconds'] < 3600 and row_term['elapsed_seconds'] < 3600
for f, value in read(P2 / 'acceptance-candidate.json')['files_sha256'].items():
    p = Path(f) if Path(f).is_absolute() else CWD / f
    assert sha(p) == value, f'changed final artifact/code: {f}'
checks = dict(schema='parallel_probes.lead_checks.v1', status='verified',
    focused_tests=dict(passed=22, wall_seconds=6.14, command='python -m pytest -q probes/dora_owner_learning/tests/test_route_access.py probes/source_rweak_row_cross/tests/test_owner_row_robustness.py'),
    fixed_score_reduction_exact=True, fixed_credit_reduction_exact=True,
    fresh_native_row_consumer_exact=True, fresh_row_reduction_v2_exact=True,
    probe1_cumulative_model_seconds=post_term['cumulative_model_execution_seconds'],
    probe2_model_seconds=row_term['elapsed_seconds'],
    total_model_gpu_hours=(post_term['cumulative_model_execution_seconds'] + row_term['elapsed_seconds']) / 3600,
    case_checks=case_checks, verification_script_sha256=sha(Path(__file__)))
save(P1 / 'lead-checks.json', checks)
save(P2 / 'lead-checks.json', checks)
print(json.dumps(dict(status='verified', samples=43, row_cases=2, tests=22, p1_statistics=stats,
    first_fork_row_partition=dict(labels), total_model_gpu_hours=checks['total_model_gpu_hours']), indent=2))
