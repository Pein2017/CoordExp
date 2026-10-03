"""Offline full570 annotation accounting and rule burdens for each arm's versions."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from probes.full_label_fit.experiment import evaluate_versions
from probes.rule_stability.objectives import trajectory_diagnostics


def _version(key):
    if type(key) is int and 0 <= key <= 16:
        return key
    if isinstance(key, str) and key.isdigit() and str(int(key)) == key and 0 <= int(key) <= 16:
        return int(key)
    raise ValueError('greedy version must be an integer0..16 or its canonical decimal string')


def evaluate(images, versions_records):
    """Reuse frozen matching; each call supplies exactly one arm and its fresh version0."""
    if not isinstance(versions_records, Mapping) or not versions_records:
        raise ValueError('offline evaluation requires version0 and at least one greedy version')
    frozen = {}
    for key, records in versions_records.items():
        version = _version(key)
        name = 'zero' if version == 0 else str(version)
        if name in frozen:
            raise ValueError('greedy versions contain an ambiguous duplicate version')
        if not isinstance(records, (list, tuple)):
            raise ValueError('greedy version records must be a sequence')
        # Legacy whole-image records may lack arm; project locally without rewriting evidence.
        frozen[name] = [dict(record, arm='greedy') for record in records]
    if 'zero' not in frozen:
        raise ValueError('offline evaluation requires this arm\'s own version0 baseline')
    result = evaluate_versions(images, frozen)
    result['schema'] = 'rule-stability-full570-metrics-v1'
    for version, records in frozen.items():
        scored = result['scored'][version]
        diagnostics = [trajectory_diagnostics(record) for record in records]
        for record, analysis in zip(records, diagnostics, strict=True):
            row = scored['images'][record['image_id']]
            row['rule_burdens'] = analysis['burdens']
            row['overlap_distribution'] = analysis['overlap_distribution']
        aggregate = {key: sum(analysis['burdens'][key] for analysis in diagnostics)
                     for key in diagnostics[0]['burdens'] if key != 'longest_event_burst'}
        aggregate['longest_event_burst'] = max(a['burdens']['longest_event_burst'] for a in diagnostics)
        scored['totals']['rule_burdens'] = aggregate
        for key, value in aggregate.items():
            if key not in scored['totals']:
                scored['totals'][key] = value
        distributions = [analysis['overlap_distribution'] for analysis in diagnostics]
        pair_count = sum(d['pair_count'] for d in distributions)
        scored['totals']['overlap_distribution'] = {
            key: sum(d[key] for d in distributions)
            for key in ('pair_count', 'strict_gt_09_pairs', 'exact_09_pairs', 'positive_pairs')}
        scored['totals']['overlap_distribution'].update(
            max=max((d['max'] for d in distributions if d['max'] is not None), default=None),
            mean=(sum((d['mean'] or 0) * d['pair_count'] for d in distributions) / pair_count
                  if pair_count else None))
    result['scored']['0'] = result['scored'].pop('zero')
    result['transitions']['0'] = result['transitions'].pop('zero')
    tie = result['transitions'].get('image_4134_annotation_294005')
    if tie is not None:
        tie['0'] = tie.pop('zero')
    for transition in result['transitions'].values():
        if isinstance(transition, dict) and transition.get('adjacent', {}).get('from_version') == 'zero':
            transition['adjacent']['from_version'] = '0'
    result['scored'] = {str(v): result['scored'][str(v)] for v in sorted(_version(k) for k in versions_records)}
    result['baseline_version'] = '0'
    result['greedy_versions'] = list(result['scored'])
    return result


def _metric_versions(metrics):
    if (not isinstance(metrics, Mapping) or metrics.get('schema') != 'rule-stability-full570-metrics-v1'
            or metrics.get('baseline_version') != '0' or not isinstance(metrics.get('scored'), Mapping)):
        raise ValueError('comparison requires existing rule-stability metrics with own version0 baseline')
    names = list(metrics['scored'])
    if not names or any(not isinstance(key, str) for key in names):
        raise ValueError('comparison requires canonical saved version keys')
    versions = sorted(_version(key) for key in names)
    if versions != list(range(versions[-1] + 1)):
        raise ValueError('comparison requires complete greedy versions0..last')
    expected = [str(version) for version in versions]
    if metrics.get('greedy_versions') != expected:
        raise ValueError('declared greedy versions differ from saved metrics')
    transitions = metrics.get('transitions')
    if not isinstance(transitions, Mapping) or any(name not in transitions for name in expected):
        raise ValueError('comparison requires each arm\'s existing own-baseline transitions')
    for name in expected:
        totals = metrics['scored'][name].get('totals', {})
        if type(totals.get('denominator')) is not int or totals['denominator'] != 570:
            raise ValueError('comparison denominator must remain full570 in every version')
    return expected


def _numeric_differences(left, right):
    """Subtract comparable finite scalars; an undefined overlap mean remains null."""
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError('comparison scalar metrics must be mappings')
    fields = [{key for key, value in row.items() if type(value) in (int, float, bool) or value is None}
              for row in (left, right)]
    if fields[0] != fields[1]:
        raise ValueError('comparison scalar metric fields differ between arms')
    differences = {}
    for key in sorted(fields[0]):
        a, b = left[key], right[key]
        if any(value is not None and not math.isfinite(value) for value in (a, b)):
            raise ValueError('comparison metrics contain a nonfinite scalar')
        differences[key] = b - a if a is not None and b is not None else None
    return differences


def compare_arms(a_metrics, b_metrics):
    """Report B-A at every saved version; preserve each arm's own baseline ledger."""
    versions = _metric_versions(a_metrics)
    if _metric_versions(b_metrics) != versions:
        raise ValueError('comparison arms must have the same complete version keys')
    identities = [metrics.get('denominator_annotation_ids') for metrics in (a_metrics, b_metrics)]
    if (identities[0] != identities[1] or not isinstance(identities[0], list) or len(identities[0]) != 570
            or len({tuple(key) for key in identities[0]}) != 570):
        raise ValueError('comparison full570 annotation identities differ or are incomplete')
    image_ids = {str(key[0]) for key in identities[0]}
    if len(image_ids) != 18:
        raise ValueError('comparison must cover all18 image identities')
    rows = {}
    for version in versions:
        a, b = a_metrics['scored'][version], b_metrics['scored'][version]
        images = [{str(key): value for key, value in arm.get('images', {}).items()} for arm in (a, b)]
        if any(set(image_rows) != image_ids or len(arm.get('images', {})) != 18
               for image_rows, arm in zip(images, (a, b), strict=True)):
            raise ValueError('comparison version image identities are incomplete or different')
        transitions = {label: deepcopy(metrics['transitions'][version])
                       for label, metrics in (('A', a_metrics), ('B', b_metrics))}
        rows[version] = {
            'metric_differences': _numeric_differences(a['totals'], b['totals']),
            'rule_burden_differences': _numeric_differences(a['totals'].get('rule_burdens'), b['totals'].get('rule_burdens')),
            'overlap_distribution_differences': _numeric_differences(a['totals'].get('overlap_distribution'), b['totals'].get('overlap_distribution')),
            'own_baseline_transitions': {**transitions, 'count_differences': _numeric_differences(transitions['A'], transitions['B'])},
            'images': {image_id: {
                'metric_differences': _numeric_differences(images[0][image_id], images[1][image_id]),
                'rule_burden_differences': _numeric_differences(images[0][image_id].get('rule_burdens'), images[1][image_id].get('rule_burdens')),
            } for image_id in sorted(image_ids, key=int)},
        }
    endpoint = versions[-1]
    return {'schema': 'rule-stability-arm-comparison-v1', 'contrast': 'B_minus_A', 'denominator': 570,
            'versions': versions, 'endpoint_version': endpoint, 'per_version': rows,
            'endpoint': deepcopy(rows[endpoint]),
            'limitations': 'Training-internal annotation and rule contrasts; each arm keeps its own fresh version0 baseline. No rescore, pooled baseline, generalization claim, physical-FN proof, or quality gate.'}
