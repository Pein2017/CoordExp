"""Exact final paired reduction; --verify recomputes without writes/model work."""
import argparse
import json
from pathlib import Path

from probes.parallel_owner_research.history import ROOT, binding, continuation_ledger, digest, publish, read, require, validate_packet


def natural_rows(arm):
    result = {}
    for shard in (0, 1):
        for line in (ROOT / f'endpoint-{arm}/shard-{shard}/rows.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['kind'] == 'natural':
                require(row['example_id'] not in result, 'duplicate natural example')
                result[row['example_id']] = row
    require(len(result) == 384, 'natural384 complete')
    return result


def aggregate(rows, ids):
    selected = [rows[example_id] for example_id in sorted(ids)]
    result = {'images': len(selected), 'scores': {}, 'burden': {}}
    for threshold in ('50', '60', '80'):
        score = {key: sum(r['free_score'][threshold][key] for r in selected) for key in ('tp', 'fp', 'fn')}
        score['f1'] = 2 * score['tp'] / (2 * score['tp'] + score['fp'] + score['fn'])
        result['scores'][threshold] = score
    for key in selected[0]['burden']:
        result['burden'][key] = sum(r['burden'][key] for r in selected)
    return result


def paired(before, after, ids):
    result = {}
    for threshold in ('50', '60', '80'):
        gained, lost, retained = [], [], []
        for example_id in sorted(ids):
            old = set(before[example_id]['free_score'][threshold]['owners'])
            new = set(after[example_id]['free_score'][threshold]['owners'])
            gained.extend(f'{example_id}:{owner}' for owner in sorted(new - old))
            lost.extend(f'{example_id}:{owner}' for owner in sorted(old - new))
            retained.extend(f'{example_id}:{owner}' for owner in sorted(old & new))
        result[threshold] = {'gained': gained, 'lost': lost, 'retained': retained,
                             'counts': {'gained': len(gained), 'lost': len(lost), 'retained': len(retained), 'net': len(gained) - len(lost)}}
    return result


def compute():
    packet = validate_packet(read(ROOT / 'preparation/packet-v2.json'))
    endpoint = read(packet['sources']['endpoint_source']['path'])
    rows = {'Stable50': {}, 'fixed_P': natural_rows('fixed_P'), 'mixed_PQ': natural_rows('mixed_PQ')}
    for frozen in endpoint['eval_records']:
        parsed = frozen['stable_parsed']
        ledger = continuation_ledger('', parsed['raw_decode_text'], frozen, 0, len(frozen['stable_ids']), frozen['stable_score']['stop_reason'])
        require(ledger['free_score'] == frozen['stable_score'], 'retained Stable50 parser/score replay')
        rows['Stable50'][frozen['example_id']] = {**ledger, 'example_id': frozen['example_id'], 'split': frozen['split']}
    panel_ids = {'union384': set(rows['Stable50']), 'targets2': {c['example_id'] for c in packet['cases']}}
    panel_ids['reference56'] = {r['example_id'] for r in endpoint['eval_records'] if r['split'] == 'reference56'}
    panel_ids['train256'] = {r['example_id'] for r in endpoint['eval_records'] if r['split'] != 'dev128'}
    panel_ids['dev128'] = {r['example_id'] for r in endpoint['eval_records'] if r['split'] == 'dev128'}
    panel_ids['other328'] = panel_ids['union384'] - panel_ids['reference56']
    panel_ids['other326_nonreference_nontarget'] = panel_ids['other328'] - panel_ids['targets2']
    require(not panel_ids['targets2'] & panel_ids['reference56'], 'target/reference disjointness')
    require([len(panel_ids[k]) for k in ('targets2', 'reference56', 'other326_nonreference_nontarget')] == [2, 56, 326], 'disjoint scientific denominators')
    require(panel_ids['targets2'] | panel_ids['reference56'] | panel_ids['other326_nonreference_nontarget'] == panel_ids['union384'], 'disjoint panels cover union384')
    natural = {name: {arm: aggregate(arm_rows, ids) for arm, arm_rows in rows.items()} for name, ids in panel_ids.items()}
    pairs = {name: {f'{after}_vs_{before}': paired(rows[before], rows[after], ids)
                    for before, after in [('Stable50', 'fixed_P'), ('Stable50', 'mixed_PQ'), ('fixed_P', 'mixed_PQ')]}
             for name, ids in panel_ids.items()}
    target_reads = {arm: read(ROOT / f'endpoint-{arm}/result.json')['targets'] for arm in rows}
    endpoint_cost = {arm: {key: read(ROOT / f'endpoint-{arm}/result.json')[key]
                          for key in ('allocated_gpu_hours', 'model_forwards', 'image_forwards')} for arm in rows}
    train_cost = {}
    for arm in ('fixed_P', 'mixed_PQ'):
        receipt = read(ROOT / f'full-{arm}/receipt.json')
        terminal_bindings = receipt['terminals']
        terminals = [read(b['path']) for b in terminal_bindings]
        cold = read(ROOT / f'full-{arm}/cold-check.json')
        train_cost[arm] = {
            'updates': receipt['updates'], 'world_size': receipt['world_size'],
            'terminals': terminals,
            'cold_check': binding(ROOT / f'full-{arm}/cold-check.json'),
            'cold_resource_fields': {k: v for k, v in cold.items() if any(word in k for word in ('elapsed', 'peak_', 'forward', 'load', 'resource'))},
        }
    return {'schema': 'parallel_owner_history.paired_reduction.v1', 'status': 'candidate_verified',
            'claim_scope': 'two trusted target rows; one fixed counterfactual earlier-order swap; exposed384 preservation, no fresh transfer',
            'history_packet': binding(ROOT / 'preparation/packet-v2.json'), 'training_input': binding(ROOT / 'preparation/training-inputs.json'),
            'producer': binding(Path(__file__).resolve()), 'natural': natural, 'paired': pairs, 'target_reads': target_reads,
            'endpoint_cost': endpoint_cost, 'training_cost': train_cost,
            'endpoint_results': {arm: binding(ROOT / f'endpoint-{arm}/result.json') for arm in rows},
            'stop_rule': 'Both32-update arms, both coldchecks, union384 natural endpoints and4conditional/model complete; no extension.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = compute()
    path = ROOT / 'paired-reduction.json'
    if args.verify:
        require(read(path) == result, 'paired reduction differs on fresh replay')
    else:
        publish(path, result)
    print(json.dumps({'status': result['status'], 'sha256': digest(result),
                      'union384': result['natural']['union384'],
                      'paired_union50': {k: v['50']['counts'] for k, v in result['paired']['union384'].items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
