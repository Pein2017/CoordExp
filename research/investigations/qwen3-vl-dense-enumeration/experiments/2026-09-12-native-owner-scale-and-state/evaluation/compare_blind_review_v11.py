"""CPU-only exact-ID join of frozen physical review and sealed source map."""
from collections import Counter, defaultdict
from pathlib import Path
from probes.native_owner_scale import evaluation as e

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/evaluation')


def main():
    review_path = ROOT / 'candidate-review-cards-v11/blind32-review.json'
    map_path = ROOT / 'candidate-natural-v11-consumer/blind-review-source-map.json'
    queue_path = ROOT / 'candidate-natural-v11-consumer/blind-review-queue.jsonl'
    assert e.file_hash(review_path) == 'b290f7d25aa0de49ab59dda13a27ad36934c5d5628ed6b943e179486563e188e'
    review, mapping, queue = e.read(review_path), e.read(map_path)['rows'], e.read_jsonl(queue_path)
    assert review['source_queue'] == e.binding(queue_path)
    assert [x['image_id'] for x in review['images']] == [x['image_id'] for x in queue]
    assert len(queue) == len({x['image_id'] for x in queue}) == 32
    keys = {(x['image_id'], p['proposal_id']) for x in queue for p in x['proposals']}
    assert len(keys) == sum(len(x['proposals']) for x in queue) == 419
    sources = defaultdict(set)
    source_rows = defaultdict(list)
    source_unique = set()
    arms = ('Stable50', 'scaled_terminal')
    for row in mapping:
        key = (row['image_id'], row['proposal_id'])
        assert key in keys and row['source_arm'] in arms
        identity = (key, row['source_arm'], row['source_prediction_index'])
        assert identity not in source_unique
        source_unique.add(identity)
        sources[key].add(row['source_arm'])
        source_rows[key].append(row)
    assert set(sources) == keys
    assignments = Counter()
    categories = {}
    owners = []
    owner_ids = set()
    image_results = []
    for image in review['images']:
        assert image['unreviewed_proposal_ids'] == []
        current = []
        for category, entries in [('owner', image['owners']), ('non_owner_or_group', image['non_owner_or_group']), ('unresolved', image['unresolved'])]:
            for entry in entries:
                entry_keys = [(image['image_id'], pid) for pid in entry['proposal_ids']]
                assert entry_keys and all(key in keys for key in entry_keys)
                for key in entry_keys:
                    assignments[key] += 1
                    categories[key] = category
                if category == 'owner':
                    assert entry['owner_id'] not in owner_ids
                    owner_ids.add(entry['owner_id'])
                    presence = {arm: any(arm in sources[key] for key in entry_keys) for arm in arms}
                    change = 'retained' if all(presence.values()) else ('gained' if presence['scaled_terminal'] else 'lost')
                    record = {'image_id': image['image_id'], 'owner_id': entry['owner_id'], 'description': entry['description'], 'proposal_ids': entry['proposal_ids'], 'extent_or_class_caveats': entry['extent_or_class_caveats'], 'presence': presence, 'change': change}
                    owners.append(record); current.append(record)
        image_results.append({'image_id': image['image_id'], 'owner_changes': dict(Counter(x['change'] for x in current)), 'reviewed_owner_clusters': len(current)})
    assert set(assignments) == keys and set(assignments.values()) == {1}
    per_arm = {}
    for arm in arms:
        selected = {key for key in keys if arm in sources[key]}
        per_arm[arm] = {'reviewed_physical_owner_presence': sum(o['presence'][arm] for o in owners), 'unique_proposal_denominator': len(selected), 'proposal_categories': dict(Counter(categories[key] for key in selected)), 'source_prediction_incidence': dict(Counter(categories[key] for key in selected for row in source_rows[key] if row['source_arm'] == arm))}
    result = {'schema': 'native_owner_scale_state.blind_review_comparison.v1', 'status': 'exact_id_join_complete_root_acceptance_pending', 'bindings': {'review': e.binding(review_path), 'source_map': e.binding(map_path), 'queue': e.binding(queue_path), 'script': e.binding(Path(__file__))}, 'validation': {'frozen_images': 32, 'queue_unique_keys': len(keys), 'map_unique_keys': len(sources), 'review_unique_keys': len(assignments), 'review_assignment_exactly_once': True, 'map_rows': len(mapping), 'multi_arm_proposal_keys': sum(len(value) > 1 for value in sources.values())}, 'review_categories': dict(Counter(categories.values())), 'reviewed_owner_clusters': len(owners), 'owner_changes': dict(Counter(o['change'] for o in owners)), 'per_arm': per_arm, 'images': image_results, 'owners': owners, 'boundary': 'Class-agnostic reviewed physical-owner presence within frozen mixed proposals, retaining recorded class/extent caveats. Not strict detection TP or exhaustive recall. Unresolved proposals stay in denominators and imply neither negative labels nor gains/losses. Group proposals are reported separately, not generalized to all unmatched/FP proposals. No GT support inference added.'}
    output = ROOT / 'candidate-review-cards-v11/blind-review-comparison.json'
    assert not output.exists()
    e.publish(output, result)
    print({key: result[key] for key in ('validation', 'review_categories', 'reviewed_owner_clusters', 'owner_changes', 'per_arm')})


if __name__ == '__main__':
    main()
