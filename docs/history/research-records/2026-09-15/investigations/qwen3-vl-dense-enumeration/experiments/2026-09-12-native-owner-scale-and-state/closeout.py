"""One-off root closeout: verify sealed evidence and persist reviewed observations."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state')
EVAL = ROOT / 'evaluation'
CARDS = EVAL / 'candidate-review-cards-v11'


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def check_bound(item):
    assert binding(Path(item['path']))['sha256'] == item['sha256'], item['path']


def publish(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2, ensure_ascii=False)
        out.write('\n')


def main():
    review = read(CARDS / 'blind32-review.json')
    joined = read(CARDS / 'blind-review-comparison.json')
    manifest = read(CARDS / 'blind32-manifest.json')
    assert binding(CARDS / 'blind32-review.json')['sha256'] == 'b290f7d25aa0de49ab59dda13a27ad36934c5d5628ed6b943e179486563e188e'
    for item in joined['bindings'].values():
        check_bound(item)
    queue = [json.loads(line) for line in Path(joined['bindings']['queue']['path']).read_text().splitlines()]
    mapping = read(Path(joined['bindings']['source_map']['path']))['rows']
    expected = {(x['image_id'], p['proposal_id']) for x in queue for p in x['proposals']}
    assert len(expected) == 419 and len(queue) == 32
    assert {x['image_id'] for x in manifest['items']} == {x['image_id'] for x in review['images']} == {x['image_id'] for x in queue}
    sources = {(x['image_id'], x['proposal_id']): x['source_arm'] for x in mapping}
    assert len(sources) == len(mapping) == 419 and set(sources) == expected
    assigned, categories, presence, changes = Counter(), Counter(), Counter(), Counter()
    for image in review['images']:
        check_bound(image['card'])
        assert not image['unreviewed_proposal_ids']
        for category in ['owners', 'non_owner_or_group', 'unresolved']:
            for entry in image[category]:
                keys = [(image['image_id'], pid) for pid in entry['proposal_ids']]
                assigned.update(keys)
                categories['owner' if category == 'owners' else category] += len(keys)
                if category == 'owners':
                    arms = {sources[key] for key in keys}
                    presence.update(arms)
                    changes['retained' if len(arms) == 2 else 'gained' if 'scaled_terminal' in arms else 'lost'] += 1
    assert set(assigned) == expected and set(assigned.values()) == {1}
    assert dict(categories) == joined['review_categories']
    assert dict(changes) == joined['owner_changes']
    assert all(presence[arm] == joined['per_arm'][arm]['reviewed_physical_owner_presence'] for arm in presence)

    # All 11 cards were viewed by root before this closeout. These are bounded
    # observations, not another exhaustive annotation set or negative labels.
    observations = {
        417044: 'Natural trained output escapes the left-edge donut loop and localizes many distinct central/right donuts, including reviewed unlabeled P6 and the second c target. The giant right-side donut extent is absent. Possible local same-owner overlap remains; zero strict repeats does not establish physical uniqueness.',
        210457: 'The real blue-cup c target is learned, but many new cup boxes drift over tree, person, legs and background. This is a new severe failure despite EOS and aggregate improvement, not a clean repair.',
        219546: 'Both reviewed spoons are naturally localized; many real table utensils remain. This is not exhaustive scene correctness.',
        477415: 'More real people and reviewed foreground c are represented; a huge background person extent and ambiguous chair/body boxes remain. Dense occlusion limits identity claims.',
        351017: 'Right-person c is clear and the old bottle loop is reduced. Table is physically represented with a different extent despite failing target IoU; matching miss is not physical absence.',
        59571: 'Oven/cooktop c appears and the left bottle loop is reduced. Cameraman c is not matched, though a partial person box remains; some wrong bottle boxes persist.',
        388795: 'Book repetition is reduced and one c meets geometric matching. Fine adjacent book-spine identity remains uncertain at card resolution; overlapping right-seat extents remain.',
        528944: 'The rear large wooden-bowl c still is not represented; some utensil changes occur. No universal repair.',
        25274: 'Reviewed crowd person is recovered and many incumbents remain; clipped dense people are not exhaustively audited.',
        99937: 'Book repetition decreases but the remote remains poorly matched. The large upper television is lost while lower displays and several desk objects remain: real preservation debt.',
        323322: 'Plant class appears with a broader/lower extent that fails c matching; physical plant presence and extent correctness are different. Sinks, toilet and vases remain.',
    }
    admitted = read(CARDS / 'admitted11-manifest.json')
    assert set(observations) == {x['image_id'] for x in admitted['items']}
    for item in admitted['items']:
        check_bound(item['card'])
    visual = {'schema': 'native_owner_scale_state.root_admitted_visual.v1', 'status': 'lead-accepted_bounded_observations', 'reviewer': '/root', 'reviewed_with': 'view_image', 'source_blind': False, 'manifest': binding(CARDS / 'admitted11-manifest.json'), 'boundary': 'All 11 paired cards personally viewed. Not exhaustive scene-owner labels; no new negative-training authorization. Geometric c matches do not prove all c owners absent from every baseline drifted extent.', 'images': [{**item, 'observation': observations[item['image_id']]} for item in admitted['items']]}
    completion = read(EVAL / 'completion-v11.json')
    paired = {}
    for arm, bound in completion['rows'].items():
        check_bound(bound)
        rows = [json.loads(line) for line in Path(bound['path']).read_text().splitlines()]
        assert len(rows) == len({x['image_id'] for x in rows}) == 640
        assert all(not x['prefix_ids'] and not x['forced_ids'] and x['remaining_budget'] == 3084 and len(x['action_ids']) <= 3084 for x in rows)
        paired[arm] = {x['image_id']: x for x in rows}
    assert set(paired['Stable50']) == set(paired['scaled_terminal'])
    assert all(x['stop_reason'] == 'im_end' for x in paired['scaled_terminal'].values())
    assert all(x['adapter_fingerprint'] == 'a7dcb56ea71ee8ab37a944778b7947c78ca22dfd321a0dcc59dde5227acecc80' for x in paired['scaled_terminal'].values())
    for item in visual['images']:
        item['natural_counters'] = {arm: {'tokens': len(rows[item['image_id']]['action_ids']), 'valid_predictions': rows[item['image_id']]['parsed']['valid_prediction_count'], 'parser_drops': rows[item['image_id']]['parsed']['dropped_prediction_count'], 'strict_repeats': rows[item['image_id']]['overlap_counts']['95'], 'stop_reason': rows[item['image_id']]['stop_reason']} for arm, rows in paired.items()}
    visual_path = CARDS / 'root-admitted11-review.json'
    publish(visual_path, visual)
    paths = {
        'acquisition': ROOT / 'scale/acquisition-full-v2.json',
        'admission': ROOT / 'scale/visual-review-full-v2/final-reviews.json',
        'training': ROOT / 'scale/training/preparation/training-completion-v2.json',
        'evaluation': EVAL / 'completion-v11.json',
        'blind_review': CARDS / 'blind32-review.json',
        'blind_join': CARDS / 'blind-review-comparison.json',
        'root_visual': visual_path,
        'repeat_projection': ROOT / 'repeat/raw128-projection.json',
        'state_consumer': ROOT / 'state/panel-v1/consumer.json',
        'closeout_script': Path(__file__),
    }
    receipt = {'schema': 'native_owner_scale_state.iteration.v1', 'status': 'lead-accepted_closed', 'completed_utc': datetime.now(timezone.utc).isoformat(), 'technical_status': 'fit_cold_natural_consumer_and_review_join_accepted', 'scientific_disposition': 'promising_natural_transfer_with_preservation_and_new_loop_debt', 'promotion': 'not_promoted', 'frozen_output_budget': 3084, 'bindings': {name: binding(path) for name, path in paths.items()}, 'fresh256': completion['strata']['fresh256'], 'trusted_c_geometry': completion['trusted_target_thresholds'], 'burden640': {arm: x['burden'] for arm, x in completion['arms'].items()}, 'blind32': {'review_categories': dict(categories), 'physical_presence': dict(presence), 'owner_changes': dict(changes), 'boundary': joined['boundary'], 'root_spot_checks': [434996, 93437]}, 'limitations': ['Two original fresh-baseline slice rows have post-hoc identity backfill, not live model-identity receipts.', 'Admission contains 16 packages but only 11 images; no isolated sample-count effect or large-scale claim.', 'Blind32 has 123 unresolved proposals; equal reviewed physical presence is not an equivalence result.', 'GT-unmatched is not false-object evidence. Reviewed owner presence retains class and extent caveats.', 'Local spoon/donut recovery coexists with cup210457 new repetition and desk99937 television loss.'], 'stop': 'All frozen lanes closed. No additional model execution, automatic promotion, publication or architecture change authorized by this receipt.'}
    publish(ROOT / 'iteration-receipt.json', receipt)
    print(json.dumps({'receipt': binding(ROOT / 'iteration-receipt.json'), 'blind_presence': dict(presence), 'blind_changes': dict(changes), 'root_visual_images': len(visual['images']), 'natural_paired_rows': 640}))


if __name__ == '__main__':
    main()
