"""Rebind unchanged root labels after the verified alias-provenance-only repair."""

import json
from pathlib import Path

from finalize_admission import ROOT, bind


def main():
    old_path = ROOT / 'physical-admission/root-training-decisions-v1.json'
    assert bind(old_path)['sha256'] == '252dccb303149a7fc50af20130f8d88ad109455d9ef9819895ab8ba485cf4fc2'
    new_index = ROOT / 'physical-admission-join/final-v2/consumer-review-index-v2.json'
    assert bind(new_index)['sha256'] == '7696236c72cd5f5e84dbd872069e2cfad98ccfa9a103a787ef8c00d8d5ea56c2'
    decisions = json.loads(old_path.read_text())
    old_index = json.loads(Path(decisions['review_indexes'][0]['path']).read_text())
    repaired = json.loads(new_index.read_text())
    assert old_index['groups'] == repaired['groups']
    assert old_index['source_bindings'] == repaired['source_bindings']
    assert len(old_index['rows']) == len(repaired['rows']) == 502
    changed = 0
    for before, after in zip(old_index['rows'], repaired['rows']):
        assert all(before[key] == after[key] for key in before)
        additions = set(after) - set(before)
        assert additions <= {'same_image_aliases', 'execution_aliases_same_visual_group'}
        changed += bool(additions)
    assert changed == 140
    decisions['review_indexes'] = [bind(new_index)]
    decisions['transport_correction'] = {
        'supersedes_binding_only': bind(old_path),
        'proof': bind(ROOT / 'physical-admission-join/final-v2/transport-correction-proof-v2.json'),
        'root_ruling': 'Labels, canonical histories, source c/w/h and selection unchanged. Only required alias-provenance transport fields were added.',
    }
    output = ROOT / 'physical-admission/root-training-decisions-v2.json'
    with output.open('x') as stream:
        json.dump(decisions, stream, indent=2)
        stream.write('\n')
    assert json.loads(output.read_text())['decisions'] == json.loads(old_path.read_text())['decisions']
    print(json.dumps({'root_decisions': bind(output), 'same_labels': True, 'alias_only_rows': changed}))


if __name__ == '__main__':
    main()
