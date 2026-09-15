"""Salvage view provenance, never hidden reasoning or missing physical labels."""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput')
SOURCE = ROOT / 'physical-admission-recovery'
OUT = ROOT / 'physical-admission-recovery-salvage'
TRACE = Path('/data/CoordExp/.codex/sessions/2026/09/13/rollout-2026-09-13T06-15-40-01a09968-3eab-7123-90cf-d9b0f7a5da91.jsonl')


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'receipt-v1.json').exists(), 'never overwrite a salvage receipt'
    index = json.loads((SOURCE / 'review-index-v1.json').read_text())
    groups = {g['visual_group_id']: g for g in index['groups']}
    assert len(groups) == 80 and len(index['rows']) == 98
    cutoff = TRACE.stat().st_size
    digest = hashlib.sha256()
    calls, replies = {}, {}
    unresolved_view_references = []
    path_re = re.compile(r'view_image\(\s*\{\s*path\s*:\s*[\"\x27]([^\"\x27]+)[\"\x27]')
    read_bytes = 0
    with TRACE.open('rb') as stream:
        for line in stream:
            if read_bytes + len(line) > cutoff:
                break
            digest.update(line)
            read_bytes += len(line)
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get('type') != 'response_item':
                continue
            payload = item.get('payload', {})
            kind, call_id = payload.get('type'), payload.get('call_id')
            if kind in ('custom_tool_call', 'function_call'):
                code = payload.get('input', payload.get('arguments', ''))
                if not isinstance(code, str) or 'view_image' not in code or str(SOURCE) not in code:
                    continue
                paths = [Path(p) for p in path_re.findall(code) if p.startswith(str(SOURCE) + '/')]
                if len(paths) != 1:
                    unresolved_view_references.append({'timestamp': item.get('timestamp'), 'call_id': call_id})
                    continue
                path = paths[0]
                match = re.search(r'PAR-\d{4}', path.name)
                if match is None or match.group() not in groups:
                    continue
                calls[call_id] = {
                    'call_id': call_id, 'timestamp': item.get('timestamp'),
                    'group_id': match.group(), 'path': str(path),
                    'kind': 'crop' if path.parent.name == 'crops' else 'card',
                    'reviewer': '/root/owner_overlay', 'model': 'gpt-5.6-luna',
                    'detail_original_requested': bool(re.search(r'detail\s*:\s*[\"\x27]original[\"\x27]', code)),
                }
            elif kind in ('custom_tool_call_output', 'function_call_output') and call_id in calls:
                output = payload.get('output')
                count = sum(isinstance(block, dict) and block.get('type') in ('input_image', 'image')
                            for block in output) if isinstance(output, list) else 0
                replies[call_id] = {'returned_at': item.get('timestamp'), 'returned_image_blocks': count}

    events = []
    file_bindings = {}
    for call_id, event in calls.items():
        event.update(replies.get(call_id, {'returned_at': None, 'returned_image_blocks': 0}))
        event['completed_single_image_return'] = event['returned_image_blocks'] == 1
        path = Path(event['path'])
        assert path.is_file(), path
        file_bindings.setdefault(str(path), binding(path))
        event['file'] = file_bindings[str(path)]
        events.append(event)
    complete = [event for event in events if event['completed_single_image_return']]
    viewed_groups = {event['group_id'] for event in complete}
    saved = []
    # Copy only explicit, already persisted assessment artifacts. Never copy the
    # raw trace, private reasoning, image payloads, or unrelated session data.
    for name in ('partial-review-notes-v1.json', 'handoff-quiescence-v1.json',
                 'partial-assessments-salvage.json', 'physical-review-v1.json'):
        source = SOURCE / name
        if not source.exists():
            continue
        data = source.read_bytes()
        json.loads(data)
        target = OUT / ('saved-' + name)
        with target.open('xb') as stream:
            stream.write(data)
        saved.append({'source': binding(source), 'snapshot': binding(target)})
    ledger = {'schema': 'owner_successor_scale.salvaged_view_events.v1',
              'events': events, 'unresolved_view_references': unresolved_view_references}
    ledger_path = OUT / 'individual-view-events-v1.json'
    with ledger_path.open('x') as stream:
        json.dump(ledger, stream, indent=2, sort_keys=True)
        stream.write('\n')
    notes = json.loads((SOURCE / 'partial-review-notes-v1.json').read_text())
    receipt = {
        'schema': 'owner_successor_scale.evidence_salvage.v1',
        'status': 'partial_evidence_preserved_not_training_admission',
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_index': binding(SOURCE / 'review-index-v1.json'),
        'source_trace_prefix': {'path': str(TRACE), 'captured_complete_line_bytes': read_bytes,
                                'sha256': digest.hexdigest(), 'raw_trace_copied': False},
        'view_ledger': binding(ledger_path), 'saved_assessment_artifacts': saved,
        'counts': {'frozen_groups': 80, 'frozen_jobs': 98,
                   'direct_literal_view_calls': len(calls),
                   'completed_single_image_returns': len(complete),
                   'unique_viewed_groups': len(viewed_groups),
                   'returned_views_by_kind': dict(Counter(e['kind'] for e in complete)),
                   'unique_cards': len({e['path'] for e in complete if e['kind'] == 'card'}),
                   'unique_crops': len({e['path'] for e in complete if e['kind'] == 'crop'}),
                   'unresolved_view_references': len(unresolved_view_references),
                   'old_provisional_assessments': len(notes.get('notes', []))},
        'old_notes_reviewer': notes.get('reviewer'),
        'groups_without_completed_direct_view': sorted(set(groups) - viewed_groups),
        'last_completed_view_at': max((e['returned_at'] for e in complete), default=None),
        'luna_final_review_present_at_capture': (SOURCE / 'physical-review-v1.json').exists(),
        'luna_partial_salvage_present_at_capture': (SOURCE / 'partial-assessments-salvage.json').exists(),
        'boundary': 'A returned image proves a view event, not a saved assessment, correct label, exact-history newness, or training admission. No missing decisions or private reasoning were reconstructed.',
        'training_admission': False, 'negative_labels_created': 0,
    }
    with (OUT / 'receipt-v1.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'receipt': str(OUT / 'receipt-v1.json'), 'counts': receipt['counts'],
                      'last_completed_view_at': receipt['last_completed_view_at'],
                      'luna_final_review_present': receipt['luna_final_review_present_at_capture'],
                      'luna_partial_salvage_present': receipt['luna_partial_salvage_present_at_capture']}))


if __name__ == '__main__':
    main()
