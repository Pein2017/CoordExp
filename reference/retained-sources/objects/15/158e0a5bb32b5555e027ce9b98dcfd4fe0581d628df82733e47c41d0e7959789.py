"""Task-local Git and canonical-dirty preservation receipt; no repository writes."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
CANONICAL = Path('/data/CoordExp/.worktrees/research-probes')


def git(cwd, *args):
    return subprocess.check_output(
        ['git', '--no-optional-locks', *args], cwd=cwd, text=True
    )


def dirty(cwd, *, hashes=False):
    parts = iter(git(cwd, 'status', '--porcelain=v1', '-z', '--untracked-files=all').split('\0'))
    result = {}
    for item in parts:
        if not item:
            continue
        status, name = item[:2], item[3:]
        record = {'status': status}
        if 'R' in status or 'C' in status:
            record['from'] = next(parts)
        path = Path(cwd) / name
        if hashes:
            record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        result[name] = record
    return result


worktrees = []
for block in git(CANONICAL, 'worktree', 'list', '--porcelain').strip().split('\n\n'):
    fields = {}
    for line in block.splitlines():
        key, _, value = line.partition(' ')
        fields[key] = value
    path = fields['worktree']
    fields['exists'] = Path(path).is_dir()
    if fields['exists']:
        try:
            fields['dirty'] = dirty(path, hashes=Path(path) == CANONICAL)
        except subprocess.CalledProcessError as exc:
            fields['status_error'] = exc.returncode
    worktrees.append(fields)
canonical = next(w for w in worktrees if w['worktree'] == str(CANONICAL))
baseline = json.loads((EVIDENCE / 'isolation.json').read_text())
old = baseline['canonical_dirty']
new = canonical['dirty']
changed = [p for p in old if new.get(p) != old[p]]
added = sorted(set(new) - set(old))
receipt = {
    'at': datetime.now(timezone.utc).isoformat(),
    'worktrees': worktrees,
    'canonical_head': canonical['HEAD'],
    'canonical_locked': 'locked' in canonical,
    'canonical_dirty_count': len(new),
    'canonical_dirty_matches_initial': new == old,
    'canonical_changed_initial_paths': changed,
    'canonical_added_dirty_paths': added,
}
target = EVIDENCE / sys.argv[1]
target.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'worktrees'}))
print(json.dumps({'inventory': str(target), 'worktrees': [
    {'path': w['worktree'], 'branch': w.get('branch'), 'dirty_count': len(w.get('dirty', {})), 'locked': 'locked' in w}
    for w in worktrees
]}))
