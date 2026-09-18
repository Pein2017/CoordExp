#!/usr/bin/env python3
"""Read-only research layout, catalog, source integrity and historical-link checks.

This validates local knowledge plumbing, not scientific truth or model artifacts.
No model imports, filesystem writes, alias creation, or automatic repair.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

CAPTURE = Path('docs/history/research-records/2026-09-15-root-collapse')
RESEARCH = Path('research')
CATALOG = RESEARCH / 'experiments/catalog.jsonl'
ROOT_NAMES = {'index.md', 'CONVENTIONS.md', 'story.md', 'glossary.md',
              'alternatives.md', 'questions', 'literature', 'experiments'}
LIFECYCLES = {'planned', 'ready', 'running', 'blocked', 'paused', 'closed', 'superseded'}
EVIDENCE_STATES = {'none', 'partial', 'unreviewed', 'accepted', 'invalid'}
LINK = re.compile(r'!?\[[^\]\n]*\]\(([^)\n]+)\)')
FENCED = re.compile(r'(?ms)^\s*```[^\n]*\n.*?^\s*```[^\n]*$')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or any(c in name for c in '\0\r\n'):
        raise ValueError('invalid repository path')
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'non-local repository path: {name}')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'path escapes checkout: {name}')
    return root / path


def logical_relative(root: Path, name: str, bundle: dict) -> str | None:
    if name.startswith('/'):
        for prefix in (str(root.resolve()), bundle['canonical_root']):
            if name == prefix or name.startswith(prefix + '/'):
                name = name[len(prefix):].lstrip('/')
                break
        else:
            return None
    value = posixpath.normpath(name)
    return None if value == '..' or value.startswith('../') else value


def load_bundle(root: Path) -> dict:
    latest = json.loads(local_path(root, str(CAPTURE / 'manifest.json')).read_text())
    previous_path = local_path(root, latest['previous_manifest'])
    if digest(previous_path.read_bytes()) != latest['previous_manifest_sha256']:
        raise ValueError('previous source manifest changed')
    previous = json.loads(previous_path.read_text())
    consumers = json.loads(local_path(root, str(CAPTURE / 'consumer-sources.json')).read_text())
    return {'canonical_root': latest['canonical_root'],
            'captures': [previous, latest, consumers],
            'retirements': latest['preexisting_retirements'],
            'exposure': latest['research_json_exposure']}


def resolve_reference(root: Path, bundle: dict, document: str, target: str) -> dict:
    """Use original document coordinates for frozen sources; never repair a live link."""
    parts = urlsplit(target.strip().strip('<>'))
    if parts.scheme or parts.netloc:
        return {'kind': 'external', 'target': target, 'exists': None}
    doc = logical_relative(root, document, bundle)
    if doc is None:
        raise ValueError('document outside checkout')
    captures = bundle['captures']
    source, owner = doc, None
    for capture in captures:
        match = next((e for e in capture['files'] if e['archive'] == doc), None)
        if match:
            source, owner = match['source'], capture
            break
    # Removed original paths can be used explicitly with the resolver.
    if owner is None and not local_path(root, doc).exists():
        for capture in reversed(captures):
            if any(e['source'] == doc for e in capture['files']):
                owner = capture
                break
    link_path = unquote(parts.path)
    candidate = (link_path if link_path.startswith('/') else
                 posixpath.join(posixpath.dirname(source), link_path)) if link_path else source
    logical = logical_relative(root, candidate, bundle)
    if logical is None:
        return {'kind': 'external' if link_path.startswith('/') else 'invalid',
                'target': target, 'exists': None}
    resolved = logical
    if owner is not None:
        ordered = [owner] + [c for c in reversed(captures) if c is not owner]
        for capture in ordered:
            match = next((e for e in capture['files'] if e['source'] == logical), None)
            if match:
                resolved = match['archive']
                break
        else:
            for capture in ordered:
                prefix = capture.get('source_prefix')
                if prefix and (logical == prefix or logical.startswith(prefix + '/')):
                    proposed = capture['archive_prefix'] + logical[len(prefix):]
                    if local_path(root, proposed).is_dir():
                        resolved = proposed
                        break
    try:
        exists = local_path(root, resolved).exists()
    except ValueError:
        return {'kind': 'invalid', 'logical_path': logical, 'exists': None}
    retired = next((e for e in bundle.get('retirements', []) if e['path'] == resolved), None)
    result = {'kind': 'local', 'logical_path': logical, 'path': resolved,
              'fragment': parts.fragment, 'exists': exists}
    if retired and not exists:
        result['recovery_git_spec'] = retired['recovery_git_spec']
        result['availability'] = 'git_recoverable_not_materialized'
    return result


def check_sources(root: Path, bundle: dict) -> list[str]:
    errors: list[str] = []
    seen_archives: set[str] = set()
    retirements = {e['path']: e for e in bundle.get('retirements', [])}
    if len(retirements) != len(bundle.get('retirements', [])):
        errors.append('duplicate retirement')
    used_retirements = set()
    for capture in bundle['captures']:
        seen_sources = set()
        for entry in capture['files']:
            source, archived = entry['source'], entry['archive']
            if source in seen_sources or archived in seen_archives:
                errors.append(f'duplicate source mapping: {source}')
            seen_sources.add(source)
            seen_archives.add(archived)
            try:
                path = local_path(root, archived)
                if not path.is_file() and archived in retirements:
                    retired = retirements[archived]
                    if retired['sha256'] != entry['sha256'] or retired['source'] != source:
                        errors.append(f'retirement identity mismatch: {source}')
                    used_retirements.add(archived)
                    continue
                data = path.read_bytes()
                if digest(data) != entry['sha256'] or len(data) != entry['bytes']:
                    errors.append(f'source bytes changed: {archived}')
            except (OSError, ValueError) as exc:
                errors.append(f'source unavailable: {archived}: {exc}')
    if used_retirements != set(retirements):
        errors.append('retirement is not an exact missing captured source')
    return errors


def check_git_sources(root: Path, bundle: dict) -> list[str]:
    """Compare capture claims with actual baseline Git blobs, not self-hashes alone."""
    errors = []
    requests = []
    for capture in bundle['captures']:
        baseline = capture['baseline_head']
        if not re.fullmatch('[0-9a-f]{40}', baseline):
            raise ValueError('invalid baseline commit')
        prefix = capture.get('source_prefix')
        if prefix:
            tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', baseline, '--', prefix], cwd=root)
            expected = {r.split(b'\t', 1)[1].decode() for r in tree.split(b'\0') if r
                        and r.split(b' ', 1)[0] != b'120000'}
            captured = {e['source'] for e in capture['files'] if e['git_tracked_at_capture']
                        and (e['source'] == prefix or e['source'].startswith(prefix + '/'))}
            if expected != captured:
                errors.append(f'baseline source coverage mismatch: {prefix}')
        requests.extend((f"{baseline}:{e['source']}", e) for e in capture['files']
                        if e['git_tracked_at_capture'])
    for retired in bundle.get('retirements', []):
        commit, spec = retired['retired_commit'], retired['recovery_git_spec']
        if not re.fullmatch('[0-9a-f]{40}', commit) or spec != commit + '^:' + retired['path']:
            raise ValueError('invalid retirement recovery binding')
        delta = subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-status',
                                         '-r', commit, '--', retired['path']], cwd=root, text=True).strip()
        if delta != 'D\t' + retired['path']:
            errors.append(f'retirement is not a committed exact deletion: {retired["path"]}')
        requests.append((spec, retired))
    payload = ''.join(spec + '\n' for spec, _ in requests).encode()
    result = subprocess.run(['git', 'cat-file', '--batch'], input=payload, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=root, check=True)
    stream = io.BytesIO(result.stdout)
    for spec, entry in requests:
        header = stream.readline().decode().strip().split()
        if len(header) != 3 or header[1] != 'blob':
            errors.append(f'Git source unavailable: {spec}')
            continue
        data = stream.read(int(header[2]))
        stream.read(1)
        if digest(data) != entry['sha256']:
            errors.append(f'Git source differs from capture: {spec}')
    return errors


def check_state(root: Path, path: str, unit_id: str) -> tuple[list[str], dict]:
    errors = []
    state = json.loads(local_path(root, path).read_text())
    if state.get('schema_version') != 1 or state.get('unit_id') != unit_id:
        errors.append(f'{path}: state identity mismatch')
    if state.get('lifecycle') not in LIFECYCLES or state.get('evidence') not in EVIDENCE_STATES:
        errors.append(f'{path}: invalid lifecycle/evidence axis')
    for key in ('disposition', 'state_as_of', 'boundary', 'next_action'):
        if not isinstance(state.get(key), str) or not state[key].strip():
            errors.append(f'{path}: missing {key}')
    if not isinstance(state.get('not_authorized'), list) or not all(
            isinstance(x, str) for x in state.get('not_authorized', [])):
        errors.append(f'{path}: invalid not_authorized list')
    for key in ('protocol', 'state_source', 'result'):
        value = state.get(key)
        if key == 'result' and value is None and state.get('evidence') != 'accepted':
            continue
        if not value or not local_path(root, value).is_file():
            errors.append(f'{path}: missing {key} target')
    return errors, state


def check_catalog(root: Path, rows: list[dict], bundle: dict) -> list[str]:
    errors, ids, protocols, states, readings = [], set(), set(), set(), set()
    for row in rows:
        uid = row['id']
        if uid in ids:
            errors.append(f'duplicate unit ID: {uid}')
        ids.add(uid)
        if not row.get('title') or not row.get('kind') or not row.get('topics'):
            errors.append(f'{uid}: missing title/kind/topics')
        for topic in row.get('topics', []):
            if not local_path(root, f'research/questions/{topic}.md').is_file():
                errors.append(f'{uid}: missing question {topic}')
        for name in ('record_root', 'reading_entry'):
            if not local_path(root, row[name]).exists():
                errors.append(f'{uid}: missing {name}')
        for name in ('protocols', 'result_records'):
            if not isinstance(row.get(name), list):
                errors.append(f'{uid}: {name} must be a list')
                continue
            for path in row[name]:
                if not local_path(root, path).is_file():
                    errors.append(f'{uid}: missing catalog target {path}')
        protocols.update(row.get('protocols', []))
        readings.add(row['reading_entry'])
        state_path = row.get('state')
        if row.get('tracking') == 'current':
            if not state_path or state_path in states:
                errors.append(f'{uid}: missing/duplicate current state')
                continue
            states.add(state_path)
            found, state = check_state(root, state_path, uid)
            errors.extend(found)
            if 'handoff' in Path(row['reading_entry']).name or row['reading_entry'].startswith('docs/history/'):
                errors.append(f'{uid}: transport/history owns current route')
            if state.get('result') and state['result'] not in row['result_records']:
                errors.append(f'{uid}: state result not catalogued')
        elif row.get('tracking') != 'historical' or state_path is not None:
            errors.append(f'{uid}: invalid historical/current ownership')
    expected_protocols = {e['archive'] for c in bundle['captures'] for e in c['files']
                          if e['source'].startswith('research/') and Path(e['source']).name == 'unit.md'}
    if not expected_protocols <= protocols:
        errors.append('catalog omits preserved protocols')
    expected_roots = set()
    for capture in bundle['captures']:
        for entry in capture['files']:
            src = entry['source']
            if not src.startswith('research/') or '/experiments/' not in src:
                continue
            tail = src.split('/experiments/', 1)[1]
            first = tail.split('/', 1)[0]
            if first not in {'index.md', 'history', 'catalog.jsonl'}:
                expected_roots.add(Path(first).stem if '/' not in tail else first)
    if not expected_roots <= ids:
        errors.append('catalog omits preserved experiment roots: ' + ','.join(sorted(expected_roots - ids)))
    actual_states = {str(p.relative_to(root)) for p in (root / RESEARCH / 'experiments').rglob('state.json')}
    if states != actual_states:
        errors.append('orphan/missing current state')
    entry = (root / RESEARCH / 'index.md').read_text()
    for state in states:
        if posixpath.relpath(state, 'research') not in entry:
            errors.append(f'frontier omits current state: {state}')
    return errors


def links_in(text: str) -> list[str]:
    return LINK.findall(FENCED.sub('', text))


def check_layout(root: Path) -> list[str]:
    errors = []
    actual = {p.name for p in (root / RESEARCH).iterdir()}
    if actual != ROOT_NAMES:
        errors.append(f'research root differs: extra={sorted(actual - ROOT_NAMES)}, missing={sorted(ROOT_NAMES - actual)}')
    for path in (root / RESEARCH).rglob('*'):
        if path.is_symlink() or path.suffix in {'.py', '.pyc', '.sh', '.log'} or path.name == '__pycache__':
            errors.append(f'executable/cache/alias in research: {path.relative_to(root)}')
    return errors


def check_live_links(root: Path, bundle: dict, documents: list[Path]) -> tuple[list[str], int, int]:
    errors, local_count, external_count = [], 0, 0
    for document in documents:
        rel = str(document.relative_to(root))
        for target in links_in(document.read_text()):
            ref = resolve_reference(root, bundle, rel, target)
            if ref['kind'] == 'external':
                external_count += 1
            elif ref['kind'] != 'local' or not ref['exists']:
                errors.append(f'{rel}: unresolved live link {target}')
            else:
                local_count += 1
    return errors, local_count, external_count


def check_exposure(root: Path, bundle: dict) -> list[str]:
    rows = bundle.get('exposure', {}).get('records', [])
    if not rows:
        return ['missing frozen exposure inventory']
    errors = []
    base = local_path(root, 'docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration')
    actual = {str(p.relative_to(root)) for p in base.rglob('*.json')
              if '2026-09-12-parallel-owner-research' not in str(p)}
    if actual != {r['archive_path'] for r in rows}:
        errors.append('frozen exposure file-set changed')
    for row in rows:
        path = local_path(root, row['archive_path'])
        if not path.is_file() or digest(path.read_bytes()) != row['sha256']:
            errors.append(f'exposure source changed: {row["archive_path"]}')
    return errors


def run_check(root: Path, bundle: dict) -> dict[str, Any]:
    rows = [json.loads(line) for line in local_path(root, str(CATALOG)).read_text().splitlines() if line.strip()]
    errors = check_sources(root, bundle) + check_git_sources(root, bundle)
    errors += check_layout(root) + check_catalog(root, rows, bundle) + check_exposure(root, bundle)
    documents = sorted((root / RESEARCH).rglob('*.md'))
    live_errors, valid_links, external = check_live_links(root, bundle, documents)
    errors += live_errors
    gaps = []
    for capture in bundle['captures']:
        for entry in capture['files']:
            path = local_path(root, entry['archive'])
            if path.suffix != '.md' or not path.is_file():
                continue
            for target in links_in(path.read_text()):
                ref = resolve_reference(root, bundle, entry['archive'], target)
                if ref['kind'] == 'local' and not ref['exists'] and 'recovery_git_spec' not in ref:
                    gaps.append({'source': entry['source'], 'target': target})
    total = sum(len(c['files']) for c in bundle['captures'])
    return {'ok': not errors, 'errors': errors, 'source_versions': total,
            'materialized_source_versions': total - len(bundle.get('retirements', [])),
            'git_recoverable_preexisting_retirements': len(bundle.get('retirements', [])),
            'catalog_entries': len(rows), 'catalogued_protocols': sum(len(r['protocols']) for r in rows),
            'current_state_owners': sum(r['tracking'] == 'current' for r in rows),
            'live_documents': len(documents), 'valid_live_local_links': valid_links,
            'unverified_external_live_handles': external,
            'frozen_exposure_records': len(bundle['exposure']['records']),
            'historical_unresolved_links': len(gaps), 'historical_gap_examples': gaps[:12],
            'scope': 'Local knowledge plumbing/source identity only; no scientific replication, external-artifact certification, or authorization to resume.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    resolve = sub.add_parser('resolve')
    resolve.add_argument('document')
    resolve.add_argument('target')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        bundle = load_bundle(root)
        result = (run_check(root, bundle) if args.command == 'check' else
                  resolve_reference(root, bundle, args.document, args.target))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(not result['ok']) if args.command == 'check' else 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
