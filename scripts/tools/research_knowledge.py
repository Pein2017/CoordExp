#!/usr/bin/env python3
"""Read-only research layout, catalog, path routing and exposure-data checks.

This validates current knowledge plumbing, not scientific truth or model artifacts.
It does not recover historical implementation sources. No model imports,
filesystem writes, alias creation, or automatic repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import sys
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

# The documented direct script entry and module entry share the same checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tools.document_locations import DocumentLocations

CAPTURE = Path('docs/history/research-records/2026-09-15-root-collapse')
RESEARCH = Path('research')
CATALOG = RESEARCH / 'experiments/catalog.jsonl'
ROOT_NAMES = {'index.md', 'CONVENTIONS.md', 'story.md', 'glossary.md',
              'alternatives.md', 'assets.md', 'questions', 'literature', 'experiments'}
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
    previous = json.loads(previous_path.read_text())
    consumers = json.loads(local_path(root, str(CAPTURE / 'consumer-sources.json')).read_text())
    return {'canonical_root': latest['canonical_root'],
            'captures': [previous, latest, consumers],
            'exposure': latest['research_json_exposure'], 'locations': DocumentLocations(root)}


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
                    current = bundle['locations'].target(proposed) if 'locations' in bundle else proposed
                    if local_path(root, current).is_dir():
                        resolved = proposed
                        break
    if 'locations' in bundle:
        resolved = bundle['locations'].target(resolved)
    try:
        exists = local_path(root, resolved).exists()
    except ValueError:
        return {'kind': 'invalid', 'logical_path': logical, 'exists': None}
    result = {'kind': 'local', 'logical_path': logical, 'path': resolved,
              'fragment': parts.fragment, 'exists': exists}
    if not exists:
        historical = historical_source_reference(root, doc, logical)
        if historical is not None:
            return {**historical, 'fragment': parts.fragment}
    return result


def git_bytes(root: Path, *arguments: str) -> bytes:
    """Read an exact Git object; never check out, execute or recreate a file."""
    return subprocess.run(
        ['git', '-C', str(root), *arguments], check=True, capture_output=True,
        timeout=5,
    ).stdout


def historical_source_reference(root: Path, document: str, target: str) -> dict | None:
    """Resolve only unchanged closed-unit source citations at the document commit.

    This proves a document-version reference exists, not that the source was
    executed or that the current implementation is equivalent. Current routers,
    edited/new documents, missing data and ordinary broken links get no fallback.
    """
    doc, source = Path(document), Path(target)
    if (len(doc.parts) < 4 or doc.parts[:2] != ('research', 'experiments')
            or source.parts[:1] not in (('src',), ('probes',), ('scripts',))
            or source.suffix not in {'.py', '.sh'}):
        return None
    try:
        document_path = local_path(root, document)
        local_path(root, target)
        state = json.loads(local_path(root, str(Path(*doc.parts[:3]) / 'state.json')).read_text())
        if state.get('lifecycle') not in {'closed', 'superseded'}:
            return None
        commit = git_bytes(root, 'log', '-1', '--format=%H', 'HEAD', '--', document).decode().strip()
        if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
            return None
        if git_bytes(root, 'show', f'{commit}:{document}') != document_path.read_bytes():
            return None
        entry = git_bytes(root, 'ls-tree', '-z', commit, '--', target)
        if not entry.startswith((b'100644 blob ', b'100755 blob ')):
            return None
        content = git_bytes(root, 'show', f'{commit}:{target}')
        return {'kind': 'historical-source', 'exists': False,
                'document': document, 'document_commit': commit, 'git_path': target,
                'sha256': digest(content), 'size_bytes': len(content),
                'scope': 'document-version source reference only; not current execution or receipt recovery'}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


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


def check_catalog(root: Path, rows: list[dict], bundle: dict | None = None) -> list[str]:
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
    actual_states = {str(p.relative_to(root)) for p in (root / RESEARCH / 'experiments').rglob('state.json')}
    if states != actual_states:
        errors.append('orphan/missing current state')
    entry = (root / RESEARCH / 'index.md').read_text()
    if not catalog_linked(entry):
        errors.append('frontier omits experiment catalog link')
    return errors


def check_migration_catalog(root: Path, rows: list[dict], bundle: dict) -> list[str]:
    """Audit the preserved historical intake, separately from current ownership."""
    errors = []
    protocols = {path for row in rows for path in row.get('protocols', [])}
    ids = {row['id'] for row in rows}
    expected_protocols = {(bundle['locations'].target(e['archive']) if 'locations' in bundle else e['archive'])
                          for c in bundle['captures'] for e in c['files']
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
    return errors


def catalog_linked(text: str) -> bool:
    """Recognize an actual inline catalog link, not an example or image.

    The catalog above owns complete state discovery. This only checks its
    entry route; selected result/state links still pass normal live-link checks.
    """
    visible = FENCED.sub('', re.sub(r'(?s)<!--.*?-->', '', text))
    visible = re.sub(r'`+[^`\n]*`+', '', visible)
    for target in re.findall(r'(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)', visible):
        parts = urlsplit(target.strip().strip('<>'))
        if parts.scheme or parts.netloc or parts.path.startswith('/'):
            continue
        path = posixpath.normpath(posixpath.join('research', unquote(parts.path)))
        if path == CATALOG.as_posix():
            return True
    return False


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
    code_suffixes = {'.py', '.pyi', '.pyc', '.pyo', '.sh', '.bash', '.zsh', '.js',
                     '.ts', '.c', '.h', '.cpp', '.cu', '.ipynb', '.patch', '.diff', '.yaml', '.yml'}
    for path in (root / 'docs').rglob('*'):
        if path == root / 'docs/catalog.yaml':
            continue  # Documentation metadata, not an executable configuration.
        if path.suffix in code_suffixes or path.name == '__pycache__':
            errors.append(f'code/cache in docs: {path.relative_to(root)}')
    return errors


def check_live_links(root: Path, bundle: dict, documents: list[Path],
                     historical_sources: list[dict] | None = None) -> tuple[list[str], int, int]:
    errors, local_count, external_count = [], 0, 0
    for document in documents:
        rel = str(document.relative_to(root))
        for target in links_in(document.read_text()):
            ref = resolve_reference(root, bundle, rel, target)
            if ref['kind'] == 'external':
                external_count += 1
            elif ref['kind'] == 'historical-source':
                if historical_sources is not None:
                    historical_sources.append(ref)
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


def run_check(root: Path, bundle: dict | None = None) -> dict[str, Any]:
    rows = [json.loads(line) for line in local_path(root, str(CATALOG)).read_text().splitlines() if line.strip()]
    include_history = bundle is not None
    errors = check_layout(root) + check_catalog(root, rows)
    if include_history:
        errors += check_migration_catalog(root, rows, bundle) + check_exposure(root, bundle)
    else:
        bundle = {'canonical_root': str(root.resolve()), 'captures': []}
    all_documents = sorted((root / RESEARCH).rglob('*.md'))
    # Historical scientific records remain useful in research. They do not become
    # live frontiers merely by moving, and old unresolved citations stay disclosed.
    historical_roots = [root / row['record_root'] for row in rows if row['tracking'] == 'historical']
    documents = [p for p in all_documents if not any(p.is_relative_to(d) for d in historical_roots)]
    historical_sources = []
    live_errors, valid_links, external = check_live_links(root, bundle, documents, historical_sources)
    errors += live_errors
    return {'ok': not errors, 'errors': errors,
            'historical_source_references': historical_sources,
            'catalog_entries': len(rows), 'catalogued_protocols': sum(len(r['protocols']) for r in rows),
            'current_state_owners': sum(r['tracking'] == 'current' for r in rows),
            'live_documents': len(documents), 'retained_research_documents': len(all_documents) - len(documents),
            'valid_live_local_links': valid_links,
            'unverified_external_live_handles': external,
            'frozen_exposure_records': len(bundle['exposure']['records']) if include_history else None,
            'historical_checks': 'checked' if include_history else 'not_requested',
            'scope': 'Current knowledge ownership and links, plus explicit historical checks when requested; no source-code recovery or scientific revalidation.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check')
    check.add_argument('--live-only', action='store_true', help='Check current knowledge without migration snapshots or exposure inputs')
    resolve = sub.add_parser('resolve')
    resolve.add_argument('document')
    resolve.add_argument('target')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        bundle = None if args.command == "check" and args.live_only else load_bundle(root)
        result = (run_check(root, bundle) if args.command == 'check' else
                  resolve_reference(root, bundle, args.document, args.target))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(not result['ok']) if args.command == 'check' else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
