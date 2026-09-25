"""Historical references must never masquerade as current executable files."""
import json
import subprocess

import pytest

from scripts.tools import research_knowledge as knowledge


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    document = 'research/experiments/closed/results.md'
    target = 'probes/old/run.py'
    path = tmp_path / document
    path.parent.mkdir(parents=True)
    text = b'[producer](../../../probes/old/run.py)\n'
    path.write_bytes(text)
    (path.parent/'state.json').write_text(json.dumps({'lifecycle': 'closed'}))
    commit = 'a' * 40

    def git(root, *args):
        assert root == tmp_path
        if args == ('log', '-1', '--format=%H', 'HEAD', '--', document):
            return (commit + '\n').encode()
        if args == ('show', f'{commit}:{document}'):
            return text
        if args == ('ls-tree', '-z', commit, '--', target):
            return b'100644 blob ' + b'b'*40 + b'\tprobes/old/run.py\x00'
        if args == ('show', f'{commit}:{target}'):
            return b'original source bytes\n'
        raise subprocess.CalledProcessError(128, args)

    monkeypatch.setattr(knowledge, 'git_bytes', git)
    return document, target


def test_original_document_and_git_blob_are_explicit_historical_reference(tmp_path, frozen):
    document, target = frozen
    ref = knowledge.historical_source_reference(tmp_path, document, target)
    assert ref['kind'] == 'historical-source' and ref['exists'] is False
    assert ref['document_commit'] == 'a' * 40
    assert ref['sha256'] == knowledge.digest(b'original source bytes\n')
    refs = []
    errors, live, external = knowledge.check_live_links(
        tmp_path, {'canonical_root': str(tmp_path), 'captures': []},
        [tmp_path/document], refs)
    assert errors == [] and live == external == 0 and len(refs) == 1
    assert not (tmp_path/target).exists()


def test_changed_document_and_active_state_cannot_use_old_git_source(tmp_path, frozen):
    document, target = frozen
    path = tmp_path/document
    old = path.read_bytes()
    path.write_bytes(old + b'changed\n')
    assert knowledge.historical_source_reference(tmp_path, document, target) is None
    path.write_bytes(old)
    (path.parent/'state.json').write_text(json.dumps({'lifecycle': 'running'}))
    assert knowledge.historical_source_reference(tmp_path, document, target) is None


@pytest.mark.parametrize('target', ['missing.json', 'research/missing.md',
                                   'probes/old/missing.py', '../escape.py'])
def test_missing_data_unknown_source_and_escape_stay_errors(tmp_path, frozen, target):
    document, _ = frozen
    assert knowledge.historical_source_reference(tmp_path, document, target) is None


def test_current_router_and_git_symlink_are_not_historical_source(tmp_path, frozen, monkeypatch):
    document, target = frozen
    assert knowledge.historical_source_reference(tmp_path, 'research/index.md', target) is None
    original = knowledge.git_bytes

    def symlink(root, *args):
        if args[0] == 'ls-tree':
            return b'120000 blob ' + b'b'*40 + b'\tprobes/old/run.py\x00'
        return original(root, *args)

    monkeypatch.setattr(knowledge, 'git_bytes', symlink)
    assert knowledge.historical_source_reference(tmp_path, document, target) is None
