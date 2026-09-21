"""Lead CPU acceptance: replay saved consumers without changing sealed outputs."""
import collections
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

R = Path(__file__).resolve().parent
sys.path.insert(0, '/data/CoordExp/.worktrees/research-probes')
read = lambda p: json.loads(Path(p).read_text())
torch.set_num_threads(2)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    terminal, panel = read(R / 'terminal.json'), read(R / 'panel.json')
    archive_file = R / 'candidate-records' / 'manifest.json'
    archive = read(archive_file) if archive_file.exists() else {}
    bindings = terminal['bindings'] + terminal['runtime_artifacts'] + panel['sources']
    verified = {}
    for item in bindings:
        original_path = str(Path(item['path']).resolve())
        path = Path(archive.get(original_path, {}).get('path', original_path))
        if original_path not in verified:
            verified[original_path] = sha(path)
        assert verified[original_path] == item['sha256'], str(path)
        if 'size_bytes' in item:
            assert path.stat().st_size == item['size_bytes'], str(path)

    saved_writes = {}
    def intercept(path, content, *args, **kwargs):
        assert path in (R / 'result.json', R / 'saved-verification.json'), str(path)
        saved_writes[path.name] = json.loads(content)
        return len(content)

    with patch.object(Path, 'write_text', intercept), contextlib.redirect_stdout(io.StringIO()):
        module('lead_saved_verifier', R / 'verify_saved.py').verify()
        module('lead_saved_reducer', R / 'reduce.py').main()
    for name, actual in saved_writes.items():
        assert actual == read(R / name), name
    result = saved_writes['result.json']
    fresh_only = read(R / 'fresh-result.json')
    fresh = {k: v for k, v in result['images'].items() if v['cohort'] == 'fresh'}
    assert len(fresh) == 128
    for name in ('fresh', 'healthy', 'unhealthy'):
        assert result[name] == fresh_only[name]
    assert fresh == fresh_only['images']

    rows = torch.load(R / 'effective-readout.pt', weights_only=True)
    factors = torch.load(R / 'coefficients.pt', weights_only=True)
    norms = rows['output_rows'].double().norm(dim=1)
    assert torch.equal(norms, factors['norms'])
    assert torch.equal(norms.median() / norms, factors['factors'])

    original = [json.loads(x) for x in (R / 'cohort/exact128records.jsonl').read_text().splitlines()]
    runtime = [json.loads(x) for x in (R / 'runtime-records.jsonl').read_text().splitlines()]
    manifest = read(R / 'cohort/cohort_manifest.json')
    source_root = Path(manifest['source_population']['processed_root'])
    exclusions = set(read(R / 'cohort-acceptance.json')['excluded'])
    assert {str(x['image_id']) for x in original} == set(fresh)
    assert not ({int(x['image_id']) for x in original} & {int(x) for x in exclusions})
    canonical = lambda x: json.dumps(x, sort_keys=True)
    for a, b in zip(original, runtime, strict=True):
        assert a['image_id'] == b['image_id']
        assert collections.Counter(map(canonical, a['objects'])) == collections.Counter(map(canonical, b['objects']))
        assert (source_root / a['images'][0]).resolve() == (R / b['images'][0]).resolve()
    assert sum(len(panel['banks'][iid]) for iid in fresh) == result['fresh']['known_bank'] == 919

    shadow = read(R / 'shadow-summary.json')
    assert sum(x['shadow']['active_steps'] for x in fresh.values()) == shadow['active_steps']
    assert sum(x['shadow']['disagreements'] for x in fresh.values()) == shadow['disagreements']
    roles, transitions = collections.Counter(), collections.Counter()
    for value in fresh.values():
        item = value['shadow']
        roles.update(item['roles']); transitions.update(item['transitions'])
        assert value['first_policy_divergence'] == (item['first']['offset'] if item['first'] else None)
        if item['disagreements'] == 1:
            assert not value['G'] and not value['L']
    assert dict(roles) == shadow['roles'] and dict(transitions) == shadow['transitions']

    physical = read(R / 'physical-review/result.json')
    assert physical['user_stop_honored'] and physical['physical_population_estimate'] is None
    assert physical['known_bank_unchanged']
    assert physical['reported_images'] == len(physical['reported_image_ids']) == 25
    assert len(physical['unreviewed_images']) == 7
    assert len(physical['invalid_view_inventory_entries']) == 2
    jobs = read(R / 'cost-and-jobs.json')
    surviving_pids = [pid for pid in jobs['owned_pids'] if Path(f'/proc/{pid}').exists()]
    live = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(arg.endswith(b'/readout_norm_fresh.py') or arg == b'probes.training_set_completion.readout_norm_fresh' for arg in args):
            live.append(int(path.parent.name))
    assert not surviving_pids and not live, (surviving_pids, live)
    assert jobs['all_exit_zero']
    report = dict(status='passed', no_model_calls=True, verified_unique_bindings=len(verified),
                  binding_sha256=verified, preserved_candidate_records=archive,
                  exact_saved_consumer_replays=list(saved_writes),
                  fresh=result['fresh'], healthy=result['healthy'], unhealthy=result['unhealthy'],
                  shadow=dict(active_steps=shadow['active_steps'], disagreements=shadow['disagreements'],
                              roles=dict(roles), transitions=dict(transitions)),
                  sparse_capture_count=saved_writes['saved-verification.json']['first_shadow_captures'],
                  live_owned_jobs=live, surviving_original_pids=surviving_pids,
                  physical_review_scope=physical['status'], original_terminal_sha256=sha(R / 'terminal.json'))
    (R / 'lead-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('binding_sha256', 'fresh', 'healthy', 'unhealthy')}, indent=2))


if __name__ == '__main__':
    main()
