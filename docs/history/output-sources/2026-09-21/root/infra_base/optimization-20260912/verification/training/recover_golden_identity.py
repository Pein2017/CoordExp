"""Recover the pre-rename golden using only an in-memory planner replacement."""

import ast
import dataclasses
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path('/data/CoordExp/.worktrees/coordexp-infras')
OUTPUT = Path(__file__).with_name('golden-identity.json')
OLD_REF = '70e576f96^'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tests/training'))
import test_micro_step_payload_identity as fixture
import src.packing.planner as planner


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True)


def summarize(payload):
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


with pytest.MonkeyPatch.context() as patch:
    current = fixture._canonical_payload_bytes(fixture._assemble(patch))
current_source = Path(planner.__file__).read_text()
old_source = git('show', f'{OLD_REF}:src/packing/planner.py')
old_test = ast.parse(git('show', f'{OLD_REF}:tests/training/test_micro_step_payload_identity.py'))
old_constants = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in old_test.body
    if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id.startswith('GOLDEN_PAYLOAD_')
}

# Re-execution also reconstructs prefix hashes derived at module import time.
exec(compile(old_source, planner.__file__, 'exec'), planner.__dict__)
# Keep import aliases coherent after replacing planner classes/functions.
for name, module in list(sys.modules.items()):
    if name.startswith('src.') and module is not planner:
        for key, value in vars(module).copy().items():
            if getattr(value, '__module__', None) == 'src.packing.planner':
                symbol = getattr(value, '__name__', '')
                if hasattr(planner, symbol):
                    setattr(module, key, getattr(planner, symbol))
with pytest.MonkeyPatch.context() as patch:
    recovered = fixture._canonical_payload_bytes(fixture._assemble(patch))

changes = []


def compare(old, new, path='$'):
    if dataclasses.is_dataclass(old):
        assert dataclasses.is_dataclass(new), path
        assert [f.name for f in dataclasses.fields(old)] == [f.name for f in dataclasses.fields(new)], path
        for field in dataclasses.fields(old):
            compare(getattr(old, field.name), getattr(new, field.name), path + '.' + field.name)
    elif isinstance(old, dict):
        assert isinstance(new, dict) and list(old) == list(new), path
        for key in old:
            compare(old[key], new[key], path + '.' + str(key))
    elif isinstance(old, (tuple, list)):
        assert type(old) is type(new) and len(old) == len(new), path
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            compare(left, right, path + f'[{index}]')
    elif hasattr(old, '__dict__'):
        compare(vars(old), vars(new), path)
    elif old != new:
        changes.append({'path': path, 'old': old, 'current': new})


compare(pickle.loads(recovered), pickle.loads(current))
golden = {'bytes': old_constants['GOLDEN_PAYLOAD_BYTE_LENGTH'], 'sha256': old_constants['GOLDEN_PAYLOAD_SHA256']}
assert summarize(recovered) == golden
assert [item['path'] for item in changes] == [
    '$[0].metadata.pack_plan.policy_identity.algorithm_version',
    '$[0].metadata.pack_plan.plan_sha256',
    '$[0].metadata.pack_plan.fragment_sha256',
]
receipt = {
    'schema': 'coordexp-infras-golden-identity-recovery-v1',
    'command': f'python {Path(__file__).resolve()}',
    'cwd': str(REPO),
    'head': git('rev-parse', 'HEAD').strip(),
    'historical_ref': OLD_REF,
    'historical_commit': git('rev-parse', OLD_REF).strip(),
    'historical_planner_source_sha256': hashlib.sha256(old_source.encode()).hexdigest(),
    'current_planner_source_sha256': hashlib.sha256(current_source.encode()).hexdigest(),
    'historical_golden': golden,
    'recovered_historical_payload': summarize(recovered),
    'current_payload': summarize(current),
    'historical_golden_recovered_exactly': True,
    'changed_leaf_count': len(changes),
    'changes': changes,
    'all_other_decoded_fields_equal': True,
    'method': 'Current assembler and fixture; replace only planner with historical source in memory, including derived import-time hashes and imported symbol aliases; compare every decoded field.',
}
OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps({'receipt': str(OUTPUT), 'recovered': summarize(recovered), 'current': summarize(current), 'changed_leaves': len(changes)}, sort_keys=True))
