"""Replay a maintained-package-only CPU preflight and consumer tests."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

workspace = Path('/data/CoordExp/.worktrees/research-probes')
evidence = Path(__file__).parent
with tempfile.TemporaryDirectory(prefix='logit-lens-independent-') as temporary:
    root = Path(temporary)
    shutil.copyfile(workspace/'pytest.ini', root/'pytest.ini')
    shutil.copytree(workspace/'src', root/'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(workspace/'probes/logit_lens', root/'probes/logit_lens', ignore=shutil.ignore_patterns('__pycache__'))
    (root/'verify.py').write_text('''import json,sys
from pathlib import Path
forbidden='/data/CoordExp/.worktrees/'
def guard(event,args):
    if event == 'open' and isinstance(args[0],str) and args[0].startswith(forbidden):
        raise RuntimeError('worktree access forbidden: '+args[0])
sys.addaudithook(guard)
from probes.logit_lens.preflight import check
report=check(output_root=Path('preflight-output'))
original=json.loads(Path(sys.argv[1]).with_name('logit-lens-original-reader.json').read_text())
assert report['images']==original
original_config=json.loads(Path(sys.argv[1]).with_name('logit-lens-original-config.json').read_text())
assert report['config']['config']==original_config
assert report['config']['config']['debug']['smoke'] is False
assert not any(name.startswith('scripts.research') for name in sys.modules)
report['worktree_access_forbidden']=True
report['original_saved_reader_exact_parity']=True
Path(sys.argv[1]).write_text(json.dumps(report,sort_keys=True))
print('INDEPENDENT_PREFLIGHT_EXACT_PARITY',len(report['images']))
''')
    environment = {**os.environ, 'PYTHONPATH': ''}
    subprocess.run(['python','verify.py',str(evidence/'logit-lens-independent.json')], cwd=root, env=environment, check=True)
    subprocess.run(['python','-m','pytest','-q','-p','no:cacheprovider','probes/logit_lens/tests'], cwd=root, env=environment, check=True)
print('INDEPENDENT_LOGIT_LENS_CPU_ACCEPTANCE_OK')
