"""CPU acceptance replay; copies maintained code, forbids retired provider access."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

workspace = Path('/data/CoordExp/.worktrees/research-probes')
evidence = Path(__file__).parent
with tempfile.TemporaryDirectory(prefix='human13-independent-') as temporary:
    root = Path(temporary)
    shutil.copyfile(workspace/'pytest.ini', root/'pytest.ini')
    shutil.copytree(workspace/'src', root/'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(workspace/'probes/human13', root/'probes/human13', ignore=shutil.ignore_patterns('__pycache__'))
    script = '''import json,sys
from pathlib import Path
forbidden='/data/CoordExp/.worktrees/human13-output-qp-identity-generalization'
def guard(event,args):
    if event == 'open' and isinstance(args[0],str) and args[0].startswith(forbidden):
        raise RuntimeError('retired provider access forbidden: '+args[0])
sys.addaudithook(guard)
from probes.human13 import output_qp as q
from probes.human13.magnitude_finite import _load_finite_candidate_receipt
from src.config.inference import load_research_infer_config
config=load_research_infer_config(q.SOURCE_CONFIG).config.model_dump(mode='json')
original=json.loads(Path(sys.argv[1]).with_name('logit-lens-original-config.json').read_text())
assert config==original and config['debug']['smoke'] is False
binding=q.check_bindings()
assert binding['status']=='passed' and binding['panel_owner_count']==392
base=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-human13-shared-output-qp-same-panel-overfit/20260831T-n4-v1')
ids=q.STAGES['N4'].image_ids
result=q.aggregate_results(results=[base/f'verify-candidate-rp1-{i}'/f'verify-{i}-candidate-rp1p0.json' for i in ids],post_source=[base/f'verify-post-source-{i}'/f'verify-{i}-source-rp1p0.json' for i in ids],stage='N4')
legacy=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-human13-dora-magnitude-finite-overfit/20260901T-n2-adamw-ce-v1/human13-n2-dora-magnitude-finite.json')
receipt,_=_load_finite_candidate_receipt(legacy)
assert not any(name.startswith('scripts.research') for name in sys.modules)
out=Path(sys.argv[1]);out.write_text(json.dumps({'binding':binding,'existing_n4_readback':result,'legacy_n2_sha256':receipt['candidate_sha256'],'retired_provider_access_forbidden':True},sort_keys=True))
'''
    (root/'verify.py').write_text(script)
    environment = {**os.environ, 'PYTHONPATH': ''}
    subprocess.run(['python','verify.py',str(evidence/'human13-independent.json')], cwd=root, env=environment, check=True)
    subprocess.run(['python','-m','pytest','-q','-p','no:cacheprovider','probes/human13/tests'], cwd=root, env=environment, check=True)
print('INDEPENDENT_HUMAN13_CPU_ACCEPTANCE_OK')
