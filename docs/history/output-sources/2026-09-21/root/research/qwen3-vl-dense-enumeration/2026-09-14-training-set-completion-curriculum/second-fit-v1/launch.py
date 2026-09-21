import hashlib, json, os, shlex, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CWD=Path('/data/CoordExp/.worktrees/research-probes')
SESSION='coordexp-second-fit-v1'
manifest=ROOT.parent/'second-fit-preparation-v1/manifest.json'
assert manifest.is_file() and not (ROOT/'launch.json').exists()
assert subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode != 0
assert Path(sys.executable).parent == Path(os.environ['CONDA_PREFIX'])/'bin'
command='cd '+shlex.quote(str(CWD))+' && exec python '+shlex.quote(str(ROOT/'run.py'))+' > '+shlex.quote(str(ROOT/'phase.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',SESSION,'-e','PATH='+os.environ['PATH'],'-e','CONDA_PREFIX='+os.environ['CONDA_PREFIX'],command],check=True)
pid=int(subprocess.check_output(['tmux','display-message','-p','-t',SESSION,'#{pane_pid}'],text=True).strip())
os.kill(pid,0)
receipt={'status':'launched_pending_model_evidence','session':SESSION,'pid':pid,'command':command,'selected_python':sys.executable,'manifest':{'path':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()},'phase_log':str(ROOT/'phase.log'),'expected_terminal':str(ROOT/'terminal.json')}
with (ROOT/'launch.json').open('x') as f: json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
