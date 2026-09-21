import json, subprocess, time
from pathlib import Path

ROOT = Path('/data/CoordExp')
OUT = ROOT / 'outputs/label_studio_coco_refinement/probes/runtime_probe_20260715.json'
probes = []

def run(name, cmd, cwd=ROOT, timeout=30):
    t = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        rc, so, se = p.returncode, p.stdout, p.stderr
    except Exception as e:
        rc, so, se = 124, '', repr(e)
    probes.append({'name': name, 'command': ' '.join(cmd), 'cwd': str(cwd),
                   'exit': rc, 'elapsed_sec': round(time.monotonic()-t, 3),
                   'stdout': so[-4000:], 'stderr': se[-4000:]})

run('git_revision', ['git', 'rev-parse', 'HEAD'])
for name, cmd in [('node_version', ['node', '--version']), ('corepack_version', ['corepack', '--version']),
                  ('yarn_version', ['yarn', '--version']), ('pnpm_version', ['pnpm', '--version']),
                  ('npm_version', ['npm', '--version'])]: run(name, cmd)
web = ROOT / 'label-studio/web'
run('web_node_modules_present', ['bash', '-lc', 'if test -d node_modules; then echo present; else echo absent; exit 1; fi'], web)
run('web_declared_scripts', ['node', '-e', "const p=require('./package.json'); console.log(Object.keys(p.scripts).join('\\n'))"], web)
run('web_yarn_scripts', ['yarn', 'run'], web, 20)
run('web_npm_run_listing', ['npm', 'run'], web, 20)
run('web_package_manager_files', ['bash', '-lc', "find . -maxdepth 1 -type f \( -name '*lock*' -o -name '.yarnrc*' \) -print | sort"], web)
run('ms_python_version', ['python', '--version'])
run('label_studio_import', ['python', '-c', 'import label_studio; print(label_studio.__file__)'], ROOT/'label-studio')
run('django_import', ['python', '-c', 'import django; print(django.get_version())'], ROOT/'label-studio')
run('pip_show_label_studio', ['python', '-m', 'pip', 'show', 'label-studio'])
run('pip_show_django', ['python', '-m', 'pip', 'show', 'Django'])
run('manage_check', ['python', 'label_studio/manage.py', 'check'], ROOT/'label-studio', 45)
run('python_requirements_metadata', ['python', '-c', "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d.get('tool',{}).get('poetry',{}).get('name')); print(d.get('tool',{}).get('poetry',{}).get('version'))"], ROOT/'label-studio')

obj = {'probe': 'label-studio runtime probe', 'timestamp_utc': '2026-07-15',
       'root': str(ROOT), 'git_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
       'probes': probes, 'network_policy': 'no installs/downloads; local version/check/import only'}
OUT.write_text(json.dumps(obj, indent=2) + '\n')
print(OUT)
