import json,os,subprocess,time
from pathlib import Path

cmd = ['python', 'probes/training_set_completion/readout_direction/runtime.py', 'run', '--panel', '/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json', '--plan', '/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-readout-direction-control/execution-plan.json', '--output', '/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-readout-direction-control/runtime/scaleout-gpu2-tied', '--device', 'cuda:0', '--max-forwards', '9000', '--max-seconds', '3000', '--max-bytes', '1932735283', '--cell-id', 'tied-885-failure--reflected', '--cell-id', 'tied-885-failure--sign22', '--cell-id', 'tied-885-failure--sign26', '--cell-id', 'tied-5586-failure--sign19', '--cell-id', 'tied-5586-failure--sign23', '--cell-id', 'tied-14038-failure--original', '--cell-id', 'tied-14038-failure--sign20', '--cell-id', 'tied-14038-failure--sign24', '--cell-id', 'tied-632-failure--full', '--cell-id', 'tied-632-failure--sign21', '--cell-id', 'tied-632-failure--sign25', '--cell-id', 'tied-417044-failure--reflected', '--cell-id', 'tied-417044-failure--sign22', '--cell-id', 'tied-417044-failure--sign26']
cwd = '/data/CoordExp/.worktrees/research-probes'
log_path = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-readout-direction-control/runtime/scaleout-gpu2-tied.log')
exit_path = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-readout-direction-control/runtime/scaleout-gpu2-tied.exit.json')
gpu = 2
log_path.parent.mkdir(parents=True, exist_ok=True)
started = time.time()
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu)
env["PYTHONUNBUFFERED"] = "1"
try:
    with log_path.open("w") as stream:
        completed = subprocess.run(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    returncode = completed.returncode
    error = None
except BaseException as exc:
    returncode = None
    error = repr(exc)
ended = time.time()
payload = {
    "schema": "readout_direction.scaleout_exit.v1",
    "status": "ended" if returncode == 0 else "failed",
    "wrapper_pid": os.getpid(),
    "gpu": gpu,
    "command": cmd,
    "cwd": cwd,
    "log": str(log_path),
    "exit_code": returncode,
    "exit_status_observed": returncode is not None,
    "error": error,
    "started_at": started,
    "ended_at": ended,
    "wall_seconds": ended - started,
}
tmp = exit_path.with_suffix(exit_path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, exit_path)
raise SystemExit(returncode if returncode is not None else 1)
