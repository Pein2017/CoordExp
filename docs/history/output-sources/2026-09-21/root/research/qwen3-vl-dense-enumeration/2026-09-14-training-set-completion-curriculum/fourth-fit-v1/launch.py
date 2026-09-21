"""Launch the reviewed fourth-fit controller in one detached tmux session."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKTREE = Path("/data/CoordExp/.worktrees/research-probes").resolve()
MANIFEST = ROOT.parent / "fourth-fit-preparation-v1/manifest.json"
RUN = ROOT / "run.py"
MS_PYTHON = Path("/root/miniconda3/envs/ms/bin/python").resolve()
SESSION = "coordexp-fourth-fit-v1"


def main() -> None:
    assert MANIFEST.is_file(), f"continuation manifest missing: {MANIFEST}"
    assert RUN.is_file() and MS_PYTHON.is_file()
    assert Path(sys.executable).resolve() == MS_PYTHON, "launch must use selected ms interpreter"
    assert subprocess.run(["tmux", "has-session", "-t", SESSION], capture_output=True).returncode != 0
    command = "cd " + shlex.quote(str(WORKTREE)) + " && exec " + shlex.quote(str(MS_PYTHON)) + " " + shlex.quote(str(RUN)) + " controller > " + shlex.quote(str(ROOT / "phase.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "-e", "PATH=" + str(MS_PYTHON.parent) + ":" + os.environ.get("PATH", ""), "-e", "CONDA_PREFIX=" + str(MS_PYTHON.parent.parent), command], check=True)
    pid = int(subprocess.check_output(["tmux", "display-message", "-p", "-t", SESSION, "#{pane_pid}"], text=True).strip())
    os.kill(pid, 0)
    receipt = {"schema": "training_set_completion.fourth_fit_launch.v1", "status": "launched_pending_model_evidence",
               "session": SESSION, "pid": pid, "command": command, "selected_python": str(MS_PYTHON),
               "manifest": {"path": str(MANIFEST), "sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), "size_bytes": MANIFEST.stat().st_size},
               "run_wrapper": str(RUN), "phase_log": str(ROOT / "phase.log"), "expected_terminal": str(ROOT / "terminal.json")}
    with (ROOT / "launch.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
