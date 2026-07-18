from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "launch_coco_refinement_gate_a.sh"


def test_fixed_gate_a_launcher_reports_stable_direct_endpoint(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--print-config"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        f"repo_root={REPO_ROOT}",
        f"runtime_root={REPO_ROOT / 'outputs/coco_refinement/gate-a-20260717'}",
        "bind_url=http://127.0.0.1:53662/",
        "browser_url=http://localhost:53662/",
    ]


def test_fixed_gate_a_launcher_rejects_unknown_arguments() -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--port", "12345"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "fixed at 53662" in result.stderr


def test_fixed_gate_a_launcher_explains_prebind_validation() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "validates the full workspace before binding the port" in source
    assert "VS Code can offer Forward/Open after Uvicorn reports" in source
    assert "--allow-browser-port-remap" in source


def test_fixed_gate_a_launcher_releases_an_existing_listener() -> None:
    listener = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import socket; "
                "s=socket.socket(); "
                "s.bind(('127.0.0.1', 0)); "
                "s.listen(); "
                "print(s.getsockname()[1], flush=True); "
                "s.accept()"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert listener.stdout is not None
        port = int(listener.stdout.readline().strip())
        command = (
            f"source {shlex.quote(str(LAUNCHER))}; "
            f"release_listeners_on_port {port}"
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        assert listener.wait(timeout=5) == -15
        assert f"Port {port} is occupied" in result.stdout
        assert f"Port {port} was released cleanly" in result.stdout
    finally:
        if listener.poll() is None:
            listener.kill()
            listener.wait(timeout=5)
