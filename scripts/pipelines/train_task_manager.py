#!/usr/bin/env python
"""Queue-driven tmux training manager.

This module owns the orchestration logic for sequential experiment execution:
- attach to an already running tmux session or launch a new command
- wait before monitoring (`monitor_after_seconds`)
- stop on flexible criteria (`global_step`, `elapsed_seconds`)
- stop tmux safely, then optionally wait for GPU memory to drain to zero
- continue to the next queued task

Queue format:
- JSONL
- blank lines and lines beginning with `#` are ignored

Backwards-compatible queue entries still work:
    {"session":"a_only","launch":false,"config":"configs/...yaml","stop_after_step":300,"require_eval":true}

Preferred flexible queue entries use:
    {
      "session": "a_only",
      "launch": false,
      "config": "configs/...yaml",
      "monitor_after_seconds": 7200,
      "poll_seconds": 600,
      "criteria_mode": "all",
      "criteria": [
        {"type": "global_step", "value": 300, "require_eval": true}
      ],
      "post_stop_wait_seconds": 15,
      "gpu_zero": {
        "enabled": true,
        "devices": "all",
        "timeout_seconds": 300,
        "poll_seconds": 5,
        "memory_threshold_mb": 0
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config.loader import ConfigLoader


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"[ERROR] {message}")


@dataclass(frozen=True)
class ManagerDefaults:
    poll_seconds: float
    run_dir_timeout_s: float
    stop_timeout_s: float
    post_stop_wait_s: float
    gpu_zero_timeout_s: float
    gpu_zero_poll_seconds: float


@dataclass(frozen=True)
class ProgressState:
    source: str
    max_step: int
    max_eval_step: int


@dataclass(frozen=True)
class CriterionSpec:
    kind: str
    value: float
    require_eval: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, line_no: int) -> "CriterionSpec":
        kind = str(raw.get("type", "") or "").strip()
        if kind not in {"global_step", "elapsed_seconds"}:
            fail(
                f"Queue line {line_no}: criterion type must be one of "
                f"'global_step' or 'elapsed_seconds', got {kind!r}."
            )

        value_raw = raw.get("value")
        try:
            value = float(value_raw)
        except (TypeError, ValueError):
            fail(f"Queue line {line_no}: criterion value must be numeric, got {value_raw!r}.")
        if value < 0:
            fail(f"Queue line {line_no}: criterion value must be >= 0, got {value}.")

        require_eval = bool(raw.get("require_eval", False))
        if kind != "global_step" and require_eval:
            fail(
                f"Queue line {line_no}: require_eval is only valid for global_step criteria."
            )

        return cls(kind=kind, value=value, require_eval=require_eval)

    def describe(self) -> str:
        if self.kind == "global_step":
            suffix = " + eval" if self.require_eval else ""
            return f"global_step>={int(self.value)}{suffix}"
        return f"elapsed_seconds>={self.value:g}"

    def is_satisfied(self, *, progress: ProgressState | None, elapsed_seconds: float) -> bool:
        if self.kind == "elapsed_seconds":
            return elapsed_seconds >= self.value
        if progress is None:
            return False
        if self.require_eval:
            return progress.max_eval_step >= int(self.value)
        return progress.max_step >= int(self.value)


@dataclass(frozen=True)
class GPUZeroSpec:
    enabled: bool
    devices: tuple[int, ...] | None
    timeout_seconds: float
    poll_seconds: float
    memory_threshold_mb: int

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        defaults: ManagerDefaults,
        fallback_gpus: str | None,
        line_no: int,
    ) -> "GPUZeroSpec":
        if raw is None:
            return cls(
                enabled=False,
                devices=None,
                timeout_seconds=defaults.gpu_zero_timeout_s,
                poll_seconds=defaults.gpu_zero_poll_seconds,
                memory_threshold_mb=0,
            )

        enabled = bool(raw.get("enabled", True))
        devices = parse_gpu_devices(raw.get("devices"), fallback_gpus=fallback_gpus, line_no=line_no)

        timeout_raw = raw.get("timeout_seconds", defaults.gpu_zero_timeout_s)
        poll_raw = raw.get("poll_seconds", defaults.gpu_zero_poll_seconds)
        threshold_raw = raw.get("memory_threshold_mb", 0)

        timeout_seconds = parse_nonnegative_float(timeout_raw, label="gpu_zero.timeout_seconds", line_no=line_no)
        poll_seconds = parse_positive_float(poll_raw, label="gpu_zero.poll_seconds", line_no=line_no)
        threshold_mb = parse_nonnegative_int(
            threshold_raw,
            label="gpu_zero.memory_threshold_mb",
            line_no=line_no,
        )

        return cls(
            enabled=enabled,
            devices=devices,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            memory_threshold_mb=threshold_mb,
        )


@dataclass(frozen=True)
class TaskSpec:
    name: str
    session: str
    launch: bool
    detach_after_launch: bool
    command: str
    config_raw: str
    launcher: str
    gpus: str
    run_dir_raw: str
    env: Mapping[str, Any]
    monitor_after_seconds: float
    poll_seconds: float
    criteria_mode: str
    criteria: tuple[CriterionSpec, ...]
    stop_timeout_s: float
    post_stop_wait_seconds: float
    gpu_zero: GPUZeroSpec
    line_no: int

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        defaults: ManagerDefaults,
        line_no: int,
    ) -> "TaskSpec":
        session = str(raw.get("session", "") or "").strip()
        if not session:
            fail(f"Queue line {line_no}: session must be a non-empty string.")

        launch = bool(raw.get("launch", True))
        detach_after_launch = bool(raw.get("detach_after_launch", False))
        command = str(raw.get("command", "") or "")
        config_raw = str(raw.get("config", "") or "")
        launcher = str(raw.get("launcher", "") or "scripts/train.sh")
        gpus = str(raw.get("gpus", "") or "")
        run_dir_raw = str(raw.get("run_dir", "") or "")
        env = raw.get("env", {})
        if env and not isinstance(env, Mapping):
            fail(f"Queue line {line_no}: env must be an object/dict.")

        monitor_after_seconds = parse_nonnegative_float(
            raw.get("monitor_after_seconds", 0),
            label="monitor_after_seconds",
            line_no=line_no,
        )
        poll_seconds = parse_positive_float(
            raw.get("poll_seconds", defaults.poll_seconds),
            label="poll_seconds",
            line_no=line_no,
        )
        stop_timeout_s = parse_positive_float(
            raw.get("stop_timeout_seconds", defaults.stop_timeout_s),
            label="stop_timeout_seconds",
            line_no=line_no,
        )
        post_stop_wait_seconds = parse_nonnegative_float(
            raw.get("post_stop_wait_seconds", defaults.post_stop_wait_s),
            label="post_stop_wait_seconds",
            line_no=line_no,
        )

        criteria_mode = str(raw.get("criteria_mode", "all") or "all").strip().lower()
        if criteria_mode not in {"all", "any"}:
            fail(f"Queue line {line_no}: criteria_mode must be 'all' or 'any'.")

        criteria = parse_task_criteria(
            raw,
            line_no=line_no,
            detach_after_launch=detach_after_launch,
        )
        if launch and not command and not config_raw:
            fail(
                f"Queue line {line_no}: launch=true requires either command or config."
            )
        if detach_after_launch and not launch:
            fail(
                f"Queue line {line_no}: detach_after_launch=true requires launch=true."
            )

        gpu_zero = GPUZeroSpec.from_mapping(
            raw.get("gpu_zero"),
            defaults=defaults,
            fallback_gpus=gpus or None,
            line_no=line_no,
        )

        name = str(raw.get("name", "") or f"task-line-{line_no}")
        return cls(
            name=name,
            session=session,
            launch=launch,
            detach_after_launch=detach_after_launch,
            command=command,
            config_raw=config_raw,
            launcher=launcher,
            gpus=gpus,
            run_dir_raw=run_dir_raw,
            env=dict(env),
            monitor_after_seconds=monitor_after_seconds,
            poll_seconds=poll_seconds,
            criteria_mode=criteria_mode,
            criteria=criteria,
            stop_timeout_s=stop_timeout_s,
            post_stop_wait_seconds=post_stop_wait_seconds,
            gpu_zero=gpu_zero,
            line_no=line_no,
        )

    def requires_run_progress(self) -> bool:
        return any(criterion.kind == "global_step" for criterion in self.criteria)

    def criteria_description(self) -> str:
        if not self.criteria:
            return "none (detach_after_launch)"
        return f" {self.criteria_mode} ".join(c.describe() for c in self.criteria)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue-driven tmux training manager.")
    parser.add_argument("--queue-file", required=True, help="JSONL queue file path.")
    parser.add_argument("--default-poll-seconds", type=float, default=15.0)
    parser.add_argument("--default-run-dir-timeout-s", type=float, default=300.0)
    parser.add_argument("--default-stop-timeout-s", type=float, default=180.0)
    parser.add_argument("--default-post-stop-wait-s", type=float, default=0.0)
    parser.add_argument("--default-gpu-zero-timeout-s", type=float, default=300.0)
    parser.add_argument("--default-gpu-zero-poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def parse_nonnegative_float(value: Any, *, label: str, line_no: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        fail(f"Queue line {line_no}: {label} must be numeric, got {value!r}.")
    if parsed < 0:
        fail(f"Queue line {line_no}: {label} must be >= 0, got {parsed}.")
    return parsed


def parse_positive_float(value: Any, *, label: str, line_no: int) -> float:
    parsed = parse_nonnegative_float(value, label=label, line_no=line_no)
    if parsed <= 0:
        fail(f"Queue line {line_no}: {label} must be > 0, got {parsed}.")
    return parsed


def parse_nonnegative_int(value: Any, *, label: str, line_no: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        fail(f"Queue line {line_no}: {label} must be an integer, got {value!r}.")
    if parsed < 0:
        fail(f"Queue line {line_no}: {label} must be >= 0, got {parsed}.")
    return parsed


def parse_gpu_devices(
    raw: Any,
    *,
    fallback_gpus: str | None,
    line_no: int,
) -> tuple[int, ...] | None:
    candidate = raw
    if candidate is None and fallback_gpus:
        candidate = fallback_gpus

    if candidate is None:
        return None

    if isinstance(candidate, str):
        text = candidate.strip()
        if not text or text.lower() == "all":
            return None
        parts = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(candidate, Sequence) and not isinstance(candidate, (bytes, bytearray)):
        parts = [str(item).strip() for item in candidate if str(item).strip()]
    else:
        fail(f"Queue line {line_no}: gpu_zero.devices must be 'all', a string, or a list.")

    devices: list[int] = []
    for part in parts:
        try:
            devices.append(int(part))
        except ValueError:
            fail(f"Queue line {line_no}: invalid GPU id {part!r} in gpu_zero.devices.")
    return tuple(devices)


def parse_task_criteria(
    raw: Mapping[str, Any],
    *,
    line_no: int,
    detach_after_launch: bool,
) -> tuple[CriterionSpec, ...]:
    criteria_raw = raw.get("criteria")
    criteria: list[CriterionSpec] = []
    if criteria_raw is not None:
        if not isinstance(criteria_raw, Sequence) or isinstance(criteria_raw, (str, bytes, bytearray)):
            fail(f"Queue line {line_no}: criteria must be a list.")
        for item in criteria_raw:
            if not isinstance(item, Mapping):
                fail(f"Queue line {line_no}: each criterion must be an object.")
            criteria.append(CriterionSpec.from_mapping(item, line_no=line_no))

    if not criteria and "stop_after_step" in raw:
        stop_after_step = parse_nonnegative_int(
            raw.get("stop_after_step"),
            label="stop_after_step",
            line_no=line_no,
        )
        criteria.append(
            CriterionSpec(
                kind="global_step",
                value=float(stop_after_step),
                require_eval=bool(raw.get("require_eval", True)),
            )
        )

    if not criteria and detach_after_launch:
        return tuple()

    if not criteria:
        fail(
            f"Queue line {line_no}: define either criteria=[...] or the legacy stop_after_step field."
        )
    return tuple(criteria)


def resolve_repo_relative_path(raw: str, *, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def resolve_config_path(config_raw: str, *, repo_root: Path) -> Path:
    if not config_raw:
        fail("resolve_config_path requires a non-empty config path.")
    if config_raw.startswith("/"):
        return Path(config_raw).resolve()
    if config_raw.endswith(".yaml"):
        return (repo_root / config_raw).resolve()
    return (repo_root / "configs" / f"{config_raw}.yaml").resolve()


def load_queue(queue_path: Path, *, defaults: ManagerDefaults) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for line_no, raw_line in enumerate(queue_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"Invalid JSON on line {line_no} of {queue_path}: {exc}")
        if not isinstance(payload, Mapping):
            fail(f"Queue line {line_no}: task entry must be a JSON object.")
        tasks.append(TaskSpec.from_mapping(payload, defaults=defaults, line_no=line_no))
    if not tasks:
        fail(f"Queue file {queue_path} did not contain any tasks.")
    return tasks


def run_subprocess(
    args: Sequence[str],
    *,
    capture_output: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            check=True,
            text=True,
            capture_output=capture_output,
            cwd=str(cwd) if cwd is not None else None,
        )
    except FileNotFoundError:
        fail(f"Required command not found: {args[0]}")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        detail = stderr or stdout or repr(exc)
        fail(f"Command failed: {' '.join(shlex.quote(a) for a in args)} :: {detail}")


def tmux_session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def tmux_session_command(session: str) -> str:
    result = subprocess.run(
        ["tmux", "display-message", "-p", "-t", session, "#{pane_current_command}"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def tmux_command_is_idle(command: str) -> bool:
    return command in {"", "bash", "zsh", "sh", "fish"}


def build_launch_command(task: TaskSpec) -> str:
    env_map = {str(key): str(value) for key, value in task.env.items()}
    env_map.setdefault("COORDEXP_TRAIN_HEARTBEAT", "1")

    parts = [f"{key}={shlex.quote(value)}" for key, value in sorted(env_map.items())]
    if task.command:
        parts.append(task.command)
    else:
        if task.config_raw:
            parts.append(f"config={shlex.quote(task.config_raw)}")
        if task.gpus:
            parts.append(f"gpus={shlex.quote(task.gpus)}")
        parts.extend(["bash", shlex.quote(task.launcher)])
    return " ".join(parts)


def launch_task_in_tmux(task: TaskSpec, *, repo_root: Path) -> None:
    launch_cmd = build_launch_command(task)
    tmux_cmd = f"cd {shlex.quote(str(repo_root))} && {launch_cmd}"
    if tmux_session_exists(task.session):
        current = tmux_session_command(task.session)
        if not tmux_command_is_idle(current):
            fail(
                f"tmux session {task.session!r} is busy with {current!r}; "
                f"refusing to stack a new launch on top."
            )
        log(f"Launching into existing tmux session {task.session!r}.")
        run_subprocess(["tmux", "send-keys", "-t", task.session, tmux_cmd, "C-m"], capture_output=False)
    else:
        log(f"Creating detached tmux session {task.session!r}.")
        run_subprocess(["tmux", "new-session", "-d", "-s", task.session, tmux_cmd], capture_output=False)


def discover_run_dir_for_config(config_path: Path, *, since_epoch: float | None) -> Path | None:
    cfg = ConfigLoader.load_materialized_training_config(str(config_path))
    base_out = Path(str(cfg.training.output_dir)).resolve()
    if not base_out.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for resolved_config_path in base_out.rglob("resolved_config.json"):
        try:
            stat = resolved_config_path.stat()
        except FileNotFoundError:
            continue
        if since_epoch is not None and stat.st_mtime + 2.0 < since_epoch:
            continue
        try:
            payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        config_path_raw = str(payload.get("config_path", "") or "")
        matches = True
        if config_path_raw:
            try:
                matches = Path(config_path_raw).resolve() == config_path.resolve()
            except Exception:
                matches = False
        if matches:
            candidates.append((stat.st_mtime, resolved_config_path.parent.resolve()))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def wait_for_run_dir(
    task: TaskSpec,
    *,
    config_path: Path,
    started_epoch: float | None,
    defaults: ManagerDefaults,
) -> Path:
    deadline = time.monotonic() + defaults.run_dir_timeout_s
    while time.monotonic() <= deadline:
        run_dir = discover_run_dir_for_config(config_path, since_epoch=started_epoch)
        if run_dir is not None:
            return run_dir
        if tmux_session_exists(task.session):
            current = tmux_session_command(task.session)
            if tmux_command_is_idle(current):
                fail(
                    f"tmux session {task.session!r} became idle before a run directory "
                    f"was discovered for {config_path}."
                )
        time.sleep(task.poll_seconds)
    fail(f"Timed out waiting for run directory discovery from {config_path}.")


def probe_run_progress(run_dir: Path) -> ProgressState:
    heartbeat_path = run_dir / "train_heartbeat.rank0.jsonl"
    logging_path = run_dir / "logging.jsonl"

    if heartbeat_path.exists():
        max_step = 0
        max_eval_step = 0
        for raw in heartbeat_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            step = _record_step(payload)
            if step is None:
                continue
            max_step = max(max_step, step)
            if payload.get("event") == "evaluate":
                max_eval_step = max(max_eval_step, step)
        return ProgressState(source="heartbeat", max_step=max_step, max_eval_step=max_eval_step)

    if logging_path.exists():
        max_step = 0
        max_eval_step = 0
        for raw in logging_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            step = _record_step(payload)
            if step is None:
                continue
            max_step = max(max_step, step)
            if any(str(key).startswith("eval_") for key in payload):
                max_eval_step = max(max_eval_step, step)
        return ProgressState(source="logging", max_step=max_step, max_eval_step=max_eval_step)

    return ProgressState(source="none", max_step=0, max_eval_step=0)


def _record_step(payload: Mapping[str, Any]) -> int | None:
    for key in ("global_step", "step"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    step_progress = payload.get("global_step/max_steps")
    if isinstance(step_progress, str):
        prefix = step_progress.split("/", 1)[0].strip()
        try:
            return int(prefix)
        except ValueError:
            return None
    return None


def maybe_wait_before_monitoring(task: TaskSpec) -> None:
    if task.monitor_after_seconds <= 0:
        return
    log(
        f"Task {task.name!r}: waiting {task.monitor_after_seconds:g}s before monitoring "
        f"(poll interval {task.poll_seconds:g}s)."
    )
    deadline = time.monotonic() + task.monitor_after_seconds
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        sleep_for = min(remaining, max(1.0, min(task.poll_seconds, 60.0)))
        time.sleep(sleep_for)


def wait_for_criteria(
    task: TaskSpec,
    *,
    run_dir: Path | None,
    activation_started_at: float,
) -> None:
    maybe_wait_before_monitoring(task)
    last_report = ""
    idle_warning_emitted = False
    while True:
        elapsed = time.monotonic() - activation_started_at
        progress = probe_run_progress(run_dir) if run_dir is not None else None
        criterion_results = [
            criterion.is_satisfied(progress=progress, elapsed_seconds=elapsed)
            for criterion in task.criteria
        ]
        if task.criteria_mode == "all":
            satisfied = all(criterion_results)
        else:
            satisfied = any(criterion_results)

        progress_desc = "source=none step=n/a eval_step=n/a"
        if progress is not None:
            progress_desc = (
                f"source={progress.source} step={progress.max_step} "
                f"eval_step={progress.max_eval_step}"
            )
        report = (
            f"{progress_desc} elapsed={elapsed:.0f}s "
            f"criteria={task.criteria_description()}"
        )
        if report != last_report:
            log(f"Task {task.name!r}: waiting on {report}")
            last_report = report

        if satisfied:
            log(f"Task {task.name!r}: stop criterion met.")
            return

        if not tmux_session_exists(task.session):
            fail(f"tmux session {task.session!r} disappeared before stop criterion was met.")
        current = tmux_session_command(task.session)
        if tmux_command_is_idle(current):
            if not idle_warning_emitted:
                log(
                    f"Task {task.name!r}: tmux session {task.session!r} currently reports "
                    f"an idle shell; continuing to watch artifacts until the criterion is met."
                )
                idle_warning_emitted = True
        else:
            idle_warning_emitted = False
        time.sleep(task.poll_seconds)


def interrupt_tmux_session(task: TaskSpec) -> None:
    if not tmux_session_exists(task.session):
        fail(f"tmux session {task.session!r} disappeared while stopping.")

    initial_command = tmux_session_command(task.session)
    log(
        f"Sending initial Ctrl-C to tmux session {task.session!r} "
        f"(current command: {initial_command or 'unknown'})."
    )
    run_subprocess(["tmux", "send-keys", "-t", task.session, "C-c"], capture_output=False)

    deadline = time.monotonic() + task.stop_timeout_s
    next_signal_at = time.monotonic() + 10.0
    while time.monotonic() <= deadline:
        if not tmux_session_exists(task.session):
            fail(f"tmux session {task.session!r} disappeared while stopping.")
        current = tmux_session_command(task.session)
        if tmux_command_is_idle(current):
            log(f"tmux session {task.session!r} is idle after stop.")
            return
        if time.monotonic() >= next_signal_at:
            log(
                f"Sending Ctrl-C to tmux session {task.session!r} "
                f"(current command: {current or 'unknown'})."
            )
            run_subprocess(["tmux", "send-keys", "-t", task.session, "C-c"], capture_output=False)
            next_signal_at = time.monotonic() + 10.0
        time.sleep(2.0)
    fail(
        f"Timed out waiting {task.stop_timeout_s:g}s for tmux session {task.session!r} to stop."
    )


def maybe_wait_after_stop(task: TaskSpec) -> None:
    if task.post_stop_wait_seconds <= 0:
        return
    log(
        f"Task {task.name!r}: waiting {task.post_stop_wait_seconds:g}s after tmux stop "
        f"before GPU drain checks."
    )
    time.sleep(task.post_stop_wait_seconds)


def wait_for_gpu_zero(task: TaskSpec) -> None:
    spec = task.gpu_zero
    if not spec.enabled:
        return

    deadline = time.monotonic() + spec.timeout_seconds
    target_desc = "all GPUs" if spec.devices is None else f"GPUs {list(spec.devices)}"
    log(
        f"Task {task.name!r}: waiting for {target_desc} to reach <= "
        f"{spec.memory_threshold_mb} MB used."
    )

    while time.monotonic() <= deadline:
        used_by_gpu = query_gpu_memory_used_mb()
        if spec.devices is None:
            relevant = used_by_gpu
        else:
            missing = [gpu_id for gpu_id in spec.devices if gpu_id not in used_by_gpu]
            if missing:
                fail(
                    f"Task {task.name!r}: nvidia-smi did not report requested GPUs {missing}."
                )
            relevant = {gpu_id: used_by_gpu[gpu_id] for gpu_id in spec.devices}
        if all(memory_mb <= spec.memory_threshold_mb for memory_mb in relevant.values()):
            log(f"Task {task.name!r}: GPU drain check passed.")
            return
        log(
            f"Task {task.name!r}: GPU memory still in use -> "
            + ", ".join(f"{gpu_id}:{memory_mb}MB" for gpu_id, memory_mb in sorted(relevant.items()))
        )
        time.sleep(spec.poll_seconds)

    fail(
        f"Task {task.name!r}: timed out waiting {spec.timeout_seconds:g}s for GPU memory to drain."
    )


def query_gpu_memory_used_mb() -> dict[int, int]:
    result = run_subprocess(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    used_by_gpu: dict[int, int] = {}
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            index_raw, mem_raw = [part.strip() for part in line.split(",", 1)]
            used_by_gpu[int(index_raw)] = int(mem_raw)
        except ValueError:
            fail(f"Unexpected nvidia-smi output row: {line!r}")
    return used_by_gpu


def resolve_task_run_dir(
    task: TaskSpec,
    *,
    repo_root: Path,
    defaults: ManagerDefaults,
    task_started_epoch: float | None,
) -> Path | None:
    if not task.requires_run_progress():
        return None

    if task.run_dir_raw:
        run_dir = resolve_repo_relative_path(task.run_dir_raw, repo_root=repo_root)
        if not run_dir.is_dir():
            fail(f"Task {task.name!r}: run_dir not found: {run_dir}")
        return run_dir

    if not task.config_raw:
        fail(
            f"Task {task.name!r}: step-based criteria require either run_dir or config."
        )

    config_path = resolve_config_path(task.config_raw, repo_root=repo_root)
    if not config_path.is_file():
        fail(f"Task {task.name!r}: config not found: {config_path}")

    if task.launch:
        return wait_for_run_dir(
            task,
            config_path=config_path,
            started_epoch=task_started_epoch,
            defaults=defaults,
        )

    run_dir = discover_run_dir_for_config(config_path, since_epoch=None)
    if run_dir is None:
        fail(
            f"Task {task.name!r}: could not discover an existing run directory from {config_path}."
        )
    return run_dir


def process_task(task: TaskSpec, *, repo_root: Path, defaults: ManagerDefaults) -> None:
    log(f"Task {task.name!r}: session={task.session} criteria={task.criteria_description()}")

    activation_started_at = time.monotonic()
    task_started_epoch: float | None = None

    if task.launch:
        launch_task_in_tmux(task, repo_root=repo_root)
        task_started_epoch = time.time()
    else:
        if not tmux_session_exists(task.session):
            fail(
                f"Task {task.name!r}: attach-only task requires existing tmux session {task.session!r}."
            )

    run_dir = resolve_task_run_dir(
        task,
        repo_root=repo_root,
        defaults=defaults,
        task_started_epoch=task_started_epoch,
    )
    if run_dir is not None:
        log(f"Task {task.name!r}: using run_dir={run_dir}")

    if task.detach_after_launch:
        log(f"Task {task.name!r}: launched and detached; leaving session running.")
        return

    wait_for_criteria(task, run_dir=run_dir, activation_started_at=activation_started_at)
    interrupt_tmux_session(task)
    maybe_wait_after_stop(task)
    wait_for_gpu_zero(task)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    queue_path = resolve_repo_relative_path(args.queue_file, repo_root=repo_root)
    if not queue_path.is_file():
        fail(f"Queue file not found: {queue_path}")

    defaults = ManagerDefaults(
        poll_seconds=args.default_poll_seconds,
        run_dir_timeout_s=args.default_run_dir_timeout_s,
        stop_timeout_s=args.default_stop_timeout_s,
        post_stop_wait_s=args.default_post_stop_wait_s,
        gpu_zero_timeout_s=args.default_gpu_zero_timeout_s,
        gpu_zero_poll_seconds=args.default_gpu_zero_poll_seconds,
    )
    tasks = load_queue(queue_path, defaults=defaults)
    log(f"Loaded {len(tasks)} task(s) from {queue_path}.")

    for index, task in enumerate(tasks, start=1):
        log(f"Processing task {index}/{len(tasks)}: {task.name!r}")
        process_task(task, repo_root=repo_root, defaults=defaults)

    log("Queue completed successfully.")


if __name__ == "__main__":
    main()
