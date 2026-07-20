from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts import run_coco_refinement as launcher


def _launch_kwargs(tmp_path: Path) -> dict[str, Any]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    return {
        "repo_root": repo_root,
        "runtime_root": repo_root / "outputs/coco_refinement/gate-a",
        "host": "127.0.0.1",
        "port": 19172,
        "startup_timeout": 11,
        "shutdown_timeout": 7,
    }


def test_launcher_orders_runtime_lifecycle_and_pins_server_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[object] = []
    repo_root = tmp_path / "repo"
    runtime_root = repo_root / "outputs/coco_refinement/gate-a"
    receipt_store = object()
    app = object()
    config = object()

    class Runtime:
        def start(self) -> None:
            events.append("start")

        def shutdown(self, *, timeout: float) -> None:
            events.append(("shutdown", timeout))

    runtime = Runtime()

    def make_receipts(path: Path) -> object:
        events.append(("receipts", path))
        return receipt_store

    def make_runtime(repo_root: Path, **kwargs: object) -> Runtime:
        receipt_factory = kwargs.pop("inference_receipt_store_factory")
        publisher_factory = kwargs.pop("terminal_publisher_factory")
        events.append(("create", repo_root, kwargs))
        assert callable(receipt_factory)
        assert callable(publisher_factory)
        assert receipt_factory() is receipt_store
        return runtime

    def make_app(selected_runtime: object, **kwargs: object) -> object:
        events.append(("app", selected_runtime, kwargs))
        return app

    def make_config(selected_app: object, **kwargs: object) -> object:
        events.append(("config", selected_app, kwargs))
        return config

    class Server:
        def __init__(self, selected_config: object) -> None:
            events.append(("server", selected_config))

        def run(self) -> None:
            events.append("serve")

    launcher.run_server(
        **_launch_kwargs(tmp_path),
        receipt_store_factory=make_receipts,
        runtime_factory=make_runtime,
        app_factory=make_app,
        config_factory=make_config,
        server_factory=Server,
    )

    assert events[0] == (
        "create",
        repo_root,
        {
            "runtime_root": runtime_root,
            "reload": False,
            "startup_cleanup_timeout": 7,
            "startup_timeout": 11,
            "workers": 1,
        },
    )
    assert events[1] == ("receipts", runtime_root / "roi-receipts.jsonl")
    assert events[2] == "start"
    assert events[3] == (
        "app",
        runtime,
        {"bind_host": "127.0.0.1", "port": 19172},
    )
    assert events[4] == (
        "config",
        app,
        {
            "host": "127.0.0.1",
            "port": 19172,
            "reload": False,
            "workers": 1,
            "access_log": False,
            "timeout_graceful_shutdown": 7,
        },
    )
    assert events[5:] == [("server", config), "serve", ("shutdown", 7)]
    output = capsys.readouterr().out
    assert "Validating the complete train/val workspace" in output
    assert "Workspace validation complete" in output
    assert "Handing off to Uvicorn at http://127.0.0.1:19172" in output
    assert "VS Code can forward the port after the listener is ready" in output


def test_launcher_passes_explicit_browser_origin_only_to_app_factory(
    tmp_path: Path,
) -> None:
    app_calls: list[dict[str, object]] = []

    class Runtime:
        def start(self) -> None:
            pass

        def shutdown(self, *, timeout: float) -> None:
            assert timeout == 7

    def make_app(_runtime: object, **kwargs: object) -> object:
        app_calls.append(kwargs)
        return object()

    kwargs = _launch_kwargs(tmp_path)
    kwargs["browser_origin"] = "http://localhost:53662"
    kwargs["allow_browser_port_remap"] = True
    launcher.run_server(
        **kwargs,
        receipt_store_factory=lambda _path: object(),
        runtime_factory=lambda _repo_root, **_kwargs: Runtime(),
        app_factory=make_app,
        config_factory=lambda _app, **_kwargs: object(),
        server_factory=lambda _config: type("Server", (), {"run": lambda self: None})(),
    )

    assert app_calls == [
        {
            "bind_host": "127.0.0.1",
            "port": 19172,
            "browser_origin": "http://localhost:53662",
            "allow_browser_port_remap": True,
        }
    ]


@pytest.mark.parametrize("failure_at", ["app", "serve"])
def test_app_or_serve_failure_still_shuts_down_runtime(
    tmp_path: Path, failure_at: str
) -> None:
    events: list[object] = []

    class Runtime:
        def start(self) -> None:
            events.append("start")

        def shutdown(self, *, timeout: float) -> None:
            events.append(("shutdown", timeout))

    runtime = Runtime()

    def make_app(_runtime: object, **_kwargs: object) -> object:
        events.append("app")
        if failure_at == "app":
            raise RuntimeError("app failed")
        return object()

    class Server:
        def __init__(self, _config: object) -> None:
            pass

        def run(self) -> None:
            events.append("serve")
            raise RuntimeError("serve failed")

    with pytest.raises(RuntimeError, match=f"{failure_at} failed"):
        launcher.run_server(
            **_launch_kwargs(tmp_path),
            receipt_store_factory=lambda _path: object(),
            runtime_factory=lambda _repo_root, **_kwargs: runtime,
            app_factory=make_app,
            config_factory=lambda _app, **_kwargs: object(),
            server_factory=Server,
        )

    assert events[-1] == ("shutdown", 7)
    assert events.count(("shutdown", 7)) == 1


def test_runtime_assembly_failure_never_builds_bind_capable_objects(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def fail_runtime(_repo_root: Path, **_kwargs: object) -> object:
        events.append("create")
        raise RuntimeError("preflight failed")

    def unexpected(*_args: object, **_kwargs: object) -> object:
        events.append("bind")
        raise AssertionError("bind-capable factory must not be called")

    with pytest.raises(RuntimeError, match="preflight failed"):
        launcher.run_server(
            **_launch_kwargs(tmp_path),
            receipt_store_factory=lambda _path: object(),
            runtime_factory=fail_runtime,
            app_factory=unexpected,
            config_factory=unexpected,
            server_factory=unexpected,
        )

    assert events == ["create"]


def test_start_failure_is_propagated_and_runtime_is_released(tmp_path: Path) -> None:
    events: list[object] = []

    class Runtime:
        def start(self) -> None:
            events.append("start")
            raise RuntimeError("start failed")

        def shutdown(self, *, timeout: float) -> None:
            events.append(("shutdown", timeout))

    with pytest.raises(RuntimeError, match="start failed"):
        launcher.run_server(
            **_launch_kwargs(tmp_path),
            receipt_store_factory=lambda _path: object(),
            runtime_factory=lambda _repo_root, **_kwargs: Runtime(),
            app_factory=lambda *_args, **_kwargs: pytest.fail("app must not build"),
            config_factory=lambda *_args, **_kwargs: pytest.fail(
                "config must not build"
            ),
            server_factory=lambda *_args, **_kwargs: pytest.fail(
                "server must not build"
            ),
        )

    assert events == ["start", ("shutdown", 7)]


@pytest.mark.parametrize("host", ["localhost", "0.0.0.0", "192.168.1.3"])
def test_cli_rejects_non_numeric_or_non_loopback_hosts(host: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        launcher._parser().parse_args(["--host", host])


@pytest.mark.parametrize(
    "origin",
    [
        "https://localhost:53662",
        "http://example.com:53662",
        "http://localhost",
        "http://localhost:0",
        "http://localhost:80",
        "http://localhost:65536",
        "http://user@localhost:53662",
        "http://localhost:53662/",
        "http://localhost:53662/path",
        "http://localhost:53662?query=yes",
        "http://localhost:53662#fragment",
    ],
)
def test_cli_rejects_invalid_browser_origins(origin: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        launcher._parser().parse_args(["--browser-origin", origin])


def test_cli_accepts_one_explicit_localhost_browser_origin() -> None:
    args = launcher._parser().parse_args(
        ["--browser-origin", "http://localhost:53662"]
    )

    assert args.browser_origin == "http://localhost:53662"


def test_cli_accepts_explicit_browser_port_remap_opt_in() -> None:
    args = launcher._parser().parse_args(
        [
            "--browser-origin",
            "http://localhost:53662",
            "--allow-browser-port-remap",
        ]
    )

    assert args.allow_browser_port_remap is True


@pytest.mark.parametrize(
    ("value", "message"),
    [(True, "requires an explicit browser origin"), (1, "must be boolean")],
)
def test_programmatic_browser_port_remap_configuration_fails_before_factories(
    tmp_path: Path, value: object, message: str
) -> None:
    calls: list[str] = []

    def unexpected(*_args: object, **_kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("no factory may run for invalid remap configuration")

    kwargs = _launch_kwargs(tmp_path)
    kwargs["allow_browser_port_remap"] = value
    with pytest.raises(ValueError, match=message):
        launcher.run_server(
            **kwargs,
            receipt_store_factory=unexpected,
            runtime_factory=unexpected,
            app_factory=unexpected,
            config_factory=unexpected,
            server_factory=unexpected,
        )
    assert calls == []


@pytest.mark.parametrize("port", ["0", "8080", "65536", "not-a-port"])
def test_cli_rejects_invalid_ports(port: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        launcher._parser().parse_args(["--port", port])


@pytest.mark.parametrize("timeout", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_startup_timeout(timeout: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        launcher._parser().parse_args(["--startup-timeout", timeout])


def test_cli_defaults_to_production_startup_timeout() -> None:
    args = launcher._parser().parse_args([])

    assert args.startup_timeout == 300


@pytest.mark.parametrize("timeout", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_shutdown_timeout(timeout: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        launcher._parser().parse_args(["--shutdown-timeout", timeout])


def test_default_paths_do_not_depend_on_the_callers_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    args = launcher._parser().parse_args([])

    assert args.repo_root == launcher.REPO_ROOT
    assert launcher._resolve_runtime_root(args.repo_root, args.runtime_root) == (
        launcher.REPO_ROOT / launcher.DEFAULT_RUNTIME_RELATIVE
    ).resolve()


def test_relative_runtime_root_is_anchored_below_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    assert launcher._resolve_runtime_root(
        repo_root, Path("outputs/coco_refinement/gate-a")
    ) == (
        repo_root / "outputs/coco_refinement/gate-a"
    ).resolve()


@pytest.mark.parametrize(
    "value",
    [
        Path("."),
        Path("public_data/coco/rescale_32_1024_bbox_len12000"),
        Path("public_data/coco/rescale_32_1024_bbox/images"),
        Path("outputs/coco_refinement"),
        Path("outputs/label_studio_coco_refinement/state"),
        Path("../outside"),
    ],
)
def test_runtime_root_must_stay_in_standalone_output_namespace(
    tmp_path: Path, value: Path
) -> None:
    repo_root = tmp_path / "repo"
    with pytest.raises(ValueError, match="outputs/coco_refinement"):
        launcher._resolve_runtime_root(repo_root, value)


def test_programmatic_legacy_port_rejection_precedes_every_factory(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def unexpected(*_args: object, **_kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("no factory may run for the reserved legacy port")

    kwargs = _launch_kwargs(tmp_path)
    kwargs["port"] = 8080
    with pytest.raises(ValueError, match="reserved"):
        launcher.run_server(
            **kwargs,
            receipt_store_factory=unexpected,
            runtime_factory=unexpected,
            app_factory=unexpected,
            config_factory=unexpected,
            server_factory=unexpected,
        )

    assert calls == []


@pytest.mark.parametrize(
    "origin",
    [
        "https://localhost:53662",
        "http://example.com:53662",
        "http://localhost",
        "http://localhost:0",
        "http://localhost:80",
        "http://localhost:65536",
        "http://user@localhost:53662",
        "http://localhost:53662/path",
        "http://localhost:53662?query=yes",
        "http://localhost:53662#fragment",
        53662,
    ],
)
def test_programmatic_browser_origin_rejection_precedes_every_factory(
    tmp_path: Path, origin: object
) -> None:
    calls: list[str] = []

    def unexpected(*_args: object, **_kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("no factory may run for an invalid browser origin")

    kwargs = _launch_kwargs(tmp_path)
    kwargs["browser_origin"] = origin
    with pytest.raises(ValueError, match="browser origin"):
        launcher.run_server(
            **kwargs,
            receipt_store_factory=unexpected,
            runtime_factory=unexpected,
            app_factory=unexpected,
            config_factory=unexpected,
            server_factory=unexpected,
        )

    assert calls == []


@pytest.mark.parametrize("field", ["startup_timeout", "shutdown_timeout"])
@pytest.mark.parametrize(
    "value",
    [0, -1, False, True, 1.0, float("nan"), float("inf"), -float("inf")],
)
def test_programmatic_timeout_requires_positive_int_before_every_factory(
    tmp_path: Path, field: str, value: object
) -> None:
    calls: list[str] = []

    def unexpected(*_args: object, **_kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("no factory may run for an invalid timeout")

    kwargs = _launch_kwargs(tmp_path)
    kwargs[field] = value
    with pytest.raises(ValueError, match=rf"{field} must be a positive integer"):
        launcher.run_server(
            **kwargs,
            receipt_store_factory=unexpected,
            runtime_factory=unexpected,
            app_factory=unexpected,
            config_factory=unexpected,
            server_factory=unexpected,
        )

    assert calls == []
