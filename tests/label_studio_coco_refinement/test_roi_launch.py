from __future__ import annotations

import ast
import json
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.config.fingerprint import sha256_json
from src.config.inference import (
    INFER_CONFIG_LOADER_VERSION,
    InferConfig,
    ResolvedInferConfig,
)
from src.inference.parsing import PARSER_ID, PARSER_POLICY
import src.label_studio_coco_refinement.roi_runtime as roi_runtime_module
from src.label_studio_coco_refinement.inference_profiles import (
    EngineProfile,
    EngineProfileStore,
    ProfileContractError,
)
from src.label_studio_coco_refinement.roi_launch import (
    DEFAULT_ENGINE_FACTORY_TARGET,
    InternalRoiProfileBinding,
    ROI_LAUNCH_SCHEMA_VERSION,
    RoiLaunchConfigError,
    RoiLaunchError,
    RoiLaunchManager,
    build_resident_engine_from_profile,
    load_roi_launch_config,
)
from src.label_studio_coco_refinement import roi_launch as roi_launch_module
from src.label_studio_coco_refinement.resident_inference import (
    CancellationMetadata,
    ResidentInferenceCancelled,
)
from src.label_studio_coco_refinement.roi_runtime import (
    RESIDENT_ADAPTER_ID,
    build_resident_profile_binding,
)
from src.label_studio_coco_refinement.roi_transform import ROI_TRANSFORM_ID
from src.qwen.images import QWEN_IMAGE_PROCESSOR_KWARGS


def _traceback_text(error: BaseException) -> str:
    return "".join(traceback.format_exception(error))


class _CurrentTargets:
    def current_target(self, frozen: Any) -> Any:  # pragma: no cover - not invoked.
        raise AssertionError(f"unexpected target lookup: {frozen!r}")


class _FakeEngine:
    def __init__(
        self,
        profile: EngineProfile,
        *,
        infer: Any | None = None,
        closed: list[str] | None = None,
    ) -> None:
        self.profile = build_resident_profile_binding(profile)
        self._infer = infer or (lambda *_args, **_kwargs: profile.name)
        self._closed = closed
        self.name = profile.name

    def infer_one(self, *args: Any, **kwargs: Any) -> Any:
        return self._infer(self.name, *args, **kwargs)

    def close(self) -> None:
        if self._closed is not None:
            self._closed.append(self.name)


class _IdentityPart:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return dict(self.payload)


class _StrictAttestingService:
    def __init__(
        self,
        *,
        profiles: EngineProfileStore,
        receipts: Any,
        **_kwargs: Any,
    ) -> None:
        self.profiles = profiles
        self.receipts = receipts

    def infer(self, *, engine: Any, marker: str, target: Any) -> dict[str, Any]:
        assert target.project_id == "project-a"
        profile = self.profiles.profiles()["accepted"]
        binding = build_resident_profile_binding(profile)
        roi_runtime_module._attest_loaded_engine(
            engine=engine,
            profile=profile,
            binding=binding,
        )
        return {"delegate_result": engine.infer_one(marker)}


class _ActivationRecordingService:
    def __init__(
        self,
        *,
        profiles: EngineProfileStore,
        receipts: Any,
        **_kwargs: Any,
    ) -> None:
        self.profiles = profiles
        self.receipts = receipts
        self.profile_receipts: list[dict[str, Any]] = []

    def infer(self, *, engine: Any, target: Any) -> dict[str, Any]:
        active = self.profiles.active(target.project_id)
        assert active.fingerprint == target.profile_fingerprint
        assert engine.profile == build_resident_profile_binding(active)
        receipt = active.to_receipt_dict()
        self.profile_receipts.append(receipt)
        return {
            "project_id": target.project_id,
            "profile_name": receipt["profile_name"],
            "delegate_result": engine.infer_one(),
        }


def test_operator_json_is_strict_and_paths_are_config_relative(tmp_path: Path) -> None:
    config_path = tmp_path / "operator" / "roi.json"
    payload = _launch_payload(
        profile_store_path="../state/profiles.json",
        receipt_store_path="../state/receipts.jsonl",
        insertion_ack_timeout_seconds=None,
    )
    _write_json(config_path, payload)

    config = load_roi_launch_config(config_path)

    assert config.config_path == config_path.resolve()
    assert config.profile_store_path == (tmp_path / "state/profiles.json").resolve()
    assert config.receipt_store_path == (tmp_path / "state/receipts.jsonl").resolve()
    assert config.insertion_ack_timeout_seconds is None
    assert config.engine_factory.target == DEFAULT_ENGINE_FACTORY_TARGET
    assert config.engine_factory.config_copy() == {}
    with pytest.raises(TypeError):
        config.profile_selectors["new"] = "profile"  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update({"unknown": True}), "unknown"),
        (lambda payload: payload["bind"].update({"scheme": "http"}), "unknown"),
        (
            lambda payload: payload["engine_factory"].update({"args": []}),
            "unknown",
        ),
        (
            lambda payload: payload.update({"insertion_ack_timeout_seconds": 0}),
            "positive",
        ),
        (
            lambda payload: payload.update({"insertion_ack_timeout_seconds": True}),
            "positive",
        ),
    ],
)
def test_operator_json_rejects_unknown_keys_and_implicit_ack_policy(
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    payload = _launch_payload()
    mutate(payload)
    config_path = tmp_path / "roi.json"
    _write_json(config_path, payload)

    with pytest.raises(RoiLaunchConfigError, match=match):
        load_roi_launch_config(config_path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload, key: payload.update({key: "private-value"}),
        lambda payload, key: payload["bind"].update({key: "private-value"}),
        lambda payload, key: payload["engine_factory"].update({key: "private-value"}),
    ],
)
def test_unknown_schema_keys_are_counted_but_never_echoed(
    tmp_path: Path,
    mutate: Any,
) -> None:
    credential_key = "token=super-secret"
    payload = _launch_payload()
    mutate(payload, credential_key)
    config_path = tmp_path / "roi.json"
    _write_json(config_path, payload)

    with pytest.raises(RoiLaunchConfigError) as error:
        load_roi_launch_config(config_path)

    message = str(error.value)
    assert "unknown_count=1" in message
    assert credential_key not in message
    assert "super-secret" not in message
    assert "private-value" not in message


@pytest.mark.parametrize(
    ("encoded", "private_key"),
    [
        (
            '{"token=super-secret":1,"token=super-secret":2}',
            "token=super-secret",
        ),
        (
            '{"bind":{"token=super-secret":1,"token=super-secret":2}}',
            "token=super-secret",
        ),
        (
            '{"engine_factory":{"token=super-secret":1,"token=super-secret":2}}',
            "token=super-secret",
        ),
        (
            '{"schema_version":"private-value","schema_version":"private-value-2"}',
            "schema_version",
        ),
    ],
)
def test_duplicate_json_keys_are_rejected_without_echoing_observed_key_or_value(
    tmp_path: Path,
    encoded: str,
    private_key: str,
) -> None:
    config_path = tmp_path / "roi.json"
    config_path.write_text(encoded, encoding="utf-8")

    with pytest.raises(RoiLaunchConfigError) as error:
        load_roi_launch_config(config_path)

    assert str(error.value) == "duplicate JSON key is forbidden"
    assert private_key not in str(error.value)
    assert "super-secret" not in str(error.value)
    assert "private-value" not in str(error.value)


def test_nonfinite_value_under_credential_shaped_key_never_echoes_parse_path(
    tmp_path: Path,
) -> None:
    credential_key = "token=super-secret"
    config_path = tmp_path / "roi.json"
    payload = _launch_payload(factory_config={credential_key: "__NUMBER__"})
    encoded = json.dumps(payload).replace('"__NUMBER__"', "1e400")
    config_path.write_text(encoded, encoding="utf-8")

    with pytest.raises(RoiLaunchConfigError) as error:
        load_roi_launch_config(config_path)

    assert str(error.value) == "launch config contains a non-finite JSON number"
    assert credential_key not in str(error.value)
    assert "super-secret" not in str(error.value)


def test_schema_and_config_path_errors_never_echo_attacker_controlled_values(
    tmp_path: Path,
) -> None:
    private_schema = "token=super-secret"
    config_path = tmp_path / "roi.json"
    payload = _launch_payload()
    payload["schema_version"] = private_schema
    _write_json(config_path, payload)

    with pytest.raises(RoiLaunchConfigError) as schema_error:
        load_roi_launch_config(config_path)

    assert str(schema_error.value) == "unsupported ROI launch schema"
    assert private_schema not in str(schema_error.value)

    private_path = tmp_path / "token=super-secret" / "missing.json"
    with pytest.raises(RoiLaunchConfigError) as path_error:
        load_roi_launch_config(private_path)

    assert str(path_error.value) == "operator config path is unavailable"
    assert "super-secret" not in _traceback_text(path_error.value)


def test_operator_json_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"coordexp-roi-launch-v1",'
        '"schema_version":"coordexp-roi-launch-v1"}',
        encoding="utf-8",
    )
    with pytest.raises(RoiLaunchConfigError, match="duplicate JSON key"):
        load_roi_launch_config(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    payload = json.dumps(_launch_payload()).replace(
        '"insertion_ack_timeout_seconds": 30.0',
        '"insertion_ack_timeout_seconds": NaN',
    )
    nonfinite.write_text(payload, encoding="utf-8")
    with pytest.raises(RoiLaunchConfigError, match="non-finite"):
        load_roi_launch_config(nonfinite)


@pytest.mark.parametrize("number", ["1e400", "-1e400", "NaN", "Infinity", "-Infinity"])
def test_operator_json_rejects_nonfinite_numbers_at_every_nesting_depth(
    tmp_path: Path,
    number: str,
) -> None:
    config_path = tmp_path / "roi.json"
    payload = _launch_payload(
        factory_config={"nested": [{"values": ["__NUMBER__"]}]},
    )
    encoded = json.dumps(payload).replace('"__NUMBER__"', number)
    config_path.write_text(encoded, encoding="utf-8")

    with pytest.raises(RoiLaunchConfigError, match="non-finite"):
        load_roi_launch_config(config_path)


@pytest.mark.parametrize(
    "path_key",
    ["checkpoint_path", "checkpoint_paths", "checkpointPath", "checkpointPaths"],
)
def test_operator_json_rejects_path_lists_instead_of_leaving_them_unresolved(
    tmp_path: Path,
    path_key: str,
) -> None:
    config_path = tmp_path / "roi.json"
    _write_json(
        config_path,
        _launch_payload(factory_config={path_key: ["a", "b"]}),
    )

    with pytest.raises(RoiLaunchConfigError, match="exactly empty"):
        load_roi_launch_config(config_path)


@pytest.mark.parametrize(
    "change",
    [
        {"bind": {"host": "0.0.0.0", "port": 8123}},
        {"factory_config": {"endpoint": "http://192.168.1.3:8123/infer"}},
        {"factory_config": {"worker_bind": {"host": "::", "port": 8123}}},
    ],
)
def test_every_operator_endpoint_and_bind_must_be_loopback(
    tmp_path: Path,
    change: dict[str, Any],
) -> None:
    payload = _launch_payload(
        bind=change.get("bind"),
        factory_config=change.get("factory_config"),
    )
    config_path = tmp_path / "roi.json"
    _write_json(config_path, payload)

    with pytest.raises(RoiLaunchConfigError, match="loopback|exactly empty"):
        load_roi_launch_config(config_path)


@pytest.mark.parametrize(
    "factory_config",
    [
        {"host": "127.0.0.1.evil.example"},
        {"address": "2130706433"},
        {"host": "3232235777"},
        {"host": "0300.0250.0001.0001"},
        {"host": "0xC0A80101"},
        {"base_url": "http://127.0.0.1.evil.example/infer"},
        {"callbackUrl": "https://example.com/callback"},
        {"baseURLValue": "2130706433"},
        {"worker_bind": {"host": "localhost.evil", "port": 8123}},
        {"binding": "0.0.0.0:8123"},
        {"listen": "[::ffff:127.0.0.1]:8123"},
        {"nested": [{"tag": "http://10.0.0.8:8123/infer"}]},
        {"tag": "//example.com:8123/infer"},
        {"tag": "192.168.1.8:8123"},
        {"tag": "3232235777"},
        {"tag": "0300.0250.0001.0001"},
        {"tag": "0xC0A80101"},
        {"tag": "[2001:db8::1]:8123"},
        {"tag": "example\u3002com:8123"},
        {"tag": "ｈｔｔｐ://192.168.1.8:8123/infer"},
        {"tag": "http%3A%2F%2F192.168.1.8%3A8123%2Finfer"},
        {"tag": "example.com:443"},
        {"socket": {"transport": "unix"}},
        {"endpoint": "http://0177.0.0.1:8123/infer"},
        {"endpoint": "http://127.0.0.1%2eevil.example:8123/infer"},
        {"endpoint": "http://user:secret@localhost:8123/infer"},
        {"listen": "user:secret@localhost:8123"},
        {"3232235777": "hidden-in-key"},
        {"http://192.168.1.8:8123": "hidden-in-key"},
        {"h%6fst": "hidden-in-key"},
        {"ｈｏｓｔ": "hidden-in-key"},
    ],
)
def test_factory_config_rejects_every_remote_or_ambiguous_nonempty_payload(
    tmp_path: Path,
    factory_config: dict[str, Any],
) -> None:
    config_path = tmp_path / "roi.json"
    _write_json(config_path, _launch_payload(factory_config=factory_config))

    with pytest.raises(RoiLaunchConfigError, match="exactly empty"):
        load_roi_launch_config(config_path)


@pytest.mark.parametrize(
    "factory_config",
    [
        {"host": "localhost"},
        {"address": "127.0.0.1"},
        {"base_url": "https://[::1]:8123/infer"},
        {"worker_bind": {"host": "localhost", "port": 8124}},
        {"host": "127.0.0.1", "port": 8124},
        {"listen": "[::1]:8125"},
        {"nested": [{"tag": "http://127.0.0.1:8126/infer"}]},
        {"tag": "//localhost:8126"},
        {"tag": "127.0.0.1:8127"},
    ],
)
def test_factory_config_rejects_even_explicit_loopback_nonempty_payload(
    tmp_path: Path,
    factory_config: dict[str, Any],
) -> None:
    config_path = tmp_path / "roi.json"
    _write_json(config_path, _launch_payload(factory_config=factory_config))

    with pytest.raises(RoiLaunchConfigError, match="exactly empty"):
        load_roi_launch_config(config_path)


@pytest.mark.parametrize(
    "factory_config",
    [
        {"tag": "neutral"},
        {"count": 1},
        {"enabled": False},
        {"nested": {}},
        {"items": []},
    ],
)
def test_factory_config_rejects_neutral_nonempty_payload(
    tmp_path: Path,
    factory_config: dict[str, Any],
) -> None:
    config_path = tmp_path / "roi.json"
    _write_json(config_path, _launch_payload(factory_config=factory_config))

    with pytest.raises(RoiLaunchConfigError, match="exactly empty"):
        load_roi_launch_config(config_path)


def test_factory_config_rejection_never_echoes_paths_or_credentials(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "roi.json"
    private_value = "/private/model?token=super-secret"
    _write_json(
        config_path,
        _launch_payload(factory_config={"credential": private_value}),
    )

    with pytest.raises(RoiLaunchConfigError) as error:
        load_roi_launch_config(config_path)

    assert str(error.value) == (
        "engine_factory.config must be exactly empty for the resident factory"
    )
    assert private_value not in str(error.value)
    assert "super-secret" not in str(error.value)


@pytest.mark.parametrize(
    "target",
    [
        "fake.engines:create",
        "src.label_studio_coco_refinement.roi_launch:_import_engine_factory",
        "builtins:dict",
    ],
)
def test_operator_json_rejects_every_alternate_factory_without_echoing_target(
    tmp_path: Path,
    target: str,
) -> None:
    payload = _launch_payload()
    payload["engine_factory"]["target"] = target
    config_path = tmp_path / "roi.json"
    _write_json(config_path, payload)

    with pytest.raises(RoiLaunchConfigError) as error:
        load_roi_launch_config(config_path)

    assert str(error.value) == (
        "engine_factory.target must be the checked-in resident factory"
    )
    assert target not in str(error.value)


def test_profile_selectors_must_map_one_to_one_onto_saved_profiles(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "roi.json"
    _write_json(
        config_path,
        _launch_payload(
            profile_selectors={
                "first-selector": "accepted",
                "second-selector": "accepted",
            }
        ),
    )

    with pytest.raises(RoiLaunchConfigError, match="one-to-one"):
        load_roi_launch_config(config_path)


def test_saved_profile_endpoint_must_also_be_loopback(tmp_path: Path) -> None:
    profile = _profile(
        tmp_path / "profile",
        name="remote",
        endpoint="http://10.0.0.7:8123/infer",
    )
    config_path = _configured_store(tmp_path, [profile])

    with pytest.raises(RoiLaunchConfigError, match="loopback"):
        RoiLaunchManager(
            config_path,
            current_targets=_CurrentTargets(),
            engine_factory_loader=lambda _target: lambda **_kwargs: None,
        )


def test_malformed_saved_ipv6_url_does_not_escape_in_full_traceback(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    store_path = (tmp_path / "state/profiles.json").resolve()
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    private_url = "http://[token=super-secret"
    payload["profiles"]["accepted"]["endpoint"] = private_url
    store_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RoiLaunchConfigError) as error:
        RoiLaunchManager(
            config_path,
            current_targets=_CurrentTargets(),
            engine_factory_loader=lambda _target: lambda **_kwargs: None,
        )

    assert str(error.value) == "profile store could not be loaded"
    assert error.value.__cause__ is None
    assert private_url not in str(error.value)
    assert private_url not in _traceback_text(error.value)
    assert "super-secret" not in str(error.value)
    assert "super-secret" not in _traceback_text(error.value)


def test_operator_relative_path_resolution_error_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "operator/roi.json"
    private_marker = "token=super-secret"
    _write_json(
        config_path,
        _launch_payload(profile_store_path=f"../{private_marker}/profiles.json"),
    )
    original_resolve = Path.resolve

    def resolve(path: Path, *, strict: bool = False) -> Path:
        if private_marker in str(path):
            raise OSError(private_marker)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)

    with pytest.raises(RoiLaunchConfigError) as error:
        load_roi_launch_config(config_path)

    assert str(error.value) == "profile_store_path path is unavailable"
    assert error.value.__cause__ is None
    assert private_marker not in str(error.value)
    assert private_marker not in _traceback_text(error.value)


def test_unknown_profile_selector_values_report_count_only(tmp_path: Path) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    private_profile_name = "token-super-secret"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["profile_selectors"] = {"private-selector": private_profile_name}
    _write_json(config_path, payload)

    with pytest.raises(RoiLaunchConfigError) as error:
        RoiLaunchManager(
            config_path,
            current_targets=_CurrentTargets(),
            engine_factory_loader=lambda _target: lambda **_kwargs: None,
        )

    assert str(error.value) == (
        "profile_selectors reference unknown saved profiles: count=1"
    )
    assert error.value.__cause__ is None
    assert private_profile_name not in str(error.value)
    assert private_profile_name not in _traceback_text(error.value)
    assert "super-secret" not in str(error.value)
    assert "super-secret" not in _traceback_text(error.value)


def test_engine_factory_loader_failure_is_sanitized_in_full_traceback(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    private_failure = "token=super-secret-loader"

    def loader(_target: str) -> Any:
        raise RuntimeError(private_failure)

    with pytest.raises(RoiLaunchConfigError) as error:
        RoiLaunchManager(
            config_path,
            current_targets=_CurrentTargets(),
            engine_factory_loader=loader,
        )

    assert str(error.value) == "engine_factory.target could not be resolved"
    assert error.value.__cause__ is None
    assert private_failure not in str(error.value)
    assert private_failure not in _traceback_text(error.value)


def test_safe_profile_projection_has_no_paths_endpoints_or_factory_config(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: lambda **_kwargs: None,
    )

    projection = manager.profile_options()

    assert projection == (
        {
            "selector": "accepted-selector",
            "display_label": "accepted-selector",
            "default_canvas": {"width": 1024, "height": 1024},
            "processor_factor": 32,
            "bounds": {
                "min_axis_pixels": 32,
                "max_axis_pixels": 2048,
                "max_total_pixels": 2_097_152,
            },
            "generation_deadline_seconds": 20.0,
        },
    )
    encoded = json.dumps(projection)
    assert "endpoint" not in encoded
    assert "artifacts" not in encoded
    assert str(tmp_path.resolve()) not in encoded
    manager.close()


def test_config_rejects_credential_or_path_shaped_profile_name_without_echoing_it(
    tmp_path: Path,
) -> None:
    private_name = "https://user:secret@example.com/model?token=do-not-leak"
    config_path = tmp_path / "operator/roi.json"
    _write_json(
        config_path,
        _launch_payload(
            profile_selectors={"safe-selector": private_name},
        ),
    )

    with pytest.raises(RoiLaunchConfigError) as error:
        load_roi_launch_config(config_path)

    assert "safe 1-64 character token" in str(error.value)
    assert "secret" not in str(error.value)
    assert "example.com" not in str(error.value)


def test_selected_store_profile_name_mismatch_fails_without_echoing_private_name(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    store_path = (tmp_path / "state/profiles.json").resolve()
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    private_name = "https://user:secret@example.com/model?token=do-not-leak"
    persisted = payload["profiles"]["accepted"]
    persisted["name"] = private_name
    persisted.pop("profile_fingerprint")
    persisted.pop("identity_fingerprints")
    store_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RoiLaunchConfigError) as error:
        RoiLaunchManager(
            config_path,
            current_targets=_CurrentTargets(),
            engine_factory_loader=lambda _target: lambda **_kwargs: None,
        )

    assert str(error.value) == (
        "profile store keys must exactly match persisted profile names"
    )
    assert "secret" not in str(error.value)
    assert "example.com" not in str(error.value)


def test_profile_store_rejects_unsafe_raw_name_even_when_key_matches_without_echo(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    store_path = (tmp_path / "state/profiles.json").resolve()
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    private_name = "https://user:secret@example.com/model?token=do-not-leak"
    persisted = payload["profiles"].pop("accepted")
    persisted["name"] = private_name
    persisted.pop("profile_fingerprint")
    persisted.pop("identity_fingerprints")
    payload["profiles"][private_name] = persisted
    store_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RoiLaunchConfigError) as error:
        RoiLaunchManager(
            config_path,
            current_targets=_CurrentTargets(),
            engine_factory_loader=lambda _target: lambda **_kwargs: None,
        )

    assert str(error.value) == (
        "persisted profile names must be safe 1-64 character tokens"
    )
    assert "secret" not in str(error.value)
    assert "example.com" not in str(error.value)


def test_selected_safe_profile_name_is_credential_free_in_projection_and_receipt(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: lambda **_kwargs: None,
    )

    encoded = json.dumps(
        {
            "options": manager.profile_options(),
            "receipt": profile.to_receipt_dict(),
        }
    )

    assert "accepted" in encoded
    assert "@" not in encoded
    assert "?" not in encoded
    assert "secret" not in encoded
    manager.close()


@pytest.mark.parametrize("ack_timeout", [37.5, None])
def test_engine_loads_once_and_service_shares_receipt_authority_and_ack_policy(
    tmp_path: Path,
    ack_timeout: float | None,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(
        tmp_path,
        [profile],
        insertion_ack_timeout_seconds=ack_timeout,
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        calls.append((profile.name, config))
        config["mutated"] = True
        return _FakeEngine(profile)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda target: factory
        if target == DEFAULT_ENGINE_FACTORY_TARGET
        else None,
    )

    first = manager.engine_for("accepted-selector")
    second = manager.engine_for("accepted-selector")

    assert first is second
    assert calls == [("accepted", {"mutated": True})]
    assert manager.config.engine_factory.config_copy() == {}
    assert manager.receipt_store is manager.inference_receipt_resolver
    assert manager.receipt_store is manager.inference_service.receipts
    assert manager.inference_service.insertion_ack_timeout_seconds == ack_timeout
    manager.close()


def test_manager_infer_exposes_exact_read_only_service_attestation_surface(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    delegate_calls: list[tuple[str, str]] = []
    runtime_identity = json.loads(profile.runtime_identity_json)

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        engine = _FakeEngine(
            profile,
            infer=lambda name, marker: delegate_calls.append((name, marker))
            or "strict-result",
        )
        engine.resolved = _resolved_from_profile(profile)
        engine.transformers_version = profile.transformers_version
        engine.processor_kwargs = dict(QWEN_IMAGE_PROCESSOR_KWARGS)
        engine.runtime = SimpleNamespace(
            model_identity=runtime_identity["model"],
            qwen=SimpleNamespace(
                processor_identity=_IdentityPart(runtime_identity["processor"]),
                token_identity=_IdentityPart(runtime_identity["tokenizer"]),
            ),
            secret_path="/operator/private/model",
        )
        engine.secret_token = "must-not-escape"
        engine.alternate_generate = lambda: "unguarded"
        return engine

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
        service_factory=_StrictAttestingService,
    )

    target = SimpleNamespace(
        project_id="project-a",
        profile_fingerprint=profile.fingerprint,
    )
    response = manager.infer(
        selector="accepted-selector",
        marker="request-1",
        target=target,
    )
    managed = manager.engine_for("accepted-selector")

    assert response == {"delegate_result": "strict-result"}
    assert delegate_calls == [("accepted", "request-1")]
    assert isinstance(managed.resolved, ResolvedInferConfig)
    assert managed.transformers_version == profile.transformers_version
    assert managed.processor_kwargs == dict(QWEN_IMAGE_PROCESSOR_KWARGS)
    assert managed.runtime.model_identity == runtime_identity["model"]
    resolved_copy = managed.resolved
    assert resolved_copy is not None
    resolved_copy.config_dict["debug"]["smoke"] = False
    kwargs_copy = managed.processor_kwargs
    assert kwargs_copy is not None
    kwargs_copy["do_resize"] = True
    assert manager.infer(
        selector="accepted-selector",
        marker="request-2",
        target=target,
    ) == {"delegate_result": "strict-result"}
    assert delegate_calls == [
        ("accepted", "request-1"),
        ("accepted", "request-2"),
    ]
    with pytest.raises(AttributeError):
        _ = managed.secret_token
    with pytest.raises(AttributeError):
        _ = managed.alternate_generate
    with pytest.raises(AttributeError):
        _ = managed.secret_path
    manager.close()


def test_project_scoped_selector_switch_changes_later_receipts_without_cross_project_leak(
    tmp_path: Path,
) -> None:
    alpha = _profile(tmp_path / "alpha-profile", name="alpha")
    beta = _profile(tmp_path / "beta-profile", name="beta")
    config_path = _configured_store(tmp_path, [alpha, beta])
    factory_calls: list[str] = []

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        factory_calls.append(profile.name)
        return _FakeEngine(profile)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
        service_factory=_ActivationRecordingService,
    )

    alpha_a = SimpleNamespace(
        project_id="project-a",
        profile_fingerprint=alpha.fingerprint,
    )
    alpha_b = SimpleNamespace(
        project_id="project-b",
        profile_fingerprint=alpha.fingerprint,
    )
    beta_a = SimpleNamespace(
        project_id="project-a",
        profile_fingerprint=beta.fingerprint,
    )
    first_a = manager.infer(selector="alpha-selector", target=alpha_a)
    first_b = manager.infer(selector="alpha-selector", target=alpha_b)
    switched_a = manager.infer(selector="beta-selector", target=beta_a)

    assert first_a["profile_name"] == "alpha"
    assert first_b["profile_name"] == "alpha"
    assert switched_a["profile_name"] == "beta"
    assert manager.profile_store.active("project-a").fingerprint == beta.fingerprint
    assert manager.profile_store.active("project-b").fingerprint == alpha.fingerprint
    assert [
        receipt["profile_name"]
        for receipt in manager.inference_service.profile_receipts
    ] == ["alpha", "alpha", "beta"]
    assert factory_calls == ["alpha", "beta"]
    manager.close()


def test_internal_profile_contract_is_credential_free_and_current_is_verified(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: _FakeEngine(profile)
        ),
    )

    selected = manager.resolve_selected_profile(
        project_id="project-a",
        selector="accepted-selector",
    )
    current = manager.current_profile(project_id="project-a")

    assert (
        selected
        == current
        == InternalRoiProfileBinding(
            fingerprint=profile.fingerprint,
            processor_factor=32,
            default_width=1024,
            default_height=1024,
            min_axis_pixels=32,
            max_axis_pixels=2048,
            max_total_pixels=2_097_152,
            deadline_seconds=20.0,
        )
    )
    encoded = json.dumps(selected.__dict__)
    assert "endpoint" not in encoded
    assert "artifact" not in encoded
    assert "path" not in encoded
    assert str(tmp_path.resolve()) not in encoded
    manager.close()


def test_failed_profile_switch_preserves_prior_project_activation(
    tmp_path: Path,
) -> None:
    alpha = _profile(tmp_path / "alpha-profile", name="alpha")
    beta_root = tmp_path / "beta-profile"
    beta = _profile(beta_root, name="beta")
    config_path = _configured_store(tmp_path, [alpha, beta])
    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: _FakeEngine(profile)
        ),
    )
    manager.activate_profile(project_id="project-a", selector="alpha-selector")
    (beta_root / "base/weights.bin").write_bytes(b"drift-before-activation")

    with pytest.raises(RoiLaunchError) as error:
        manager.activate_profile(project_id="project-a", selector="beta-selector")

    assert error.value.__cause__ is None
    assert manager.profile_store.active("project-a").fingerprint == alpha.fingerprint
    manager.close()


def test_partial_activation_write_then_raise_restores_prior_profile(
    tmp_path: Path,
) -> None:
    alpha = _profile(tmp_path / "alpha-profile", name="alpha")
    beta = _profile(tmp_path / "beta-profile", name="beta")
    config_path = _configured_store(tmp_path, [alpha, beta])
    private_failure = "token=super-secret-partial-activation"

    class PartialWriteStore:
        def __init__(self, path: Path) -> None:
            self.delegate = EngineProfileStore(path)

        def profiles(self) -> dict[str, EngineProfile]:
            return self.delegate.profiles()

        def active(self, project_id: str, *, verify: bool = True) -> EngineProfile:
            return self.delegate.active(project_id, verify=verify)

        def activate(self, project_id: str, profile_name: str) -> EngineProfile:
            activated = self.delegate.activate(project_id, profile_name)
            if profile_name == "beta":
                raise RuntimeError(private_failure)
            return activated

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        profile_store_factory=PartialWriteStore,
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: _FakeEngine(profile)
        ),
    )
    manager.activate_profile(project_id="project-a", selector="alpha-selector")

    with pytest.raises(RoiLaunchError, match="prior activation restored") as error:
        manager.activate_profile(project_id="project-a", selector="beta-selector")

    assert error.value.__cause__ is None
    assert private_failure not in _traceback_text(error.value)
    assert manager.profile_store.active("project-a").fingerprint == alpha.fingerprint
    manager.activate_profile(project_id="project-a", selector="alpha-selector")
    manager.close()


def test_mismatched_activation_return_restores_prior_profile(tmp_path: Path) -> None:
    alpha = _profile(tmp_path / "alpha-profile", name="alpha")
    beta = _profile(tmp_path / "beta-profile", name="beta")
    config_path = _configured_store(tmp_path, [alpha, beta])

    class MismatchedReturnStore:
        def __init__(self, path: Path) -> None:
            self.delegate = EngineProfileStore(path)

        def profiles(self) -> dict[str, EngineProfile]:
            return self.delegate.profiles()

        def active(self, project_id: str, *, verify: bool = True) -> EngineProfile:
            return self.delegate.active(project_id, verify=verify)

        def activate(self, project_id: str, profile_name: str) -> EngineProfile:
            activated = self.delegate.activate(project_id, profile_name)
            if profile_name == "beta":
                return self.delegate.profiles()["alpha"]
            return activated

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        profile_store_factory=MismatchedReturnStore,
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: _FakeEngine(profile)
        ),
    )
    manager.activate_profile(project_id="project-a", selector="alpha-selector")

    with pytest.raises(RoiLaunchError, match="prior activation restored") as error:
        manager.activate_profile(project_id="project-a", selector="beta-selector")

    assert error.value.__cause__ is None
    assert manager.profile_store.active("project-a").fingerprint == alpha.fingerprint
    manager.activate_profile(project_id="project-a", selector="alpha-selector")
    manager.close()


def test_rollback_failure_terminally_quarantines_only_affected_project(
    tmp_path: Path,
) -> None:
    alpha = _profile(tmp_path / "alpha-profile", name="alpha")
    beta = _profile(tmp_path / "beta-profile", name="beta")
    config_path = _configured_store(tmp_path, [alpha, beta])
    activation_secret = "token=super-secret-activation"
    rollback_secret = "token=super-secret-rollback"

    class RollbackFailureStore:
        def __init__(self, path: Path) -> None:
            self.delegate = EngineProfileStore(path)
            self.alpha_calls = 0

        def profiles(self) -> dict[str, EngineProfile]:
            return self.delegate.profiles()

        def active(self, project_id: str, *, verify: bool = True) -> EngineProfile:
            return self.delegate.active(project_id, verify=verify)

        def activate(self, project_id: str, profile_name: str) -> EngineProfile:
            activated = self.delegate.activate(project_id, profile_name)
            if project_id == "project-a" and profile_name == "alpha":
                self.alpha_calls += 1
                if self.alpha_calls > 1:
                    raise RuntimeError(rollback_secret)
            if project_id == "project-a" and profile_name == "beta":
                raise RuntimeError(activation_secret)
            return activated

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        profile_store_factory=RollbackFailureStore,
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: _FakeEngine(profile)
        ),
        service_factory=_ActivationRecordingService,
    )
    manager.activate_profile(project_id="project-a", selector="alpha-selector")

    with pytest.raises(RoiLaunchError, match="terminally unavailable") as error:
        manager.activate_profile(project_id="project-a", selector="beta-selector")

    rendered = _traceback_text(error.value)
    assert error.value.__cause__ is None
    assert activation_secret not in rendered
    assert rollback_secret not in rendered
    target_a = SimpleNamespace(
        project_id="project-a",
        profile_fingerprint=alpha.fingerprint,
    )
    with pytest.raises(RoiLaunchError, match="terminally unavailable"):
        manager.infer(selector="alpha-selector", target=target_a)
    assert manager.inference_service.profile_receipts == []

    target_b = SimpleNamespace(
        project_id="project-b",
        profile_fingerprint=alpha.fingerprint,
    )
    assert (
        manager.infer(selector="alpha-selector", target=target_b)["profile_name"]
        == "alpha"
    )
    manager.close()


def test_failed_first_activation_without_prior_is_terminally_quarantined(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    private_failure = "token=super-secret-first-activation"

    class FirstActivationFailureStore:
        def __init__(self, path: Path) -> None:
            self.delegate = EngineProfileStore(path)

        def profiles(self) -> dict[str, EngineProfile]:
            return self.delegate.profiles()

        def active(self, project_id: str, *, verify: bool = True) -> EngineProfile:
            return self.delegate.active(project_id, verify=verify)

        def activate(self, project_id: str, profile_name: str) -> EngineProfile:
            self.delegate.activate(project_id, profile_name)
            raise RuntimeError(private_failure)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        profile_store_factory=FirstActivationFailureStore,
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: _FakeEngine(profile)
        ),
    )

    with pytest.raises(RoiLaunchError, match="terminally unavailable") as error:
        manager.activate_profile(
            project_id="project-a",
            selector="accepted-selector",
        )

    assert error.value.__cause__ is None
    assert private_failure not in _traceback_text(error.value)
    with pytest.raises(RoiLaunchError, match="terminally unavailable"):
        manager.activate_profile(
            project_id="project-a",
            selector="accepted-selector",
        )
    manager.close()


def test_target_profile_mismatch_does_not_change_existing_project_activation(
    tmp_path: Path,
) -> None:
    alpha = _profile(tmp_path / "alpha-profile", name="alpha")
    beta = _profile(tmp_path / "beta-profile", name="beta")
    config_path = _configured_store(tmp_path, [alpha, beta])
    factory_calls: list[str] = []

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        factory_calls.append(profile.name)
        return _FakeEngine(profile)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
        service_factory=_ActivationRecordingService,
    )
    manager.activate_profile(project_id="project-a", selector="alpha-selector")
    mismatched_target = SimpleNamespace(
        project_id="project-a",
        profile_fingerprint=alpha.fingerprint,
    )

    with pytest.raises(RoiLaunchError, match="does not match"):
        manager.infer(selector="beta-selector", target=mismatched_target)

    assert manager.profile_store.active("project-a").fingerprint == alpha.fingerprint
    assert manager.inference_service.profile_receipts == []
    assert factory_calls == ["alpha"]
    manager.close()


def test_concurrent_switches_are_serialized_without_partial_activation(
    tmp_path: Path,
) -> None:
    alpha = _profile(tmp_path / "alpha-profile", name="alpha")
    beta = _profile(tmp_path / "beta-profile", name="beta")
    config_path = _configured_store(tmp_path, [alpha, beta])
    alpha_factory_entered = threading.Event()
    release_alpha_factory = threading.Event()
    completed: list[str] = []

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        if profile.name == "alpha":
            alpha_factory_entered.set()
            assert release_alpha_factory.wait(timeout=5)
        return _FakeEngine(profile)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
    )
    first = threading.Thread(
        target=lambda: (
            manager.activate_profile(
                project_id="project-a",
                selector="alpha-selector",
            ),
            completed.append("alpha"),
        )
    )
    second = threading.Thread(
        target=lambda: (
            manager.activate_profile(
                project_id="project-a",
                selector="beta-selector",
            ),
            completed.append("beta"),
        )
    )
    first.start()
    assert alpha_factory_entered.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    with pytest.raises(ProfileContractError, match="no active profile"):
        manager.profile_store.active("project-a", verify=False)
    assert completed == []
    release_alpha_factory.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert completed == ["alpha", "beta"]
    assert manager.profile_store.active("project-a").fingerprint == beta.fingerprint
    manager.close()


@pytest.mark.parametrize(
    "selector",
    [
        "https://user:secret@example.com/profile?token=hidden",
        "../private/profile",
        " alpha-selector",
        "unknown-safe-selector",
    ],
)
def test_request_selector_validation_is_generic_and_never_echoes_input(
    tmp_path: Path,
    selector: str,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: _FakeEngine(profile)
        ),
    )

    with pytest.raises(RoiLaunchError) as engine_error:
        manager.engine_for(selector)
    with pytest.raises(RoiLaunchError) as activation_error:
        manager.activate_profile(project_id="project-a", selector=selector)
    target = SimpleNamespace(
        project_id="project-a",
        profile_fingerprint=profile.fingerprint,
    )
    with pytest.raises(RoiLaunchError) as inference_error:
        manager.infer(selector=selector, target=target)

    assert str(engine_error.value) == "invalid or unknown ROI profile selector"
    assert str(activation_error.value) == "invalid or unknown ROI profile selector"
    assert str(inference_error.value) == "invalid or unknown ROI profile selector"
    assert selector not in str(engine_error.value)
    assert selector not in str(activation_error.value)
    assert selector not in str(inference_error.value)
    assert "secret" not in str(engine_error.value)
    assert "secret" not in str(activation_error.value)
    assert "secret" not in str(inference_error.value)
    manager.close()


def test_concurrent_first_engine_load_executes_factory_exactly_once(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    factory_entered = threading.Event()
    release_factory = threading.Event()
    calls: list[str] = []
    results: list[Any] = []

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        calls.append(profile.name)
        factory_entered.set()
        assert release_factory.wait(timeout=5)
        return _FakeEngine(profile)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
    )
    threads = [
        threading.Thread(
            target=lambda: results.append(manager.engine_for("accepted-selector"))
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    assert factory_entered.wait(timeout=5)
    assert calls == ["accepted"]
    release_factory.set()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 4
    assert all(engine is results[0] for engine in results)
    assert calls == ["accepted"]
    manager.close()


def test_failed_first_load_is_terminally_cached_for_sequential_and_concurrent_retry(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    factory_entered = threading.Event()
    release_factory = threading.Event()
    calls: list[str] = []
    failures: list[BaseException] = []
    private_failure = "token=super-secret-loader-failure"

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        calls.append(profile.name)
        factory_entered.set()
        assert release_factory.wait(timeout=5)
        raise RuntimeError(private_failure)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
    )

    def attempt() -> None:
        try:
            manager.engine_for("accepted-selector")
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(3)]
    for thread in threads:
        thread.start()
    assert factory_entered.wait(timeout=5)
    release_factory.set()
    for thread in threads:
        thread.join(timeout=5)
    attempt()

    assert all(not thread.is_alive() for thread in threads)
    assert calls == ["accepted"]
    assert len(failures) == 4
    assert all(isinstance(exc, RoiLaunchError) for exc in failures)
    assert all(exc.__cause__ is None for exc in failures)
    assert all(private_failure not in str(exc) for exc in failures)
    assert all(private_failure not in _traceback_text(exc) for exc in failures)
    manager.close()


def test_one_global_single_flight_serializes_different_profiles(
    tmp_path: Path,
) -> None:
    alpha = _profile(tmp_path / "alpha-profile", name="alpha")
    beta = _profile(tmp_path / "beta-profile", name="beta")
    config_path = _configured_store(tmp_path, [alpha, beta])
    alpha_entered = threading.Event()
    release_alpha = threading.Event()
    beta_attempting = threading.Event()
    beta_entered = threading.Event()
    counter_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def infer(profile_name: str) -> str:
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if profile_name == "alpha":
                alpha_entered.set()
                assert release_alpha.wait(timeout=5)
            else:
                beta_entered.set()
            return profile_name
        finally:
            with counter_lock:
                active -= 1

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        return _FakeEngine(profile, infer=infer)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
    )
    alpha_engine = manager.engine_for("alpha-selector")
    beta_engine = manager.engine_for("beta-selector")
    results: list[str] = []

    first = threading.Thread(target=lambda: results.append(alpha_engine.infer_one()))

    def run_beta() -> None:
        beta_attempting.set()
        results.append(beta_engine.infer_one())

    second = threading.Thread(target=run_beta)
    first.start()
    assert alpha_entered.wait(timeout=5)
    second.start()
    assert beta_attempting.wait(timeout=5)
    assert not beta_entered.wait(timeout=0.05)
    release_alpha.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert beta_entered.is_set()
    assert maximum_active == 1
    assert results == ["alpha", "beta"]
    manager.close()


def test_profile_artifact_drift_fails_before_engine_factory(tmp_path: Path) -> None:
    profile_root = tmp_path / "profile"
    profile = _profile(profile_root, name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    factory_calls: list[str] = []
    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: lambda **kwargs: factory_calls.append(
            kwargs["profile"].name
        ),
    )
    (profile_root / "base/weights.bin").write_bytes(b"drifted")

    with pytest.raises(RoiLaunchError) as error:
        manager.engine_for("accepted-selector")

    assert error.value.__cause__ is None
    assert factory_calls == []
    manager.close()


def test_loaded_engine_and_retained_proxy_reverify_artifacts_before_every_generation(
    tmp_path: Path,
) -> None:
    profile_root = tmp_path / "profile"
    profile = _profile(profile_root, name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    generation_calls: list[str] = []

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        return _FakeEngine(
            profile,
            infer=lambda name: generation_calls.append(name) or name,
        )

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
    )
    retained = manager.engine_for("accepted-selector")
    assert retained.infer_one() == "accepted"
    assert manager.engine_for("accepted-selector") is retained
    (profile_root / "base/weights.bin").write_bytes(b"same-path-drift")

    with pytest.raises(RoiLaunchError, match="profile verification"):
        retained.infer_one()
    with pytest.raises(RoiLaunchError, match="profile verification"):
        manager.engine_for("accepted-selector").infer_one()

    assert generation_calls == ["accepted"]
    manager.close()


def test_loaded_engine_reverifies_profile_store_fingerprint_before_generation(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: _FakeEngine(profile)
        ),
    )
    retained = manager.engine_for("accepted-selector")
    payload = json.loads(manager.config.profile_store_path.read_text(encoding="utf-8"))
    persisted = payload["profiles"]["accepted"]
    persisted["runtime_identity"]["tokenizer"]["sha256"] = "drifted-tokenizer"
    persisted.pop("profile_fingerprint")
    persisted.pop("identity_fingerprints")
    manager.config.profile_store_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(RoiLaunchError, match="profile verification"):
        retained.infer_one()

    manager.close()


def test_profile_drift_during_factory_load_closes_engine_and_terminally_fails(
    tmp_path: Path,
) -> None:
    profile_root = tmp_path / "profile"
    profile = _profile(profile_root, name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    closed: list[str] = []
    calls: list[str] = []

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        calls.append(profile.name)
        engine = _FakeEngine(profile, closed=closed)
        (profile_root / "base/weights.bin").write_bytes(b"drift-during-load")
        return engine

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
    )

    with pytest.raises(RoiLaunchError) as first:
        manager.engine_for("accepted-selector")
    with pytest.raises(RoiLaunchError) as second:
        manager.engine_for("accepted-selector")

    assert first.value.__cause__ is None
    assert second.value.__cause__ is None
    assert calls == ["accepted"]
    assert closed == ["accepted"]
    manager.close()


def test_managed_engine_does_not_forward_alternate_or_secret_delegate_attributes(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    delegate = _FakeEngine(profile)
    delegate.secret_token = "must-not-escape"
    delegate.alternate_generate = lambda: "unguarded"
    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: lambda **_kwargs: delegate,
    )

    managed = manager.engine_for("accepted-selector")

    assert managed.profile == build_resident_profile_binding(profile)
    with pytest.raises(AttributeError):
        _ = managed.secret_token
    with pytest.raises(AttributeError):
        _ = managed.alternate_generate
    manager.close()


def test_managed_engine_failure_is_sanitized_in_full_traceback(tmp_path: Path) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    private_failure = "token=super-secret-engine-inference"

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config

        def fail(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(private_failure)

        return _FakeEngine(profile, infer=fail)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
    )
    engine = manager.engine_for("accepted-selector")

    with pytest.raises(RoiLaunchError) as error:
        engine.infer_one()

    assert str(error.value) == "resident engine inference failed"
    assert error.value.__cause__ is None
    assert private_failure not in str(error.value)
    assert private_failure not in _traceback_text(error.value)
    manager.close()


def test_managed_engine_cancellation_reason_is_canonicalized(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    private_reason = "token=super-secret-cancellation"
    cancellation = ResidentInferenceCancelled(
        CancellationMetadata(
            requested=True,
            reason=private_reason,
            observed_by_backend=True,
            backend_started=True,
            cuda_synchronized=True,
            deadline_seconds=20.0,
        )
    )

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config

        def cancel(*_args: Any, **_kwargs: Any) -> Any:
            raise cancellation

        return _FakeEngine(profile, infer=cancel)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
    )
    engine = manager.engine_for("accepted-selector")

    with pytest.raises(ResidentInferenceCancelled) as error:
        engine.infer_one()

    assert error.value.metadata.reason == "client_cancelled"
    assert error.value.__cause__ is None
    assert private_reason not in _traceback_text(error.value)
    manager.close()


def test_close_is_deterministic_idempotent_and_invalidates_retained_engines(
    tmp_path: Path,
) -> None:
    alpha = _profile(tmp_path / "alpha-profile", name="alpha")
    beta = _profile(tmp_path / "beta-profile", name="beta")
    config_path = _configured_store(tmp_path, [alpha, beta])
    closed: list[str] = []

    def factory(*, profile: EngineProfile, config: dict[str, Any]) -> _FakeEngine:
        del config
        return _FakeEngine(profile, closed=closed)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: factory,
    )
    beta_engine = manager.engine_for("beta-selector")
    manager.engine_for("alpha-selector")

    manager.close()
    manager.close()

    assert closed == ["alpha", "beta"]
    with pytest.raises(RoiLaunchError, match="closed"):
        beta_engine.infer_one()
    with pytest.raises(RoiLaunchError, match="closed"):
        manager.engine_for("alpha-selector")


def test_close_waits_for_inflight_generation_and_all_concurrent_callers_finish_together(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    generation_entered = threading.Event()
    release_generation = threading.Event()
    close_done = [threading.Event(), threading.Event()]
    close_failures: list[BaseException] = []

    def infer(_name: str) -> str:
        generation_entered.set()
        assert release_generation.wait(timeout=5)
        return "done"

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: _FakeEngine(profile, infer=infer)
        ),
    )
    retained = manager.engine_for("accepted-selector")
    generation_result: list[str] = []
    generation = threading.Thread(
        target=lambda: generation_result.append(retained.infer_one())
    )
    generation.start()
    assert generation_entered.wait(timeout=5)

    def close(index: int) -> None:
        try:
            manager.close()
        except BaseException as exc:
            close_failures.append(exc)
        finally:
            close_done[index].set()

    first = threading.Thread(target=close, args=(0,))
    second = threading.Thread(target=close, args=(1,))
    first.start()
    assert manager._closed.wait(timeout=5)
    second.start()
    assert not close_done[0].wait(timeout=0.05)
    assert not close_done[1].wait(timeout=0.05)
    release_generation.set()
    generation.join(timeout=5)
    first.join(timeout=5)
    second.join(timeout=5)

    assert generation_result == ["done"]
    assert close_failures == []
    assert close_done[0].is_set() and close_done[1].is_set()
    with pytest.raises(RoiLaunchError, match="closed"):
        retained.infer_one()


def test_concurrent_close_callers_observe_the_same_terminal_failure(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    config_path = _configured_store(tmp_path, [profile])
    close_entered = threading.Event()
    release_close = threading.Event()
    private_failure = "token=super-secret-close-failure"
    errors: list[BaseException] = []

    class FailingCloseEngine(_FakeEngine):
        def close(self) -> None:
            close_entered.set()
            assert release_close.wait(timeout=5)
            raise RuntimeError(private_failure)

    manager = RoiLaunchManager(
        config_path,
        current_targets=_CurrentTargets(),
        engine_factory_loader=lambda _target: (
            lambda *, profile, config: FailingCloseEngine(profile)
        ),
    )
    manager.engine_for("accepted-selector")

    def close() -> None:
        try:
            manager.close()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=close)
    second = threading.Thread(target=close)
    first.start()
    assert close_entered.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    assert errors == []
    release_close.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert len(errors) == 2
    assert all(isinstance(error, RoiLaunchError) for error in errors)
    assert all(error.__cause__ is None for error in errors)
    assert all(private_failure not in str(error) for error in errors)
    assert all(private_failure not in _traceback_text(error) for error in errors)
    with pytest.raises(RoiLaunchError) as sequential:
        manager.close()
    assert sequential.value.__cause__ is None
    assert private_failure not in str(sequential.value)
    assert private_failure not in _traceback_text(sequential.value)


def test_default_factory_uses_current_resident_components_and_empty_typed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    sentinel = object()
    captured: dict[str, Any] = {}

    def load(*, resolved: ResolvedInferConfig, profile: Any) -> object:
        captured["resolved"] = resolved
        captured["profile"] = profile
        return sentinel

    monkeypatch.setattr(
        roi_launch_module.ResidentRoiInferenceEngine,
        "load",
        staticmethod(load),
    )

    result = build_resident_engine_from_profile(profile=profile, config={})

    assert result is sentinel
    assert isinstance(captured["resolved"].config, InferConfig)
    assert captured["resolved"].fingerprint == profile.resolved_infer_config_fingerprint
    assert captured["profile"] == build_resident_profile_binding(profile)
    with pytest.raises(RoiLaunchConfigError, match="empty factory config"):
        build_resident_engine_from_profile(
            profile=profile, config={"host": "localhost"}
        )

    tree = ast.parse(Path(roi_launch_module.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "src.infer" not in imported_modules
    assert not any(module.startswith("src.infer.") for module in imported_modules)


def test_default_factory_load_failure_is_sanitized_in_full_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path / "profile", name="accepted")
    private_failure = "token=super-secret-default-loader"

    def fail_load(**_kwargs: Any) -> Any:
        raise RuntimeError(private_failure)

    monkeypatch.setattr(
        roi_launch_module.ResidentRoiInferenceEngine,
        "load",
        staticmethod(fail_load),
    )

    with pytest.raises(RoiLaunchError) as error:
        build_resident_engine_from_profile(profile=profile, config={})

    assert str(error.value) == "resident engine factory failed"
    assert error.value.__cause__ is None
    assert private_failure not in str(error.value)
    assert private_failure not in _traceback_text(error.value)


def _configured_store(
    tmp_path: Path,
    profiles: list[EngineProfile],
    *,
    insertion_ack_timeout_seconds: float | None = 30.0,
) -> Path:
    state = tmp_path / "state"
    profile_store_path = state / "profiles.json"
    store = EngineProfileStore(profile_store_path)
    for profile in profiles:
        store.save(profile)
    selectors = {f"{profile.name}-selector": profile.name for profile in profiles}
    config_path = tmp_path / "operator/roi.json"
    payload = _launch_payload(
        profile_store_path="../state/profiles.json",
        receipt_store_path="../state/receipts.jsonl",
        insertion_ack_timeout_seconds=insertion_ack_timeout_seconds,
        profile_selectors=selectors,
    )
    _write_json(config_path, payload)
    return config_path


def _launch_payload(
    *,
    bind: dict[str, Any] | None = None,
    profile_store_path: str = "profiles.json",
    receipt_store_path: str = "receipts.jsonl",
    insertion_ack_timeout_seconds: float | None = 30.0,
    profile_selectors: dict[str, str] | None = None,
    factory_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ROI_LAUNCH_SCHEMA_VERSION,
        "bind": bind or {"host": "127.0.0.1", "port": 8123},
        "profile_store_path": profile_store_path,
        "receipt_store_path": receipt_store_path,
        "insertion_ack_timeout_seconds": insertion_ack_timeout_seconds,
        "profile_selectors": profile_selectors or {"accepted-selector": "accepted"},
        "engine_factory": {
            "target": DEFAULT_ENGINE_FACTORY_TARGET,
            "config": factory_config or {},
        },
    }


def _profile(
    root: Path,
    *,
    name: str,
    endpoint: str = "http://127.0.0.1:8123/infer",
) -> EngineProfile:
    root.mkdir(parents=True)
    base = root / "base"
    base.mkdir()
    (base / "weights.bin").write_bytes(b"weights-v1")
    model_config = root / "config.json"
    model_config.write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "vision_config": {"patch_size": 16, "spatial_merge_size": 2},
            }
        ),
        encoding="utf-8",
    )
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text('{"vocab":{"a":1}}', encoding="utf-8")
    processor = root / "preprocessor_config.json"
    processor.write_text(
        '{"patch_size":16,"merge_size":2}',
        encoding="utf-8",
    )
    strict_payload: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "name": "roi-launch-test",
            "artifact_root": str(root / "outputs"),
            "collision_policy": "fail",
        },
        "model": {
            "base_model": str(base),
            "dtype": "bf16",
            "attn_implementation": "eager",
            "processor": {"do_resize": False},
            "runtime_patches": {"patch_embed_linearization": "enabled"},
        },
        "data": {"input_jsonl": str(root / "source.jsonl")},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {"system": "detect", "user": "find all objects"},
        },
        "backend": {"type": "hf"},
        "generation": {
            "batch_size": 1,
            "max_new_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        "scoring": {"enabled": True},
        "artifacts": {
            "write_token_trace": True,
            "write_parse_diagnostics": True,
        },
        "debug": {"smoke": True, "dry_run": False},
        "adapter": None,
        "embedding_delta": None,
    }
    config = InferConfig.model_validate(strict_payload)
    config_dict = config.model_dump(mode="json")
    resolved = ResolvedInferConfig(
        config=config,
        config_dict=config_dict,
        fingerprint=sha256_json(config_dict),
        schema_version=1,
        loader_version=INFER_CONFIG_LOADER_VERSION,
        entry_config_path=root / "infer.yaml",
        sources=(),
        path_origins={},
    )
    return EngineProfile.capture(
        name=name,
        endpoint=endpoint,
        artifact_paths={
            "base_weights": base,
            "model_config": model_config,
            "tokenizer": tokenizer,
            "processor": processor,
        },
        resolved_config=resolved,
        roi_inference={
            "processor_factor": 32,
            "default_width": 1024,
            "default_height": 1024,
            "min_axis_pixels": 32,
            "max_axis_pixels": 2048,
            "max_total_pixels": 2_097_152,
            "deadline_seconds": 20.0,
        },
        parser_identity={"id": PARSER_ID, "policy": PARSER_POLICY},
        adapter_identity={"id": RESIDENT_ADAPTER_ID},
        transform_identity={"id": ROI_TRANSFORM_ID},
        transformers_version="4.57.3",
        processor_kwargs={"do_resize": False, "return_tensors": "pt"},
        runtime_identity={
            "model": {"backend": "hf", "device": "cuda:0"},
            "processor": {"patch_size": 16, "merge_size": 2},
            "tokenizer": {"sha256": "test-tokenizer"},
        },
    )


def _resolved_from_profile(profile: EngineProfile) -> ResolvedInferConfig:
    payload = json.loads(profile.resolved_config_json)
    roi_sidecar = payload.pop("roi_inference")
    assert isinstance(roi_sidecar, dict)
    config = InferConfig.model_validate(payload)
    config_dict = config.model_dump(mode="json")
    return ResolvedInferConfig(
        config=config,
        config_dict=config_dict,
        fingerprint=sha256_json(config_dict),
        schema_version=config.schema_version,
        loader_version=INFER_CONFIG_LOADER_VERSION,
        entry_config_path=Path("/__test_resident_profile__.json"),
        sources=(),
        path_origins={},
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
