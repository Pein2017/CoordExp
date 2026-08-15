"""CPU contract for the one-image v5 parity-only entry.

Frozen failure-mode matrix (OpenSpec task 6.3 mechanics):

| Invariant | Executable owner | Minimal counterexample | Closing evidence |
|---|---|---|---|
| Exact image, seeds, and RP order | v5 runner | another image/group or RP 1.10 first | success/failure order tests |
| Native engine releases before exact replay | per-RP phase runner | HF surface opens while vLLM is live | event-order assertions |
| Parity only | v5 runner | witness, ledger, optimizer, or owner call | backend exposes only allowed seams |
| Immutable root and zero-action dry run | CLI/run admission | existing root or dry-run callback | immutability/dry-run tests |
| Durable terminal and parity diagnostics | append-only publisher | failure raises without error field/terminal | failure receipt test |
| Exact Source and fp32/SDPA lineage | surface receipt | drifted model/surface identity | success receipt assertions |
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.research import collect_human13_rp_crossover as acquisition


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/research/run_human13_rp_crossover_parity_v5.py"
)
VOCAB_SIZE = acquisition.NATURAL_STOP_TOKEN_ID + 2
PROMPT_IDS = (10, 11, 12)
UNIFORM_LOGPROB = -math.log(VOCAB_SIZE)


def _subject() -> ModuleType:
    assert SCRIPT.is_file(), "the task-6.3 parity-only entry does not exist"
    spec = importlib.util.spec_from_file_location("parity_v5_subject", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _frozen(rp: float) -> SimpleNamespace:
    return SimpleNamespace(
        source_checkpoint_path="/source/checkpoint",
        source_checkpoint_payload_sha256=_digest("source"),
        base_model_path="/source/base",
        adapter_tensor_path="/source/adapter.safetensors",
        adapter_sha256=_digest("adapter"),
        special_embedding_tensor_path="/source/special.safetensors",
        special_embedding_sha256=_digest("special"),
        manifest_path="/source/manifest.json",
        manifest_sha256=_digest("manifest"),
        panel_path="/source/panel.jsonl",
        panel_sha256=_digest("panel"),
        tokenizer_path="/source/tokenizer.json",
        tokenizer_sha256=_digest("tokenizer"),
        prompt_config_path="/source/prompt.yaml",
        prompt_config_sha256=_digest("prompt"),
        prompt_policy_fingerprint=_digest("prompt-policy"),
        alias_bank_sha256=_digest("aliases"),
        c_leaf_path=f"/leaf/rp-{rp}.yaml",
        c_leaf_sha256=_digest(f"leaf:{rp}"),
        qualification_learning_rate_ray=(3e-7, 1e-6, 3e-6, 1e-5, 3e-5),
        default_qualification_learning_rate=3e-6,
    )


def _request_receipt(
    request: acquisition.AcquisitionRequest,
) -> acquisition.NativeRequestReceipt:
    return acquisition.NativeRequestReceipt(
        request_id=request.request_id,
        seed=request.seed,
        physical_batch_index=request.physical_batch_index,
        request_order_in_batch=request.request_order_in_batch,
        sampling_params=acquisition.expected_native_sampling_evidence(request),
        prompt_token_ids_sha256=acquisition.token_ids_sha256(PROMPT_IDS),
        model_id="source-model",
        model_identity_sha256=_digest("vllm-model"),
        session_identity_sha256=_digest("vllm-session"),
    )


def _output_receipt(
    request: acquisition.AcquisitionRequest, *, sampled_error: float
) -> acquisition.NativeOutputReceipt:
    generated = (acquisition.NATURAL_STOP_TOKEN_ID,)
    request_receipt = _request_receipt(request)
    return acquisition.NativeOutputReceipt(
        native_request_receipt_sha256=request_receipt.content_sha256,
        request_id=request.request_id,
        seed=request.seed,
        physical_batch_index=request.physical_batch_index,
        request_order_in_batch=request.request_order_in_batch,
        prompt_token_ids=PROMPT_IDS,
        source_sha256=_digest("source"),
        manifest_sha256=_digest("manifest"),
        model_id="source-model",
        tokenizer_id="source-tokenizer",
        processor_id="source-processor",
        processor_order=("repetition_penalty", "temperature", "log_softmax"),
        sampler_backend_id="vllm:test:processed_logprobs",
        generated_token_ids=generated,
        processed_logprobs=(UNIFORM_LOGPROB + sampled_error,),
        terminal_kind="natural_stop",
    )


@dataclass
class FakeBackend:
    rp: float
    events: list[str]
    sampled_error: float = 0.0
    sampler_open: bool = False

    def open_sampler(self, frozen: Any) -> object:
        assert frozen.c_leaf_sha256 == _digest(f"leaf:{self.rp}")
        assert not self.sampler_open
        self.sampler_open = True
        self.events.append(f"rp:{self.rp}:sampler:open")
        return object()

    def sample_batch(
        self, sampler: object, batch: acquisition.AcquisitionBatch, params: Any
    ) -> acquisition.NativeBatchReceipt:
        del sampler
        assert self.sampler_open
        assert len(batch.requests) == len(params) == 4
        self.events.append(
            f"rp:{self.rp}:batch:{batch.batch_index}:"
            f"{batch.requests[0].seed}-{batch.requests[-1].seed}"
        )
        return acquisition.NativeBatchReceipt(
            requests=tuple(_request_receipt(item) for item in batch.requests),
            outputs=tuple(
                _output_receipt(item, sampled_error=self.sampled_error)
                for item in batch.requests
            ),
        )

    def close_sampler(self, sampler: object) -> None:
        del sampler
        assert self.sampler_open
        self.sampler_open = False
        self.events.append(f"rp:{self.rp}:sampler:close")

    def open_packed_surface(self, frozen: Any) -> object:
        del frozen
        assert not self.sampler_open
        self.events.append(f"rp:{self.rp}:hf:open")
        return object()

    def packed_raw_logits(
        self, packed: object, execution: acquisition.AcquisitionExecution
    ) -> acquisition.PackedRawLogits:
        del packed
        self.events.append(f"rp:{self.rp}:hf:forward")
        identities = tuple(
            (trajectory.identity.request_id, token.token_index)
            for trajectory in execution.group.trajectories
            for token in trajectory.generated_tokens
        )
        return acquisition.PackedRawLogits(
            request_ids=tuple(item[0] for item in identities),
            token_indices=tuple(item[1] for item in identities),
            logits=torch.zeros((len(identities), VOCAB_SIZE), dtype=torch.float32),
        )

    def parity_surface_lineage(
        self, packed: object, execution: acquisition.AcquisitionExecution
    ) -> dict[str, Any]:
        del packed, execution
        self.events.append(f"rp:{self.rp}:hf:lineage")
        return {
            "model_plan_sha256": _digest(f"model-plan:{self.rp}"),
            "mixed_precision": "fp32",
            "attn_implementation": "sdpa",
            "batch_size": 1,
            "forward_owner": (
                "scripts.research.human13_rp_crossover_live_packs:"
                "default_exact_history_forward"
            ),
            "materialize_forward_count": 16,
            "exact_history_model_forward_count": 16,
            "packed_token_count": 64,
            "logical_token_count": 64,
        }

    def close_packed_surface(self, packed: object) -> None:
        del packed
        self.events.append(f"rp:{self.rp}:hf:close")

    def resource_snapshot(self) -> SimpleNamespace:
        self.events.append(f"rp:{self.rp}:resources")
        return SimpleNamespace(
            measurement_scope="fake_live",
            peak_host_rss_bytes=2048,
            cuda_peak_allocated_bytes=4096,
            cuda_peak_reserved_bytes=8192,
            decode_token_count=0,
            packed_token_count=0,
            logical_token_count=0,
            forward_count=0,
            row_bytes=0,
            artifact_bytes=0,
        )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dry_run_is_zero_action_and_zero_write(tmp_path: Path) -> None:
    subject = _subject()
    calls: list[object] = []
    output_root = tmp_path / "v5"

    result = subject.main(
        [],
        _output_root=output_root,
        _runner=lambda **kwargs: calls.append(kwargs),
    )

    assert result == 0
    assert calls == []
    assert not output_root.exists()


def test_direct_script_dry_run_uses_the_repository_import_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=SCRIPT.parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["filesystem_writes"] == 0


def test_success_runs_exact_rp_order_and_publishes_one_terminal(
    tmp_path: Path,
) -> None:
    subject = _subject()
    events: list[str] = []
    output_root = tmp_path / "v5"

    terminal = subject.run_v5_parity_qualification(
        output_root=output_root,
        backend_factory=lambda rp, root: FakeBackend(rp, events),
        frozen_loader=_frozen,
    )

    assert terminal["status"] == "passed"
    assert terminal["completed_rps"] == [1.0, 1.1]
    assert terminal["route_disposition"] == "full_panel_successor_allowed"
    assert events == [
        "rp:1.0:sampler:open",
        "rp:1.0:batch:0:30001-30004",
        "rp:1.0:batch:1:30005-30008",
        "rp:1.0:batch:2:30009-30012",
        "rp:1.0:batch:3:30013-30016",
        "rp:1.0:sampler:close",
        "rp:1.0:hf:open",
        "rp:1.0:hf:forward",
        "rp:1.0:hf:lineage",
        "rp:1.0:hf:close",
        "rp:1.0:resources",
        "rp:1.1:sampler:open",
        "rp:1.1:batch:0:30001-30004",
        "rp:1.1:batch:1:30005-30008",
        "rp:1.1:batch:2:30009-30012",
        "rp:1.1:batch:3:30013-30016",
        "rp:1.1:sampler:close",
        "rp:1.1:hf:open",
        "rp:1.1:hf:forward",
        "rp:1.1:hf:lineage",
        "rp:1.1:hf:close",
        "rp:1.1:resources",
    ]
    assert sorted(path.name for path in output_root.iterdir()) == [
        "rp100",
        "rp110",
        "terminal.json",
    ]
    assert len(tuple(output_root.glob("terminal*.json"))) == 1
    persisted = _load(output_root / "terminal.json")
    assert persisted == terminal
    assert persisted["image_ids"] == [1584]
    assert persisted["seeds"] == list(range(30001, 30017))
    assert persisted["parity_tolerance"] == {
        "per_token_nats": 0.02,
        "group_mean_nats": 0.002,
    }
    for slug, rp in (("rp100", 1.0), ("rp110", 1.1)):
        evidence = _load(output_root / slug / "parity-evidence.json")
        assert evidence["status"] == "passed"
        assert evidence["repetition_penalty"] == rp
        assert evidence["image_id"] == 1584
        assert evidence["seeds"] == list(range(30001, 30017))
        assert evidence["native_request_count"] == 16
        assert evidence["native_batch_count"] == 4
        assert evidence["surface_lineage"]["mixed_precision"] == "fp32"
        assert evidence["surface_lineage"]["attn_implementation"] == "sdpa"
        assert evidence["surface_lineage"]["batch_size"] == 1
        assert evidence["source_lineage"]["source_checkpoint_payload_sha256"] == (
            _digest("source")
        )
        assert evidence["error"] is None
        assert len(evidence["content_sha256"]) == 64
        assert (output_root / slug / "native-acquisition.json").is_file()
        assert (output_root / slug / "replayed-group.json").is_file()
        assert (output_root / slug / "replay-receipt.json").is_file()


def test_first_parity_failure_is_durable_and_stops_before_second_rp(
    tmp_path: Path,
) -> None:
    subject = _subject()
    events: list[str] = []
    output_root = tmp_path / "v5"

    terminal = subject.run_v5_parity_qualification(
        output_root=output_root,
        backend_factory=lambda rp, root: FakeBackend(
            rp, events, sampled_error=0.5 if rp == 1.0 else 0.0
        ),
        frozen_loader=_frozen,
    )

    assert terminal["status"] == "failed"
    assert terminal["completed_rps"] == []
    assert terminal["failed_rp"] == 1.0
    assert terminal["route_disposition"] == "retire_exact_on_policy_route"
    assert not any(event.startswith("rp:1.1") for event in events)
    assert events[-2:] == ["rp:1.0:hf:close", "rp:1.0:resources"]
    error = terminal["error"]
    assert error["exception_type"] == "PolicyReplayError"
    assert error["phase"] == "exact_surface_replay"
    assert error["error_field"] == {
        "token_count": 16,
        "tokens_over_tolerance": 16,
        "max_absolute_error_nats": pytest.approx(0.5),
        "mean_absolute_error_nats": pytest.approx(0.5),
        "max_error_request_id": "human13:1584:rp-crossover:qualification:30001",
        "max_error_token_index": 0,
    }
    evidence = _load(output_root / "rp100" / "parity-evidence.json")
    assert evidence["status"] == "failed"
    assert evidence["error"] == error
    assert (output_root / "rp100" / "native-acquisition.json").is_file()
    assert not (output_root / "rp110").exists()
    assert _load(output_root / "terminal.json")["error"] == error


def test_infrastructure_failure_holds_without_claiming_negative_parity(
    tmp_path: Path,
) -> None:
    subject = _subject()
    events: list[str] = []
    output_root = tmp_path / "v5"

    class BrokenExactSurface(FakeBackend):
        def open_packed_surface(self, frozen: Any) -> object:
            del frozen
            assert not self.sampler_open
            self.events.append(f"rp:{self.rp}:hf:open")
            raise RuntimeError("exact surface unavailable")

    terminal = subject.run_v5_parity_qualification(
        output_root=output_root,
        backend_factory=lambda rp, root: BrokenExactSurface(rp, events),
        frozen_loader=_frozen,
    )

    assert terminal["status"] == "failed"
    assert terminal["route_disposition"] == "hold_infrastructure_failure"
    assert terminal["error"]["phase"] == "exact_surface_replay"
    assert terminal["error"]["error_field"] is None
    assert not any(event.startswith("rp:1.1") for event in events)
    assert (output_root / "terminal.json").is_file()


def test_frozen_lineage_error_is_durable_before_any_backend_action(
    tmp_path: Path,
) -> None:
    subject = _subject()
    output_root = tmp_path / "v5"
    calls: list[object] = []

    terminal = subject.run_v5_parity_qualification(
        output_root=output_root,
        backend_factory=lambda rp, root: calls.append((rp, root)),
        frozen_loader=lambda rp: SimpleNamespace(
            source_checkpoint_path=f"/missing-lineage/{rp}"
        ),
    )

    assert calls == []
    assert terminal["status"] == "failed"
    assert terminal["route_disposition"] == "hold_infrastructure_failure"
    assert terminal["error"]["phase"] == "frozen_source_admission"
    assert "omitted source_checkpoint_payload_sha256" in terminal["error"]["message"]
    assert terminal["phase_status_by_rp"] == {
        "rp100": {
            "native_acquisition": "not_started",
            "exact_surface_replay": "not_started",
        }
    }
    assert (output_root / "rp100" / "parity-evidence.json").is_file()
    assert (output_root / "terminal.json").is_file()


def test_existing_root_fails_before_backend_or_frozen_input_action(
    tmp_path: Path,
) -> None:
    subject = _subject()
    output_root = tmp_path / "v5"
    output_root.mkdir()
    calls: list[object] = []

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        subject.run_v5_parity_qualification(
            output_root=output_root,
            backend_factory=lambda rp, root: calls.append((rp, root)),
            frozen_loader=lambda rp: calls.append(rp),
        )

    assert calls == []
    assert tuple(output_root.iterdir()) == ()


def test_execute_requires_the_existing_model_gpu_authority_flag(
    tmp_path: Path,
) -> None:
    subject = _subject()
    calls: list[object] = []
    output_root = tmp_path / "v5"

    with pytest.raises(PermissionError, match="user-model-gpu-authority"):
        subject.main(
            ["--execute"],
            _output_root=output_root,
            _runner=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []
    assert not output_root.exists()
