from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from probes.human13 import magnitude_finite as finite
from probes.human13.magnitude_finite import (
    ADAPTER_TENSOR_NAME,
    CANDIDATE_NAME,
    MATERIALIZATION_RECEIPT_NAME,
    RECEIPT_NAME,
    CapturedSpecimen,
    aggregate_natural_gate,
    aggregate_n2_natural_gate,
    materialize_finite_adapter,
    scan_full_vocabulary_margin,
    train_finite_candidate,
)
from probes.human13.magnitude_qp import (
    MagnitudeSurface,
    MechanicalInvalid,
    _tensor_sha256,
)
from src.qwen.special_token_embeddings import (
    SelectedDeltaOutputHead,
    SpecialTokenSelection,
)


class _Language(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.tensor([0.0]))
        self.right = nn.Parameter(torch.tensor([0.0]))
        self.frozen = nn.Parameter(torch.tensor(7.0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        scale = torch.cat((self.left, self.right))
        return SimpleNamespace(last_hidden_state=x * scale)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language = _Language()


class _FailOnThirdHeadCall(nn.Module):
    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head
        self.calls = 0

    @property
    def weight(self) -> nn.Parameter:
        return self.head.weight

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("post-update margin failure")
        return self.head(states)


def _fixture() -> tuple[
    _Model, SelectedDeltaOutputHead, MagnitudeSurface, tuple[CapturedSpecimen, ...]
]:
    model = _Model()
    base = nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        base.weight.copy_(
            torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
        )
    head = SelectedDeltaOutputHead(
        base,
        SpecialTokenSelection(token_strings=("<unseen>",), token_ids=(3,)),
        nn.Parameter(torch.tensor([[4.0, 4.0]]), requires_grad=False),
    )
    parameters = (model.language.left, model.language.right)
    names = ("language.left", "language.right")
    surface = MagnitudeSurface(
        model=model,
        language_module_name="language",
        names=names,
        functional_names=("left", "right"),
        parameters=parameters,
        shapes=((1,), (1,)),
        tensor_hashes=tuple(_tensor_sha256(item) for item in parameters),
        scalar_count=2,
    )
    specimens = (
        CapturedSpecimen(
            image_id=6040,
            language_args=(),
            language_kwargs={"x": torch.tensor([[[1.0, 0.0]]])},
            decision_positions=torch.tensor([0]),
            target_token_ids=torch.tensor([0]),
            route_receipt={"token_count": 1},
        ),
        CapturedSpecimen(
            image_id=16228,
            language_args=(),
            language_kwargs={"x": torch.tensor([[[0.0, 1.0]]])},
            decision_positions=torch.tensor([0]),
            target_token_ids=torch.tensor([1]),
            route_receipt={"token_count": 1},
        ),
    )
    return model, head, surface, specimens


def _run(
    *,
    language: nn.Module,
    head: nn.Module,
    surface: MagnitudeSurface,
    specimens: tuple[CapturedSpecimen, ...],
    output_dir: Path,
) -> dict[str, object]:
    return train_finite_candidate(
        language=language,
        head=head,
        surface=surface,
        specimens=specimens,
        output_dir=output_dir,
        steps=1,
        learning_rate=0.1,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        check_every=1,
        seed=17,
    )


def test_shared_finite_update_and_actual_full_vocabulary_scan(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    model, head, surface, specimens = _fixture()
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name not in surface.names
    }
    head_before = tuple(parameter.detach().clone() for parameter in head.parameters())

    result = _run(
        language=model.language,
        head=head,
        surface=surface,
        specimens=specimens,
        output_dir=tmp_path,
    )

    candidate = load_file(str(tmp_path / CANDIDATE_NAME))
    # Sensitivity witness: omitting either image's backward leaves its disjoint
    # magnitude at Source; both nonzero tensors therefore require one shared step.
    assert candidate["language.left"].item() != 0.0
    assert candidate["language.right"].item() != 0.0
    assert result["optimizer"]["steps_executed"] == 1
    assert result["measurements"]["training_language_forward_count"] == 2
    assert all(parameter.item() == 0.0 for parameter in surface.parameters)
    assert all(
        torch.equal(dict(model.named_parameters())[name], value)
        for name, value in frozen_before.items()
    )
    assert all(
        torch.equal(parameter, value)
        for parameter, value in zip(head.parameters(), head_before, strict=True)
    )

    # A route-only scan over tokens 0/1 would miss token 3; the actual selected-
    # delta wrapper makes it the exhaustive full-vocabulary worst competitor.
    scan = scan_full_vocabulary_margin(
        head(torch.tensor([[1.0, 1.0]])), torch.tensor([0])
    )
    assert scan["worst_competitor_token_id"] == 3
    assert scan["full_vocabulary_exhaustive"] is True


def test_exception_after_shared_update_restores_exact_source(tmp_path: Path) -> None:
    model, head, surface, specimens = _fixture()
    source_hashes = tuple(_tensor_sha256(item) for item in surface.parameters)

    with pytest.raises(RuntimeError, match="post-update margin failure"):
        _run(
            language=model.language,
            head=_FailOnThirdHeadCall(head),
            surface=surface,
            specimens=specimens,
            output_dir=tmp_path,
        )

    assert tuple(_tensor_sha256(item) for item in surface.parameters) == source_hashes
    assert not (tmp_path / CANDIDATE_NAME).exists()


def test_receipt_is_not_published_before_source_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, head, surface, specimens = _fixture()

    def fail_restore(*_args: object) -> None:
        raise MechanicalInvalid("injected restoration failure")

    monkeypatch.setattr(finite, "_restore_source", fail_restore)
    with pytest.raises(MechanicalInvalid, match="injected restoration failure"):
        _run(
            language=model.language,
            head=head,
            surface=surface,
            specimens=specimens,
            output_dir=tmp_path,
        )

    assert (tmp_path / CANDIDATE_NAME).exists()
    assert not (tmp_path / RECEIPT_NAME).exists()


@pytest.mark.parametrize("card_kind", ["missing", "markdown", "json"])
def test_materialization_mapping_and_natural_gate_order_is_nongating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    card_kind: str,
) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    source_adapter = tmp_path / "source"
    source_adapter.mkdir()
    (source_adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(finite.same_panel.BASE_MODEL),
                "peft_type": "LORA",
                "use_dora": True,
                "target_modules": ["q_proj"],
                "r": 1,
                "lora_alpha": 1,
            }
        )
    )
    if card_kind != "missing":
        (source_adapter / "README.md").write_text("source adapter\n")
    if card_kind == "json":
        from src.artifacts.model_card import package_model_card

        package_model_card(source_adapter)
    source = {}
    candidate = {}
    for index in range(finite.CANDIDATE_TENSOR_COUNT):
        target = f"model.language_model.layers.{index}.q_proj"
        saved = f"base_model.model.{target}"
        source[f"{saved}.lora_A.weight"] = torch.tensor([[float(index)]])
        source[f"{saved}.lora_B.weight"] = torch.tensor([[float(-index)]])
        source[f"{saved}.lora_magnitude_vector"] = torch.tensor([1.0])
        candidate[f"{target}.lora_magnitude_vector.default.weight"] = torch.tensor(
            [float(index + 10)]
        )
    save_file(
        source,
        str(source_adapter / ADAPTER_TENSOR_NAME),
        metadata={"format": "pt"},
    )
    candidate_path = tmp_path / CANDIDATE_NAME
    save_file(candidate, str(candidate_path))
    monkeypatch.setattr(
        finite,
        "EXPECTED_CANDIDATE_SHA256",
        finite.same_panel.sha256_file(candidate_path),
    )
    candidate_receipt = tmp_path / RECEIPT_NAME
    candidate_receipt_body = {
        "disposition": "FINITE_MARGIN_PASS",
        "candidate_path": str(candidate_path),
        "candidate_sha256": finite.same_panel.sha256_file(candidate_path),
        "image_ids": list(finite.IMAGE_IDS),
        "owner_count": finite.OWNER_COUNT,
        "decision_count": finite.DECISION_COUNT,
    }
    candidate_receipt.write_text(
        json.dumps(
            {
                **candidate_receipt_body,
                "content_sha256": finite.same_panel.sha256_json(candidate_receipt_body),
            }
        )
    )
    output_adapter = tmp_path / "final-adapter"

    materialized = materialize_finite_adapter(
        candidate_receipt_path=candidate_receipt,
        output_adapter=output_adapter,
        source_adapter=source_adapter,
    )

    observed = load_file(str(output_adapter / ADAPTER_TENSOR_NAME))
    assert not (output_adapter / "README.md").exists()
    assert (output_adapter / "model_card.json").exists() == (card_kind != "missing")
    assert materialized["schema_version"].endswith(".v2")
    files, metadata = finite._verify_materialization_files(
        output_adapter, materialized, stage="n2"
    )
    assert set(files) == {"adapter_config.json", ADAPTER_TENSOR_NAME}
    assert metadata["mode"] == "current_json"
    for key, value in source.items():
        if ".lora_magnitude_vector" not in key:
            assert torch.equal(observed[key], value)
    for live_name, value in candidate.items():
        saved_name = "base_model.model." + live_name.removesuffix(".default.weight")
        assert torch.equal(observed[saved_name], value)

    def loaded_name(saved_name: str) -> str:
        normalized = saved_name.removeprefix("base_model.model.")
        if normalized.endswith(".lora_magnitude_vector"):
            return normalized + ".default.weight"
        return normalized.removesuffix(".weight") + ".default.weight"

    live = {
        loaded_name(name): nn.Parameter(value.clone())
        for name, value in observed.items()
    }
    model = SimpleNamespace(named_parameters=lambda: tuple(live.items()))
    assert (
        finite._validate_live_adapter_readback(model, output_adapter)["tensor_count"]
        == 3 * finite.CANDIDATE_TENSOR_COUNT
    )
    next(iter(live.values())).data.add_(1)
    with pytest.raises(MechanicalInvalid, match="tensor persistence"):
        finite._validate_live_adapter_readback(model, output_adapter)

    with safe_open(
        output_adapter / ADAPTER_TENSOR_NAME, framework="pt", device="cpu"
    ) as handle:
        assert handle.metadata() == {"format": "pt"}
    assert materialized["mapped_magnitude_count"] == finite.CANDIDATE_TENSOR_COUNT
    assert materialized["model_load_count"] == 0
    assert (output_adapter / MATERIALIZATION_RECEIPT_NAME).is_file()
    assert (
        finite.same_panel._natural_order_violation_row_count(
            ((0.2, 0.1, 0.3, 0.2), (0.1, 0.9, 0.2, 1.0), (0.15, 0.0, 0.2, 0.1))
        )
        == 2
    )
    assert (
        finite.same_panel._natural_order_violation_row_count(
            ((0.1, 0.9, 0.2, 1.0), (0.15, 0.0, 0.2, 0.1), (0.2, 0.1, 0.3, 0.2))
        )
        == 0
    )

    primary = [
        {
            "image_id": image_id,
            "matched_owner_count": {"50": owners, "60": owners, "80": owners},
            "duplicate_count": 0,
            "unmatched_prediction_count": 0,
            "malformed_count": 0,
            "cap_debt": False,
            "natural_eos": True,
            "natural_order_violation": True,
            "natural_order_violation_row_count": 2,
        }
        for image_id, owners in zip(finite.IMAGE_IDS, (15, 50), strict=True)
    ]
    monitor = [
        {**item, "matched_owner_count": dict(item["matched_owner_count"])}
        for item in primary
    ]
    gate = aggregate_n2_natural_gate(primary, monitor)
    assert gate["disposition"] == "N2_PASS"
    assert gate["natural_order_monitor"] == {
        "gating": False,
        "rp1_0_violation_image_count": 2,
        "rp1_0_violation_row_count": 4,
        "rp1_10_violation_image_count": 2,
        "rp1_10_violation_row_count": 4,
    }
    primary[0]["unmatched_prediction_count"] = 1
    assert (
        aggregate_n2_natural_gate(primary, monitor)["disposition"]
        == "M_ONLY_REPLAY_PASS_NATURAL_FAIL"
    )


def test_n4_gate_and_stage_receipt_binding_fail_closed(tmp_path: Path) -> None:
    n4_ids = finite.same_panel.STAGES["N4"].image_ids
    assert finite._candidate_name("n4") == (
        "human13-n4-dora-magnitude-finite.safetensors"
    )
    assert finite._receipt_name("n4") == "human13-n4-dora-magnitude-finite.json"
    assert finite._materialization_receipt_name("n4") == (
        "human13-n4-materialization.json"
    )
    primary = [
        {
            "image_id": image_id,
            "matched_owner_count": {
                "50": owners,
                "60": owners,
                "80": owners,
            },
            "duplicate_count": 0,
            "unmatched_prediction_count": 0,
            "malformed_count": 0,
            "cap_debt": False,
            "natural_eos": True,
        }
        for image_id, owners in zip(n4_ids, (25, 15, 33, 50), strict=True)
    ]
    monitor = [{**item} for item in primary]
    gate = aggregate_natural_gate(primary, monitor, stage="n4")
    assert gate["disposition"] == "N4_PASS"
    assert gate["required_iou50_owner_count"] == 123
    assert gate["image_ids"] == [4134, 6040, 13923, 16228]

    primary[0]["image_id"] = 6040
    with pytest.raises(MechanicalInvalid, match="frozen N4 image order"):
        aggregate_natural_gate(primary, monitor, stage="n4")

    candidate_path = tmp_path / "human13-n4-dora-magnitude-finite.safetensors"
    candidate_path.write_bytes(b"n4 candidate")
    receipt_path = tmp_path / "human13-n4-dora-magnitude-finite.json"
    body = {
        "disposition": "FINITE_MARGIN_PASS",
        "stage": "N4",
        "candidate_path": str(candidate_path),
        "candidate_sha256": finite.same_panel.sha256_file(candidate_path),
        "image_ids": list(n4_ids),
        "owner_count": 123,
        "decision_count": 1147,
    }
    receipt_path.write_text(
        json.dumps({**body, "content_sha256": finite.same_panel.sha256_json(body)})
    )
    receipt, observed_path = finite._load_finite_candidate_receipt(
        receipt_path, stage="n4"
    )
    assert receipt["stage"] == "N4"
    assert observed_path == candidate_path

    body["stage"] = "N2"
    receipt_path.write_text(
        json.dumps({**body, "content_sha256": finite.same_panel.sha256_json(body)})
    )
    with pytest.raises(MechanicalInvalid, match="binding differs"):
        finite._load_finite_candidate_receipt(receipt_path, stage="n4")


def test_n13_stage_registration_and_gate_fail_closed() -> None:
    n13 = finite.same_panel.STAGES["N13"]
    assert finite._stage("n13") == ("N13", n13)
    assert n13.image_ids == (
        1584,
        2299,
        2685,
        4134,
        5001,
        6040,
        7511,
        10707,
        13348,
        13923,
        14038,
        14439,
        16228,
    )
    assert n13.owner_count == 392
    assert n13.decision_state_count == 3637

    primary = [
        {
            "image_id": image_id,
            "matched_owner_count": {
                "50": owners,
                "60": owners,
                "80": owners,
            },
            "duplicate_count": 0,
            "unmatched_prediction_count": 0,
            "malformed_count": 0,
            "cap_debt": False,
            "natural_eos": True,
        }
        for image_id, owners in zip(
            n13.image_ids,
            (19, 46, 29, 37, 23, 15, 44, 19, 15, 21, 47, 27, 50),
            strict=True,
        )
    ]
    monitor = [{**item} for item in primary]
    gate = aggregate_natural_gate(primary, monitor, stage="n13")
    assert gate["disposition"] == "N13_PASS"
    assert gate["required_iou50_owner_count"] == 392
    assert gate["all_natural_eos"] is True

    primary[-1]["natural_eos"] = False
    assert (
        aggregate_natural_gate(primary, monitor, stage="n13")["disposition"]
        == "M_ONLY_REPLAY_PASS_NATURAL_FAIL"
    )


def test_training_cli_defaults_n2_and_dispatches_n4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = [
        "--output-dir",
        str(tmp_path),
        "--steps",
        "300",
        "--learning-rate",
        "0.003",
        "--weight-decay",
        "0",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.999",
        "--adam-eps",
        "1e-8",
        "--check-every",
        "10",
        "--seed",
        "0",
    ]
    assert finite._training_parser().parse_args(required).stage == "n2"

    observed: dict[str, object] = {}

    def fake_run_stage(**kwargs: object) -> dict[str, str]:
        observed.update(kwargs)
        return {"disposition": "FINITE_MARGIN_PASS"}

    monkeypatch.setattr(finite, "run_stage", fake_run_stage)
    assert finite.main(["--stage", "n4", *required]) == 0
    assert observed["stage"] == "n4"
    assert "adapter_path" not in observed
    assert "candidate_receipt" not in observed

    observed.clear()
    assert finite.main(["--stage", "n13", *required]) == 0
    assert observed["stage"] == "n13"

    with pytest.raises(SystemExit):
        finite._training_parser().parse_args(["--stage", "N13", *required])


def test_n13_materialize_and_readback_cli_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[dict[str, object]] = []

    def fake_materialize(**kwargs: object) -> dict[str, str]:
        observed.append(kwargs)
        return {"disposition": "materialized_unmerged"}

    def fake_verify(**kwargs: object) -> dict[str, str]:
        observed.append(kwargs)
        return {"disposition": "N13_PASS"}

    monkeypatch.setattr(finite, "materialize_finite_adapter", fake_materialize)
    monkeypatch.setattr(finite, "verify_adapter", fake_verify)
    assert (
        finite.main(
            [
                "materialize",
                "--stage",
                "n13",
                "--candidate-receipt",
                str(tmp_path / "candidate.json"),
                "--output-adapter",
                str(tmp_path / "adapter"),
            ]
        )
        == 0
    )
    assert (
        finite.main(
            [
                "readback",
                "--stage",
                "n13",
                "--adapter-path",
                str(tmp_path / "adapter"),
                "--output-receipt",
                str(tmp_path / "acceptance.json"),
            ]
        )
        == 0
    )
    assert [item["stage"] for item in observed] == ["n13", "n13"]


def test_new_producer_n2_receipt_keeps_legacy_pin_and_all_payload_bindings(tmp_path):
    candidate = tmp_path / finite.CANDIDATE_NAME
    candidate.write_bytes(b'a new explicit N2 candidate')
    receipt_path = tmp_path / finite.RECEIPT_NAME
    body = {
        'schema_version': finite.CANDIDATE_SCHEMA,
        'producer': {'module': finite.PRODUCER_MODULE, 'source_sha256': 'a' * 64},
        'disposition': 'FINITE_MARGIN_PASS', 'stage': 'N2',
        'candidate_path': str(candidate), 'candidate_sha256': finite.same_panel.sha256_file(candidate),
        'image_ids': list(finite.IMAGE_IDS), 'owner_count': finite.OWNER_COUNT,
        'decision_count': finite.DECISION_COUNT,
    }
    def write(value):
        receipt_path.write_text(json.dumps({**value, 'content_sha256': finite.same_panel.sha256_json(value)}))
    write(body)
    assert finite._load_finite_candidate_receipt(receipt_path)[1] == candidate
    for changed in (
        {**body, 'schema_version': 'human13_dora_magnitude_finite_candidate.v1'},
        {**body, 'schema_version': 'human13_dora_magnitude_finite_candidate.v1', 'stage': None},
        {**body, 'stage': 'N4'},
        {**body, 'decision_count': finite.DECISION_COUNT - 1},
        {**body, 'owner_count': finite.OWNER_COUNT - 1},
        {**body, 'candidate_sha256': 'b' * 64},
        {**body, 'producer': {'module': 'old.script', 'source_sha256': 'a' * 64}},
    ):
        write(changed)
        with pytest.raises(MechanicalInvalid):
            finite._load_finite_candidate_receipt(receipt_path)
    write(body)
    candidate.write_bytes(b'corrupted candidate')
    with pytest.raises(MechanicalInvalid, match='binding differs'):
        finite._load_finite_candidate_receipt(receipt_path)


def test_legacy_materialization_verifies_archived_card_without_restoring_it(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for name in ("adapter_config.json", finite.ADAPTER_TENSOR_NAME):
        (adapter / name).write_bytes(b"retained payload")
    archived = tmp_path / "archived-card.md"
    archived.write_bytes(b"original card\n")
    card_hash = finite.same_panel.sha256_file(archived)
    manifest = tmp_path / "archive.json"
    manifest.write_text(json.dumps({
        "schema": "coordexp.output_source_archive.v1",
        "files": [{"source": str(adapter / "README.md"), "archive": str(archived),
                   "sha256": card_hash}],
    }))
    receipt = {
        "schema_version": "human13_n2_dora_magnitude_materialization.v1",
        "adapter_files": {
            name: finite.same_panel.sha256_file(adapter / name)
            for name in ("adapter_config.json", finite.ADAPTER_TENSOR_NAME)
        } | {"README.md": card_hash},
    }
    payload, metadata = finite._verify_materialization_files(
        adapter, receipt, stage="n2", archive_manifest=manifest
    )
    assert set(payload) == {"adapter_config.json", finite.ADAPTER_TENSOR_NAME}
    assert metadata["mode"] == "historical_markdown"
    assert metadata["binding"]["resolution"] == "archived_exact_path"
    assert not (adapter / "README.md").exists()
    archived.write_bytes(b"changed card")
    with pytest.raises(MechanicalInvalid, match="metadata is unavailable"):
        finite._verify_materialization_files(adapter, receipt, stage="n2", archive_manifest=manifest)


def test_current_materialization_rejects_changed_metadata_without_archive_fallback(tmp_path):
    for name in ("adapter_config.json", finite.ADAPTER_TENSOR_NAME, "model_card.json"):
        (tmp_path / name).write_bytes(b"current payload")
    receipt = {
        "schema_version": "human13_n2_dora_magnitude_materialization.v2",
        "adapter_files": {
            name: finite.same_panel.sha256_file(tmp_path / name)
            for name in ("adapter_config.json", finite.ADAPTER_TENSOR_NAME)
        },
        "metadata_files": {"model_card.json": finite.same_panel.sha256_file(tmp_path / "model_card.json")},
    }
    finite._verify_materialization_files(tmp_path, receipt, stage="n2")
    (tmp_path / "model_card.json").write_bytes(b"changed metadata")
    with pytest.raises(MechanicalInvalid, match="metadata hashes differ"):
        finite._verify_materialization_files(tmp_path, receipt, stage="n2")
