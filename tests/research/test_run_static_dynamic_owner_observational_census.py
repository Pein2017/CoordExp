from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from scripts.research import run_static_dynamic_owner_observational_census as census
from scripts.research import run_static_dynamic_owner_interface_experiment as runner
from scripts.research import run_static_post_llm_image_field_probe as static


IMAGE_TOKEN = 5
S_TERMINAL = 103
A_COMMIT = 104
NATIVE_BOX_END = 151649
NATIVE_COMMIT = 151669
HIDDEN = 4


def _generated_row_boundaries(
    rows: list[list[int]],
    *,
    strict_owner_by_row: dict[int, str],
) -> list[dict[str, object]]:
    """Mirror the immutable H0 ledger's global physical-row receipts."""

    boundaries: list[dict[str, object]] = []
    offset = 0
    for row_index, row in enumerate(rows):
        owner_id = strict_owner_by_row.get(row_index)
        boundaries.append({
            "prediction_index": row_index,
            "generated_order": row_index,
            "row_start_step": offset,
            "closure_step": offset + len(row) - 1,
            "closure_token_id": row[-1],
            "match_status": "tp" if owner_id is not None else "unmatched",
            "gt_owner_id": owner_id,
        })
        offset += len(row)
    return boundaries


class TinyBlock(torch.nn.Module):
    _coordexp_decoder_layer = True

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        torch.nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.proj(hidden_states)


class TinyMerger(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        torch.nn.init.eye_(self.proj.weight)

    def forward(self, image_states: torch.Tensor) -> torch.Tensor:
        return self.proj(image_states)


class TinyQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.embed = torch.nn.Embedding(256, HIDDEN)
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([TinyBlock() for _ in range(28)])
        self.model.language_model.norm = torch.nn.LayerNorm(HIDDEN)
        self.visual = torch.nn.Module()
        self.visual.merger = TinyMerger()

    def forward(self, *, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        self.forward_calls += 1
        hidden = self.embed(input_ids)
        positions = [index for index, value in enumerate(input_ids[0].tolist()) if value == IMAGE_TOKEN]
        merged = self.visual.merger(hidden[:, positions, :])
        hidden = hidden.clone()
        hidden[:, positions, :] = merged
        for layer in self.model.language_model.layers:
            hidden = layer(hidden)
        hidden = self.model.language_model.norm(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _contract() -> census.WrapperContract:
    return census.WrapperContract(
        checkpoint="S",
        wrapper="object_box_closed",
        terminal_token_id=S_TERMINAL,
        terminal_token_name="box_end",
        h0_prefix_sha256="a" * 64,
        mrope_sha256="b" * 64,
        prefix_token_ids_sha256="c" * 64,
    )


def _capture() -> census.NativePrefillCapture:
    model = TinyQwen()
    ids = torch.tensor([[9, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 101, S_TERMINAL]])
    positions = torch.arange(ids.shape[1], dtype=torch.long).reshape(1, -1).repeat(3, 1).unsqueeze(1)
    return census.capture_native_prefill(
        model,
        checkpoint="S",
        image_id=1,
        input_ids=ids,
        image_positions=[1, 2, 3, 4],
        terminal_bindings=[
            {
                "owner_id": "gt:1:0",
                "position": 6,
                "token_id": S_TERMINAL,
                "source": "native_h0_complete_row",
                "prefix_sha256": "a" * 64,
            },
            {
                "owner_id": "gt:1:1",
                "position": 6,
                "token_id": S_TERMINAL,
                "source": "h0_complete_row_terminal_state",
                "prefix_sha256": "a" * 64,
            },
        ],
        contract=_contract(),
        position_ids=positions,
    )


def test_single_numeric_gpu_and_shard_partition_contract() -> None:
    assert census.validate_single_numeric_cuda_visible_devices("7") == {
        "raw": "7",
        "tokens": ["7"],
        "selected_physical_device": "7",
        "count": 1,
    }
    with pytest.raises(census.CensusContractError, match="exactly one numeric"):
        census.validate_single_numeric_cuda_visible_devices("7,8")
    assert census.parse_shard_selector("S:1/3") == ("S", 1, 3)
    assert census.parse_shard_selector("1/3", default_checkpoint="A") == ("A", 1, 3)
    assert census.partition_image_ids([4, 1, 2, 3], shard_index=0, shard_count=2) == (1, 3)
    assert census.partition_image_ids([4, 1, 2, 3], shard_index=1, shard_count=2) == (2, 4)


def test_cli_help_distinguishes_cpu_contract_live_capture_and_merge() -> None:
    help_text = " ".join(census.build_parser().format_help().split())
    assert "contract" in help_text and "without loading a model" in help_text
    assert "capture" in help_text and "load and attest" in help_text
    assert "merge" in help_text and "CPU-only" in help_text


def test_native_prefill_captures_all_layers_terminal_queries_once_and_cleans_hooks() -> None:
    capture = _capture()
    assert capture.receipt["forward_count"] == 1
    assert capture.receipt["cleanup_complete"] is True
    assert set(capture.states) == set(census.LAYER_NAMES)
    assert all(count == 1 for count in capture.receipt["hook_counts"].values())
    assert set(capture.terminal_states) == set(census.LAYER_NAMES) - {"merger_output"}
    assert set(capture.terminal_states["final_norm"]) == {"gt:1:0", "gt:1:1"}
    assert all("state_sha256" in receipt for receipt in capture.receipt["state_receipts"].values())
    assert all("values" not in receipt for receipt in capture.receipt["state_receipts"].values())


def test_terminal_query_binding_rejects_synthetic_and_prefix_mismatch() -> None:
    ids = torch.tensor([[S_TERMINAL]])
    with pytest.raises(census.TechnicalInvalid, match="native H0"):
        census.validate_terminal_query_binding(
            {"owner_id": "x", "position": 0, "token_id": S_TERMINAL, "source": "synthetic", "prefix_sha256": "a" * 64},
            input_ids=ids,
            contract=_contract(),
        )
    with pytest.raises(census.TechnicalInvalid, match="prefix hash"):
        census.validate_terminal_query_binding(
            {"owner_id": "x", "position": 0, "token_id": S_TERMINAL, "source": "native_h0_complete_row", "prefix_sha256": "d" * 64},
            input_ids=ids,
            contract=_contract(),
        )


def test_overlap_shared_cells_are_not_reused_as_owner_prototypes() -> None:
    owners = [
        {"owner_id": "a", "image_id": 1, "class_name": "person", "bbox": [0, 0, 75, 100], "image_size": [100, 100]},
        {"owner_id": "b", "image_id": 1, "class_name": "person", "bbox": [25, 0, 100, 100], "image_size": [100, 100]},
    ]
    regions = census.build_owner_regions(owners, image_grid_thw=[1, 2, 2], merge_size=1)
    assert regions["a"]["shared_only"]
    assert regions["b"]["shared_only"]
    assert regions["a"]["exclusive"] == {}
    assert regions["a"]["not_measured_reason"] == "exclusive_support_vanished"
    state = torch.ones((4, HIDDEN))
    prototype, receipt = census.weighted_owner_prototype(state, regions["a"]["exclusive"])
    assert prototype is None
    assert receipt["available"] is False


def test_census_controls_are_target_blind_and_deterministic() -> None:
    capture = _capture()
    owners = [
        {"owner_id": "gt:1:0", "image_id": 1, "class_name": "person", "bbox": [0, 0, 60, 100], "image_size": [100, 100]},
        {"owner_id": "gt:1:1", "image_id": 1, "class_name": "person", "bbox": [40, 0, 100, 100], "image_size": [100, 100]},
    ]
    first = census.compute_observational_census(capture, owners, image_grid_thw=[1, 2, 2], merge_size=1)
    second = census.compute_observational_census(capture, owners, image_grid_thw=[1, 2, 2], merge_size=1)
    assert first["rows"] == second["rows"]
    assert first["target_blind_controls"] is True
    assert first["no_efficacy_thresholds"] is True
    assert len(first["rows"]) == len(census.LAYER_NAMES) * 2
    controls = first["rows"][0]["controls"]
    assert controls["shuffled_owner"]["target_blind"] is True
    assert controls["shuffled_owner"]["shuffle_seed"] == census.DEFAULT_SHUFFLE_SEED
    permutation = controls["shuffled_owner"]["permutation"]
    assert controls["shuffled_owner"]["derangement"] is True
    assert all(owner_id != donor_id for owner_id, donor_id in permutation.items())
    assert "denominator" in first["rows"][0]["geometry_labels"]["density"]
    assert first["rows"][0]["geometry_labels"]["uses_layer_state"] is False


def test_probe_free_state_readouts_and_shuffle_derangement_are_explicit() -> None:
    capture = _capture()
    owners = [
        {"owner_id": "gt:1:0", "image_id": 1, "class_name": "person", "bbox": [0, 0, 50, 50], "image_size": [100, 100]},
        {"owner_id": "gt:1:1", "image_id": 1, "class_name": "person", "bbox": [50, 0, 100, 50], "image_size": [100, 100]},
        {"owner_id": "gt:1:2", "image_id": 1, "class_name": "car", "bbox": [0, 50, 50, 100], "image_size": [100, 100]},
    ]
    result = census.compute_observational_census(capture, owners, image_grid_thw=[1, 2, 2], merge_size=1)
    row = next(item for item in result["rows"] if item["owner_id"] == "gt:1:0" and item["layer"] == "final_norm")
    assert row["geometry_labels"]["interpretation"] == "labels_and_geometry_baselines_only"
    assert row["state_readouts"]["foreground_state_contrast"]["uses_layer_state"] is True
    assert row["state_readouts"]["class_state_contrast"]["uses_layer_state"] is True
    density_class = row["state_readouts"]["density_conditioned_class_state_contrast"]
    assert density_class["uses_layer_state"] is True
    assert density_class["conditioning"] == "nearest_fractional_occupancy_per_class_role"
    permutation = row["controls"]["shuffled_owner"]["permutation"]
    assert set(permutation) == {item["owner_id"] for item in owners}
    assert all(owner_id != donor_id for owner_id, donor_id in permutation.items())

    single = census.compute_observational_census(capture, owners[:1], image_grid_thw=[1, 2, 2], merge_size=1)
    single_row = next(item for item in single["rows"] if item["layer"] == "final_norm")
    shuffled = single_row["controls"]["shuffled_owner"]
    assert shuffled["derangement"] is False
    assert shuffled["own_cosine"]["not_measured_reason"] == "derangement_requires_at_least_two_owners"


def _identity() -> dict[str, object]:
    result = {
        key: {"sha256": f"{index + 1:064x}"}
        for index, key in enumerate(("config", "panel", "cohort", "h0", "runtime", "prefix", "wrapper", "mrope"))
    }
    result["wrapper"] = {"sha256": census.sha256_json(_contract().recipe_receipt())}
    return result


def _runtime_attestation(device: str) -> dict[str, object]:
    return {
        "status": "validated",
        "passed": True,
        "physical_device_id": device,
        "physical_device_uuid": f"GPU-{device * 8}",
        "pid": 1,
    }


def _row_identity(image_id: int) -> dict[str, object]:
    identity = _identity()
    identity["prefix"] = {"sha256": census.sha256_json({"prefix": image_id})}
    identity["mrope"] = {"sha256": census.sha256_json({"mrope": image_id})}
    return identity


def _manifest(index: int) -> dict[str, object]:
    image_ids = [1, 2, 3, 4]
    assigned = census.partition_image_ids(image_ids, shard_index=index, shard_count=2)
    identity = _identity()
    for key in ("prefix", "mrope"):
        identity[key] = {
            "sha256": census.sha256_json(
                sorted(_row_identity(image_id)[key]["sha256"] for image_id in assigned)
            )
        }
    return census.build_shard_manifest(
        checkpoint="S",
        shard_index=index,
        shard_count=2,
        image_ids=image_ids,
        identity=identity,
        runtime_attestation=_runtime_attestation("7"),
        contract=_contract(),
    )


def _valid_row(image_id: int, manifest: dict[str, object]) -> dict[str, object]:
    identity = _row_identity(image_id)
    capture = {
        "schema_version": census.RECEIPT_SCHEMA_VERSION,
        "unit_id": census.UNIT_ID,
        "checkpoint": "S",
        "image_id": image_id,
        "forward_count": 1,
        "cleanup_complete": True,
        "passed": True,
        "nonfinite_state_count": 0,
        "hook_counts": {name: 1 for name in census.LAYER_NAMES},
        "input_ids_sha256": identity["prefix"]["sha256"],
        "mrope_sha256": identity["mrope"]["sha256"],
    }
    p1 = {
        "schema_version": census.P1_CENSUS_SCHEMA_VERSION,
        "unit_id": census.UNIT_ID,
        "checkpoint": "S",
        "image_id": image_id,
        "layer_names": list(census.LAYER_NAMES),
        "owner_count": 1,
        "rows": [{"owner_id": f"gt:{image_id}:0", "layer": layer} for layer in census.LAYER_NAMES],
        "capture_receipt": capture,
    }
    return {
        "schema_version": census.P1_CENSUS_SCHEMA_VERSION,
        "status": "valid",
        "checkpoint": "S",
        "image_id": image_id,
        "identity": identity,
        "runtime_attestation": manifest["runtime_attestation"],
        "capture_receipt": capture,
        "p1_census": p1,
    }


def test_shard_write_collision_and_exact_cpu_merge(tmp_path) -> None:
    first_manifest = _manifest(0)
    first_rows = [_valid_row(image_id, first_manifest) for image_id in (1, 3)]
    first = census.write_shard(tmp_path / "s0", manifest=first_manifest, census_rows=first_rows)
    second_manifest = _manifest(1)
    second_manifest["runtime_attestation"] = _runtime_attestation("8")
    second_rows = [_valid_row(image_id, second_manifest) for image_id in (2, 4)]
    census.write_shard(tmp_path / "s1", manifest=second_manifest, census_rows=second_rows)
    merged = census.merge_shards([tmp_path / "s0", tmp_path / "s1"], output_dir=tmp_path / "merged", expected_checkpoint="S", expected_image_ids=[1, 2, 3, 4])
    assert merged["receipt"]["image_ids"] == [1, 2, 3, 4]
    assert merged["receipt"]["p1_row_count"] == 4
    assert [item["attestation"]["physical_device_id"] for item in merged["receipt"]["per_shard_runtime_attestations"]] == ["7", "8"]
    with pytest.raises(census.CensusContractError, match="fail-collision"):
        changed_rows = [_valid_row(image_id, first_manifest) for image_id in (1, 3)]
        changed_rows[1]["changed"] = True
        census.write_shard(tmp_path / "s0", manifest=first_manifest, census_rows=changed_rows)
    assert first["receipt"]["status"] == "valid"


def _write_merge_fixture(tmp_path) -> None:
    for index, images in ((0, (1, 3)), (1, (2, 4))):
        manifest = _manifest(index)
        if index == 1:
            manifest["runtime_attestation"] = _runtime_attestation("8")
        census.write_shard(
            tmp_path / f"s{index}",
            manifest=manifest,
            census_rows=[_valid_row(image_id, manifest) for image_id in images],
        )


def test_merge_rejects_stale_manifest_hash_invalid_attestation_and_row_capture_tamper(tmp_path) -> None:
    stale_root = tmp_path / "stale"
    _write_merge_fixture(stale_root)
    manifest_path = stale_root / "s0" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collision_policy"] = "tampered"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(census.CensusContractError, match="manifest hash mismatch"):
        census.merge_shards([stale_root / "s0", stale_root / "s1"], output_dir=stale_root / "merged")

    attestation_root = tmp_path / "attestation"
    _write_merge_fixture(attestation_root)
    manifest_path = attestation_root / "s0" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_attestation"] = {"status": "not_applicable", "passed": False}
    manifest_bytes = census.canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    receipt_path = attestation_root / "s0" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = census.sha256_bytes(manifest_bytes)
    receipt_path.write_bytes(census.canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(census.CensusContractError, match="attestation must be validated"):
        census.merge_shards([attestation_root / "s0", attestation_root / "s1"], output_dir=attestation_root / "merged")

    row_root = tmp_path / "row"
    _write_merge_fixture(row_root)
    rows_path = row_root / "s0" / "p1-census.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["capture_receipt"]["input_ids_sha256"] = "f" * 64
    rows[0]["p1_census"]["capture_receipt"]["input_ids_sha256"] = "f" * 64
    rows_bytes = b"".join(census.canonical_json_bytes(row) + b"\n" for row in rows)
    rows_path.write_bytes(rows_bytes)
    receipt_path = row_root / "s0" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["census_rows_sha256"] = census.sha256_bytes(rows_bytes)
    receipt_path.write_bytes(census.canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(census.CensusContractError, match="prefix identity differs from capture"):
        census.merge_shards([row_root / "s0", row_root / "s1"], output_dir=row_root / "merged")


def test_nonfinite_capture_fails_closed() -> None:
    class BadMerger(TinyMerger):
        def forward(self, image_states: torch.Tensor) -> torch.Tensor:
            result = super().forward(image_states)
            return result.masked_fill(torch.ones_like(result, dtype=torch.bool), float("nan"))

    # The final norm hook sees the non-finite model output and must reject it.
    model = TinyQwen()
    model.visual.merger = BadMerger()
    ids = torch.tensor([[9, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 101, S_TERMINAL]])
    positions = torch.arange(ids.shape[1], dtype=torch.long).reshape(1, -1).repeat(3, 1).unsqueeze(1)
    with pytest.raises(census.TechnicalInvalid, match="non-finite"):
        census.capture_native_prefill(
            model,
            checkpoint="S",
            image_id=1,
            input_ids=ids,
            image_positions=[1, 2, 3, 4],
            terminal_bindings=[{"owner_id": "x", "position": 6, "token_id": S_TERMINAL, "source": "native_h0_complete_row", "prefix_sha256": "a" * 64}],
            contract=_contract(),
            position_ids=positions,
        )


class ProductionShapedTestAdapter(runner.ExperimentRuntimeAdapter):
    """A CPU adapter with the real adapter's image-runtime/H0 shape.

    It is intentionally supplied directly with ``test_only=True``.  Production
    code must instead call the owning runner's live attestor.
    """

    def __init__(
        self,
        image_ids: tuple[int, ...],
        *,
        physical_device_id: str,
        checkpoint: str = "S",
        test_only: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.test_only = test_only
        self.model = TinyQwen()
        self.physical_device_id = physical_device_id
        self.attest_calls = 0
        self.parse_calls = 0
        self.exact_model_input_calls = 0
        self._wrapper_contract = static.WrapperContract(
            assistant_format="object_box_commit" if checkpoint == "A" else "object_box_closed",
            object_ref_start_token_id=100,
            object_ref_end_token_id=101,
            box_start_token_id=102,
            box_end_token_id=S_TERMINAL,
            coordinate_token_start_id=10,
            commit_token_id=A_COMMIT if checkpoint == "A" else None,
        )
        identities = _identity()
        self._census_identity = {
            "config_sha256": identities["config"]["sha256"],
            "panel_sha256": identities["panel"]["sha256"],
            "cohort_sha256": identities["cohort"]["sha256"],
            "h0": {"run_manifest_sha256": identities["h0"]["sha256"]},
            "resolved_config_fingerprint": identities["runtime"]["sha256"],
        }
        self.panel_rows: dict[str, runner.ImageRuntime] = {}
        self.h0_ledger_records: dict[str, dict[str, dict[str, object]]] = {}
        for image_id in image_ids:
            self._add_image_runtime(image_id)

    @property
    def model_device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def wrapper_contract(self) -> static.WrapperContract:
        return self._wrapper_contract

    def _add_image_runtime(self, image_id: int) -> None:
        owner_id = f"gt:{image_id}:0"
        covered_owner_id = f"covered:{image_id}"
        prompt_ids = torch.tensor([[9 + image_id, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 101]], dtype=torch.long)
        span = static.derive_image_span(
            prompt_ids,
            image_token_id=IMAGE_TOKEN,
            image_grid_thw=torch.tensor([1, 2, 2]),
            merge_size=1,
        )
        terminal_row = [100, 42, 101, 102, 10, 11, 12, 13, S_TERMINAL]
        if self.checkpoint == "A":
            terminal_row.append(A_COMMIT)
        runtime = runner.ImageRuntime(
            image_id=str(image_id),
            raw={"image_id": image_id},
            prompt_ids=prompt_ids,
            prompt_record=SimpleNamespace(),
            native_inputs={"pixel_values": torch.zeros((1, 1), dtype=torch.float32)},
            image_grid_thw=torch.tensor([1, 2, 2]),
            image_token_id=IMAGE_TOKEN,
            merge_size=1,
            image_span=span,
            owners=[{
                "owner_id": owner_id,
                "category": "person",
                "pixel_bbox": [10, 10, 60, 60],
            }],
            h0={
                "rows": [terminal_row],
                "generated_token_ids": terminal_row,
                "generated_token_ids_sha256": census.sha256_token_ids(terminal_row),
            },
            width=100,
            height=100,
            owner_mapping={owner_id: {"source_panel_object_index": 0}},
        )
        self.panel_rows[str(image_id)] = runtime
        self.h0_ledger_records[str(image_id)] = {
            owner_id: {
                "natural_boundary": 1,
                "covered_owner_ids": (covered_owner_id,),
                "latest_covered_owner_id": covered_owner_id,
                "exact_prefix_token_ids": tuple(terminal_row),
                "exact_prefix_sha256": census.sha256_token_ids(terminal_row),
                "generated_row_boundaries": _generated_row_boundaries(
                    [terminal_row],
                    strict_owner_by_row={0: covered_owner_id},
                ),
            }
        }

    def attest_runtime(self) -> dict[str, object]:
        self.attest_calls += 1
        return {
            "status": "validated",
            "passed": True,
            "physical_device_id": self.physical_device_id,
            "physical_device_uuid": f"GPU-{self.physical_device_id * 8}",
            "pid": 1,
        }

    def exact_model_inputs(
        self,
        runtime: runner.ImageRuntime,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, object], torch.Tensor, str]:
        self.exact_model_input_calls += 1
        ids = input_ids.to(device=self.model_device, dtype=torch.long)
        positions = torch.arange(ids.shape[1], dtype=torch.long).reshape(1, -1).repeat(3, 1).unsqueeze(1)
        return (
            {
                "input_ids": ids,
                "position_ids": positions,
                "attention_mask": torch.ones_like(ids) if attention_mask is None else attention_mask,
                "pixel_values": torch.zeros((1, 1), dtype=torch.float32),
                "use_cache": False,
                "return_dict": True,
            },
            positions,
            census.sha256_json({"mrope": runtime.image_id}),
        )

    def parse_row(self, row_tokens: list[int], runtime: runner.ImageRuntime, *, row_index: int) -> dict[str, object]:
        assert row_index == 0
        assert row_tokens == runtime.h0["rows"][0]
        self.parse_calls += 1
        return {
            "owner_match": {
                "status": "unique",
                "owner_id": f"covered:{runtime.image_id}",
            }
        }


class BoundaryHistoryAdapter(ProductionShapedTestAdapter):
    def __init__(self, *, checkpoint: str) -> None:
        super().__init__((1,), physical_device_id="7", checkpoint=checkpoint)
        runtime = self.panel_rows["1"]
        closure = A_COMMIT if checkpoint == "A" else S_TERMINAL
        first = [100, 40, 101, 102, 10, 11, 12, 13, S_TERMINAL]
        unmatched = [100, 41, 101, 102, 14, 15, 16, 17, S_TERMINAL]
        if checkpoint == "A":
            first.append(A_COMMIT)
            unmatched.append(A_COMMIT)
        assert first[-1] == closure and unmatched[-1] == closure
        history = [*first, *unmatched]
        runtime.h0 = {
            "rows": [first, unmatched],
            "generated_token_ids": history,
            "generated_token_ids_sha256": census.sha256_token_ids(history),
        }
        runtime.owners = [
            {"owner_id": "gt:1:0", "category": "person", "pixel_bbox": [0, 0, 50, 100]},
            {"owner_id": "gt:1:1", "category": "car", "pixel_bbox": [50, 0, 100, 100]},
        ]
        self.h0_ledger_records["1"] = {
            "gt:1:0": {
                "natural_boundary": 1,
                "covered_owner_ids": ("covered:1",),
                "latest_covered_owner_id": "covered:1",
                "exact_prefix_token_ids": tuple(history),
                "exact_prefix_sha256": census.sha256_token_ids(history),
                "generated_row_boundaries": _generated_row_boundaries(
                    [first, unmatched],
                    strict_owner_by_row={0: "covered:1"},
                ),
            },
            "gt:1:1": {
                "natural_boundary": 0,
                "covered_owner_ids": (),
                "latest_covered_owner_id": None,
                "exact_prefix_token_ids": (),
                "exact_prefix_sha256": census.sha256_token_ids([]),
                "generated_row_boundaries": _generated_row_boundaries(
                    [first, unmatched],
                    strict_owner_by_row={0: "covered:1"},
                ),
            },
        }

    def parse_row(self, row_tokens: list[int], runtime: runner.ImageRuntime, *, row_index: int) -> dict[str, object]:
        del row_tokens, runtime
        self.parse_calls += 1
        if row_index == 0:
            return {"owner_match": {"status": "unique", "owner_id": "covered:1"}}
        assert row_index == 1
        return {"owner_match": {"status": "unmatched", "owner_id": None}}


class AuthoritativeLedgerBoundaryAdapter(ProductionShapedTestAdapter):
    """A-shaped reproduction of the image-14038 target-23 prefix boundary."""

    TARGET_OWNER = "gt:14038:23"
    COVERED = (
        "gt:14038:2",
        "gt:14038:3",
        "gt:14038:4",
        "gt:14038:9",
        "gt:14038:14",
        "gt:14038:18",
    )

    def __init__(self) -> None:
        super().__init__((14038,), physical_device_id="7", checkpoint="A")
        self.model.embed = torch.nn.Embedding(152671, HIDDEN)
        self._wrapper_contract = static.WrapperContract(
            assistant_format="object_box_commit",
            object_ref_start_token_id=100,
            object_ref_end_token_id=101,
            box_start_token_id=102,
            box_end_token_id=NATIVE_BOX_END,
            coordinate_token_start_id=10,
            commit_token_id=NATIVE_COMMIT,
        )
        runtime = self.panel_rows["14038"]
        rows = [
            [120 + row_index, NATIVE_BOX_END, NATIVE_COMMIT]
            for row_index in range(15)
        ]
        generated = [token for row in rows for token in row]
        exact_prefix = [token for row in rows[:14] for token in row]
        strict_owner_by_row = {
            1: self.COVERED[0],
            2: self.COVERED[1],
            3: self.COVERED[2],
            4: self.COVERED[3],
            5: self.COVERED[4],
            7: self.COVERED[5],
            14: self.TARGET_OWNER,
        }
        runtime.h0 = {
            "rows": rows,
            "generated_token_ids": generated,
            "generated_token_ids_sha256": census.sha256_token_ids(generated),
        }
        runtime.owners = [{
            "owner_id": self.TARGET_OWNER,
            "category": "book",
            "pixel_bbox": [985, 480, 1060, 502],
        }]
        runtime.width = 1200
        runtime.height = 800
        generated_boundaries = _generated_row_boundaries(
            rows,
            strict_owner_by_row=strict_owner_by_row,
        )
        # One complete wrapper row is category-invalid and therefore absent
        # from scored prediction boundaries; absence is authoritative
        # unmatched, not permission to rerun the local parser as a matcher.
        del generated_boundaries[10]
        for prediction_index, boundary in enumerate(generated_boundaries):
            boundary.update({
                "prediction_index": prediction_index,
                "object_span_id": f"coco2017_val_000000014038:span-{prediction_index}",
                "span_end_step": int(boundary["closure_step"]) - 1,
                "closure_token_text": "<|commit|>",
                "commit_step": boundary["closure_step"],
            })
        self.h0_ledger_records["14038"] = {
            self.TARGET_OWNER: {
                "natural_boundary": 6,
                "valid_prediction_count": 14,
                "covered_owner_ids": self.COVERED,
                "latest_covered_owner_id": self.COVERED[-1],
                "exact_prefix_token_ids": tuple(exact_prefix),
                "exact_prefix_sha256": census.sha256_token_ids(exact_prefix),
                "generated_row_boundaries": generated_boundaries,
            }
        }

    def parse_row(self, row_tokens: list[int], runtime: runner.ImageRuntime, *, row_index: int) -> dict[str, object]:
        assert row_tokens == runtime.h0["rows"][row_index]
        self.parse_calls += 1
        # Deliberately adversarial: the local parser maps globally unmatched
        # duplicate rows to owners, including the queried target.  The global
        # one-to-one H0 ledger remains the only strict-coverage authority.
        local_owner = self.TARGET_OWNER if row_index == 8 else f"local:{row_index}"
        return {"owner_match": {"status": "unique", "owner_id": local_owner}}


@pytest.mark.parametrize("checkpoint", ["S", "A"])
def test_real_h0_bridge_uses_physical_prefix_closure_allows_unmatched_and_root(checkpoint: str) -> None:
    adapter = BoundaryHistoryAdapter(checkpoint=checkpoint)
    payload = census._existing_experiment_payload(adapter, image_id=1, owner_ids=("gt:1:0", "gt:1:1"))
    assert len(payload["terminal_bindings"]) == 1
    binding = payload["terminal_bindings"][0]
    assert binding["natural_boundary"] == 1
    assert binding["terminal_row_index"] == 1
    assert binding["latest_covered_owner_id"] == "covered:1"
    assert binding["physical_terminal_owner_id"] is None
    assert payload["terminal_query_statuses"]["gt:1:1"] == {
        "status": "not_measured",
        "natural_boundary": 0,
        "not_measured_reason": "root_history_has_no_preceding_terminal",
    }
    capture = census.capture_native_prefill(
        adapter.model,
        checkpoint=checkpoint,
        image_id=1,
        input_ids=payload["input_ids"],
        image_positions=payload["image_positions"],
        terminal_bindings=payload["terminal_bindings"],
        terminal_prefix_hashes=payload["terminal_prefix_hashes"],
        terminal_query_statuses=payload["terminal_query_statuses"],
        contract=payload["contract"],
        position_ids=payload["position_ids"],
        model_inputs=payload["model_inputs"],
    )
    result = census.compute_observational_census(
        capture,
        payload["owners"],
        image_grid_thw=payload["image_grid_thw"],
        merge_size=payload["merge_size"],
    )
    root_row = next(row for row in result["rows"] if row["owner_id"] == "gt:1:1" and row["layer"] == "final_norm")
    assert root_row["retrieval"]["own_cosine"]["not_measured_reason"] == "root_history_has_no_preceding_terminal"

    root_only = census._existing_experiment_payload(adapter, image_id=1, owner_ids=("gt:1:1",))
    assert root_only["terminal_bindings"] == []
    root_capture = census.capture_native_prefill(
        adapter.model,
        checkpoint=checkpoint,
        image_id=1,
        input_ids=root_only["input_ids"],
        image_positions=root_only["image_positions"],
        terminal_bindings=[],
        terminal_prefix_hashes={},
        terminal_query_statuses=root_only["terminal_query_statuses"],
        contract=root_only["contract"],
        position_ids=root_only["position_ids"],
        model_inputs=root_only["model_inputs"],
    )
    assert root_capture.receipt["forward_count"] == 1
    assert root_capture.terminal_states["final_norm"] == {}


def test_real_h0_bridge_uses_authoritative_global_matches_and_latest_physical_closure() -> None:
    adapter = AuthoritativeLedgerBoundaryAdapter()
    payload = census._existing_experiment_payload(
        adapter,
        image_id=14038,
        owner_ids=(adapter.TARGET_OWNER,),
    )

    assert adapter.parse_calls == 15
    assert len(payload["terminal_bindings"]) == 1
    binding = payload["terminal_bindings"][0]
    assert binding["terminal_row_index"] == 13
    assert binding["physical_prefix_row_count"] == 14
    assert binding["physical_terminal_closure_step"] == 41
    assert binding["strict_covered_owner_ids"] == list(adapter.COVERED)
    assert binding["latest_covered_owner_id"] == "gt:14038:18"
    assert binding["physical_terminal_match_status"] == "unmatched"
    assert binding["physical_terminal_owner_id"] is None
    assert binding["intervening_unmatched_row_count"] == 6
    assert len(binding["intervening_unmatched_rows_sha256"]) == 64
    status = payload["terminal_query_statuses"][adapter.TARGET_OWNER]
    assert status["latest_covered_owner_id"] == "gt:14038:18"
    assert status["physical_terminal_match_status"] == "unmatched"
    assert status["physical_terminal_owner_id"] is None
    assert status["intervening_unmatched_row_count"] == 6
    assert status["intervening_unmatched_rows_sha256"] == binding["intervening_unmatched_rows_sha256"]


def test_a14038_actual_shaped_ledger_survives_runner_loader_and_census_bridge(
    tmp_path,
) -> None:
    adapter = AuthoritativeLedgerBoundaryAdapter()
    source_record = adapter.h0_ledger_records["14038"][adapter.TARGET_OWNER]
    ledger_path = tmp_path / "a14038-h0-ledger.json"
    ledger_path.write_text(
        json.dumps({
            "unit_id": runner.UNIT_ID,
            "checkpoint": "A",
            "records": [{
                "image_id": 14038,
                "gt_owner_id": adapter.TARGET_OWNER,
                "natural_boundary": source_record["natural_boundary"],
                "covered_owner_ids": list(source_record["covered_owner_ids"]),
                "latest_covered_owner_id": source_record["latest_covered_owner_id"],
                "exact_prefix_token_ids": list(source_record["exact_prefix_token_ids"]),
                "exact_prefix_sha256": source_record["exact_prefix_sha256"],
                "strict_complete_row": False,
                "native_tp": False,
                "native_fn": True,
                "parse_status": "accepted_with_drops",
                "valid_prediction_count": source_record["valid_prediction_count"],
                "generated_row_boundaries": source_record["generated_row_boundaries"],
            }],
        }) + "\n",
        encoding="utf-8",
    )
    digest = runner.sha256_file(ledger_path)
    loaded, _identity = runner._load_h0_ledger_records(
        sources={"h0_ledgers": [{"path": str(ledger_path), "sha256": digest}]},
        manifest_sources={"h0_ledgers": [digest]},
        checkpoint="A",
    )
    loaded_record = loaded["14038"][adapter.TARGET_OWNER]
    assert loaded_record["valid_prediction_count"] == 14
    assert len(loaded_record["generated_row_boundaries"]) == 14
    assert loaded_record["generated_row_boundaries"][10]["generated_order"] == 11

    adapter.h0_ledger_records = loaded
    payload = census._existing_experiment_payload(
        adapter,
        image_id=14038,
        owner_ids=(adapter.TARGET_OWNER,),
    )
    binding = payload["terminal_bindings"][0]
    assert binding["terminal_row_index"] == 13
    assert binding["strict_covered_owner_ids"] == list(adapter.COVERED)
    assert binding["physical_terminal_match_status"] == "unmatched"


def test_real_h0_bridge_rejects_authoritative_boundary_owner_mismatch() -> None:
    adapter = AuthoritativeLedgerBoundaryAdapter()
    record = adapter.h0_ledger_records["14038"][adapter.TARGET_OWNER]
    boundaries = record["generated_row_boundaries"]
    assert isinstance(boundaries, list)
    boundaries[7]["gt_owner_id"] = "gt:14038:999"

    with pytest.raises(census.TechnicalInvalid, match="strict covered-owner sequence"):
        census._existing_experiment_payload(
            adapter,
            image_id=14038,
            owner_ids=(adapter.TARGET_OWNER,),
        )


def _cohort(tmp_path, image_ids: tuple[int, ...]) -> object:
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps({
            "unit_id": census.UNIT_ID,
            "events": [
                {
                    "image_id": image_id,
                    "gt_owner_id": f"gt:{image_id}:0",
                    "checkpoint_status": {"S": {"disposition": "established"}},
                }
                for image_id in image_ids
            ],
        }) + "\n",
        encoding="utf-8",
    )
    return cohort


def _run_live(tmp_path, *, adapter: ProductionShapedTestAdapter, cohort, image_shard: str, output_name: str) -> dict[str, object]:
    return census.run_observational_census(
        "live",
        checkpoint="S",
        infer_config=tmp_path / "infer.yaml",
        panel=tmp_path / "panel.jsonl",
        cohort=cohort,
        h0_dir=None,
        ledger=None,
        image_shard=image_shard,
        output_dir=tmp_path / output_name,
        adapter=adapter,
    )


def test_production_shaped_adapter_captures_each_assigned_image_once_and_measures_cyclic_controls(tmp_path) -> None:
    cohort = _cohort(tmp_path, (1, 2, 3, 4))
    adapter = ProductionShapedTestAdapter((1, 2, 3, 4), physical_device_id="7")
    result = _run_live(tmp_path, adapter=adapter, cohort=cohort, image_shard="S:0/2", output_name="shard")

    assert result["receipt"]["status"] == "valid"
    assert result["receipt"]["row_count"] == 2
    assert adapter.attest_calls == 1
    assert adapter.model.forward_calls == 2
    assert adapter.parse_calls == 2
    assert adapter.exact_model_input_calls == 2
    rows = result["rows"]
    assert [row["image_id"] for row in rows] == [1, 3]
    partner_by_image = {1: 3, 3: 1}
    for row in rows:
        assert row["capture_receipt"]["forward_count"] == 1
        terminal_binding = row["capture_receipt"]["terminal_bindings"][0]
        assert terminal_binding["natural_boundary"] == 1
        assert terminal_binding["terminal_row_index"] == 0
        assert terminal_binding["latest_covered_owner_id"] == f"covered:{row['image_id']}"
        assert terminal_binding["physical_terminal_owner_id"] == f"covered:{row['image_id']}"
        for readout in row["p1_census"]["rows"]:
            cyclic = readout["controls"]["cyclic_next_image_same_normalized_geometry"]
            if readout["layer"] == "merger_output":
                assert cyclic["own_cosine"]["not_measured_reason"] == "missing_h0_terminal_query_state"
                continue
            assert cyclic["own_cosine"]["available"] is True
            binding = cyclic["partner_binding"]
            assert binding["partner_image_id"] == partner_by_image[row["image_id"]]
            assert binding["partner_image_grid_thw"] == [1, 2, 2]
            assert len(binding["partner_prefix_token_ids_sha256"]) == 64
            assert len(binding["partner_state_sha256"]) == 64
    serialized = (tmp_path / "shard" / "p1-census.jsonl").read_text(encoding="utf-8")
    assert "tensor(" not in serialized


def test_real_adapter_attestation_failure_fails_closed_without_test_attestation_fallback(tmp_path, monkeypatch) -> None:
    cohort = _cohort(tmp_path, (1, 2))
    adapter = ProductionShapedTestAdapter((1, 2), physical_device_id="7", test_only=False)

    def _fail_live_attestation(_adapter) -> object:
        raise runner.OrchestrationError("attestation boom")

    monkeypatch.setattr(runner, "_attest_live_runtime", _fail_live_attestation)
    with pytest.raises(census.TechnicalInvalid, match="attestation failed closed"):
        _run_live(tmp_path, adapter=adapter, cohort=cohort, image_shard="S:0/1", output_name="failed")
    assert adapter.attest_calls == 0
    assert adapter.model.forward_calls == 0


def test_single_image_shard_is_an_explicit_cross_image_preflight_hold(tmp_path) -> None:
    cohort = _cohort(tmp_path, (1,))
    adapter = ProductionShapedTestAdapter((1,), physical_device_id="7")
    with pytest.raises(census.CensusContractError, match="preflight hold: cyclic cross-image"):
        _run_live(tmp_path, adapter=adapter, cohort=cohort, image_shard="S:0/1", output_name="held")
    assert adapter.model.forward_calls == 0


def test_capture_failure_quarantines_every_assigned_image_without_partial_valid_rows(tmp_path) -> None:
    class BadSecondImageAdapter(ProductionShapedTestAdapter):
        def exact_model_inputs(
            self,
            runtime: runner.ImageRuntime,
            input_ids: torch.Tensor,
            *,
            attention_mask: torch.Tensor | None = None,
        ) -> tuple[dict[str, object], torch.Tensor, str]:
            payload, positions, mrope = super().exact_model_inputs(
                runtime,
                input_ids,
                attention_mask=attention_mask,
            )
            if runtime.image_id == "2":
                positions = positions[:2]
                payload["position_ids"] = positions
            return payload, positions, mrope

    cohort = _cohort(tmp_path, (1, 2))
    adapter = BadSecondImageAdapter((1, 2), physical_device_id="7")
    result = _run_live(tmp_path, adapter=adapter, cohort=cohort, image_shard="S:0/1", output_name="quarantined")
    assert result["receipt"]["status"] == "quarantined"
    assert result["receipt"]["row_count"] == 2
    assert {row["image_id"] for row in result["rows"]} == {1, 2}
    assert {row["status"] for row in result["rows"]} == {"technical_invalid"}
    assert adapter.model.forward_calls == 1


def test_live_shards_merge_common_recipe_identities_with_distinct_physical_attestations(tmp_path) -> None:
    cohort = _cohort(tmp_path, (1, 2, 3, 4))
    first = _run_live(
        tmp_path,
        adapter=ProductionShapedTestAdapter((1, 2, 3, 4), physical_device_id="7"),
        cohort=cohort,
        image_shard="S:0/2",
        output_name="s0",
    )
    second = _run_live(
        tmp_path,
        adapter=ProductionShapedTestAdapter((1, 2, 3, 4), physical_device_id="8"),
        cohort=cohort,
        image_shard="S:1/2",
        output_name="s1",
    )
    assert first["manifest"]["identity"]["config"] == second["manifest"]["identity"]["config"]
    assert first["manifest"]["identity"]["wrapper"] == second["manifest"]["identity"]["wrapper"]
    assert len({row["identity"]["prefix"]["sha256"] for row in first["rows"]}) == 2
    assert len({row["identity"]["mrope"]["sha256"] for row in first["rows"]}) == 2
    assert first["manifest"]["runtime_attestation"]["physical_device_id"] == "7"
    assert second["manifest"]["runtime_attestation"]["physical_device_id"] == "8"
    merged = census.merge_shards(
        [tmp_path / "s0", tmp_path / "s1"],
        output_dir=tmp_path / "merged",
        expected_checkpoint="S",
        expected_image_ids=[1, 2, 3, 4],
    )
    assert merged["receipt"]["image_ids"] == [1, 2, 3, 4]
    assert [item["attestation"]["physical_device_id"] for item in merged["receipt"]["per_shard_runtime_attestations"]] == ["7", "8"]
