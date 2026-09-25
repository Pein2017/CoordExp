from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from probes.coordinate_representation.coordinate_codebook_alignment import semantic_assay as assay
from src.qwen import coordinate_codebook


class _EarlyCodebook:
    mode = "early_patch_edges"
    enabled = True
    spatial_merge_size = 2

    def __init__(self) -> None:
        self._edge_coordinates_override: torch.Tensor | None = None

    def patch_edge_coordinates(self, grid: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        return coordinate_codebook._patch_edge_coordinates(grid, self.spatial_merge_size, device=device)

    @contextmanager
    def override_patch_edges(self, edges: torch.Tensor):
        assert self._edge_coordinates_override is None
        self._edge_coordinates_override = edges.detach()
        try:
            yield
        finally:
            self._edge_coordinates_override = None

    def inject_early(
        self,
        hidden_states: torch.Tensor,
        grid: torch.Tensor,
        *,
        edge_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden_states


class _LateCodebook:
    mode = "late_masked_center"
    enabled = True
    spatial_merge_size = 2


class _FakeModel(nn.Module):
    def __init__(self, kind: str, codebook: Any, *, fail_on_forward: int | None = None) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.kind = kind
        self.coordinate_codebook = codebook
        self.forward_count = 0
        self.fail_on_forward = fail_on_forward

    @staticmethod
    def _logit_row(center: torch.Tensor) -> torch.Tensor:
        bins = torch.arange(1000, device=center.device, dtype=torch.float32)
        row = torch.empty(1002, device=center.device, dtype=torch.float32)
        row[:1000] = -2.0 - ((bins - center * 1000.0) / 20.0).square()
        row[1000:] = torch.tensor([0.0, -0.5], device=center.device)
        return row

    def forward(self, *, input_ids: torch.Tensor, image_grid_thw: torch.Tensor,
                pixel_values: torch.Tensor, attention_mask: torch.Tensor,
                logits_to_keep: int) -> SimpleNamespace:
        self.forward_count += 1
        assert logits_to_keep == 2
        grid = image_grid_thw
        if self.kind == "early16":
            edges = self.coordinate_codebook._edge_coordinates_override
            hidden = torch.zeros((int(grid[0, 1] * grid[0, 2]), 4), device=input_ids.device)
            self.coordinate_codebook.inject_early(hidden, grid, edge_coordinates=edges)
            if edges is None:
                edges = self.coordinate_codebook.patch_edge_coordinates(grid, device=input_ids.device)
            x_center = edges[:, :2].mean()
            y_center = edges[:, 2:].mean()
        else:
            height = int(grid[0, 1]) // self.coordinate_codebook.spatial_merge_size
            width = int(grid[0, 2]) // self.coordinate_codebook.spatial_merge_size
            addresses = coordinate_codebook._normalized_addresses(height, width, device=input_ids.device)
            x_center, y_center = addresses[:, 0].mean(), addresses[:, 1].mean()
        if self.fail_on_forward == self.forward_count:
            raise RuntimeError("synthetic forward failure")
        logits = torch.stack((self._logit_row(x_center), self._logit_row(y_center)))
        return SimpleNamespace(logits=logits.unsqueeze(0))


def _prepared_cases() -> list[dict[str, Any]]:
    cases = []
    grids = ((1, 4, 6), (1, 6, 4), (1, 8, 6), (1, 6, 8))
    for image_id, grid in zip(assay.CASE_IMAGE_IDS, grids, strict=True):
        prompt = [100 + image_id % 100, 200 + image_id % 100]
        x_target, y_target = 300 + image_id % 100, 600 - image_id % 100
        x_prefix = [11, 12]
        case = {
            "image_id": image_id,
            "row_id": f"coco2017_val_{image_id:012d}",
            "cohort": "fit_coco_ordinary",
            "first_description": "fixture",
            "annotation_unique_first_description": True,
            "first_gt_coord_bins": [x_target, y_target],
            "prompt_token_count": len(prompt),
            "prompt_token_ids_sha256": assay._json_digest(prompt),
            "media_sha256": [f"media-{image_id}"],
            "grid_thw": list(grid),
            "merged_grid_hw": [grid[1] // 2, grid[2] // 2],
            "raw_patch_count": grid[1] * grid[2],
            "first_x_query": {"coordinate_role": "x1", "response_prefix_token_ids": x_prefix,
                              "target_token_id": x_target},
            "first_y_query": {"coordinate_role": "y1", "response_prefix_token_ids": [*x_prefix, x_target],
                              "target_token_id": y_target, "earlier_gt_coordinates": ["x1"]},
            "prepared": {
                "native_inputs": {
                    "input_ids": torch.tensor([prompt], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long),
                    "pixel_values": torch.ones((1, 3, 4, 4), dtype=torch.float32),
                    "image_grid_thw": torch.tensor([grid], dtype=torch.long),
                },
                "executed_prompt_token_ids": prompt,
                "executed_media_sha256": [f"media-{image_id}"],
            },
        }
        cases.append(case)
    return cases


def _fake_prepare_replay(model: Any, native_inputs: Any, *, prompt_token_ids: Any,
                         continuation_token_ids: Any, compact_logits: bool) -> SimpleNamespace:
    assert compact_logits is True
    assert len(continuation_token_ids) == 1
    ids = [*prompt_token_ids, *continuation_token_ids]
    inputs = dict(native_inputs)
    inputs.update(input_ids=torch.tensor([ids], dtype=torch.long),
                  attention_mask=torch.ones((1, len(ids)), dtype=torch.long), logits_to_keep=2)
    return SimpleNamespace(inputs=inputs)


def _native_caller_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = copy.deepcopy(_prepared_cases())
    rows = []
    for index, case in enumerate(cases):
        grid = case["grid_thw"]
        case["dataset_row_number"] = (23, 24, 29, 30)[index]
        case["merged_token_count"] = grid[1] * grid[2] // 4
        case["first_description"] = f"description-{index}"
        case["first_gt_coord_bins"] = [case["first_x_query"]["target_token_id"], case["first_y_query"]["target_token_id"]]
        coords = [*case["first_gt_coord_bins"], 999, 999]
        rows.append({
            "image_id": case["image_id"],
            "_admission": {"row_id": case["row_id"], "cohort": case["cohort"]},
            "objects": [{
                "coco_ann_id": 1000 + index,
                "desc": case["first_description"],
                "bbox_2d": [f"<|coord_{value}|>" for value in coords],
            }],
        })
    return cases, rows


def _qwen(kind: str, *, fail_on_forward: int | None = None) -> SimpleNamespace:
    codebook = _EarlyCodebook() if kind == "early16" else _LateCodebook()
    model = _FakeModel(kind, codebook, fail_on_forward=fail_on_forward).eval()
    return SimpleNamespace(model=model, token_identity=SimpleNamespace(coordinate_token_ids=tuple(range(1000))))


def test_patch_edge_warp_preserves_non_square_processor_order_and_other_axis() -> None:
    grid = torch.tensor([[1, 4, 6]], dtype=torch.long)
    edges = coordinate_codebook._patch_edge_coordinates(grid, 2, device=torch.device("cpu"))
    assert edges.shape == (24, 4)
    assert torch.allclose(edges[:4], torch.tensor([
        [0.0, 1 / 6, 0.0, 1 / 4],
        [1 / 6, 2 / 6, 0.0, 1 / 4],
        [0.0, 1 / 6, 1 / 4, 2 / 4],
        [1 / 6, 2 / 6, 1 / 4, 2 / 4],
    ]))
    x_warped = assay._warp_patch_edges(edges, axis=0, sign=1)
    y_warped = assay._warp_patch_edges(edges, axis=1, sign=-1)
    assert torch.equal(x_warped[:, 2:], edges[:, 2:])
    assert torch.equal(y_warped[:, :2], edges[:, :2])
    assert torch.equal(x_warped[[0, 1], :2].min(), torch.tensor(0.0))
    assert torch.equal(x_warped[-1, 1], torch.tensor(1.0))
    assert torch.all(x_warped[:, 0] <= x_warped[:, 1])
    assert torch.all(y_warped[:, 2] <= y_warped[:, 3])


@pytest.mark.parametrize("kind", ("early16", "late16"))
def test_checkpoint_runner_checks_actual_callers_causal_prefixes_and_wrong_axis(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assay, "prepare_replay", _fake_prepare_replay)
    qwen = _qwen(kind)
    result = assay.run_checkpoint_semantic_assay(qwen, checkpoint=kind, prepared_cases=_prepared_cases())
    assert result["teacher_forwards"] == 24
    assert result["max_teacher_forwards_both_checkpoints"] == 48
    assert qwen.model.forward_count == 24
    for row in result["cases"]:
        assert len(row["conditions"]) == 6
        assert row["identity_full_vocabulary_logit_max_abs"] <= assay.IDENTITY_TOLERANCE
        assert row["identity_coordinate_logit_max_abs_by_role"] == {"x1": 0.0, "y1": 0.0}
        assert all(len(readout["coordinate_logits"]) == 1000
                   for condition in row["conditions"].values()
                   for readout in condition.values())
        assert all(len(readout["coordinate_cdf"]) == 1000
                   for condition in row["conditions"].values()
                   for readout in condition.values())
        assert row["calls"]["identity_override"]["address_override_invocations"] == 1
        for intervention in ("x_plus", "x_minus", "y_plus", "y_minus"):
            assert row["calls"][intervention]["address_override_invocations"] == 1
        assert row["directional_responses"]["x1"]["plus_minus"][
            "plus_minus_conditional_mean_coordinate"] > 0
        assert abs(row["directional_responses"]["x1"]["wrong_axis_plus_minus"][
            "plus_minus_conditional_mean_coordinate"]) < 1e-6
        assert row["directional_responses"]["y1"]["plus_minus"][
            "plus_minus_conditional_mean_coordinate"] > 0
        assert abs(row["directional_responses"]["y1"]["wrong_axis_plus_minus"][
            "plus_minus_conditional_mean_coordinate"]) < 1e-6
        assert row["merged_grid_hw"][0] != row["merged_grid_hw"][1]


def test_readout_uses_full_vocabulary_mass_and_distribution_direction() -> None:
    def logits(center: int, non_coordinate: float = 0.0) -> torch.Tensor:
        values = torch.full((1002,), -8.0)
        values[:1000] = -2.0 - ((torch.arange(1000) - center) / 4.0).square()
        values[1000:] = torch.tensor([non_coordinate, non_coordinate - 0.5])
        return values

    result = assay._readout(logits(400, 0.0), tuple(range(1000)))
    probabilities = torch.softmax(logits(400, 0.0)[:1000], dim=0)
    expected_mean = float((probabilities * (torch.arange(1000) / 1000)).sum())
    expected_mass = float(torch.exp(torch.logsumexp(logits(400, 0.0)[:1000], 0)
                                     - torch.logsumexp(logits(400, 0.0), 0)))
    assert result["full_vocabulary_logsumexp"] == pytest.approx(float(torch.logsumexp(logits(400, 0.0), 0)))
    assert result["coordinate_family_mass"] == pytest.approx(expected_mass)
    assert result["conditional_mean_coordinate_b_over_1000"] == pytest.approx(expected_mean)
    assert result["mass_weighted_first_moment"] == pytest.approx(expected_mean * expected_mass)
    assert result["mass_weighted_first_moment"] != pytest.approx(expected_mean)

    conditions = {}
    for name, x_center, y_center, noncoord in (
        ("correct", 400, 500, 0.0), ("identity_override", 400, 500, 0.0),
        ("x_plus", 450, 500, -0.2), ("x_minus", 350, 500, 0.0),
        ("y_plus", 400, 550, 0.0), ("y_minus", 400, 450, 0.0),
    ):
        conditions[name] = {
            "x1": assay._readout(logits(x_center, noncoord), tuple(range(1000))),
            "y1": assay._readout(logits(y_center), tuple(range(1000))),
        }
    response = assay._directional_responses(conditions)
    assert response["x1"]["plus_minus"]["plus_minus_conditional_mean_coordinate"] > 0
    assert response["x1"]["wrong_axis_plus_minus"]["coordinate_distribution_wasserstein1"] == pytest.approx(0)
    assert response["x1"]["plus_minus"]["plus_minus_coordinate_family_mass"] != pytest.approx(0)
    assert response["y1"]["plus_minus"]["coordinate_distribution_wasserstein1"] > 0


@pytest.mark.parametrize("kind", ("early16", "late16"))
def test_caller_restores_override_after_forward_exception(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assay, "prepare_replay", _fake_prepare_replay)
    qwen = _qwen(kind, fail_on_forward=2)
    codebook = qwen.model.coordinate_codebook
    original_inject = getattr(codebook, "inject_early", None)
    original_addresses = coordinate_codebook._normalized_addresses
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        assay.run_checkpoint_semantic_assay(qwen, checkpoint=kind, prepared_cases=_prepared_cases())
    if kind == "early16":
        assert codebook._edge_coordinates_override is None
        assert codebook.inject_early.__func__ is original_inject.__func__
    else:
        assert coordinate_codebook._normalized_addresses is original_addresses


def test_late_checked_caller_rejects_swapped_non_square_grid_and_restores() -> None:
    original = coordinate_codebook._normalized_addresses
    with pytest.raises(AssertionError, match="differs from frozen"):
        with assay._checked_late_forward(expected_merged_hw=(2, 3), axis=0, sign=1):
            coordinate_codebook._normalized_addresses(3, 2, device=torch.device("cpu"))
    assert coordinate_codebook._normalized_addresses is original


@pytest.mark.parametrize("identity_field", ("executed_prompt_token_ids", "executed_media_sha256"))
def test_mutated_prompt_or_media_holds_before_teacher_forward(
    identity_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assay, "prepare_replay", _fake_prepare_replay)
    qwen = _qwen("early16")
    cases = _prepared_cases()
    cases[2]["prepared"][identity_field][0] = -1 if identity_field.endswith("token_ids") else "mutated"
    with pytest.raises(ValueError, match="executed prompt|executed media"):
        assay.run_checkpoint_semantic_assay(qwen, checkpoint="early16", prepared_cases=cases)
    assert qwen.model.forward_count == 0


def test_native_caller_binds_normalized_gt_prompt_media_and_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases, input_rows = _native_caller_fixture()
    normalized = {}
    for case, input_row in zip(cases, input_rows, strict=True):
        normalized[case["image_id"]] = {
            "row_id": case["row_id"],
            "cohort": case["cohort"],
            "image_plan": {
                "backend_prompt_token_count": case["prompt_token_count"],
                "image_content_sha256": case["media_sha256"][0],
                "observed_image_grid_thw": case["grid_thw"],
                "merged_visual_tokens": case["merged_token_count"],
            },
        }
    rows_by_id = {row["_admission"]["row_id"]: row for row in input_rows}

    def ground_truth(row: Any) -> list[dict[str, Any]]:
        source = rows_by_id[row["row_id"]]["objects"][0]
        return [{
            "owner_id": str(source["coco_ann_id"]),
            "description": str(source["desc"]),
            "coord_bins": assay._ground_truth_signature(rows_by_id[row["row_id"]])[0]["coord_bins"],
            "bbox": [0.0, 0.0, 1.0, 1.0],
        }]

    monkeypatch.setattr(assay.evaluation, "_config", lambda _admission, _dataset: {"native": True})
    monkeypatch.setattr(
        assay.evaluation,
        "_normalize_case",
        lambda _qwen, row, _config, _dataset: normalized[row["image_id"]],
    )
    monkeypatch.setattr(assay.evaluation, "_gt", ground_truth)
    calls: list[tuple[str, str, bool]] = []

    def build_requests(_qwen: Any, config: Any, rows: Any) -> tuple[list[Any], list[Any]]:
        row = rows[0]
        assert config == {"native": True}
        calls.append((row["row_id"], "build", True))
        return [SimpleNamespace(request_id=row["row_id"])], []

    def prepare(processor: Any, requests: Any, *, device: str, record_media_identity: bool) -> Any:
        assert processor == "native-processor"
        assert device == "cpu" and record_media_identity is True
        request = requests[0]
        case = next(item for item in cases if item["row_id"] == request.request_id)
        calls.append((request.request_id, "prepare", record_media_identity))
        prepared = case["prepared"]
        return SimpleNamespace(
            request_ids=(request.request_id,),
            prompt_token_ids=(tuple(prepared["executed_prompt_token_ids"]),),
            media_sha256=tuple(prepared["executed_media_sha256"]),
            image_grids=(tuple(case["grid_thw"]),),
            inputs=prepared["native_inputs"],
        )

    monkeypatch.setattr(assay, "build_bound_native_requests", build_requests)
    monkeypatch.setattr(assay, "prepare_native_inputs", prepare)
    qwen = _qwen("early16")
    qwen.processor = "native-processor"
    prepared, identities = assay._prepare_native_cases(
        qwen, plan_cases=cases, admission={}, dataset=Path("/frozen/train-1024.coord.jsonl"), input_rows=input_rows
    )
    assert len(prepared) == len(identities) == 4
    assert len(calls) == 8
    for case, identity in zip(prepared, identities, strict=True):
        assert list(case["prepared"]["executed_prompt_token_ids"]) == identity["prompt_token_ids"]
        assert list(case["prepared"]["executed_media_sha256"]) == identity["media_sha256"]
        assert identity["grid_thw"] == case["grid_thw"]
        assert identity["ground_truth"][0]["coord_bins"][:2] == case["first_gt_coord_bins"]


def test_real_frozen_native_caller_binds_all_four_cases_without_model_load() -> None:
    from src.config.loader import load_train_config
    from src.qwen import load_qwen_components

    repo = Path(__file__).resolve().parents[3]
    config = load_train_config(
        repo / "configs/research/coordinate_codebook_alignment/qualification-single.yaml"
    ).config
    frozen = assay._frozen_inputs(assay.FROZEN_PLAN_PATH, assay.DEFAULT_ADMISSION_PATH)
    expected_model = Path(frozen["admission"]["source_config"]["model"]["base_model"]).resolve()
    assert Path(config.model.base_model).resolve() == expected_model
    qwen = load_qwen_components(config, load_model=False)
    assert qwen.model is None
    prepared_cases, identities = assay._prepare_native_cases(
        qwen,
        plan_cases=frozen["plan"]["cases"],
        admission=frozen["admission"],
        dataset=frozen["dataset"],
        input_rows=frozen["rows"],
    )
    assert qwen.model is None
    assert tuple(int(case["image_id"]) for case in prepared_cases) == assay.CASE_IMAGE_IDS
    assert len(identities) == 4
    for case, identity in zip(prepared_cases, identities, strict=True):
        assert identity["prompt_token_count"] == case["prompt_token_count"]
        assert identity["prompt_token_ids_sha256"] == case["prompt_token_ids_sha256"]
        assert identity["media_sha256"] == case["media_sha256"]
        assert identity["grid_thw"] == case["grid_thw"]
        assert identity["ground_truth"][0]["description"] == case["first_description"]
        assert identity["ground_truth"][0]["coord_bins"][:2] == case["first_gt_coord_bins"]
        assert identity["image_plan_content_sha256"] != identity["media_sha256"][0]


def _run_orchestration_fixture(
    tmp_path: Path, *, fail_early: bool = False
) -> tuple[Any, list[dict[str, Any]], list[str], Any]:
    roots = {}
    for label in ("early16", "late16"):
        root = tmp_path / label / "step-984"
        root.mkdir(parents=True)
        roots[label] = root
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    cases = _prepared_cases()
    plan = {
        "checkpoint_bindings": {
            label: {"checkpoint_root": str(roots[label]), "payload_bindings": []}
            for label in ("early16", "late16")
        },
        "coordinate_token_ids": list(range(1000)),
        "cases": cases,
    }
    frozen = {
        "plan": plan,
        "admission": {"source_config": {"model": {"base_model": str(base_model)}}},
        "dataset": Path("/frozen/train-1024.coord.jsonl"),
        "rows": [],
        "plan_binding": {"path": "/frozen/semantic-assay-plan.json", "sha256": "plan", "size_bytes": 1},
        "admission_binding": {"path": "/frozen/admission.json", "sha256": "admission", "size_bytes": 1},
        "input_bindings": [],
        "dataset_binding": {"path": "/frozen/train-1024.coord.jsonl", "sha256": "dataset", "size_bytes": 1},
        "checkpoint_payload_bindings": {"early16": [], "late16": []},
    }
    loaded: list[str] = []

    def load(checkpoint: str, _device: str, _admission: Any) -> tuple[Any, dict[str, Any]]:
        label = Path(checkpoint).parent.name
        loaded.append(label)
        qwen = _qwen(label, fail_on_forward=2 if fail_early and label == "early16" else None)
        return qwen, {
            "checkpoint_root": checkpoint,
            "payload_bindings": [],
            "launch": {
                "backend": "hf",
                "model_path": str(base_model),
                "model_dtype": "bf16",
                "backend_options": {
                    "hf": {
                        "attn_implementation": "flash_attention_2",
                        "patch_embed_linearization": "enabled",
                        "adapter_runtime": "live_promoted",
                        "coordinate_codebook_path": f"{checkpoint}/coordinate_codebook",
                    }
                },
                "adapter": {"type": "dora", "name": "default", "path": f"{checkpoint}/adapter"},
                "embedding_delta": {"path": f"{checkpoint}/special_token_embeddings", "source_gate_root": None},
            },
        }

    return (frozen, cases, loaded, load)


def test_cli_loads_checkpoints_sequentially_enforces_48_and_persists_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, cases, loaded, load = _run_orchestration_fixture(tmp_path)
    monkeypatch.setattr(assay, "_frozen_inputs", lambda *_args: frozen)
    identities = [{
        "native_request_config_sha256": assay._json_digest({"native": True}),
        "pixel_values_sha256": assay._pixel_tensor_sha256(case["prepared"]["native_inputs"]),
        "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
        "media_sha256": case["media_sha256"],
    } for case in cases]
    monkeypatch.setattr(assay, "_prepare_native_cases", lambda *_args, **_kwargs: (cases, identities))
    monkeypatch.setattr(assay.evaluation, "_load_runtime", load)
    monkeypatch.setattr(assay.evaluation, "_config", lambda *_args: {"native": True})
    monkeypatch.setattr(assay, "prepare_replay", _fake_prepare_replay)
    output = tmp_path / "semantic-assay.json"
    result = assay.run(Path("plan.json"), Path("admission.json"), output, device="cpu")
    persisted = json.loads(output.read_text())
    assert loaded == ["early16", "late16"]
    assert result["status"] == persisted["status"] == "complete"
    assert result["teacher_forwards"] == persisted["teacher_forwards"] == 48
    assert result["checkpoint_teacher_forwards"] == {"early16": 24, "late16": 24}
    with pytest.raises(FileExistsError, match="refusing to replace"):
        assay.run(Path("plan.json"), Path("admission.json"), output, device="cpu")


def test_cli_persists_hold_and_stops_admitting_after_checkpoint_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, cases, loaded, load = _run_orchestration_fixture(tmp_path, fail_early=True)
    monkeypatch.setattr(assay, "_frozen_inputs", lambda *_args: frozen)
    identities = [{
        "native_request_config_sha256": assay._json_digest({"native": True}),
        "pixel_values_sha256": assay._pixel_tensor_sha256(case["prepared"]["native_inputs"]),
        "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
        "media_sha256": case["media_sha256"],
    } for case in cases]
    monkeypatch.setattr(assay, "_prepare_native_cases", lambda *_args, **_kwargs: (cases, identities))
    monkeypatch.setattr(assay.evaluation, "_load_runtime", load)
    monkeypatch.setattr(assay.evaluation, "_config", lambda *_args: {"native": True})
    monkeypatch.setattr(assay, "prepare_replay", _fake_prepare_replay)
    output = tmp_path / "semantic-assay-hold.json"
    with pytest.raises(RuntimeError, match="persisted HOLD"):
        assay.run(Path("plan.json"), Path("admission.json"), output, device="cpu")
    persisted = json.loads(output.read_text())
    assert loaded == ["early16"]
    assert persisted["status"] == "HOLD"
    assert persisted["teacher_forwards"] == 2
    assert persisted["checkpoint_teacher_forwards"] == {"early16": 2}


def test_cli_holds_before_late_forwards_when_common_runtime_config_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, cases, loaded, base_load = _run_orchestration_fixture(tmp_path)

    def load(checkpoint: str, device: str, admission: Any) -> tuple[Any, dict[str, Any]]:
        qwen, identity = base_load(checkpoint, device, admission)
        if Path(checkpoint).parent.name == "late16":
            identity["launch"]["backend_options"]["hf"]["adapter_runtime"] = "mutated"
        return qwen, identity

    identities = [{
        "native_request_config_sha256": assay._json_digest({"native": True}),
        "pixel_values_sha256": assay._pixel_tensor_sha256(case["prepared"]["native_inputs"]),
        "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
        "media_sha256": case["media_sha256"],
    } for case in cases]
    monkeypatch.setattr(assay, "_frozen_inputs", lambda *_args: frozen)
    monkeypatch.setattr(assay, "_prepare_native_cases", lambda *_args, **_kwargs: (cases, identities))
    monkeypatch.setattr(assay.evaluation, "_load_runtime", load)
    monkeypatch.setattr(assay.evaluation, "_config", lambda *_args: {"native": True})
    monkeypatch.setattr(assay, "prepare_replay", _fake_prepare_replay)
    output = tmp_path / "semantic-assay-config-hold.json"
    with pytest.raises(RuntimeError, match="persisted HOLD"):
        assay.run(Path("plan.json"), Path("admission.json"), output, device="cpu")
    persisted = json.loads(output.read_text())
    assert loaded == ["early16", "late16"]
    assert persisted["status"] == "HOLD"
    assert persisted["teacher_forwards"] == 24
    assert persisted["checkpoint_teacher_forwards"] == {"early16": 24, "late16": 0}
