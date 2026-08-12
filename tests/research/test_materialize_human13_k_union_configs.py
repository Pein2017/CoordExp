from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts.research import build_human13_k_union_manifest as manifest_builder
import scripts.research.materialize_human13_k_union_configs as materializer


CONFIG_ROOT = Path("configs/coordexp_swift/research/human13_k_union")
BASE_ARMS = (
    "frozen_source",
    "full_gt_capacity",
    "A0",
    "A1",
    "A3",
    "A4",
    "A7",
    "A8-prime",
)


def _write_manifest(path: Path, *, h_mid: bool = True) -> tuple[Path, str, str]:
    def request(
        mode: str, *, seed: int | None = None
    ) -> manifest_builder.RequestIdentity:
        if mode == "source_greedy":
            return manifest_builder.RequestIdentity(
                backend="hf",
                backend_version="test-hf",
                mode="source_greedy",
                n=1,
                seed=None,
                physical_batch_index=0,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_new_tokens=3084,
            )
        assert seed is not None
        return manifest_builder.RequestIdentity(
            backend="vllm",
            backend_version="test-vllm",
            mode="k_sampled",
            n=1,
            seed=seed,
            physical_batch_index=(seed - 21001) // 4,
            temperature=0.4,
            top_p=0.95,
            repetition_penalty=1.10,
            max_new_tokens=512,
        )

    images: list[manifest_builder.ImageInput] = []
    for frozen in manifest_builder.load_frozen_panel():
        if frozen.image_id == 2299:
            g_index = 8 if h_mid else 2
            g_owner = frozen.owners[g_index]
            h_owner = frozen.owners[3]
            source = manifest_builder.TrajectoryInput(
                trajectory_id="source:2299",
                request=request("source_greedy"),
                token_ids=(31, 32, 999),
                terminal_token_index=2,
                stop_reason="im_end",
                parser_status="complete",
                rows=(
                    manifest_builder.PredictionRowInput(
                        row_id="source:2299:g",
                        row_index=0,
                        category=g_owner.category,
                        bbox=g_owner.bbox,
                        token_start=0,
                        token_end=2,
                        final_coordinate_token_index=1,
                    ),
                ),
            )
        else:
            source = manifest_builder.TrajectoryInput(
                trajectory_id=f"source:{frozen.image_id}",
                request=request("source_greedy"),
                token_ids=(999,),
                terminal_token_index=0,
                stop_reason="im_end",
                parser_status="complete",
                rows=(),
            )

        sampled: list[manifest_builder.TrajectoryInput] = []
        for seed in manifest_builder.EXPECTED_K_SEEDS:
            if frozen.image_id == 2299 and seed == 21001:
                sampled.append(
                    manifest_builder.TrajectoryInput(
                        trajectory_id="sampled:2299:21001",
                        request=request("k_sampled", seed=seed),
                        token_ids=(11, 12, 21, 22, 999),
                        terminal_token_index=4,
                        stop_reason="im_end",
                        parser_status="complete",
                        rows=(
                            manifest_builder.PredictionRowInput(
                                row_id="sampled:2299:21001:prior",
                                row_index=0,
                                category=h_owner.category,
                                bbox=(0.0, 0.0, 1.0, 1.0),
                                token_start=0,
                                token_end=2,
                                final_coordinate_token_index=1,
                            ),
                            manifest_builder.PredictionRowInput(
                                row_id="sampled:2299:21001:target",
                                row_index=1,
                                category=h_owner.category,
                                bbox=h_owner.bbox,
                                token_start=2,
                                token_end=4,
                                final_coordinate_token_index=3,
                            ),
                        ),
                    )
                )
            else:
                sampled.append(
                    manifest_builder.TrajectoryInput(
                        trajectory_id=f"sampled:{frozen.image_id}:{seed}",
                        request=request("k_sampled", seed=seed),
                        token_ids=(999,),
                        terminal_token_index=0,
                        stop_reason="im_end",
                        parser_status="complete",
                        rows=(),
                    )
                )
        images.append(
            manifest_builder.ImageInput(
                image_id=frozen.image_id,
                owners=frozen.owners,
                source=source,
                sampled=tuple(sampled),
                panel_row_sha256=frozen.panel_row_sha256,
                image_sha256=frozen.image_sha256,
            )
        )

    built = manifest_builder.build_manifest(
        binding=manifest_builder.default_binding(),
        images=tuple(images),
        require_full_panel=True,
    )
    manifest_sha = manifest_builder.canonical_write(built, path)
    frozen_projection = [
        {
            "image_id": image.image_id,
            "selected_rows": [
                {
                    "owner_id": row.owner_id,
                    "row_id": row.row_id,
                    "token_ids": list(row.token_ids),
                    "target_token_mask": list(row.target_token_mask),
                }
                for row in image.selected_rows
            ],
        }
        for image in built.images
    ]
    frozen_sha = hashlib.sha256(
        json.dumps(
            frozen_projection,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return path, manifest_sha, frozen_sha


def _write_census(path: Path, *, frozen_sha: str, applicable: bool = True) -> Path:
    drift = 0.1249 if applicable else 0.7499
    required_margin = drift + 1.0e-4
    packed_margins = (0.1, 0.3)
    sites = [
        {
            "site_index": index,
            "image_id": "2299",
            "owner_id": "gt:2299:3",
            "token_offset": index,
            "target_token_id": 21 + index,
            "token_role": "coordinate",
            "packed_finite": True,
            "hf_finite": True,
            "aligned_finite": True,
            "packed_target_margin": packed_margin,
            "hf_target_margin": packed_margin + site_drift,
            "absolute_margin_drift": site_drift,
        }
        for index, (packed_margin, site_drift) in enumerate(
            zip(packed_margins, (drift, 0.02), strict=True)
        )
    ]
    payload = {
        "schema_version": "human13_k_union_no_update_census.v1",
        "trie": {
            "original_row_count": 1,
            "unique_row_count": 1,
            "exact_duplicate_count": 0,
            "images": [
                {
                    "image_id": "2299",
                    "nodes": [
                        {
                            "prefix_token_ids": [],
                            "viable_child_token_ids": [21],
                            "actual_top1_token_id": 21,
                            "actual_top1_is_viable_child": True,
                            "strongest_viable_child_token_id": 21,
                            "strongest_viable_child_margin": 1.0,
                            "top_tie_count": 1,
                        },
                        {
                            "prefix_token_ids": [21],
                            "viable_child_token_ids": [22],
                            "actual_top1_token_id": 22,
                            "actual_top1_is_viable_child": True,
                            "strongest_viable_child_token_id": 22,
                            "strongest_viable_child_margin": 1.0,
                            "top_tie_count": 1,
                        },
                    ],
                    "projected_token_ids": [21, 22],
                    "projected_owner_ids": ["gt:2299:3"],
                    "reached_native_leaf": True,
                }
            ],
        },
        "coherent_chain": {
            "site_count": len(sites),
            "sites": sites,
            "first_non_argmax_site": None,
            "minimum_strict_margin": min(packed_margins),
            "tie_site_count": 0,
            "token_role_counts": {"coordinate": len(sites)},
        },
        "aligned_surface": {
            "all_finite": True,
            "maximum_absolute_margin_drift": drift,
            "site_count": len(sites),
        },
        "a8_prime": {
            "applicable": applicable,
            "blocked": not applicable,
            "block_reason": (None if applicable else "required_margin_exceeds_0_5"),
            "required_margin": required_margin,
            "violating_site_count": (
                sum(margin < required_margin for margin in packed_margins)
                if applicable
                else 0
            ),
        },
        "frozen_targets": {
            "byte_identical": True,
            "sha256_before": frozen_sha,
            "sha256_after": frozen_sha,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_static_configs_freeze_exact_approved_arm_contracts() -> None:
    paths = sorted(CONFIG_ROOT.glob("*.yaml"))
    configs = [materializer.load_arm_config(path) for path in paths]

    assert tuple(config.arm_id for config in configs) == BASE_ARMS + ("A6",)
    assert {config.source for config in configs} == {materializer.FROZEN_SOURCE}
    assert {config.milestones for config in configs} == {(0, 1, 2, 4, 8, 16)}
    assert {config.global_max_length for config in configs} == {12_000}

    by_arm = {config.arm_id: config for config in configs}
    assert by_arm["A0"].arm_name == "A0 shared no-H background"
    assert by_arm["frozen_source"].updates is False
    assert by_arm["frozen_source"].optimizer is None
    for arm_id in set(by_arm) - {"frozen_source"}:
        config = by_arm[arm_id]
        assert config.updates is True
        assert config.trainable_surface == materializer.LANGUAGE_DORA_ONLY
        assert config.optimizer == materializer.FROZEN_ADAMW
        assert config.scheduler == materializer.FROZEN_SCHEDULER
        assert config.max_grad_norm == 1.0
    assert by_arm["A0"].coefficients == (0.0, 1.0, 1.0)
    assert by_arm["A7"].coefficients == (1.0, 0.0, 1.0)
    assert by_arm["full_gt_capacity"].coefficients == (1.0, 0.0, 0.0)
    for arm_id in ("A1", "A3", "A4", "A6", "A8-prime"):
        assert by_arm[arm_id].coefficients == (1.0, 1.0, 1.0)
    assert all(config.renormalize_active_families is False for config in configs)


def test_strict_config_rejects_unknown_or_drifted_fields(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIG_ROOT / "04_a3.yaml").read_text(encoding="utf-8"))
    raw["optimizer"]["learning_rate"] = 2.0e-5
    raw["surprise"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(materializer.MaterializationError, match="unknown fields"):
        materializer.load_arm_config(path)
    raw.pop("surprise")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="learning_rate"):
        materializer.load_arm_config(path)

    raw = yaml.safe_load((CONFIG_ROOT / "02_a0.yaml").read_text(encoding="utf-8"))
    raw["arm_name"] = "replay-only control"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="arm_name"):
        materializer.load_arm_config(path)

    raw = yaml.safe_load(
        (CONFIG_ROOT / "00_frozen_source.yaml").read_text(encoding="utf-8")
    )
    raw["updates"] = True
    raw["trainable_surface"] = yaml.safe_load(
        (CONFIG_ROOT / "02_a0.yaml").read_text(encoding="utf-8")
    )["trainable_surface"]
    raw["optimizer"] = yaml.safe_load(
        (CONFIG_ROOT / "02_a0.yaml").read_text(encoding="utf-8")
    )["optimizer"]
    raw["scheduler"] = yaml.safe_load(
        (CONFIG_ROOT / "02_a0.yaml").read_text(encoding="utf-8")
    )["scheduler"]
    raw["max_grad_norm"] = 1.0
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="update matrix"):
        materializer.load_arm_config(path)


def test_materializer_omits_a6_without_sealed_eligible_donor_and_a8_without_binding(
    tmp_path: Path,
) -> None:
    receipt = materializer.materialize_plans(
        output_root=tmp_path / "runs",
        run_id="screen-001",
        config_root=CONFIG_ROOT,
    )

    assert receipt["mode"] == "dry_run"
    assert receipt["actions"] == materializer.ZERO_MODEL_ACTIONS
    assert [plan["arm_id"] for plan in receipt["plans"]] == [
        arm for arm in BASE_ARMS if arm != "A8-prime"
    ]
    assert receipt["omitted_arms"] == [
        {"arm_id": "A6", "reason": "sealed_eligible_h_mid_donor_unavailable"},
        {"arm_id": "A8-prime", "reason": "sealed_a8_census_binding_unavailable"},
    ]


def test_materializer_binds_applicable_a6_and_a8_and_isolates_every_arm(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, frozen_sha = _write_manifest(tmp_path / "manifest.json")
    census = _write_census(tmp_path / "census.json", frozen_sha=frozen_sha)

    receipt = materializer.materialize_plans(
        output_root=tmp_path / "runs",
        run_id="screen-002",
        config_root=CONFIG_ROOT,
        census_path=census,
        manifest_path=manifest,
    )

    assert [plan["arm_id"] for plan in receipt["plans"]] == [
        *BASE_ARMS,
        "A6",
    ]
    assert receipt["omitted_arms"] == []
    roots = [plan["output_root"] for plan in receipt["plans"]]
    state_roots = [plan["optimizer_state_root"] for plan in receipt["plans"]]
    assert len(set(roots)) == len(roots)
    assert len(set(state_roots)) == len(state_roots)
    assert all(root is None for root in state_roots[:1])
    assert all(root is not None for root in state_roots[1:])
    assert len({plan["source"]["adapter_sha256"] for plan in receipt["plans"]}) == 1
    assert len({plan["fresh_state_id"] for plan in receipt["plans"][1:]}) == 8

    a6 = next(plan for plan in receipt["plans"] if plan["arm_id"] == "A6")
    assert (
        a6["a6_donor_binding"]["manifest_identity"]["manifest_sha256"] == manifest_sha
    )
    assert a6["a6_donor_binding"]["frozen_targets_sha256"] == frozen_sha
    assert a6["a6_donor_binding"]["donors"] == (
        {
            "image_id": 2299,
            "owner_id": "gt:2299:3",
            "target_row_id": "sampled:2299:21001:target",
            "donor_trajectory_id": "sampled:2299:21001",
            "donor_prefix_token_ids": (11, 12),
            "donor_prior_row_ids": ("sampled:2299:21001:prior",),
            "h_mid_eligible": True,
        },
    )
    from scripts.research import run_human13_k_union_overfit as runner

    raw_binding = a6["a6_donor_binding"]
    typed_binding = runner.Human13A6DonorBinding(
        schema_version=raw_binding["schema_version"],
        manifest_identity=runner.Human13ManifestIdentity(
            **raw_binding["manifest_identity"]
        ),
        frozen_targets_sha256=raw_binding["frozen_targets_sha256"],
        artifact_sha256=raw_binding["artifact_sha256"],
        applicable=raw_binding["applicable"],
        donors=tuple(
            runner.Human13A6DonorRecord(**donor) for donor in raw_binding["donors"]
        ),
    )
    assert typed_binding.artifact_sha256 == runner._a6_donor_artifact_sha256(
        typed_binding
    )
    a8 = next(plan for plan in receipt["plans"] if plan["arm_id"] == "A8-prime")
    assert a8["a8_census_binding"]["required_margin"] == pytest.approx(0.125)


def test_a8_binding_is_fail_closed_on_blocked_or_unbound_census(tmp_path: Path) -> None:
    manifest, _, frozen_sha = _write_manifest(tmp_path / "manifest.json")
    blocked = _write_census(
        tmp_path / "blocked.json", frozen_sha=frozen_sha, applicable=False
    )
    receipt = materializer.materialize_plans(
        output_root=tmp_path / "runs",
        run_id="screen-003",
        config_root=CONFIG_ROOT,
        census_path=blocked,
        manifest_path=manifest,
    )
    assert {item["arm_id"]: item["reason"] for item in receipt["omitted_arms"]}[
        "A8-prime"
    ] == "required_margin_exceeds_0_5"

    raw = json.loads(blocked.read_text(encoding="utf-8"))
    raw["a8_prime"]["applicable"] = True
    raw["a8_prime"]["blocked"] = False
    raw["a8_prime"]["required_margin"] = 0.1
    blocked.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="derived.*drift"):
        materializer.materialize_plans(
            output_root=tmp_path / "runs",
            run_id="screen-004",
            config_root=CONFIG_ROOT,
            census_path=blocked,
            manifest_path=manifest,
        )

    truncated = dict(raw)
    truncated.pop("coherent_chain")
    blocked.write_text(json.dumps(truncated), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="missing fields"):
        materializer.materialize_plans(
            output_root=tmp_path / "runs",
            run_id="screen-005",
            config_root=CONFIG_ROOT,
            census_path=blocked,
            manifest_path=manifest,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("site_drift", "site drift is self-asserted"),
        ("maximum_drift", "maximum drift is not derived"),
        ("truncated_site", "coherent chain is incomplete"),
        ("frozen_target", "preserve frozen target bytes"),
    ),
)
def test_a8_rejects_truncated_or_self_asserted_census_content(
    tmp_path: Path, mutation: str, error: str
) -> None:
    manifest, _, frozen_sha = _write_manifest(tmp_path / "manifest.json")
    census = _write_census(tmp_path / "census.json", frozen_sha=frozen_sha)
    raw = json.loads(census.read_text(encoding="utf-8"))
    if mutation == "site_drift":
        raw["coherent_chain"]["sites"][0]["absolute_margin_drift"] = 0.0
    elif mutation == "maximum_drift":
        raw["aligned_surface"]["maximum_absolute_margin_drift"] = 0.0
    elif mutation == "truncated_site":
        raw["coherent_chain"]["sites"].pop()
    else:
        raw["frozen_targets"]["sha256_after"] = "0" * 64
    census.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(materializer.MaterializationError, match=error):
        materializer.materialize_plans(
            output_root=tmp_path / "runs",
            run_id=f"bad-a8-{mutation}",
            config_root=CONFIG_ROOT,
            manifest_path=manifest,
            census_path=census,
        )


def test_a6_is_derived_from_manifest_and_omitted_without_h_mid(tmp_path: Path) -> None:
    manifest, _, _ = _write_manifest(tmp_path / "manifest.json", h_mid=False)
    receipt = materializer.materialize_plans(
        output_root=tmp_path / "runs",
        run_id="no-h-mid",
        config_root=CONFIG_ROOT,
        manifest_path=manifest,
    )
    assert "A6" not in {plan["arm_id"] for plan in receipt["plans"]}
    assert {item["arm_id"]: item["reason"] for item in receipt["omitted_arms"]}[
        "A6"
    ] == "sealed_eligible_h_mid_donor_unavailable"


def test_manifest_bound_materialization_still_imports_no_model_runtime(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _write_manifest(tmp_path / "manifest.json")
    code = """
import json
import sys
import tempfile
from pathlib import Path
import scripts.research.materialize_human13_k_union_configs as materializer

root = Path(tempfile.mkdtemp(prefix='human13-manifest-proof-'))
manifest = Path(sys.argv[1])
materializer.materialize_plans(
    output_root=root / 'runs',
    run_id='manifest-proof',
    config_root=Path('configs/coordexp_swift/research/human13_k_union'),
    manifest_path=manifest,
)
forbidden = sorted(
    name for name in sys.modules
    if name == 'torch'
    or name.startswith('torch.')
    or name == 'transformers'
    or name.startswith('transformers.')
    or name == 'accelerate'
    or name.startswith('accelerate.')
    or name == 'src.qwen'
    or name.startswith('src.qwen.')
    or name == 'scripts.research.run_human13_k_union_overfit'
)
print(json.dumps(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize("mutation", ("owner", "row", "trajectory"))
def test_materializer_rejects_resealed_semantic_manifest_mutation(
    tmp_path: Path, mutation: str
) -> None:
    manifest, _, _ = _write_manifest(tmp_path / "manifest.json")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    target_image = next(image for image in raw["images"] if image["selected_rows"])
    selected = target_image["selected_rows"][0]
    selected[
        {"owner": "owner_id", "row": "row_id", "trajectory": "trajectory_id"}[mutation]
    ] = "invented"
    encoded = (
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    manifest.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(f"{manifest}.sha256").write_text(
        f"{digest}  {manifest.name}\n", encoding="ascii"
    )
    with pytest.raises(
        materializer.MaterializationError, match="canonical full-panel admission"
    ):
        materializer.materialize_plans(
            output_root=tmp_path / "runs",
            run_id=f"bad-{mutation}",
            config_root=CONFIG_ROOT,
            manifest_path=manifest,
        )


def test_dry_run_cli_performs_zero_runtime_action_and_writes_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = materializer.main(
        [
            "--config-root",
            str(CONFIG_ROOT),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "cli-proof",
        ]
    )

    assert code == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["actions"] == materializer.ZERO_MODEL_ACTIONS
    assert not (tmp_path / "runs").exists()


def test_default_import_and_dry_run_do_not_import_model_runtime() -> None:
    code = """
import json
import sys
from pathlib import Path
import scripts.research.materialize_human13_k_union_configs as materializer
materializer.materialize_plans(
    output_root=Path('/tmp/human13-zero-action-proof'),
    run_id='import-proof',
    config_root=Path('configs/coordexp_swift/research/human13_k_union'),
)
forbidden = sorted(
    name for name in sys.modules
    if name == 'torch'
    or name.startswith('torch.')
    or name == 'transformers'
    or name.startswith('transformers.')
    or name == 'accelerate'
    or name.startswith('accelerate.')
    or name == 'src.qwen'
    or name.startswith('src.qwen.')
    or name == 'scripts.research.run_human13_k_union_overfit'
)
print(json.dumps(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []
