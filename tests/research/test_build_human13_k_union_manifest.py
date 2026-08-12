from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research import build_human13_k_union_manifest as manifest_builder
from src.common.errors import ArtifactContractError
from src.rollout_calibration.state_bank import ImageIdentity


EXPECTED_IMAGE_IDENTITIES = (
    (1584, "06b9d29a50b896f1bec14a267a57016723e54e205a1d1a40088237d95ce91206"),
    (2299, "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"),
    (2685, "84514a8aed88aba07163aa5b1be5c6e0ee75da4351496cad062d1e355a261409"),
    (4134, "60dfd1369e0efa83dfa6c7d4035f4d9d66ca6ba9a0ec6b760349d0a0e30d7b34"),
    (5001, "faaecb19a8b681495f02e18493b8ae01c96d022767f25ded19f5cbec9d95cf31"),
    (6040, "585c27309e9130400849315b4bf49d99717f700507239f2bd4dc6fece3cbe894"),
    (7511, "2843c07959515a93d2791183c998462d76b34100185e57a258d8949f112296e7"),
    (10707, "eec1e22dc3ed6ff70d35dd771e05a20024264fdecaf5a187fdad616e6d20e5d6"),
    (13348, "14f222b4cbf6d90e60eb90eb72aebc0f83cebaf150b4c5999f107bb11294fc36"),
    (13923, "c5c32b9999259b6041e92815693b975f8ee2291b29a6577d983685b6d486796f"),
    (14038, "055f28bbd181590b7a7c4844bf488d7f1be752d39916e07034c279f5f387acd2"),
    (14439, "d09e1ef4ec3bcbfbe3e16a2f9b3ff92a743e4dc061eaca4add29f59787567f73"),
    (16228, "5aade4c6e9dbdbf3bd64072e813bae02975a342b84c312efc91983ac63300d49"),
)


def test_default_binding_is_exactly_hash_and_unit_bound() -> None:
    binding = manifest_builder.default_binding()

    assert binding.unit_id == "2026-08-12-human13-k-union-to-greedy-overfit-screen"
    assert binding.purpose == "overfit_only"
    assert binding.panel.panel_sha256 == (
        "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"
    )
    assert (
        tuple((image.image_id, image.image_sha256) for image in binding.panel.images)
        == EXPECTED_IMAGE_IDENTITIES
    )


def test_binding_exposes_exact_source_prompt_tokenizer_and_matcher_identity() -> None:
    binding = manifest_builder.default_binding()

    assert binding.source.checkpoint_path.endswith(
        "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444"
    )
    assert binding.source.adapter_sha256 == (
        "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da"
    )
    assert binding.source.special_embedding_sha256 == (
        "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2"
    )
    assert binding.surface.prompt_policy_fingerprint == (
        "0b4fa411f289ccc29e3f6b59c65d32689ac04e6a014891cbe2dc2d976de634c1"
    )
    assert binding.surface.tokenizer_sha256 == (
        "ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8"
    )
    assert binding.surface.wrapper == "object_box_closed"
    assert binding.surface.parser == "compact_object_box_closed_only"
    assert binding.matcher.algorithm == "cardinality_first_max_total_iou"
    assert binding.matcher.same_category is True
    assert binding.matcher.owner_iou_threshold == 0.50
    assert binding.matcher.duplicate_iou_threshold == 0.95
    assert binding.matcher.duplicate_comparison == "strictly_greater"


def test_wrong_panel_hash_is_rejected_before_manifest_construction() -> None:
    binding = manifest_builder.default_binding()
    wrong = replace(
        binding,
        panel=replace(binding.panel, panel_sha256="0" * 64),
    )

    with pytest.raises(ValueError, match="panel SHA-256"):
        manifest_builder.build_manifest(
            binding=wrong, images=(), require_full_panel=False
        )


def test_generic_state_bank_still_rejects_the_blind_panel(tmp_path: Path) -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        ImageIdentity(
            image_id=1584,
            path=tmp_path / "blind.jpg",
            width=1,
            height=1,
            content_sha256="1" * 64,
        )

    assert exc_info.value.code == "state_bank.blind_cohort"


def test_canonical_write_is_byte_stable_and_has_one_digest(tmp_path: Path) -> None:
    built = manifest_builder.build_manifest(
        binding=manifest_builder.default_binding(),
        images=(),
        require_full_panel=False,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first_digest = manifest_builder.canonical_write(built, first_path)
    second_digest = manifest_builder.canonical_write(built, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")
    assert first_digest == second_digest
    assert json.loads(first_path.read_text(encoding="utf-8"))["schema_version"] == (
        "human13_k_union_manifest.v1"
    )
    assert (tmp_path / "first.json.sha256").read_text(encoding="ascii") == (
        f"{first_digest}  first.json\n"
    )
    assert manifest_builder.load_manifest(first_path, require_full_panel=False) == built


def test_dry_run_summary_declares_zero_model_and_optimizer_actions() -> None:
    built = manifest_builder.build_manifest(
        binding=manifest_builder.default_binding(),
        images=(),
        require_full_panel=False,
    )

    summary = manifest_builder.dry_run_summary(built)

    assert summary["unit_id"] == "2026-08-12-human13-k-union-to-greedy-overfit-screen"
    assert summary["purpose"] == "overfit_only"
    assert summary["panel_sha256"] == (
        "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"
    )
    assert summary["declared_image_count"] == 13
    assert summary["actions"] == {
        "model_load": 0,
        "decode": 0,
        "forward": 0,
        "backward": 0,
        "optimizer_mutation": 0,
        "checkpoint_write": 0,
        "gpu_allocation": 0,
    }


def _request(mode: str, *, seed: int | None = None) -> manifest_builder.RequestIdentity:
    if mode == "source_greedy":
        return manifest_builder.RequestIdentity(
            backend="hf",
            backend_version="4.57.1",
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


def _empty_sample(seed: int) -> manifest_builder.TrajectoryInput:
    return manifest_builder.TrajectoryInput(
        trajectory_id=f"k-{seed}",
        request=_request("k_sampled", seed=seed),
        token_ids=(900, 999),
        terminal_token_index=1,
        stop_reason="im_end",
        parser_status="complete",
        rows=(),
    )


def _dense_fixture() -> manifest_builder.ImageInput:
    source = manifest_builder.TrajectoryInput(
        trajectory_id="source",
        request=_request("source_greedy"),
        token_ids=(
            100,
            11,
            12,
            13,
            14,
            21,
            22,
            23,
            24,
            31,
            32,
            33,
            34,
            41,
            42,
            43,
            44,
            999,
        ),
        terminal_token_index=17,
        stop_reason="im_end",
        parser_status="complete",
        rows=(
            manifest_builder.PredictionRowInput(
                "source-a", 0, "person", (0.0, 0.0, 10.0, 10.0), 1, 5, 3
            ),
            manifest_builder.PredictionRowInput(
                "source-dense-duplicate",
                1,
                "person",
                (0.1, 0.1, 10.1, 10.1),
                5,
                9,
                7,
            ),
            manifest_builder.PredictionRowInput(
                "source-unmatched",
                2,
                "person",
                (30.0, 30.0, 40.0, 40.0),
                9,
                13,
                11,
            ),
            manifest_builder.PredictionRowInput(
                "source-invalid",
                3,
                "person",
                (50.0, 50.0, 60.0, 60.0),
                13,
                17,
                15,
                geometry_valid=False,
            ),
        ),
    )
    first_k = manifest_builder.TrajectoryInput(
        trajectory_id="k-21001",
        request=_request("k_sampled", seed=21001),
        token_ids=(200, 51, 52, 53, 54, 61, 62, 63, 64, 999),
        terminal_token_index=9,
        stop_reason="im_end",
        parser_status="complete",
        rows=(
            manifest_builder.PredictionRowInput(
                "k-b", 0, "person", (0.1, 0.1, 10.1, 10.1), 1, 5, 3
            ),
            manifest_builder.PredictionRowInput(
                "k-b-duplicate",
                1,
                "person",
                (0.11, 0.11, 10.11, 10.11),
                5,
                9,
                7,
            ),
        ),
    )
    second_k = manifest_builder.TrajectoryInput(
        trajectory_id="k-21002",
        request=_request("k_sampled", seed=21002),
        token_ids=(200, 71, 72, 73, 74, 999),
        terminal_token_index=5,
        stop_reason="im_end",
        parser_status="complete",
        rows=(
            manifest_builder.PredictionRowInput(
                "k-b-lower-quality",
                0,
                "person",
                (0.2, 0.2, 10.0, 10.0),
                1,
                5,
                3,
            ),
        ),
    )
    samples = [first_k, second_k]
    samples.extend(_empty_sample(seed) for seed in range(21003, 21017))
    return manifest_builder.ImageInput(
        image_id=2299,
        owners=(
            manifest_builder.OwnerInput(
                "gt:2299:0", "person", (0.0, 0.0, 10.0, 10.0), 0
            ),
            manifest_builder.OwnerInput(
                "gt:2299:1", "person", (0.1, 0.1, 10.1, 10.1), 1
            ),
            manifest_builder.OwnerInput(
                "gt:2299:2", "person", (70.0, 70.0, 80.0, 80.0), 2
            ),
        ),
        source=source,
        sampled=tuple(samples),
    )


def _build_dense_fixture() -> manifest_builder.Human13KUnionManifest:
    return manifest_builder.build_manifest(
        binding=manifest_builder.default_binding(),
        images=(_dense_fixture(),),
        require_full_panel=False,
    )


def test_each_materialized_image_requires_all_exact_k_seeds() -> None:
    image = _dense_fixture()

    with pytest.raises(ValueError, match="seeds 21001..21016"):
        manifest_builder.build_manifest(
            binding=manifest_builder.default_binding(),
            images=(replace(image, sampled=image.sampled[:-1]),),
            require_full_panel=False,
        )


def test_duplicate_classification_precedes_dense_owner_matching() -> None:
    image = _build_dense_fixture().images[0]

    source = next(row for row in image.trajectories if row.trajectory_id == "source")
    assert source.duplicate_row_ids == ("source-dense-duplicate",)
    assert "source-dense-duplicate" not in source.matched_row_ids
    assert image.g_owner_ids == ("gt:2299:0",)
    assert image.h_owner_ids == ("gt:2299:1",)
    assert image.m_owner_ids == ("gt:2299:2",)
    positive_sets = (
        set(source.matched_row_ids),
        set(image.replay_row_ids),
        set(image.target_row_ids),
        set(image.candidate_row_ids),
    )
    assert all("source-dense-duplicate" not in positive for positive in positive_sets)


def test_clean_prefix_deletes_only_duplicate_spans_without_retokenizing() -> None:
    image = _build_dense_fixture().images[0]
    source = next(row for row in image.trajectories if row.trajectory_id == "source")

    assert source.prefix.raw_token_ids == (
        100,
        11,
        12,
        13,
        14,
        21,
        22,
        23,
        24,
        31,
        32,
        33,
        34,
        41,
        42,
        43,
        44,
    )
    assert source.prefix.clean_token_ids == (
        100,
        11,
        12,
        13,
        14,
        31,
        32,
        33,
        34,
        41,
        42,
        43,
        44,
    )
    assert source.prefix.removed_row_ids == ("source-dense-duplicate",)


def test_duplicate_events_preserve_every_original_raw_decision_state() -> None:
    image = _build_dense_fixture().images[0]
    events = {event.duplicate_row_id: event for event in image.duplicate_events}

    assert set(events) == {"source-dense-duplicate", "k-b-duplicate"}
    source_event = events["source-dense-duplicate"]
    assert source_event.retained_row_id == "source-a"
    assert source_event.decision_prefix_token_ids == (100, 11, 12, 13, 14, 21, 22)
    assert source_event.target_token_id == 23
    assert source_event.target_token_index == 7
    k_event = events["k-b-duplicate"]
    assert k_event.decision_prefix_token_ids == (200, 51, 52, 53, 54, 61, 62)
    assert k_event.target_token_id == 63


def test_target_selection_is_max_iou_then_seed_then_row_index() -> None:
    image = _build_dense_fixture().images[0]

    assert len(image.selected_rows) == 1
    selected = image.selected_rows[0]
    assert selected.owner_id == "gt:2299:1"
    assert selected.row_id == "k-b"
    assert selected.seed == 21001
    assert selected.token_ids == (51, 52, 53, 54)
    assert selected.target_token_mask == (True, True, True, True)


def test_unmatched_invalid_duplicate_and_terminal_sites_are_masked() -> None:
    image = _build_dense_fixture().images[0]
    source = next(row for row in image.trajectories if row.trajectory_id == "source")

    assert source.replay_token_mask == (
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    assert source.duplicate_target_mask == (
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    assert source.replay_token_mask[source.terminal_token_index] is False
    assert source.duplicate_target_mask[source.terminal_token_index] is False


def test_global_denominators_count_physical_owners_images_and_all_events() -> None:
    denominators = _build_dense_fixture().denominators

    assert denominators.panel_image_count == 13
    assert denominators.target_image_count == 1
    assert denominators.target_owner_count == 1
    assert denominators.replay_image_count == 1
    assert denominators.replay_owner_count == 1
    assert denominators.duplicate_image_count == 1
    assert denominators.duplicate_event_count == 2


def test_zero_eligible_denominators_remain_explicit_zero_counts() -> None:
    image = _dense_fixture()
    no_owner_image = replace(image, owners=())

    built = manifest_builder.build_manifest(
        binding=manifest_builder.default_binding(),
        images=(no_owner_image,),
        require_full_panel=False,
    )

    assert built.denominators.target_owner_count == 0
    assert built.denominators.target_image_count == 0
    assert built.denominators.replay_owner_count == 0
    assert built.denominators.replay_image_count == 0
    assert built.denominators.duplicate_event_count == 2


def test_nonempty_canonical_manifest_round_trips_all_frozen_records(
    tmp_path: Path,
) -> None:
    built = _build_dense_fixture()
    path = tmp_path / "dense.json"

    digest = manifest_builder.canonical_write(built, path)
    loaded = manifest_builder.load_manifest(path, require_full_panel=False)

    assert loaded == built
    assert (tmp_path / "dense.json.sha256").read_text(encoding="ascii") == (
        f"{digest}  dense.json\n"
    )


def test_load_rejects_altered_arm_identity_even_with_matching_digest(
    tmp_path: Path,
) -> None:
    built = _build_dense_fixture()
    path = tmp_path / "altered.json"
    manifest_builder.canonical_write(built, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["arms"][0]["target_scope"] = "H"
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "altered.json.sha256").write_text(
        f"{digest}  altered.json\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="arm identities"):
        manifest_builder.load_manifest(path, require_full_panel=False)


def _empty_source(image_id: int) -> manifest_builder.TrajectoryInput:
    return manifest_builder.TrajectoryInput(
        trajectory_id=f"source-{image_id}",
        request=_request("source_greedy"),
        token_ids=(900, 999),
        terminal_token_index=1,
        stop_reason="im_end",
        parser_status="complete",
        rows=(),
    )


def test_full_panel_rejects_substituted_bytes_before_count_admission(
    tmp_path: Path,
) -> None:
    images = []
    for image_index, (image_id, _) in enumerate(EXPECTED_IMAGE_IDENTITIES):
        owner_count = 32 if image_index == 12 else 30
        owners = tuple(
            manifest_builder.OwnerInput(
                owner_id=f"gt:{image_id}:{owner_index}",
                category="person",
                bbox=(
                    float(owner_index),
                    0.0,
                    float(owner_index + 1),
                    1.0,
                ),
                source_object_index=owner_index,
            )
            for owner_index in range(owner_count)
        )
        images.append(
            manifest_builder.ImageInput(
                image_id=image_id,
                owners=owners,
                source=_empty_source(image_id),
                sampled=tuple(_empty_sample(seed) for seed in range(21001, 21017)),
            )
        )
    substituted_panel = tmp_path / "substituted-panel.jsonl"
    substituted_panel.write_text(
        "".join(
            json.dumps({"image_id": image_id, "objects": []}, separators=(",", ":"))
            + "\n"
            for image_id, _ in EXPECTED_IMAGE_IDENTITIES
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="panel SHA-256"):
        manifest_builder.build_manifest(
            binding=manifest_builder.default_binding(),
            images=tuple(images),
            panel_path=substituted_panel,
            require_full_panel=True,
        )


def test_load_rejects_raw_row_provenance_mutation_with_matching_digest(
    tmp_path: Path,
) -> None:
    built = _build_dense_fixture()
    path = tmp_path / "row-mutated.json"
    manifest_builder.canonical_write(built, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    source = document["images"][0]["trajectories"][0]
    source["rows"][1]["bbox"] = [70.0, 70.0, 80.0, 80.0]
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "row-mutated.json.sha256").write_text(
        f"{digest}  row-mutated.json\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="meaning-bearing projections"):
        manifest_builder.load_manifest(path, require_full_panel=False)


def test_canonical_write_rejects_invalid_direct_manifest(tmp_path: Path) -> None:
    built = _build_dense_fixture()
    invalid_image = replace(built.images[0], g_owner_ids=())
    invalid = replace(built, images=(invalid_image,))

    with pytest.raises(ValueError, match="meaning-bearing projections"):
        manifest_builder.canonical_write(invalid, tmp_path / "invalid.json")

    assert not (tmp_path / "invalid.json").exists()


def _trajectory_with_owner_rows(
    trajectory_id: str,
    request: manifest_builder.RequestIdentity,
    owner_indices: tuple[int, ...],
) -> manifest_builder.TrajectoryInput:
    token_ids = [100]
    rows = []
    for row_index, owner_index in enumerate(owner_indices):
        token_start = len(token_ids)
        token_ids.extend((1000 + owner_index * 4 + offset for offset in range(4)))
        rows.append(
            manifest_builder.PredictionRowInput(
                row_id=f"{trajectory_id}-row-{owner_index}",
                row_index=row_index,
                category="person",
                bbox=(
                    float(owner_index * 20),
                    0.0,
                    float(owner_index * 20 + 10),
                    10.0,
                ),
                token_start=token_start,
                token_end=token_start + 4,
                final_coordinate_token_index=token_start + 2,
            )
        )
    terminal_token_index = len(token_ids)
    token_ids.append(999)
    return manifest_builder.TrajectoryInput(
        trajectory_id=trajectory_id,
        request=request,
        token_ids=tuple(token_ids),
        terminal_token_index=terminal_token_index,
        stop_reason="im_end",
        parser_status="complete",
        rows=tuple(rows),
    )


def test_owner_and_residual_orders_follow_source_object_index_not_owner_id() -> None:
    owners = tuple(
        manifest_builder.OwnerInput(
            owner_id=f"gt:2299:{owner_index}",
            category="person",
            bbox=(
                float(owner_index * 20),
                0.0,
                float(owner_index * 20 + 10),
                10.0,
            ),
            source_object_index=owner_index,
        )
        for owner_index in range(13)
    )
    source = _trajectory_with_owner_rows(
        "source-ordered", _request("source_greedy"), (2, 10)
    )
    first_k = _trajectory_with_owner_rows(
        "k-21001-ordered", _request("k_sampled", seed=21001), (3, 11)
    )
    image = manifest_builder.ImageInput(
        image_id=2299,
        owners=owners,
        source=source,
        sampled=(first_k,) + tuple(_empty_sample(seed) for seed in range(21002, 21017)),
    )

    record = manifest_builder.build_manifest(
        binding=manifest_builder.default_binding(),
        images=(image,),
        require_full_panel=False,
    ).images[0]

    assert record.g_owner_ids == ("gt:2299:2", "gt:2299:10")
    assert record.h_owner_ids == ("gt:2299:3", "gt:2299:11")
    assert record.m_owner_ids == (
        "gt:2299:0",
        "gt:2299:1",
        "gt:2299:4",
        "gt:2299:5",
        "gt:2299:6",
        "gt:2299:7",
        "gt:2299:8",
        "gt:2299:9",
        "gt:2299:12",
    )
    assert tuple(row.owner_id for row in record.selected_rows) == (
        "gt:2299:3",
        "gt:2299:11",
    )
    assert record.target_row_ids == (
        "k-21001-ordered-row-3",
        "k-21001-ordered-row-11",
    )
    assert record.candidate_row_ids == (
        "k-21001-ordered-row-3",
        "k-21001-ordered-row-11",
    )


def test_matching_normalizes_coco_category_case_and_whitespace() -> None:
    owner = manifest_builder.OwnerInput(
        owner_id="gt:2299:0",
        category=" Wine   Glass ",
        bbox=(0.0, 0.0, 10.0, 10.0),
        source_object_index=0,
    )
    source = manifest_builder.TrajectoryInput(
        trajectory_id="source-category-normalized",
        request=_request("source_greedy"),
        token_ids=(100, 11, 12, 13, 14, 999),
        terminal_token_index=5,
        stop_reason="im_end",
        parser_status="complete",
        rows=(
            manifest_builder.PredictionRowInput(
                row_id="source-wine-glass",
                row_index=0,
                category="wine GLASS",
                bbox=(0.0, 0.0, 10.0, 10.0),
                token_start=1,
                token_end=5,
                final_coordinate_token_index=3,
            ),
        ),
    )
    image = manifest_builder.ImageInput(
        image_id=2299,
        owners=(owner,),
        source=source,
        sampled=tuple(_empty_sample(seed) for seed in range(21001, 21017)),
    )

    record = manifest_builder.build_manifest(
        binding=manifest_builder.default_binding(),
        images=(image,),
        require_full_panel=False,
    ).images[0]

    assert record.g_owner_ids == ("gt:2299:0",)
    assert record.m_owner_ids == ()
    assert record.replay_row_ids == ("source-wine-glass",)
