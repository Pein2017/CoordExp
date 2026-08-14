from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from PIL import Image
import pytest
import torch
from src.qwen.fa2 import Fa2VarlenPlan
from src.qwen.forward import QwenForwardInputs, QwenForwardReceipt
from src.qwen.images import QwenImageEncoding, QwenNoResizeImagePlan

from scripts.research.build_human13_k_union_manifest import (
    ArmIdentity,
    GlobalDenominatorIdentity,
    Human13KUnionManifest,
    ImageRecord,
    OwnerRecord,
    PredictionRowInput,
    PrefixRecord,
    RequestIdentity,
    TrajectoryRecord,
    default_binding,
)
from scripts.research.build_human13_on_policy_frontier import (
    FrontierCandidateAlias,
    FrontierImage,
    candidate_aliases_for_owners,
)
from scripts.research.human13_greedy_compiler import (
    EXACT_ALIAS_COUNT,
    CompilerLedger,
    SourceBoundaryInput,
    _admit_compiler_ledger_for_test,
    _build_compiler_ledger_for_test,
    _build_nested_arm_artifacts_for_test,
    _build_source_forward_runtime_for_test,
    _construct_source_forward_runtime,
    _admit_source_greedy_decode_for_test,
    _bind_packed_compiler_logits_for_test,
    _greedy_compiler_loss_for_test,
    _greedy_compiler_numerator_for_test,
    _require_source_decode,
    admit_compiler_compact_logits,
    admit_compiler_compact_logits_from_packed_plan,
    admit_source_greedy_decode,
    bind_packed_compiler_logits,
    build_compiler_ledger,
    combined_loss,
    gather_compiler_compact_logits,
    greedy_compiler_loss as public_greedy_compiler_loss,
    greedy_compiler_numerator as public_greedy_compiler_numerator,
    greedy_compiler_site_score,
    load_compiler_ledger,
    PackedCompilerLineage,
)
from scripts.research.human13_on_policy_scoring import (
    prepare_on_policy_candidate_scoring,
)
from scripts.research.human13_rp_crossover_live_packs import CompilerPackedRow


SOURCE = "a" * 64
ACQUISITION = "b" * 64
CREDIT = "c" * 64
STOP = 99
IMAGE_TOKEN_ID = 151655


def _image_encoding(image_sha256: str) -> QwenImageEncoding:
    plan = QwenNoResizeImagePlan(
        example_id="fixture-image",
        image_path=Path("/nonexistent/fixture.png"),
        width=2,
        height=2,
        patch_size=1,
        merge_size=1,
        temporal_patch_size=1,
        required_spatial_factor=1,
        raw_pixels=4,
        raw_patch_rows=4,
        expected_pixel_values_width=3,
        image_grid_thw=(1, 2, 2),
        merged_visual_tokens=4,
        max_raw_pixels=4,
        max_merged_visual_tokens=4,
        image_content_sha256=image_sha256,
        decoded_width=2,
        decoded_height=2,
    )
    return QwenImageEncoding(
        plan,
        torch.zeros((4, 3)),
        torch.tensor([[1, 2, 2]], dtype=torch.long),
    )


@dataclass(frozen=True)
class _ImagePlan:
    merge_size: int = 2


@dataclass(frozen=True)
class _ImageEncoding:
    image_grid_thw: tuple[int, int, int] = (1, 2, 2)
    merged_visual_tokens: int = 1
    plan: _ImagePlan = field(default_factory=_ImagePlan)
    pixel_values: torch.Tensor = field(default_factory=lambda: torch.zeros((4, 2)))


@dataclass(frozen=True)
class _Skeleton:
    example_id: str
    input_ids: tuple[int, ...]
    prompt_token_count: int
    image_pad_physical_start: int = 1
    image_pad_physical_end: int = 2
    image_grid_thw: tuple[int, int, int] = (1, 2, 2)
    merge_size: int = 2
    image_token_id: int = IMAGE_TOKEN_ID
    image_encoding: _ImageEncoding = field(default_factory=_ImageEncoding)


def _request(mode: str, *, seed: int | None) -> RequestIdentity:
    return RequestIdentity(
        backend="fixture",
        backend_version="1",
        mode=mode,  # type: ignore[arg-type]
        n=1,
        seed=seed,
        physical_batch_index=0,
        temperature=0.7,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=512,
    )


def _panel_fixture(
    alias_counts: tuple[int, ...] = (EXACT_ALIAS_COUNT,),
    *,
    repetition_penalty: float = 1.0,
) -> tuple[Human13KUnionManifest, tuple[SourceBoundaryInput, ...]]:
    images: list[ImageRecord] = []
    for image_offset, alias_count in enumerate(alias_counts):
        image_id = 7000 + image_offset
        owner_a = f"owner:{image_id}:a"
        owner_b = f"owner:{image_id}:b"
        owner_missing = f"owner:{image_id}:missing"
        split = min(2, alias_count)
        owner_by_row: list[str] = [owner_a] * split + [owner_b] * (alias_count - split)
        rows: list[PredictionRowInput] = []
        raw_tokens: list[int] = []
        row_ids: list[str] = []
        for index, owner_id in enumerate(owner_by_row):
            row_id = f"row:{image_id}:{index:03d}"
            row_ids.append(row_id)
            if image_offset == 0 and owner_id == owner_a:
                token_id = 7 if index == 0 else 8
            elif image_offset == 0 and owner_id == owner_b:
                token_id = 8 if index == split else 9
            else:
                token_id = 10 + image_offset
            start = len(raw_tokens)
            raw_tokens.extend((token_id, 200 + image_offset))
            rows.append(
                PredictionRowInput(
                    row_id=row_id,
                    row_index=index,
                    category="cat",
                    bbox=(0.0, 0.0, 10.0, 10.0),
                    token_start=start,
                    token_end=start + 2,
                    final_coordinate_token_index=start + 1,
                )
            )
        source = TrajectoryRecord(
            trajectory_id=f"frozen-source:{image_id}",
            request=_request("source_greedy", seed=None),
            raw_token_ids=(41, STOP),
            terminal_token_index=1,
            stop_reason="im_end",
            parser_status="complete",
            rows=(),
            prefix=PrefixRecord((41,), (41,), ()),
            retained_row_ids=(),
            duplicate_row_ids=(),
            matched_row_ids=(),
            replay_token_mask=(False, False),
            duplicate_target_mask=(False, False),
        )
        sampled = TrajectoryRecord(
            trajectory_id=f"frozen-k:{image_id}",
            request=_request("k_sampled", seed=21001),
            raw_token_ids=tuple(raw_tokens),
            terminal_token_index=None,
            stop_reason="length",
            parser_status="complete",
            rows=tuple(rows),
            prefix=PrefixRecord(tuple(raw_tokens), tuple(raw_tokens), ()),
            retained_row_ids=tuple(row_ids),
            duplicate_row_ids=(),
            matched_row_ids=tuple(row_ids),
            replay_token_mask=(False,) * len(raw_tokens),
            duplicate_target_mask=(False,) * len(raw_tokens),
        )
        owners = (
            OwnerRecord(
                owner_a,
                "cat",
                (0.0, 0.0, 10.0, 10.0),
                0,
                "H",
                (),
                tuple(row_ids[:split]),
            ),
            OwnerRecord(
                owner_b,
                "cat",
                (0.0, 0.0, 10.0, 10.0),
                1,
                "H",
                (),
                tuple(row_ids[split:]),
            ),
            OwnerRecord(
                owner_missing,
                "cat",
                (20.0, 0.0, 30.0, 10.0),
                2,
                "H",
                (),
                (),
            ),
        )
        images.append(
            ImageRecord(
                image_id=image_id,
                panel_row_sha256=None,
                image_sha256=f"{image_offset + 1:064x}",
                owners=owners,
                trajectories=(source, sampled),
                duplicate_events=(),
                selected_rows=(),
                g_owner_ids=(),
                h_owner_ids=(owner_a, owner_b, owner_missing),
                m_owner_ids=(),
                replay_row_ids=(),
                target_row_ids=(),
                candidate_row_ids=tuple(row_ids),
            )
        )
    manifest = Human13KUnionManifest(
        schema_version="human13_k_union_manifest.v1",
        binding=default_binding(),
        images=tuple(images),
        arms=(ArmIdentity("fixture", "none", True),),
        denominators=GlobalDenominatorIdentity(len(images), 0, 0, 0, 0, 0, 0),
        full_panel=False,
    )
    boundaries: list[SourceBoundaryInput] = []
    for image in images:
        owner_ids = {image.h_owner_ids[0], image.h_owner_ids[1]}
        aliases = candidate_aliases_for_owners(image, owner_ids=owner_ids)
        boundaries.append(
            SourceBoundaryInput(
                image=FrontierImage(
                    image_id=image.image_id,
                    trajectory_id=f"source-rp:{repetition_penalty}:{image.image_id}",
                    generated_token_ids=(41, STOP),
                    parser="compact_object_box_closed_only",
                    parser_status="accepted",
                    stop_reason="im_end",
                    rows=(),
                    canonical_owner_ids=(),
                    constrained_protected_owner_ids=(),
                    covered_h_owner_ids=(image.h_owner_ids[-1],),
                    uncovered_h_owner_ids=tuple(sorted(owner_ids)),
                    candidate_aliases=aliases,
                    duplicate_events=(),
                    terminal_token_index=1,
                ),
                prompt_token_ids=(7, STOP, 1),
                repetition_penalty=repetition_penalty,
                image_sha256=cast(str, image.image_sha256),
            )
        )
    return manifest, tuple(boundaries)


def _ledger(
    manifest: Human13KUnionManifest,
    boundaries: tuple[SourceBoundaryInput, ...],
) -> CompilerLedger:
    return _build_compiler_ledger_for_test(
        manifest,
        boundaries,
        source_sha256=SOURCE,
        acquisition_sha256=ACQUISITION,
        trajectory_credit_sha256=CREDIT,
    )


def _manifest_sha256(manifest: Human13KUnionManifest) -> str:
    return hashlib.sha256(
        (
            json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    ).hexdigest()


def _raw(
    site: Any, values: dict[int, float], *, requires_grad: bool = False
) -> torch.Tensor:
    return torch.tensor(
        [values.get(token_id, 0.0) for token_id in site.compact_token_ids],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )


def greedy_compiler_loss(
    raw_logits: dict[str, torch.Tensor], ledger: CompilerLedger
) -> torch.Tensor:
    return _greedy_compiler_loss_for_test(raw_logits, ledger)


def greedy_compiler_numerator(
    raw_logits: dict[str, torch.Tensor],
    ledger: CompilerLedger,
    *,
    site_ids: tuple[str, ...],
) -> torch.Tensor:
    return _greedy_compiler_numerator_for_test(raw_logits, ledger, site_ids=site_ids)


def test_binds_exact_309_alias_bank_and_normalizes_owner_then_alias() -> None:
    # Catches accepting a sampled support set or normalizing all aliases globally.
    manifest, boundaries = _panel_fixture()
    ledger = _ledger(manifest, boundaries)
    site = ledger.images[0].site
    assert site is not None
    assert ledger.frozen_alias_count == 309
    assert len(site.alias_children) == 309
    weights = dict(zip(site.valid_token_ids, site.valid_token_weights, strict=True))
    assert weights[7] == pytest.approx(0.25)
    assert weights[8] == pytest.approx(0.25 + 0.5 / 307)
    assert weights[9] == pytest.approx(306 * 0.5 / 307)
    assert sum(site.valid_token_weights) == pytest.approx(1.0)
    assert tuple(binding.token_id for binding in site.alias_children[:2]) == (7, 8)
    assert all(
        not hasattr(binding, "suffix_token_ids") for binding in site.alias_children
    )


def test_rejects_nonexact_alias_bank_and_fresh_support_leakage() -> None:
    # Catches treating new K rows as valid children or weakening exact-309 admission.
    short_manifest, short_boundaries = _panel_fixture((308,))
    with pytest.raises(ValueError, match="309"):
        _ledger(short_manifest, short_boundaries)

    manifest, boundaries = _panel_fixture()
    forged = FrontierCandidateAlias(
        owner_id=boundaries[0].image.uncovered_h_owner_ids[0],
        row_id="fresh-k-row",
        trajectory_id="fresh-k",
        seed=99999,
        owner_iou=1.0,
        token_ids=(12345,),
    )
    changed = replace(
        boundaries[0],
        image=replace(
            boundaries[0].image,
            candidate_aliases=(*boundaries[0].image.candidate_aliases, forged),
        ),
    )
    with pytest.raises(ValueError, match="frozen manifest aliases"):
        _ledger(manifest, (changed,))


def test_rejects_incomplete_frozen_h_owner_partition() -> None:
    # Catches silently dropping a trusted H owner from both coverage sets.
    manifest, boundaries = _panel_fixture()
    omitted = replace(
        boundaries[0],
        image=replace(boundaries[0].image, covered_h_owner_ids=()),
    )
    with pytest.raises(ValueError, match="exact partition"):
        _ledger(manifest, (omitted,))


def test_valid_logmeanexp_is_bounded_and_bad_is_realized_source_child() -> None:
    # Catches unnormalized logsumexp and choosing the largest arbitrary invalid token.
    manifest, boundaries = _panel_fixture()
    ledger = _ledger(manifest, boundaries)
    site = ledger.images[0].site
    assert site is not None and site.bad_token_id == STOP
    raw = _raw(site, {7: 1.0, 8: 2.0, 9: 3.0, STOP: -4.0})
    score = greedy_compiler_site_score(raw, site, repetition_penalty=1.0)
    hand = math.log(
        0.25 * math.exp(1.0)
        + (0.25 + 0.5 / 307) * math.exp(2.0)
        + (306 * 0.5 / 307) * math.exp(3.0)
    )
    assert score.valid_score.item() == pytest.approx(hand, abs=1e-6)
    assert score.valid_score.item() <= score.max_valid_score.item() + 1e-6
    assert score.bad_score.item() == pytest.approx(-4.0)
    assert 12345 not in site.compact_token_ids


def test_greedy_policy_is_sign_aware_rp_without_temperature() -> None:
    # Catches sampling-policy temperature/top-p/top-k leaking into the compiler.
    manifest, boundaries = _panel_fixture(repetition_penalty=1.10)
    ledger = _ledger(manifest, boundaries)
    site = ledger.images[0].site
    assert site is not None
    raw = _raw(site, {7: 2.2, 8: 0.5, 9: -0.3, STOP: -2.0})
    score = greedy_compiler_site_score(raw, site, repetition_penalty=1.10)
    weights = dict(zip(site.valid_token_ids, site.valid_token_weights, strict=True))
    hand = math.log(
        weights[7] * math.exp(2.0)
        + weights[8] * math.exp(0.5)
        + weights[9] * math.exp(-0.3)
    )
    assert score.valid_score.item() == pytest.approx(hand, abs=1e-6)
    assert score.bad_score.item() == pytest.approx(-2.2)
    assert ledger.repetition_penalty == 1.10


def test_source_decode_receipt_proves_each_token_under_exact_rp() -> None:
    # Catches relabeling one durable Source frontier as the other RP surface.
    manifest, boundaries = _panel_fixture(repetition_penalty=1.10)
    boundary = boundaries[0]
    runtime = _build_source_forward_runtime_for_test(
        source_sha256=SOURCE,
        model_identity_sha256="d" * 64,
        tokenizer_identity_sha256="e" * 64,
        model_vocab_size=STOP + 3,
        tokenizer_vocab_size=STOP + 3,
    )
    vocab = runtime.vocab_size
    raw = torch.full((2, vocab), -20.0, dtype=torch.float32)
    raw[0, 41] = 10.0
    raw[1, STOP] = 10.0
    receipt = _admit_source_greedy_decode_for_test(
        boundary,
        raw,
        runtime=runtime,
        manifest_sha256=_manifest_sha256(manifest),
    )
    assert receipt.repetition_penalty == 1.10
    assert receipt.generated_token_ids == (41, STOP)
    forged = object.__new__(type(receipt))
    for name, value in receipt.__dict__.items():
        object.__setattr__(forged, name, value)
    object.__setattr__(forged, "repetition_penalty", 1.0)
    with pytest.raises(ValueError, match="absent or forged"):
        _require_source_decode(forged)
    wrong = raw.clone()
    wrong[1, 42] = 11.0
    with pytest.raises(ValueError, match="greedy argmax"):
        _admit_source_greedy_decode_for_test(
            boundary,
            wrong,
            runtime=runtime,
            manifest_sha256=receipt.manifest_sha256,
        )


def test_source_decode_rejects_truncated_vocab_before_argmax() -> None:
    # Catches hiding a higher-logit token just beyond the submitted row width.
    manifest, boundaries = _panel_fixture()
    boundary = boundaries[0]
    runtime = _build_source_forward_runtime_for_test(
        source_sha256=SOURCE,
        model_identity_sha256="d" * 64,
        tokenizer_identity_sha256="e" * 64,
        model_vocab_size=STOP + 3,
        tokenizer_vocab_size=STOP + 3,
    )
    truncated = torch.full((2, STOP + 1), -20.0)
    truncated[0, 41] = 10.0
    truncated[1, STOP] = 10.0
    with pytest.raises(ValueError, match="exact runtime vocabulary width"):
        _admit_source_greedy_decode_for_test(
            boundary,
            truncated,
            runtime=runtime,
            manifest_sha256=_manifest_sha256(manifest),
        )
    complete = torch.nn.functional.pad(truncated, (0, 2), value=-20.0)
    complete[0, STOP + 1] = 11.0
    with pytest.raises(ValueError, match="greedy argmax"):
        _admit_source_greedy_decode_for_test(
            boundary,
            complete,
            runtime=runtime,
            manifest_sha256=_manifest_sha256(manifest),
        )
    with pytest.raises(ValueError, match="loaded Source runtime"):
        admit_source_greedy_decode(
            boundary,
            runtime=runtime,
            manifest_sha256=_manifest_sha256(manifest),
            prompt_skeleton=_Skeleton(
                example_id="source-prompt:7000",
                input_ids=boundary.prompt_token_ids,
                prompt_token_count=len(boundary.prompt_token_ids),
                image_encoding=cast(Any, _image_encoding(boundary.image_sha256)),
            ),
        )


def test_margin_crossing_and_satisfied_site_gradient() -> None:
    # Catches the hinge sign, margin, or a nonzero gradient after satisfaction.
    manifest, boundaries = _panel_fixture()
    ledger = _ledger(manifest, boundaries)
    site = ledger.images[0].site
    assert site is not None
    violating = _raw(site, {7: 2.0, 8: 2.0, 9: 2.0, STOP: 3.0}, requires_grad=True)
    loss = greedy_compiler_loss({site.site_id: violating}, ledger)
    assert loss.item() == pytest.approx(1.0001, abs=1e-6)
    loss.backward()
    assert violating.grad is not None
    assert violating.grad[site.compact_token_ids.index(STOP)].item() == pytest.approx(
        1.0
    )
    assert violating.grad[site.compact_token_ids.index(7)].item() < 0.0

    satisfied = _raw(site, {7: 4.0, 8: 4.0, 9: 4.0, STOP: 1.0}, requires_grad=True)
    zero = greedy_compiler_loss({site.site_id: satisfied}, ledger)
    zero.backward()
    assert zero.item() == 0.0
    assert satisfied.grad is not None
    assert torch.equal(satisfied.grad, torch.zeros_like(satisfied.grad))


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ("no_boundary", "no_premature_source_boundary"),
        ("no_remaining", "no_trusted_remaining"),
        ("no_aliases", "no_valid_aliases"),
    ),
)
def test_absent_site_reasons_contribute_zero(change: str, reason: str) -> None:
    # Catches silent missing sites or an absent image disappearing from denominator N.
    manifest, boundaries = _panel_fixture()
    boundary = boundaries[0]
    if change == "no_boundary":
        boundary = replace(
            boundary,
            image=replace(
                boundary.image, terminal_token_index=None, generated_token_ids=(41,)
            ),
        )
    elif change == "no_remaining":
        boundary = replace(
            boundary,
            image=replace(
                boundary.image,
                covered_h_owner_ids=manifest.images[0].h_owner_ids,
                uncovered_h_owner_ids=(),
                candidate_aliases=(),
            ),
        )
    else:
        missing = manifest.images[0].h_owner_ids[-1]
        boundary = replace(
            boundary,
            image=replace(
                boundary.image,
                covered_h_owner_ids=manifest.images[0].h_owner_ids[:2],
                uncovered_h_owner_ids=(missing,),
                candidate_aliases=(),
            ),
        )
    ledger = _ledger(manifest, (boundary,))
    assert ledger.images[0].site is None
    assert ledger.images[0].absent_reason == reason
    loss = greedy_compiler_loss({}, ledger)
    assert loss.item() == 0.0


def test_selectors_are_detached_python_evidence() -> None:
    # Catches selector/alias weights accidentally becoming differentiable tensors.
    manifest, boundaries = _panel_fixture()
    ledger = _ledger(manifest, boundaries)
    site = ledger.images[0].site
    assert site is not None
    assert all(type(value) is float for value in site.valid_token_weights)
    assert all(type(value) is int for value in site.valid_token_ids)
    raw = _raw(site, {7: 0.0, 8: 0.0, 9: 0.0, STOP: 1.0}, requires_grad=True)
    greedy_compiler_loss({site.site_id: raw}, ledger).backward()
    assert raw.grad is not None


def test_lineage_round_trip_and_self_consistent_forgery_rejection() -> None:
    # Catches accepting an RP/source/manifest substitution or a rehashed changed ledger.
    manifest, boundaries = _panel_fixture()
    ledger = _ledger(manifest, boundaries)
    loaded = load_compiler_ledger(
        ledger.to_dict(),
        manifest,
        boundaries,
        source_sha256=SOURCE,
        acquisition_sha256=ACQUISITION,
        trajectory_credit_sha256=CREDIT,
    )
    assert loaded == ledger
    assert loaded.content_sha256 == ledger.content_sha256

    document = ledger.to_dict()
    document["images"][0]["site"]["valid_token_weights"][0] += 0.01
    preimage = {
        key: value for key, value in document.items() if key != "content_sha256"
    }
    document["content_sha256"] = hashlib.sha256(
        json.dumps(preimage, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError):
        load_compiler_ledger(
            document,
            manifest,
            boundaries,
            source_sha256=SOURCE,
            acquisition_sha256=ACQUISITION,
            trajectory_credit_sha256=CREDIT,
        )

    _, rp_boundaries = _panel_fixture(repetition_penalty=1.10)
    with pytest.raises(ValueError, match="stored compiler ledger differs"):
        load_compiler_ledger(
            ledger.to_dict(),
            manifest,
            rp_boundaries,
            source_sha256=SOURCE,
            acquisition_sha256=ACQUISITION,
            trajectory_credit_sha256=CREDIT,
        )


def test_unadmitted_compiler_ledgers_cannot_drive_public_loss() -> None:
    # Catches a directly constructed or deserialized ledger entering training.
    manifest, boundaries = _panel_fixture()
    ledger = _ledger(manifest, boundaries)
    site = ledger.images[0].site
    assert site is not None
    logits = {site.site_id: _raw(site, {7: 0.0, 8: 0.0, 9: 0.0, STOP: 1.0})}
    with pytest.raises(ValueError, match="compact-logit receipt"):
        public_greedy_compiler_loss(cast(Any, logits), ledger)
    copied = CompilerLedger.from_dict(ledger.to_dict())
    with pytest.raises(ValueError, match="compact-logit receipt"):
        public_greedy_compiler_loss(cast(Any, logits), copied)
    with pytest.raises((TypeError, ValueError)):
        build_compiler_ledger(manifest, boundaries, object())  # type: ignore[arg-type]


def test_one_global_image_denominator_is_pack_and_gradient_invariant() -> None:
    # Catches pack-local means or applying N more than once across microsteps.
    manifest, boundaries = _panel_fixture((155, 154))
    ledger = _ledger(manifest, boundaries)
    sites = [image.site for image in ledger.images]
    assert all(site is not None for site in sites)
    site_a, site_b = sites
    assert site_a is not None and site_b is not None
    full_logits = {
        site_a.site_id: _raw(
            site_a, {7: 0.0, 8: 0.0, 9: 0.0, STOP: 1.0}, requires_grad=True
        ),
        site_b.site_id: _raw(site_b, {11: 0.0, STOP: 2.0}, requires_grad=True),
    }
    full = greedy_compiler_loss(full_logits, ledger)
    full.backward()
    full_grads: dict[str, torch.Tensor] = {}
    for key, value in full_logits.items():
        assert value.grad is not None
        full_grads[key] = value.grad.detach().clone()

    packed_logits = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in full_logits.items()
    }
    numerator = sum(
        (
            greedy_compiler_numerator(
                {site_id: packed_logits[site_id]}, ledger, site_ids=(site_id,)
            )
            for site_id in packed_logits
        ),
        torch.zeros((), dtype=torch.float32),
    )
    packed = numerator / ledger.logical_image_count
    packed.backward()
    assert torch.allclose(full, packed, atol=1e-7, rtol=0.0)
    for key, value in packed_logits.items():
        assert value.grad is not None
        assert torch.allclose(value.grad, full_grads[key], atol=1e-7, rtol=0.0)


def test_nested_arms_reuse_byte_identical_shared_artifacts_and_fixed_combination() -> (
    None
):
    # Catches reacquisition/recrediting in B or a tunable compiler coefficient.
    manifest, boundaries = _panel_fixture()
    ledger = _ledger(manifest, boundaries)
    nested = _build_nested_arm_artifacts_for_test(
        acquisition_artifact=b"exact-acquisition-bytes",
        trajectory_credit_artifact=b"exact-credit-ledger-bytes",
        compiler_ledger=ledger,
    )
    assert nested.arm_a_shared_artifacts == nested.arm_b_shared_artifacts
    assert nested.arm_a_shared_artifacts[0] is nested.arm_b_shared_artifacts[0]
    assert nested.arm_a_shared_artifacts[1] is nested.arm_b_shared_artifacts[1]
    assert nested.arm_a_compiler_artifact is None
    assert nested.arm_b_compiler_artifact == ledger.canonical_bytes
    trajectory = torch.tensor(2.0, requires_grad=True)
    compiler = torch.tensor(3.0, requires_grad=True)
    total = combined_loss(trajectory, compiler)
    total.backward()
    assert total.item() == 5.0
    assert trajectory.grad is not None and trajectory.grad.item() == 1.0
    assert compiler.grad is not None and compiler.grad.item() == 1.0


def test_packed_position_gather_uses_exact_source_row_and_compact_tokens() -> None:
    # Catches local-vs-packed remap errors or retaining a full-vocabulary row.
    manifest, boundaries = _panel_fixture()
    ledger = _ledger(manifest, boundaries)
    site = ledger.images[0].site
    assert site is not None
    prepared = prepare_on_policy_candidate_scoring(
        frontier_images={7000: boundaries[0].image},
        prompt_skeletons={
            7000: _Skeleton(
                example_id="source-prompt:7000",
                input_ids=(7, STOP, 1),
                prompt_token_count=3,
            )
        },
        global_max_length=4096,
    )
    packed_segment = next(
        segment
        for pack in prepared.packed_plan.packs
        for segment in pack.pack.segments
        if segment.example_id == site.packed_segment_id
    )
    expected_position = packed_segment.start + site.local_causal_position
    vocab = max(site.compact_token_ids) + 2
    full_row = torch.arange(vocab, dtype=torch.float32).unsqueeze(0)
    packed = _bind_packed_compiler_logits_for_test(
        prepared,
        site,
        pack_index=packed_segment.pack_index,
        logits_position_ids=(expected_position,),
        raw_logits=full_row,
    )
    compact = gather_compiler_compact_logits((packed,), ledger)
    assert tuple(compact) == (site.site_id,)
    assert torch.equal(
        compact[site.site_id],
        torch.tensor(site.compact_token_ids, dtype=torch.float32),
    )
    assert compact[site.site_id].shape == (len(site.compact_token_ids),)
    wrong_prompt = prepare_on_policy_candidate_scoring(
        frontier_images={7000: boundaries[0].image},
        prompt_skeletons={
            7000: _Skeleton(
                example_id="wrong-prompt:7000",
                input_ids=(8, IMAGE_TOKEN_ID, 1),
                prompt_token_count=3,
            )
        },
        global_max_length=4096,
    )
    with pytest.raises(ValueError, match="prompt token digest"):
        _bind_packed_compiler_logits_for_test(
            wrong_prompt,
            site,
            pack_index=packed_segment.pack_index,
            logits_position_ids=(expected_position,),
            raw_logits=full_row,
        )
    wrong_prefix_frontier = replace(
        boundaries[0].image,
        generated_token_ids=(42, STOP),
    )
    wrong_prefix = prepare_on_policy_candidate_scoring(
        frontier_images={7000: wrong_prefix_frontier},
        prompt_skeletons={
            7000: _Skeleton(
                example_id="wrong-prefix:7000",
                input_ids=(7, STOP, 1),
                prompt_token_count=3,
            )
        },
        global_max_length=4096,
    )
    with pytest.raises(ValueError, match="Source prefix token digest"):
        _bind_packed_compiler_logits_for_test(
            wrong_prefix,
            site,
            pack_index=packed_segment.pack_index,
            logits_position_ids=(expected_position,),
            raw_logits=full_row,
        )
    with pytest.raises(ValueError, match="causal position"):
        _bind_packed_compiler_logits_for_test(
            prepared,
            site,
            pack_index=packed_segment.pack_index,
            logits_position_ids=(expected_position + 1,),
            raw_logits=full_row,
        )
    with pytest.raises(ValueError, match="exactly"):
        gather_compiler_compact_logits((packed, packed), ledger)


def test_public_loss_requires_bound_compact_logit_receipt() -> None:
    # Catches direct, copied, or ledger-substituted compact tensors driving training.
    manifest, boundaries = _panel_fixture()
    admitted = _admit_compiler_ledger_for_test(_ledger(manifest, boundaries))
    site = admitted.images[0].site
    assert site is not None
    prepared = prepare_on_policy_candidate_scoring(
        frontier_images={7000: boundaries[0].image},
        prompt_skeletons={
            7000: _Skeleton(
                example_id="source-prompt:7000",
                input_ids=(7, STOP, 1),
                prompt_token_count=3,
            )
        },
        global_max_length=4096,
    )
    segment = next(
        value
        for pack in prepared.packed_plan.packs
        for value in pack.pack.segments
        if value.example_id == site.packed_segment_id
    )
    position = segment.start + site.local_causal_position
    raw = torch.zeros((1, max(site.compact_token_ids) + 2), requires_grad=True)
    packed = bind_packed_compiler_logits(
        prepared,
        admitted,
        site_id=site.site_id,
        pack_index=segment.pack_index,
        logits_position_ids=(position,),
        raw_logits=raw,
    )
    copied_packed = object.__new__(type(packed))
    for name, value in packed.__dict__.items():
        object.__setattr__(copied_packed, name, value)
    with pytest.raises(ValueError, match="forged"):
        admit_compiler_compact_logits((copied_packed,), admitted)
    forged_site = replace(site, prompt_token_sha256="f" * 64)
    with pytest.raises((TypeError, ValueError)):
        bind_packed_compiler_logits(
            prepared,
            cast(Any, forged_site),
            site_id=site.site_id,
            pack_index=segment.pack_index,
            logits_position_ids=(position,),
            raw_logits=raw,
        )
    receipt = admit_compiler_compact_logits((packed,), admitted)
    loss = public_greedy_compiler_loss(receipt, admitted)
    loss.backward()
    assert raw.grad is not None
    with pytest.raises(ValueError, match="compact-logit receipt"):
        public_greedy_compiler_loss(
            cast(Any, gather_compiler_compact_logits((packed,), admitted)), admitted
        )
    copied = object.__new__(type(receipt))
    for name, value in receipt.__dict__.items():
        object.__setattr__(copied, name, value)
    with pytest.raises(ValueError, match="absent or forged"):
        public_greedy_compiler_loss(copied, admitted)
    forged = object.__new__(type(receipt))
    for name, value in receipt.__dict__.items():
        object.__setattr__(forged, name, value)
    object.__setattr__(forged, "compiler_ledger_sha256", "f" * 64)
    with pytest.raises(ValueError, match="absent or forged"):
        public_greedy_compiler_loss(forged, admitted)


def test_external_packed_plan_binder_rejects_a_physical_position_mismatch() -> None:
    """Materializer rows must bind the exact co-packed plan, never a second plan."""

    manifest, boundaries = _panel_fixture()
    admitted = _admit_compiler_ledger_for_test(_ledger(manifest, boundaries))
    site = admitted.images[0].site
    assert site is not None
    prepared = prepare_on_policy_candidate_scoring(
        frontier_images={7000: boundaries[0].image},
        prompt_skeletons={
            7000: _Skeleton(
                example_id="source-prompt:7000",
                input_ids=(7, STOP, 1),
                prompt_token_count=3,
            )
        },
        global_max_length=4096,
    )
    segment = next(
        value
        for pack in prepared.packed_plan.packs
        for value in pack.pack.segments
        if value.example_id == site.packed_segment_id
    )
    position = segment.start + site.local_causal_position
    raw = torch.zeros((1, max(site.compact_token_ids) + 2), requires_grad=True)
    lineage = PackedCompilerLineage(
        acquisition_sha256=admitted.acquisition_sha256,
        trajectory_credit_sha256=admitted.trajectory_credit_sha256,
        repetition_penalty=admitted.repetition_penalty,
    )

    receipt = admit_compiler_compact_logits_from_packed_plan(
        prepared.packed_plan,
        admitted,
        rows=(
            CompilerPackedRow(
                site_id=site.site_id,
                pack_index=segment.pack_index,
                logits_position_ids=(position,),
                raw_logits=raw,
            ),
        ),
        lineage=lineage,
    )
    public_greedy_compiler_numerator(receipt, admitted).backward()
    assert raw.grad is not None

    with pytest.raises(ValueError, match="causal position"):
        admit_compiler_compact_logits_from_packed_plan(
            prepared.packed_plan,
            admitted,
            rows=(
                CompilerPackedRow(
                    site_id=site.site_id,
                    pack_index=segment.pack_index,
                    logits_position_ids=(position + 1,),
                    raw_logits=raw.detach().clone().requires_grad_(True),
                ),
            ),
            lineage=lineage,
        )


def test_compiler_evidence_is_compact_not_full_vocabulary() -> None:
    # Catches persisting a full-vocabulary Python row in the compiler receipt.
    manifest, boundaries = _panel_fixture()
    ledger = _ledger(manifest, boundaries)
    site = ledger.images[0].site
    assert site is not None
    assert set(site.compact_token_ids) == {*site.valid_token_ids, site.bad_token_id}
    assert len(site.compact_token_ids) == 4
    payload = asdict(site)
    assert "raw_logits" not in payload
    assert "full_vocab" not in payload


def _source_forward_fixture(tmp_path: Path) -> tuple[Any, ...]:
    """Real PNG, hash-pinned lazy encoding, and a position-keyed stub model."""

    image_path = tmp_path / "source.png"
    Image.new("RGB", (2, 2), color=(3, 5, 7)).save(image_path)
    image_bytes = image_path.read_bytes()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    processor = _StubImageProcessor()
    plan = QwenNoResizeImagePlan(
        example_id="source-image",
        image_path=image_path,
        width=2,
        height=2,
        patch_size=1,
        merge_size=2,
        temporal_patch_size=1,
        required_spatial_factor=2,
        raw_pixels=4,
        raw_patch_rows=4,
        expected_pixel_values_width=3,
        image_grid_thw=(1, 2, 2),
        merged_visual_tokens=1,
        max_raw_pixels=4,
        max_merged_visual_tokens=4,
        image_content_sha256=image_sha256,
        decoded_width=2,
        decoded_height=2,
    )
    encoding = QwenImageEncoding(plan, None, None, processor)
    model = _StubSourceModel(IMAGE_TOKEN_ID + 2, {0: 41, 1: STOP, 2: 41, 3: STOP})
    runtime = _construct_source_forward_runtime(
        source_sha256=SOURCE,
        runtime_artifact_sha256="9" * 64,
        model_identity_sha256="d" * 64,
        tokenizer_identity_sha256="e" * 64,
        model_vocab_size=IMAGE_TOKEN_ID + 2,
        tokenizer_vocab_size=IMAGE_TOKEN_ID + 2,
        model=model,
        image_processor=processor,
    )
    skeleton = _Skeleton(
        example_id="source-prompt:7000",
        input_ids=(7, IMAGE_TOKEN_ID, 1),
        prompt_token_count=3,
        image_encoding=cast(Any, encoding),
    )
    return image_sha256, processor, encoding, model, runtime, skeleton


@dataclass
class _StubImageProcessor:
    def __call__(self, *, images: Any, return_tensors: str, do_resize: bool) -> Any:
        assert return_tensors == "pt" and do_resize is False and len(images) == 1
        return {
            "pixel_values": torch.zeros((4, 3), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


class _StubSourceModel(torch.nn.Module):
    """Row content depends only on the causal position the model was asked for."""

    def __init__(self, vocab_size: int, argmax_by_position: dict[int, int]) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.argmax_by_position = argmax_by_position
        self.calls: list[tuple[int, ...]] = []
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs: Any) -> Any:
        positions = tuple(
            int(value) for value in kwargs["logits_to_keep"].reshape(-1).tolist()
        )
        self.calls.append(positions)
        logits = torch.full(
            (1, len(positions), self.vocab_size), -20.0, dtype=torch.float32
        )
        for row, position in enumerate(positions):
            logits[0, row, self.argmax_by_position[position]] = 10.0 + 0.01 * position
        return SimpleNamespace(logits=logits)


def _forged_forward_inputs(
    history: tuple[int, ...],
    *,
    labeled_position: int,
    selected_row: int,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> QwenForwardInputs:
    """Correct causal label, different selected causal row."""

    length = len(history)
    plan = Fa2VarlenPlan(
        segment_boundaries=(0, length),
        segment_lengths=(length,),
        cu_seq_lens_q=torch.tensor([0, length], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, length], dtype=torch.int32),
        max_length_q=length,
        max_length_k=length,
        attention_mask=None,
        branch_evidence_required=False,
    )
    receipt = QwenForwardReceipt(
        pack_index=0,
        pack_length=length,
        segment_count=1,
        input_ids_shape=(1, length),
        position_ids_shape=(4, 1, length),
        position_row_meaning=("text", "temporal", "height", "width"),
        pixel_values_shape=tuple(int(item) for item in pixel_values.shape),
        image_grid_thw=((1, 2, 2),),
        placeholder_token_count=1,
        expected_visual_token_count=1,
        labels_passed=False,
        use_cache=False,
        logits_to_keep=(labeled_position,),
        inputs_embeds_used=False,
        fa2_varlen_plan=plan,
    )
    return QwenForwardInputs(
        pack_index=0,
        input_ids=torch.tensor([list(history)], dtype=torch.long),
        position_ids=torch.zeros((4, 1, length), dtype=torch.long),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        fa2_varlen_plan=plan,
        receipt=receipt,
        logits_to_keep=torch.tensor([selected_row], dtype=torch.long),
        logits_position_ids=(labeled_position,),
    )


def test_source_greedy_rejects_caller_selected_causal_row(tmp_path: Path) -> None:
    # Catches certifying Source greedy from a caller-selected wrong causal row.
    manifest, boundaries = _panel_fixture()
    image_sha256, processor, _, model, runtime, skeleton = _source_forward_fixture(
        tmp_path
    )
    boundary = replace(
        boundaries[0],
        prompt_token_ids=(7, IMAGE_TOKEN_ID, 1),
        image_sha256=image_sha256,
    )
    pixels = torch.zeros((4, 3), dtype=torch.float32)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    forged = (
        _forged_forward_inputs(
            (7, IMAGE_TOKEN_ID, 1),
            labeled_position=2,
            selected_row=0,
            pixel_values=pixels,
            image_grid_thw=grid,
        ),
        _forged_forward_inputs(
            (7, IMAGE_TOKEN_ID, 1, 41),
            labeled_position=3,
            selected_row=1,
            pixel_values=pixels,
            image_grid_thw=grid,
        ),
    )
    with pytest.raises(ValueError, match="caller-supplied forward inputs"):
        admit_source_greedy_decode(
            boundary,
            runtime=runtime,
            manifest_sha256=_manifest_sha256(manifest),
            prompt_skeleton=skeleton,
            forward_inputs=forged,
        )
    assert model.calls == []
    receipt = admit_source_greedy_decode(
        boundary,
        runtime=runtime,
        manifest_sha256=_manifest_sha256(manifest),
        prompt_skeleton=skeleton,
    )
    assert receipt.generated_token_ids == (41, STOP)
    assert tuple(row.causal_position for row in receipt.forward_rows) == (2, 3)
    # The model itself read exactly the internally derived teacher-forced rows.
    assert model.calls == [(2, 3)]
    wrong_image = _Skeleton(
        example_id=skeleton.example_id,
        input_ids=skeleton.input_ids,
        prompt_token_count=skeleton.prompt_token_count,
        image_encoding=cast(
            Any,
            QwenImageEncoding(
                replace(
                    cast(Any, skeleton.image_encoding).plan,
                    image_content_sha256="f" * 64,
                ),
                None,
                None,
                processor,
            ),
        ),
    )
    with pytest.raises(ValueError, match="image identity"):
        admit_source_greedy_decode(
            boundary,
            runtime=runtime,
            manifest_sha256=_manifest_sha256(manifest),
            prompt_skeleton=wrong_image,
        )
    assert model.calls == [(2, 3)]
