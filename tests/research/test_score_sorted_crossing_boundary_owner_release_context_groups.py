"""Regression coverage for the crossing-boundary scorer's fresh-context semantics.

Kept in its own file so the frozen contract file
(``test_score_sorted_crossing_boundary_owner_release.py``) stays owned by its
author.  This file pins only the logical-context-group identity rule, which a
real fake-backend end-to-end run on the authoritative v2 plan (image ``10707``,
batch 8) failed on twice:

1. the *same* owner legitimately scores several request families from one
   prefill -- the natural-release and native-next-action families share the bare
   context group, and the three forced-``D_C`` coordinate families share the
   ``D_C`` group -- so those must collapse to one group id each, not one per
   family; and
2. an uncached fallback re-read of a boundary is a *separate* execution on its
   own fresh state, so it must not be mistaken for a reused cache;

while a genuinely reused group -- two different owners, or the same owner twice
on the same backend -- must still fail closed.
"""

from __future__ import annotations

import pytest

from scripts.research import score_sorted_crossing_boundary_owner_release as sut


def _group(
    *,
    owner_id: str = "gt:10707:16",
    context_id: str = "10707:boundary-004",
    variant: str = "at_p_plus_e",
    appended_digest: str = "appended-d-c",
    scoring_backend: str = sut.KV_CACHE_BACKEND,
) -> str:
    return sut._context_group_id(  # noqa: SLF001
        gt_owner_id=owner_id,
        context_id=context_id,
        variant=variant,
        appended_digest=appended_digest,
        scoring_backend=scoring_backend,
    )


# ---------------------------------------------------------------------------
# 1. One prefill per (owner, context, variant, appended prefix, backend).
# ---------------------------------------------------------------------------


def test_request_families_sharing_one_prefill_share_one_group_id() -> None:
    # The release and native-next-action families are both scored at the bare
    # context, so they resolve to the identical logical group and must not each
    # claim a fresh cache.
    bare = sut.sha256_json([])
    assert _group(appended_digest=bare) == _group(appended_digest=bare)


def test_bare_context_and_forced_dc_are_distinct_groups() -> None:
    # Different literal roots => genuinely different prefills.
    assert _group(appended_digest=sut.sha256_json([])) != _group(
        appended_digest=sut.sha256_json([151646, 1, 151647, 151648])
    )


def test_different_owners_never_share_a_group_id() -> None:
    assert _group(owner_id="gt:10707:16") != _group(owner_id="gt:10707:17")


def test_different_variants_and_contexts_never_share_a_group_id() -> None:
    assert _group(variant="at_p") != _group(variant="at_p_plus_e")
    assert _group(context_id="10707:boundary-004") != _group(
        context_id="10707:boundary-005"
    )


# ---------------------------------------------------------------------------
# 2. The uncached fallback is a separate execution, not a reused cache.
# ---------------------------------------------------------------------------


def test_uncached_fallback_of_the_same_boundary_is_a_distinct_group() -> None:
    cached = _group(scoring_backend=sut.KV_CACHE_BACKEND)
    uncached = _group(scoring_backend=sut.UNCACHED_BACKEND)
    assert cached != uncached
    # Both may therefore appear in one shard's freshness ledger.
    sut.assert_fresh_context_per_owner([cached, uncached])


# ---------------------------------------------------------------------------
# 3. A genuine reuse still fails closed.
# ---------------------------------------------------------------------------


def test_repeating_one_group_on_one_backend_still_fails_closed() -> None:
    group_id = _group()
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.assert_fresh_context_per_owner([group_id, group_id])


def test_cross_owner_reuse_of_a_literal_group_id_fails_closed() -> None:
    shared = _group(owner_id="gt:10707:16")
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.assert_fresh_context_per_owner(
            [shared, _group(owner_id="gt:10707:17"), shared]
        )


# ---------------------------------------------------------------------------
# 4. Manifest lineage shape tolerance (the .v1 -> .v2 seal-record migration).
# ---------------------------------------------------------------------------


def test_declared_digest_map_reads_both_the_scalar_and_the_record_shape() -> None:
    flat = {"analysis/receipt.json": "a" * 64}
    records = {
        "analysis/receipt.json": {
            "path": "analysis/receipt.json",
            "byte_size": 12,
            "sha256": "a" * 64,
        }
    }
    listed = [{"path": "analysis/receipt.json", "byte_size": 12, "sha256": "a" * 64}]
    expected = {"analysis/receipt.json": "a" * 64}
    for value in (flat, records, listed):
        assert sut.declared_digest_map(value, label="lineage") == expected


def test_lineage_digest_map_reads_both_the_v1_and_v2_key_spellings() -> None:
    digest = {"analysis/receipt.json": "b" * 64}
    v1 = {"prevalence_input_file_sha256": digest}
    v2 = {"prevalence_input_files": {"analysis/receipt.json": {"sha256": "b" * 64}}}
    assert sut.lineage_digest_map(v1, surface="prevalence") == digest
    assert sut.lineage_digest_map(v2, surface="prevalence") == digest


def test_builder_source_reads_both_the_scalar_and_the_record_shape() -> None:
    assert sut.builder_source_sha256({"builder_source_sha256": "c" * 64}) == "c" * 64
    assert (
        sut.builder_source_sha256({"builder_source": {"sha256": "c" * 64, "byte_size": 1}})
        == "c" * 64
    )


def test_unknown_lineage_shape_fails_closed() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.declared_digest_map(17, label="lineage")
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.lineage_digest_map({}, surface="census")
