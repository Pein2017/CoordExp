from __future__ import annotations

import random

import pytest

from src.detection.rollin import (
    ObjectInstanceId,
    RollinState,
    make_prefix_rollin_state,
    sample_prefix_rollin_state,
)


def test_uniform_inclusive_k_sampler_can_emit_every_depth() -> None:
    objects = tuple(ObjectInstanceId(f"obj-{i}") for i in range(4))
    seen: set[int] = set()
    rng = random.Random(123)

    for _ in range(1000):
        state = sample_prefix_rollin_state(objects, rng=rng)
        seen.add(state.k)

    assert seen == {0, 1, 2, 3, 4}


def test_rollin_state_partitions_emitted_and_remaining() -> None:
    objects = (
        ObjectInstanceId("A"),
        ObjectInstanceId("B"),
        ObjectInstanceId("C"),
    )
    state = make_prefix_rollin_state(
        objects,
        permutation=(
            ObjectInstanceId("C"),
            ObjectInstanceId("A"),
            ObjectInstanceId("B"),
        ),
        k=2,
    )

    assert state.k == 2
    assert state.emitted == (ObjectInstanceId("C"), ObjectInstanceId("A"))
    assert state.remaining == (ObjectInstanceId("B"),)
    assert set(state.emitted).isdisjoint(state.remaining)
    assert set(state.emitted) | set(state.remaining) == set(objects)


def test_k_equals_n_has_empty_remaining() -> None:
    state = make_prefix_rollin_state(
        (ObjectInstanceId("A"), ObjectInstanceId("B")),
        permutation=(ObjectInstanceId("B"), ObjectInstanceId("A")),
        k=2,
    )

    assert state.emitted == (ObjectInstanceId("B"), ObjectInstanceId("A"))
    assert state.remaining == ()


def test_k_equals_zero_has_all_objects_remaining() -> None:
    state = make_prefix_rollin_state(
        (ObjectInstanceId("A"), ObjectInstanceId("B")),
        permutation=(ObjectInstanceId("B"), ObjectInstanceId("A")),
        k=0,
    )

    assert state.emitted == ()
    assert state.remaining == (ObjectInstanceId("B"), ObjectInstanceId("A"))


def test_sampler_uses_random_permutation_contract() -> None:
    objects = tuple(ObjectInstanceId(f"obj-{i}") for i in range(5))
    rng = random.Random(20260507)
    expected_rng = random.Random(20260507)
    expected_permutation = tuple(expected_rng.sample(objects, k=len(objects)))
    expected_k = expected_rng.randint(0, len(objects))

    state = sample_prefix_rollin_state(objects, rng=rng)

    assert state.permutation == expected_permutation
    assert state.k == expected_k
    assert state.emitted == expected_permutation[:expected_k]
    assert state.remaining == expected_permutation[expected_k:]


def test_sampler_rejects_duplicate_object_ids_without_consuming_rng() -> None:
    rng = random.Random(12345)
    before = rng.getstate()

    with pytest.raises(ValueError, match="duplicate object_instance_ids"):
        sample_prefix_rollin_state(
            (ObjectInstanceId("A"), ObjectInstanceId("A")),
            rng=rng,
        )

    assert rng.getstate() == before


def test_rollin_rejects_duplicate_object_instance_ids() -> None:
    with pytest.raises(ValueError, match="duplicate object_instance_ids"):
        make_prefix_rollin_state(
            (ObjectInstanceId("A"), ObjectInstanceId("A")),
            permutation=(ObjectInstanceId("A"), ObjectInstanceId("A")),
            k=1,
        )


@pytest.mark.parametrize(
    "permutation",
    [
        (ObjectInstanceId("A"),),
        (ObjectInstanceId("A"), ObjectInstanceId("B"), ObjectInstanceId("C")),
        (ObjectInstanceId("A"), ObjectInstanceId("A")),
        (ObjectInstanceId("B"), ObjectInstanceId("A"), ObjectInstanceId("A")),
    ],
)
def test_rollin_rejects_permutation_mismatch(
    permutation: tuple[ObjectInstanceId, ...],
) -> None:
    with pytest.raises(ValueError, match="permutation must contain each object"):
        make_prefix_rollin_state(
            (ObjectInstanceId("A"), ObjectInstanceId("B")),
            permutation=permutation,
            k=1,
        )


@pytest.mark.parametrize("k", [-1, 3])
def test_rollin_rejects_k_outside_inclusive_range(k: int) -> None:
    with pytest.raises(ValueError, match=r"k must satisfy 0 <= k <= 2"):
        make_prefix_rollin_state(
            (ObjectInstanceId("A"), ObjectInstanceId("B")),
            permutation=(ObjectInstanceId("B"), ObjectInstanceId("A")),
            k=k,
        )


@pytest.mark.parametrize("k", [False, 1.0, "1"])
def test_rollin_rejects_non_integer_k(k: object) -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        make_prefix_rollin_state(
            (ObjectInstanceId("A"), ObjectInstanceId("B")),
            permutation=(ObjectInstanceId("B"), ObjectInstanceId("A")),
            k=k,  # type: ignore[arg-type]
        )


def test_rollin_state_constructor_enforces_k_partition_invariants() -> None:
    with pytest.raises(ValueError, match="emitted must equal permutation\\[:k\\]"):
        RollinState(
            k=1,
            permutation=(ObjectInstanceId("A"), ObjectInstanceId("B")),
            emitted=(ObjectInstanceId("B"),),
            remaining=(ObjectInstanceId("B"),),
        )


def test_rollin_state_constructor_normalizes_sequence_fields_to_tuples() -> None:
    state = RollinState(
        k=1,
        permutation=[ObjectInstanceId("A"), ObjectInstanceId("B")],  # type: ignore[arg-type]
        emitted=[ObjectInstanceId("A")],  # type: ignore[arg-type]
        remaining=[ObjectInstanceId("B")],  # type: ignore[arg-type]
    )

    assert state.permutation == (ObjectInstanceId("A"), ObjectInstanceId("B"))
    assert state.emitted == (ObjectInstanceId("A"),)
    assert state.remaining == (ObjectInstanceId("B"),)
    assert isinstance(state.permutation, tuple)
    assert isinstance(state.emitted, tuple)
    assert isinstance(state.remaining, tuple)


def test_rollin_state_constructor_rejects_duplicate_permutation_ids() -> None:
    with pytest.raises(ValueError, match="permutation contains duplicate"):
        RollinState(
            k=1,
            permutation=(ObjectInstanceId("A"), ObjectInstanceId("A")),
            emitted=(ObjectInstanceId("A"),),
            remaining=(ObjectInstanceId("A"),),
        )


def test_rollin_state_constructor_rejects_non_partitioned_remaining() -> None:
    with pytest.raises(ValueError, match="remaining must equal permutation\\[k:\\]"):
        RollinState(
            k=1,
            permutation=(ObjectInstanceId("A"), ObjectInstanceId("B")),
            emitted=(ObjectInstanceId("A"),),
            remaining=(ObjectInstanceId("A"),),
        )
