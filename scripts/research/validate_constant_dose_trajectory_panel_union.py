#!/usr/bin/env python3
"""Validate the frozen 2,432-image Source trajectory-panel union."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import glob
import hashlib
import json
from pathlib import Path
import random
from typing import Any


ROLLOUT_SCHEMA = "current_seeded_sampled_rollouts.v1"
SPLIT_SCHEMA = "label_only_candidate_pool_split.v1"
SPLIT_SEED = 19
GREEDY_SEED = 31000
SAMPLED_SEED_ORDER = tuple(range(31001, 31017))
SAMPLED_SEEDS = frozenset(SAMPLED_SEED_ORDER)
FROZEN_PRODUCER_SCRIPT_SHA256 = (
    "9413a0d891daa59040b99043f719d5d7ee8b1007c260c00df030a98c589861c3"
)
DEFAULT_PRODUCER_SCRIPT = (
    Path(__file__).resolve().with_name("run_current_seeded_sampled_rollouts.py")
)
BAND_NAMES = (
    "sparse_1_to_3",
    "medium_4_to_7",
    "dense_8_to_15",
    "very_dense_16_plus",
)
SPLIT_COUNTS = {
    "train_candidate": 2048,
    "development": 256,
    "heldout": 128,
}
SPLIT_PER_BAND_COUNTS = {
    "train_candidate": dict.fromkeys(BAND_NAMES, 512),
    "development": dict.fromkeys(BAND_NAMES, 64),
    "heldout": dict.fromkeys(BAND_NAMES, 32),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_image_ids_sha256(image_ids: Sequence[str]) -> str:
    value = "".join(f"{image_id}\n" for image_id in image_ids).encode()
    return hashlib.sha256(value).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact is not an object: {path}")
    return dict(value)


def _image_id(value: Any, *, context: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{context} has no usable image_id")
    if isinstance(value, (int, str)) and str(value).strip():
        return str(value).strip()
    raise ValueError(f"{context} has no usable image_id")


def _expand(paths: Sequence[str | Path]) -> list[Path]:
    result: set[Path] = set()
    for item in paths:
        matches = [Path(path) for path in glob.glob(str(item))] or [Path(item)]
        for path in matches:
            if not path.is_file():
                raise FileNotFoundError(path)
            result.add(path.resolve())
    return sorted(result)


def _object_count_band(object_count: int) -> str:
    if 1 <= object_count <= 3:
        return "sparse_1_to_3"
    if object_count <= 7:
        return "medium_4_to_7"
    if object_count <= 15:
        return "dense_8_to_15"
    return "very_dense_16_plus"


def _candidate_pool_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise ValueError(f"blank candidate-pool row {line_number}")
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError(f"candidate-pool row {line_number} is not an object")
        image_id = _image_id(
            value.get("image_id"), context=f"candidate-pool row {line_number}"
        )
        if image_id in seen:
            raise ValueError(f"candidate pool duplicates image_id {image_id}")
        objects = value.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError(f"candidate-pool row {line_number} has empty objects")
        seen.add(image_id)
        rows.append((image_id, _object_count_band(len(objects))))
    if len(rows) != sum(SPLIT_COUNTS.values()):
        raise ValueError(f"candidate pool has {len(rows)} images, expected 2432")
    return rows


def _reconstruct_split(
    rows: Sequence[tuple[str, str]],
) -> dict[str, list[str]]:
    rows_by_band = {band: [] for band in BAND_NAMES}
    for image_id, band in rows:
        rows_by_band[band].append(image_id)
    expected_per_band_total = sum(
        counts[BAND_NAMES[0]] for counts in SPLIT_PER_BAND_COUNTS.values()
    )
    for band, image_ids in rows_by_band.items():
        if len(image_ids) != expected_per_band_total:
            raise ValueError(
                f"candidate pool has {len(image_ids)} images in {band}, "
                f"expected {expected_per_band_total}"
            )

    assignments: dict[str, str] = {}
    generator = random.Random(SPLIT_SEED)
    for band in BAND_NAMES:
        candidates = list(rows_by_band[band])
        generator.shuffle(candidates)
        offset = 0
        for split_name in SPLIT_COUNTS:
            next_offset = offset + SPLIT_PER_BAND_COUNTS[split_name][band]
            for image_id in candidates[offset:next_offset]:
                assignments[image_id] = split_name
            offset = next_offset
    return {
        split_name: [
            image_id
            for image_id, _band in rows
            if assignments.get(image_id) == split_name
        ]
        for split_name in SPLIT_COUNTS
    }


def _split_ids(path: Path, *, split_name: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise ValueError(f"blank {split_name} row {line_number}")
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError(f"{split_name} row {line_number} is not an object")
        image_id = _image_id(
            value.get("image_id"), context=f"{split_name} row {line_number}"
        )
        if image_id in seen:
            raise ValueError(f"{split_name} duplicates image_id {image_id}")
        seen.add(image_id)
        result.append(image_id)
    return result


def _validate_split(pool: Path, receipt_path: Path) -> set[str]:
    pool_rows = _candidate_pool_rows(pool)
    pool_bands = dict(pool_rows)
    expected_membership = _reconstruct_split(pool_rows)
    receipt = _load_object(receipt_path)
    if receipt.get("schema_version") != SPLIT_SCHEMA:
        raise ValueError("split receipt has the wrong schema_version")
    if receipt.get("seed") != SPLIT_SEED:
        raise ValueError("split receipt does not use frozen seed 19")
    if receipt.get("requested_counts") != SPLIT_COUNTS:
        raise ValueError("split receipt does not request exact 2048/256/128 counts")
    if receipt.get("requested_per_band_counts") != SPLIT_PER_BAND_COUNTS:
        raise ValueError(
            "split receipt does not request exact 512/64/32 per-band counts"
        )
    source = receipt.get("input")
    if not isinstance(source, Mapping):
        raise ValueError("split receipt lacks candidate-pool input evidence")
    source_ids = [image_id for image_id, _band in pool_rows]
    if (
        source.get("sha256") != _sha256_file(pool)
        or source.get("count") != len(pool_rows)
        or source.get("ordered_image_ids_sha256")
        != _ordered_image_ids_sha256(source_ids)
    ):
        raise ValueError("split receipt does not bind the supplied candidate pool")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(SPLIT_COUNTS):
        raise ValueError(
            "split receipt does not name exactly train/development/heldout"
        )

    observed_sets: list[set[str]] = []
    for split_name, expected_count in SPLIT_COUNTS.items():
        entry = outputs[split_name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"split receipt {split_name} entry is invalid")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"split receipt {split_name} lacks a path")
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_file() or entry.get("sha256") != _sha256_file(path):
            raise ValueError(f"split receipt hash disagrees for {split_name}")
        ids = _split_ids(path, split_name=split_name)
        if len(ids) != expected_count or entry.get("count") != expected_count:
            raise ValueError(f"split {split_name} count is not {expected_count}")
        if entry.get("ordered_image_ids_sha256") != _ordered_image_ids_sha256(ids):
            raise ValueError(
                f"split receipt ordered image hash disagrees for {split_name}"
            )
        try:
            observed_per_band = {
                band: sum(pool_bands[image_id] == band for image_id in ids)
                for band in BAND_NAMES
            }
        except KeyError as exc:
            raise ValueError(
                f"split {split_name} contains image outside candidate pool: "
                f"{exc.args[0]}"
            ) from exc
        expected_per_band = SPLIT_PER_BAND_COUNTS[split_name]
        if (
            observed_per_band != expected_per_band
            or entry.get("per_band_counts") != expected_per_band
        ):
            raise ValueError(f"split {split_name} has wrong per-band membership")
        if ids != expected_membership[split_name]:
            raise ValueError(
                f"split {split_name} is not deterministic seed-19 membership"
            )
        observed_sets.append(set(ids))

    if any(
        left & right
        for index, left in enumerate(observed_sets)
        for right in observed_sets[index + 1 :]
    ):
        raise ValueError("split membership overlaps")
    pool_ids = set(pool_bands)
    if set().union(*observed_sets) != pool_ids:
        raise ValueError("split membership is not the exact candidate-pool union")
    return pool_ids


def _exact_number(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float(value) == expected
    )


def _decode_seeds(
    config: Mapping[str, Any], *, mode: str, path: Path
) -> tuple[int, ...]:
    expected = {
        "temperature": 0.0 if mode == "greedy" else 0.4,
        "top_p": 1.0 if mode == "greedy" else 0.95,
        "repetition_penalty": 1.0,
    }
    for field, expected_value in expected.items():
        if not _exact_number(config.get(field), expected_value):
            raise ValueError(f"{mode} artifact has non-canonical {field}: {path}")
    max_new_tokens = config.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens != 512
    ):
        raise ValueError(f"{mode} artifact has non-canonical max_new_tokens: {path}")
    raw_seeds = config.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError(f"{mode} artifact lacks declared seeds: {path}")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise ValueError(f"{mode} artifact has non-integer seeds: {path}")
    seeds = tuple(raw_seeds)
    if mode == "greedy":
        if seeds != (GREEDY_SEED,):
            raise ValueError(
                f"greedy artifact does not declare only seed 31000: {path}"
            )
    elif seeds != tuple(sorted(set(seeds))) or not set(seeds) <= SAMPLED_SEEDS:
        raise ValueError(f"sampled artifact has non-canonical sampled seeds: {path}")
    return seeds


def _prompt_policy(artifact: Mapping[str, Any], path: Path) -> tuple[str, set[str]]:
    metadata = artifact.get("prompt_metadata")
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError(f"artifact lacks prompt_metadata: {path}")
    chat_hashes: set[str] = set()
    metadata_keys: set[str] = set()
    for raw_key, raw_entry in metadata.items():
        key = str(raw_key)
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"prompt metadata {key} is not an object: {path}")
        prompt_ids = raw_entry.get("prompt_token_ids")
        if (
            not isinstance(prompt_ids, list)
            or not prompt_ids
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in prompt_ids
            )
        ):
            raise ValueError(f"prompt metadata {key} lacks exact prompt IDs: {path}")
        if raw_entry.get("prompt_token_ids_sha256") != _sha256_json(prompt_ids):
            raise ValueError(
                f"prompt metadata {key} has a prompt-ID hash mismatch: {path}"
            )
        chat_hash = raw_entry.get("chat_text_sha256")
        if (
            not isinstance(chat_hash, str)
            or len(chat_hash) != 64
            or any(character not in "0123456789abcdef" for character in chat_hash)
        ):
            raise ValueError(f"prompt metadata {key} lacks chat-text hash: {path}")
        chat_hashes.add(chat_hash)
        metadata_keys.add(key)
    if len(chat_hashes) != 1:
        raise ValueError(f"artifact does not have one chat-text prompt policy: {path}")
    return next(iter(chat_hashes)), metadata_keys


def _load_panel(
    paths: Sequence[Path], *, mode: str
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any], str]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    model_identity: dict[str, Any] | None = None
    prompt_fingerprint: str | None = None
    panel_declared_seeds: set[int] = set()
    for path in paths:
        artifact = _load_object(path)
        if artifact.get("schema_version") != ROLLOUT_SCHEMA:
            raise ValueError(f"unsupported rollout schema: {path}")
        config = artifact.get("config")
        rollouts = artifact.get("rollouts")
        if (
            not isinstance(config, Mapping)
            or config.get("decode_mode") != mode
            or not isinstance(rollouts, list)
            or not rollouts
        ):
            raise ValueError(f"invalid {mode} rollout artifact: {path}")
        if config.get("sampling_is_not_infer_config") is not True:
            raise ValueError(
                f"{mode} artifact does not attest request-scoped sampling: {path}"
            )
        config_seeds = _decode_seeds(config, mode=mode, path=path)
        panel_declared_seeds.update(config_seeds)
        identity = artifact.get("model_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{mode} artifact lacks model_identity: {path}")
        if model_identity is None:
            model_identity = dict(identity)
        elif model_identity != dict(identity):
            raise ValueError(f"{mode} artifacts do not share exact model_identity")
        fingerprint, metadata_keys = _prompt_policy(artifact, path)
        if prompt_fingerprint is None:
            prompt_fingerprint = fingerprint
        elif prompt_fingerprint != fingerprint:
            raise ValueError(f"{mode} artifacts do not share prompt-policy fingerprint")

        artifact_pairs: list[tuple[str, int]] = []
        example_by_image: dict[str, str] = {}
        for index, row in enumerate(rollouts, 1):
            if not isinstance(row, Mapping) or row.get("decode_mode") != mode:
                raise ValueError(f"invalid {mode} rollout row {index}: {path}")
            image_id = _image_id(row.get("image_id"), context=f"{path} row {index}")
            example_id = _image_id(
                row.get("example_id"), context=f"{path} row {index} example_id"
            )
            raw_seed = row.get("seed")
            if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
                raise ValueError(f"{path} row {index} lacks an integer seed")
            seed = raw_seed
            if seed not in config_seeds:
                raise ValueError(f"{path} row {index} seed is not declared by config")
            previous_example_id = example_by_image.setdefault(image_id, example_id)
            if previous_example_id != example_id:
                raise ValueError(
                    f"{mode} artifact image {image_id} has conflicting example_ids: "
                    f"{path}"
                )
            key = (image_id, seed)
            if key in rows:
                raise ValueError(f"duplicate image-seed pair {key}")
            rows[key] = dict(row)
            artifact_pairs.append(key)
        if (
            len(set(example_by_image.values())) != len(example_by_image)
            or set(example_by_image.values()) != metadata_keys
        ):
            raise ValueError(
                f"{mode} artifact prompt metadata does not cover exact rows: {path}"
            )
        image_order = list(
            dict.fromkeys(image_id for image_id, _seed in artifact_pairs)
        )
        expected_pairs = [
            (image_id, seed) for image_id in image_order for seed in config_seeds
        ]
        if artifact_pairs != expected_pairs:
            raise ValueError(
                f"{mode} artifact rows are not contiguous request-major seed blocks: "
                f"{path}"
            )
    expected_panel_seeds = {GREEDY_SEED} if mode == "greedy" else SAMPLED_SEEDS
    if panel_declared_seeds != expected_panel_seeds:
        raise ValueError(f"{mode} panel config does not cover exact canonical seeds")
    assert model_identity is not None and prompt_fingerprint is not None
    return rows, model_identity, prompt_fingerprint


def validate_panel_union(
    *,
    candidate_pool: Path,
    split_receipt: Path,
    old_greedy: Sequence[Path],
    old_sampled: Sequence[Path],
    new_greedy: Sequence[Path],
    new_sampled: Sequence[Path],
    producer_script: Path | None = None,
) -> dict[str, Any]:
    """Prove the requested fixed-panel, identity, and split invariants."""

    producer = (
        (producer_script or DEFAULT_PRODUCER_SCRIPT).expanduser().resolve(strict=True)
    )
    producer_sha256 = _sha256_file(producer)
    if producer_sha256 != FROZEN_PRODUCER_SCRIPT_SHA256:
        raise ValueError(
            "producer script SHA does not match the frozen audited "
            "request-scoped producer"
        )
    pool_ids = _validate_split(candidate_pool, split_receipt)
    old_greedy_rows, old_identity, old_prompt = _load_panel(old_greedy, mode="greedy")
    old_sampled_rows, sampled_identity, sampled_prompt = _load_panel(
        old_sampled, mode="sampled"
    )
    new_greedy_rows, new_identity, new_prompt = _load_panel(new_greedy, mode="greedy")
    new_sampled_rows, new_sampled_identity, new_sampled_prompt = _load_panel(
        new_sampled, mode="sampled"
    )
    if not (old_identity == sampled_identity == new_identity == new_sampled_identity):
        raise ValueError("old/new artifacts do not have exact model_identity parity")
    if len({old_prompt, sampled_prompt, new_prompt, new_sampled_prompt}) != 1:
        raise ValueError("old/new artifacts do not have prompt-policy parity")
    old_ids = {image_id for image_id, seed in old_greedy_rows if seed == GREEDY_SEED}
    new_ids = {image_id for image_id, seed in new_greedy_rows if seed == GREEDY_SEED}
    if set(old_greedy_rows) != {(image_id, GREEDY_SEED) for image_id in old_ids} or set(
        new_greedy_rows
    ) != {(image_id, GREEDY_SEED) for image_id in new_ids}:
        raise ValueError("greedy panel is not exactly one seed-31000 rollout per image")
    for label, rows, expected_ids in (
        ("old", old_sampled_rows, old_ids),
        ("new", new_sampled_rows, new_ids),
    ):
        expected = {
            (image_id, seed) for image_id in expected_ids for seed in SAMPLED_SEEDS
        }
        if set(rows) != expected:
            raise ValueError(
                f"{label} sampled panel is not exactly seeds 31001..31016 per image"
            )
    if (
        len(old_ids) != 256
        or len(new_ids) != 2176
        or old_ids & new_ids
        or old_ids | new_ids != pool_ids
    ):
        raise ValueError(
            "old/new panel is not a disjoint 256 + 2176 = 2432 candidate-pool union"
        )
    return {
        "schema_version": "constant_dose_trajectory_panel_union.v1",
        "passed": True,
        "image_count": len(pool_ids),
        "old_image_count": len(old_ids),
        "new_image_count": len(new_ids),
        "sampled_seed_count": len(SAMPLED_SEEDS),
        "prompt_policy_fingerprint": old_prompt,
        "execution_metadata": {
            "physical_batch_size": 1,
            "sampling_order": "request_major",
            "rng_reset": "per_image_seed",
            "producer_script": str(producer),
            "producer_script_sha256": producer_sha256,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--split-receipt", type=Path, required=True)
    for name in ("old-greedy", "old-sampled", "new-greedy", "new-sampled"):
        parser.add_argument(f"--{name}", action="append", required=True)
    parser.add_argument("--producer-script", type=Path, default=DEFAULT_PRODUCER_SCRIPT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_panel_union(
        candidate_pool=args.candidate_pool.resolve(strict=True),
        split_receipt=args.split_receipt.resolve(strict=True),
        old_greedy=_expand(args.old_greedy),
        old_sampled=_expand(args.old_sampled),
        new_greedy=_expand(args.new_greedy),
        new_sampled=_expand(args.new_sampled),
        producer_script=args.producer_script,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
