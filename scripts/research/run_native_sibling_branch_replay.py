#!/usr/bin/env python3
"""Run one-request-at-a-time native sibling-row branch replays.

This module is deliberately experiment-local.  It does not add a decoder
constraint, change the shared inference backend, or teach the model a new
object state.  A naturally emitted parent prefix (and, optionally, one
complete naturally emitted sibling row) is reconstructed exactly, then the
existing Qwen decoder samples fresh suffixes.  Every model call is made with a
physical request cardinality of one so that a branch outcome is attributable to
its own prompt and random seed.

The source-consistent primary mode uses the existing verified sampled runtime
attestation for the bfloat16 model.  ``--runtime-dtype fp32`` is a separate
robustness mode: it uses the same model after an explicit float32 conversion
and calls the backend's local sampling context without weakening the shared
production attestation guard.  Results from the two modes must not be merged
as if they were one runtime lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "native_sibling_branch_replay"
DEFAULT_CONFIG = Path(
    "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml"
)
DEFAULT_SOURCE_JSONL = Path(
    "/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/"
    "val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
)
DEFAULT_ATTESTATION = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/runtime-attestation-v2/"
    "request-scoped-sampling-three-policy-cuda.json"
)
DEFAULT_ROOT_SEED = 2026071301
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_NUM_SEEDS = 32


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _temporary_cwd(path: Path):
    import contextlib

    @contextlib.contextmanager
    def manager():
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    return manager()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay naturally emitted sibling rows with physical batch size one."
    )
    parser.add_argument("--mode", choices=("run", "merge", "check"), default="run")
    parser.add_argument("--image-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--sampled-runtime-attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument("--donor-bundle", type=Path)
    parser.add_argument("--donor-prefix-token-count", type=int)
    parser.add_argument(
        "--branch-row-span",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Complete generated-token span [start,end) copied from the donor bundle.",
    )
    parser.add_argument(
        "--branch-row-index",
        type=int,
        help="Complete-row index in the donor bundle; an alternative to --branch-row-span.",
    )
    parser.add_argument("--branch-row-bundle", type=Path)
    parser.add_argument("--branch-label", default="native-sibling")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--seed-root", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
    )
    parser.add_argument(
        "--runtime-dtype",
        choices=("config", "fp32"),
        default="config",
        help="config uses the source bfloat16 runtime; fp32 is an isolated robustness mode.",
    )
    parser.add_argument(
        "--no-greedy-control",
        action="store_true",
        help="Do not execute the paired greedy control call.",
    )
    parser.add_argument(
        "--exact-donor-token-prompt",
        action="store_true",
        help=(
            "Use the donor bundle's already-tokenized prompt plus natural row tokens "
            "directly. This is required for nested natural prefixes that cannot be "
            "losslessly reconstructed by decoding and re-tokenizing the continuation."
        ),
    )
    parser.add_argument(
        "--use-local-sampling-context",
        action="store_true",
        help=(
            "Execute sampled requests through the backend's local sampling context "
            "instead of rebinding a persisted runtime attestation. The receipt records "
            "this weaker lineage explicitly."
        ),
    )
    parser.add_argument(
        "--shard-receipt",
        action="append",
        type=Path,
        help="Input shard receipt for --mode merge or --mode check (repeatable).",
    )
    parser.add_argument("--merged-output", type=Path)
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    """Validate scientific invariants before loading a model or touching output."""

    if args.mode == "run":
        required = {
            "--image-id": args.image_id,
            "--output-root": args.output_root,
            "--donor-bundle": args.donor_bundle,
            "--donor-prefix-token-count": args.donor_prefix_token_count,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit("run requires " + ", ".join(missing))
        if args.branch_row_span is not None and args.branch_row_index is not None:
            raise SystemExit("choose one of --branch-row-span or --branch-row-index")
        if args.branch_row_bundle is not None and args.branch_row_span is None and args.branch_row_index is None:
            raise SystemExit("--branch-row-bundle requires a row span or row index")
        if args.donor_prefix_token_count < 0:
            raise SystemExit("--donor-prefix-token-count must be non-negative")
    elif args.mode in {"merge", "check"}:
        if not args.shard_receipt:
            raise SystemExit(f"{args.mode} requires at least one --shard-receipt")
        if args.mode == "merge" and args.merged_output is None:
            raise SystemExit("merge requires --merged-output")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.num_seeds <= 0:
        raise SystemExit("--num-seeds must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must satisfy 0 <= shard-index < shard-count")
    if not math.isfinite(float(args.temperature)) or float(args.temperature) != DEFAULT_TEMPERATURE:
        raise SystemExit("this unit freezes temperature at 0.4")
    if not math.isfinite(float(args.top_p)) or float(args.top_p) != DEFAULT_TOP_P:
        raise SystemExit("this unit freezes top_p at 0.95")
    if not math.isfinite(float(args.repetition_penalty)) or float(args.repetition_penalty) != DEFAULT_REPETITION_PENALTY:
        raise SystemExit("this unit freezes repetition penalty at 1.0")
    if args.seeds is not None:
        if not args.seeds:
            raise SystemExit("--seed requires at least one integer")
        if len(set(args.seeds)) != len(args.seeds):
            raise SystemExit("--seed values must be unique")
        if any(seed < -(2**63) or seed > 2**63 - 1 for seed in args.seeds):
            raise SystemExit("--seed values must fit signed 64-bit integers")


def _decode_result_payload(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    value = bundle.get("decode_result", bundle)
    if not isinstance(value, Mapping):
        raise ValueError("donor bundle decode_result must be a JSON object")
    return value


def donor_tokens(bundle: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """Return immutable prompt and generated IDs from a donor bundle."""

    decode = _decode_result_payload(bundle)
    prompt = decode.get("prompt_token_ids")
    generated = decode.get("generated_token_ids")
    if not isinstance(prompt, list) or not isinstance(generated, list):
        raise ValueError("donor bundle requires prompt_token_ids and generated_token_ids")
    return [int(x) for x in prompt], [int(x) for x in generated]


def _donor_lineage(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Extract identity-bearing evidence without trusting a receipt alone.

    Wave-Two branch sources may be either the historical terminal bundle or a
    call bundle emitted by this runner.  The latter stores source lineage in
    its nested ``donor`` record and runtime identity in ``runtime``; both are
    checked against the result-bound decode receipt below before the caller
    compares them with the active runtime.
    """

    schema_version = str(bundle.get("schema_version", ""))
    is_current_call_bundle = schema_version == f"{SCHEMA_VERSION}.call_bundle.v1"
    execution = bundle.get("execution_evidence")
    source_record: Mapping[str, Any]
    runtime_record: Mapping[str, Any] = {}
    if is_current_call_bundle:
        source_record = bundle.get("donor") if isinstance(bundle.get("donor"), Mapping) else {}
        runtime_value = bundle.get("runtime")
        if not source_record:
            raise ValueError("current call bundle lacks donor lineage")
        if not isinstance(runtime_value, Mapping):
            raise ValueError("current call bundle lacks runtime lineage")
        runtime_record = runtime_value
        execution = {
            "image_id": bundle.get("image_id"),
            "source_image_sha256": source_record.get("source_image_sha256"),
            "source_width": source_record.get("source_image_width"),
            "source_height": source_record.get("source_image_height"),
        }
    elif not isinstance(execution, Mapping):
        raise ValueError("donor bundle lacks execution_evidence")
    else:
        source_record = execution
    decode = _decode_result_payload(bundle)
    receipt = decode.get("execution_receipt", execution.get("decode_execution_receipt"))
    if not isinstance(receipt, Mapping):
        raise ValueError("donor bundle lacks decode execution receipt")
    image_id = execution.get("image_id")
    source_sha = execution.get("source_image_sha256")
    source_width = execution.get("source_width")
    source_height = execution.get("source_height")
    if image_id is None or not isinstance(source_sha, str):
        raise ValueError("donor execution evidence lacks image identity")
    model_identity = decode.get("model_identity")
    tokenizer_identity = decode.get("tokenizer_identity")
    generation_fingerprint = decode.get("generation_config_fingerprint")
    if not isinstance(model_identity, Mapping) or not isinstance(tokenizer_identity, Mapping):
        raise ValueError("donor decode lacks model/tokenizer identity")
    if not isinstance(generation_fingerprint, str):
        raise ValueError("donor decode lacks generation-config fingerprint")
    if is_current_call_bundle:
        if runtime_record.get("model_identity") != dict(model_identity):
            raise ValueError("current call bundle runtime model identity disagrees with decode")
        if runtime_record.get("tokenizer_identity") != dict(tokenizer_identity):
            raise ValueError("current call bundle runtime tokenizer identity disagrees with decode")
        if runtime_record.get("generation_config_fingerprint") != generation_fingerprint:
            raise ValueError("current call bundle runtime generation fingerprint disagrees with decode")
        runtime_attention = runtime_record.get("attention_implementation")
        receipt_attention = receipt.get("attention_implementation")
        if runtime_attention is None or receipt_attention != runtime_attention:
            raise ValueError("current call bundle runtime attention disagrees with decode receipt")
        prompt_evidence = bundle.get("prompt")
        prompt_token_ids = decode.get("prompt_token_ids")
        if not isinstance(prompt_evidence, Mapping) or not isinstance(prompt_token_ids, list):
            raise ValueError("current call bundle lacks prompt lineage")
        if prompt_evidence.get("prompt_token_count") != len(prompt_token_ids):
            raise ValueError("current call bundle prompt token count disagrees with decode")
        if prompt_evidence.get("prompt_token_ids_sha256") != _sha256_json(prompt_token_ids):
            raise ValueError("current call bundle prompt hash disagrees with decode")
    return {
        "image_id": str(int(image_id)),
        "source_image_sha256": source_sha,
        "source_width": None if source_width is None else int(source_width),
        "source_height": None if source_height is None else int(source_height),
        "model_identity": dict(model_identity),
        "tokenizer_identity": dict(tokenizer_identity),
        "model_identity_sha256": _sha256_json(model_identity),
        "tokenizer_identity_sha256": _sha256_json(tokenizer_identity),
        "model_identity_fingerprint": receipt.get("model_identity_fingerprint"),
        "tokenizer_identity_fingerprint": receipt.get("tokenizer_identity_fingerprint"),
        "generation_config_fingerprint": generation_fingerprint,
        "attention_implementation": receipt.get("attention_implementation"),
        "runtime_identity": receipt.get("runtime_identity"),
        "source_execution_receipt_schema_version": receipt.get("schema_version"),
    }


def validate_donor_lineage(
    bundle: Mapping[str, Any],
    *,
    image_id: str,
    source_image_sha256: str | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    model_identity: Mapping[str, Any] | None = None,
    tokenizer_identity: Mapping[str, Any] | None = None,
    generation_config_fingerprint: str | None = None,
    attention_implementation: str | None = None,
    strict_runtime: bool = False,
) -> dict[str, Any]:
    """Fail fast when a donor belongs to another image or runtime lineage."""

    lineage = _donor_lineage(bundle)
    if lineage["image_id"] != str(int(image_id)):
        raise ValueError(
            f"donor image id {lineage['image_id']} does not match requested image {image_id}"
        )
    if source_image_sha256 is not None and lineage["source_image_sha256"] != source_image_sha256:
        raise ValueError("donor source image digest does not match active source image")
    if source_width is not None and lineage["source_width"] not in {None, int(source_width)}:
        raise ValueError("donor source image width does not match active source image")
    if source_height is not None and lineage["source_height"] not in {None, int(source_height)}:
        raise ValueError("donor source image height does not match active source image")
    if strict_runtime:
        if model_identity is not None and lineage["model_identity"] != dict(model_identity):
            raise ValueError("donor model identity does not match active runtime")
        if tokenizer_identity is not None and lineage["tokenizer_identity"] != dict(tokenizer_identity):
            raise ValueError("donor tokenizer identity does not match active runtime")
        if generation_config_fingerprint is not None and lineage["generation_config_fingerprint"] != generation_config_fingerprint:
            raise ValueError("donor generation-config fingerprint does not match active runtime")
        if attention_implementation is not None and lineage["attention_implementation"] not in {None, attention_implementation}:
            raise ValueError("donor attention implementation does not match active runtime")
    return {
        key: value
        for key, value in lineage.items()
        if key not in {"model_identity", "tokenizer_identity"}
    }


def complete_row_spans(tokens: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Extract complete compact rows without interpreting their category text."""

    from src.analysis.sampled_rescue_transition.comparison import trajectory_rows

    return tuple((int(start), int(end)) for start, end in trajectory_rows(tokens))


def reconstruct_donor_prefix(
    bundle: Mapping[str, Any],
    prefix_token_count: int,
) -> dict[str, Any]:
    """Reconstruct and validate an exact natural boundary from donor IDs."""

    prompt, generated = donor_tokens(bundle)
    count = int(prefix_token_count)
    if count < 0 or count > len(generated):
        raise ValueError("donor prefix count is outside generated token range")
    prefix = generated[:count]
    spans = complete_row_spans(prefix)
    if count and (not spans or spans[-1][1] != count):
        raise ValueError("donor prefix must end at a complete row boundary")
    return {
        "donor_prompt_token_ids": prompt,
        "donor_prefix_token_ids": prefix,
        "donor_prefix_token_count": count,
        "donor_prompt_token_ids_sha256": _sha256_json(prompt),
        "donor_prefix_token_ids_sha256": _sha256_json(prefix),
        "reconstructed_prompt_token_ids": [*prompt, *prefix],
        "reconstructed_prompt_token_ids_sha256": _sha256_json([*prompt, *prefix]),
        "prefix_row_spans": [list(span) for span in spans],
    }


def extract_branch_row_tokens(
    bundle: Mapping[str, Any],
    *,
    row_span: Sequence[int] | None = None,
    row_index: int | None = None,
    prefix_token_count: int,
    relative_suffix: bool = False,
) -> dict[str, Any] | None:
    """Select one complete naturally emitted row to append after a parent prefix."""

    if row_span is not None and row_index is not None:
        raise ValueError("row_span and row_index are mutually exclusive")
    if row_span is None and row_index is None:
        return None
    _, generated = donor_tokens(bundle)
    spans = complete_row_spans(generated)
    if row_span is not None:
        if len(row_span) != 2:
            raise ValueError("row_span must contain start and end")
        start, end = (int(row_span[0]), int(row_span[1]))
        selected = (start, end)
        if selected not in spans:
            raise ValueError("requested branch span is not a complete natural row")
    else:
        index = int(row_index)
        if index < 0 or index >= len(spans):
            raise ValueError("branch row index is outside complete-row range")
        selected = spans[index]
    start, end = selected
    # A row selected from the original full trajectory is indexed relative to
    # that trajectory and must occur after the parent prefix.  A fresh child
    # bundle is generated from the already reconstructed parent prompt, so its
    # generated-token coordinate system starts at zero; the caller validates
    # that parent prompt before opting into this relative-suffix mode.
    if not relative_suffix and start != int(prefix_token_count):
        raise ValueError("original-donor branch row must start exactly at the parent prefix")
    row_tokens = generated[start:end]
    return {
        "source_token_span": [start, end],
        "token_ids": [int(x) for x in row_tokens],
        "token_count": end - start,
        "token_ids_sha256": _sha256_json(row_tokens),
        "natural_complete_row": True,
    }


def reconstruct_branch_prompt(
    donor_bundle: Mapping[str, Any],
    *,
    prefix_token_count: int,
    branch_bundle: Mapping[str, Any] | None = None,
    branch_row_span: Sequence[int] | None = None,
    branch_row_index: int | None = None,
) -> dict[str, Any]:
    """Build the exact token sequence used by the prompt builder."""

    prefix = reconstruct_donor_prefix(donor_bundle, prefix_token_count)
    branch_source = branch_bundle or donor_bundle
    relative_suffix = branch_bundle is not None
    if relative_suffix:
        branch_prompt, _ = donor_tokens(branch_source)
        expected_parent_prompt = prefix["reconstructed_prompt_token_ids"]
        if branch_prompt != expected_parent_prompt:
            raise ValueError(
                "distinct branch bundle prompt must equal donor prompt plus exact parent prefix"
            )
    branch = extract_branch_row_tokens(
        branch_source,
        row_span=branch_row_span,
        row_index=branch_row_index,
        prefix_token_count=prefix_token_count,
        relative_suffix=relative_suffix,
    )
    branch_ids = [] if branch is None else branch["token_ids"]
    combined = [*prefix["donor_prefix_token_ids"], *branch_ids]
    return {
        **prefix,
        "branch": branch,
        "assistant_continuation_token_ids": combined,
        "assistant_continuation_token_ids_sha256": _sha256_json(combined),
        "expected_prompt_token_ids": [
            *prefix["donor_prompt_token_ids"],
            *combined,
        ],
        "expected_prompt_token_ids_sha256": _sha256_json(
            [*prefix["donor_prompt_token_ids"], *combined]
        ),
    }


def _build_reconstructed_prompt_record(
    raw: Any,
    template_config: Any,
    *,
    processor: Any,
    row_index: int,
    tokenizer: Any,
    continuation_token_ids: Sequence[int],
) -> Any:
    """Build a canonical prompt, omitting an empty assistant continuation."""

    from src.inference.prompt import AssistantContinuation, build_prompt_record

    kwargs: dict[str, Any] = {}
    if continuation_token_ids:
        continuation_text = tokenizer.decode(
            list(continuation_token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        kwargs["assistant_continuation"] = AssistantContinuation(text=continuation_text)
    return build_prompt_record(
        raw,
        template_config,
        processor=processor,
        row_index=row_index,
        **kwargs,
    )


def select_runtime_prompt_token_ids(
    *,
    built_prompt_token_ids: Sequence[int],
    expected_prompt_token_ids: Sequence[int],
    exact_donor_token_prompt: bool,
) -> tuple[list[int], str]:
    """Choose the executed prompt while preserving the ordinary image plan.

    The standard path requires a byte-for-byte token round trip through the
    prompt builder. Nested natural prefixes can contain an already-tokenized
    assistant continuation whose decode/re-tokenize form is not identical.
    The explicit exact-donor path therefore trusts the attested donor token IDs
    after proving that its base prompt is exactly the prompt built from the
    active image and template.
    """

    built = [int(token) for token in built_prompt_token_ids]
    expected = [int(token) for token in expected_prompt_token_ids]
    if exact_donor_token_prompt:
        if expected[: len(built)] != built:
            raise RuntimeError(
                "exact donor prompt does not extend the active base prompt: "
                f"base_len={len(built)} expected_len={len(expected)}"
            )
        return expected, "exact_donor_token_ids"
    if built != expected:
        mismatch_index = next(
            (
                index
                for index, (actual, target) in enumerate(
                    zip(built, expected, strict=False)
                )
                if actual != target
            ),
            min(len(built), len(expected)),
        )
        raise RuntimeError(
            "reconstructed donor prompt does not round-trip through tokenizer: "
            f"actual_len={len(built)} expected_len={len(expected)} "
            f"first_mismatch={mismatch_index} "
            f"actual_window={built[max(0, mismatch_index - 4):mismatch_index + 5]} "
            f"expected_window={expected[max(0, mismatch_index - 4):mismatch_index + 5]}"
        )
    return built, "decoded_and_retokenized_continuation"


def _default_seed(
    *,
    seed_root: int,
    image_id: str,
    parent_prompt_hash: str,
    parent_prefix_hash: str,
    index: int,
) -> int:
    """Derive paired seeds without allowing a branch label to change them."""

    payload = {
        "schema_version": f"{SCHEMA_VERSION}.seed_schedule.v1",
        "seed_root": int(seed_root),
        "image_id": str(int(image_id)),
        "parent_prompt_token_ids_sha256": str(parent_prompt_hash),
        "parent_prefix_token_ids_sha256": str(parent_prefix_hash),
        "index": int(index),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def _seed_schedule(
    args: argparse.Namespace,
    *,
    image_id: str,
    parent_prompt_hash: str,
    parent_prefix_hash: str,
    branch_label: str | None = None,
) -> list[int]:
    # ``branch_label`` is retained only for source compatibility with an early
    # local caller; it is intentionally ignored so paired sibling arms share
    # exactly the same random seed vector.
    del branch_label
    if args.seeds is not None:
        seeds = [int(seed) for seed in args.seeds]
    else:
        seeds = [
            _default_seed(
                seed_root=int(args.seed_root),
                image_id=str(image_id),
                parent_prompt_hash=parent_prompt_hash,
                parent_prefix_hash=parent_prefix_hash,
                index=index,
            )
            for index in range(int(args.num_seeds))
        ]
    if len(set(seeds)) != len(seeds):
        raise ValueError("sampling seed schedule contains duplicate seeds")
    return seeds[int(args.shard_index) :: int(args.shard_count)]


def _seed_schedule_identity(
    args: argparse.Namespace,
    *,
    image_id: str,
    parent_prompt_hash: str,
    parent_prefix_hash: str,
    selected_seeds: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}.seed_schedule.v1",
        "algorithm": "explicit-cli" if args.seeds is not None else "sha256-first-63-bits",
        "seed_root": int(args.seed_root),
        "image_id": str(int(image_id)),
        "parent_prompt_token_ids_sha256": str(parent_prompt_hash),
        "parent_prefix_token_ids_sha256": str(parent_prefix_hash),
        "branch_label_excluded": True,
        "requested_seed_count": len(args.seeds) if args.seeds is not None else int(args.num_seeds),
        "selected_seed_count": len(selected_seeds),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "selected_seeds": [int(seed) for seed in selected_seeds],
    }


def _request_id(
    *,
    image_id: str,
    branch_label: str,
    prompt_hash: str,
    kind: str,
    index: int,
    seed: int | None,
) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in branch_label)
    suffix = f"{kind}-index-{index}"
    if seed is not None:
        suffix += f"-seed-{int(seed)}"
    return f"native-sibling:{image_id}:branch-{safe}:prompt-{prompt_hash[:12]}:{suffix}"


def _model_dtype_summary(model: Any) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    for parameter in model.parameters():
        counter[str(parameter.dtype)] += int(parameter.numel())
    return {
        "parameter_dtype_counts": dict(sorted(counter.items())),
        "parameter_dtype_names": sorted(counter),
    }


def _attention_implementation(model: Any, configured: str) -> str:
    value = getattr(getattr(model, "config", None), "_attn_implementation", None)
    return str(value or configured)


def _model_to_fp32(model: Any) -> None:
    try:
        model.to(dtype=__import__("torch").float32)
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        raise RuntimeError("full-model fp32 conversion failed") from exc


def _raw_for_image(raw_rows: Sequence[Any], image_id: str) -> Any:
    for row in raw_rows:
        source = row.metadata.get("source") if hasattr(row.metadata, "get") else None
        candidate = source.get("image_id") if hasattr(source, "get") else None
        if candidate is not None and str(int(candidate)) == str(int(image_id)):
            return row
        if str(getattr(row, "example_id", "")) == str(image_id):
            return row
    raise ValueError(f"image id {image_id} was not found in source JSONL")


_NATURAL_TERMINATION_REASONS = frozenset(
    {
        "eos",
        "eos_token",
        "im_end",
        "stop",
        "terminal_eos",
    }
)
_TOKEN_LIMIT_TERMINATION_REASONS = frozenset(
    {
        "length",
        "max_new_tokens",
        "max_tokens",
        "token_limit",
        "token_limit_reached",
        "truncated",
    }
)


def _horizon_termination_classification(
    *,
    stop_reason: str,
    parse_status: str,
    complete_row_count: int,
    dropped_prediction_count: int,
) -> str:
    """Classify whether the H=4 projection ended naturally or was censored.

    A token-limit stop is never treated as natural, even when it happens just
    after the fourth complete row.  Parser drops likewise make the result
    invalid for terminal interpretation.  Empty output is valid only when the
    parser produced no drops and the backend reports an EOS-style stop.
    """

    normalized_reason = str(stop_reason).strip().lower().replace("-", "_")
    if normalized_reason in _TOKEN_LIMIT_TERMINATION_REASONS:
        return "token_limit_truncated"
    parser_valid = parse_status in {"accepted", "empty"} and int(dropped_prediction_count) == 0
    if not parser_valid:
        return "invalid_or_malformed"
    if int(complete_row_count) >= 4:
        return "horizon_reached"
    if normalized_reason in _NATURAL_TERMINATION_REASONS:
        return "natural_termination_before_horizon"
    return "invalid_or_malformed"


def _parse_result_receipt(result: Any, raw: Any) -> dict[str, Any]:
    from src.analysis.sampled_rescue_transition.comparison import trajectory_rows
    from src.inference.parsing import parse_compact_object_box_closed

    parsed = parse_compact_object_box_closed(
        result.parser_text,
        row_id=result.request_id,
        row_index=0,
        image_width=raw.image.width,
        image_height=raw.image.height,
    )
    spans = trajectory_rows(result.generated_token_ids)
    horizon_spans = tuple(spans[:4])
    parse_artifact = parsed.to_artifact_dict()
    horizon_termination = _horizon_termination_classification(
        stop_reason=str(getattr(result, "stop_reason", "")),
        parse_status=str(parse_artifact["parse_status"]),
        complete_row_count=len(spans),
        dropped_prediction_count=int(parse_artifact["dropped_prediction_count"]),
    )
    parser_valid = str(parse_artifact["parse_status"]) in {"accepted", "empty"} and int(
        parse_artifact["dropped_prediction_count"]
    ) == 0
    horizon_projection = {
        "horizon_rows": 4,
        "first_four_complete_row_spans": [list(span) for span in horizon_spans],
        "first_four_complete_row_token_ids": [
            [int(token) for token in result.generated_token_ids[start:end]]
            for start, end in horizon_spans
        ],
        "natural_termination_before_horizon": horizon_termination
        == "natural_termination_before_horizon",
        "termination_classification": horizon_termination,
        "stop_reason": str(getattr(result, "stop_reason", "")),
        "parser_valid": parser_valid,
    }
    if not spans:
        parser_status = "no_complete_rows"
    elif parse_artifact["parse_status"] in {"accepted", "accepted_with_drops"}:
        parser_status = "complete_rows_with_optional_drops"
    else:
        parser_status = str(parse_artifact["parse_status"])
    return {
        "parser_status": parser_status,
        "parse_result": parse_artifact,
        "complete_row_spans": [list(span) for span in spans],
        "complete_row_count": len(spans),
        "horizon_projection": horizon_projection,
    }


def _call_bundle(
    *,
    result: Any,
    raw: Any,
    image_id: str,
    branch_label: str,
    sampling_seed: int | None,
    donor_evidence: Mapping[str, Any],
    prompt_evidence: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    parser = _parse_result_receipt(result, raw)
    decode = result.to_artifact_dict()
    return {
        "schema_version": f"{SCHEMA_VERSION}.call_bundle.v1",
        "request_id": result.request_id,
        "image_id": str(image_id),
        "branch_label": str(branch_label),
        "sampling_seed": sampling_seed,
        "donor": dict(donor_evidence),
        "prompt": dict(prompt_evidence),
        "runtime": dict(runtime_evidence),
        "executed_call_attestation": {
            "physical_batch_size": 1,
            "request_count": 1,
            "request_id": str(result.request_id),
            "batch_cardinality_asserted_by_runner": True,
            "backend_batch_request_order_fingerprint": (
                None
                if result.execution_receipt is None
                else result.execution_receipt.batch_request_order_fingerprint
            ),
        },
        "raw_generated_token_ids": [int(x) for x in result.generated_token_ids],
        "raw_generated_text": str(result.raw_generated_text),
        "decode_result": decode,
        **parser,
    }


def _execution_contract(
    *,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_fingerprint: str,
    config_dtype: str,
    runtime_dtype: str,
    dtype_summary: Mapping[str, Any],
    attention_implementation: str,
    policy: Mapping[str, Any],
    sampling_attestation_mode: str,
) -> dict[str, Any]:
    return {
        "physical_batch_size": 1,
        "model_config_dtype": str(config_dtype),
        "runtime_dtype_mode": str(runtime_dtype),
        "actual_model_parameter_dtypes": dict(dtype_summary),
        "attention_implementation": str(attention_implementation),
        "decode_generation_policy": dict(policy),
        "generation_config_fingerprint": generation_fingerprint,
        "model_identity": dict(model_identity),
        "tokenizer_identity": dict(tokenizer_identity),
        "sampling_attestation_mode": str(sampling_attestation_mode),
    }


def merge_shard_receipts(paths: Sequence[Path]) -> dict[str, Any]:
    """Merge shard receipts with deterministic ordering and collision checks."""

    resolved = sorted({Path(path).expanduser().resolve() for path in paths}, key=str)
    if len(resolved) != len(paths):
        raise ValueError("duplicate shard receipt path")
    receipts = [_read_json(path) for path in resolved]
    if not receipts:
        raise ValueError("at least one shard receipt is required")
    contracts = [_sha256_json(receipt.get("execution_contract")) for receipt in receipts]
    if len(set(contracts)) != 1:
        raise ValueError("shard execution contracts do not match")
    calls: list[dict[str, Any]] = []
    for receipt, shard_path in zip(receipts, resolved, strict=True):
        if receipt.get("schema_version") != f"{SCHEMA_VERSION}.shard_receipt.v1":
            raise ValueError(f"unsupported shard receipt schema: {shard_path}")
        for call in receipt.get("calls", []):
            if not isinstance(call, Mapping):
                raise ValueError(f"non-object call summary in {shard_path}")
            calls.append({**dict(call), "shard_receipt": str(shard_path)})
    request_ids = [str(call.get("request_id")) for call in calls]
    if any(not value or value == "None" for value in request_ids):
        raise ValueError("every merged call must have a request_id")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("duplicate request_id across shards")
    sampled = [call for call in calls if call.get("sampling_seed") is not None]
    seeds = [int(call["sampling_seed"]) for call in sampled]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate sampling_seed across shards")
    calls.sort(key=lambda call: str(call["request_id"]))
    seed_to_request = {str(int(call["sampling_seed"])): str(call["request_id"]) for call in sampled}
    merged = {
        "schema_version": f"{SCHEMA_VERSION}.merged_receipt.v1",
        "execution_contract": receipts[0]["execution_contract"],
        "execution_contract_sha256": contracts[0],
        "shard_receipts": [str(path) for path in resolved],
        "call_count": len(calls),
        "calls": calls,
        "seed_to_request_id": dict(sorted(seed_to_request.items())),
        "stable_attribution_check": {
            "request_ids_unique": True,
            "sampling_seeds_unique": True,
            "physical_batch_size_all_one": all(
                int(receipts[0]["execution_contract"].get("physical_batch_size", 0)) == 1
                for _ in calls
            ),
        },
    }
    return merged


def check_shard_receipts(paths: Sequence[Path]) -> dict[str, Any]:
    merged = merge_shard_receipts(paths)
    if not merged["stable_attribution_check"]["physical_batch_size_all_one"]:
        raise ValueError("merged shards do not attest physical batch size one")
    return {
        "schema_version": f"{SCHEMA_VERSION}.merge_check.v1",
        "execution_contract_sha256": merged["execution_contract_sha256"],
        "shard_count": len(merged["shard_receipts"]),
        "call_count": merged["call_count"],
        "stable_attribution_check": merged["stable_attribution_check"],
    }


def _run_one_request(
    *,
    backend: Any,
    request: Any,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_fingerprint: str,
    capability: Any | None,
) -> Any:
    # The one-element list is intentional: changing it to a grouped call would
    # change the scientific unit and invalidate the per-seed attribution.
    if request.generation_policy.mode == "sampled" and capability is not None:
        return backend.generate_batch_with_verified_runtime_attestation(
            [request],
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            verified_runtime_attestation=capability,
        )[0]
    if request.generation_policy.mode == "sampled":
        # Used by the optional full-model fp32 robustness mode and by an
        # explicitly requested exploratory local-sampling run. The shared
        # production guard remains unchanged; receipts record that no
        # persisted capability was used.
        return backend._generate_batch(  # noqa: SLF001 - experiment-local seam
            [request],
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            execution_context="sampling_attestation",
        )[0]
    return backend.generate_batch(
        [request],
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )[0]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Load one image, reconstruct one branch, and execute one-call requests."""

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import (
        DecodeGenerationPolicy,
        DecodeRequest,
        HFGenerateBackend,
        load_and_rebind_sampled_runtime_attestation_aggregate,
    )
    from src.inference.image_plan import (
        materialize_image_plan_batch,
        verify_processor_model_vision_parity,
    )
    from src.inference.pipeline import _processor_config, _template_config, _tokenizer_identity
    from src.inference.runtime import assemble_runtime

    donor_path = args.donor_bundle.expanduser().resolve(strict=True)
    donor = _read_json(donor_path)
    branch_source_path = args.branch_row_bundle.expanduser().resolve(strict=True) if args.branch_row_bundle else donor_path
    branch_source = donor if branch_source_path == donor_path else _read_json(branch_source_path)
    reconstruction = reconstruct_branch_prompt(
        donor,
        prefix_token_count=int(args.donor_prefix_token_count),
        branch_bundle=(
            branch_source
            if args.branch_row_bundle and branch_source_path != donor_path
            else None
        ),
        branch_row_span=args.branch_row_span,
        branch_row_index=args.branch_row_index,
    )
    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    if args.runtime_dtype == "fp32" and resolved.config.model.dtype != "bf16":
        raise SystemExit("fp32 robustness mode expects a bfloat16 source config")
    raw = _raw_for_image(load_raw_examples(source_path), str(args.image_id))
    source_image_path = Path(raw.image.path).expanduser().resolve(strict=True)
    source_image_sha256 = _sha256_file(source_image_path)
    donor_lineage = validate_donor_lineage(
        donor,
        image_id=str(args.image_id),
        source_image_sha256=source_image_sha256,
        source_width=int(raw.image.width),
        source_height=int(raw.image.height),
        strict_runtime=False,
    )
    branch_lineage = None
    if args.branch_row_bundle and branch_source_path != donor_path:
        branch_lineage = validate_donor_lineage(
            branch_source,
            image_id=str(args.image_id),
            source_image_sha256=source_image_sha256,
            source_width=int(raw.image.width),
            source_height=int(raw.image.height),
            strict_runtime=False,
        )
    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    if args.runtime_dtype == "fp32":
        _model_to_fp32(qwen.model)
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    continuation_ids = reconstruction["assistant_continuation_token_ids"]
    prompt_record = _build_reconstructed_prompt_record(
        raw,
        _template_config(resolved.config),
        processor=qwen.processor,
        row_index=0,
        tokenizer=qwen.tokenizer,
        continuation_token_ids=([] if args.exact_donor_token_prompt else continuation_ids),
    )
    prompt_token_ids, prompt_construction_mode = select_runtime_prompt_token_ids(
        built_prompt_token_ids=prompt_record.prompt_token_ids,
        expected_prompt_token_ids=reconstruction["expected_prompt_token_ids"],
        exact_donor_token_prompt=bool(args.exact_donor_token_prompt),
    )
    if _sha256_json(prompt_token_ids) != reconstruction["expected_prompt_token_ids_sha256"]:
        raise RuntimeError("reconstructed prompt hash mismatch")
    image_plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    model_inputs = image_plan.model_inputs_by_row_id[prompt_record.row_id]
    generation_fingerprint = sha256_json(resolved.config.generation.model_dump(mode="json"))
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    sampled_policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=int(args.max_new_tokens),
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
    )
    greedy_policy = DecodeGenerationPolicy.greedy(
        max_new_tokens=int(args.max_new_tokens),
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
    )
    dtype_summary = _model_dtype_summary(qwen.model)
    config_dtype = str(resolved.config.model.dtype)
    actual_attention = _attention_implementation(qwen.model, resolved.config.model.attn_implementation)
    if actual_attention != "sdpa":
        raise SystemExit(f"this unit requires SDPA, observed {actual_attention}")
    if args.runtime_dtype == "config" and config_dtype != "bf16":
        raise SystemExit(f"source-consistent mode requires bfloat16 config, observed {config_dtype}")
    if args.runtime_dtype == "config" and "torch.bfloat16" not in dtype_summary["parameter_dtype_names"]:
        raise SystemExit("source-consistent mode did not load any bfloat16 model parameters")
    if args.runtime_dtype == "fp32" and dtype_summary["parameter_dtype_names"] != ["torch.float32"]:
        raise SystemExit("fp32 robustness mode did not convert all model parameters to float32")
    donor_lineage = validate_donor_lineage(
        donor,
        image_id=str(args.image_id),
        source_image_sha256=source_image_sha256,
        source_width=int(raw.image.width),
        source_height=int(raw.image.height),
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
        attention_implementation=actual_attention,
        strict_runtime=args.runtime_dtype == "config",
    )
    if args.branch_row_bundle and branch_source_path != donor_path:
        branch_lineage = validate_donor_lineage(
            branch_source,
            image_id=str(args.image_id),
            source_image_sha256=source_image_sha256,
            source_width=int(raw.image.width),
            source_height=int(raw.image.height),
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            attention_implementation=actual_attention,
            strict_runtime=args.runtime_dtype == "config",
        )
    sampling_attestation_mode = "local-direct-sampling-context"
    capability = None
    if args.runtime_dtype == "config" and not args.use_local_sampling_context:
        capability = load_and_rebind_sampled_runtime_attestation_aggregate(
            args.sampled_runtime_attestation.expanduser().resolve(strict=True),
            decode_generation_policy_fingerprint=sampled_policy.fingerprint,
            backend=backend,
        )
        sampling_attestation_mode = "persisted-bf16-capability"
    prompt_hash = _sha256_json(prompt_token_ids)
    parent_prompt_hash = reconstruction["reconstructed_prompt_token_ids_sha256"]
    parent_prefix_hash = reconstruction["donor_prefix_token_ids_sha256"]
    seeds = _seed_schedule(
        args,
        image_id=str(args.image_id),
        parent_prompt_hash=parent_prompt_hash,
        parent_prefix_hash=parent_prefix_hash,
    )
    seed_schedule_identity = _seed_schedule_identity(
        args,
        image_id=str(args.image_id),
        parent_prompt_hash=parent_prompt_hash,
        parent_prefix_hash=parent_prefix_hash,
        selected_seeds=seeds,
    )
    runtime_evidence = _execution_contract(
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_fingerprint=generation_fingerprint,
        config_dtype=config_dtype,
        runtime_dtype=args.runtime_dtype,
        dtype_summary=dtype_summary,
        attention_implementation=actual_attention,
        policy=sampled_policy.to_artifact_dict(),
        sampling_attestation_mode=sampling_attestation_mode,
    )
    greedy_runtime_evidence = {
        **runtime_evidence,
        "decode_generation_policy": greedy_policy.to_artifact_dict(),
        "sampling_attestation_mode": "not_applicable_greedy",
    }
    donor_evidence = {
        "source_bundle": str(donor_path),
        "source_bundle_sha256": _sha256_file(donor_path),
        "source_request_id": _decode_result_payload(donor).get("request_id", donor.get("request_id")),
        "branch_source_bundle": str(branch_source_path),
        "branch_source_bundle_sha256": _sha256_file(branch_source_path),
        "parent_prefix_token_count": reconstruction["donor_prefix_token_count"],
        "parent_prefix_token_ids_sha256": reconstruction["donor_prefix_token_ids_sha256"],
        "branch_row": reconstruction["branch"],
        "donor_lineage": donor_lineage,
        "branch_lineage": branch_lineage,
        "source_image_path": str(source_image_path),
        "source_image_sha256": source_image_sha256,
        "source_image_width": int(raw.image.width),
        "source_image_height": int(raw.image.height),
    }
    prompt_evidence = {
        "donor_prompt_token_count": len(reconstruction["donor_prompt_token_ids"]),
        "prompt_token_count": len(prompt_token_ids),
        "prompt_token_ids_sha256": prompt_hash,
        "assistant_continuation_token_count": len(continuation_ids),
        "assistant_continuation_token_ids_sha256": reconstruction["assistant_continuation_token_ids_sha256"],
        "exact_donor_prompt_plus_continuation": True,
        "prompt_construction_mode": prompt_construction_mode,
    }
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    calls: list[dict[str, Any]] = []
    if not args.no_greedy_control:
        greedy_request = DecodeRequest(
            request_id=_request_id(
                image_id=str(args.image_id),
                branch_label=args.branch_label,
                prompt_hash=prompt_hash,
                kind="greedy",
                index=0,
                seed=None,
            ),
            prompt_token_ids=list(prompt_token_ids),
            model_inputs=model_inputs,
            generation_policy=greedy_policy,
        )
        result = _run_one_request(
            backend=backend,
            request=greedy_request,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_fingerprint=generation_fingerprint,
            capability=None,
        )
        bundle = _call_bundle(
            result=result,
            raw=raw,
            image_id=str(args.image_id),
            branch_label=args.branch_label,
            sampling_seed=None,
            donor_evidence=donor_evidence,
            prompt_evidence=prompt_evidence,
            runtime_evidence={**greedy_runtime_evidence, "decode_mode": "greedy", "physical_batch_size": 1},
        )
        path = output_root / "greedy-call-bundle.json"
        _write_json_once(path, bundle)
        calls.append({
            "request_id": result.request_id,
            "sampling_seed": None,
            "decode_mode": "greedy",
            "bundle_path": str(path),
            "complete_row_count": bundle["complete_row_count"],
        })
    for index, seed in enumerate(seeds):
        request = DecodeRequest(
            request_id=_request_id(
                image_id=str(args.image_id),
                branch_label=args.branch_label,
                prompt_hash=prompt_hash,
                kind="sample",
                index=index,
                seed=seed,
            ),
            prompt_token_ids=list(prompt_token_ids),
            model_inputs=model_inputs,
            generation_policy=sampled_policy,
            sampling_seed=int(seed),
        )
        result = _run_one_request(
            backend=backend,
            request=request,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_fingerprint=generation_fingerprint,
            capability=capability,
        )
        bundle = _call_bundle(
            result=result,
            raw=raw,
            image_id=str(args.image_id),
            branch_label=args.branch_label,
            sampling_seed=int(seed),
            donor_evidence=donor_evidence,
            prompt_evidence=prompt_evidence,
            runtime_evidence={**runtime_evidence, "decode_mode": "sampled", "physical_batch_size": 1},
        )
        path = output_root / f"sample-{index:04d}-seed-{int(seed)}-call-bundle.json"
        _write_json_once(path, bundle)
        calls.append({
            "request_id": result.request_id,
            "sampling_seed": int(seed),
            "decode_mode": "sampled",
            "bundle_path": str(path),
            "complete_row_count": bundle["complete_row_count"],
        })
    contract = dict(runtime_evidence)
    contract["source_donor_sha256"] = donor_evidence["source_bundle_sha256"]
    contract["branch_source_donor_sha256"] = donor_evidence["branch_source_bundle_sha256"]
    contract["prompt_token_ids_sha256"] = prompt_hash
    receipt = {
        "schema_version": f"{SCHEMA_VERSION}.shard_receipt.v1",
        "image_id": str(args.image_id),
        "branch_label": str(args.branch_label),
        "execution_contract": contract,
        "execution_contract_sha256": _sha256_json(contract),
        "donor": donor_evidence,
        "prompt": prompt_evidence,
        "sampling_seeds": [int(seed) for seed in seeds],
        "seed_schedule_identity": seed_schedule_identity,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "calls": sorted(calls, key=lambda call: str(call["request_id"])),
    }
    _write_json_once(output_root / "receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli_args(args)
    if args.mode == "run":
        receipt = run_experiment(args)
        print(json.dumps({"output_root": str(args.output_root.resolve()), "call_count": len(receipt["calls"])}, indent=2))
    elif args.mode == "check":
        result = check_shard_receipts(args.shard_receipt)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = merge_shard_receipts(args.shard_receipt)
        _write_json_once(args.merged_output.expanduser().resolve(), result)
        print(json.dumps({"merged_output": str(args.merged_output.resolve()), "call_count": result["call_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
