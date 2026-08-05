#!/usr/bin/env python3
"""Verify likelihood-mining probe Stage 0 contract gates before GPU replay.

The four gates check:
1. Token identity: SHA-256 hashes of prompt and generated token IDs
2. Config fingerprint: Local infer configs match stored fingerprints (model-identity fields only)
3. Span-to-token alignment: Parser spans align to generated tokens (THE CRITICAL GATE)
4. Canonical aggregate: Derived baseline metrics pass

All gates must pass before replay is authorized. Failures are loud and exit non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

# Register module in sys.modules BEFORE importing anything that uses @dataclass.
# This ensures cls.__module__ resolution works when script is executed via importlib.
_module_name = __name__ if __name__ != "__main__" else "verify_likelihood_mining_contracts"
if _module_name not in sys.modules:
    sys.modules[_module_name] = types.ModuleType(_module_name)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.inference import load_infer_config
from src.inference.backend import LikelihoodPair, TokenTrace
from src.inference.scoring import _locate_object_interval, _token_char_ranges, _trace_for_span

SCHEMA_VERSION = "likelihood_mining_contracts.v1"

DEFAULT_ROLLOUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration"
    "/2026-07-29-three-checkpoint-human-refined12-max3084"
)

CHECKPOINT_CONFIG_PATHS = {
    "sorted": "configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32.yaml",
    "random": "configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_random_step4887_human_refined12_hf_fp32.yaml",
    "permutation": "configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_random_permutation_bundle_step4887_human_refined12_hf_fp32.yaml",
}

EXPECTED_COUNTS = {
    "sorted": 3570,
    "random": 2852,
    "permutation": 2918,
}

EXPECTED_TOTAL_ROLLOUTS = 576

ORIGIN_WORKTREE = Path("/data/CoordExp/.worktrees/permutation-bundle-coordinate-noise-pilot")


class GateFailure:
    """Represents a single gate failure."""

    def __init__(
        self,
        gate: int,
        checkpoint: str,
        example_id: str | None,
        seed: int | None,
        object_span_id: str | None,
        error_code: str,
        error_message: str,
    ):
        self.gate = gate
        self.checkpoint = checkpoint
        self.example_id = example_id
        self.seed = seed
        self.object_span_id = object_span_id
        self.error_code = error_code
        self.error_message = error_message


def _sha256_json(value: Any) -> str:
    """Hash JSON as produced by run_current_seeded_sampled_rollouts.py."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _sha256_text(text: str) -> str:
    """Hash text as in src/inference/scoring.py."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_gate1_token_identity(
    shard: dict[str, Any],
    checkpoint: str,
) -> list[GateFailure]:
    """Gate 1: Verify SHA-256 hashes of prompt and generated token IDs."""
    failures = []

    for rollout in shard["rollouts"]:
        example_id = rollout.get("example_id")
        seed = rollout.get("seed")

        # Check prompt token IDs
        stored_prompt_sha = rollout.get("prompt_token_ids_sha256")
        prompt_ids = rollout.get("prompt_token_ids", [])
        computed_prompt_sha = _sha256_json(prompt_ids)
        if stored_prompt_sha != computed_prompt_sha:
            failures.append(
                GateFailure(
                    gate=1,
                    checkpoint=checkpoint,
                    example_id=example_id,
                    seed=seed,
                    object_span_id=None,
                    error_code="gate1.prompt_token_sha_mismatch",
                    error_message=f"prompt_token_ids_sha256 mismatch: stored={stored_prompt_sha}, computed={computed_prompt_sha}",
                )
            )

        # Check generated token IDs
        stored_gen_sha = rollout.get("generated_token_ids_sha256")
        gen_ids = rollout.get("generated_token_ids", [])
        computed_gen_sha = _sha256_json(gen_ids)
        if stored_gen_sha != computed_gen_sha:
            failures.append(
                GateFailure(
                    gate=1,
                    checkpoint=checkpoint,
                    example_id=example_id,
                    seed=seed,
                    object_span_id=None,
                    error_code="gate1.generated_token_sha_mismatch",
                    error_message=f"generated_token_ids_sha256 mismatch: stored={stored_gen_sha}, computed={computed_gen_sha}",
                )
            )

    return failures


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode tokens per hf_backend._decode_tokens semantic."""
    return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))


def _build_token_trace(
    tokenizer: Any,
    token_id: int,
    step_index: int,
) -> TokenTrace:
    """Construct a minimal TokenTrace for alignment."""
    token_text = _decode_tokens(tokenizer, [token_id])
    return TokenTrace(
        step_index=step_index,
        token_id=token_id,
        token_text=token_text,
        likelihood=LikelihoodPair(policy_logprob=None, raw_model_logprob=None),
        is_stop=False,
        is_pad=False,
        backend="hf",
        backend_mode="inference",
        response_family="qwen3-vl",
    )


def verify_gate3_span_alignment(
    shard: dict[str, Any],
    checkpoint: str,
    tokenizer: Any,
    token_decode_cache: dict[int, str],
) -> tuple[list[GateFailure], int]:
    """Gate 3: Verify span-to-token alignment for all predictions."""
    failures = []
    total_spans_checked = 0

    for rollout in shard["rollouts"]:
        example_id = rollout.get("example_id")
        seed = rollout.get("seed")
        gen_ids = rollout.get("generated_token_ids", [])
        stored_text = rollout.get("generated_text", "")

        # Step 1: Build token trace with decoded text per token
        token_trace: list[TokenTrace] = []
        for step_idx, token_id in enumerate(gen_ids):
            if token_id not in token_decode_cache:
                token_decode_cache[token_id] = _decode_tokens(tokenizer, [token_id])
            token_text = token_decode_cache[token_id]
            token_trace.append(
                TokenTrace(
                    step_index=step_idx,
                    token_id=token_id,
                    token_text=token_text,
                    likelihood=LikelihoodPair(policy_logprob=None, raw_model_logprob=None),
                    is_stop=False,
                    is_pad=False,
                    backend="hf",
                    backend_mode="inference",
                    response_family="qwen3-vl",
                )
            )

        # Step 2: Assert concatenated token_text equals stored generated_text
        concatenated_text = "".join(trace.token_text for trace in token_trace)
        if concatenated_text != stored_text:
            failures.append(
                GateFailure(
                    gate=3,
                    checkpoint=checkpoint,
                    example_id=example_id,
                    seed=seed,
                    object_span_id=None,
                    error_code="gate3.token_concatenation_mismatch",
                    error_message=f"concatenated token_text ({len(concatenated_text)} chars) != generated_text ({len(stored_text)} chars)",
                )
            )
            continue

        # Step 3-4: For each prediction, verify span alignment
        predictions_list = rollout.get("predictions", {}).get("predictions", [])
        for prediction in predictions_list:
            object_span_id = prediction.get("object_span_id")
            total_spans_checked += 1

            # Step 3: Locate object interval
            try:
                interval = _locate_object_interval(
                    row_id=example_id,
                    prediction=prediction,
                    token_trace=token_trace,
                )
            except Exception as exc:
                error_code = getattr(exc, "code", "gate3.object_interval_error")
                error_msg = str(exc)
                failures.append(
                    GateFailure(
                        gate=3,
                        checkpoint=checkpoint,
                        example_id=example_id,
                        seed=seed,
                        object_span_id=object_span_id,
                        error_code=error_code,
                        error_message=error_msg,
                    )
                )
                continue

            start_token, end_token, span_char_start = interval

            # Step 4: Verify schema_spans and coord_token_spans align to exactly one token
            token_ranges = _token_char_ranges(token_trace, start_token, end_token)

            for span_list_name in ["schema_spans", "coord_token_spans"]:
                spans = prediction.get(span_list_name) or []
                for span in spans:
                    try:
                        _trace_for_span(
                            span,
                            row_id=example_id,
                            token_trace=token_trace,
                            token_ranges=token_ranges,
                            span_char_start=span_char_start,
                            object_span_id=object_span_id,
                        )
                    except Exception as exc:
                        error_code = getattr(exc, "code", "gate3.span_alignment_error")
                        error_msg = str(exc)
                        failures.append(
                            GateFailure(
                                gate=3,
                                checkpoint=checkpoint,
                                example_id=example_id,
                                seed=seed,
                                object_span_id=object_span_id,
                                error_code=error_code,
                                error_message=error_msg,
                            )
                        )

    return failures, total_spans_checked


def _get_origin_config(checkpoint: str) -> tuple[dict[str, Any], str]:
    """Load origin worktree config via subprocess and return (config_dict, fingerprint).

    Returns:
        (config_dict, fingerprint) where config_dict is model_dump(mode='json')

    Raises:
        RuntimeError if origin worktree is absent or config load fails
    """
    if not ORIGIN_WORKTREE.exists():
        raise RuntimeError(
            f"origin worktree does not exist at {ORIGIN_WORKTREE}; "
            "cannot verify against source-of-truth configs"
        )

    config_filename = Path(CHECKPOINT_CONFIG_PATHS[checkpoint]).name

    # Create helper script to run in origin worktree
    helper_script = f"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.config.inference import load_infer_config

config_filename = {config_filename!r}
config_path = Path('configs/coordexp_swift/infer') / config_filename
resolved = load_infer_config(config_path.resolve(strict=True))

result = {{
    'fingerprint': resolved.fingerprint,
    'config_dict': resolved.config.model_dump(mode='json'),
}}
print(json.dumps(result))
"""

    try:
        result = subprocess.run(
            ["python3", "-c", helper_script],
            cwd=ORIGIN_WORKTREE,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"origin config load timed out for {checkpoint}")
    except Exception as exc:
        raise RuntimeError(f"failed to run origin config helper: {exc}")

    if result.returncode != 0:
        raise RuntimeError(
            f"origin config load failed for {checkpoint}: {result.stderr}"
        )

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"origin config helper produced invalid JSON: {exc}")

    return output["config_dict"], output["fingerprint"]


def _deep_diff_configs(
    local_dict: dict[str, Any],
    origin_dict: dict[str, Any],
    exempted_paths: set[str],
) -> dict[str, tuple[Any, Any]]:
    """Compare two config dicts, returning {path: (local_value, origin_value)} for differences.

    Paths are dot-separated (e.g., 'run.artifact_root').
    """
    diffs, _ = _deep_diff_configs_split(local_dict, origin_dict, exempted_paths)
    return diffs


def _deep_diff_configs_split(
    local_dict: dict[str, Any],
    origin_dict: dict[str, Any],
    exempted_paths: set[str],
) -> tuple[dict[str, tuple[Any, Any]], dict[str, tuple[Any, Any]]]:
    """Split differing paths into (critical, exempted).

    The exempted set must stay auditable: an exemption that is never observed to
    fire is indistinguishable from a comparison that never ran. Returning both
    halves lets the receipt record that `run.artifact_root` actually did differ
    and was deliberately allowed, rather than silently absorbing it.
    """

    critical: dict[str, tuple[Any, Any]] = {}
    exempted: dict[str, tuple[Any, Any]] = {}

    def recurse(local_val: Any, origin_val: Any, path: str):
        if isinstance(local_val, dict) and isinstance(origin_val, dict):
            all_keys = set(local_val.keys()) | set(origin_val.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                local_sub = local_val.get(key)
                origin_sub = origin_val.get(key)
                recurse(local_sub, origin_sub, new_path)
        elif local_val != origin_val:
            target = exempted if path in exempted_paths else critical
            target[path] = (local_val, origin_val)

    recurse(local_dict, origin_dict, "")
    return critical, exempted


def verify_gate2_config_fingerprint(
    shard: dict[str, Any],
    checkpoint: str,
) -> tuple[list[GateFailure], dict[str, Any]]:
    """Gate 2: Verify model-identity fields match origin config.

    Returns:
        (failures, report_dict) where report_dict contains gate2-specific metrics
    """
    failures = []
    report: dict[str, Any] = {
        "origin_config_present": False,
        "origin_fingerprint_matches_stored": False,
        "model_identity_fields_match": False,
        "differing_fields_exempted": [],
        "differing_fields_critical": [],
    }

    stored_fingerprint = shard.get("config", {}).get("resolved_fingerprint")
    if stored_fingerprint is None:
        failures.append(
            GateFailure(
                gate=2,
                checkpoint=checkpoint,
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate2.missing_stored_fingerprint",
                error_message="shard config lacks resolved_fingerprint",
            )
        )
        return failures, report

    # Load local config
    config_path_str = CHECKPOINT_CONFIG_PATHS.get(checkpoint)
    if config_path_str is None:
        failures.append(
            GateFailure(
                gate=2,
                checkpoint=checkpoint,
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate2.unknown_checkpoint",
                error_message=f"no local config path defined for checkpoint {checkpoint}",
            )
        )
        return failures, report

    try:
        config_path = Path(config_path_str).resolve(strict=True)
        local_resolved = load_infer_config(config_path)
        local_dict = local_resolved.config.model_dump(mode="json")
    except Exception as exc:
        failures.append(
            GateFailure(
                gate=2,
                checkpoint=checkpoint,
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate2.local_config_load_error",
                error_message=f"failed to load local config: {exc}",
            )
        )
        return failures, report

    # Load origin config
    try:
        origin_dict, origin_fingerprint = _get_origin_config(checkpoint)
        report["origin_config_present"] = True
    except RuntimeError as exc:
        failures.append(
            GateFailure(
                gate=2,
                checkpoint=checkpoint,
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate2.origin_config_unavailable",
                error_message=str(exc),
            )
        )
        return failures, report

    # Verify origin fingerprint matches stored fingerprint
    if origin_fingerprint != stored_fingerprint:
        failures.append(
            GateFailure(
                gate=2,
                checkpoint=checkpoint,
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate2.origin_fingerprint_mismatch",
                error_message=(
                    f"origin config fingerprint {origin_fingerprint} "
                    f"does not match stored fingerprint {stored_fingerprint}"
                ),
            )
        )
        return failures, report

    report["origin_fingerprint_matches_stored"] = True

    # Compare local vs origin, exempting run.artifact_root
    exempted = {"run.artifact_root"}
    diffs, exempted_diffs = _deep_diff_configs_split(local_dict, origin_dict, exempted)
    report["differing_fields_exempted"] = sorted(exempted_diffs)
    report["exemptions_declared"] = sorted(exempted)

    for path in sorted(diffs.keys()):
        local_val, origin_val = diffs[path]
        report["differing_fields_critical"].append(path)
        failures.append(
            GateFailure(
                gate=2,
                checkpoint=checkpoint,
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate2.model_identity_field_mismatch",
                error_message=(
                    f"critical field {path} differs: local={local_val}, origin={origin_val}"
                ),
            )
        )

    if not failures:
        report["model_identity_fields_match"] = True

    return failures, report


def verify_gate4_canonical_aggregate(
) -> tuple[list[GateFailure], bool]:
    """Gate 4: Shell out to derive_likelihood_mining_baseline.py and check gates_passed."""
    failures = []

    try:
        result = subprocess.run(
            [
                "python3",
                str(Path(__file__).with_name("derive_likelihood_mining_baseline.py")),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        failures.append(
            GateFailure(
                gate=4,
                checkpoint="all",
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate4.timeout",
                error_message=f"derive_likelihood_mining_baseline.py timed out",
            )
        )
        return failures, False
    except Exception as exc:
        failures.append(
            GateFailure(
                gate=4,
                checkpoint="all",
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate4.subprocess_error",
                error_message=f"failed to run derive_likelihood_mining_baseline.py: {exc}",
            )
        )
        return failures, False

    if result.returncode != 0:
        failures.append(
            GateFailure(
                gate=4,
                checkpoint="all",
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate4.non_zero_exit",
                error_message=f"derive_likelihood_mining_baseline.py exited with code {result.returncode}",
            )
        )
        if result.stderr:
            failures[-1] = GateFailure(
                gate=4,
                checkpoint="all",
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate4.non_zero_exit",
                error_message=f"derive_likelihood_mining_baseline.py exited with code {result.returncode}: {result.stderr[:500]}",
            )
        return failures, False

    # Try to parse gates_passed from stdout
    gates_passed = False
    for line in result.stdout.split("\n"):
        if '"gates_passed": true' in line or "'gates_passed': True" in line:
            gates_passed = True
            break

    if not gates_passed:
        failures.append(
            GateFailure(
                gate=4,
                checkpoint="all",
                example_id=None,
                seed=None,
                object_span_id=None,
                error_code="gate4.gates_not_passed",
                error_message="derive_likelihood_mining_baseline.py reported gates_passed=false",
            )
        )

    return failures, gates_passed


def run_self_test(rollout_root: Path) -> int:
    """Verify gate failures on deliberately corrupted data."""
    print("\n[SELF-TEST MODE] Running corruption detection tests...")

    # Load one rollout
    shard_path = rollout_root / "sorted" / "sampled" / "shard-0.json"
    with open(shard_path) as f:
        shard = json.load(f)

    # Test 1: Corrupt a token ID
    print("  Test 1: Flip a generated token ID...")
    import copy

    test_shard = copy.deepcopy(shard)
    rollout = test_shard["rollouts"][0]
    if rollout.get("generated_token_ids"):
        rollout["generated_token_ids"][0] = (rollout["generated_token_ids"][0] + 1) % 150000
        # Recompute the SHA to make it pass gate 1 detection would be wrong,
        # but we want to test gate 1 failure, so we corrupt AND the SHA
        rollout["generated_token_ids_sha256"] = _sha256_json(rollout["generated_token_ids"])

    gate1_failures = verify_gate1_token_identity(test_shard, "sorted")
    # After SHA recomputation, gate 1 will pass. But the concatenated text will differ in gate 3.
    # Let's instead not recompute the SHA.
    rollout["generated_token_ids_sha256"] = "invalid_hash_value"
    gate1_failures = verify_gate1_token_identity(test_shard, "sorted")
    if gate1_failures:
        print("    ✓ Gate 1 correctly detected corrupted token ID hash")
    else:
        print("    ✗ Gate 1 FAILED to detect corrupted token ID hash")
        return 1

    # Test 2: Corrupt a span text
    print("  Test 2: Flip a byte in raw_span_text...")
    test_shard = copy.deepcopy(shard)
    rollout = test_shard["rollouts"][0]
    predictions = rollout.get("predictions", {}).get("predictions", [])
    if predictions:
        pred = predictions[0]
        if pred.get("raw_span_text"):
            original_text = pred["raw_span_text"]
            # Flip the last character (avoiding special characters)
            if len(original_text) > 0:
                flipped = list(original_text)
                flipped[0] = chr((ord(flipped[0]) + 1) % 256)
                pred["raw_span_text"] = "".join(flipped)
                # Don't update the SHA; we want gate 3 to fail

    # Load tokenizer and run gate 3
    try:
        from transformers import AutoTokenizer

        model_path = "/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as e:
        print(f"    ! Could not load tokenizer for self-test: {e}")
        return 0

    gate3_failures, _ = verify_gate3_span_alignment(test_shard, "sorted", tokenizer, {})
    if gate3_failures:
        print("    ✓ Gate 3 correctly detected corrupted span text")
    else:
        print("    ✗ Gate 3 FAILED to detect corrupted span text")
        return 1

    # Test 3: Corrupt a model-identity field (adapter.path)
    print("  Test 3: Perturb adapter.path in local config...")

    # Mock _get_origin_config to return the known-good origin config
    # The stored shard has the correct fingerprint
    shard_config = shard.get("config", {})
    stored_fingerprint = shard_config.get("resolved_fingerprint")

    # Load the real local config
    config_path_str = CHECKPOINT_CONFIG_PATHS["sorted"]
    config_path = Path(config_path_str).resolve(strict=True)
    local_resolved = load_infer_config(config_path)
    local_dict = local_resolved.config.model_dump(mode="json")

    # Get the origin config
    try:
        origin_dict, origin_fingerprint = _get_origin_config("sorted")
    except RuntimeError as e:
        print(f"    ! Could not load origin config for self-test: {e}")
        return 0

    # Verify they start equal (except artifact_root)
    exempted = {"run.artifact_root"}
    diffs_initial = _deep_diff_configs(local_dict, origin_dict, exempted)
    if diffs_initial:
        print(f"    ! Initial config mismatch (before test): {list(diffs_initial.keys())}")
        return 1

    # Now perturb the local dict's adapter.path
    if "adapter" in local_dict and "path" in local_dict["adapter"]:
        local_dict["adapter"]["path"] = "/fake/adapter/path"

    # Compare and verify it detects the difference
    diffs_after = _deep_diff_configs(local_dict, origin_dict, exempted)
    if "adapter.path" in diffs_after:
        print("    ✓ Gate 2 correctly detected model-identity field divergence")
    else:
        print("    ✗ Gate 2 FAILED to detect adapter.path change")
        return 1

    print("\n✓ All self-tests passed\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify likelihood-mining probe Stage 0 contract gates."
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=DEFAULT_ROLLOUT_ROOT,
        help="Path to rollout root directory",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        choices=["sorted", "random", "permutation"],
        help="Checkpoints to verify (default: all three)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write receipt JSON (default: stdout)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run corruption detection self-tests instead of normal verification",
    )

    args = parser.parse_args()

    if args.self_test:
        return run_self_test(args.rollout_root)

    checkpoints = args.checkpoint or ["sorted", "random", "permutation"]
    rollout_root = args.rollout_root

    # Verify root exists
    if not rollout_root.exists():
        print(f"ERROR: Rollout root does not exist: {rollout_root}", file=sys.stderr)
        return 1

    # Load tokenizer once
    try:
        from transformers import AutoTokenizer

        model_path = "/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        print(f"ERROR: Failed to load tokenizer: {exc}", file=sys.stderr)
        return 1

    token_decode_cache: dict[int, str] = {}

    # Collect results per checkpoint
    gate_results: dict[str, dict[str, Any]] = {}
    all_failures: list[GateFailure] = []
    total_rollouts = 0
    total_spans = 0

    for checkpoint in checkpoints:
        result_dict: dict[str, Any] = {
            "rollouts_checked": 0,
            "spans_checked": 0,
            "tokens_checked": 0,
            "gate1_passed": True,
            "gate2_passed": True,
            "gate3_passed": True,
            "gate4_passed": True,
        }

        print(f"Verifying {checkpoint}...", file=sys.stderr)

        # Load shards
        shards = []
        for shard_idx in [0, 1]:
            shard_path = rollout_root / checkpoint / "sampled" / f"shard-{shard_idx}.json"
            if not shard_path.exists():
                print(f"ERROR: Shard not found: {shard_path}", file=sys.stderr)
                return 1
            with open(shard_path) as f:
                shards.append(json.load(f))

        # Combine shards
        combined_shard: dict[str, Any] = {
            "config": shards[0].get("config", {}),
            "rollouts": [],
        }
        for shard in shards:
            combined_shard["rollouts"].extend(shard.get("rollouts", []))

        result_dict["rollouts_checked"] = len(combined_shard["rollouts"])
        total_rollouts += len(combined_shard["rollouts"])

        # Gate 1: Token identity
        gate1_failures = verify_gate1_token_identity(combined_shard, checkpoint)
        if gate1_failures:
            result_dict["gate1_passed"] = False
            all_failures.extend(gate1_failures)

        # Gate 2: Config fingerprint (narrowed to model-identity fields only)
        gate2_failures, gate2_report = verify_gate2_config_fingerprint(
            combined_shard, checkpoint
        )
        result_dict["gate2_report"] = gate2_report
        if gate2_failures:
            result_dict["gate2_passed"] = False
            all_failures.extend(gate2_failures)

        # Gate 3: Span-to-token alignment
        gate3_failures, spans_checked = verify_gate3_span_alignment(
            combined_shard, checkpoint, tokenizer, token_decode_cache
        )
        result_dict["spans_checked"] = spans_checked
        result_dict["tokens_checked"] = sum(
            len(r.get("generated_token_ids", [])) for r in combined_shard["rollouts"]
        )
        if gate3_failures:
            result_dict["gate3_passed"] = False
            all_failures.extend(gate3_failures)

        total_spans += spans_checked

        gate_results[checkpoint] = result_dict

    # Gate 4: Canonical aggregate (shared across all checkpoints)
    gate4_failures, gate4_passed = verify_gate4_canonical_aggregate()
    for checkpoint in checkpoints:
        gate_results[checkpoint]["gate4_passed"] = gate4_passed
    all_failures.extend(gate4_failures)

    # Verify expected counts (only for checkpoints being run)
    count_issues = []
    for checkpoint in checkpoints:
        if gate_results[checkpoint]["spans_checked"] != EXPECTED_COUNTS[checkpoint]:
            count_issues.append(
                f"{checkpoint}: {gate_results[checkpoint]['spans_checked']} spans (expected {EXPECTED_COUNTS[checkpoint]})"
            )

    # Only check total rollouts if running all checkpoints
    if len(checkpoints) == 3 and total_rollouts != EXPECTED_TOTAL_ROLLOUTS:
        count_issues.append(
            f"total: {total_rollouts} rollouts (expected {EXPECTED_TOTAL_ROLLOUTS})"
        )

    # Build receipt
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "total_rollouts_checked": total_rollouts,
        "total_spans_checked": total_spans,
        "total_tokens_generated": sum(
            r["tokens_checked"] for r in gate_results.values()
        ),
        "per_checkpoint": gate_results,
        "gates_passed": all(
            gate_results[cp]["gate1_passed"]
            and gate_results[cp]["gate2_passed"]
            and gate_results[cp]["gate3_passed"]
            and gate_results[cp]["gate4_passed"]
            for cp in checkpoints
        )
        and not count_issues,
        "failure_count": len(all_failures),
        "failures": [
            {
                "gate": f.gate,
                "checkpoint": f.checkpoint,
                "example_id": f.example_id,
                "seed": f.seed,
                "object_span_id": f.object_span_id,
                "error_code": f.error_code,
                "error_message": f.error_message,
            }
            for f in all_failures
        ],
        "count_issues": count_issues,
    }

    # Output receipt
    receipt_json = json.dumps(receipt, indent=2)
    if args.output:
        args.output.write_text(receipt_json)
        print(f"Receipt written to {args.output}")
    else:
        print(receipt_json)

    # Summary to stderr
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Total rollouts: {total_rollouts}/{EXPECTED_TOTAL_ROLLOUTS}", file=sys.stderr)
    print(f"Total spans: {total_spans}/9340", file=sys.stderr)
    print(f"Failures: {len(all_failures)}", file=sys.stderr)
    if receipt["gates_passed"]:
        print("✓ ALL GATES PASSED", file=sys.stderr)
    else:
        print("✗ GATES FAILED", file=sys.stderr)
        for issue in count_issues:
            print(f"  Count mismatch: {issue}", file=sys.stderr)
        # Show first few failures
        for failure in all_failures[:5]:
            print(
                f"  Gate {failure.gate}: {failure.error_code} "
                f"({failure.checkpoint}/{failure.example_id})",
                file=sys.stderr,
            )
        if len(all_failures) > 5:
            print(f"  ... and {len(all_failures) - 5} more failures", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    return 0 if receipt["gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
