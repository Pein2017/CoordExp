#!/usr/bin/env python3
"""
Test script to verify coordinate tokens are correctly added and tokenized as single tokens.

This script specifically checks:
1. Coordinate tokens exist in the vocabulary
2. Each <|coord_i|> token is encoded as a SINGLE token (not multiple tokens)
3. Round-trip encoding/decoding preserves coordinate tokens

Usage:
    python scripts/test_coord_tokenization.py --checkpoint model_cache/Qwen3-VL-8B-Instruct-coordexp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test coordinate token tokenization (single-token encoding verification)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_cache/Qwen3-VL-8B-Instruct-coordexp"),
        help="Path to the checkpoint directory with expanded vocabulary.",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Optional: Path to base checkpoint (without coord tokens) for comparison.",
    )
    return parser.parse_args()


def test_single_token_encoding(tokenizer, token: str) -> tuple[bool, list[int], str]:
    """
    Test if a token is encoded as a single token.

    Returns:
        (is_single_token, token_ids, decoded_text)
    """
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids)
    is_single = len(token_ids) == 1

    return is_single, token_ids, decoded


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"[✗] Checkpoint directory does not exist: {args.checkpoint}")
        return

    print("=" * 80)
    print("COORDINATE TOKEN TOKENIZATION TEST")
    print("=" * 80)
    print(f"\n[+] Loading tokenizer from {args.checkpoint}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, trust_remote_code=True
        )
    except Exception as e:
        print(f"[✗] Failed to load tokenizer: {e}")
        return

    vocab_size = len(tokenizer)
    print(f"[+] Vocabulary size: {vocab_size}")

    # Check for coord_tokens.json
    tokens_path = args.checkpoint / "coord_tokens.json"
    expected_tokens = None
    if tokens_path.exists():
        print(f"[+] Found coord_tokens.json at {tokens_path}")
        with tokens_path.open("r", encoding="utf-8") as f:
            expected_tokens = json.load(f)
        print(f"[+] Expected {len(expected_tokens)} coordinate tokens")
    else:
        print("[!] coord_tokens.json not found, will test common coordinate tokens")

    # Build test tokens
    if expected_tokens:
        test_tokens = expected_tokens
    else:
        # Default test tokens
        test_tokens = [f"<|coord_{i}|>" for i in range(0, 1000)]
        if "<|coord_*|>" not in test_tokens:
            test_tokens = ["<|coord_*|>"] + test_tokens

    print(f"\n[+] Testing {len(test_tokens)} coordinate tokens...")

    # Test 1: Verify tokens exist in vocabulary
    print("\n" + "=" * 80)
    print("TEST 1: Vocabulary Presence Check")
    print("=" * 80)

    missing_tokens = []
    found_tokens = []

    for token in test_tokens:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.convert_tokens_to_ids(token)
            found_tokens.append((token, token_id))
        else:
            missing_tokens.append(token)

    if missing_tokens:
        print(f"[✗] {len(missing_tokens)} tokens are missing from vocabulary")
        print(f"    First 10 missing: {missing_tokens[:10]}")
        if len(missing_tokens) > 10:
            print(f"    ... and {len(missing_tokens) - 10} more")
    else:
        print(f"[✓] All {len(test_tokens)} tokens are present in vocabulary")

    if found_tokens:
        print("\n[+] Sample tokens in vocabulary:")
        for token, token_id in found_tokens[:5]:
            print(f"    {token} -> ID {token_id}")
        if len(found_tokens) > 5:
            print(f"    ... and {len(found_tokens) - 5} more")

    # Test 2: Single-token encoding verification (CRITICAL TEST)
    print("\n" + "=" * 80)
    print("TEST 2: Single-Token Encoding Verification (CRITICAL)")
    print("=" * 80)
    print(
        "This test verifies that each <|coord_i|> token is encoded as a SINGLE token."
    )
    print(
        "If a token is split into multiple tokens, it indicates a tokenization issue.\n"
    )

    # Test a sample of tokens (not all to avoid too much output)
    sample_indices = (
        [0, 1, 10, 100, 500, 999, 1000]
        if len(test_tokens) > 1000
        else range(len(test_tokens))
    )
    if "<|coord_*|>" in test_tokens:
        sample_tokens = ["<|coord_*|>"] + [
            test_tokens[i] for i in sample_indices if i < len(test_tokens)
        ]
    else:
        sample_tokens = [test_tokens[i] for i in sample_indices if i < len(test_tokens)]

    # Also test a few random tokens
    import random

    if len(test_tokens) > 20:
        random.seed(42)
        random_indices = random.sample(
            range(len(test_tokens)), min(10, len(test_tokens) - len(sample_tokens))
        )
        sample_tokens.extend([test_tokens[i] for i in random_indices])

    single_token_count = 0
    multi_token_count = 0
    multi_token_examples = []

    print(
        f"[+] Testing {len(sample_tokens)} sample tokens for single-token encoding...\n"
    )

    for token in sample_tokens:
        if token not in tokenizer.get_vocab():
            continue

        is_single, token_ids, decoded = test_single_token_encoding(tokenizer, token)

        if is_single:
            single_token_count += 1
            token_id = token_ids[0]
            # Verify round-trip
            if decoded.strip() == token:
                status = "✓"
            else:
                status = "⚠"
                print(f"  {status} {token}: ID {token_id} (decoded: '{decoded}')")
        else:
            multi_token_count += 1
            multi_token_examples.append((token, token_ids, decoded))
            print(f"  ✗ {token}: Split into {len(token_ids)} tokens: {token_ids}")
            print(f"      Decoded: '{decoded}'")

    print("\n[Results]")
    print(f"  Single-token encoding: {single_token_count}/{len(sample_tokens)} tokens")
    print(f"  Multi-token encoding: {multi_token_count}/{len(sample_tokens)} tokens")

    if multi_token_examples:
        print(
            f"\n[✗] FAILURE: {len(multi_token_examples)} tokens are split into multiple tokens!"
        )
        print("    This indicates a tokenization problem.")
        print("    Examples:")
        for token, token_ids, decoded in multi_token_examples[:5]:
            print(f"      {token} -> {len(token_ids)} tokens: {token_ids}")
    else:
        print("\n[✓] SUCCESS: All tested tokens are encoded as single tokens!")

    # Test 3: Test all tokens in vocabulary (if not too many)
    if len(found_tokens) <= 1000:
        print("\n" + "=" * 80)
        print("TEST 3: Comprehensive Single-Token Check (All Tokens)")
        print("=" * 80)

        all_single = 0
        all_multi = 0
        all_multi_examples = []

        for token, token_id in found_tokens:
            is_single, token_ids, decoded = test_single_token_encoding(tokenizer, token)
            if is_single:
                all_single += 1
            else:
                all_multi += 1
                all_multi_examples.append((token, token_ids))

        print("[Results]")
        print(f"  Single-token encoding: {all_single}/{len(found_tokens)} tokens")
        print(f"  Multi-token encoding: {all_multi}/{len(found_tokens)} tokens")

        if all_multi_examples:
            print(
                f"\n[✗] FAILURE: {len(all_multi_examples)} tokens are split into multiple tokens!"
            )
            print("    First 10 examples:")
            for token, token_ids in all_multi_examples[:10]:
                print(f"      {token} -> {len(token_ids)} tokens: {token_ids}")
        else:
            print(
                f"\n[✓] SUCCESS: All {len(found_tokens)} tokens are encoded as single tokens!"
            )

    # Test 4: Test in context (tokens in a sentence)
    print("\n" + "=" * 80)
    print("TEST 4: Context Encoding Test")
    print("=" * 80)
    print("Testing coordinate tokens when embedded in text sequences.\n")

    context_tests = [
        "The box is at <|coord_100|> and <|coord_500|>.",
        "Coordinates: <|coord_0|>, <|coord_250|>, <|coord_999|>",
        "Wildcard: <|coord_*|> represents any coordinate.",
    ]

    all_context_passed = True
    for test_text in context_tests:
        token_ids = tokenizer.encode(test_text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids)

        # Extract coordinate tokens from original text
        import re

        coord_tokens_in_text = re.findall(r"<\|coord_[*0-9]+\|>", test_text)

        print(f"  Original: {test_text}")
        print(f"  Encoded: {len(token_ids)} token IDs")
        print(f"  Decoded: {decoded}")

        # Check if all coord tokens are preserved
        preserved = all(token in decoded for token in coord_tokens_in_text)
        if preserved:
            print("  [✓] All coordinate tokens preserved")
        else:
            print("  [✗] Some coordinate tokens lost in round-trip")
            all_context_passed = False

        # Check if coord tokens appear as single tokens in the sequence
        # This is a heuristic: if the token count is reasonable, coord tokens are likely single
        expected_min_tokens = len(test_text.split())  # Rough estimate
        if len(token_ids) <= expected_min_tokens + len(coord_tokens_in_text) * 2:
            print("  [✓] Token count reasonable (coord tokens likely single)")
        else:
            print(
                "  [⚠] Token count higher than expected (coord tokens might be split)"
            )
            all_context_passed = False
        print()

    if all_context_passed:
        print("[✓] All context encoding tests passed")
    else:
        print("[✗] Some context encoding tests failed")

    # Test 5: Compare with base model (if provided)
    if args.base_checkpoint and args.base_checkpoint.exists():
        print("\n" + "=" * 80)
        print("TEST 5: Comparison with Base Model")
        print("=" * 80)

        try:
            base_tokenizer = AutoTokenizer.from_pretrained(
                args.base_checkpoint, trust_remote_code=True
            )
            print(f"[+] Loaded base tokenizer (vocab size: {len(base_tokenizer)})")

            # Test how base model tokenizes coordinate tokens (should be multiple tokens)
            test_token = "<|coord_100|>"
            base_ids = base_tokenizer.encode(test_token, add_special_tokens=False)
            coord_ids = tokenizer.encode(test_token, add_special_tokens=False)

            print(f"\n[Comparison] Token: {test_token}")
            print(f"  Base model: {len(base_ids)} tokens -> {base_ids}")
            print(f"  Coord model: {len(coord_ids)} tokens -> {coord_ids}")

            if len(base_ids) > 1 and len(coord_ids) == 1:
                print(
                    f"  [✓] Base model splits token ({len(base_ids)} tokens), coord model uses single token"
                )
            elif len(base_ids) == 1:
                print("  [⚠] Base model also uses single token (unexpected)")
            else:
                print(
                    "  [✗] Both models split token (coord model should use single token)"
                )
        except Exception as e:
            print(f"[!] Could not load base tokenizer: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if missing_tokens:
        print(f"[✗] Vocabulary check: {len(missing_tokens)} tokens missing")
    else:
        print("[✓] Vocabulary check: All tokens present")

    if multi_token_count > 0 or (len(found_tokens) <= 1000 and all_multi > 0):
        print("[✗] Single-token encoding: FAILED (some tokens are split)")
        print("    This is a CRITICAL issue - coordinate tokens must be single tokens!")
    else:
        print("[✓] Single-token encoding: PASSED (all tokens are single tokens)")

    if all_context_passed:
        print("[✓] Context encoding: PASSED")
    else:
        print("[✗] Context encoding: FAILED")

    print("\n[✓] Test complete")


if __name__ == "__main__":
    main()
