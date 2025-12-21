"""
Verification script to check if coordinate tokens were successfully added to the vocabulary.

Usage:
    python scripts/verify_coord_vocab.py [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify coordinate tokens in expanded vocabulary."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct-coordexp"
        ),
        help="Path to the expanded checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"[✗] Checkpoint directory does not exist: {args.checkpoint}")
        return

    print(f"[+] Loading tokenizer from {args.checkpoint}")
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
    if tokens_path.exists():
        print(f"[+] Found coord_tokens.json at {tokens_path}")
        with tokens_path.open("r", encoding="utf-8") as f:
            expected_tokens = json.load(f)
        print(f"[+] Expected {len(expected_tokens)} coordinate tokens")
        print(f"[+] First 5 tokens: {expected_tokens[:5]}")
        print(f"[+] Last 5 tokens: {expected_tokens[-5:]}")
    else:
        print(f"[!] coord_tokens.json not found at {tokens_path}")
        expected_tokens = None

    # Check for wildcard token
    wildcard_token = "<|coord_*|>"
    if wildcard_token in tokenizer.get_vocab():
        token_id = tokenizer.convert_tokens_to_ids(wildcard_token)
        print(
            f"[✓] Wildcard token '{wildcard_token}' found in vocabulary (ID: {token_id})"
        )
    else:
        print(f"[✗] Wildcard token '{wildcard_token}' NOT found in vocabulary")

    # Check for critical coordinate tokens (0–999 is the canonical experiment range;
    # coord_1000 is optional and may be absent if we only add 1000 bins.)
    sample_tokens = [
        "<|coord_0|>",  # Critical: left/top edge (pixel = 0)
        "<|coord_1|>",
        "<|coord_100|>",
        "<|coord_500|>",
        "<|coord_999|>",  # Canonical far edge bin in our experiments
    ]
    optional_tokens = [
        "<|coord_1000|>",  # Optional: right/bottom edge bin if 1001 tokens are added
    ]
    found_count = 0
    for token in sample_tokens + optional_tokens:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"[✓] Token '{token}' found (ID: {token_id})")
            found_count += 1
        else:
            prefix = "[!]" if token in optional_tokens else "[✗]"
            print(f"{prefix} Token '{token}' NOT found")

    # Verify range completeness
    print("\n[+] Checking coordinate token range completeness...")
    coord_tokens_in_vocab = [
        tok
        for tok in tokenizer.get_vocab().keys()
        if tok.startswith("<|coord_") and tok != "<|coord_*|>"
    ]

    # Sort numerically by extracting the number (not alphabetically)
    def extract_coord_num(token: str) -> int:
        try:
            # Extract number from <|coord_N|>
            num_str = token.replace("<|coord_", "").replace("|>", "")
            return int(num_str)
        except ValueError:
            return -1

    coord_tokens_in_vocab.sort(key=extract_coord_num)
    if coord_tokens_in_vocab:
        first_coord = coord_tokens_in_vocab[0]
        last_coord = coord_tokens_in_vocab[-1]
        print(f"    - First coordinate token: {first_coord}")
        print(f"    - Last coordinate token: {last_coord}")
        print(
            f"    - Total coordinate tokens (excluding wildcard): {len(coord_tokens_in_vocab)}"
        )

        # Check if we have a contiguous coord range.
        if first_coord == "<|coord_0|>" and last_coord == "<|coord_999|>":
            if len(coord_tokens_in_vocab) == 1000:
                print("[✓] Full range [0, 999] is present (1000 tokens)")
            else:
                print(
                    f"[!] Range endpoints [0, 999] but count is {len(coord_tokens_in_vocab)} (expected 1000)"
                )
        elif first_coord == "<|coord_0|>" and last_coord == "<|coord_1000|>":
            if len(coord_tokens_in_vocab) == 1001:
                print("[✓] Full range [0, 1000] is present (1001 tokens)")
            else:
                print(
                    f"[!] Range endpoints [0, 1000] but count is {len(coord_tokens_in_vocab)} (expected 1001)"
                )
        elif first_coord == "<|coord_1|>" and last_coord == "<|coord_1000|>":
            print(
                "[✗] Missing coord_0! Range is [1, 1000] instead of [0, 999] or [0, 1000]"
            )
            print("    This means coordinates at pixel=0 cannot be represented.")
        else:
            print(f"[!] Unexpected range: {first_coord} to {last_coord}")

    # Check all coord tokens if we have the list
    if expected_tokens:
        print(f"\n[+] Verifying all {len(expected_tokens)} expected tokens...")
        missing = []
        for token in expected_tokens:
            if token not in tokenizer.get_vocab():
                missing.append(token)

        if missing:
            print(f"[✗] {len(missing)} tokens are missing from vocabulary")
            print(f"[!] First 10 missing: {missing[:10]}")
        else:
            print(
                f"[✓] All {len(expected_tokens)} expected tokens are present in vocabulary"
            )

    # Check special tokens
    print("\n[+] Special tokens info:")
    print(
        f"    - Additional special tokens count: {len(tokenizer.additional_special_tokens)}"
    )
    if tokenizer.additional_special_tokens:
        print(
            f"    - First 5 additional special tokens: {tokenizer.additional_special_tokens[:5]}"
        )
        print(
            f"    - Last 5 additional special tokens: {tokenizer.additional_special_tokens[-5:]}"
        )

    # Test encoding/decoding with edge cases
    print("\n[+] Testing encoding/decoding...")
    test_cases = [
        "Hello <|coord_*|> world <|coord_100|> test",
        "Edge cases: <|coord_0|> and <|coord_1000|>",
    ]
    all_passed = True
    for test_text in test_cases:
        encoded = tokenizer.encode(test_text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        print(f"    - Original: {test_text}")
        print(f"    - Encoded IDs: {encoded}")
        print(f"    - Decoded: {decoded}")

        # Check if all coord tokens in the test are preserved
        import re

        coord_tokens_in_text = re.findall(r"<\|coord_[*0-9]+\|>", test_text)
        preserved = all(token in decoded for token in coord_tokens_in_text)
        if preserved:
            print("    [✓] All coordinate tokens preserved")
        else:
            print("    [✗] Some coordinate tokens lost in round-trip")
            all_passed = False

    if all_passed:
        print("[✓] All encoding/decoding tests passed")
    else:
        print("[✗] Some encoding/decoding tests failed")

    print("\n[✓] Verification complete")


if __name__ == "__main__":
    main()
