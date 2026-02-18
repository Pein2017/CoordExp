import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


def _env_flag(name: str) -> bool:
    v = str(os.environ.get(name, "") or "").strip().lower()
    return v not in {"", "0", "false", "no"}


def _load_tokenizers(model_or_tokenizer_path: str) -> List[Any]:
    """Load both fast and slow tokenizers when available.

    Decode behavior can differ between fast/slow implementations. Since Stage-2
    relies on precise span math, we probe both to avoid false confidence.
    """

    transformers = pytest.importorskip("transformers")
    AutoTokenizer = getattr(transformers, "AutoTokenizer", None)
    if AutoTokenizer is None:
        pytest.skip("transformers.AutoTokenizer is not available")

    toks: List[Any] = []
    errors: List[str] = []

    for use_fast in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(
                model_or_tokenizer_path,
                trust_remote_code=True,
                use_fast=use_fast,
            )
            toks.append(tok)
        except BaseException as exc:  # pragma: no cover
            errors.append(f"use_fast={use_fast}: {exc}")

    if not toks:
        raise RuntimeError(
            f"Failed to load tokenizer from {model_or_tokenizer_path!r}; errors: {errors}"
        )

    # De-dup if both branches return the same object type/impl.
    uniq: List[Any] = []
    seen: set[Tuple[str, bool]] = set()
    for tok in toks:
        key = (tok.__class__.__name__, bool(getattr(tok, "is_fast", False)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(tok)

    return uniq


def _decode_full(tokenizer: Any, token_ids: List[int]) -> str:
    return tokenizer.decode(
        [int(t) for t in token_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _decode_joined(tokenizer: Any, token_ids: List[int]) -> Tuple[str, List[str]]:
    pieces = [
        tokenizer.decode(
            [int(t)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for t in token_ids
    ]
    return "".join(pieces), pieces


def _summarize_mismatch(a: str, b: str, *, context: int = 80) -> str:
    # Return a concise diff summary without relying on external diff tools.
    if a == b:
        return "(no mismatch)"

    # Find first differing index.
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1

    a_win = a[max(0, i - context) : i + context]
    b_win = b[max(0, i - context) : i + context]
    return (
        "first_diff_char_index="
        + str(i)
        + "\n"
        + f"full_decode_window={a_win!r}\n"
        + f"joined_decode_window={b_win!r}\n"
        + f"len(full_decode)={len(a)} len(joined_decode)={len(b)}"
    )


def _run_canary_for_one_tokenizer(tok: Any, *, samples: List[str]) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []

    for text in samples:
        ids = tok.encode(text, add_special_tokens=False)
        ids = [int(t) for t in ids]
        if not ids:
            mismatches.append(
                {
                    "text": text,
                    "reason": "encode produced empty token ids",
                }
            )
            continue

        full = _decode_full(tok, ids)
        joined, pieces = _decode_joined(tok, ids)

        if full != joined:
            # Summarize mismatch and keep enough info to reproduce locally.
            mismatches.append(
                {
                    "text": text,
                    "reason": "stage2_fragment_mismatch",
                    "ids_len": int(len(ids)),
                    "diff": _summarize_mismatch(full, joined),
                    "full_decode": full,
                    "joined_decode": joined,
                    # Keep only a short prefix of pieces to keep messages bounded.
                    "pieces_head": pieces[:40],
                }
            )

    # Randomized canary: probe arbitrary token-id sequences (deterministic seed).
    # This catches tokenizer behaviors that apply cross-token normalization.
    if not mismatches:
        import random

        rng = random.Random(0)
        try:
            n_tokens = int(len(tok))
        except Exception:
            n_tokens = 0

        n_cases = int(os.environ.get("COORDEXP_TOKENIZER_CANARY_RANDOM_CASES", "50") or 50)
        max_len = int(os.environ.get("COORDEXP_TOKENIZER_CANARY_MAX_LEN", "64") or 64)
        n_cases = max(0, min(n_cases, 200))
        max_len = max(1, min(max_len, 256))

        if n_tokens > 0:
            for case_i in range(n_cases):
                L = rng.randint(1, max_len)
                ids = [int(rng.randrange(n_tokens)) for _ in range(L)]
                try:
                    full = _decode_full(tok, ids)
                    joined, pieces = _decode_joined(tok, ids)
                except Exception as exc:  # pragma: no cover
                    mismatches.append(
                        {
                            "text": None,
                            "reason": f"decode raised: {exc}",
                            "ids_len": int(len(ids)),
                            "ids_head": ids[:64],
                        }
                    )
                    break

                if full != joined:
                    mismatches.append(
                        {
                            "text": None,
                            "reason": f"random_ids_mismatch case={case_i}",
                            "ids_len": int(len(ids)),
                            "ids_head": ids[:64],
                            "diff": _summarize_mismatch(full, joined),
                            "full_decode": full,
                            "joined_decode": joined,
                            "pieces_head": pieces[:40],
                        }
                    )
                    break

    return mismatches


def test_tokenizer_decode_frame_consistency_canary():
    """Canary for decode frame mismatches with the *real* tokenizer.

    We compare:
      1) tokenizer.decode(token_ids)  (list decode)
      2) ''.join(tokenizer.decode([tid]) for tid in token_ids)  (token-piece join)

    If they differ, then any char-span math must be explicit about the chosen
    frame (piece-join frame is what rollout parsing uses).

    This test is gated because it depends on a local tokenizer/model directory.
    """

    if not _env_flag("COORDEXP_RUN_TOKENIZER_CANARY"):
        pytest.skip("Set COORDEXP_RUN_TOKENIZER_CANARY=1 to run tokenizer canary")

    repo_root = Path(__file__).resolve().parent.parent

    # Preferred: explicit path. Fallback: a stable local checkpoint path if present.
    default_path = "/data/home/xiaoyan/AIteam/data/Qwen3-VL"
    tok_path = str(os.environ.get("COORDEXP_TOKENIZER_PATH", default_path) or "").strip()
    p = Path(tok_path) if tok_path else Path()

    if (not tok_path) or (not p.exists()):
        fallback = repo_root / "output" / "1-26" / "checkpoint-1516-merged"
        if fallback.exists():
            tok_path = str(fallback)
            p = fallback
        else:
            if not tok_path:
                pytest.skip("No tokenizer path provided")
            pytest.skip(f"Tokenizer path does not exist: {tok_path}")

    tokenizers = _load_tokenizers(tok_path)

    # Stage-2-relevant synthetic fragments in CoordJSON objects[] format.
    samples = [
        '{"objects": [{"desc": "a", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}',
        '{"objects":[{"desc":"a","bbox_2d":[<|coord_1|>,<|coord_2|>,<|coord_3|>,<|coord_4|>]}]}',
        '{"objects": [{"desc": "a b", "bbox_2d": [<|coord_12|>, <|coord_34|>, <|coord_256|>, <|coord_480|>]}]}',
        # Include stop marker token if present in vocab.
        '{"objects": [{"desc": "x", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_1|>, <|coord_1|>]}]}<|im_end|>',
        # Include backslashes/quotes to stress escape handling.
        '{"objects": [{"desc": "a\\\"b", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}',
    ]

    all_mismatches: List[Dict[str, Any]] = []

    for tok in tokenizers:
        mm = _run_canary_for_one_tokenizer(tok, samples=samples)
        if not mm:
            continue
        for entry in mm:
            e = dict(entry)
            e["tokenizer_class"] = tok.__class__.__name__
            e["is_fast"] = bool(getattr(tok, "is_fast", False))
            all_mismatches.append(e)

    if all_mismatches:
        lines = [
            "Tokenizer decode frame mismatch detected.",
            f"tokenizer_path={tok_path}",
            f"num_tokenizers_tested={len(tokenizers)}",
            f"num_mismatches={len(all_mismatches)}",
            "",
        ]
        for i, m in enumerate(all_mismatches, start=1):
            lines.append(
                f"[{i}] tokenizer_class={m.get('tokenizer_class')} is_fast={m.get('is_fast')}"
            )
            lines.append(f"    reason={m.get('reason')}")
            if m.get("text") is not None:
                lines.append(f"    text={m.get('text')!r}")
            if m.get("ids_len") is not None:
                lines.append(f"    ids_len={m.get('ids_len')}")
            if m.get("ids_head") is not None:
                lines.append(f"    ids_head={m.get('ids_head')}")
            if m.get("diff"):
                lines.append(f"    diff_summary=\n{m.get('diff')}")
            if m.get("full_decode") is not None:
                lines.append(
                    f"    full_decode_head={str(m.get('full_decode') or '')[:200]!r}"
                )
            if m.get("joined_decode") is not None:
                lines.append(
                    f"    joined_decode_head={str(m.get('joined_decode') or '')[:200]!r}"
                )
            if m.get("pieces_head") is not None:
                lines.append(f"    pieces_head={m.get('pieces_head')!r}")
            lines.append("")

        pytest.fail("\n".join(lines))
