"""Deterministic CPU check that full-vocabulary sampling is not constrained sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch


def run(seed: int = 190019) -> dict[str, Any]:
    # Five successive row-tail steps.  Token 7 is a grammar escape and is
    # deliberately given nonzero mass in the native full-vocabulary case.
    target = [10, 11, 12, 13, 20]
    coordinate_family = [10, 11, 12, 13, 14, 15]
    vocab = list(range(7)) + coordinate_family + [20]
    logits_by_step = []
    for index, token in enumerate(target):
        logits = torch.full((max(vocab) + 1,), -4.0, dtype=torch.float32)
        logits[7] = 1.0
        logits[token] = 0.3 if index < 4 else 0.8
        logits_by_step.append(logits)

    def sample(*, constrained: bool) -> tuple[list[int], int]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        emitted: list[int] = []
        escapes = 0
        for logits in logits_by_step:
            active = logits.clone()
            if constrained:
                mask = torch.ones_like(active, dtype=torch.bool)
                mask[coordinate_family] = False
                active[mask] = -torch.inf
            token = int(torch.multinomial(torch.softmax(active, dim=-1), 1, generator=generator).item())
            emitted.append(token)
            escapes += int(token == 7)
        return emitted, escapes

    native, native_escapes = sample(constrained=False)
    constrained, constrained_escapes = sample(constrained=True)
    replay, replay_escapes = sample(constrained=False)
    if native != replay or native_escapes != replay_escapes:
        raise AssertionError("fixed CPU RNG stream did not replay")
    if native_escapes <= 0 or constrained_escapes != 0:
        raise AssertionError("CPU control failed to distinguish native and constrained sampling")
    return {
        "schema": "recurrence_conditional_mass.self_consistency.v1",
        "seed": seed,
        "target_tail": target,
        "native_full_vocabulary": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "sampled_tokens": native,
            "grammar_escape_count": native_escapes,
            "retry_on_escape": False,
        },
        "constrained_coordinate_control": {
            "allowed_token_ids": coordinate_family,
            "sampled_tokens": constrained,
            "grammar_escape_count": constrained_escapes,
            "forced_coordinate_sampling": True,
        },
        "interpretation": "The constrained control is a self-consistency diagnostic only; its zero escape rate is not the native q estimate.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
