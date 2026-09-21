"""CPU-only contract check for the Lane B row accounting reducer."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from probes.training_set_completion.native_row_choice.runtime import (
    _falsification,
    _validate_state_result,
)


def main() -> None:
    tokens = [151646, 8987, 151647, 151648, 151670, 151670, 151735, 151692, 151649]
    row = {
        "token_ids": tokens,
        "token_logprobs": [-1.0] * len(tokens),
        "row_sum_logprob": -float(len(tokens)),
        "positions": [[0, index, index] for index in range(len(tokens))],
    }
    result = {
        "actual_greedy_id": "actual",
        "candidate_sets": {
            "A": {"row_ids": ["actual"], "row_logsumexp": row["row_sum_logprob"]},
            "C": {"row_ids": [], "row_logsumexp": None},
            "N": {"row_ids": [], "row_logsumexp": None},
        },
        "rows": {"actual": row},
    }
    _validate_state_result(result)
    falsification = _falsification(copy.deepcopy(result))
    assert falsification["passed"]
    assert all(item["rejected"] for item in falsification["checks"])
    print("native_row_choice selfcheck: PASS")


if __name__ == "__main__":
    main()
