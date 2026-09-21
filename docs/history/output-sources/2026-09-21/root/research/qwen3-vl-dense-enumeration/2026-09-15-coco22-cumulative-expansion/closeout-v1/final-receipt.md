# COCO22 final checkpoint — BLOCKED

The lead closed this task at a reproduced native readback publication failure. The fixed256 main arm was not launched; Source was not triggered. No review or repair work remains running.

## Accepted data freeze

- **22 images / 376 trusted owners = old227 + new149.** Old11 route objects are exactly preserved.
- New149 =142 visually verified GT +7 verified COCO80 unlabeled owners. One additional physical utensil has unknown class and remains outside the teacher.
- Raw proposal ledger: **173 PROCESSED +648 HOLD =821**. Prior lead-closed review packages account for461 raw IDs; this does not mean461 owners were verified. Unreviewed, ambiguous, unlocatable and candidate-only entries remain HOLD. Exhaustive review is not a launch gate.
- Full annotations preserve338 raw GT +87 unlabeled records. The accepted physical reference has398 owners.

## Exact bindings

| Artifact | SHA256 |
|---|---|
| annotations-v5 | `8a3ddfc03edb3064de417e25e444383dfdc83cc1a08a6bbcee08ddd7435e1959` |
| teacher bank | `270fe47723918a992092b822b2f78ccc2a42ce42160177381fe8a50ec33f93b6` |
| evaluation sidecar | `6cb0a9ac32b389a81d8047f3f4fdcd70c7dd4fd1771e91edbc339eb496fa3224` |
| GT admission | `c97615a9635889088b190231446e0d9a42e18b3b5fba62fa94b73bc5cb678c35` |
| fixed256 main config | `0f375c14d8ad44e102fc2051732335b457eda937f1311a51873a547452108c95` |

The source is `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/trial-v1/S/training/checkpoints/step-00256/adapter`, fingerprint `0a2d96f2f4a8c89c8e3db5cc743ebf4e4a22e7656d5a70437e68c66b59441a87`; exact base model and paired special-token embedding paths are in the JSON receipt.

## Actual validation

-29 tests passed (13.45s). Data binding, exact old routes, parser and evaluation validation passed.
-8-rank real training qualification completed3 ×2 updates:132 image exposures /88 model calls, with fresh optimizer and saved checkpoints. The lead cold-checked counters, rank agreement and checkpoint payloads. microbatch1 admitted;2/3 failed the frozen numeric tolerances.
- All66 native readback requests completed; all3 workers exited0. batch2 and batch3 each passed22/22 detection-parity comparisons. Candidate batch3 took552.72s versus serial697.21s.
- **Controller exited1:** `ValueError: qualification publication`. On actual image25274, the in-memory owner assignment tuples become JSON lists; direct equality fails, while canonical JSON roundtrip equality passes. The result file's `completed` flag is not an admitted controller terminal.

## Final boundary

**BLOCKED:** the data packet is accepted, but runtime release is not. Fix/replay the publication gate on existing evidence and obtain cold-step0/runtime admission before any main launch. These are next-task gates, not work silently continued here. The main arm and Source were not launched.

All native subagents and this task's runtime workers have ended. See [machine-readable final receipt](final-receipt.json), [failure reproduction](readback-publication-failure.json), and [held launch command](launch-commands.md).
