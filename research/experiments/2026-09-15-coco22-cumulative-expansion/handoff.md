# COCO22 cumulative expansion: closed handoff

## Current truth

This research unit is **closed and lead-accepted**. It is not a launch queue.

Read in this order:

1. [state.json](state.json) for lifecycle/disposition.
2. [results.md](results.md) for the accepted trajectory, exact metrics, frozen identities and final artifact.
3. [unit.md](unit.md) for the frozen protocol plus chronological execution history.
4. [visual-context-analysis.md](visual-context-analysis.md) only for bounded supporting visual analysis; it does not redefine the frozen teacher or accepted result.

Do not resume proposal enumeration, refresh the frozen teacher, trigger Source, add another seed/dose, or continue to a larger image cohort under this unit.

## Scientific purpose and accepted boundary

Question: can the prior 227-owner continuation model absorb 11 additional image conditions while preserving the old 11-image tasks through full 22-image replay under the same fixed-teacher recipe?

Accepted answer: yes, for this in-sample frozen panel and recipe.

- Frozen population: 22 images, 376 trusted COCO80 owners = 227 old + 149 new.
- Main arm: prior Sample-arm final256 parameters, fresh AdamW.
- Objective: sample-equal CE + 0.01 shared geometry hinge, `geo_sorted_xy`.
- Dose: 256 full-cohort updates; all 22 images participate in every update.
- Execution: 8 ranks, qualified microbatch1; native readback batch3.
- Natural greedy: saved128 and saved256 reach 376/376, FN0, annotation-relative F1=1.0, with complete-output review closed.
- Preservation: all old 227 owners remain matched at every saved checkpoint.
- Conditional Source: **not triggered**, because the main arm succeeded.

This result establishes cumulative in-sample fitting only. It does not establish held-out generalization, memory-limited continual learning, general owner binding, a visual-sink mechanism, or an iterative rollout-correction algorithm.

## Exact accepted artifacts

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion`

Final adapter:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1/S/training/checkpoints/step-00256/adapter`

Adapter fingerprint:
`513d3daef205c39ae17bf60f7c69959b9881bc2d4f29e7c007090358ac76a2ed`

Frozen annotation ledger:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/annotations-v5/annotations.jsonl`
SHA256 `8a3ddfc03edb3064de417e25e444383dfdc83cc1a08a6bbcee08ddd7435e1959`

Frozen teacher bank:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/data-v1/bank.json`
SHA256 `270fe47723918a992092b822b2f78ccc2a42ce42160177381fe8a50ec33f93b6`

Lead final receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/evaluation-v1/lead-final-receipt.json`

Per-image final scoring:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/evaluation-v1/S-step-00256.json`

Trajectory:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/evaluation-v1/trajectory-summary.json`

## Historical blocked checkpoint and repair

An earlier closeout was **BLOCKED at result publication**, not at model learning. Python assignment tuples were serialized through JSON as lists, so direct Python equality failed even though the underlying readback requests had completed. Preserve that receipt as historical evidence:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/closeout-v1/final-receipt.json`

The user then authorized the narrow runtime repair and continuation. The producer was changed to emit JSON-native assignment lists; the existing real-entry qualification replay passed without regenerating the scientific data. The accepted main run then completed normally. See `runtime-repair-v1/admission.json` under the output root.

The later visibility ruling excludes prior-only tiny objects with virtually no model-input visual information from future target construction. It did **not** retroactively change this frozen 376-owner denominator. The tiny 510122 kite was already HOLD/outside the frozen target set. See `runtime-repair-v1/user-visibility-ruling.json`.

## Maintained implementation shape

The COCO22 lane remains under `probes/training_set_completion/`, but reusable mechanics now use repository owners rather than importing execution helpers from unrelated probe directions:

- native row parsing: `src.eval.native_rows`
- bound native request reconstruction: `src.inference.bound_requests`
- owned child waiter: `src.runtime.process_completion`
- language-decoder activation checkpointing: `src.qwen.checkpointing`
- COCO22 regressions: `probes/training_set_completion/tests/test_coco22_runtime.py`

Direction-specific normalization/consensus policy may remain inside `training_set_completion`; do not force scientific policy into generic runtime modules merely for symmetry.

## Continuation rule

There is no remaining action inside this unit. Any next experiment must define a new question relative to this fixed-SFT baseline and the earlier Human13 overfitting result. In particular, a new stage should state what changed factor is expected to improve physical owner discovery or natural greedy realization beyond fixed-teacher replay, rather than treating a larger panel as automatic progress.
