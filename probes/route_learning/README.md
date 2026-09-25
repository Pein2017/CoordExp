# Route learning

This family owns explicit teacher/route construction, fixed-teacher fitting,
Source256 paired completion and complete-output ranking. These are different
objectives and controls, not modes of a universal trainer. Scientific state,
parameters and accepted outcomes belong to the [experiment catalog](../../research/experiments/catalog.jsonl).

## Implementation owners

| Need | Owner | Choices that remain with the recipe |
|---|---|---|
| Reviewed literal routes and masks | `route_bank.py`, `repair_bank.py`, acquisition/review modules | Owner admission, retained context, teacher geometry and order |
| Fixed-teacher fitting | `training.py`, `coco227_training.py`, `coco22_training.py`, `dual_start_distributed.py` | Frozen partitions, optimizer, dose and acceptance |
| Paired Source256 completion | `source256_training.py`, `source256_normalized_training.py` | Variant validation, explicit CE denominators and enrichment |
| Complete-output ranking | `source256_ranking_training.py` | Ranking objective and reference-cache lifecycle; not the paired CE loop |
| Literal compact replay | `replay.py` | Batch composition, masks and gradients |
| Distributed reduction | `distributed.py` | Global denominator, SUM without an extra world-size divisor, rank partition |

Reusable token/text mapping belongs to `src.inference.token_text`, native input
and continuation to `src.qwen`, saved-row accounting to `src.eval.saved_rows`,
and the fixed model composition to [model profiles](../model_profiles/README.md).
The Source256 inference config is `probes/model_profiles/configs/source256.yaml`.

## Identity and publication

`src.artifacts.utf8_json` owns this lineage's unescaped UTF-8, compact sorted
JSON and terminal newline. It is deliberately different from the strict journal
encoder. `training.publish` rejects an existing file; `source256_data.publish`
accepts it only when bytes agree. These are not interchangeable policies.

Owned-child spawn/wait/termination belongs to `src.runtime.owned_process`.
Callers still choose devices, deadline and retry permission. A late observation
of a completed process does not establish its historical completion time.

## Validation and execution scope

Tests are in `tests/`, including CPU/Gloo four-rank paired execution and serial
AdamW comparison, literal masks and collision policies. Use CUDA-hidden bounded
CPU tests. Source-bound integration tests require their original preparations;
they must not be made green by rewriting old source hashes.

Module entries such as `python -m probes.route_learning.training --help` select
the maintained code. Running a frozen recipe requires its declared inputs and a
newly authorized output/identity context. Renaming a package does not reopen old
grants, recreate arbitrary teachers or prove real-Qwen numerical parity. Routine
fitting remains only where actual consumers/control value justify maintenance.
