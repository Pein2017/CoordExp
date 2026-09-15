---
title: Earliest Shared-Prefix Branch Pilot Results
description: Artifact-backed results for the three-case greedy versus sampled branch intervention.
type: result
authority: non_normative_research
unit_id: 2026-07-21-earliest-shared-prefix-branch-pilot
status: complete
evidence_status: executed_with_native_replay_parity
updated: 2026-07-21
---

# Scope and execution

This pilot tested whether a sampled branch can be introduced at its first
exact token divergence and then carried by greedy continuation. It used the
same frozen image, prompt, checkpoint, coordinate representation, and
generation configuration as the certified route audit. It did not train a
model and it did not inject an object detector or object slots.

The exact commands were:

```text
CUDA_VISIBLE_DEVICES=0 python scripts/research/run_earliest_shared_prefix_branch_pilot.py \
  --manifest research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-earliest-shared-prefix-branch-pilot/cases.json \
  --infer-config configs/coordexp_infras/infer/qwen3_vl_2b_description_first_geometry_sorted_pure_cross_entropy_type_gate_dora_step4887_same_covered_set_prefix_order.yaml \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-earliest-shared-prefix-branch-pilot/person-5001.json \
  --device cuda:0 --case-id person-5001-first-sampled-only --force

CUDA_VISIBLE_DEVICES=1 python scripts/research/run_earliest_shared_prefix_branch_pilot.py \
  --manifest research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-earliest-shared-prefix-branch-pilot/cases.json \
  --infer-config configs/coordexp_infras/infer/qwen3_vl_2b_description_first_geometry_sorted_pure_cross_entropy_type_gate_dora_step4887_same_covered_set_prefix_order.yaml \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-earliest-shared-prefix-branch-pilot/person-7511.json \
  --device cuda:0 --case-id person-7511-first-sampled-only --force

CUDA_VISIBLE_DEVICES=2 python scripts/research/run_earliest_shared_prefix_branch_pilot.py \
  --manifest research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-earliest-shared-prefix-branch-pilot/cases.json \
  --infer-config configs/coordexp_infras/infer/qwen3_vl_2b_description_first_geometry_sorted_pure_cross_entropy_type_gate_dora_step4887_same_covered_set_prefix_order.yaml \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-earliest-shared-prefix-branch-pilot/wine-glass-2685.json \
  --device cuda:0 --case-id wine-glass-2685-first-sampled-only --force
```

The durable copies are in [artifacts](artifacts/), with checksums in
[SHA256SUMS.txt](artifacts/SHA256SUMS.txt):

| Case | Artifact SHA-256 checksum |
| --- | --- |
| `person-5001-first-sampled-only` | `2df056a802c17e5c67ec2c29108919c9139c9ff4f195d74005b3751031fef410` |
| `person-7511-first-sampled-only` | `93405a80d1d7dfdb7663c1e170e63055a4215563dfdfab1e2dd28d2016b6d362` |
| `wine-glass-2685-first-sampled-only` | `a48b3e2485f2d74e64f4f9ff196e42d7dc3a6d529ae9312f895dc619e4f18f09` |

All three native replays passed exact target-row and released-suffix token
parity. The runner now compares the complete expected suffix horizon and
checks a terminal record separately; a malformed or terminal current row does
not trigger another suffix replay.

# Native reference receipts

The fixed budget was sixteen complete rows. `Unresolved` counts rows with an
unmatched or ambiguous prediction receipt; it is not a hallucination count.
The pilot keeps entity and geometry evidence separate.

| Case | Longest common generated-token prefix | First divergence | Native unique owners | Native duplicate owners | Native unresolved rows | Native malformed rows |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 5001 | 4 | `x1` | 9 | 0 | 7 | 0 |
| 7511 | 5 | `y1` | 6 | 0 | 4 | 0 |
| 2685 | 4 | `x1` | 8 | 2 (`-81`, `1226144`) | 3 | 0 |

The first sampled-only target owners were `5001:1269467` at sampled row 5,
`7511:-169` at sampled row 4, and `2685:-78` at sampled row 6.

# Per-rung receipts

`Current` means the target owner was recognized in the intervened current row.
`Suffix` means it was found only in the released greedy suffix. `Eligible`
means that the target owner was recognized in the declared target row after at
least one token was released by the decoder; a fully supplied target row is
never treated as decoder acquisition evidence. `Added` and `Removed` are
changes in the fixed-budget unique-owner set relative to the native reference.
The duplicate, unresolved, and malformed columns are corresponding deltas.

## Image 5001, target owner `5001:1269467`

| Rung | Forced tokens | Released tokens | Current | Suffix | Eligible | Added owners | Removed owners | Duplicate delta | Unresolved delta | Malformed delta |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Level A, first divergence | 5 | 4 | no | no | no | — | — | 0 | 0 | 0 |
| Level A, y1 | 6 | 3 | no | no | no | — | — | 0 | 0 | 0 |
| Level A, x2 | 7 | 2 | no | no | no | — | — | 0 | 0 | 0 |
| Level A, y2 | 8 | 1 | no | no | no | — | — | 0 | 0 | 0 |
| Level A, box end | 9 | 0 | no | no | no | — | — | 0 | 0 | 0 |
| Level B, target row start | 1 | 8 | no | no | no | `-107` | — | 0 | -1 | 0 |
| Level B, description end | 3 | 6 | no | no | no | `-107` | — | 0 | -1 | 0 |
| Level B, box start | 4 | 5 | no | no | no | `-107` | — | 0 | -1 | 0 |
| Level B, x1 | 5 | 4 | yes | no | yes | `-107`, `1269467` | — | 0 | -2 | 0 |
| Level B, y1 | 6 | 3 | yes | no | yes | `-107`, `1269467` | — | 0 | -2 | 0 |
| Level B, x2 | 7 | 2 | yes | no | yes | `-107`, `1269467` | — | 0 | -2 | 0 |
| Level B, y2 | 8 | 1 | yes | no | yes | `-107`, `1269467` | — | 0 | -2 | 0 |
| Level B, box end | 9 | 0 | yes | no | no | `-107`, `1269467` | — | 0 | -2 | 0 |

## Image 7511, target owner `7511:-169`

| Rung | Forced tokens | Released tokens | Current | Suffix | Eligible | Added owners | Removed owners | Duplicate delta | Unresolved delta | Malformed delta |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Level A, first divergence | 6 | 4 | no | no | no | — | — | 0 | 0 | 0 |
| Level A, y1 | 7 | 3 | no | no | no | — | — | 0 | 0 | 0 |
| Level A, x2 | 8 | 2 | no | no | no | — | — | 0 | 0 | 0 |
| Level A, y2 | 9 | 1 | no | no | no | — | — | 0 | 0 | 0 |
| Level A, box end | 10 | 0 | no | no | no | — | — | 0 | 0 | 0 |
| Level B, target row start | 1 | 8 | no | no | no | — | — | 0 | 0 | 0 |
| Level B, description end | 3 | 6 | no | no | no | — | — | 0 | 0 | 0 |
| Level B, box start | 4 | 5 | no | no | no | — | — | 0 | 0 | 0 |
| Level B, x1 | 5 | 4 | yes | no | yes | `-169` | — | 0 | 0 | 0 |
| Level B, y1 | 6 | 3 | yes | no | yes | `-169` | — | 0 | 0 | 0 |
| Level B, x2 | 7 | 2 | yes | no | yes | `-169` | — | 0 | 0 | 0 |
| Level B, y2 | 8 | 1 | yes | no | yes | `-169` | — | 0 | 0 | 0 |
| Level B, box end | 9 | 0 | yes | no | no | `-169` | — | 0 | 0 | 0 |

## Image 2685, target owner `2685:-78`

| Rung | Forced tokens | Released tokens | Current | Suffix | Eligible | Added owners | Removed owners | Duplicate delta | Unresolved delta | Malformed delta |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Level A, first divergence | 5 | 4 | no | no | no | — | — | 0 | +1 | 0 |
| Level A, y1 | 6 | 3 | no | no | no | — | — | 0 | +1 | 0 |
| Level A, x2 | 7 | 2 | no | no | no | — | — | 0 | +1 | 0 |
| Level A, y2 | 8 | 1 | no | no | no | `-86`, `-90` | — | +1 | -1 | 0 |
| Level A, box end | 9 | 0 | no | no | no | `-86`, `-90` | — | +1 | -1 | 0 |
| Level B, target row start | 1 | 9 | no | no | no | `-86`, `-90` | — | 0 | 0 | 0 |
| Level B, description end | 4 | 6 | no | no | no | `-86`, `-90` | — | 0 | 0 | 0 |
| Level B, box start | 5 | 5 | no | no | no | `-86`, `-90` | — | 0 | 0 | 0 |
| Level B, x1 | 6 | 4 | yes | no | yes | `-78`, `-86`, `-90` | `-83` | 0 | 0 | 0 |
| Level B, y1 | 7 | 3 | yes | no | yes | `-78`, `-86`, `-90` | `-83` | +1 | -1 | 0 |
| Level B, x2 | 8 | 2 | yes | no | yes | `-78`, `-86`, `-90` | `-83` | 0 | 0 | 0 |
| Level B, y2 | 9 | 1 | yes | no | yes | `-78`, `-86`, `-90` | `-83` | 0 | 0 | 0 |
| Level B, box end | 10 | 0 | yes | no | no | `-78`, `-86`, `-90` | `-83` | 0 | 0 | 0 |

# Interpretation

The three cases show a consistent local pattern, but not a population result:

1. For all three images, forcing only the first-row branch from the exact
   first divergence did not acquire the later sampled-only target owner.
   It can nevertheless alter later coverage in one case, so early token
   divergence is causally relevant but is not sufficient as a treatment.
2. Supplying the sampled history before the target row was necessary in all
   three cases. Once the current row reached its first differing coordinate
   (`x1` for images 5001 and 2685, `x1` for image 7511 after a `y1` global
   divergence), greedy continuation could acquire the target owner in these
   receipts.
3. The target was always found in the intervened current row, never in the
   released suffix. Therefore this pilot supports a current-row owner-
   resolution effect conditioned on sampled history; it does not demonstrate
   that a released greedy suffix autonomously discovers the target.
4. Image 2685 also shows the entity/geometry distinction. At Level B `x1`,
   the target physical wine-glass owner is recognized while the later
   coordinate tokens are not an exact copy of the sampled donor geometry.
   The result is eligible as entity acquisition, but it is not evidence that
   the full box is correct.
5. Fully supplied target rows are retained as intervention controls, but their
   `Eligible` value is explicitly false. They cannot be used as training or
   causal-acquisition evidence.

The most useful next step is therefore not to train on sampled coordinate
tokens. It is to collect more independently reviewed sampled-only owners and
test a history-conditioned current-row transition objective, while keeping
entity acquisition and coordinate quality as separate targets. The observed
harm deltas also show why a future treatment must monitor duplicates and
unresolved rows rather than reward a longer rollout alone.

These three cases are diagnostic examples only. They justify a larger,
reviewed screen; they do not establish that one particular prefix or
coordinate boundary is universally causal.

**Replay note.** Producer scripts deleted from `research-probes` on 2026-08-28 (reclaim-research-probes-lifecycle); replay them from tag `research-base-v2`: `git worktree add <tmp> research-base-v2`.
