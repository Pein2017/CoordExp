# Implementation handoff

This handoff transports the planned
[Human-13 K-Union-to-Greedy Overfit Screen](unit.md). The unit owns scientific
meaning. The linked OpenSpec change owns implementation scope and completion.
The Superpowers plan owns execution order only. This file authorizes none of
them to start.

## Current disposition

- Planning artifacts: authorized and being reviewed.
- Implementation: **HOLD** until the user explicitly authorizes it.
- Model/GPU execution: **HOLD** behind a later, separate decision.
- Available accelerator ceiling after authorization: eight GPUs.
- Current worktree: `/data/CoordExp/.worktrees/research-probes` on branch
  `research-probes`; reverify before continuation.

The user explicitly asked not to over-audit or over-design. Use the shortest
conclusion-bearing path. Optional telemetry, a generalized candidate engine,
another evidence journal, or speculative fallbacks are not prerequisites.

## Authority/read order

1. [Research unit](unit.md): cohort, `G/H/M`, prefixes, duplicates, arms,
   outcomes, claims, and stop rules.
2. [Reconciled review](review.md): independent findings and accepted
   corrections.
3. [OpenSpec proposal](../../../../../openspec/changes/add-human13-k-union-greedy-overfit-probe/proposal.md),
   [design](../../../../../openspec/changes/add-human13-k-union-greedy-overfit-probe/design.md),
   [tasks](../../../../../openspec/changes/add-human13-k-union-greedy-overfit-probe/tasks.md),
   and [delta spec](../../../../../openspec/changes/add-human13-k-union-greedy-overfit-probe/specs/coordexp-swift-human13-k-union-greedy-probe/spec.md).
4. [Superpowers implementation plan](../../../../../docs/superpowers/plans/2026-08-12-human13-k-union-greedy-overfit-probe.md).
5. Existing panel admission/static-source units and current Swift owner docs
   only when the linked artifacts route there.

If these conflict, stop. Research meaning is never inferred from the execution
plan, and implementation never begins from this handoff alone.

## Frozen decisions

### Discovery and readout

- Panel: exact hash-bound Human-13 `geo_sorted_xy` panel in the unit.
- Source: plain S step-2444 checkpoint; vision/aligner frozen, language-tower
  DoRA trainable with fresh AdamW per arm.
- Discovery: K=16 per image as four successive physical batches of four
  independent `n=1` requests, explicit seeds `21001..21016`, temperature
  `0.4`, top-p `0.95`, repetition penalty `1.10`, max-new-tokens `512`.
- Clean-greedy outcome: Source-matched HF, physical batch size one,
  repetition penalty `1.0`, original prompt.
- Stage 1 targets: `H` only. `M` is gradient-neutral and never a negative.

### Prefix and duplicate handling

- `P_raw` preserves exact natural state and owns duplicate unlikelihood.
- `P_clean` removes every later complete row whose class-agnostic predicted-box
  IoU with an earlier retained row exceeds `0.95`, preserving exact retained
  token spans without re-tokenization.
- Chronological duplicate classification runs before owner matching across
  Source and all K trajectories. A later duplicate is never eligible for
  `G/U/H`, replay, a positive target, or A4—even if it could otherwise match a
  distinct dense GT owner—and remains only raw-context negative provenance.
- Unmatched non-duplicate, malformed, and invalid spans remain context-only and
  masked.
- Every frozen duplicate event contributes one uncapped unlikelihood site at
  its final box-closing coordinate token under the original raw state, computed
  stably from target versus non-target logits rather than `1-softmax`.
- Duplicate-free output is a goal/readout, not a hard promotion gate.
- Every Stage-1 terminal target is masked.

### Arms

Run only the approved set after authorization:

- Frozen Source and separate full-GT body-CE capacity control;
- A0 shared no-H background control (Source replay plus shared duplicate UL);
- A1 coherent full-H chain CE;
- A3 uniform independent-H1 hub accumulated at one parameter state;
- A4 once-per-image atomic any-valid native-row mass;
- A7 A3 without Source replay;
- A8-prime coherent full-H violation-only token bottleneck paired with A1; and
- A6 natural-donor H1 only if the frozen ledger contains an eligible `H_mid`
  case.

A2, A5, random permutation sweeps, candidate-tree training, online refresh,
GT-IoU coordinate search, K-miss supervision, and an external bridge are not
Stage 1.

### Packing and GPUs

- Reuse one-row, no-padding varlen FlashAttention-2 with isolated segments and
  per-segment MRoPE reset under `global_max_length=12000`.
- Stable descending-length first-fit for independent segments.
- A1/A8-prime/full-GT: one coherent segment per image. A4: one atomic candidate
  group per image.
- All packs in one panel exposure use frozen global denominators and accumulate
  before exactly one AdamW step.
- Packing claims padding/launch efficiency only, not prefix-KV, image-encoder,
  or forward-FLOP reuse.
- Parallelize independent arms: one world-size-one Accelerate process per GPU,
  at most eight. Do not build an eight-rank DDP arm.

### Frozen initial update

- Train language-tower DoRA only; freeze vision, aligner, token embeddings, and
  base weights.
- AdamW: learning rate `1e-5`, betas `(0.9,0.999)`, epsilon `1e-8`, no weight
  decay, global gradient clip `1.0`.
- Initial scheduler: cosine, zero warmup, sixteen-update horizon.
- Normalized family weights: A0 `(H,replay,dup)=(0,1,1)`;
  A1/A3/A4/A6/A8-prime `(1,1,1)`; A7 `(1,0,1)`; full-GT `(1,0,0)`.
- Any later 100-update run restarts Source and fresh AdamW with its own schedule;
  it is not a resume of the sixteen-update arm.

## Minimal implementation surface

The linked plans currently select:

```text
scripts/research/build_human13_k_union_manifest.py
scripts/research/collect_human13_k16_vllm.py
scripts/research/census_human13_k_union_trie.py
scripts/research/run_human13_k_union_overfit.py
scripts/research/materialize_human13_k_union_configs.py
scripts/research/analyze_human13_k_union.py
scripts/research/launch_human13_k_union_matrix.py
src/losses/human13_k_union.py
configs/coordexp_swift/research/human13_k_union/*.yaml
tests/research/test_*human13_k_union*.py
tests/losses/test_human13_k_union.py
```

Use existing packing, forward, runtime, artifact, backend, and checkpoint
interfaces. Add no generic StateBank exception and no second trainer unless an
interface-level failing test proves the chosen experiment adapter impossible;
if that happens, return to OpenSpec instead of widening scope silently.

## Evidence sequence after authorization

1. CPU-only manifest/collector/loss/census/packing-plan/analyzer tests and a
   dry-run with zero model actions.
2. Bounded standards and research-intent review; resolve P0/P1 only.
3. After separate model/GPU authorization, acquire all thirteen Source HF
   clean-greedy rows and all 208 K requests, seal the full manifest, and run the
   no-update census. Stop before backward.
4. After a distinct update authorization, select one eligible image from that
   sealed manifest and run the complete real path: pack, forward/backward, one
   update, checkpoint write/read, HF clean greedy, matching, analyzer.
5. Return the measured mechanics and cost. Do not infer matrix authority.
6. After a second explicit decision, run applicable arms through exposures
   `0,1,2,4,8,16`, at most eight concurrently, and stop at the first complete
   table.

No automatic extension to `32,64,100`, objective retuning, target refresh,
checkpoint promotion, OpenSpec archive, commit, push, or publication is
authorized.

## Execution authority received on 2026-08-12

The user subsequently authorized this task to implement and verify the complete
OpenSpec path, run the full-panel Source/K discovery and census, execute the
production-shaped update slice, and launch the applicable matrix on at most
eight GPUs.  The same instruction authorizes in-scope dynamic execution
decisions and scoped commits while requiring preservation of unrelated work and
explicitly forbidding over-audit and over-design.  This single instruction
satisfies the earlier implementation, discovery, update-slice, and matrix
authority gates; it does not change the frozen scientific semantics, mechanical
stop rules, or the explicit launcher `--execute` guard.  A fresh 100-update
screen, checkpoint promotion, push, publication, stable-spec sync, and OpenSpec
archive remain outside the current completion objective unless separately
chosen after the first complete matrix table.

## Dirty-worktree boundary

At the time of this handoff, the checkout also contains unrelated modified
research router files and untracked OwnerBridge probe files. Preserve them.
Inspect and stage only explicit Human-13/OpenSpec/Superpowers paths if the user
later asks for a commit; never broad-stage or clean this worktree.

## Planned artifact root

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-08-12-human13-k-union-to-greedy-overfit-screen/<run-id>/
```

No output root is created during planning. A failed or partial run identifier
is immutable and is never reused.
