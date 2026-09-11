---
title: Human13 Shared Output-QP Same-Panel Overfit
description: A staged test of whether one output-only residual shared across the fixed Human13 panel can compile all 392 canonical owners under natural greedy.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: completed
unit_id: 2026-08-31-human13-shared-output-qp-identity-generalization
topic: qwen3-vl-dense-enumeration
status: completed_human13_overfit_pass
evidence_status: immutable_n13_primary_acceptance
updated: 2026-08-31
---

# Human13 Shared Output-QP Same-Panel Overfit

## Pre-execution semantic correction

The stable unit ID and path retain the original `identity-generalization`
label for backlinks. That label no longer owns the estimand. Before any model
forward, the user superseded the proposed held-out and leave-one-out study with
the direct multi-image continuation of the Image2299 overfit question:

> After one fixed image could be overfit, can the same output-only mechanism
> overfit thirteen fixed images jointly?

The former held-out plan produced no execution or evidence. Generalization is
a separate successor only after same-panel capacity is established.

## Decision and estimand

Starting from the exact four-coordinate `geo_sorted_xy=(x1,y1)` step-2444
Source, can one output-only delta shared by all fixed Human13 images compile
the complete 392-owner target policy under fresh-cold original-prompt natural
greedy?

All thirteen images may contribute their canonical target routes, hidden
states, token constraints, and evaluator rows. That is intentional same-panel
overfit, not leakage relative to this estimand. The intervention must still be
one shared matrix: no image-ID branch, per-image residual, per-image checkpoint,
or inference-time payload selection is allowed. Frozen hidden states may encode
image and prefix identity, so a successful shared matrix may act as a finite
lookup table; that still counts as overfit success here.

### Primary success predicate

At repetition penalty `1.0`, one immutable `Delta W_out` must, on one fresh
process per verification cell:

1. decode from each of the 13 original prompts without teacher forcing;
2. reach strict same-category, global one-to-one, actual-pixel Intersection
   over Union at least `0.5` coverage of all `392 / 392` GT owners;
3. incur zero confirmed duplicate, unsupported, malformed, or token-cap debt;
   and
4. terminate with a valid natural end-of-sequence token.

Report legacy-12 `346 / 346` and Image2299 `46 / 46` before the pooled result.
Exact canonical token replay is a stronger diagnostic receipt, not a separate
requirement when an owner-equivalent complete route satisfies the predicate.
Repetition penalty `1.10` is a nonblocking robustness monitor.

The strongest permitted positive claim is:

> On the frozen Human13 panel and exact step-2444 Source, one output-only delta
> shared across all thirteen images compiled a fresh-cold natural-greedy
> 392-owner same-panel policy without hard debt.

This is finite-panel shared-output overfit or compilability only. It does not
establish held-out transfer, identity or distributional generalization,
semantic sharing, hidden-state internalization, base-model learning, or
production readiness.

## Frozen Source and panel identity

The first model process must fail closed unless every identity below matches.

### Source

- checkpoint:
  `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`
- base model:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- adapter tensor SHA-256:
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`
- special-embedding tensor SHA-256:
  `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`
- base `config.json` SHA-256:
  `c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de`
- base `tokenizer.json` SHA-256:
  `ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8`
- prompt/template authority:
  `configs/coordexp_infras/infer/qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml`
- prompt-config SHA-256:
  `d2217208bc3e419bc9d8c621b4842da262d216780eb1dfc88eade0edf1358f6b`
- launch config: derive one experiment-local leaf that changes only the run
  root and decision-bearing `generation.batch_size=1`; freeze its resolved
  bytes and hash before the first forward and never write the historical H0
  artifact root;
- backend: Hugging Face, full-model `fp32`, Scaled Dot-Product Attention,
  physical batch size one for decision-bearing decode;
- geometry: four-coordinate `XYXY`, stable `geo_sorted_xy=(x1,y1)`;
- wrapper/parser: native `object_box_closed` / strict expected parser;
- primary decode: temperature `0`, top-p `1`, repetition penalty `1.0`,
  `max_new_tokens=3084`.

`Delta W_out` changes logits only. It must not modify the tied input embedding
tensor or any checkpoint file.

### Panel

The historical artifacts were exactly reconstructed and verified before this
protocol correction:

| Artifact | Rows / owners | SHA-256 |
|---|---:|---|
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/evaluation-inputs/human-refined-12.coord.jsonl` | 12 / 346 | `cfe4f693133287f9e6c561fc094710c642f049aec3d50f44dea98b764ba2aa85` |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission/evaluation-inputs/human-refined-13.coord.jsonl` | 13 / 392 | `01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8` |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission/receipt.json` | admission receipt | `ad78c174897509dca07c24c57d12c897fdad42724d83af08147d79c9644c5414` |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl` | 13 / 392 | `5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23` |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.receipt.json` | ordering receipt | `cd1273f627f7bdfcb16e9ca6e4a50e9d51f0163081ea7458d6d3deabe9bb82c0` |

The ordered image IDs are `1584, 2299, 2685, 4134, 5001, 6040, 7511,
10707, 13348, 13923, 14038, 14439, 16228`. All 13 are target-bearing under
the full-GT predicate. The earlier `12 target-bearing + 1 preservation-only`
statement described a narrower missing-owner ledger and is inapplicable here.

## Intervention and solver contract

Keep the base model frozen and use

\[
z(h)=hE^\top+h\Delta W_{out}^\top.
\]

For each admitted stage:

1. serialize every included image's complete canonical x-then-y GT route;
2. teacher-force that route at zero residual and capture the final hidden state
   at every target-token decision;
3. solve one minimum-Frobenius-norm selected-row `Delta W_out` with registered
   target-versus-full-vocabulary margin `0.01`;
4. use cutting planes if needed, then require a final full-vocabulary margin
   certificate with zero positive slack; and
5. write one immutable payload before warm and fresh-cold natural decode.

Teacher-forced feasibility is necessary mechanics evidence, not behavioral
success. A candidate passes only through the primary natural-greedy predicate.
No alternate subset, route order, margin, alias, sign, row deletion, or payload
selection may be chosen after observing a decode.

## Frozen nested ladder

Panel owner counts freeze the burden-spanning subsets before the first model
forward: N=2 uses the minimum and maximum burden, while N=4 adds the nearest
one-third and two-third burden ranks; image ID breaks count ties.

| Stage | Images | Canonical owners |
|---|---|---:|
| N=2 | `6040, 16228` | 65 |
| N=4 | `4134, 6040, 13923, 16228` | 123 |
| N=13 | all ordered panel IDs | 392 |

### Stage -1: identity and zero-delta parity

1. Re-hash Source, panel, image bytes, tokenizer, prompt, wrapper, parser,
   evaluator, geometry, and decode settings.
2. Freeze canonical route bytes and the N=2/N=4/N=13 owner ledgers.
3. Reproduce Source natural greedy at repetition penalties `1.0` and `1.10`.
4. Prove zero-delta logits and decodes equal Source.
5. For each later payload, run a Source-A -> candidate-B -> Source-A process
   sandwich so loader contamination cannot masquerade as a treatment effect.

Stage -1 produces no scientific result. Any mismatch is `MECHANICAL_INVALID`.

### N=2, N=4, and N=13

At each N, solve and evaluate on the same registered images. All target states
are allowed because same-panel fit is the estimand. Promote only after the
current N satisfies the primary predicate on its own owners and hard-debt/EOS
conditions. N=2 failure stops N=4; N=4 failure stops N=13.

The original registered N13 solve and its identical diagnostic replay stopped
at a numerical boundary before payload creation. The later authorized recovery
retained the exact terminal dual on each active subproblem and bounded
continuation to three `4,000`-iteration segments; a status-0/no-new-cut point
could receive same-active-problem `ftol=0` certificate polish. It did not change
targets, constraints, objective, margin, certificate tolerance, or candidate
selection. Each decision-bearing candidate gets one fresh-process
natural-greedy replay at repetition penalty `1.0`; no warm candidate replay owns
acceptance. Repetition penalty `1.10` is a nonblocking monitor.

## Execution evidence

The completed [results](results.md) record owns current evidence. N2 passes on
`65 / 65` owners, N4 on `123 / 123`, and N13 on `392 / 392`, each at
IoU50, IoU60, and IoU80 with zero hard debt, natural EOS, exact canonical
replay, and exact Source-A -> candidate-B -> Source-A restoration.

The accepted N13 run is
`20260831T-n13-v5-certificate-polish-corrected`. Its census is `P=3637`,
`U=832`, 772 selected rows, hidden-span rank 2048, and 2,807,764 registered
constraints. The unchanged exhaustive certificate reports maximum FP64
violation `1.0842741027028424e-05` against tolerance `2e-5` and minimum
FP32 hook margin `0.00998687744140625` for the frozen `0.01` target margin.
Thirteen fresh candidate processes cover all owners and thirteen fresh
post-candidate Source processes restore four exact route-identity fields. The
extended immutable acceptance receipt classifies the result as
`HUMAN13_OVERFIT_PASS`.

## Required monitors

### Identity and resource

- every Source, panel, image, prompt, tokenizer, wrapper, evaluator, route, and
  payload hash;
- process/model-load count, generation and teacher-forced capture count;
- wall time, peak GPU reserved bytes, peak host resident memory, worker count,
  cache bytes, and final artifact bytes;
- measured N=2 scaling projected to N=4 and N=13 before promotion.

### Capacity and optimization

- selected rows, registered constraints, active constraints, free variables,
  slack, primal/dual residuals, and final full-vocabulary certificate;
- Frobenius norm total and per owner/constraint as N grows;
- parameter participation-ratio effective rank, first-direction energy, and
  rank for 95% energy;
- largest row-energy share and largest baseline deficit/active constraint;
- protected-state numerical rank and null dimension as diagnostics only, never
  as a launch or success gate.

### Functional behavior

- per-image gained, retained, and lost owners; duplicate, unsupported,
  malformed, ambiguity, cap, and EOS status; first decode divergence;
- warm versus fresh-cold equality and Source-A/B/A restoration;
- top-1 flips, coordinate-token rank shifts, and logit Kullback-Leibler
  divergence on registered Source and target states;
- functional effective rank of the induced logit-change matrix
  `H_eval Delta W_out^T`.

If and only if the largest row holds at least `90%` of residual energy or one
constraint holds at least `50%` of squared baseline-deficit mass, run one
drop-dominant-row or leave-largest-constraint-out sensitivity. Do not build a
full ablation grid speculatively.

## Optional post-success identity-permutation null

This null classifies a successful N=13 fit; it does not gate or revoke it and
was not run after the primary success stop rule fired.
Keep the same target-token multiset and token-slot types, but permute which
matched hidden-state decisions receive those targets within frozen baseline-
deficit bins. This holds output-row popularity and approximate local difficulty
fixed while breaking the correct owner/state pairing.

- Comparable feasibility, norm, and functional rank supports an
  identity-agnostic finite lookup-capacity explanation.
- Lower burden for the correct pairing supports only panel-bounded semantic
  specificity.

Neither outcome is a generalization result.

## Decision and stop rules

- **MECHANICAL_INVALID:** any identity, parser, ledger, zero-delta, warm/cold,
  full-vocabulary, or solver-certificate failure. Repair mechanics without a
  scientific claim.
- **N2_FAIL / N4_FAIL:** the registered smaller stage does not satisfy natural
  greedy or hard-debt/EOS conditions. Stop scaling.
- **OUTPUT_SURFACE_BOUNDED_NEGATIVE:** a valid registered program is infeasible
  or cannot produce the target behavior. This is evidence about this output
  surface and recipe only, not about semantic information or generalization.
- **HUMAN13_OVERFIT_PASS:** the single N=13 payload satisfies all primary
  conditions. Stop and report the finite-panel result; do not require the null
  or open a larger parameter surface.

This unit does not authorize language-tower, aligner, vision-tower, tied-input-
embedding, per-image-payload, or rank-escalation treatments. `HOLD_PRODUCTION`
remains unconditional.

## Code ownership and execution state

The experiment-local runner is
`scripts/research/run_human13_output_qp_same_panel.py`; it reuses the existing
step-2444 loading, prompt, cold-decode, and strict owner-evaluation seams while
keeping the solver and selected-row hook local to this research consumer. Its
invariant tests are in `tests/research/test_human13_output_qp_same_panel.py`.
No shared infrastructure or checkpoint mutation was introduced.

The five historical input artifacts listed above match their published hashes.
N2, N4, and N13 payloads, natural-decode receipts, Source-restoration receipts,
and aggregate acceptances are durable under the output roots named in the
results. The N13 success stop rule has fired; architecture and production remain
unpromoted.
