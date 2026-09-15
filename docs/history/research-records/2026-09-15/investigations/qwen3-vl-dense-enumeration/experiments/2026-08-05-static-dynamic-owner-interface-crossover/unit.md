---
title: Static-Dynamic Owner Interface Crossover
description: No-training causal probe of post-LLM image state, completed-row history, and their interaction on dense owner enumeration.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-08-05-static-dynamic-owner-interface-crossover
topic: qwen3-vl-dense-enumeration
status: active
evidence_status: pending
updated: 2026-08-05
---

# Static-Dynamic Owner Interface Crossover

## Owning question and decision boundary

On each checkpoint's native full wrapper, do frozen post-LLM image states and
completed-row history each causally control source-specific free-row release,
and does their interaction improve three-row unique-owner utility?

This unit supersedes the older plan to make skipped-owner suffix recovery the
treatment. Gained, retained, and lost suffix owners remain decision-bearing
endpoints, but the treatments are the static image field and dynamic completed-
row state. The unit performs no training, optimizer step, weight update, new
token, wrapper change, architecture change, A2 run, broader panel, or
production promotion.

The unit must finish the declared P1--P4 matrix even when effects are weak,
null, harmful, readout-only, or surprising. Efficacy is never a technical stop
rule. A mechanically invalid cell is repaired once and rerun; if the repair
cannot establish validity, that cell is retained as `invalid/uninterpretable`
rather than converted into a scientific null.

## Competing explanations

- **H1 -- no usable static owner carrier at the tested interface.** The tested
  image regions and post-LLM states do not provide separately consumable owner
  evidence after qualified positive controls. A null alone is reported more
  narrowly as no target-specific effect at the tested interface/region.
- **H2 -- static carrier present, salience/routing failure.** Owner-selective
  image-key or residual intervention transfers to a strict complete row while
  dynamic coverage intervention adds little.
- **H3 -- dynamic coverage failure.** Static owner access is qualified, while
  completed-row state causally reinforces, suppresses, or reallocates owner
  mass incorrectly.
- **H4 -- coupled bottleneck.** The static-by-dynamic interaction improves
  strict owner release and three-row gained-minus-lost utility beyond either
  factor alone.
- **H5 -- neither tested handle is behaviorally causal.** Both qualified
  interfaces are mechanically valid but do not transfer to complete natural-
  prefix rows. The conclusion is limited to these interfaces and operators.

The final classification may retain more than one explanation when strata
differ. A3-only evidence remains checkpoint-specific and cannot be attributed
to `<|commit|>`, InfoNCE, or owner binding without a later matched control.

## Frozen substrates

### S: primary plain checkpoint

- checkpoint: `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`
- base: `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- adapter SHA-256: `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`
- special-embedding tensor SHA-256: `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`
- geometry: four-coordinate `XYXY`
- ordering: stable x-then-y, `geo_sorted_xy`
- wrapper: native `object_box_closed`
- status: archived final/best step; val200 mAP `0.434394` is provenance,
  not evidence on this unit's 13 images.

### A: sensitized A3 comparator

- checkpoint: `/data/CoordExp/.worktrees/owner-commit-binding/outputs/prod/coordexp_swift/owner_commit_patch_binding/a3_patch_binding_4epoch/checkpoints/step-2445`
- base: `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-owner-commit-natural-adjacent`
- adapter SHA-256: `b7fcb74a3dc7e8ed251b7b7513389164d1728358f4d99119e2335652711d21b7`
- special-embedding tensor SHA-256: `66ddb6f658340e69e94195dfb2e9f4a31f72f5536307749c640a6d30bc899eb5`
- commit token ID: `151669`
- wrapper: native `object_box_commit`
- sealed rp1.10 P0 result: sensitivity/provenance only.

### Matched inference recipe

Both strata use HF, fp32, SDPA, greedy decoding,
`repetition_penalty=1.0`, and `max_new_tokens=3084`. Each keeps its native
tokenizer, embedding profile, full wrapper, and parser. Prompt bytes must match
within the compatible prompt contract, but checkpoint-native wrapper bytes are
not forced to match one another. Comparisons and estimands are always
intervention-minus-own-baseline within checkpoint; raw S-versus-A comparison
does not identify the commit token or training objective.

## Panel and x-then-y derivation

Source panel:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission/evaluation-inputs/human-refined-13.coord.jsonl`
(SHA-256 `01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8`).
Its admission receipt is SHA-256
`ad78c174897509dca07c24c57d12c897fdad42724d83af08147d79c9644c5414`.

The admitted rows contain the same 392 manually calibrated physical owners but
are serialized y-then-x. Before H0, build one experiment-owned derived view by
stable sorting every row's `objects` list on decoded `(x1, y1,
source_object_index)`. Preserve every object mapping under canonical JSON
identity apart from list order (the derived JSONL is canonically reserialized),
preserve image identities and row order, and retain a declared
image path only when it resolves to the same hash-verified file from the
derived location. Emit per-row
source-to-derived index maps, object-multiset hashes, coordinate arity checks,
and whole-file/source hashes. S and A consume this same derived x-then-y owner
view; their wrappers remain native. No completed legacy denominator is
rewritten.

Every result reports the legacy-12 subset and image `2299` separately before a
pooled 13-image summary. Unknown or unmatched entities are neutral, and entity
identity and geometry match axes remain separate.

## H0: all-image substrate materialization

Attempt 26 native baselines: 13 images for S and 13 for A. Record resolved
config, base/adapter/embedding/tokenizer hashes, prompt and wrapper bytes,
image-token span, position IDs/MRoPE metadata, generated token IDs, row
boundaries, parse outcomes, owner matching, exact serialized-prefix hashes,
and legacy-12 versus image-2299 summaries.

H0 is not reused from the archived S val200 result or the sealed A rp1.10 run.
It materializes checkpoint-specific natural greedy prefixes under the matched
recipe and supplies all later exact-boundary identities.

## Frozen candidate owner pool and materialization

The preregistered pool is 32 GT owners over eight images. IDs are
`gt:<image_id>:<source-panel-object-index>`; boxes below are pixel XYXY.
Historical labels (`TP`, `SUP`, `A3R`, `NK16`, same-class separated `SS`,
same-class overlap `SO`, cross-class `XC`) select candidates only and must be
re-established under S step-2444 H0 before interpretation.

| Image | Four candidate owners |
| --- | --- |
| 2299 | `1` person `[519,51,632,263]` TP; `11` tie `[452,148,467,172]` A3R; `15` person `[990,188,1110,375]` diagnostic FN; `2` person `[760,63,865,262]` NK16 |
| 4134 | `28` tie `[774,376,850,710]` TP; `22` tie `[629,288,644,324]` SUP/NK16/SS; `29` wine glass `[472,389,488,423]` SUP/A3R/SS; `13` person `[1000,276,1043,354]` SUP/A3R |
| 5001 | `0` person `[764,41,846,169]` TP; `10` person `[1085,108,1151,222]` SUP/A3R/SS; `15` person `[74,172,267,854]` NK16; `17` bicycle `[901,286,984,466]` SUP/A3R/XC |
| 6040 | `11` person `[621,422,668,473]` TP; `13` person `[785,424,816,469]` SUP/A3R/SO; `10` person `[265,422,281,441]` SUP/NK16; `4` person `[1130,381,1200,481]` NK16 |
| 7511 | `0` kite `[370,229,399,290]` TP; `14` person `[1064,489,1074,520]` A3R; `1` kite `[999,305,1018,326]` SUP/NK16/XC; `3` person `[45,466,54,481]` NK16 |
| 10707 | `1` remote `[772,84,806,206]` TP; `16` bottle `[272,661,318,751]` SUP/NK16/XC; `11` bottle `[293,627,327,736]` TP/A3R; `9` bottle `[416,602,456,670]` NK16 |
| 14038 | `0` potted plant `[971,75,1085,304]` TP/A3R; `19` book `[1006,452,1051,467]` SUP/NK16/SO; `23` book `[985,480,1060,502]` SUP/NK16/SO; `12` book `[1007,372,1073,389]` ambiguity-FN/A3R |
| 16228 | `0` umbrella `[663,245,907,331]` TP; `11` person `[1036,315,1079,372]` SUP/A3R/XC; `38` person `[90,383,118,410]` SUP/NK16/SO; `47` person `[944,412,1029,539]` SUP/A3R/SS |

Materialize the final registry exactly once after H0:

1. Remap by unique `coco_ann_id`; if unavailable, require a unique
   `(category, pixel_bbox)` match. Ambiguous owners are indeterminate.
2. Re-establish native TP/FN, strict complete-row matches, trusted support,
   and natural-prefix row boundaries independently for S and A. Historical A3
   K16 labels remain comparator-only. Image-2299 old FN-support labels are not
   transferred because its earlier calibration gate failed.
3. For each target B, select the earliest checkpoint-native natural boundary
   after at least one strict covered owner where B remains uncovered and has
   independently verified support. Let A be the latest strict covered owner at
   that boundary. If no verified B exists, record `no_verified_B`; do not score
   it as a negative.
4. Map fractional GT-box overlap to the merger grid and split overlap cases
   into A-exclusive, B-exclusive, and shared-core cells. Shared-core evidence
   is region/density evidence only. Record absolute image positions and MRoPE
   hashes.
5. Retain 24--32 events, preserving all eight images when mechanically
   possible, at least one re-established native TP per image, and the SS/SO/XC
   strata. A missing candidate is replaced only by a same-image, same-stratum
   owner under the same frozen rules; otherwise it is retained as
   indeterminate. Never substitute val200 indices.

Prefixes are checkpoint-specific. An event may be valid in only one checkpoint
and is never treated as a cross-checkpoint paired observation.

## P1: static image-field matrix

### Observational census

For the final registry, capture image-position residuals at merger output,
block-0 input, block outputs 0--27, and final norm. Compute foreground,
`log1p` fractional occupancy/density, density-conditioned class, owner
retrieval, same-class retrieval, hardest-negative margin, and coordinate/area/
shuffled-owner/cross-image controls. Shared cells may not leak into both query
and prototype. These readouts describe where information is accessible; they
cannot pass P1 without strict free-row transfer.

### Query-to-image key family

At each valid natural row-start boundary, run:

| ID | Arm |
| --- | --- |
| `K00` | native |
| `K01` | byte-identical self/no-op |
| `K10` | B-exclusive image-key spotlight |
| `K11` | already-covered A-exclusive image-key removal |
| `K12` | equal-area background spotlight |
| `K13` | same-class competitor spotlight, or explicit `not_applicable` |

The image-key mask may affect only declared image keys for the current row
query. It must preserve token IDs, absolute positions, MRoPE, wrapper, and
non-image/future keys. Extend the existing recompute seam to release one full
row; teacher-forced likelihood alone is diagnostic.

### Post-LLM residual-field family

Implement one bounded multi-image-position post-block hook and run at blocks
13 and 23:

| ID | Arm |
| --- | --- |
| `R00` | exact self/no-op field replacement |
| `R10` | target owner-exclusive residual replaced by norm-matched background |
| `R11` | equal-count A/B owner-exclusive residual swap at fixed destination positions |
| `R12` | shared-core knockout for SO events, density/region interpretation only |

Use a deterministic equal-size subset when A/B exclusive cell counts differ;
record the selection. Block 27 repeats `R00` and `R10` on the re-established
native-TP qualification subset as a mechanical sentinel. Any block-27
behavioral effect invalidates the residual-field seam. Block 0--2 DeepStack
injection is observational only and is not silently equated with later blocks.

Native-TP support removal/replacement qualifies the actuator. Failure to move
a native TP is a scientific operator-null if every mechanical receipt passes;
it narrows the conclusion to the tested operator and region rather than proving
that the owner is absent.

## P2: dynamic history matrix

At the exact natural boundary after latest covered owner A, use post-block-23
residual replacement while preserving the token sequence, row length,
destination positions, wrapper, and MRoPE:

| ID | Arm |
| --- | --- |
| `D00` | native |
| `D01` | exact self/no-op |
| `D10` | latest terminal carrier norm-matched mute: A3 `<|commit|>`, S `<|box_end|>` |
| `D11` | latest row's last-coordinate state mute |
| `D12` | equal-length earlier-row terminal carrier mute |
| `D20` | whole latest completed-row state mute through the attested multi-position hook |
| `D21` | same-parent, equal-token-length, same-class whole-row donor replacement when eligible; otherwise explicit `not_applicable` |

`D21` is a diagnostic actuator control. Its donor must be captured from an
identical parent history and written at the recipient's destination positions;
blind cached post-RoPE K/V or arbitrary donor-cache swapping is prohibited.
S `<|box_end|>` and A `<|commit|>` are native analogues, not declared semantic
equivalents.

Release one complete row and then a fixed three rows or natural STOP. Classify
the carrier within checkpoint as suppressive/reallocating, reinforcing, idle,
or distributed. A classification requires strict complete-row transfer;
readout or margin movement alone is reported separately.

## P3: fixed 2x2 crossover

The preselected static factor is `K11`, covered-A-exclusive key removal. The
preselected dynamic factor is `D10`, latest terminal-carrier mute. This pairing
uses disjoint key-eligibility and residual seams and avoids two unordered
residual hooks on one layer.

| Cell | Static | Dynamic |
| --- | --- | --- |
| `Y00` | native | intact |
| `Y10` | `K11` | intact |
| `Y01` | native | `D10` |
| `Y11` | `K11` | `D10` |

All four cells share a checkpoint-native boundary, token/prefix hash, release
budget, parser, and owner matcher. Run a one-row local release and a persistent
three-row release. In the persistent arm, A's key removal remains active for
all three rows, and the terminal carrier of each newly accepted complete row
is muted once at the next boundary. Natural STOP ends the horizon and is
counted, not overridden. Forced clean AB/BA same-covered-set prefixes are
optional diagnostics on eligible equal-contract cases and never substitute
for natural-prefix evidence.

For an outcome `Y`, report:

```text
Delta_static  = Y10 - Y00
Delta_dynamic = Y01 - Y00
tau           = (Y11 - Y10) - (Y01 - Y00)
```

No component efficacy threshold gates execution of the 2x2. Failed actuator
qualification changes interpretation, not whether the planned cells are run.

## P4: frozen gradient-path audit

With model weights frozen and no optimizer, audit three fixed objectives on
the exact native prefixes:

1. static target-B complete-row negative log likelihood, with gradients
   retained to block-23 B-exclusive image residuals and matched background;
2. dynamic uncovered-B versus covered-A complete-row margin, with gradients
   retained to the latest terminal carrier and latest-row span;
3. their fixed sum as a coupled reachability diagnostic.

Record finite gradient norms, target/control region ratios, non-target owner
effects, grammar/STOP/invalid mass, and whether gradients reach the declared
post-LLM interfaces rather than only the LM head. Detached visual states cannot
support a static gradient claim. P4 is a reachability audit, not evidence that
an SGD update would improve rollout behavior and not authorization to train.

## Endpoints

The primary local endpoint is a source-specific complete row: correct physical
owner description/class evidence, all four XYXY coordinates, valid native
wrapper closure, and `<|commit|>` for A. Also record description and geometry
margins, B versus best uncovered/covered owner, row-entry versus
`<|im_end|>`, covered-A repeat, non-target damage, valid rows, duplicates,
unmatched, malformed, length, premature STOP, and over-continuation.

For each three-row arm relative to its own baseline owner set `S0`:

```text
G = S_arm - S0        gained
K = S_arm & S0        retained
L = S0 - S_arm        lost
net = |G| - |L|
```

Report repeat hazard at `t+1`, `t+2`, and `t+3`; newly covered unique owners
per released row; and STOP behavior as verified remaining support is exhausted.
Unknown/unmatched entities remain neutral. A generic length or continuation
increase without the intended physical owner is not owner recovery.

## Mechanical validity

Every cell records and checks:

- base/checkpoint/embedding/tokenizer/config/panel identity;
- exact prompt, wrapper, prefix token IDs, serialized-prefix hash, and row
  boundary;
- image span, absolute positions, MRoPE hash, target masks, tensor
  shape/dtype/device, finite values, hook count, and hook cleanup;
- self/no-op generation identity and selected-token numeric drift at or below
  `1e-4` where floating comparison is required;
- non-target position zero drift for residual replacement;
- full-row parse and source-specific physical-owner match.

Wrong token/prefix identity, retokenization, missing MRoPE, leaked hooks,
masking non-image/future keys, a single-position patch labeled whole-row,
position-confounded donor cache, missing native wrapper/commit, or a block-27
effect is technical invalidity. Repair once and rerun the affected cell; if it
still fails, preserve the receipt and classify it `invalid/uninterpretable`.

Native-TP null, generic continuation, readout-only movement, forced-prefix-only
movement, one-row gain lost by row three, negative net utility, and valid null
interaction are scientific outcomes. None stops the remaining planned matrix.

## Resource and execution contract

Recheck live GPU processes, utilization, and memory before every launch; never
evict unrelated work. When all eight A100s remain available, shard H0 and
events explicitly by checkpoint and image across GPUs 0--7. Each GPU writes an
independent shard under a unique run root; one CPU finalizer verifies complete
shard identity before publishing summaries. Never let concurrent workers write
the same artifact path or mutate shared model caches.

External artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-05-static-dynamic-owner-interface-crossover/<run-id>/
```

Required artifacts include `runtime_identity.json`, derived-panel receipt,
H0 ledgers, `cohort.json`, exact-prefix and intervention manifests,
per-event results, P1 census, P4 gradient receipt, `summary.json`, and
`results.md`. Large tensors remain outside Git.

## Fixed follow-up ladder

After all P1--P4 cells are attempted:

- valid local gain already proceeds to the declared three-row horizon;
- local gain with suffix loss is classified as displacement, with no new
  treatment sweep;
- valid margin/readout movement without a strict row is classified
  non-transfer;
- mechanically valid null is classified as null/H5-compatible, with no
  layer/head/temperature/panel expansion;
- A3-only effect yields a checkpoint-specific recommendation; A2 remains a
  future attribution condition;
- an invalid seam gets its one exact repair/rerun, then remains
  uninterpretable;
- at most one shortest in-scope diagnostic may be added for an unexpected
  result when it distinguishes two live H1--H5 explanations without changing
  the checkpoint, panel, wrapper, interface, architecture, or no-training
  boundary. Its question and disposition must be recorded before launch.

## Completion and recommendation contract

The unit is complete only when:

1. all 26 H0 baselines are attempted and reported by legacy-12/image-2299;
2. the 32-owner candidate pool is materialized into a 24--32 event registry or
   every deficit is explicitly indeterminate under the frozen rule;
3. every declared P1--P4 cell is valid or explicitly
   `invalid/uninterpretable` with the repair receipt;
4. within-checkpoint deltas, `G/K/L/net`, validity dispositions, and
   static/dynamic/interaction classifications are published;
5. H1--H5 are classified at the supported scope and the result recommends or
   rejects concrete training-objective families without silently promoting an
   architecture or starting training.

The result must say which of these routes follows, if any: higher-dimensional
owner-addressable static field; class/geometry-aware field; scalar occupancy or
density only as inventory/remaining mass; owner-selective query-key salience;
dynamic covered-owner rejection and remaining-owner reallocation; on-policy
multi-row/final-set credit; matched static/dynamic/combined pilot; or no tested
frozen-state treatment. A production run always requires a later explicit
decision.
