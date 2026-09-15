# Screened owner-data microproof

Date: 2026-09-12. Status: **lead-accepted, completed negative pilot, closed**.
The authorized32-update fit, cold check and385natural endpoints completed.
See [results](results.md). No promotion, extension or further model work.
Owner: `/root/astra_xhigh_data_flywheel`; root owns scientific acceptance,
label rulings, any launch and portfolio scheduling.

## Source-bound observation

The user correctly identified substantial usable object discovery in the
417044 donut result. The source is **Stable50 under an inference-only
coordinate-carrier KV intervention**, not a newly trained detector or an
unconditionally generated teacher. Image, prompt, native text/positions and
weights are unchanged; two completed native rows and the next structural
opener precede the free suffix. See the owning
[instance-state result](../instance-state/results.md) for mechanism scope.

The exact source `instance-state/panel-v1/417044-owner_a-coordinates.json`
contains 340 tokens, 34 parser-valid rows and EOS. The immutable consumer
reports 11 annotated matches / 23 unmatched at IoU50, no parser drops and
zero strict later-row IoU > .95 events. None of those unmatched rows are
automatically hallucinations or automatically new physical owners.

This lane did CPU extraction and actual visual review only. It did not
consume fresh-transfer images, mutate GT, load a model, train or change the
registered duplication predicate. Source hashes, literal tokens and review
cards are in the raw sidecar, not copied from display text.

## Conservative candidate ledger

| Candidate type | Count | Row IDs |
|---|---:|---|
| GT matched, keep | 11 | P0, P11, P14, P21, P26, P27, P28, P29, P31, P32, P33 |
| Unlabeled single-owner candidate, keep | 14 | P1, P2, P6, P7, P9, P10, P12, P15, P17, P18, P19, P22, P24, P30 |
| Physical duplicate, neutral | 2 | P3 → P1; P5 → P2 |
| Partial owner extent, neutral | 2 | P4, P20 |
| Multi-owner/background extent, neutral | 2 | P23, P25 |
| Unresolved extent or description, neutral | 3 | P8, P13, P16 |

These are reviewer proposals, not replacement ground truth. P2 and P7 have
medium-high rather than high confidence; P2's literal extent is imperfect.
P13/P16 are real chocolate-coated oblong pastries, but this visual evidence
does not settle their literal donut description sufficiently for this first
full-row pilot. P8 is a touching rear pastry with uncertain extent/category.
These three conservative holds do not deny that they may be valid donuts.

Of the 14 proposed unlabeled owners, **P1 was already supplied in the native
prefix**; 13 are free-suffix discoveries. All 11 annotated matches survive
screening. Original GT still has four unmatched annotated owners, and the
photo has additional unreviewed objects. No exhaustive-label or perfect-set
claim follows.

Physical duplication differs from the frozen strict overlap statistic:
P3/P1 IoU is 0.942857; P5/P2 is 0.363190. Neither triggers > .95. Their
neutral exclusion from this proposed teacher is visually supported and does
not retrospectively rewrite either source metric or threshold.

## Exact cleaned trajectory and changed conditions

Proposed retained order:

```text
P0 P1 P2 P6 P7 P9 P10 P11 P12 P14 P15 P17 P18 P19
P21 P22 P24 P26 P27 P28 P29 P30 P31 P32 P33
```

Each row is the **unchanged original full row**, including category,
structural tokens and four coordinate tokens. No box is repaired or class
renamed. This is 249 row tokens; a separate display-only EOS makes 250.
EOS has no proposed supervised loss, since the screened set is known to be
incomplete. This deliberately tests owner-sequence compilation, not learning
a declaration that all objects have been found.

Removal is a genuine intervention on conditioning. Only P0, P1 and P2
retain their original token history. P6 now follows P2 rather than P5, after
deleting P3/P4/P5. All 22 later retained rows have changed prefix tokens and
positions; later removals add further gaps. The artifact explicitly contains
both each original prefix and each proposed compacted prefix. The generated
25 literal records are therefore **synthetic teacher-forced trajectories**,
not 25 intervention-verified conditional capabilities at those new states.
No retained downstream log probability has been inferred from the old run.

An illustrative 340-position source mask marks 249 selected row tokens and
91 neutral tokens (90 removed-row tokens plus EOS). That mask is not the
proposed cleaned-prefix fit: keeping masked bad rows in the input would
preserve a different history distribution. Zero local loss is not guaranteed
zero parameter movement or probability change. The pilot proposes no
unlikelihood/negative loss on any held row.

## One proposed pilot, not launched

Root subsequently accepted the screened25 ledger and froze the following
single32-update experiment, including conditional KL0 and the strong25-owner
criterion. These terms are no longer open alternatives. Training is reserved
on physical GPUs0/1; after root released the other lanes, the cold endpoint
is frozen on all8GPUs with49/48/48/48/48/48/48/48natural jobs. The earlier
six-shard `preparation-v1` is preserved as **SUPERSEDED, never launched**;
the active executable packet is `preparation-v2/pilot.json`.

**Question:** From unchanged Stable50, can learning the screened 25-row
teacher on this one exposed image recover its reviewed owners in a natural
original-prompt rollout, while preserving legacy behavior?

**Contrast:** untouched Stable50 versus one fixed 32-update checkpoint from
the screened literal-row data. This is a finite-image feasibility microproof,
not evidence that pseudo-label augmentation beats GT-only training; that
would require a matched control outside this pilot. No raw-bad-teacher arm,
layer search, repeated seed/dose sweep, architecture, new objective, or
checkpoint selection on outputs is proposed.

Reuse `probes.parallel_owner_research/training.py`, its qualified two-rank
route and the literal records emitted here. Proposed objective:

```text
mean_25(sum_complete_row_token_NLL)
+ 100 * mean_56(normal Stable50 token KL)
+ 10 * mean_56(existing one-sided margin protection)
```

Normal56 is the existing image-disjoint bank (417044 excluded). Every step
uses all 25 literal positives, each exactly once per rank with the existing
replication normalization. No conditional KL is proposed: the new compacted
history has no independently verified incumbent-successor spans, and
preserving Stable50 on target states risks preserving its loop. Root explicitly
authorized that conditional-zero setting for this packet. Use the
qualified language-only DoRA surface and existing fresh AdamW settings,
without changing the base model, special tokens, vision tower or decode.

The existing engine accepts exactly one complete row per positive record;
these 25 records fit that public interface. No unsupported multi-row target
or new full-sequence trainer is assumed. At two ranks this implies 25
positive plus 28 sharded normal live forwards per rank per update, or 1,696
live training forwards per rank for 32 updates, plus reference preparation
and cold checks. The launch packet must bind actual instrumented reference
counts and invocation guards; there is no portfolio spending ceiling.

**Primary evidence:** a cold natural greedy decode from the original image
and prompt, with no supplied detection rows, KV patch or output filter,
same 3,084-token cap. Report one-to-one reviewed-owner recovery over the
25-row candidate ledger, separately for 11 annotated and 14 unlabeled
owners, plus gains/losses against natural Stable50. Geometry matching is
only candidate-owner evidence, requiring a bounded visual check on ambiguous
new/duplicate rows. Report the originally supplied P1 separately from the
13 free-suffix discoveries. Do not select a checkpoint using those outputs.

**Proposed strong microproof criterion:** naturally recover all 25 admitted
owners at IoU50, preserve the baseline reviewed owners, reach EOS with no
parser drops, and not reintroduce P3/P5-type physical duplicates or
P23/P25-type group extents. Report partial attainment honestly rather than
automatically extending the run. Exact target-token imitation is not
required. Root froze this strong criterion before fit; useful partial recovery
will be reported separately without changing the threshold after the endpoint.

**Retention:** cold natural evaluation on the already exposed union384,
reporting this trained image separately, normal56 separately and the
remaining 327 images separately. For all legacy comparisons retain original
GT/global-matcher counts at IoU50/60/80, gains and losses, strict > .95
repeats, parse/geometry drops, output volume and cap hits. A promising
within-image compilation must not be promoted if aggregate legacy owner
recovery regresses or abnormal burden increases. Fresh transfer256 is neither
selection data nor an evaluation dependency of this microproof.

**Stop:** after root review freezes the candidate ledger and packet, one
32-update fit and its single cold endpoint plus retention read. A failed
conditional or numerical plumbing check is technical-invalid, not a
scientific null; bounded repair retains the same question. Scientific
failure/partial success does not automatically trigger more updates, an
alternative teacher, new penalties or another data-mining round.

The strongest alternative after a success is image-specific sequence
memorization. That still answers whether this screened discovery can be
learned back into the original model; it does not establish a self-improving
data cycle on unseen images. The next cycle would need a separately frozen
cohort and independent candidate acceptance, not model-consensus alone.

## Evidence and reproduction

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/`

- `candidate-sidecar.json`: all 34 original rows, exact full row/prefix
  tokens, classes/geometry, GT match metadata, proposed decisions, confidence,
  source hashes and actual viewed image paths.
- `cleaned-trajectory.json`: unchanged retained rows, 22 explicit changed
  conditions, display-only EOS and removed IDs.
- `proposed-literal-positive-records.json`: 25 exact cleaned-prefix records;
  **not an authorized executable training packet**.
- `cards/manifest.json`: 34 individual crop cards and five single-image
  review sheets, with source crop coordinates and hashes.
- `validation.json`: counts, no GT/model/GPU mutation, source and output hashes.

CPU reconstruction command:

```bash
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-12-parallel-owner-research/data-flywheel/build_candidates.py
```

The renderer displays original-context crops and the same crops with only
the literal box; it does not repair image contents or infer new geometry.
All five review sheets, the original image and the original shared GT/pred
card were actually viewed by this worker. Root separately reviewed the
decision-bearing P2/P3/P4/P5/P20/P23/P25 cards; that supports those bounded
rulings, not automatic acceptance of every other row.
