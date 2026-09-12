# Fixed32 result: fewer valid repeats, more invalid boxes — not promoted

## Decision

**The candidate fails the frozen joint-quality criteria. Stop at32 updates;
do not extend this coordinate-only unlikelihood recipe or promote its adapter.**
Training and the complete exposed384 natural evaluation finished and verified.
The earlier OOM was technical-invalid and is separately preserved; the accepted
runtime uses activation recomputation without changing the scientific packet.

| All384, natural unprocessed output | Stable50 anchor | Dedup32 |
|---|---:|---:|
| IoU50 TP |1899|1896|
| IoU50 FP |1522|1431|
| IoU50 F1 |0.606032|0.614288|
| Strict later repeats, IoU>0.95 |582|487|
| Later overlap, IoU>0.90 |606|509|
| Later overlap, IoU>0.80 |644|549|
| Geometry-invalid dropped rows |788|888|
| Other malformed dropped rows |4|4|
| Raw object-row starts |4213|4219|
| Generated tokens |40200|40278|
| Capped images |4|4|

The IoU50 ledger gains8, loses11 and retains1888 old annotated owners.
IoU60 TP1787→1783; IoU80 TP1354→1353. All seven specifically protected target
owners remain matched at IoU50. The same four images cap:
351017,417044,477415,502725; no cap was resolved or newly introduced.

Strict repeats decrease16.32% and F1 improves, but TP50 decreases and parser
drops increase100. Those two failures are explicit frozen vetoes. The native
`invalid_predictions` count is0 on both sides because geometry-invalid rows
are rejected by the parser **before** becoming valid prediction objects. It
must not be used to hide the888 geometry-invalid output rows.

## What actually changed

The lead inspected all10 paired visualizations: online8, dev59571 (largest
repeat increase), and remaining-train20781 (an old-match loss). Native raw
tokens and parser reason records were checked alongside the plots. The
renderer’s `dup-cand` is a pairwise visual hint, **not** the experiment’s
once-per-later-row metric; invalid boxes are absent from the valid-box plots.

### Dominant failure: repetition shifts into invalid geometry

In351017, strict repeats136→56, but raw row starts stay309 and generated
length stays3084. Geometry-invalid rows increase161→240. A repeated candidate
row includes:

```text
<|object_ref_start|>bottle<|object_ref_end|><|box_start|><|coord_0|><|coord_0|><|coord_0|><|coord_61|><|box_end|>
```

Here x1=x2=0. Its row still exists in the rollout, but no longer counts as a
valid duplicate. This case contributes80 of the net95 fewer strict repeats,
co-occurring with79 additional invalid rows. This is not a claim of exact
one-to-one alignment between every old and new row; it is strong direct
evidence against interpreting the aggregate decline as clean row removal.

477415 and502725 show the same failure direction: raw343 starts and3084 tokens
remain, while invalid rows increase.502725 includes repeated boxes with
y1=y2=999.417044 still emits308 valid predictions and309 starts; its broad
spurious donut boxes change extents/bands rather than stopping.248167 retains
26 valid rows and260 tokens despite strict repeats11→8, consistent with
reboxing/threshold movement rather than verified removal.

This is more specific than “the regularizer did nothing.” It changed the
distribution, but the negative set excluded invalid geometry and the loss
only reduced sampled-coordinate confidence. The observed outcome is
consistent with an inexpensive alternative: make a bad valid box invalid
instead of cease repeating an instance. No model intent is inferred.

### Genuine local reduction also occurs

274509 is a useful counterexample to a blanket null: raw starts44→22,
tokens398→200, strict repeats10→0 and geometry drops1→0, while all10 old
annotated owners remain. The repeated book-region output visibly contracts.
This supports a local output-reduction effect, not complete physical-owner
preservation for unannotated objects or general success.

158044 gains one annotated match while producing29 rather than27 valid rows;
it is not simple row removal or proof of a newly discovered physical instance.
The zero-repeat9813 control remains visually stable with9 old matches and
100 tokens. Some old IoU50 losses elsewhere are localization/threshold changes,
not proven disappearance of the physical object. All claims remain
annotation-relative; no GT correction was made.

## Scope and transfer

| Stratum | TP50 | FP50 | Strict repeats | Parser drops |
|---|---:|---:|---:|---:|
| Online8 |34→35|598→491|495→390|780→866|
| Reference56 |416→416|222→224|0→0|1→1|
| Remaining train192 |835→831|286→290|0→0|4→3|
| Dev128 |614→614|416→426|87→97|7→22|

The apparent benefit is concentrated on the trained repetitive cases.
Development F1 falls0.639250→0.635940, with6 gained/6 lost owners. In59571,
repeats19→31 and geometry drops1→16 accompany two lost old matches. The
reference set retains all416 old IoU50 matches but does not establish global
preservation. This is an exposed development result, not independent
confirmation or generalization evidence.

The scheduled online signal is nonmonotone:495 repeat rows at refresh0,
760 at8,567 at16,448 at24, and390 in the final natural online8 read. The32-step
checkpoint is the sole candidate; no intermediate checkpoint was selected.

## Execution and acceptance evidence

- Eight-rank shared DoRA,32 updates; online8 and reference56; no positive CE
  or GT-derived negative labels. Every refreshed exact trajectory supplies its
  own detached duplicate selection and frozen-Stable50 teacher reference.
- First invocation failed on SDPA memory before any completed global update;
  its logs, inputs and producer source are under `technical-invalid-attempt-01`.
- Retry used28 language-block non-reentrant checkpoint wrappers, preserving
  eval mode and bypassing generation. The real246-token rank1 control had
  exactly zero loss, full-gradient relative-L2 and max-absolute differences
  between checkpoint off/on. Two control forwards performed no optimizer step.
- The first two actual capped-stream updates passed. All32 updates have
  identical reduced gradient, adapter and Adam hashes across eight ranks;
  frozen weights remain unchanged. Final adapter parameter L2 displacement
  is0.1799273.
- Accepted training used2048 student training replays,88 teacher replays,
 32 online continuations and2 parity forwards. Sixteen model loads;55,596
  outer model forwards;2170 image forwards. Static teacher cache3,692,781,960B;
  peak allocated CUDA37,509,283,328B. No trajectory, hard case, dtype or loss
  was shortened/changed to repair the OOM.
- Natural evaluation:384 continuations,40,278 generated tokens, all8 shards
  exit0. The accepted pipeline took2456.87s wall time. Summed training/eval
  worker GPU allocation time was15,958.15s (4.43 GPU-hours), not an utilization
  percentage; the separately archived failed attempt is additional cost.
- Lead fresh verification:28 CPU tests, training receipt verification, exact
  native parser/metric readback,10 image inspections and raw invalid-row checks.
  The one monitor-registration attempt failed before arming; root remained
  active and observed completion through inotify. No monitor or model job is
  pending for this unit. Unrelated shared-GPU activity was not touched.

## Research conclusion and stop

This tests **sampled four-coordinate unlikelihood**, not every possible
deduplication signal. It does not falsify the user's simple0.95 detection rule
or prove a missing ledger/KV architecture. It does falsify treating fewer
parser-valid repeats as sufficient success for this32-step implementation.

If this direction is revisited, the next design must address the concrete
invalid-box alternative and distinguish row/instance suppression from sampled
coordinate suppression. Simply increasing this coefficient or extending its
dose is not the next justified action. A validity-aware bad-action definition
and row-level credit are hypotheses for a **new** bounded contrast, not changes
silently added to this closed run. No further training was launched.

The original Stable50 anchor remains unchanged; no default model/config, GT,
confirmation axis or production artifact was promoted. Existing dirty work
is preserved and no Git commit/publication or memory-store update was made.

## Artifact entry points

- [Raw root](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup)
- [Input packet](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup/inputs.json)
- [Training receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup/training/receipt.json)
- [Native384 reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup/evaluation/reduction.json)
- [Decision/raw-row diagnostics](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup/interpretation.json)
- [Visual observations](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup/visual-review.json)
- [Visualization manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup/evaluation/visualizations/manifest.json)
