# CPU preparation candidate

Status **corrected candidate**, awaiting lead replay; original CPU candidate was HOLD for crop mapping. Not lead-accepted. Worker `926-worker`, UUID
`01a0de41-cc56-7a62-8c56-c2d9850b95b5`, was verified through lead-worker transport
as `gpt-6-astra` / `low`. Lead `01a0dd7c-0899-7b81-90a2-2f50da3476d1` retains
scientific interpretation, acceptance and GPU release. No GPU generation, teacher
call, training, commit or publication occurred. CPU commands have completed.

The [unit](unit.md) is the sole scientific owner and contains all latest rulings.
The final anchor is **untied+axis001 step2444**. The intervening tied choice and
its preparation are superseded, not an additional arm. The user's performance
preference is not a causal finding about untying. Current user decisions are
settled: COCO80 with existing instance/group semantics; same 2B teacher may use
enlarged/repeated queries; no preset precision ratio; no runtime/GPU-hour cap;
prefer 8 GPUs. Future CE/type-gate/expected-axis-margin requirements are recorded
only; training remains outside this package.

Earlier K4 support did not prove natural greedy compilation; annotation matching
did not prove physical identity. This unit tests supervision acquisition, not
learning or a perception upper bound. No scientific recovery result exists yet.

## Artifacts and data

Output root: `/data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-26/`.
Active data/policy: **`preparation-v3/`**. Its `evaluator/data_manifest.json` binds
source and output identities. `truth.json` contains full current objects;
`provenance.json` contains per-object source/current/status crosswalks. The
acquisition directory holds only the visible view and model/query policy.
`preparation-v1`, `preparation-v2` and `preparation-v4-tied` remain preserved;
v2's unchanged native CPU evidence is explicitly reused below.

Human13 JSONL SHA256:
`5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23`.
The adjacent receipt and all 13 image-byte bindings pass. There are 392 reports:
197 negative human-added annotation IDs and 195 retained IDs. Retained boxes may
be human-refined; original-COCO edits/deletions were not reconstructed.

Refined5 frozen working SHA256:
`1d8d7c6d63e982f2d5060fa96a7cbc825c0314246276b181aaefff9ee80fed85`.
All six snapshot-receipt file hashes, manifest hash, crosswalk and five image-byte
bindings pass. Image references resolve through the actual source workspace
bindings, not the copied snapshot directory.

| Image | Source | Current | Retained IDs | Edited retained | Added region IDs | Deleted IDs |
|---|---:|---:|---:|---:|---:|---:|
| 7116 | 6 | 5 | 5 | 2 | 0 | 1 |
| 309264 | 10 | 14 | 7 | 0 | 7 | 3 |
| 351017 | 25 | 49 | 20 | 1 | 29 | 5 |
| 417044 | 15 | 63 | 14 | 0 | 49 | 1 |
| 477415 | 27 | 47 | 27 | 4 | 20 | 0 |
| Total | 83 | 178 | 73 | 7 | 105 | 10 |

There are no retained category edits. The Label Studio exporter source was read
but not executed: a missing COCO ID gets a deterministic negative ID from image
ID and region key. The snapshot establishes **105 added annotation regions**,
not 105 proven physically new owners. It lacks deleted-to-new redraw identity.
That is the precise unresolved physical crosswalk; no invented IDs, replacement
mask or IoU owner clusters fill it. Image 7116 has no hidden additions and remains
a preservation/deletion control, without a hidden-recall denominator.

Union: 18 distinct images / 570 reports, 302 hidden annotation IDs and 268 visible
IDs. Human13 remains development data; refined5 is confirmation for the frozen
protocol, not globally unseen data. COCO80 approximate completeness does not
silently resolve group/part, deleted/redrawn or extent ambiguity.

## Anchor identity and exposure

Final checkpoint:
`/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444`.
Base: `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
Load its DoRA adapter and **independent input/output** embedding deltas. Each delta
is `[1004,2048]`, selecting 1000 coordinates plus four wrappers. The candidate
policy hashes 11 files including both base weight shards and all four executable
adapter/delta files. Those four files match `inference_payload_manifest.json`.
The historical manifest also lists a now-missing `adapter/README.md`; do not claim
its full historical aggregate reverified. No old receipt or payload was modified.
See `model-inspection.json` and the final `untied-evidence-reuse.json`.

`exposure.json` binds current train/val bytes to their pipeline manifest. The
anchor's resolved config uses the broad train source; refined5 images occur there
with 83 source reports, 66 exactly matching current refined objects. The eval
source contains all 392 current Human13 reports. Evaluation exposure is distinct
from gradient supervision. The historical dataset-to-cache byte binding has not
been reconstructed, so no globally unseen label/image claim is made. This is not
a Human13-specialized fitted derivative.

## Frozen finite schedule and isolation

Exact prompts/policies are in `preparation-v3/acquisition/policy.json`; scientific
choices are in the unit. Initial schedule: shared original-image empty-history
greedy; K4 full-image samples; K4 fixed overlapping 5/8 corner-region samples,
rounded outward to 32px, native scale, no hidden-derived crop. FP32/SDPA,
patch-embed linearization enabled, batch one per request, 3084 new-token maximum,
EOS `<|im_end|>`, repetition penalty 1, no inherited generation defaults; samples
T=.7/top-p=1/top-k=0, paired seeds 92601..92604. Region order is TL/TR/BL/BR.
Enlargement is allowed but not silently added to this first contrast; residual
failure evidence may motivate a later finite proposal.

All candidates and parser-invalid records survive. Saved same-model queries
provide per-candidate same-category overlap scores against other queries, plus
counts at IoU>=.5. Literal duplicates retain their raw IDs. These are correlated
screening scores, not owner identity or trustworthy-positive admission. No extra
verification forward or automatic positive promotion occurs. K1–K4 and their
few discrete support-score levels reuse saved outputs for recovery/error/cost
curves; no dense threshold sweep or confirmation-driven threshold selection.

The visible schema allowlists image identity/dimensions and nonnegative objects.
Requests use no labels; candidate screening sees only predictions and visible
known-label flags. Full truth or negative IDs passed at this boundary fail closed.
The evaluator bank is not supplied to acquisition/screening. This is a tested
functional boundary, not adversarial same-user filesystem sandboxing. Future
teacher/reviewer inputs must remain blind to this report and evaluator bank.

`leakage-real18-final.json`: changing hidden coordinates, descriptions, categories,
IDs and counts leaves all 162 requests and screening inputs identical. Injecting
a hidden-count dependency produces the expected equality failure. The tests also
show evaluator sensitivity to changed truth, frozen-output tamper rejection,
crop mapping, truth-view rejection, complete eight-rank sharding/readback, and
unreleased-policy rejection before model load. `cpu-red-visible-boundary.log`
contains the real pre-guard failure; `cpu-green-restored-untied.log` has 7 passing
tests. No GPU numeric or real-model composition claim follows from CPU checks.

## Native CPU evidence and costs

`untied-evidence-reuse.json` verifies that v3's only policy differences from v2 are
screening metadata/status, and that visible bytes, all payload hashes and all
request plans match. Reuse **`cpu-plan-v2/frozen.json`**, which binds processor
prompt IDs, image grids and media hashes for 162 requests. No duplicate untied
frontend run was made after the user restored that anchor.

| Arm | Requests | Summed pixels | Summed visual tokens |
|---|---:|---:|---:|
| Shared greedy | 18 | 18,083,840 | 17,660 |
| Full K4 | 72 | 72,335,360 | 70,640 |
| Region K4 | 72 | 29,724,672 | 29,028 |

Region K4 uses about 41.1% of full K4's visual tokens. Equal calls are not equal
compute; this first contrast tests restricted context at preserved pixel scale.
The alternative enlarged-to-full-budget crop would change both magnification and
restriction. `cpu-costs.json` records CPU-only costs: zero executed model calls,
generated tokens and GPU seconds. Actual generation time/tokens and any later
verification cost remain separately measured, not estimated as equal.

## Readback, commands and next boundary

`probes/hidden_human_recovery.py` reuses native loading/preparation/generation,
DoRA and untied embedding owners, the compact-row parser, saved-row assignment
and the existing `plan_data_parallel_shards`. No historical executable imports,
new verifier model, framework, training or loss change were added. Torchrun
shards requests across eight model replicas; readback rejects incomplete,
duplicate or differently bound shards. CPU sharding is checked; its real runtime
and GPU memory behavior are the next vertical-slice qualification.

Offline readback explicitly uses class-agnostic one-to-one cardinality-first
IoU>=.5 matching, with category agreement separate. It reports raw/external-review
admitted hidden recovery, new hidden matches beyond greedy, visible matches,
individual IoUs, unmatched rows, parser debt, IoU>.95 recurrence pairs,
crop-boundary candidates, review dispositions and cost. Unexecuted images are
not zero scores. Annotation matching, crop boundary or recurrence does not decide
physical identity; same-owner extent and group/part uncertainty remain review
reasons. There is no preset precision target and no automatic candidate-to-label
promotion. The synthetic CPU score check is not a model result.

Fresh CPU validation:

```bash
python -m unittest discover -s tests/probes -p test_hidden_human_recovery.py -v
python -m scripts.check_research_knowledge check
```

After lead inspection, a released policy and a clean committed source identity,
the prepared two-development-image smoke command is:

```bash
torchrun --standalone --nproc_per_node=8 -m probes.hidden_human_recovery acquire --visible /data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-26/preparation-v3/acquisition/visible.json --policy RELEASED_POLICY.json --image-ids 1584 2299 --output /data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-26/smoke-01
```

It is 18 calls / <=55,512 new tokens. No runtime/GPU-hour cap is imposed; wait for
no idle-GPU gate. Stop after the finite schedule or a concrete technical failure;
no automatic expansion. Check composition, durable raw/readback, crop mapping,
empty-history baseline, termination and actual costs. Preserve refined5 for the
frozen confirmation protocol. The separately lead-released 18-image pilot is
162 calls / <=499,608 tokens at this schedule. Finite support is not an upper bound
on perception. No training is authorized.

Owned changes: one probe, one focused test file, this unit, one catalog row and
minimal frontier pointer. Final `preparation-receipt.json` binds files/checks.
Only clean-source preparation, GPU technical qualification and lead release remain;
the settled user questions are not reopened. Worker stops at this checkpoint.

## Bounded correction 01: norm1000 crop mapping

The lead/reviewer reproduced a decision-changing mapping fault: `candidates()`
used 999 rather than the maintained `src/data/geometry.py` norm1000 convention.
The shared function feeds both `admission_inputs()` and `evaluate()` (including
cumulative screening curves). It now maps each axis by
`(offset*1000 + bin*crop_extent)/original_extent`, preserving fractional bins;
it does not invoke the pixel-rounding helper.

For scheduled crop `[384,0,1024,640]`, bins `[8,160,24,320]` now map to
`[380,100,390,200]`, not `[379.625,100,389.625,200]`. Against hidden reference
`[383,100,394,200]`, the caller-level evaluator now reports one recovered hidden
annotation at IoU .5 rather than a miss. RED recorded both this failure and the
old incorrect mapping expectation; GREEN passes all 8 tests. The same new test
checks y-offset, full-image identity and fractional mapping. No crop/model/seed,
hidden boundary or denominator changed. No frontend or GPU run was repeated.

Original report, state, probe/tests, owned patch, transport report and preparation
receipt were copied byte-for-byte to output `correction-01/original/`, with hashes
in `correction-01/original-snapshot.json`. The original preparation receipt remains
unchanged at its original path. Correction evidence and current hashes are owned
by `correction-01/correction-receipt.json`; its report is
`correction-01/lead-report.txt`. Existing native CPU request evidence is reused:
this change affects prediction readback only. Lead replay and clean-source
qualification remain required before any acquisition release.
