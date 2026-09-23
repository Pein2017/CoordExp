# Accepted three-loss epoch16 comparison; not promoted

2026-09-22. Complete package independently accepted under
[ruling01](lead-ruling-01.md) and the unchanged clauses of [unit.md](unit.md).
All model work is terminal. This fixed recipe has nearly the same training fit
as matched CE-only16 and fails the prospective bad-output and owner-recurrence
limits. It is not promoted to larger training or automatically extended to32.

The three user-default losses remain CE1/typegate0.2(all four groups)/
raw-axis-validity0.01 with margin1/999, gaussian0, segment_balanced. They executed
correctly. This result weakens the specific expectation that restoring these
defaults at this dose would repair the medium-panel failures; it does not
separately identify either auxiliary, prove them universally ineffective, or
establish the codebook component's benefit or a burst mechanism.

## Matched result

Fresh mature untied+axis001 live source, seed1729, unchanged architecture,
optimizer/LRs, data and schedule prefix;984 optimizer calls over16epochs,
7872global packs and16384image presentations. The control is the historical
CE-only step984, not epoch32. Its missing full-panel evaluations were supplied
without retraining or modifying the accepted predecessor.

| Metric | Source | CE-only16 | Three-loss16 |
|---|---:|---:|---:|
| Train IoU50 owners /9519 |5580|6505|6512|
| Train IoU80 owners /9519 |3515|4638|4635|
| Train clean images /1024 |320|452|457|
| Train plain teacher CE |1.666406|1.173919|1.178286|
| Validation IoU50 owners /2033 |1224|1235|1242|
| Validation IoU80 owners /2033 |800|744|750|
| Validation clean images /256 |104|97|93|
| Validation plain teacher CE |1.654581|1.838695|1.843367|

Teacher denominators remain90934/19541 tokens. Composite training loss is
separate. On training, the change versus CE16 gains490 and loses483 annotation
owners, with184 images improving,174 worsening and666 tied in IoU50 counts.
Retained32 clean remains9/32 while IoU50 falls337→322; additions992 clean rises
443→448. The520 dense training images change18→22 clean, but IoU50 falls
5032→5028 and IoU80 falls3232→3222. This is not broad dense-scene improvement.

Exact96 sentinel source/three-loss8/three-loss16 clean is21/28/35 and IoU80 is
396/464/505 of1294; matched CE16 is37clean/512IoU80. Epoch8 selected nothing.

## Guardrails and failure redistribution

| Source-negative to final-positive images | Train / limit | Validation / limit |
|---|---:|---:|
| Bad output |57/51, fail|21/12, fail|
| Cap |1/10|0/2|
| Annotation-owner recurrence |73/51, fail|18/12, fail|
| Consecutive owner run>=5 |2/10|0/2|

These source-relative limits, not validation coverage/CE regression, determine
the failed eligibility. CE16-relative train bad repair/persistent/new is35/40/40;
validation is7/16/10. Owner-recurrent repair/persistent/new is47/54/58 train and
7/12/15 validation. Train recurrent-image incidence rises101→112 and validation
19→27 despite some lower aggregate counts. Train parser drops368→356 coexist
with affected images75→80; validation geometry83→33 and drops92→43 coexist with
affected images23→26. Exact-row repetition and owner recurrence remain distinct.

Independent review recomputed all1280 raw source/CE16/final triplets (3840cells),
including IoU50 owner assignments and runs, and reproduced these counts with
zero disagreement. Original32 membership is preserved independently of reuse.
UNKNOWN remains annotation-unmatched and is not classified physically false.
The mature SFT exposure959/1024train and248/256validation remains disclosed.

## Acceptance evidence and closure

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-three-loss/`.
Candidate `candidate-v1/manifest.json` SHA256
`f6967d35ce9f7aa38982febf5de45f2eabcf0c8134e7d0d5d8e36abb2c03010b`.
Lead acceptance is `lead-acceptance-v1.json` under this root.

The lead verified35 candidate authority/artifact bindings and20 current/capture
pairs, identical initial903 trainable and frozen hashes on all four ranks,
984finite/applied log rows, exact LR prefix and unchanged executed schedule.
First-entry objective/gradient qualification remains
[accepted](lead-entry-acceptance.md). Weighted loss magnitudes are telemetry,
not gradient-effect estimates or evidence that a term was strong enough.

All3936 distinct cells are complete:2560new+1376reused (1280source+96CE16),
zero missing/mutated/HOLD; the source/three-loss trajectory alone has2656cells.
Fresh lead reducer replay is byte-identical, SHA256
`399f0960c8c6e1ef921c43ef949f63936995f1bb968786ca51d6c473d2d4cd14`.
Fresh lead payload readback matches all2560new cells against903 saved tensors
per checkpoint, SHA256
`862ed01932490cf2d97ba48a4cd3145399bea5d84a5e91a8790ba0df10c18824`.
Summary scientific fields also reproduce exactly; only the deliberately fresh
input-reduction path differs. Three focused final CPU tests pass; checks and
exact replay commands are retained with the acceptance receipt. Test counts
overlap prior child/lead checks and are not additive.

Recomputed cost is37007.938354730606GPU-seconds (10.279983GPUh), with4757.983802s
model-execution wall (1.321662h). All17 GPU jobs exited0 and joined, recorded
producer/supervisor PIDs are absent, and owned allocation intervals do not
overlap on any GPU. No failed GPU attempt; no model work remains. The copied
packing-metadata erratum and distinct final CPU reducer captures remain
explicit; neither changes executed producers or historical evidence.

## Next decision and delegation learning

Do not infer that a larger dose, heavier auxiliary weights or full-data training
will repair this failure. The next proposed discriminator is a small matched
teacher/native-prefix readout at concrete failures, separating legal-family
mass from within-family geometry/selection. Correct-prefix failure would favor
an unresolved learning/objective problem; generated-history-only emergence
would favor feedback amplification. This would not by itself distinguish
visual binding from all other causes. It is a proposal, not launch authority.

[Delegation notes](delegation-notes.md) preserve observed Luna/max objective
work and Luna/high reducer work. Real caller/normalizer/mutation instructions
and parent first-entry checks qualified this run without GPU repair. Local
child test mistakes were corrected before launch; full historical byte replay
required parent removal of a reducer metadata addition. Stale descriptive
packing fields were a parent omission. These are specific reusable prompting
and acceptance lessons, not an unmeasured model speed/capability ranking.
