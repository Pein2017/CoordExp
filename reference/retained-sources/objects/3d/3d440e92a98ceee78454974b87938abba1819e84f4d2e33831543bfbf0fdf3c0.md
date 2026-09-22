# Lane A qualification candidate — lead acceptance pending

Lane A real-entry qualification completed under lead-ruling-01. Lane B remains
source-inapplicable HOLD with zero scientific cells; no new source or contrast
was substituted. Admission-v3 remains unchanged (128 train / 32 calibration /
32 evaluation). This is technical qualification, not evidence of a burst remedy
or a completed calibration-validation comparison. Broad training remains held.

Stable candidate: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot/qualification/candidate-v2/manifest.json`.
SHA256: `889de33bdac53bd43be7ae2557ba637ae0dd058e49a08605035864f029b91d8f`.
All numerical fields below are reduced from saved outputs by
`probes/training_set_completion/coordinate_address_readout/reduce_qualification.py`.
The reduction replays byte-identically and rejects a deliberately mutated frozen
backbone identity. The original blocker-bound Markdown records were preserved
byte-exactly before replacement; see qualification/continuation01-records.json.

## Real-model evidence

- Exact bridge-off/native short-greedy agreement; zero-gain full-vocabulary max
  difference 0.0.
- Future-target mutation leaves earlier logits unchanged (0.0);
  the deliberately wrong same-position state changes, so this test has sensitivity.
- Selected causal hidden rows and identical-shaped head projection are exact.
  Cross-shape native FP32 drift is separately retained: 4.1961669921875e-05.
  The lead approved this mechanical comparison; the 2e-5 gate was not widened.
- Enabled coordinate-family logsumexp max error 1.90734863e-06;
  non-coordinate logits and unadmitted slots remain bitwise equal.
- Installed processor/vision rectangular merge order and image boundaries pass;
  deliberately corrupted order is rejected. Real visual banks are request-scoped
  primary-merger output, and an A/B/A request check shows no bank leakage.
- 262,401 sidecar parameters; optimizer contains only those tensors.
  Gain gradient is nonzero initially; Q/K/role gradients are zero at gain zero and
  nonzero after its first update. Every sidecar parameter group changes.
  The full original parameter hash is identical before/after.
- 16 fixed updates on 8 positive coordinate
  tokens per update: full-vocabulary CE 5.293778 -> 4.312341.
  Initial Q/K/role tensors with the learned final gain give CE
  5.292527, demonstrating a learned Q/K/role contribution.
  Cached coordinate-family/full-vocabulary CE accounting error is
  0.0. Train argmax coordinate MAE stays
  0.060500 -> 0.060500; CE learning does not establish
  geometric improvement or native owner gains.
- Cached/full replay full-vocabulary max difference 6.48498535e-05;
  chosen-token logprob max difference 1.14440918e-05.
- Tiny fit -> bound sidecar checkpoint -> fresh-process reload reproduces the
  saved native greedy sequence exactly. The frozen unique-class spoon diagnostic
  freely emits bins [140, 274, 336, 621] after a
  description-only prefix, versus target [139, 265, 330, 671].
  MAE 0.0165; no three-GT-coordinate
  prefix was supplied. Its first full box is valid; cap 8 truncates a subsequent
  row, recorded as a parser drop. This is one technical case without a baseline
  diagnostic comparison, not a calibration-quality or natural enumeration claim.

## Attempts, costs and forecast

Attempt01 failed before forward because source capture encountered PyTorch's
synthetic _classes.py file reference. The repair records synthetic external
references separately and still rejects a missing maintained source. Attempt02
failed the raw cross-shape compact-head comparison before updates. The separate
two-forward alignment-attempt03 localized the discrepancy entirely to native
head projection shape. Model-attempt04 passes the repaired full slice;
reload-attempt05 passes independent-process persistence/free generation.
Every attempt and source capture is retained; all five model processes exited.

Totals: 112.690520 allocated GPU-seconds
(0.031303 GPU-hours), 251 model forwards,
28 vision forwards, 16 optimizer updates / 128 supervised
coordinate tokens in updates, plus 160 coordinate tokens in the no-update
throughput backward. Peak measured CUDA allocation is
9595826176 bytes. The shared four-hour clock
continues from epoch 1790049699.020824; at saved closure,
13753.3 wall seconds remained. No producer remains.

The largest admitted training case (40 boxes / 160 coordinates, 371 teacher
tokens, 972 visual tokens) took 0.628767s
for a full frozen feature forward and 0.013501s
for sidecar forward/backward without another update. Projected cache payload is
1077843056 bytes from all 128 admitted shapes and coordinate counts,
with a projected largest per-case payload of 10024560 bytes.
Candidate-v1's largest-case extrapolation is superseded by this shape-specific
calculation; model evidence and execution were not repeated.

At measured 0.097997s per generated token, projecting
all 190 native cells to the 3084-token cap plus paired training and
diagnostic envelopes costs 16.134 GPU-hours, or ideally
2.017 hours across eight GPUs. Only
28 emitted tokens were measured: this is a forecast, not a long-context bound.
A 2x native-decoding slowdown gives 32.085 GPU-hours
and exceeds the remaining allocation; budget guards must stop incomplete cells
as HOLD rather than silently shrink the denominator. Proposed 256-update,
eight-image batch, LR 0.001 fits remain pending lead scientific/launch acceptance.
The full paired training/evaluation driver is not yet executed or claimed
qualified as an eight-image batch; this candidate qualifies the bounded
single-image model/capture/objective/save/reload/generation path.

## Reproduction and verification

Exact executed model commands are retained in each process.json and source capture.
The train attempt used qualify --mode train, launch-v2.json, cuda:0 and the fresh
model-attempt04 directory. Reload used --mode reload, the same launch, its saved
sidecar.pt and trained-generation.json, and fresh reload-attempt05.

```sh
python -B -m probes.training_set_completion.coordinate_address_readout.reduce_qualification --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot/qualification/lead-reduction-fresh.json
python -B -m pytest -q probes/training_set_completion/coordinate_address_readout/test_bridge.py probes/training_set_completion/visual_detail_dependence/test_runtime.py
python -B scripts/research/check_research_knowledge.py check
python -B -m src.artifacts.output_layout --root /data/CoordExp/outputs --root /data/CoordExp/.worktrees/research-probes/outputs
git diff --check
```

Nine CPU tests, research knowledge, output layout and whitespace checks pass.
The reducer command is saved-output-only and uses a fresh output path.

Owned changes stay within the two probe packages and preparation/candidate notes,
plus required source captures. Lead-owned unit/ruling/state/catalog/frontier and
the asset index remain unchanged by this worker. Human13 and Refined5 are
preserved separate references. No new durable shared asset is proposed.
Direct session reporting now replaces wake markers; no new marker was appended.

## Fixed Lane A pilot candidate
Lead ruling 02 released this continuation after accepting single-image qualification. All four fresh fits reached update 256. All 190 native cells (38 complete five-condition blocks) and all 160 calibration cells (32 complete blocks) completed. No Lane A cells are HOLD or partial. Lane B remains source-inapplicable HOLD with zero cells. No recurrence-prefix cells were executed.
The aligned sidecars did not improve held-out calibration: both coordinate CE and teacher coordinate-family argmax error worsened. Both also lost fresh-panel annotation-owner coverage and added cap/geometry debt. The development pool shows different proxy behavior and must not be pooled into a generalization claim. This does not establish a unique burst mechanism, reject every finer readout, or reject the visual-detail hypothesis. All owner results below are annotation IoU proxies; no new physical adjudication was performed and unmatched predictions remain UNKNOWN.
| Condition | Calibration CE/token | Teacher coordinate MAE | Free-box MAE on scored cases | Missing first box / 32 |
|---|---:|---:|---:|---:|
| original | 3.409525 | 0.015389 | 0.125070 (32 scored) | 0 |
| aligned-1729 | 3.553782 | 0.020423 | 0.140363 (31 scored) | 1 |
| permuted-1729 | 3.452764 | 0.015775 | 0.151558 (30 scored) | 2 |
| aligned-2718 | 3.524034 | 0.021116 | 0.153695 (32 scored) | 0 |
| permuted-2718 | 3.436619 | 0.019126 | 0.132081 (31 scored) | 1 |

Free-box error is conditional on scored boxes; missing boxes, parser drops and cap/stop metadata remain explicit per case. It is not a successful-only full-denominator score.

### fresh_natural_evaluation

| Condition | Covered owner proxies | Gained | Lost | Revisits | Max contiguous run | Invalid geometry | Cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 175 | 0 | 0 | 11 | 6 | 9 | 0 |
| aligned-1729 | 138 | 7 | 44 | 199 | 195 | 665 | 4 |
| permuted-1729 | 156 | 9 | 28 | 268 | 256 | 359 | 2 |
| aligned-2718 | 154 | 12 | 33 | 12 | 6 | 268 | 2 |
| permuted-2718 | 164 | 11 | 22 | 352 | 339 | 624 | 3 |

### development_recurrence

| Condition | Covered owner proxies | Gained | Lost | Revisits | Max contiguous run | Invalid geometry | Cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 26 | 0 | 0 | 28 | 12 | 210 | 3 |
| aligned-1729 | 42 | 23 | 7 | 6 | 2 | 271 | 1 |
| permuted-1729 | 29 | 10 | 7 | 26 | 9 | 328 | 2 |
| aligned-2718 | 32 | 8 | 2 | 20 | 10 | 566 | 2 |
| permuted-2718 | 23 | 3 | 6 | 15 | 5 | 731 | 3 |

Package cost through last producer closure: 4509.113175 seconds (75.152 minutes) from original wall start; 8270.141018 allocated GPU-seconds (2.297261 GPU-hours), including qualification failures. There were 91315 model and 538 vision forwards. Production used 1024 updates / 230144 coordinate targets and retained all final checkpoints. All 18 owned producer receipts are terminal.

Seven evaluation workers finished before the final worker; the frozen static image-block assignment left a tail. The user asked to favor more GPU and memory use for speed; no current-run queue or scientific denominator was changed. Future authorized scheduling should avoid this static tail.

Fresh evaluation density/class strata were frozen. The six development cases retain source groups and annotations but have no explicit materialized `stratum` field in the production manifest; no new development strata are claimed after outcomes. Intermediate partial-reducer telemetry is superseded by the source-captured final replay; its exact intermediate producer bytes were not separately captured before later CPU reporting repairs. See production/reducer-repairs.json. Neither issue changes saved model cells.

Saved candidate model-evidence manifest: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot/production/candidate-v1/manifest.json`, SHA256 `ed57920e7e259e62f8e5e2e2f7fabc9ca98d82d3289adf81d4a2f58cf004f2ca`. Final handoff binds this manifest, deterministic reduction replay, fresh checks and current owned records. Candidate is not lead acceptance.
