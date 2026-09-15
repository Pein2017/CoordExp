# Good sampled outputs were not realized by the first updated greedy policy

Subsequent separately authorized evidence: the
[fixed-witness likelihood/margin read](../2026-09-10-fixed-witness-route-access/results.md)
now closes the historical unmeasured-score gap below. It does not change this
unit's original natural-greedy outcome or retroactively expand its launch.

Scientific disposition: **mixed, low-yield immediate train-side outcome**.
Better complete sampled outputs exist and many received positive objective
weight, but the first historical RLOO update does not realize the strong
candidate gains in natural greedy output. This is not evidence of zero
candidate likelihood movement or an intrinsic inability to learn them.

Technical disposition: **lead-accepted**. One cold read of the retained round1
adapter completed all256 images on eight GPUs; all ranks exited normally,
merged artifacts and both CPU consumers passed, and GPU resources returned.
There was no new optimizer step, random sampling, likelihood-replay job,
development evaluation or protected confirmation read. The fixed stop is met.

## Same-panel natural greedy outcomes

All256 images /1955 GT objects, same base and original selected-token embedding,
same native prompt/media, unmerged DoRA, FP32/SDPA, no resize, T0/top-p1/RP1,
batch2 per device and3084-token per-image cap. Only the adapter and output
identity change. Matching is category-consistent global one-to-one assignment;
FP and F1 remain annotation-relative under incomplete labels.

| Outcome | Source before update | RLOO round1 after update | Difference |
|---|---:|---:|---:|
| TP at IoU50 | 1259 | 1262 | +3 |
| TP at IoU60 | 1190 | 1195 | +5 |
| TP at IoU80 | 908 | 906 | -2 |
| Recall at IoU50 | 64.3990% | 64.5524% | +0.1535 percentage points |
| F1 at IoU50 | 0.588593 | 0.585479 | -0.003114 |
| Annotation-relative FP50 | 1064 | 1094 | +30 |
| Valid predictions | 2323 | 2356 | +33 |
| Strict geometric repeats | 467 | 484 | +17 |
| Parser drops | 794 | 792 | -2 |
| Length-capped outputs | 4 | 4 | 0 |
| Complete generated action tokens | 29686 | 29972 | +286 |

At IoU50 the update gains9 object IDs and loses6, retaining1253 of1259
original matches (99.5234%). At IoU60 it gains10 and loses5; at IoU80 it
gains7 and loses9. This is not joint owner/F1/burden improvement. Both panels
retain all four capped outputs; no truncation or malformed-output filter
improves the denominator. The new run has zero parser or scoring failures.

## Which observed candidate opportunities became greedy matches?

The prior [retained K4 census](../2026-09-09-natural-candidate-opportunity/results.md)
defines these candidate classes. The following denominators are distinct
`(image, GT object)` pairs gained over the original greedy output, deduplicated
across samples. They are descriptive pools, **not a combined predicted output**.

| Fixed candidate class | Candidate samples | Distinct gained objects | Present after round1 greedy |
|---|---:|---:|---:|
| Net TP50 improvement, allowing old-object losses | 89 | 134 | 5 |
| Strictly preserves the original greedy object set | 77 | 110 | 0 |
| Also no extra FP, repeats, drops or cap | 43 | 65 | 0 |
| Strong class restricted to positive RLOO advantage | 37 | 61 | 0 |

Across all89 net-improving candidates, none has all its gained objects present
in post-update greedy. Five distinct gains appear, but those are partial
realizations of candidates that also exchanged original objects. The other
four of the actual nine newly matched greedy objects are outside this
net-improving-candidate gained-object pool; this does not imply absence from
the entire K4 bank or absence from the model's support.

For the43 strong candidates, the per-candidate gained-object incidence count
is101, while the unique-object count is65. For their37 positive-advantage
candidates these counts are75 and61. Every one of these incidences remains
absent at IoU50 after the update. Positive/zero/negative strata can overlap in
object identity; their unique counts must not be added together.

This sharpens the conclusion to **observed opportunity plus signed training
signal, without immediate greedy realization at this dose**. It does not show
that probability failed to increase: a small update can change likelihoods
without crossing greedy token choices, and shared gradients/optimizer effects
may interact. The first update also contains all1024 sampled trajectories,
not only the strong witnesses; it is not direct fitting of those37 outputs.

## Eight-GPU execution and the single-GPU tail

The configured frontend assigned32 images to each of eight ranks. Equal image
counts did not imply equal generation work:

| Rank | Generated tokens | Capped images | Decode seconds |
|---|---:|---:|---:|
| 0 | 5120 | 1 | 513.40 |
| 1 | 1957 | 0 | 171.19 |
| 2 | 1787 | 0 | 150.75 |
| 3 | 8797 | 2 | 913.42 |
| 4 | 2147 | 0 | 192.37 |
| 5 | 2746 | 0 | 241.95 |
| 6 | 5348 | 1 | 495.14 |
| 7 | 2070 | 0 | 176.12 |

All eight rank states and merge status are `completed`. The final single-GPU
period was the slow rank3 finishing its longer outputs, not a failed eight-GPU
launch or a hung worker. This run preserves the existing static batch/rank
assignment; no scheduling change is introduced.

Controller start/end:2026-09-09 15:59:12–16:16:07UTC, **1015 seconds**.
Peak per-worker reserved CUDA memory was19,853,737,984 bytes (19.85GB decimal).
Eight requested GPUs times full controller wall is a conservative allocation
envelope of2.25556 GPU-hours; it is not active-kernel time or actual simultaneous
GPU occupancy throughout the run. Sum reported rank decode time is2854.34
seconds, including inference overhead. These distinct cost measures must not
be substituted for each other. No run reached the3600-second wall ceiling.

Async wake registration failed before arming with an app-server read timeout;
the lead remained active and observed actual producer exit through Linux
pidfd. This did not affect the inference, cause a relaunch or imply a successful
monitor. A too-long initial tmux socket path was corrected before any producer
started. Both records are preserved; all resources were released normally.

## Acceptance and evidence

- [Primary summary](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization/analysis-v1/summary.json), SHA256 `d4bf3e27f61e48b0127f8f9dee97d632cb5a6ac80a3d9037d60899e30e27f2ff`.
- [Per-image outcomes and witness joins](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization/analysis-v1/cases.json), SHA256 `859de42bef9d075360db8091e52f3feb9a5e95312ab713fb45a985cbf113a903`.
- [Lead acceptance and resource receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization/lead-acceptance.json).
- [Native inference artifacts](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization/cold/source256-rloo-round1-train256-natural-v1/run_manifest.json) and [prelaunch identity](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization/prelaunch.json).
- [Separate native evaluator artifact](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization/native-detection-eval-v1/metrics.json):256 images /1955 GT /2356 scored predictions; its COCO AP is not the primary matching/F1 contrast above.

Root freshly passed9 CPU tests, including the full historical round4 fixture
and rejection of round4 as round1, inspected the consumer, executed the fresh
round1 reduction and native evaluator (both exit0), and independently checked
the0/65 and0/61 set intersections. Adapter files, authored config and input
hashes still match their prelaunch bindings; saved effective config is exactly
the validated config. The launch source receipt retains Git/dirty state and
a125-file effective Python source snapshot. Complete logs remain on disk.

The existing [consumer](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/round1_realization.py)
can reproduce this result from saved artifacts with no GPU. Use a fresh absent
output directory and do not pass `--fixture-round4`:

```bash
python -m probes.dora_owner_learning.round1_realization \
  --post-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization/cold/source256-rloo-round1-train256-natural-v1 \
  --output-dir <absent-output-directory>
```

## Stop and remaining discriminator

This closes the missing immediate greedy-outcome read. The strongest remaining
distinction is likelihood response versus greedy decision-boundary crossing.
A fixed retained-sequence before/after log-probability read would address that
diagnostic, but requires a separately bounded cost decision. None is launched;
no new reward, architecture, training dose or checkpoint selection is justified
automatically by this result.
