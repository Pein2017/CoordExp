---
title: Bounded CE controls and Source-started CE versus refreshed RLOO
description: Parallel successor portfolio after the negative SFT256 development curve.
type: investigation
role: research-portfolio
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
status: complete
evidence_status: observed_fixed_panel
updated: 2026-09-06
---

# Authorized successor portfolio

On 2026-09-06 the user authorized advancing all proposed EOS, parameter-surface,
data-breadth, and CE/RLOO comparisons, including negative findings and parallel
work. The earlier 256-step unit remains closed and immutable; these are new
bounded contrasts, not a post-outcome extension. On the same date the user
explicitly accepted RLOO reward TP50/GT with no extra unknown/EOS/length/duplicate
reward term, and original Source as the common CE/RLOO anchor.

Each linked unit owns one question. This portfolio owns only shared constants,
resource coordination, and the closure list.

## Shared evidence and invariants

- Anchor: original unmerged step-2444 Source, never the degraded SFT step256.
- Base: /data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent.
- Source adapter: /data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444/adapter.
- Source embedding: sibling special_token_embeddings, held exactly unchanged.
- Source adapter SHA-256: 49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da.
- Train256 and Dev128: previous baseline inputs-v3; exact 256/1955 and 128/891
  image/annotated-owner counts, zero overlap, canonical processed geometry.
- Existing baseline Source/full-CE artifacts are immutable controls and reused,
  not regenerated. Default config fingerprint must remain
  39d550759d78d2e3c40b17779d7efa40b2e2ada156c66129c92f732b433dd00b.
- Frozen evaluation: same natural HF FP32/SDPA, eight workers, batch2, max3084,
  greedy T0/top-p1/RP1; original prompt, no prefix forcing or order gate.
- Primary: dev128 unique annotated-owner recall at IoU50. Always report 60/80,
  gains/losses, density, geometry, duplicates, invalid/dropped/cap/EOS diagnostics,
  predictions and unmatched-as-unknown. The dev set is historically used;
  neither untouched-test nor complete-scene precision claims are permitted.

## Units and finite doses

| Unit | Changed factor | Fixed numerical course | Natural evaluations |
|---|---|---|---|
| [EOS numerator](../2026-09-06-eos-numerator-control/unit.md) | Terminal EOS CE numerator zero, original denominator | Full DoRA; 256 steps, lr1e-5, EBS64, cosine256, seed19 | dev128 at64/256; fixed train256 panel at256 |
| [DoRA surface](../2026-09-06-dora-surface-control/unit.md) | Magnitude-only; A/B frozen | Two preregistered profiles: lr1e-5 exact-recipe control, lr0.003 historical-magnitude-dose sensitivity; each256steps/EBS64/cosine256/seed19 | each dev128 at64/256 and train256 at256 |
| [Data breadth](../2026-09-06-density2048-exposure-control/unit.md) | Nested density-matched train2048 instead of256 | Full DoRA; same16384 presentations,256steps/EBS64/lr1e-5/cosine256/seed19;8 rather than64 image passes | dev128 at64/256; same old train256 diagnostic panel at256 |
| [CE/RLOO](../2026-09-06-source256-ce-rloo/unit.md) | Canonical CE versus trajectory RLOO | Two Source-started chains, four full256-image AdamW updates each, lr2.5e-6, persistent moments; RLOO K4 freshly reacquired eachround | each dev128 everyround; train256 atterminalround4 |

The magnitude lr0.003 arm is NOT a pure parameter-surface ablation. It prevents
a null at the small A/B learning rate from being mistaken for magnitude-only
capacity failure; its joint recipe must be reported separately. No adaptive LR
sweep, best-only checkpoint selection, or new seed sweep is authorized here.

Native CE forward-evaluation and checkpoint cadence also remain identical to
the sealed full-CE control (steps16/64/128/256). This avoids assuming that a
changed evaluation/checkpoint schedule cannot affect RNG or runtime state.
Only the natural-generation read is reduced to the three preregistered points
per arm above; no additional natural decodes at16/128 are implied.

CE/RLOO matching is population, optimizer-update count and hyperparameters,
NOT FLOPs, generated/replayed tokens, model calls or wall time. Those unequal
costs must be measured. No equal-compute or algorithm-superiority claim follows
merely from a better four-step result.

## Runtime and acceptance

Output root: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor.
One semantic owner per source package. CPU/data preparation may run concurrently.
Preserve each native CE arm's world8/EBS64 topology rather than silently reducing
ranks to fit parallel scheduling. At most two of our native CE jobs may share
the eight cards after a production-shaped admission checks memory/updates.
RLOO replay was previously about35.84GiB reserved per rank; do not overlap its
large FP32 replay with our other GPU producers. Other users' workloads remain
untouched. Live A10080GB shared occupancy was 0-33GB per card at initial check.

Subsequent read-only acquisition admission observed only about14-17GiB/card
allocated by the two native CE jobs and64-67GiB free. The eight-image raw RLOO
qualification sampler may overlap them: it has no backward, one sequence per
call and no returned score tensors, versus a measured prior native B2 FP32
cold peak14.95GB allocated/19.75GB reserved. It records actual per-shard peaks.
This allowance is for acquisition only, not the large complete-action replay.

No overall cap replaces the user's open resource authorization. Individual
courses remain finite as above. A wait expiry is an observation deadline, not
scientific failure. Use durable launchers and event callbacks; no polling.

Close CPU-discoverable config/default/identity failures first. One representative
real update/save/eval/cold slice must cover new EOS/magnitude behavior; the
data2048 stress case must cover any newly observed largest workload. Do not
requalify unchanged mechanics as ceremony. RLOO requires real capped/terminated
action replay and optimizer round-trip before the costly full K4 bank; missing
or corrupt provenance is technical failure, never a silent sample exclusion.

Close every unit with its observed result, including negatives, once its declared
course and fresh artifact checks finish. No QP/GSPO expansion or human-label
rewrite is part of this portfolio. Do not infer architecture promotion from a
successful execution or a positive fixed-panel count.

## Closed portfolio

All registered courses and fixed reads are complete. Scientific outcomes and
technical recovery remain owned by the four linked unit results; none is
architecture-promoted. No active producer or further dose remains.

- [EOS numerator](../2026-09-06-eos-numerator-control/results.md): primary
  development owner gains come with severe length/parser debt and later
  high-IoU regression; removing EOS supervision is not a clean quality win.
- [DoRA surface](../2026-09-06-dora-surface-control/results.md): magnitude-only
  lr1e-5 is near-null. At lr0.003 training fits strongly but development worsens;
  limited surface capacity alone does not explain the weak low-dose result.
- [Data breadth](../2026-09-06-density2048-exposure-control/results.md): broader
  training mitigates narrow-data CE degradation but does not beat Source on
  the registered primary terminal development outcome.
- [Source256 CE/RLOO](../2026-09-06-source256-ce-rloo/results.md): four refreshed
  RLOO updates end at618/589/451 development owners versus CE612/589/452 and
  Source614/585/451. The primary difference is small, not monotonic or uniform
  across IoU, and uses greater replay plus sampling expenditure.

Together these controls do not support an intrinsic CE limitation, a unique
stopping-only mechanism, or a general RLOO-superiority claim. Data breadth,
supervision of termination and dose/surface all affect this exact recipe;
the separate contrasts must not be pooled as one causal intervention.
The fixed development panel has historical use; no untouched-test or
multi-seed claim is made. Any new course needs its own user-accepted contrast.

Technical closeout reuses [qualification (preserved from archive ref `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-06-ce-controls-rloo-successor/rloo-qualification-lead-acceptance-v1.json),
[first production bank acceptance (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-06-ce-controls-rloo-successor/source256-bank1-lead-acceptance-v1.json)
and the completed [launch packet (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-06-ce-controls-rloo-successor/source256-launch-packet-v1.json), with the
full primary aggregate linked from its result. The native density reduction
and magnitude parser failures were repaired only within their affected
surfaces; original failed artifacts remain preserved and completed unaffected
runs were reused. No inference result was silently relabeled as successful.
