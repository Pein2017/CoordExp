---
title: Human13 Pure CE Independently Reproduces 392 of 392 Owners
description: From the unchanged partial-recall Source, 140 magnitude-only CE AdamW steps reproduce full cold Human13 fit and the historical candidate payload exactly.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-05-human13-pure-ce-replay
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-05
---

# Outcome

**Scientific verdict:** `HUMAN13_PURE_CE_COMPLETE_FIT_REPRODUCED`.
One fresh pure cross-entropy (CE) AdamW run from the original step-2444 Source
fits the complete fixed Human13 panel using only the 196 shared DoRA magnitude
vectors. After 140 updates, independent-process natural greedy decoding covers
all 392 annotated owners at IoU50, IoU60, and IoU80 without structural debt.

**Mechanics:** lead-accepted. Training, CPU materialization, and cold readback
completed once with unchanged native model code. All 588 saved/live unmerged
adapter tensors match exactly. Source adapter SHA-256 remains unchanged, and
no train/readback producer remains. The [unit](unit.md) and [v2 packet (preserved from archive ref `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-human13-pure-ce-replay/launch-v2.json)
own the bounded recipe and the one pre-producer launcher correction.

## Source was not already at full coverage

The lead rechecked the historical Source receipt hashes, adapter paths,
`mode=source`, absent output-QP payload, and RP1.0. The exact input provenance
is [source-baseline.json (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-human13-pure-ce-replay/source-baseline.json); these are old natural-generation
receipts reverified now, not newly generated baselines.

| Fixed panel | Source natural IoU50 / 60 / 80 | Fresh pure-CE natural IoU50 / 60 / 80 |
| --- | ---: | ---: |
| N2, 65 annotated owners | 30 / 27 / 8 | 65 / 65 / 65 |
| Human13, 392 annotated owners | 173 / 159 / 91 | 392 / 392 / 392 |

The N2 CE column comes from the separately completed [66-step add-on](../2026-09-05-dora-ce-margin-n2-ablation/results.md).
Human13 does not initialize from that N2 adapter: it independently starts at
Source and uses one shared adapter for all thirteen images. Both training runs
use correct canonical-prefix teacher forcing; all displayed natural counts
come from unassisted greedy decoding after training.

## Fresh N13 evidence

- 13 images, 392 owners, 3,637 canonical decision positions including EOS.
- Only 196 language-DoRA magnitude vectors / 573,440 scalars trained. A/B,
  base, vision, aligner, embedding delta, and output head remain frozen.
- Native full-vocabulary CE sum divided by 3,637; AdamW lr=0.003,
  betas=(0.9,0.999), eps=1e-8, weight_decay=0, seed=0, no scheduler.
- 140 executed updates of a 300-step ceiling; full margin checked every 10.
- Minimum cold target margin: 0.09504318237304688, above 0.00998 at all positions.
- Natural RP1.0: 392/392 at each IoU, exact canonical routes 13/13,
  natural EOS 13/13, duplicate/malformed/cap debt 0, valid unmatched 0.
- Cold readback: 588 exact saved/live adapter tensors; A/B checksum unchanged.
- Candidate magnitude distance from Source: 76.25761367745358.
- Training loop and finalization: 1,800.248 seconds; cold verifier: 582.099
  seconds. The native training timer excludes initial model loading/capture;
  these are not end-to-end process timings comparable to the newer N2 runner.
- 1,820 training image forwards, 182 margin forwards; peak training GPU
  reserved memory 32,312,918,016 bytes.
- The unchanged verifier also emits 13 RP1.10 monitor generations; their
  ordering violations (3 images / 7 rows) are explicitly not acceptance gates.

The candidate payload SHA-256 exactly matches the actual historical N13 CE
file after independently hashing both files:
`4bad9e8d67f0a519f0c2443d450d41695d2035b2ea8661b4327cd339ea155680`.
The standard unmerged adapter SHA-256 also matches:
`8a5ebfcacfa92be4b873fea4439fc25570a9c415be2245e20fd1da94c9ff4070`.
This is exact same-seed execution replay, not seed-robustness evidence.

## Interpretation and stop

**Observed:** a Source with partial natural owner coverage becomes a full-fit
fixed-panel model through ordinary CE and AdamW on existing internal parameters.

**Supported:** readout QP has no exclusive ability to overfit these dense images.
The premise that CE cannot fit this Human13 panel is contradicted by both the
historical run and this independent replay. No output residual or RL was needed.

**Unresolved:** whether a QP method has an advantage in equal-success weight
distance, computational cost, scaling, or held-out transfer. This replay is not
a matched optimization of those criteria and should not claim all possible QP
advantages are absent. Small-panel programmability is not population recall.

**Not claimed:** full-parameter fine-tuning, a new CE algorithm, self-rollout
learning, best-tuned loss comparison, held-out generalization, or production use.
Stop here and discuss what a new algorithm must improve beyond same-panel fit.

## Receipts and reproduction

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-human13-pure-ce-replay/run-v2/`.

The authoritative lead aggregate is `lead-acceptance.json`, SHA-256:
`f4a06a116798d24ee7bd10dfb2fdb487243bc3f856fb6c5698e638eab4b5e372`.
It binds the train, materialization, cold, and baseline evidence. The CPU-only
verifier independently checks every IoU threshold and does not trust the
legacy verifier's IoU50-only pass badge or unmatched-as-debt rule.

```bash
conda run -n ms python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-human13-pure-ce-replay/verify_result.py
```

Lead replay was byte-identical. A sensitivity test removed one actual cold
IoU80 match while retaining the legacy `N13_PASS` badge; the verifier rejected
it. Ten inherited focused tests passed freshly, and runner/helper/runtime/test
hashes remain identical to the original N13 packet. The pre-producer v1 empty
log is retained; its corrected v2 launch is the only actual model run. No
global environment file was modified, no artifact was deleted, and no further
training or monitoring is active.
