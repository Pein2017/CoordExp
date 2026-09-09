---
title: DoRA parameter-surface control
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
status: completed
evidence_status: observed_fixed_panel
updated: 2026-09-06
---

# DoRA parameter-surface control

Question: on the same train256 panel, does restricting updates to magnitude
vectors preserve or improve development behavior compared with full A/B/m DoRA?
Primary exact-recipe contrast uses lr1e-5 on both surfaces. A second fixed
magnitude-only lr0.003 profile uses the previously successful magnitude dose;
it is a joint surface/dose sensitivity, not a pure surface causal comparison.
Both run256steps/EBS64/world8/cosine256/seed19; dev64/256 andtrain256 atterminal.
Strongest alternative to a negative m-only result is insufficient optimization
at the small full-DoRA LR. Retain all588 adapter tensors; A/B and selected delta
must remain exactly Source while196 magnitudes can update. No parameter norm
or loss proxy replaces natural recall, and no claim of m-only superiority from
one tuned recipe is permitted.

Shared constants, authority and resource/acceptance boundary:
[successor portfolio](../2026-09-06-ce-controls-rloo-successor/unit.md).

## Completed result

The lr1e-5 magnitude-only arm completed all three natural reads. Dev IoU50/60/80
is615/585/449 at64 and612/585/451 at256, versus Source614/585/451. The fixed
train256 panel ends1256/1191/911 versus Source1259/1190/908. Dev stops naturally
on all128 images; train retains4 length caps. This is near-null behavior, not
proof of magnitude-only capacity failure at every dose.

Its aggregate is /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/native-ce/analysis-magnitude_small_lr-v1/aggregate.json,
SHA-256675c25c9e81bfc25a6970bab75ad77e8f9f17876cdadb1302eb6048ccbe94096.
Both magnitude checkpoints passed all256 finite/applied updates, frozen A/B,
196 changed magnitude tensors and byte-identical Source embedding checks.

The lr0.003 arm's step64 dev read completed, but step256 dev failed at native
scoring.selected_count_mismatch (selected11 versus policy8, image460339).
That failed v1 evaluation was technical-invalid, not a negative model result;
its artifacts were preserved. The full588 saved surface and scoring policy
were not relaxed during recovery.

The role-evidence parser bug is now reproduced and corrected with74 passing
tests; all2816 rows/32737 predictions in16 prior completed runs reparse exactly
unchanged. See [recovery receipt (preserved from archive ref `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-06-dora-surface-control/parser-recovery-v2.md). Only the failed dev point
and unexecuted train point were rerun, completing at14:18UTC. All six natural
reads and both aggregates are now lead-accepted. The lr0.003 terminal result is
dev522/491/367 versus Source614/585/451, despite training1688/1639/1507 versus
1259/1190/908. See [completed interpretation and receipts](results.md).
The fixed two-dose unit is closed without promotion or further dose search.
