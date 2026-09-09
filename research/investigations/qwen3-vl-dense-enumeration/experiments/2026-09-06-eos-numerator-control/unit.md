---
title: EOS numerator control
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
status: completed
evidence_status: observed_fixed_panel
updated: 2026-09-06
---

# EOS numerator control

Question: from original Source, does removing only assistant-terminal EOS CE
numerator/gradient improve dev128 natural annotated-owner recall relative to the
sealed full-CE control? Strongest alternative: broad parameter/grounding changes,
not stopping supervision, cause the loss. Keep original supervised atoms and
segment denominators; report common full-transcript CE from the same logits
separately from the optimized EOS-zero objective. Do not reuse legacy censored
transcript normalization. Full language DoRA, lr1e-5, 256-step cosine/EBS64/world8,
seed19, exact train256; evaluate dev64/256 and train256 atterminal256. A change
only in output count without recovered annotated coverage is not success.

Shared constants, authority and resource/acceptance boundary:
[successor portfolio](../2026-09-06-ce-controls-rloo-successor/unit.md).

## Completed course

All256 updates and all three natural reads are complete and lead-accepted.
Removing the EOS numerator recovers IoU50 owners versus full CE and Source,
but dev length caps rise from0 to35/128 atstep64 and70/128 atstep256; terminal
IoU80 coverage falls23 owners versus Source. This is a real recall/termination
tradeoff, not a clean repair. See [results](results.md). No promotion or extra dose.
