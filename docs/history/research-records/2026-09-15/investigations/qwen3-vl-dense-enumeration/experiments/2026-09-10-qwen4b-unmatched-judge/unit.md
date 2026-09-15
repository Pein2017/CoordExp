---
title: Small Qwen3-VL4B judge of unmatched detections
description: One fixed54-case two-question pilot with vLLM timing and a minimal Transformers comparison.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-10-qwen4b-unmatched-judge
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-10
---

Closeout: the native engine/timing path was verified, but the4B/8B two-question
judges were not qualified as automatic candidate-admission gates. The later
user-authorized autonomous search and retained screening-only Co-DETR/context
profile are owned by the [successor unit](../2026-09-10-autonomous-unmatched-evaluator/unit.md).
Original pilot, prompt-control and conditional Source2B DoRA on/off artifacts
remain unchanged; this closure is not a general instruction-following claim.

The user authorizes the cached official4B Instruct model to judge the fixed54
non-strict-repeat FP cases and asks whether vLLM gives usable latency. The
reusable profile is for the agent's internal convenience; the later explicit
correction cancels user-facing CLI, product/interface and documentation design.
Keep only the minimal internal profile/callable needed for reuse.

Question: can the4B proxy give useful coarse evidence about entity/category
presence and single-instance box quality, and at what measured inference cost?
The reference is the already accepted54-case provisional visual review, not
new GT. Twenty-one cases lack a clean binary accept/reject label. The declared
binary screen is13 clean extras versus20 problematic current rows; the three
visually redundant rows also have independent class or box errors. This
two-question input does not establish cross-prediction duplicate detection.

Profile: `probes/unmatched_judge/profiles/qwen3_vl_4b.yaml`. Official4B weights,
no detector adapter/delta, BF16 greedy, output64-token ceiling, one GT-free
composite image containing scene context and a crop. GT boxes, review labels,
confidence scores and source-error strata are not shown to the judge. Output
is exactly entity/box with yes/no/unknown; an invalid answer is retained as a
parse failure, never silently repaired or converted to a negative label.

Root owns inputs, question semantics and quality readout. The engine owner
owns the minimal vLLM/Transformers caller and timing. GPU0 only; unrelated
GPUs2-7 are active and must not be touched. No package/environment changes,
downloads,8B fallback, training, GT edits or persistent serving daemon.

Budget: one4B vLLM load, at most6 first requests individually then the remaining
48 as a batch, all54 primary cases exactly once. Distinguish initialization,
first-request warmup, next5 unseen-input hot latencies, and batch throughput.
Disable repeated-input caches for this timing; do not label batch-average time
as interactive latency. At most6 matched Transformers requests after vLLM
releases GPU0; same official chat template and same prepared images. This is
answer-level comparison, not a claim of full-vocabulary numerical parity.
Total bound is72 generated responses and30 minutes of model/runtime work on
one GPU. No automatic full rerun or parameter/prompt search. One bounded
execution configuration correction is allowed only after a concrete technical
failure inside that same bound; repeated failure stops for a user decision.

The first6 native requests are the production-shaped slice. If all6 fail the
response interface, stop before the batch; otherwise retain failures and finish
the population. Quality, parser validity and latency are separate readouts.
The goal is a roughly usable proxy, not a perfect annotation classifier. Stop
after the one pilot, timing and bounded assessment; any standing service or
training/data integration needs a separate resource/semantic decision.

## User-authorized prompt/8B continuation

After the initial pilot, the user explicitly requests other prompts, an
inference-correctness check, and an8B fallback if4B remains ineffective.
The initial54vLLM+6HF artifacts remain immutable:49/54 yes/yes, including18/20
problematic rows; the first6 HF answers and expanded prompts matched exactly.

The superseding bounded continuation is GPU0 only, at most30 additional
minutes and150 responses. First use one4B load for two prompts on the same
four clear positive/four clear problematic cases, plus four explicit
image/category controls (20 requests total). Remove the printed proposed-
category header from the composites to reduce anchoring; no GT is supplied.
The controls compare a tight bottle crop under bottle/cup/elephant claims and
the full-context bottle under its correct class. They distinguish image/class
responsiveness from mere assent without claiming exhaustive model correctness.

At most two prompt variants, no open-ended tuning. A practical routing screen
requires retaining at least3/4 clean cases and not accepting at least3/4
problematic cases; abstention can defer a bad candidate but is not a certified
negative. This is a cheap routing heuristic, not validated accuracy. If neither
4B prompt passes, switch to the cached official8B using one selected prompt
on54 cases plus the four controls. If a4B prompt passes, test that prompt once
on the full54 before deciding whether8B is necessary. Preserve all results;
no train/val labels, detector parameters, shared environment or service state
are changed. Stop after this bounded fallback, even if neither judge suffices.
