---
title: Inference Evaluation and Artifact Lessons
type: investigation
role: lessons
authority: non_normative_research
status: historical-synthesis
updated: 2026-07-17
---

# Inference, Evaluation, and Artifact Lessons

This page is a compact failure-aware checklist for research probes. It does not
replace the current [`docs/eval/WORKFLOW.md`](../../../docs/eval/WORKFLOW.md),
the evaluator contract, or stable inference specs. Those owners are linked, not
copied.

## Lessons

| ID | Failure pattern | Evidence handle | Required interpretation / cheap check | Currentity |
|---|---|---|---|---|
| INFER-001 | A high AP row is accepted without checking raw generation. | `progress/benchmarks/2026-07-03_coord_token_val200_benchmark.md`; 79 metric roots and 60 coordinate-token/compact-full val200 rows were censused. | Require raw rollout, `summary`, resolved config, confidence post-op, duplicate report, and evaluator inputs. If any are missing, downgrade the claim. | Reusable artifact rule; current paths belong to `docs/eval/WORKFLOW.md`. |
| INFER-002 | Grammar-constrained output is interpreted as natural rollout behavior. | Historical free-decode diagnostics and painted-GT decode contract. | Record `decode_surface` and effective backend kwargs. Keep grammar/trie rows as sensitivity panels; do not mix them into primary free-generation gates. | Currentity depends on backend/spec; historical constrained rows are not primary evidence. |
| INFER-003 | Invalid or malformed rows are removed before scoring. | Painted-GT decode/eval OpenSpec and native baseline benchmark. | Invalid, malformed, unparseable, empty, and truncated outputs remain in recall/debug denominators. Parseable-only tables are secondary diagnostics. | Reusable contract; verify implementation in the target worktree. |
| INFER-004 | Confidence post-op or visualization is treated as a new inference run. | `docs/eval/WORKFLOW.md`; infer comparison galleries and A5/A6 poor-IoU manifests. | Distinguish parent inference artifacts from child post-op/visualization views. Preserve parent run identity and never count transformed children as independent runs. | Current evaluation workflow rule. |
| INFER-005 | Val200 rows are compared despite different order, dataset, wrapper, or decode. | `progress/benchmarks/stage1_2b_val200_leaderboard.md` and coordinate-token snapshot. | Use `comparable_group` plus dataset/slice/row-id, template/order, checkpoint/export, decode, parser, and evaluator identity. Reject cross-group topline claims. | Reusable registry rule. |
| INFER-006 | Duplicate predictions are described as hallucination without manual or class-aware review. | `outputs/research/qwen3-vl-dense-enumeration/.../mask-reset-vs-full-bag.../manual-review.md`; Stage-2 duplication registry. | Separate exact duplicate, near duplicate, fragmented localization, category mismatch, real unlabeled entity, and unsupported hallucination. Report exact/near thresholds and affected rows. | Reusable taxonomy; thresholds remain study-specific. |
| INFER-007 | Oracle-K is treated as a deployable ensemble result. | `progress/benchmarks/2026-03-11_stage2_oracle_k_first200.md` and current evaluation workflow. | Oracle-K is additive support/recoverability analysis. Preserve baseline-FN denominator, K, temperature/seed arms, record order, and `ever_recovered`/`recover_fraction`; do not call it single-decode quality. | Historical and current workflow distinction. |
| INFER-008 | A sampling or repetition-penalty sweep is generalized from one checkpoint. | `progress/benchmarks/2026-03-11_stage2_rollout_temperature_refinement.md`; native text-coordinate sweep. | Keep checkpoint, subset, seed, batch, max tokens, stop policy, and parser fixed. Decode policy is a factor, not a universal recommendation. | Reusable experimental-design lesson. |
| INFER-009 | Row-level visualization is promoted to population evidence. | Infer `vis_poor_raw_iou50_top12/manifest.md` and painted-GT route reports. | Galleries and no-GPU route reports are selection/diagnostic receipts. They need population metrics, denominators, and matching rules before supporting a broad claim. | Receipt-vs-result boundary. |
| INFER-010 | A self-prefix improvement claim lacks a frozen paired baseline. | Painted-GT decode/eval OpenSpec; current research-unit contracts. | Require matched slice, schedule, source, model/tokenizer/processor, decode, and condition identities. Without frozen self-prefix, report safety only, not improvement/degradation. | Contract reference; execution must be revalidated. |

## Minimum inference/eval receipt

For each run, preserve: input JSONL and image-root identity, row-id/order hash,
checkpoint/adapter identity, prompt/template/order, effective generation kwargs
and decode hash, raw suffix/token trace, parser and stop reasons, normalized
rows, invalid/malformed/truncation counters, confidence/post-op provenance,
evaluator version, matching rule, denominator counts, metrics, and artifact root.
The current workflow owns file names and commands; this page only defines what a
research claim must prove.
