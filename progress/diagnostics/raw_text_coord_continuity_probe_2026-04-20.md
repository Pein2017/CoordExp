---
title: Raw-Text Coordinate Continuity Probe
date: 2026-04-20
status: consolidated-final
owner: codex
branches:
  - codex/raw-text-continuity-probe
  - codex/coord-family-comparison
supersedes:
  - output/analysis/raw-text-coord-continuity-probe-2026-04-18/report.md
  - output/analysis/raw-text-coord-continuity-probe-2026-04-18/summary.json
---

# Raw-Text Coordinate Continuity Probe

## Scope

This note is the permanent progress-layer archive for the raw-text continuity
study that was developed under the `codex/raw-text-continuity-probe` branch and
later carried forward into `codex/coord-family-comparison`.

The original runtime bundle lived under `output/analysis/` and would otherwise
be lost from git history because `output/` is ignored. This document records:

- the motivating question,
- the exact study contract,
- the key implementation and process decisions,
- the final verdicts,
- the important human review correction,
- and the stable paths of the copied artifact summaries that now live under
  `progress/diagnostics/artifacts/`.

## Research Question

The study asked whether a `raw_text_xyxy` grounding model trained with pure CE
already preserves numeric / coordinate continuity, despite being supervised
through discrete cross-entropy rather than explicit coordinate expectation.

The working hypothesis had two parts:

- good continuity:
  - GT-centered local basins may already exist in raw text and may help
    localization
- bad continuity:
  - the same continuity may also create wrong local basins around an incorrect
    prefix or previous same-class instance

The downstream decision question was narrow and explicit:

- if the only desired benefit is local coordinate continuity, is `coord_token`
  still necessary?

## Study Contract

The final study contract matched the 2026-04-18 super-power spec:

- source spec:
  - `docs/superpowers/specs/2026-04-18-raw-text-coord-continuity-probe-design.md`
- output surface:
  - `pretty_inline` JSON
- active contract:
  - `{"objects": [{"desc": ..., "bbox_2d": [x1, y1, x2, y2]}, ...]}`
- tokenizer assumption after audit:
  - digit-by-digit tokenization for numbers like `199`, `200`, `210`
- scoring principle:
  - **candidate coordinate sequence scoring**
  - not single-token logit approximation
- comparison points:
  - base `Qwen3-VL-2B-Instruct`
  - raw-text pure-CE Stage-1 checkpoint
- control lanes:
  - GT-centered basin probe
  - lexical confound control
  - image-swap control
  - self-prefix bad-basin probe
  - prefix-geometry perturbation

## Process Decisions That Mattered

Several method choices ended up being decisive. These are worth recording
because they materially changed the trustworthiness of the conclusions.

### 1. Audit first, then probe

The study did not assume the output surface. It first audited:

- raw-text serialization shape
- tokenizer behavior on `0..999`
- compact vs pretty-inline JSON ambiguity

The final answer was:

- use `pretty_inline`
- treat numbers as digit sequences, not whole-number tokens

### 2. Sequence scoring instead of token-logit shortcuts

The central methodological safeguard was to score an entire candidate
coordinate string under a fixed prefix:

- `score(k | prefix, image)`

This prevented false conclusions caused by fragmented tokenization of
multi-digit integers.

### 3. Human review corrected an important over-interpretation

The manual bbox-first review showed that:

- a pred-centered basin is not automatically a wrong-instance basin
- one hard repeated-object case was truly wrong-instance
- another strong pred-centered case was still semantically correct

This made the final wording of the bad-basin conclusion more careful and more
honest.

## Final Verdicts

The copied runtime bundle and final report support the following five verdicts:

1. Base Qwen3-VL already has raw-text numeric adjacency / coordinate continuity:
   `strongly supported`
2. Stage-1 pure-CE fine-tuning enhances that continuity:
   `partially supported`
3. Continuity is stronger under the correct image than under swapped-image
   controls:
   `strongly supported`
4. Hard repeated-object cases can form wrong local basins around the wrong
   prefix, especially at `x1/y1`:
   `strongly supported`
5. If the goal is only local continuity, `coord_token` remains necessary:
   `not supported`

## High-Signal Evidence

The strongest pieces of evidence worth preserving are:

- lexical control:
  - combined numeric-distance coefficient stayed negative and significant after
    controlling lexical similarity
  - this supported a real numeric / ordinal prior rather than a pure string
    artifact
- image control:
  - correct-image conditions produced positive GT-score lift relative to
    swapped-image controls
  - this separated visual grounding from pure language prior
- bad-basin causal intervention:
  - prefix-geometry perturbation showed that editing the previous instance's
    local geometry could move the basin of the current instance
- manual review:
  - human bbox review separated "pred-centered but correct" from
    "pred-centered and wrong-instance"

## Final Interpretation

The narrow conclusion of this study is:

**raw-text pure-CE does not need `coord_token` in order to exhibit local
coordinate continuity.**

The broader interpretation is:

- continuity is already present
- continuity is visually modulated
- continuity can still be harmful under dense repeated-object prefix geometry

Therefore, if `coord_token` is still justified, the justification is likely to
be about:

- typing discipline
- decoding stability
- parameterization cleanliness
- or instance-separation geometry

not about continuity creation alone.

## Durable Artifact Map

The runtime-root summaries copied into `progress/` are:

- copied report:
  - [progress/diagnostics/artifacts/raw_text_coord_continuity_probe_2026-04-18/report.md](artifacts/raw_text_coord_continuity_probe_2026-04-18/report.md)
- copied summary:
  - [progress/diagnostics/artifacts/raw_text_coord_continuity_probe_2026-04-18/summary.json](artifacts/raw_text_coord_continuity_probe_2026-04-18/summary.json)
- copied human findings:
  - [progress/diagnostics/artifacts/raw_text_coord_continuity_probe_2026-04-18/manual_review/human_findings.md](artifacts/raw_text_coord_continuity_probe_2026-04-18/manual_review/human_findings.md)

The original study design references remain in:

- `docs/superpowers/specs/2026-04-18-raw-text-coord-continuity-probe-design.md`
- `docs/superpowers/plans/2026-04-18-raw-text-coord-continuity-probe-design.md`

## Bottom Line

This study permanently changes the burden of proof around `coord_token`.

The default assumption should no longer be:

- "`coord_token` is required so the model can have local continuity"

The better current assumption is:

- "raw-text grounding already has continuity; the remaining question is which
  parameterization yields the best combination of stability, recall, and
  repeated-instance separation."
