# Progress Notes Conventions (Suggested)

This folder contains research notes and experiment records written along the way.
The goal is to keep them:

- paper-ready (clear provenance, decisions, and verification steps),
- reproducible where possible,
- robust to cleanup (some run artifacts may be pruned over time).

## Minimal Header (Recommended)

At the top of each progress note (`*.md`), add a small metadata block right after the title:

```text
Date: YYYY-MM-DD
Last updated: YYYY-MM-DD
Note: referenced run artifacts may be pruned; paths are best-effort pointers.
```

Guidance:

- `Date` is the anchor date of the experiment/idea being recorded (often matches the filename date).
- `Last updated` is the canonical "last written" sort key.
  Update it whenever you make a meaningful edit (new results, revised conclusions, new decisions).
- The `Note` line is optional but recommended because `output/` and external checkouts can be cleaned.

## Filenames

We prefer stable filenames to avoid breaking cross-references.

- If a note is tied to a particular day or run, include the date in the filename, e.g.
  `stage2_ab_softctx_discretization_vs_stage1_bbox_losses_2026-02-22.md`.
- If a note becomes long-lived (living document), keep the original name but use `Last updated` to reflect recency.

## What To Record (Practical)

When you reference artifacts/configs, include handles that make the note usable even if some paths are missing:

- config path(s): `configs/...yaml`
- checkpoint/run dir (best-effort): `output/...`
- evaluation settings (seed, decoding params, dataset slice/limit)
- a short decision + verification checklist

Avoid relying on fragile context like "the latest run" without an explicit date or directory name.

## Document "Types" (Informal)

We tend to write a few recurring types of notes:

- Bench / comparison: measured tables + fixed eval settings + provenance directories
- Audit: loss/logging alignment, diagnosis of scalars, efficiency notes, recommended ablations
- Diagnosis: failure modes, root causes, and config-first mitigation plans
- Runbook pointer: short stubs that point to authoritative docs under `docs/`
- Infra request: exploration tasks / integration plans (may later be promoted into `docs/` if they become stable guidance)

You don't need to label types explicitly, but keeping the structure consistent makes notes easier to scan later.

