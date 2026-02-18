---
name: full-pipeline-smoke
description: Build and run production-like smoke tests for new features by going through the full pipeline cycle (data -> transforms/packing -> training/infer/rollout -> metrics/artifacts), using YAML inheritance so smoke differs from prod only by a few runtime knobs (sample limits/max_steps/output dirs).
metadata:
  short-description: Full-cycle smoke workflow
---

# Full-Cycle Smoke Testing (Production-Like)

Use this skill whenever you add a new feature and want a **real smoke test** that exercises the **entire pipeline path** (not a unit test) while staying close to production conditions.

## Core Principle

**Smoke configs must be production configs with fewer samples.**  
Keep all main hyperparameters identical to production; only change runtime knobs.

This prevents “smoke drift” where smoke passes but production fails.

## What “Full Pipeline Cycle” Means

A real smoke test should pass through as many of these as relevant to the feature:

- Data reading + sample contract validation
- Template encode + multimodal handling (images/video, if applicable)
- Packing / position_ids contracts (if packing is enabled)
- Forward + backward + optimizer step (at least 1–3 steps)
- Any feature-specific stage (e.g., rollouts, matching, decoding, post-processing)
- Logging/metrics emission (at least one log line with the key metrics)
- Artifact creation (optional): minimal output dir, minimal vis dumps

## YAML Hierarchy Pattern (Required)

Use a consistent layout:

- Production configs live in a dedicated folder (e.g. `configs/<area>/prod/`)
  - One `base.yaml` defines shared hyperparameters
  - Variant configs override only the minimal “experiment axis”

- Smoke configs live under `configs/<area>/smoke/`
  - A single `common_prodlike.yaml` centralizes smoke runtime overrides
  - One entrypoint YAML per smoke scenario extends prod + common

Recommended inheritance order:

```yaml
extends:
  - ../prod/<variant>.yaml
  - common_prodlike.yaml
```

Reason: later entries override earlier ones, so smoke can override runtime knobs without touching prod hyperparams.

### Keep legacy paths stable (optional but recommended)

If the repo already has old smoke YAMLs referenced by docs/tests, convert them to 1–3 line wrappers:

```yaml
extends: smoke/<new_smoke_entrypoint>.yaml
```

## Allowed Smoke Overrides

Allowed keys to override in smoke entrypoints (or `common_prodlike.yaml`):

- `training.run_name`
- `training.output_dir`
- `training.logging_dir`
- `training.max_steps` (or `num_train_epochs` only for specific reasons)
- `training.save_strategy: "no"` (and keep `eval_strategy: "no"` unless evaluation is the feature)
- dataset limits:
  - `custom.train_sample_limit`, `custom.val_sample_limit`
  - or feature-equivalent limit knobs in the repo
- dataloader stability knobs:
  - `data.dataloader_num_workers: 0`
  - `data.dataloader_prefetch_factor: null`
  - `data.dataloader_persistent_workers: false`

Avoid overriding (unless the explicit purpose is to test these knobs):

- learning rates / optimizer settings
- packing configuration / max lengths
- model checkpoint / template
- decoding parameters (unless decoding itself is the feature under test)

## Smoke Design Checklist

Before writing any YAML:

1) Identify the **production config** that represents the “real run”
2) Decide the minimum number of steps to hit the feature path
   - e.g., if a schedule has multiple phases, ensure smoke reaches each phase at least once
3) Set sample limits so the dataloader can provide enough samples for those steps
4) Ensure length caps in smoke match prod (so truncation/packing contracts are exercised)

## Regression Loop: Feature Building -> Smoke -> Guardrail

Recommended loop:

1) Implement feature (code/config)
2) Add or update smoke YAML overlay(s)
3) Run the smoke command(s) and verify:
   - GPU is actually used (no hang)
   - At least one optimizer step completes
   - Key metrics counters/logs are present
   - No silent truncation/invalid parse spikes (feature-dependent)
4) Add/adjust a minimal test that runs the smoke config (if repo has such tests)
5) Document the canonical smoke command in the relevant runbook

## Runtime Guardrails (CoordExp defaults)

- Prefer YAML-first changes over adding new CLI flags.
- If local HTTP services are involved (server-mode rollouts), ensure proxy is disabled:
  - `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY`
  - ensure `NO_PROXY` contains `127.0.0.1,localhost`
- Avoid destructive git commands unless explicitly requested by the user.

