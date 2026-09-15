# Logit-lens interventions

Four explicit profiles share the frozen Source/overfit model pair and compact
object-row token conventions. The retained work is conditional on fixed generated
prefixes and selected Human13 images; it does not establish held-out or
native-from-prompt behavior.

| Entry | Maintained protocol | Original producer |
| --- | --- | --- |
| `python -m probes.logit_lens.base` | Image2299 Source/overfit generation and both exact-prefix replays; 28-layer lens with DeepStack boundary validation | `probe_image2299_logit_lens.py` |
| `python -m probes.logit_lens.causal` | Stage A Image2299 and Stage B remaining Human13 images; current-token, causal-prefix, and equal-norm random controls | `probe_logit_lens_causal_transfer.py` |
| `python -m probes.logit_lens.radius` | Blocks 24/27 radius-only, direction-only, and full-current factorial with block-28 controls | `probe_logit_lens_radius_direction.py` |
| `python -m probes.logit_lens.natural` | One block-27 direction-only prefill graft, followed by ordinary cached Source continuation | `probe_logit_lens_natural_continuation.py` |

Scientific records are indexed in [the research catalog](../../research/experiments/catalog.jsonl)
and preserved in `docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/` under the
`2026-09-08-image2299-logit-lens`, `2026-09-08-logit-lens-causal-transfer`,
`2026-09-08-logit-lens-radius-direction`, and
`2026-09-08-logit-lens-natural-continuation` directions.

Each model entry accepts a fresh `--output-root`; causal additionally accepts
`--stage-b-output-root`. Their default output paths are new `/tmp/coordexp-logit-lens-*-new`
locations, never historical run directories. Model work requires its separate
resource authorization; this migration performed no model forwards.

```bash
python -m probes.logit_lens.preflight --output-root /tmp/logit-lens-input-check
python -m pytest -q probes/logit_lens/tests
```

Preflight is a real CPU path: it loads the processor/tokenizer, prepares all 13
image/prompt inputs, verifies source/checkpoint/gate and predecessor receipt
hashes, checks exact input parity against saved baselines, reduces saved causal
traces, and resolves each natural-prefix anchor and remaining budget. It never
loads an executable model. Its destination must be absent.

Examples of the separate model profiles:

```bash
python -m probes.logit_lens.base --output-root /tmp/logit-lens-new/base
python -m probes.logit_lens.causal --output-root /tmp/logit-lens-new/causal-a
python -m probes.logit_lens.causal --stage-b-output-root /tmp/logit-lens-new/causal-b
python -m probes.logit_lens.radius --output-root /tmp/logit-lens-new/radius
python -m probes.logit_lens.natural --output-root /tmp/logit-lens-new/natural
```

Successor profiles deliberately consume their original hash-bound predecessor
artifacts. They are not an automatic pipeline that promotes outputs from a new
base run. All historical input hashes remain fixed. New executions record their
actual package helper paths/hashes with `helper_binding_scope` set to
`maintained_package_sources_at_launch`; no old code hash labels new executed code.

The source config and inherited base are byte-preserved. The public research
profile loader preserves their full effective config, including `debug.smoke=False`,
without applying the production-directory rule. Two required source-gate
evidence files are copied into `configs/` and retain the original SHA checks when
staged into a fresh output root. Their original paths are
`docs/history/architecture/proposals/2026-06-27-coordexp-infras/source-studies/special-token-embeddings.md`
and `docs/history/worktree-cleanup/2026-07-12-pvci-research-worktree-recycle/local-artifacts/69ed/outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json`.

The one-off `recover_attempt1` repair command is historical-only. Restore its
exact implementation from
`archive/research-restructure-20260909/image2299-logit-lens:scripts/research/probe_logit_lens_causal_transfer.py`
at source commit `269477a3a`; the maintained causal entry retains normal Stage A/B
execution and the cold `reduce_trace` consumer. No historical receipt is rewritten.

CPU tests retain the original scientific counterexamples and additionally reach
the real package causal-capture consumer and native continuation API. They check
pre-injection cloning, intervention locality, causal-row alignment, immutable
input capture, exact prefix IDs, one visual encoding, and cached followups.
Full-model numerical, generation, and scientific acceptance remain outstanding.

Independent-root acceptance passed 23 tests and exact original config/reader
parity across all 13 images and 60 sites, with every source-worktree file open
forbidden. No `scripts/` directory or sibling package was present.
