# Contract Diff And Loss Footguns

Use this when a planned or newly wired mechanism could silently train or evaluate the wrong contract.

## Minimal Contract Diff

| Layer | Evidence To Capture | Verdict |
| --- | --- | --- |
| Intended contract | design/spec/plan, expected math, launch intent | matched / mismatched / unproven |
| Authored config | YAML leaf, inheritance chain, explicit overrides | matched / mismatched / unproven |
| Resolved contract | materialized config, schema dataclass, defaulted keys | matched / mismatched / unproven |
| Runtime contract | dataset, collator, trainer, loss, decode/eval objects | matched / mismatched / unproven |
| Artifact contract | manifests, resolved config, metric keys, parser/drop counters | matched / mismatched / unproven |

Do not treat a smoke as trustworthy if any layer silently falls back to legacy behavior.

## Config Checks

- Dicts may deep-merge while lists replace wholesale; inspect the final resolved list.
- A leaf config that changes one inherited objective must restate sibling objectives when list replacement applies.
- Reject legacy semantic keys instead of warning-only no-ops.
- Check production and ablation surfaces for unsafe shared defaults.
- Confirm materialized configs record the objective identity that actually trained.

## Loss Checks

- Verify the differentiable scalar, not only logged labels.
- Compare raw terms and effective weighted contributions.
- Check zero-weight targets, support size, target membership, duplicate multiplicity, and EOS/type-gate composition.
- Use fp32 for probability/log-probability math even when model forward uses bf16.
- Prefer deterministic tiny-logit scalar tests before trusting aggregate loss curves.

## Numerics Receipt

For each changed loss term, record raw and weighted values, valid-count
denominator, mask density, zero-mask behavior, target support membership,
finite checks, dtype/fp32 islands, gradient path, accumulation and distributed
reduction semantics, plus metric/logging names. The gate is complete only when
one authored config, one resolved config, one encoded sample, one collated
batch, and one deterministic tiny-logit formula probe agree.
