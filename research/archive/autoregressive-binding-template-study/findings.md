---
type: investigation
title: Autoregressive Binding Template Study Findings
status: synthesized
domain: autoregressive-binding-template-ablation
updated: 2026-07-01
---

# Findings

## Durable Read

The stable conclusion across the absorbed records is that many detection failures are not explained by one generic late residual collapse. The records repeatedly separate several mechanisms:

- **Coordinate-basin routing is causal but site-sensitive.** Residual state can move coordinate-token probabilities, but the strongest route/value evidence often depends on entering through the right attention or layer-output site.
- **Visual evidence and value contribution are not the same thing.** Some regions attract attention without helpful value contribution; some route/value pockets are constructive while nearby pockets are destructive.
- **False negatives can be prefix-state or coordinate-basin failures.** Several missing objects remain conditionally recoverable under coordinate guidance or hidden-state patching, so they should not be treated as pure visual invisibility.
- **Descriptor repair is insufficient by itself.** Later June records show that descriptor-logit repair, selected-instance binding, current-box repair, and next-row routing can diverge.
- **x1 can behave like a basin key for later coordinates in selected cases.** The June 26-27 prefix-denoising records suggest that forced x1 history can alter y1 route/readout behavior, with sharp train/val and sample-specific differences.

## Evidence Limits

Most records are tiny, case-study, selected-panel, or smoke evidence. They are valuable for mechanism generation and probe design, not population validation. The synthesis should not be cited as benchmark evidence unless a linked artifact explicitly reports a population metric.

## Operational Consequences

Future probes should preserve validity counters, prefix condition, coordinate slot, layer/site, and source-region definitions in the same artifact. The investigation repeatedly shows that removing one of those axes makes the result easy to overinterpret.
