---
title: PVCI causal-proposal-bridge raw-material synthesis
type: investigation
role: archival-synthesis
authority: non_normative_research
status: complete_not_promoted
updated: 2026-08-29
---

# PVCI causal proposal bridge

## Session context

Session `019f4fc1-7c49-76b0-bc25-9ce0eacc77d1` froze a representation screen
before any own-prefix causal claim. The question was whether a supervised
pre-row visual proposal changes free-rollout behavior because the decoder
causally consumes it, rather than merely because an auxiliary loss reshapes
representations.

## Distilled result

The matched A/B/C 512-step screen completed. B and C both passed the applicable
held-out checks; tie-aware selection chose C, authorizing only a narrower causal
follow-up. The unit remains `promotion_status: not_promoted`: no detection,
recall, duplicate, STOP, own-prefix, or final-architecture claim follows from
this screen.

Canonical owner:
`/data/CoordExp/.worktrees/research-probes/research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-causal-proposal-bridge/`.

## Raw disposition

The 14G output root contains 24 rank-local subtrees, proposal-row JSONL ledgers,
runtime training receipts, and three arm families of checkpoints. The named
step-512 checkpoints and immutable receipts are enough for historical replay;
earlier steps and rank-local duplicate payloads add no new decision after the
canonical result. This pass therefore marks the entire raw root
`DISTILL-DELETE` (no active handles were present at the last live check).

No global archive checksum was recovered. The deletion receipt records the exact
path and byte count; restoration would require regenerating the run rather than
relying on a hidden archive.

