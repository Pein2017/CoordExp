# All-HF shared-surface successor

## Retired (2026-08-24) — historical continuity only

This note is superseded and is not current authority. The route it describes
executed partially and is retired; see the closeout status in
`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/unit.md`.
Durable partial-execution evidence is the 2026-08-21 K16 acquisition/replay
receipt bound in that unit's closeout. Current successor authority is
`memories/current.md` plus
`memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md`. Do not use
the line below as evidence that no GPU action ever occurred on this route.

The user selected the shared-surface fork after the cross-engine exact-policy
unit closed.  The new authority is
`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/unit.md`
with implementation contract
`openspec/changes/add-human13-all-hf-shared-surface-trajectory-credit-vertical/`.

Continuity only:

- pay extra sampling compute to remove vLLM-to-HF policy drift before judging
  the algorithm;
- use one live HF BF16/FA2 model object, no-cache stepwise K16 sampling, and
  vectorized grad replay guarded by the unchanged `0.02/0.002` parity limit;
- after parity, run one complete trajectory-credit + compiler + preservation
  update on image 1584 and audit clean greedy at RP 1.0 and 1.10;
- one-image completion is decision-bearing even if null/unsafe; only zero G
  loss plus positive protected H/net gain and no new duplicate/malformed/cap
  burden may expand to 13 images;
- historical framing at the time this note was written: no implementation,
  model, or GPU action had occurred yet. This is no longer true — the route
  later executed partially (Source audits and K16 acquisition/replay) and was
  then retired before an admitted objective/backward/optimizer step; see the
  retirement banner above.
