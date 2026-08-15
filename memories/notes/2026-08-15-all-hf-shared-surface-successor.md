# All-HF shared-surface successor

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
- no implementation, model, or GPU action has occurred yet.
