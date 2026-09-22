# Reusable artifact index

Owning unit: `/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-17-readout-norm-fresh128/`.
Status: closed candidate. All72 model executions and independent CPU verification completed. Visual review was stopped by the user and remains partial/selective; no further work is running.

- `panel.json`: exact128 fresh +4 diagnostic focus IDs, native bs4 groups/companions, cases, prompt/config, checkpoint/embedding source bindings, fixed known-owner banks, coordinate IDs, coefficients and producer hash.
- `cohort/`: exact original selected records, source/exclusion inventory, deterministic selection, original annotation targets and CPU provenance checks. `runtime-records.jsonl` rebases paths and sorts the SAME objects for the native x-y template; `cohort-acceptance.json` verifies identity/object multisets.
- `coefficients.pt`: FP64 norms, fixed median and factors, effective row SHA. `effective-readout.pt`: actual effective FP32 input/output coordinate rows, bias and IDs, copied from accepted tensors and matched by runtime SHA; no new model call. Input/output shared storage stays unchanged.
- `runtime/group-<key>/O/raw.json` and `N/raw.json`: complete native emitted tokens, literal text, image/request IDs and EOS/cap. Boxes, including invalid rows/fragments, are reproducible via the bound native parser.
- `runtime/group-<key>/{O,N}/receipt.json`: model/runtime identities, exact prompt token IDs, image grids/media and input tensor hashes, left-padding via inputs, prefill MRoPE hash, decoding settings, producer/panel/raw hashes, cost and status. Both policies share native inputs/positions. Old diagnostic receipts additionally verify exact prior outputs/stops.
- `N/receipt.json` → `shadow.samples.<image_id>.steps`: EVERY active step on the SAME N history, zero-based offset, prefix hash, field/role, unscaled full-vocabulary winner, scaled winner and both top2 margins. EOS included; post-EOS padding excluded. These are not O-trajectory decisions after the policies diverge.
- `N/first-shadow-<image_id>.pt`: sparse first disagreement only, full-vocabulary logits before/after, final actual lm_head input hidden state, token IDs and offset. At most one capture per generated image, including separate diagnostic companions. Linked tensor SHA in N receipt. Full raw token prefix + native prompt/position bindings reconstruct context.
- `result.json`: independent saved-output parser/matcher reduction with boxes, prediction IDs, known matches/misses, exact per-owner gains/losses/retention, valid/invalid/literal repeats, UNKNOWN, EOS/cap, first divergence, endpoint occupancy and shadow summaries. Fresh and diagnostic populations separate.
- `physical-review/selection.json`, blinded per-image packets and final reviews: preselected32-image stratified sample and sampling metadata preserved; user stopped exhaustive review, so actual partial/selective coverage and unresolved outputs are reported in amendment.json and reviewer returns. Do not use incomplete review for weighted population claims; never rewrites known banks.
- `terminal.json`: final candidate status and authoritative artifact hashes/command/exit/cost/verification links, written at closeout.

## Explicitly absent

No full-sequence logits or hidden states beyond first shadow disagreement, no comprehensive KV cache, layer/head activations or attention probabilities, no training gradients/optimizer, no extra model forwards for shadow capture, no scene census/new GT. If an image has no shadow disagreement it has no sparse disagreement tensor. Sparse evidence supports conditional readout analysis and subsequent targeted replay, not a complete causal circuit reconstruction.
