# Natural output-row norm equalization: four-image diagnostic

**Candidate; technical execution and independent CPU reduction/verification passed. Lead acceptance pending.** The fixed panel is complete and all owned jobs ended. No normalization variant, training or follow-up was launched.

## Scientific verdict

This single fixed output policy produces useful natural known-owner gains on477415 and417044, retaining every incumbent and ending with EOS without invalid/malformed rows or repeats. It does not repair351017, which still caps with heavy valid-box repetition. The healthy7116 damage check retains the exact same4 owners, clean EOS and37-token length. These are selected-case causal policy effects, not population rates or a promoted general repair.

The intervention changes ONLY the deployed output policy, not learned parameters. It supports a contribution of measured output-row norm structure to harmful rollout behavior on two selected images. It cannot establish that endpoint norms alone caused the original loops: the first useful-policy forks on477415/417044 are INTERIOR coordinate choices, and the policy scales all1000 coordinates globally, including their competition with non-coordinate tokens.

## Natural known-owner coverage and retention

| Image / fixed bank | Matches baseline → treated | FN baseline → treated | New / lost incumbent | Retained incumbent IDs | Tokens baseline → treated | Stop baseline → treated |
|---|---:|---:|---:|---|---:|---|
|351017 / 25|1 → 1|24 → 24|0 / 0|466970|3084 → 3084|length → length|
|417044 / 17|2 → 14|15 → 3|12 / 0|515293, source256-unlabeled-candidate-f854c4fe2c46cc3ed748|3084 → 320|length → im_end|
|477415 / 27|2 → 18|25 → 9|16 / 0|1285340, 1580313|3084 → 262|length → im_end|
|7116 / 6|4 → 4|2 → 2|0 / 0|1375632, 1739499, 176783, 176844|37 → 37|im_end → im_end|

No image loses an incumbent owner; gains are not replacements hidden by an aggregate match count. Full gained/lost/retained identity sets, matches and missing owners are in result.json.417044’s gained set includes one already-admitted `source256-unlabeled-candidate-28ab9dcf93faedebb94a`; another existing admitted owner is retained. No new owner or label was introduced. These remain frozen-bank IoU50 matches, not an exhaustive physical scene census.

## Output debt and recurrence

| Image | Invalid complete B→T | Malformed B→T | Strict valid repeats B→T | Literal invalid repeats B→T | UNKNOWN B→T | Endpoint token share B→T |
|---|---:|---:|---:|---:|---:|---:|
|351017|0→0|1→1|297→285|0→0|307→307|50.00%→34.66%|
|417044|0→0|1→0|266→0|0→0|306→18|43.43%→1.56%|
|477415|309→0|1→0|16→0|302→0|31→11|91.68%→9.48%|
|7116|0→0|0→0|0→0|0→0|0→0|6.25%→6.25%|

477415 produces29 distinct valid complete rows,18 known matches,11 UNKNOWN predictions,EOS;417044 produces32 distinct valid complete rows,14 known matches,18 UNKNOWN predictions,EOS. Their shorter sequences are not counted as success alone: owner gains/retention and eliminated repetition/debt accompany stopping. UNKNOWN stays unresolved and is not a physical FP or precision certification.

351017 is the counterexample to endpoint-rate-only reasoning: endpoint share drops50%→34.66%, but coverage remains1/25 and all308 complete boxes have x1=0. Literal repetition starts at row4 under BOTH policies. Its longest exact run remains84 rows (baseline bottle[0,0,60,76], treated bottle[0,87,91,123]); literal repeats292→279 and strict repeats297→285 remain severe. It still emits a malformed tail and caps.

Baseline417044 first literal repeat is row15 and its longest run134 rows; treated has no repeat. Baseline477415 first repeat is row9 and its longest run is291 copies of chair[0,999,999,999] from row52; treated has no repeat. Healthy7116 has no repeats/debt under either policy; its endpoint count stays1/16. Complete per-role0/999 counts, raw row sequences, first strict/literal repeats and exact run records are saved, not inferred from a chart.

## First forks: full-vocabulary evidence

| Image | Zero-based token / field | Original → treated coordinate |
|---|---|---|
|351017|7 / first-row y2|999→990|
|417044|5 / first-row y1|406→405|
|477415|14 / second-row y1|811→809|
|7116|13 / second-row x1|285→286|

All four first forks occur on identical literal prefixes; native-before argmax equals the original token, and scaled-after argmax equals the treated token. Raw full-vocabulary before/after tensors and top5/EOS details are retained. EOS logits are bitwise unchanged at the seam. Nevertheless scaling coordinate logits changes their competition with EOS/categories/syntax over the rollout, so a stopping effect cannot be attributed solely to relative coordinate ranks. The first fork locates policy separation, not a unique causal circuit or a sufficient single-token edit.

## Policy and technical acceptance

Effective FP32 W is base+shared delta at the actual SelectedDeltaOutputHead. `.weight` alone would omit that delta. Bias absence and tied base/shared delta were reasserted live. Fixed FP64 median norm is1.3125505117169813; factors span0.9406469966–1.0246542747, with coord_0=0.9672321747 and coord_999=0.9406469966. All1000 coordinate logits in every sample/step are multiplied in FP64 and rounded back to native logit dtype. Every non-coordinate logit is bitwise unchanged. This is not an endpoint ban or parser-slot intervention.

Actual input/output row hashes, base/delta hashes and every model parameter version counter agree before/after generation. Model modules, input parameters, DoRA, image/KV handling and checkpoint bytes were not altered; only the transient native logits processor changes returned scores. Different generated tokens subsequently change hidden states naturally. This does not imply input embeddings are causally irrelevant.

All4 identity batches reproduce all16 saved member token sequences/stops exactly. Original heterogeneous companions/order/padding/prompt/image/native MRoPE,FP32-SDPA/RP1 and max3084 are preserved. Treated policy applies to companions too, but scientific cohort remains ONLY the4 frozen images. No supplied history, row replacement or oracle target occurs. Each identity gated its treatment; four independent paired processes used four GPUs without redundant batches.

The accepted offline first-row counterfactual was checked on12 available coordinate contexts in identity runs and10 still-identical-prefix treated contexts (22 comparisons, NOT22 scientific trials). Maximum scaled-logit discrepancy=6.50182283e-05; maximum2*error/margin=0.001513812; all conditional argmaxes agree. Saved CPU verification also rechecks16 first-fork tensors across batch members: coordinate formula exact, every non-coordinate/EOS value exact, and focus-owner G/L sets recomputed.

The native parser, class-agnostic one-to-one IoU50 matcher and frozen per-image bank are unchanged. Label-string compatibility is separate: treated compatible matches1/14/18/4 on351017/417044/477415/7116 respectively. Formal verified-class flags are not manufactured; no unmatched prediction is upgraded.

## Cumulative mechanism synthesis and limits

1. Accepted cache/full checks closed measured execution-parity seams; the coordinate input/output rows are shared across current checkpoints. Endpoint frequency is not a clipping/drawing artifact.
2. Conditional token bridges selectively recovered owners but later relapsed; both999→998 one-bin edits failed. A particular history change can open a useful route, but arbitrary token disruption is not sufficient.
3. Offline equal-norm analysis removed17/69 endpoint wins while52 survived, identifying both readout-scale contribution and context-specific alignment. Norms alone did not explain all selected slots.
4. Natural donor substitution changed both conditioned loop forms without donor-known recovery. That established visual-input sensitivity but left foreign-history mismatch unresolved; this empty-prefix panel avoids that mismatch.
5. Fixed all-coordinate norm equalization now permits useful naturally generated, terminated routes on two original failure images with incumbent retention, while351017 persists and the one healthy image shows no measured damage. This links readout structure to rollout outcomes but does not separate early-route prevention from later maintenance effects, generic trajectory sensitivity from special endpoint effects, or coordinate-vs-full-vocabulary competition.

No claim of training-origin causality, pretraining frequency explanation, generalization, full physical precision/recall, safe deployment or learned ordinary-parameter repair follows. Existing UNKNOWN predictions remain an evidence boundary. No automatic strength/endpoint/input variant or training is authorized by this result. STOP; any further work requires a genuinely decision-changing new question selected by lead.

## Cost, mechanical repair and stable evidence

Executed8 batches,13086 model forwards,8 vision calls;1639.239 allocated GPU-seconds (0.45534 GPU-hours), peak reserved17706254336 bytes; launch-to-closeout993.3s. All8 execution exit codes0; no GPU retry. Within25000 calls and both2-hour ceilings. All owned jobs ended.

A CPU reducer path-binding helper initially received a string rather than Path; the exact pre-fix reducer and failure/fix receipt were preserved, then only that conversion was corrected. Model outputs, scientific scoring and all GPU work were unchanged. Fresh reduction and independent saved-fork/G-L checks passed. This resolved mechanical issue is not a scientific null.

- [panel.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/panel.json)
- [coefficients.pt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/coefficients.pt)
- [producer.py](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/producer.py)
- [reduce.py](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/reduce.py)
- [result.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/result.json)
- [verify_saved.py](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/verify_saved.py)
- [saved-verification.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/saved-verification.json)
- [reduction-repair.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/reduction-repair.json)
- [run.sh](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/run.sh)
- [launch.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/launch.json)
- [terminal.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/terminal.json)

Predecessors: [token-edit series](../2026-09-16-corner-loop-y1-one-bin-control/results.md), [readout/state decomposition](../2026-09-16-endpoint-loop-readout-state/results.md), [natural image/history contrast](../2026-09-16-endpoint-loop-image-history/results.md). All predecessor result/terminal/producer bytes were preserved.

Technical status: valid fixed-policy natural generation and saved-output reduction. Scientific status: bounded positive contribution on two cases, persistent failure on one, no measured damage on one healthy control; candidate pending independent lead acceptance.
