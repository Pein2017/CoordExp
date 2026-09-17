# Corner-loop phase1: conditional exit remains available; cache mismatch unsupported

**Status: phase1 lead-accepted and closed.** Lead independently recomputed all363 slot decisions from all57 saved logit files and checked87 artifact bindings, literal prefixes, trusted-owner alternatives and process termination. [Acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/lead-acceptance.json) and [CPU verifier](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/lead_verify.py). All nine cells completed and all owned jobs ended. No training, free decoding, token-forcing rollout, embedding edit or KV intervention was performed. The worker's original machine-readable candidate result remains unchanged.

## Decision

Under the identical literal P history before477415 row138, Bnormalized64, P16 and R16 all prefer the saved exit category and coordinates: person,100,545,174,663. In particular, R has not lost this local conditional exit. This is not a full-suffix rescue claim: candidate-row interiors were supplied for scoring, and no suffix was generated.

The natural contrast18/17/2 known-owner matches is reverified. At the actual shared first P/R fork (row4 y1), P prefers688 and R670, while both select x1=0. Later conditional decisions also differ. These observations support history-dependent route selection/maintenance as the live distinction; they do not establish the first differing token as harmful or explain why R never reaches P’s exit history.

Crucial boundary: P rows132–136 remain chair [0,999,999,999]; row137 is person [0,0,999,999]; row138 first leaves x1=0. Thus the frozen exit boundary already includes the category/geometry bridge at row137. We did not add a fourth logit boundary to locate that earlier transition. The R natural rows132–139 remain corner chair rows.

## Raw validation and panel

| Image | P complete / x1=0 / invalid | R complete / x1=0 / invalid | First P/R row difference |
|---|---|---|---|
|477415|157 /137 /124|342 /342 /309|row4 y1:688 vs670|
|351017|308 /308 /234|308 /308 /0|row2 x2:17 vs20|
|417044|308 /308 /239|308 /308 /0|row2 y2:374 vs373|

Rows are complete regex spans including invalid geometry. P477415 contains122 chair [0,999,999,999] rows before exit; R contains291 such rows. Repeats/invalid rows are not removed before prefix construction.351017/417044 are pre-existing loops changing form, not demonstrated newly created failures.

Frozen boundaries:477415 rows4/16/138;351017 and417044 rows1/2/16. Every checkpoint consumes the SAME full literal P prefix. Internal prefixes of each complete candidate row are also identical across checkpoints. Alternative rows are whole trusted-bank rows, never coordinate mixtures. Canonical order is not claimed uniquely legal.

## Execution parity

363 scored slot decisions: zero cache/full argmax flips, zero duplicate-bs4/full-bs1 flips, zero material parity failures. Max slot logit discrepancy is 0.000890493; maximum 2ε/top1–top2 margin ratio is 0.054710. Thus every measured margin exceeds its worst-case two-logit discrepancy bound. Exact full no-op discrepancy is0; cloned-cache first-step no-op is bitwise equal. Max full-prefix bs4 shape discrepancy is 0.000386119.

Prompt/media/grid identities, every native cached MRoPE position and full recomputation position, cache lengths, FP32/SDPA and adapter/embedding bindings passed. Native execution used actual Qwen prepare_inputs_for_generation and prompt-to-boundary token-by-token cache updates. Full execution used existing exact_history_inputs. Coordinate input embeddings and all readout parameters hash-identically across all nine cells.

Batch limitation: primary cache/full is matched bs1. The shape diagnostic uses four duplicate image/history entries, not the original heterogeneous bs4 companions/padding. At the sampled observed slots P’s top tokens match the saved P row. These checks retire a material mismatch on this panel, not every possible original-batch or runtime bug.

## Slot contrasts

The following are conditional logits at the earliest P/R divergent slot under a shared history, not predictions assembled from incompatible coordinate paths.

| Image / slot | Bnormalized64 top (margin) | P16 top (margin) | R16 top (margin) |
|---|---|---|---|
|477415 row4 y1|<|coord_696|> (0.02759)|<|coord_688|> (0.04099)|<|coord_670|> (0.02998)|
|351017 row2 x2|<|coord_20|> (0.05204)|<|coord_17|> (0.06388)|<|coord_20|> (0.05275)|
|417044 row2 y2|<|coord_374|> (0.05207)|<|coord_374|> (0.01567)|<|coord_373|> (0.01718)|

x1 preference already favors0 before repetition in both secondary first rows, including otherwise valid person boxes. It remains0 across their selected later boundaries. R’s x1=0 margins are smaller than P’s at all these selected zero-x1 contexts despite R’s worse477415 natural outcome. Frequency or a large zero logit does not identify an embedding-origin cause.

| Image / boundary | Bnormalized64 x1 top / margin | P16 x1 top / margin | R16 x1 top / margin |
|---|---|---|---|
|477415 row4|<|coord_0|> / 0.85621|<|coord_0|> / 1.22978|<|coord_0|> / 1.15956|
|477415 row16|<|coord_0|> / 1.02396|<|coord_0|> / 1.19318|<|coord_0|> / 1.01278|
|477415 row138|<|coord_100|> / 0.32858|<|coord_100|> / 0.28917|<|coord_100|> / 0.28322|
|351017 row1|<|coord_0|> / 0.91388|<|coord_0|> / 1.25395|<|coord_0|> / 1.16308|
|351017 row2|<|coord_0|> / 0.90771|<|coord_0|> / 1.11515|<|coord_0|> / 0.97359|
|351017 row16|<|coord_0|> / 1.05220|<|coord_0|> / 1.33319|<|coord_0|> / 1.03147|
|417044 row1|<|coord_0|> / 0.98701|<|coord_0|> / 1.32456|<|coord_0|> / 1.23204|
|417044 row2|<|coord_0|> / 0.95503|<|coord_0|> / 1.35952|<|coord_0|> / 1.24824|
|417044 row16|<|coord_0|> / 0.86454|<|coord_0|> / 1.09159|<|coord_0|> / 0.89139|

At477415 row16, P and R both weakly prefer chair over person (both margins approximately0.012); x1 remains0. Under the observed chair/0 internal prefix, P y1 prefers999 while R prefers658. x2/y2 values in the complete slot table are evaluated under the supplied P row’s preceding coordinates, not under each model’s earlier argmax; do not concatenate these marginals into a generated box.

|477415 row138 slot| Bnormalized64 | P16 | R16 |
|---|---|---|---|
|category_1|person (1.52819)|person (1.74406)|person (1.75405)|
|x1|<|coord_100|> (0.32858)|<|coord_100|> (0.28917)|<|coord_100|> (0.28322)|
|y1|<|coord_545|> (0.11203)|<|coord_545|> (0.13529)|<|coord_545|> (0.17029)|
|x2|<|coord_174|> (0.27341)|<|coord_174|> (0.38308)|<|coord_174|> (0.38358)|
|y2|<|coord_663|> (0.05013)|<|coord_663|> (0.14795)|<|coord_663|> (0.12136)|

At this exit x1 context, coord_0 ranks12/11/19 for B/P/R; R’s zero logit is1.9356 below coord_100. R also favors person over chair by1.7541. Thus strong corner preference is not invariant under history and category conditioning. The frozen trusted alternative is owner1307155/person [105,647,307,866]; its own whole-row prefixes were scored separately. At rows4/16, chair wins the category decision, but given that coherent person candidate’s description, all three prefer x1=100 over0. This is conditional compatibility, not proof that the exact trusted owner will be emitted or matched.

All category subtokens, four coordinate slots, admission/EOS ranks, trusted alternatives and actual candidate token ranks are available in the complete slot table and raw receipts. EOS loses to the next-row opener at all selected observed boundaries by6.57–13.24 logits. This establishes local continuation preference only, not scene exhaustiveness or an EOS-causal explanation.

## Strongest alternative and ONE next causal discriminator — not executed

The strongest surviving alternative is that accumulated earlier history, including the row137 category/geometry bridge, determines escape; the same frozen readout can support both trajectories. Broader language-DoRA changes may alter both entry and later maintenance. This panel cannot separate visual grounding, recency/copying, repetition accumulation or an embedding prior, and cannot establish correctness of a local token.

Propose ONE477415/R16 bridge-sufficiency contrast: retain its own natural first136 complete rows and replace only row137 with the literal P bridge person [0,0,999,999], versus an identity/no-op replay of its native row137 chair [0,999,999,999]. Both rows have nine tokens; retain x1=0, x2=y2=999, positions, model and image. Then permit one natural greedy continuation per branch to the existing total cap3084, bs4 under identical batching policy. This tests the composite category+y1 bridge, not category alone, a correct owner target or coord_0 necessity. The wide person row is a diagnostic history, not admitted GT.

First require identity control reproduces its bounded natural behavior under matched runtime; failure is technical and stops interpretation. Primary causal evidence would be additional freely generated known-owner coverage with fewer invalid/repeated rows and no extra cap/EOS debt; do not credit the supplied bridge as a recovered owner. Count retained old owners, complete invalid rows, repeat proxy and termination jointly. If a short recent bridge rescues while x1 remains0, it supports local history sufficiency; if not, accumulated-history/model-maintenance explanations survive. One pair, fixed horizon, no dose/seed/row/embedding/KV sweep. This phase merely recommends it; further execution needs lead decision.

## Evidence, cost and ownership

Nine cells,27 boundaries,57 candidate rows,5343 model forwards and189 vision forwards;0 generated tokens. Sum single-GPU cell elapsed time 968.07s (0.2689 allocated GPU-hours), maximum cell 179.03s. These are allocated wall times including load/checks, not measured kernel time. First bounded real-entry cell ran once and is included; remaining eight ran concurrently. No GPU reruns or failed model cells. A host wait helper lacked os.pidfd_open; libc pidfd event-wait replaced it without affecting model execution.

- [Machine-readable result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/result.json), SHA256 `0bba4cc76e568ebcee6aea8e7ad2f49b03eb7ea4ed9ae7f6d611533bed01b3d9`.
- [Frozen panel](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/panel.json): exact prefixes, source generation IDs, checkpoint/config/data provenance and bank-owner alternatives.
- [Complete slot table](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/slot-table.tsv); per-cell `runtime/<checkpoint>-<image>/receipt.json` binds saved FP32 full/cache/bs4 logit tensors.
- [Executed producer snapshot](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/producer.py), [launch](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/run.sh), [launch receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/launch.json), [reducer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/reduce.py), [closeout checks](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism/closeout.py).

Accounting clarification: the producer role named repeated_competitor at row138 scored the literal immediate predecessor row137, person [0,0,999,999]. Its source comment calling this a corner row is inaccurate. Literal tokens, raw logits and numerical results are preserved unchanged; no result treats it as the122-fold degenerate chair row. The row16 observed candidate separately scores the repeated corner chair.

All eight parallel exit files and parent exit are0; the initial cell also returned0. All nine receipts complete; no owned jobs remain. Predecessor outputs and unrelated dirty work remain unchanged. Lead acceptance and catalog/frontier integration are complete. The next causal pair is proposed only; phase2 has not run.
