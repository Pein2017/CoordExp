## Why

Current dense-enumeration checkpoints often preserve visual evidence for objects that never win a natural decode opportunity: forced or sampled paths can recover owners that greedy decoding misses, while row-local representations can identify the current owner without reliably reallocating probability mass after that row is emitted. The next production training should therefore give the model a persistent, causal owner interface between late visual representations and autoregressive row generation, rather than add another training-only probe or rely on an external covered-set controller.

## What Changes

- Add a **permanent owner bridge** that is present in both training and HF inference. It exposes `K=4` locally anchored owner atoms per merged visual carrier, routes from a final-layer boundary query, softly informs native continue/stop admission, and hard-latches one owner only for the lifetime of an emitted object row.
- Train Stage 1 entirely with teacher forcing from the accepted four-coordinate `geo_sorted_xy` step-2444 checkpoint. The four full presentations are `geo_sorted -> random-1 -> geo_sorted -> random-2`; random permutations are deterministic, artifact-bound, and refreshed between their two presentations.
- Keep in-training evaluation on the same teacher-forced forward/loss path as training and run it at the midpoint and end of each presentation: eight predetermined planned-step landmarks at approximately 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%, and 100%. Every landmark uses the same fixed `geo_sorted` eval rendering; natural greedy HF evaluation remains a separate post-checkpoint observation.
- Teach set-level awareness during Stage 1, not only during a future self-prefix phase. Router supervision rewards total probability mass on any known-uncovered owner under each gold prefix and never assigns a mandatory next-owner order.
- Add joint autoregressive, atom, set-routing, and owner-use objectives. A same-image causal swap requires the chosen owner value to change both description and geometry likelihood in the corresponding direction; every ordinary COCO unmatched atom remains unknown-neutral in the first production run.
- Declare the first production run's annotation trust class explicitly as `ordinary_partial`, matching the S checkpoint's existing COCO-80 source with the accepted 1024-pixel processor budget and `global_max_length=12000`. Trusted-exhaustive negative/null branches are implemented and mechanically tested but receive no production examples in this run; no unreviewed refined panel is silently mixed in.
- Write the active row owner after model block 20 and let blocks 21-27 form owner-conditioned row KV state. Read the next boundary query after the final RMS normalization so it can observe those upper-layer writes. This seam is a Qwen3-VL-2B architecture profile, not a free training hyperparameter.
- Preserve the native autoregressive path and native opener/stop decision. Boundary availability is a bounded soft signal; after native `<|object_ref_start|>` emission, a row-local hard atom latch is held through `<|box_end|>`. No persistent external ledger, decode-time coverage mask, forced next-owner operation, or new special token is introduced.
- Save bridge weights and identity metadata as required checkpoint payloads and load them for dynamic HF inference. Bridge checkpoints fail closed under vLLM in this change; custom vLLM execution is deferred.
- Replace the unimplemented scalar owner-density proposal with this owner-addressable production route. The superseded change carried no implementation and has been removed from the active OpenSpec set.
- Repair the confirmed eight-rank (W8) smoke failure as a **runtime collective choreography** defect, not a model-quality or objective defect. Every physical accumulation slot on every rank SHALL execute exactly one DDP-wrapped anchor forward and one combined backward, and every auxiliary `L_use` branch forward SHALL run through the unwrapped module so a rank-varying local branch count can no longer change the per-rank collective sequence.
- Add an owner-use **performance lane** that truncates each branch to its segment-local causal prefix plus the candidate row and packs independent branches into one FA2 `cu_seq_lens`-isolated forward, preserving exact description/geometry target logits, MRoPE positions, and image-token mapping.

### W8 evidence and repair scope

The bounded eight-rank smoke at `outputs/smoke/coordexp_swift_owner_bridge/...accelerate8_ebs24_train256_val64_4epoch_warmup0p1-20260810T025027Z` failed with `completed_steps: 0` and `terminal_error: RuntimeError: [gloo/transport/tcp/pair.cc:547] Connection closed by peer`. The per-rank logs at `outputs/smoke/coordexp_swift_owner_bridge_readiness/d9d9768/w8-rank-logs/none_1ulisqjw/attempt_0/{0..7}/stderr.log` show ranks 0 and 4 stopped at `last enqueued NCCL work: 77` inside the `pre_backward` rank-report `all_gather`, while ranks 1, 2, 3, 5, 6, and 7 enqueued 79 and hung on `WorkNCCL(SeqNum=78, OpType=BROADCAST)` until the 600s NCCL watchdog fired. Those log files sit under the gitignored `outputs/` tree, so they are working-tree evidence; the conclusion is additionally supported by the recorded run failure and by two static code shapes — slots that skip the wrapped forward, and rank-varying auxiliary branch forwards on the wrapped module — either of which alone makes a divergent per-rank collective count structurally possible. It is a choreography failure and carries **no** evidence about atom, route, use, or autoregressive learning quality.

Raising the standalone rank-report control timeout to 1800s (commit `d9d976840`) cannot repair this: it exceeds the 600s NCCL watchdog, so it only delays the Gloo control gather past the point where the watchdog already aborts the run. That timeout is reverted to a bounded 120s and is explicitly not recorded as the fix.

The repair changes runtime execution shape only. It introduces no pair-sampling change, no loss weight or formula change, no data/order/annotation-trust change, no EBS change, no pack-cache identity change, and no inference change. Existing global loss denominators and world-size scaling are preserved exactly.

The production question is whether this owner-addressable late visual/KV interface improves natural dense enumeration while preserving valid row generation. The Stage 1 checkpoint and its natural greedy evaluation are retained regardless of outcome. Unit/contract tests, algorithm-reference checks, the distributed collective choreography acceptance battery, an exact-path single-rank smoke, and a bounded eight-rank launch smoke establish implementation readiness only; after all of them pass on the frozen tree and independent audits have no unresolved P0/P1, this change authorizes launching the eight-GPU Stage 1 production training without an additional model-quality gate.

### Scope and protected semantics

- Keep the accepted English full-wrapper schema, four `x1,y1,x2,y2` coordinate tokens, tokenizer vocabulary, no-resize image semantics, source example identity, packed-segment isolation, and ordinary full-sequence teacher-forced CE.
- Keep the vision tower and merger frozen; jointly train the new bridge, existing language DoRA, and selected special-token embeddings with explicit optimizer groups.
- Keep inference-time bridge computation permanent. A bridge-free route is outside this change unless a later proposal establishes a higher function-representation ceiling without it.
- Keep pair sampling, the four loss terms and their weights/ramps, eligible-count numerators/denominators, world-size scaling, data order, annotation trust, EBS 24, pack-cache identity, and every inference path unchanged while repairing collective choreography and adding the owner-use performance lane.

### Deferred work

- Stage 2 self-prefix/active-rollout learning, rollout buffers, atom-id replay, snapshot policy, actor/learner scheduling, and effective rollout batch sizing.
- vLLM custom-model integration or permanent-bridge materialization into a standard checkpoint.
- External controllers, hard without-replacement masks, persistent covered-set ledgers, reinforcement learning, decoding post-processing, additional special-token redesign, layer sweeps, and performance optimization unrelated to the Stage 1 production path.

## Capabilities

### New Capabilities

- `coordexp-swift-permanent-owner-bridge`: Defines the permanent visual-owner atom inventory, order-agnostic set router, bounded boundary admission, row-local owner lifecycle, late-layer causal write/read seam, Stage 1 objectives, and mandatory HF inference composition.

### Modified Capabilities

- `coordexp-swift-config-runtime`: Adds a strict Stage 1 bridge profile, alternating presentation schedule, fixed initialization lineage, loss ramps, architecture-profile selection, and optimizer hyperparameters.
- `coordexp-swift-data-template-encoding`: Makes deterministic epoch/presentation-aware random row permutations reproducible while preserving the authored `geo_sorted` presentations and typed owner-row spans.
- `coordexp-swift-packing-forward`: Carries owner metadata through packed segments and supports the block-20 row-write/final-boundary-read forward contract without breaking segment isolation or Qwen visual replacement.
- `coordexp-swift-pack-cache-semantic-identity`: Includes owner records, realized row order, bridge profile, and forward semantics in cache identity and rejects older incompatible payloads.
- `coordexp-swift-supervision-losses`: Adds protected `L_atom`, order-agnostic uncovered-set `L_route`, and same-image causal `L_use` objectives alongside full-wrapper `L_AR`.
- `coordexp-swift-adapters-embeddings-optim`: Adds explicit bridge parameter groups, learning rates, trainable-surface validation, and zero-initialized bounded bridge updates.
- `coordexp-swift-training-artifacts`: Saves permanent bridge payloads, bridge-aware checkpoint identity, schedules, diagnostics, and final checkpoint composition.
- `coordexp-swift-infer-config-runtime`: Adds strict bridge payload/profile identity and rejects unsupported bridge-plus-vLLM configurations before model loading.
- `coordexp-swift-infer-pipeline`: Makes dynamic HF composition load and execute the bridge for generation while preserving existing inference artifacts and failure accounting.
- `coordexp-swift-infer-backend-trace`: Carries per-sequence bridge lifecycle evidence through backend-neutral HF decode results.
- `coordexp-swift-infer-scoring-artifacts`: Binds bridge composition and lifecycle sidecars into single-rank and merged inference provenance.
- `coordexp-swift-vertical-smoke`: Adds permanent-bridge single-rank and eight-rank mechanical smokes plus an algorithm-reference acceptance battery before production launch.

## Impact

- Training code: configuration, template rendering, typed supervision records, pack-cache identity, Qwen forward hooks/wrapper, model modules, loss bundle, optimizer grouping, checkpoint saving, and run diagnostics.
- Inference code: strict model-composition config, dynamic HF model assembly, bridge-aware generation state, and manifest/provenance records. Existing vLLM execution remains unchanged and explicitly unsupported for bridge checkpoints in this change.
- Data and model lineage: the first production run starts from `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444` and preserves its four-coordinate full-wrapper contract.
- Runtime: after implementation correctness, algorithm-reference, smoke, and fixed-tree audit checks pass, the production leaf is launched through eight-rank Accelerate on available GPUs. Readiness does not require a favorable recall/duplication result from a tiny smoke.
- Distributed runtime: per-slot anchor-forward construction, unwrapped auxiliary branch execution, the bounded rank-report control timeout, and the owner-use branch packing lane. Rank-report semantics, finite gates, and gate-decision reduction keep their existing meaning.
- Dependencies: no dependency upgrade is required. The implementation remains on the accepted Transformers/PEFT/Accelerate training stack and does not promise exact optimizer-state resume.
