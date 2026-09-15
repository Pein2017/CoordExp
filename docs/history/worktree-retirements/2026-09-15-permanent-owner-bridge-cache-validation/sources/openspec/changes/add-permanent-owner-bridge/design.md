## Context

See [proposal.md](proposal.md) for motivation and the capability specs for normative behavior. The accepted training spine is `src/train.py -> src/training/pipeline.py -> src/training/supervised_trainer.py`, with template, packing, Qwen forward, loss, optimizer, and artifact owners already separated. Dynamic HF inference is owned under `src/inference/`; vLLM consumes a standard materialized model and cannot execute a new stateful bridge without separate custom-model work.

The source checkpoint is the accepted four-coordinate `geo_sorted_xy` step-2444 composition. It already has a strong sequential prior. Stage 1 must therefore add an owner-addressable route without destroying the native path, while using full random row permutations often enough that the new route cannot be ignored in favor of one raster successor rule.

Current evidence motivates, but does not prove, the design:

- forced or sampled decoding can recover objects absent from greedy output;
- late visual-region and history interventions can affect owner selection;
- a row-end commit representation can retrieve its just-emitted owner;
- generic additive bridge inputs can act as activation energy rather than owner identity;
- static carriers sometimes lack recoverable support, so this bridge cannot repair all perception or merger loss.

The design intentionally turns the next production checkpoint into the primary observation. Mechanical tests and a short exact-path smoke establish execution validity only.

The first bounded eight-rank (W8) smoke failed before any optimizer step and reclassified part of that execution validity as unfinished. Its receipts are:

| Evidence | Value |
| --- | --- |
| run root | `outputs/smoke/coordexp_swift_owner_bridge/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1-20260810T025027Z` |
| `status` / `completed_steps` | `failed` / `0` |
| `terminal_error` | `RuntimeError: [gloo/transport/tcp/pair.cc:547] Connection closed by peer [192.168.7.7]:57000` |
| per-rank logs | `outputs/smoke/coordexp_swift_owner_bridge_readiness/d9d9768/w8-rank-logs/none_1ulisqjw/attempt_0/{0..7}/stderr.log` (present on disk; `outputs/` is gitignored, so these are working-tree artifacts, not repo-durable) |
| ranks 0, 4 | `last enqueued NCCL work: 77, last completed NCCL work: 77`, blocked in `pre_backward -> _gather_rank_reports -> _all_gather_cpu_bytes` |
| ranks 1, 2, 3, 5, 6, 7 | `last enqueued NCCL work: 79, last completed NCCL work: 77`, watchdog on `WorkNCCL(SeqNum=78, OpType=BROADCAST, NumelIn=80, Timeout(ms)=600000)` |
| launcher log | `outputs/smoke/coordexp_swift_owner_bridge_readiness/d9d9768/w8-train.stdout.log` |

Two ranks enqueued two fewer device collectives than the other six and then entered a CPU control gather that the other six never reached. Because the per-rank logs live under the gitignored `outputs/` tree, the diagnosis does not rest on them alone: the counters are corroborated by the recorded `run.json` failure record and by two static code shapes that make a rank-varying collective count structurally possible.

That is a divergent per-rank collective sequence, and both contributing shapes are visible in the implementation: a shadow/no-data slot builds `build_connected_shadow_loss` from parameters without any wrapped forward, and each owner-use branch runs a separate forward through the DDP-wrapped model whose eligible-pair count varies by rank. This is runtime choreography evidence only. It says nothing about atom matching, routing, owner use, or autoregressive quality, and it must not be summarized as a negative result for the objective.

Raising `_RANK_REPORT_CONTROL_TIMEOUT_SECONDS` to 1800s cannot be the repair, because 1800s exceeds the 600s NCCL watchdog that already aborted this run; the control gather would still be waiting when the watchdog tears the process group down. The bounded 120s value is restored and the control timeout is required to stay below the NCCL watchdog.

## Goals / Non-Goals

**Goals:**

- Introduce one deep module that owns static owner atoms, boundary routing, row-local commitment, late row writes, bridge losses, and bridge serialization.
- Make the current row's physical owner causally available from opener through coordinates and write that identity into upper-layer KV state before the next boundary query.
- Teach covered-versus-uncovered set reallocation during entirely teacher-forced Stage 1 trajectories without prescribing a next-owner order.
- Preserve full-wrapper CE, native token choice, native visual attention, native residuals, packed-segment isolation, and HF batched greedy inference.
- Produce a self-contained permanent-bridge checkpoint and production config from the S checkpoint, ready for the conditional direct eight-GPU launch after the declared readiness checks pass.

**Non-Goals:**

- Stage 2 self-prefix learning, rollout collection, active learner infrastructure, replay, off-policy correction, or actor throughput.
- A hard without-replacement decoder, persistent covered-set memory, external controller, constrained grammar, or post-hoc union/deduplication.
- New special tokens, a closed-set detector, merger unfreezing, vision-tower changes, or an alternative target schema.
- Layer search as a run-time/config dimension. A future architecture profile can change the seam through a new reviewed contract.
- vLLM support, bridge removal/distillation, exact optimizer-state resume, or model-quality launch gates.

## Decisions

### 1. One permanent bridge module with typed callers

Add a model-family-neutral owner module under `src/qwen/` with a versioned Qwen3-VL-2B profile and typed training/inference contexts. It owns:

- `OwnerAtomInventory`: carrier/slot identity, objectness, box, key, and value;
- `OwnerAssignment`: detached global matching from source-owner identity to atom;
- `BoundaryRoute`: atom/null scores, trustworthy-domain probabilities, and aggregate availability;
- `ActiveOwner`: one row-local atom id/value plus its lifecycle state;
- `OwnerBridgeOutputs`: loss inputs and bounded diagnostics;
- bridge payload save/load identity.

Training owns source-owner targets, presentation identity, matching eligibility, and loss reduction. HF inference owns per-sequence decode state and token transitions. Neither caller reimplements atomizer or routing formulas. Inference imports the shared Qwen bridge module and payload types, never training owners.

Alternative: scatter heads, hooks, losses, and state across existing training and inference modules. Rejected because the same owner identity would acquire several subtly different definitions and checkpoint composition could omit inference-critical pieces.

### 2. Preserve Transformers visual replacement and intercept only two named seams

The permanent bridge wraps the loaded Qwen model and installs verified, scoped interceptors at:

1. the output of zero-based language block 20; and
2. the output of the final RMS normalization immediately before the LM head.

The wrapper continues to call the canonical model with `input_ids`, image tensors, grid metadata, MRoPE inputs, and cache arguments. Transformers remains responsible for embedding lookup, image-placeholder replacement, attention/cache mechanics, and the native LM head. The bridge context supplies segment-local image positions, row positions, and boundary positions. Interceptors must prove the expected model type, 28 blocks, hidden width, module identity, call count, tensor rank, and position alignment on every new runtime construction. Activation-checkpoint recomputation is part of the same logical forward: it must restore the exact immutable typed context for the original segment and `L_use` branch, apply the identical bridge transform, and avoid emitting diagnostics or lifecycle side effects twice.

This avoids copying the Qwen forward loop while still letting an after-block-20 update flow through blocks 21-27. Context is explicit and scoped to one forward call; no process-global mutable hook state is allowed.

Alternative: request hidden states from an unmodified full forward and add an auxiliary head afterward. Rejected because it cannot write the active owner into row KV state or change native generation.

Alternative: reimplement the full Qwen decoder loop. Rejected for Stage 1 because it duplicates fast-moving upstream cache, MRoPE, and multimodal behavior and weakens parity evidence.

### 3. Fix the causal seam as an architecture profile

Register `qwen3_vl_2b_owner_bridge_v1` with:

- 28 zero-based blocks `0..27`;
- atom read and row write after block 20;
- upper integration blocks `21..27`;
- boundary route read after final RMS normalization;
- `K=4`, key width 128, value width 512, row-adapter inner width 256;
- boundary and row RMS caps `0.03` and `0.10`.

The layer choices are not separate config fields. Profile validation fails if the loaded architecture differs.

Block 20 is preferred over block 23 because it leaves seven rather than four upper blocks to turn an owner payload into language, geometry, and KV state. More importantly, the next boundary router must read above the write seam. A block-20 row write followed by a block-20 boundary query cannot observe the owner-conditioned keys/values formed by blocks 21-27.

Alternative: expose the layer as a general hyperparameter. Rejected because a change in write depth changes cache visibility and remaining integration capacity; it is an architecture contract, not an ordinary tuning scalar.

### 4. Construct four addressable atoms from each late visual carrier

For post-block-20 image-carrier state `h_i`, learned slot embedding `s_k`, and normalized carrier coordinates `p_i`, compute one shared trunk:

```text
r_ik = MLP([RMSNorm(h_i), s_k, p_i])
o_ik = W_o r_ik
b_ik = decode_box(W_b r_ik, p_i)
k_ik = L2Norm(W_k r_ik)       # 128
v_ik = RMSNorm(W_v r_ik)      # 512
```

`decode_box` produces one complete normalized `xyxy` box from carrier-relative center offsets and positive width/height parameters. The four learned slots receive deterministic quadrant-biased initialization only to break symmetry; the loss never asserts a permanent quadrant meaning. All heads derive from the same `r_ik`, avoiding key/value/box identities that can drift independently.

No class head is added. The box/objectness heads supply address and assignment; semantic identity must be present in `v_ik` and consumed by the native language model. This deliberately tests whether late image tokens can be organized into useful owner values rather than training a parallel closed-set detector.

Alternative: one atom per carrier. Rejected because one merged carrier key cannot separately address multiple superposed physical owners.

Alternative: independent class, geometry, key, and value towers. Rejected because a good probe head could describe one owner while the value used by the LM refers to another.

### 5. Use one global injective assignment per image

Let `j` index labeled source owners and `m=(i,k)` index every atom. Build a deterministic candidate graph from carriers inside the target box, one carrier-neighborhood dilation, and the nearest carrier fallback. Run one image-global Hungarian assignment with cost:

```text
C(j,m) = 5 * L1(b_m, b_j)
       + 2 * (1 - GIoU(b_m, b_j))
       - log(sigmoid(o_m) + eps)
       + 0.25 * anchor_distance(j,m)
```

The candidate graph is a computational locality aid, not an independent local matcher. Stable source-owner id, carrier index, and slot index break exact ties. Assignment decisions are detached; gradients flow through losses on the chosen predictions.

For trusted exhaustive/refined examples, matched atoms are positives and trustworthy unmatched atoms are objectness negatives. For ordinary COCO, matched atoms are positives and every unmatched atom is unknown-neutral: it contributes no objectness loss and is excluded from route normalization. No geometry-only duplicate-negative exception is used in the first production leaf. Final null supervision follows the same trusted/ordinary split. Matching first tries the locality candidate graph, then expands to the full inventory when necessary; every labeled owner must match exactly once or the example fails before any row is trained.

Alternative: independently match within each carrier. Rejected because one physical owner can become several inventory items and later appear as apparent duplicates despite a "covered" history.

### 6. Route a set, then commit one row

For a post-final-norm boundary state `g_t`, compute:

```text
q_t = L2Norm(W_q g_t)
s_tm = q_t dot k_m / 0.2
       + 0.5 * log(eps + sigmoid(o_m))
s_t_null = MLP_null(g_t)
p_t = softmax([s_t1 ... s_tPK, s_t_null])
```

The objectness term is non-positive, so low-quality atoms are suppressed but high objectness cannot add an unbounded salience bonus. Keys remain the owner-specific address.

At a teacher-forced boundary, a trustworthy-domain mask removes annotation-unknown atoms before renormalizing `p_t` for `L_route`. It does not alter the forward router distribution used for admission or inference.

Boundary admission receives only the scalar atom-versus-null evidence:

```text
r_avail = stop_gradient(logsumexp(s_atoms) - s_null)
delta_adm = RMSClip(MLP_adm(r_avail), 0.03 * RMS(g_t))
g'_t = g_t + delta_adm
next_token_logits = LMHead(g'_t)
```

It never sees `v_m` or a soft owner-value mixture. The stop-gradient separates responsibilities: native token CE trains the admission map but cannot turn a partial-annotation terminal token into implicit key/objectness/null supervision; router parameters are trained by `L_route`. This makes admission a soft harness for native continue/stop behavior without blurring several owners into one transcription payload.

After the native decoder emits `<|object_ref_start|>`:

- Stage 1 TF selects the atom globally matched to that gold row;
- HF inference selects `argmax_m s_tm` among non-null atoms from the preceding boundary;
- the selected value and atom id are latched until the row's `<|box_end|>` has completed its full forward;
- the old latch clears only after that box-end final representation and next route are available.

There is no cross-row selected-owner mask. The router can score the same atom again; suppression/reallocation must be learned from owner-conditioned history through `L_route`.

Alternative: fully soft owner mixtures through the row. Rejected because multiple owners can superpose into an unidentifiable payload and the model need not form a discrete row-owner relation.

Alternative: hard mask previously selected atoms. Rejected because it would make the decoder correct by an external set operation and prevent the experiment from testing learned KV/query reallocation.

### 7. Write owner identity into every row token

For each post-block-20 row hidden state `h` and the fixed row value `z`:

```text
u = SiLU(W_h RMSNorm(h)) * W_z RMSNorm(z)
raw_delta = W_2 u
delta = RMSClip(raw_delta, cap(step) * RMS(h))
h' = h + delta
```

`W_h` and `W_z` project to width 256, `*` is elementwise multiplication, and `W_2` is zero initialized. The multiplicative interaction prevents a constant `z` norm from acting only as a generic additive pulse. `cap(step)` ramps from zero to `0.10` in the first presentation.

Apply the same latched `z` to every token from opener through box end inclusive. The owner can therefore influence description, box opener, all coordinates, row close, and their upper-layer KV entries. The row close is injected before its final boundary query, so the next query can attend to an explicit trace of which physical owner produced the completed row.

`RMSClip(delta, alpha * RMS(h))` means:

```text
RMS(x) = sqrt(mean_i(x_i^2))
scale = min(1, alpha * RMS(h) / (RMS(delta) + eps))
RMSClip(delta, alpha * RMS(h)) = scale * delta
```

It limits vector energy, not semantic effect. A small structured update can still change attention, later nonlinearities, and token rankings substantially.

Alternative: inject once at opener. Rejected because a long description-to-coordinate span can lose binding and because one amplitude pulse can teach generic continuation more easily than persistent owner-conditioned computation.

### 8. Teach set awareness in TF without conflicting row targets

Each row keeps its stable source-owner id under all permutations. Every example also carries an explicit annotation trust class. The first production run uses only `ordinary_partial`, matching the S checkpoint's COCO source; trusted-exhaustive behavior is implemented and tested with fixtures but inactive in that production run. At boundary `t`, the gold prefix defines covered set `C_t` and known-uncovered set `U_t`. For trustworthy normalization domain `Omega_t`:

```text
P_U(t) = sum_{m matched to U_t} p_t(m | Omega_t)
L_route(t) = -log(P_U(t) + eps)                 if U_t is non-empty
L_route(t) = -log(p_t(null | Omega_t) + eps)    if trusted exhaustive and exhausted
```

Ordinary COCO exhaustion is masked for direct router-null supervision. The inherited full-wrapper `L_AR` still supervises the terminal token to preserve the S checkpoint's native grammar; this is an indirect admission/stop signal and is not treated as proof that unmatched atoms are background. Because every known covered atom remains in `Omega_t`, the normalized objective teaches mass reallocation after each completed row. Because it sums over `U_t`, it does not impose the teacher row as router target.

This separation resolves a critical conflict: if a predicted router owner `b` were fed into a teacher row for owner `a`, `L_AR` would reward ignoring `z_b`. Stage 1 instead feeds `z_a` to row `a`, while `L_route` trains the router on the whole uncovered set. Random full trajectories make the row order unpredictable from a single raster rule and make consuming the matched owner value useful for CE.

The schedule is one continuous optimizer run over four complete cache-backed presentations:

1. `geo_sorted` authored order;
2. `random-1` from hash(run seed, example id, presentation id);
3. the same validated `geo_sorted` order;
4. `random-2` from a distinct presentation id, deterministically rotated if a hash collision yields `random-1` and another permutation exists.

The repeated sorted presentations retain the S checkpoint's stable grammar/order basin. The two random presentations contribute half the full trajectories, rather than a 25/75 split that could leave a simple successor shortcut dominant.

Alternative: learn set awareness only from self-prefixes. Deferred, not rejected. Self-prefixes are necessary to expose model-generated state distribution in Stage 2, but they are not necessary to teach the basic covered/uncovered relation. Starting with TF separates representation/consumption from rollout-distribution errors and produces a deployable checkpoint sooner.

### 9. Optimize four losses with one identity-bearing counterfactual

The total loss is:

```text
L = L_AR + L_atom + 0.5 * L_route + 0.1 * L_use
```

- `L_AR`: existing full-wrapper, token-normalized teacher-forced CE.
- `L_atom`: matched objectness plus `5 * L1 + 2 * (1-GIoU)` box terms; trusted negatives only.
- `L_route`: normalized uncovered-set or trusted-null objective above.
- `L_use`: same-image two-owner causal swap.

For exactly one deterministic eligible boundary/pair `(a,b)` per source example and microstep, chosen where at least two known-uncovered owners remain, fork four one-row continuations from the exact same image and gold boundary prefix: `(y_a,z_a)`, `(y_a,z_b)`, `(y_b,z_b)`, and `(y_b,z_a)`. Each branch teacher-forces only its candidate row after the shared prefix. This prevents an earlier wrong value from changing the hidden/KV prefix used to score the other row while preserving row tokens, geometry targets, and generic activation counts. Let `D(j,z)` and `G(j,z)` be average log likelihood over description and geometry tokens for row `j` under value `z`:

```text
Delta_D = [D(a,z_a) + D(b,z_b)] - [D(a,z_b) + D(b,z_a)]
Delta_G = [G(a,z_a) + G(b,z_b)] - [G(a,z_b) + G(b,z_a)]
L_use = softplus(-Delta_D) + softplus(-Delta_G)
```

Wrapper/control tokens are excluded. Half the pair choices use same-class or spatially close competitors when source labels permit, and half are deterministic random pairs. Correct and counterfactual variants are evaluated by the same bridge/Qwen path; the implementation may batch or checkpoint these variants, but all four sides must carry gradients through the row adapter into the selected owner values and shared atomizer as well as into the native trainable path. A detached/no-grad owner-value prefix is non-conforming.

Wrong-image, globally permuted values, and bridge-zero interventions stay evaluation-only. If used as training negatives, the network could distinguish dataset/source artifacts instead of owner identity.

Alternative: hidden-state alignment loss. Rejected because it can make a probe readable without requiring the LM to use owner identity for description or coordinates.

Alternative: wrong-image margin alone. Rejected because previous additive signals can be separated by norm, energy, or continuation effects; the 2x2 swap balances those factors.

### 10. Keep the existing optimizer lineage but start fresh optimizer state

Load the S checkpoint's base-compatible language DoRA and selected embedding delta with explicit `adapter.seed_mode: warm_start_expand_dora`, binding `adapter.source_adapter_path` to the step-2444 `adapter/` payload and `adapter.repaired_embedding_payload_path` to its `special_token_embeddings/` payload. Validate both payload fingerprints before initializing new bridge weights. Freeze vision and merger. Construct explicit non-overlapping groups:

| Group | Learning rate | Contents |
| --- | ---: | --- |
| language DoRA | `5e-5` | accepted language-tower DoRA targets |
| selected embeddings | `2.5e-5` | existing full-wrapper special-token rows only |
| permanent bridge | `2.5e-4` | atomizer, keys/query, null, admission, row adapter |

Use the existing S-lineage COCO-80 train/eval files, no-resize `max_raw_pixels=1048576`, `max_merged_visual_tokens=4096`, `global_max_length=12000`, fresh AdamW, global EBS 24, cosine schedule, ten-percent warmup, and four complete presentations. The source optimizer state is intentionally not resumed because the trainable surface and loss geometry changed.

`L_route` ramps to 0.5 over the first 5% of the resolved planned-step budget, indexed by completed safe optimizer updates. `L_use` ramps to 0.1 over the first 20% on the same safe-update counter. The row cap ramps through the first presentation. The boundary cap stays at 3%. Skipped unsafe updates do not advance any ramp.

Training-time eval reuses the exact teacher-forced model context, global assignment, gold row latch, and four-loss computation under `eval()`/no-grad. It runs at the predetermined midpoint and end planned step of every presentation, yielding eight signals at approximately 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%, and 100% without per-step disruption. All eight events reuse one fingerprinted `geo_sorted` eval rendering so their curves are directly comparable. If an optimizer update is unsafe at a landmark, the eval still fires under that original planned-step event id and records the skipped update; the safe-update counter controls ramps but never reschedules eval. Natural greedy HF evaluation is intentionally post-checkpoint and is not part of the training cadence.

Alternative: freeze the LM and train bridge heads first. Rejected because atom values must co-adapt with the LM path that consumes them, and a detector-first phase can converge to box-good/value-decorative features.

### 11. Materialize presentation-specific caches behind one continuous schedule

Cache preparation builds or validates three semantic payload identities: `geo_sorted`, `random-1`, and `random-2`; the `geo_sorted` payload is consumed twice. Each payload holds complete rendered/encoded/packed rows plus annotation trust, compact owner, and boundary records. A presentation iterator concatenates the four payload uses without resetting optimizer or scheduler state and verifies that every source example occurs exactly once per presentation. The production data declaration is constant `ordinary_partial`; changing that declaration changes cache identity. No human-refined panel is mixed into the first production run without a later explicit data-scope decision.

The planned-step schedule is computed from all four presentation pack counts under global EBS 24. Presentation boundaries and their midpoint/end eval events are resolved on that planned-step clock; safe-update progress is recorded separately for ramps. Data-loader worker count can change materialization mechanics but cannot change realized permutations or pack contents.

Alternative: dynamically permute rows after tokenization. Rejected because typed spans, causal positions, pack lengths, and cache identity would become inconsistent.

Alternative: duplicate four variants into one shuffled dataset. Rejected because it loses the deliberate sorted/random phase order and weakens attribution to the curriculum.

### 12. Save one exact permanent composition

Add `owner_bridge.safetensors` and `owner_bridge_manifest.json` to every bridge-enabled checkpoint. The manifest binds:

- schema and architecture-profile versions;
- base identity available from the source checkpoint;
- source adapter fingerprint and selected-embedding fingerprint;
- tokenizer fingerprint;
- bridge tensor names, shapes, dtypes, and payload hash;
- block count/seam, K/key/value/inner widths, caps;
- training schedule, seeds, and companion payload paths.

Checkpoint publication validates adapter, selected embeddings, and bridge as one inference composition. Existing bridge-disabled checkpoint behavior remains unchanged. The run also saves bounded atom, route, swap, RMS, grammar, and gradient summaries at presentation boundaries and final.

No exact optimizer-state resume is added. Presentation-boundary checkpoints are research/inference states, not a promise that a failed training process can resume bit-exactly.

### 13. Execute the bridge through dynamic HF only

Extend strict inference composition with an explicit bridge path. `src/inference/hf_backend.py` loads the base, DoRA, selected embeddings, then permanent bridge and verifies all companion identities. `src/inference/vllm_backend.py` and execution-model materialization remain unchanged; config rejects bridge-plus-vLLM before either is invoked.

Each greedy batch element owns a `BridgeDecodeState`:

```text
atoms: optional static inventory
pending_route: optional boundary distribution
active_owner: optional hard atom id/value
phase: boundary | row | stopped
```

On prefill, the block-20 interceptor creates atoms and the final-norm interceptor creates the first pending route and bounded admission. After native argmax sampling:

- opener: latch best non-null from `pending_route`, set `phase=row`;
- ordinary row token: retain latch;
- box end: run with latch, compute new route at final norm, then clear old latch and set `phase=boundary`;
- terminal token: clear state and set `phase=stopped`.

The state is carried through the HF generation kwargs/cache lifecycle and indexed with the batch. Stage 1 supports greedy generation only; unsupported beam branching or cache reordering fails closed until it has an exact lifecycle implementation. No decode-time atom mask is applied.

### 14. Make diagnostics causal and bounded

Required training receipts are summaries, not launch gates:

- atom matched recall/box error by size and density, slot utilization, ordinary-unmatched masked count, and assignment switch rate;
- boundary `P_U`, `P_C`, `p_null`, top-k atom ids, and null/opener disagreements;
- aggregate availability versus merged-carrier/atom count, and selected-atom matched/unknown status on labeled evaluation rows;
- correct/swap description and geometry margins;
- boundary and row update RMS ratios and clipped fraction;
- per-loss numerator/denominator and gradient norm per trainable group;
- full-wrapper validity and loss by presentation.

Required first-checkpoint inference observations are natural greedy unique recall, strict duplicates, unmatched predictions, row validity, stopping, and a bounded lifecycle trace. Bridge-zero, wrong-image, and value-permutation interventions are retained for a deterministic panel after the checkpoint exists. They characterize a checkpoint; they do not decide whether Stage 1 was allowed to train or whether its checkpoint is saved.

### 15. Separate algorithm accuracy from model-quality outcome before launch

Implementation readiness uses executable invariants whose expected answers are known independently of a trained checkpoint:

1. reference-sized assignment cases must exactly match a brute-force optimum and remain injective under carrier/slot ties;
2. route probabilities and `L_route` must match an FP64 reference, remain invariant to atom permutation, and treat all uncovered-owner permutations equally;
3. the four-branch `L_use` implementation must match its scalar reference, cancel a common logit pulse, send gradients in the correct diagonal-versus-swap direction, and reach the selected values plus shared atomizer;
4. zero bridge contribution must recover bridge-disabled logits within the declared dtype tolerance;
5. a controlled nonzero row value must affect the post-final boundary through blocks 21-27, while a same-seam read cannot observe that write;
6. packed versus isolated segments, full teacher forcing versus forced-token incremental HF, and in-memory versus saved/reloaded bridge execution must agree at their shared causal positions;
7. per-sequence greedy decode state must survive unequal row phases without leakage;
8. activation-checkpoint on/off executions must agree on logits, losses, and bridge/DoRA/selected-embedding gradients for an ordinary branch and all four `L_use` branches while counting diagnostics once;
9. one-process and eight-rank smokes must complete safe optimizer steps, finite reductions/backward, atomic bridge checkpoint publication, reload, and bounded greedy decode through the exact production modules;
10. a two-rank actual-DDP test with deliberately uneven eligible owner-use branch counts and a mixed real/shadow slot assignment must record exactly one anchor forward per physical slot per rank, one combined backward per slot, one optimizer update for the window, and DoRA/selected-embedding/bridge gradients matching a single-process reference within declared tolerance;
11. the exact one-step W8 choreography digest — per-rank ordered collective kinds and counts for one planned step — must be identical across ranks before the full W8 smoke is rerun, and the full W8 smoke must then reach its bounded step budget;
12. packed owner-use branches must agree with per-branch isolated execution on branch logits, `L_use`, and gradients, with an FA2 receipt recording `cu_seq_lens` boundaries and proving no cross-branch or cross-prefix attention.

Checks 10-12 are choreography and equivalence checks. A green rerun of the W8 smoke is required, but the smoke's loss values remain mechanical evidence and never become model-quality evidence.

These checks establish algorithm and plumbing accuracy. They do not require atom AP, recall, duplication, or natural stopping to improve after a handful of smoke steps. Once the full battery — checks 1-9 plus the choreography acceptance checks 10-12 — strict OpenSpec validation, and fixed-tree audits all pass on the same frozen tree with no unresolved P0/P1, the implementation may launch the eight-GPU production run. Results carried over from a tree that predates a runtime repair do not count. Before launch it must check live GPU ownership/headroom, use an explicit unique artifact root and at-most-once command guard, and never evict unrelated work.

### 16. Give every accumulation slot exactly one DDP anchor forward

Data-parallel correctness requires each rank to enqueue the same ordered collective sequence, so the unit of choreography is the physical accumulation slot, not the eligible example. For each planned step, every rank iterates the same number of slots and each slot executes exactly one DDP-wrapped anchor forward followed by one combined backward:

- a **real** slot's anchor is the ordinary teacher-forced packed forward it already runs;
- a **shadow / no-data** slot's anchor is a deterministic differentiable dummy forward through the same wrapped model, whose output enters the existing connected-zero surface so every trainable parameter receives a defined zero gradient contribution.

The shadow anchor replaces the current parameter-only `build_connected_shadow_loss` shape, which produced no wrapped forward and therefore no DDP broadcast/allreduce for that slot. Gradient synchronization stays on the final slot of the window, so the number of reducer flushes per planned step is a function of the slot count alone.

All auxiliary `L_use` branch forwards move to the **unwrapped** module. They read and write the same `Parameter` objects, so gradients still reach DoRA, selected embeddings, and every bridge parameter through the one combined backward, but a rank-varying local branch count no longer emits rank-varying collectives. Existing eligible-count numerators, global denominators, and world-size scaling are untouched; a rank with zero eligible pairs keeps contributing a zero numerator and zero denominator exactly as before.

Alternative: keep branch forwards on the wrapped model and pad every rank to a common branch count. Rejected because it changes pair sampling and the realized eligible set to satisfy a runtime constraint, and it wastes compute on ranks that legitimately have fewer eligible pairs.

Alternative: raise the rank-report control timeout. Rejected: the observed W8 hang was bounded by the 600s NCCL watchdog, not by the control gather, so a longer control timeout only hides the divergence later. The bound stays a module-private static constant at 120s, enforced by a static test asserting it is below the effective NCCL watchdog; it is deliberately not promoted to a configurable field, since no run should be able to tune its way past a choreography defect.

### 17. Truncate and pack owner-use branches without changing their meaning

Each owner-use branch currently re-forwards the full pack to score one candidate row, so a 2x2 comparison costs four full-pack forwards on top of the ordinary forward. The performance lane keeps the objective identical and changes only the executed token window:

1. **Truncate.** Each branch executes its segment-local causal prefix up to the selected boundary plus its candidate row. Tokens outside that segment, and tokens after the candidate row, cannot influence the scored positions under a causal mask, so removing them is an exact rewrite rather than an approximation. MRoPE position ids, the image-token mapping, and the visual carrier states for that segment are preserved verbatim; description and geometry target logits must be bit-comparable to the untruncated branch within declared dtype tolerance.
2. **Pack.** Independent branches are concatenated into one forward with FA2 `cu_seq_lens` boundaries at each branch start, so attention never crosses a branch. Branches share no prefix state through attention; they share only the model parameters and the bridge inventory for that image.

The lane is required to publish an FA2 receipt with the realized `cu_seq_lens_q`/`cu_seq_lens_k` boundaries and a packed-versus-isolated equivalence result. Any disagreement in branch logits, `L_use`, or gradients fails the lane rather than being tolerated as a performance trade.

The lane does not change which pairs are selected, how `Delta_D`/`Delta_G` are formed, the softplus objective, or the loss weights. It reduces branch cost only.

Alternative: score all four branches in one shared-prefix forward with a common prefix segment. Rejected in this change because a shared prefix segment would let the branches attend to one another's rows unless a custom mask is introduced, and a custom mask is a larger attention-contract change than `cu_seq_lens` isolation.

### 18. Recover one consumed pre-run activation only through an immutable successor

The first guarded production activation consumed the singleton launch claim but failed before a run root, model construction, or optimizer mutation: rank zero remained in the old post-process-group cache payload scan while peer ranks timed out in the pre-model status broadcast. A consumed claim remains immutable even when the failure is classified as pre-run infrastructure. Repair evidence, passing smokes, free GPUs, and implementation permission do not regenerate launch authority.

The user has authorized exactly one recovery activation. Recovery therefore publishes one append-only, canonical parent-linked successor at the fixed filename `stage1-production-recovery-attempt-1-claim.json`, with `attempt_ordinal=1`, `max_recovery_attempts=1`, and a distinct `stage1-production-recovery-claim-v1` policy/schema. The original v1 singleton policy, exact field set, and validator remain unchanged so the two claims remain independently readable. The recovery intent key is derived from the recovery policy, current commit, production-config fingerprint, parent claim SHA-256 and receipt fingerprint, parent intent key, and ordinal `1`; it is not the ordinary commit/config key and does not depend on the fresh nonce. A second independent guard rejects any recovery-claim residue matching the fixed recovery namespace and rejects every ordinal or maximum other than `1`.

The successor filename is reserved by `O_EXCL` and its parent directory is fsynced **before** the nonce or successor payload is derived. That reservation is the authorization-consuming transition. It is never unlinked on any failure path: a zero-byte, partial, or malformed reserved successor is terminal evidence and makes every later recovery invocation reject before parsing it. Generic atomic-write helpers that publish only after payload construction, or that unlink on failure, are not valid reservation primitives.

The accepted parent anchors are fixed, not caller-provided knobs. They include the original claim SHA-256 `0b597d82b97855b44742b959d5e864149d9f68070701cc8ce2665e15ad82db73` and receipt fingerprint `ca087ebb4d98472188d84b4bab0cb5dcb86e14b59290f470d9246b407a20eeae`; original intent key `02bdbc993c57f4e3b7f38506d40d80af9acbc2ab327a1330ee1e86789078d3ad`; original intent SHA-256 `e270ae82ad0d4f22ad31eaea3a388944b3aecbe7924ce374b7301b5e270dcf1d`; activation SHA-256 `14864a1fff219f4afdc86e3a443666abda13f9758fa4d598d978acf42349c4e6`; preflight SHA-256 `5efe94c4ca1de2ba9b39015e12a439f9585de6037cd9694263be49f058617a3d`; worker-admission rank-0 through rank-7 SHA-256 values `6caa7c1892cea981034638e10cb645b36dc61f5c89f95236e3398f9665926b28`, `bbb3093e66e5ce02a429b5f79d2f3d6bc6a80e3e5efc437434e0cedfa77ced5b`, `5a62727265f4de4054d1f52e5793e93aad11f18cd010289031c7e7684dbe24dd`, `44741143d02e23a64566661c7f884bdb6a1e06d6ab1eb346424ce11d62a1d6c9`, `62c985251c1e13ea08dc4cbfd2556178c0ca09ffbccf35f0de6b0c84cd16ac42`, `8307b61d5e2c5bdaf9960a15e3b614ed1f0e1b430fdc836451d68952b6b08946`, `725a492e1ffb8d28f63688445dea4c2f4d0eff614607c95f6aa11545c16db5d6`, and `4a1c5c251c12c5f66bfbcf060ad0c0f073a7e8bf9651698baa9834c6858766e9`; failed run-binding SHA-256 `cda4cd4b970d12697a98965cd0ea66f27291a93b548344334a70f6aa218dbc47` and receipt fingerprint `2505ff27e404bffbd54b8a33262a9b86e8f8bb6a146310ec45540d8db52ed92f`; empty stdout SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; and stderr SHA-256 `8020c3ea8fd5be58c19af29a557386b4fe0f34d3bc5b33bb05a55b9b47827aa5`. The empty stdout anchor proves only that the file remains empty; it is retained as one bounded fact, not treated as independent execution evidence.

Recovery also binds the parent intent's exact `expected_run` artifact root, output directory, and run name and requires equality with the recovery preflight. Eligibility is evidence-positive: the exact terminal `run_binding_failed` receipt must be present with error `training.owner_bridge_launch_guard_process_exited` and a nonzero return code; the exact PID/start-time/boot identity from the anchored activation must no longer be live; the parent run-binding file must be absent; and no `run.json` under the parent-bound artifact root may carry the parent nonce. A guarded static-order regression must continue to prove that production publishes `run.json` before entering the sole model/optimizer construction path; only under that fixed ordering can the parent run absence support the bounded no-model/no-optimizer conclusion. The successor binds all fixed evidence files by exact canonical path plus SHA-256, the current frozen repository/config identity, a fresh nonce unequal to the parent nonce, and the distinct attempt identity. The original claim, intent, activation, preflight, admissions, logs, and binding failure are never deleted, renamed, overwritten, or reused.

Production workers select from a closed set containing exactly the original claim and the fixed attempt-1 successor. Exactly one readable claim must match the launch nonce; zero matches, multiple matches, or an unreadable candidate fail closed without falling back. The selected claim path and its SHA-256 are threaded into admission and the pre-model quorum receipt before distributed initialization. All successor intent, activation, admission, pre-model, run-binding, and failure evidence is attempt-scoped.

The lead-only executor may reserve the successor only after the repaired implementation is committed, fixed-target lifecycle review has passed, and source/cache/tree/GPU preflight is current. Immediately before reservation it rechecks all parent hashes, the complete recovery activation surface (including intent and pre-model paths), repository/config/cache identity, and GPU headroom. The original attempt must have no successful run binding, run root, model/optimizer heartbeat, or other evidence that execution crossed the pre-run boundary. Any parent-evidence hash drift, existing or malformed successor/residue, reused nonce, wrong ordinal, recovery-preflight run-root drift, or evidence of a successful original activation fails before reservation or process creation. If reservation, payload completion, activation, binding, or the required first finite/applied heartbeat is uncertain or fails, the retained evidence is terminal and execution stops; no third activation is authorized.

Alternative: delete or rotate the singleton claim and launch normally. Rejected because it erases the at-most-once authority boundary and makes retries indistinguishable from a first activation.

Alternative: reuse the original claim/nonce or accept a caller-selected recovery ledger. Rejected because stale workers or a second receipt root could replay authority without a unique, canonical successor.

## Risks / Trade-offs

- **[Carrier information is already lost]** K slots improve address and assignment capacity but cannot reconstruct evidence discarded by the vision tower or merger. → Record oracle assignment/atom quality and retain failure checkpoints; merger changes require a later proposal.
- **[TF set awareness does not transfer to self-prefixes]** Gold histories may encode coverage cleanly while model-generated histories do not. → Keep Stage 1 claim to TF-trained representation plus natural-rollout observation; Stage 2 remains a separate self-prefix intervention.
- **[Native KV fails to represent coverage]** Rows may remain attractors even with owner writes. → Compare teacher boundary `P_U/P_C` with box-end natural traces and preserve checkpoints where atoms/use succeed but reallocation fails.
- **[Bridge becomes a generic energy pulse]** Constant-norm injection can improve continuation without identity. → Normalize values, use bilinear interaction, zero initialization, RMS caps, balanced 2x2 swaps, and held-out wrong-image/permutation tests.
- **[Atomizer becomes box-only]** Matching and `L_atom` can improve geometry while values remain decorative. → `L_use` separately requires description and coordinate consumption; do not promote atom box recall alone as success.
- **[Slot permutation instability]** Global assignment can flip equivalent slots across steps. → Use deterministic slot/anchor initialization, tie breaking, and switch-rate receipts; do not add EMA matching unless a later checkpoint establishes the need.
- **[Partial annotations create false background/stop]** Ordinary COCO does not prove absence. → Mask every unmatched ordinary atom and final null; restrict complete negative/exhaustion supervision to trusted examples.
- **[Trusted-only branches are inactive in the first run]** The exact S-lineage production source contains only ordinary COCO, so direct background-objectness and final-null route positives are absent; unmatched values and objectness calibration may remain weakly trained. → Bind `ordinary_partial` into config/cache identity, detach aggregate availability from router parameters before admission CE, keep native terminal CE explicit, log selected matched/unmatched status and objectness/null behavior, and limit negative conclusions about stop calibration accordingly.
- **[RMS caps damage grammar or hide useful signal]** Too much update can corrupt wrappers; too little can be ignored. → Zero-init and ramp the row path, cap admission/row paths separately, log clipped fractions and validity, but do not introduce a quality gate.
- **[Paired use loss increases compute and memory]** Correct/counterfactual likelihoods add a second causal lane. → Select one pair per eligible example/microstep, reuse semantic caches, and permit activation checkpointing only with exact typed-context recomputation and gradient parity. The causal-truncation plus FA2 `cu_seq_lens` packing lane of Decision 17 is not deferred: it lands before launch and must be proven by the packed-versus-isolated equivalence and boundary receipts. Only deeper optimizations beyond that lane — cross-branch KV sharing, vision-tower/merger output reuse across branches, fused kernels, and similar — defer to post-launch profiling, and none of them may change the loss meaning.
- **[Per-rank collective divergence returns through a new conditional path]** Any future slot kind, eval branch, or early-exit that skips or adds a wrapped forward reproduces the W8 hang. → Keep the anchor forward unconditional per slot, assert the one-step choreography digest across ranks in the two-rank test, and keep the control timeout bounded so divergence surfaces quickly instead of stalling for 30 minutes.
- **[Truncation silently changes branch targets]** An off-by-one prefix bound, dropped image token, or shifted MRoPE index would change description/geometry likelihood while still looking finite. → Require packed/truncated-versus-isolated equality on branch logits, `L_use`, and gradients, and keep the untruncated path available as the reference in that comparison.
- **[Packed branches contaminate each other]** A missing `cu_seq_lens` boundary would let one branch attend to another branch's row or prefix, which is exactly the leakage the 2x2 design exists to exclude. → Publish the FA2 boundary receipt, assert no cross-branch attention, and fail the lane on any equivalence mismatch.
- **[HF generation state leaks across sequences]** A global latch would silently bind the wrong image. → Use explicit per-sequence state, batch-index lifecycle tests, and fail on unsupported cache reordering.
- **[Layer profile is too early or late]** Block 20 is reasoned, not empirically proven. → Treat the checkpoint as the test; preserve artifacts. Changing the seam later creates a new profile and checkpoint identity rather than an untracked hyperparameter tweak.

## Migration Plan

1. Keep all existing bridge-disabled configs, caches, checkpoints, and HF/vLLM behavior unchanged by default.
2. Add the versioned bridge profile, typed owner records, payload schema, and strict config fields behind one explicit Stage 1 production leaf.
3. Rebuild presentation-specific caches; never reinterpret scalar-density or owner-record-free caches.
4. Load and validate the S step-2444 composition, then initialize the new bridge and fresh optimizer state.
5. Run deterministic contract tests and a very short exact production-path smoke. These validate mechanics only.
6. Repair collective choreography before rerunning distributed evidence: restore the bounded 120s rank-report control timeout, give every accumulation slot one DDP anchor forward, move owner-use branch forwards to the unwrapped module, and land the truncated/packed branch lane with its equivalence and FA2 receipts.
7. Run the bridge algorithm-reference battery, exact-path single-rank smoke, the two-rank uneven-branch DDP acceptance, the one-step W8 choreography digest, and the full bounded eight-rank launch smoke; require finite/identity/serialization/lifecycle correctness and no unresolved audit P0/P1, but no recall or duplication threshold.
8. If those readiness checks pass and GPUs are safely available, launch the four-presentation production training with eight-rank Accelerate and record the exact process/config/artifact identity.
9. Save all presentation-boundary checkpoints and the final permanent composition, then run natural greedy HF evaluation without a promotion threshold.

Rollback is non-destructive: stop using the new config and load the original S checkpoint or any existing bridge-disabled checkpoint. A completed bridge checkpoint is retained even when model-quality results are negative; it is never converted to bridge-free inference by deleting its payload.
