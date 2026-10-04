# Retained research operators

Scientific decisions belong to [research](../research/index.md), not these files.
New methods belong to a concrete `probes/<direction>` owner with a current question
or caller. Reusable execution and integrity mechanics belong to their `src/`
owner; operator contracts belong to `tests/probes/`. Do not add another test root,
universal trainer, profile registry or compatibility alias for a retired experiment.
The entries below describe retained execution boundaries, not a launch queue.

## Implementation placement

Prefer a cohesive `probes/<direction>/` package for related probes. Reuse an
existing owner where it fits; otherwise name the package for the research
direction or shared mechanism, rather than a run, date or individual experiment.
Avoid adding one top-level module per experiment. A small, independent, stable
operator may remain a root module when a package would add no useful grouping;
do not create a framework, wrapper or one-file package just to add a directory.
Keep corresponding contracts under `tests/probes/`, mirroring package ownership.

Choose the implementation and test paths before delegation or source freezing.
Existing code need not move with every new probe: check callers and path-bound
evidence before a separately scoped migration, and preserve sealed receipts.

## Retained entries

| Capability | Entry and contract |
|---|---|
| Output-QP | `python -m probes.output_qp --capture <six-array.npz> --output <fresh.json>`; selected-output-row minimum-Frobenius solve with exhaustive supplied-state FP32 certificate |
| Readout norm | `python -m probes.readout_norm --input <explicit-arrays.json> --output <fresh.json>`; effective OUTPUT-row lower-median norm scaling, selected logits only |
| Saved row evaluator | `src.eval.saved_rows`; explicit raw/case/reference-bank inputs, class-agnostic matching, validity and recurrence separately |
| Online row-credit owner | `python -m probes.online_row_credit_owner --root <released-pair-root> --release-sha256 <exact-digest>`; one source-bound six-stage sequential invocation, separately released by the lead |
| Rule stability | `python -m probes.rule_stability cpu-smoke --output <fresh-unit-output>`; eight-rank CPU lifecycle with substituted model computation. `native-run --config <exact-lead-packet> --output <bound-output>` runs the frozen [full18/570 IoU90 unit](../research/experiments/2026-10-03-rule-stability-iou90/unit.md); generated proposals grant no native execution. |
| First-row history | `python -m probes.first_row_history prepare --output <fresh-packet-directory>` prepares tokenizer/processor inputs only. `run --config <exact-lead-release> --output <bound-output>` executes the [four-request history cross](../research/experiments/2026-10-03-first-row-history-cross/unit.md), with9 sequential forced actions and64 free actions; failed native-history fidelity stops before dependent requests. `readback --output <bound-output>` reloads the literal evidence without inference. |
| Resident rollout qualification | `python scripts/probes/coordexp_infras/vllm_dora_rollout.py --help`; bounded HF learning, resident vLLM acquisition, paired action likelihoods, and synchronous refresh. This measures infrastructure and does not change a released research backend. |

QP NPZ fields are hidden_states, target_ids, route_token_ids, base_route_logits,
top_ids and top_logits. No pickle, model loading or image/panel selection occurs.
The certificate is for the supplied states, not native generation or transfer.
Norm JSON fields are effective_output_rows, scores and token_ids. FP64 factors
multiply selected FP32 scores; input embeddings and nonselected columns remain
unchanged. Tied and untied model selection remains the caller's responsibility.

Pure functions support CPU tests. CLI qualification binds clean Git commit/tree,
required source files and input bytes before work and again before publication.
`--source-receipt` accepts only the exact current qualification identity and same
input; legacy receipts fail closed. A new qualification is not a recovered
historical run or an execution grant. External artifacts are not rewritten.

The online owner preserves issued/exited/skipped receipts and cleans only its
confirmed descendants. Execution cuts off at 2670 seconds, reserving 30 seconds
for cleanup within the 2700-second acceptance ceiling. Explicit waits use the
remaining budget; OS scheduling and I/O cannot provide an absolute wall guarantee.
`terminal.json` records cleanup and publication measurements; the durable final
`owner_terminal` stdout event and external exit receipt determine any late
finalization failure. Native acceptance requires confirmed cleanup, a finished
watcher, exit zero, and full observed owner wall at most 2700 seconds. Charge
eight slots times the larger actual internal/external owner wall, including all
cleanup and receipt finalization, uncapped. A packet or CPU test is no launch grant.

The opt-in `recall-error-floor-v4` correction recipe keeps original-prefix M in
control and uses coherent CHAIN targets in treatment: relocated M rows have
coefficient `1/m`, B rows `1/(m+k)`, with zero M contribution when `m=0`.
Common correction, geometry and containment stay fixed; v3 remains unchanged.
All three `probes.online_row_credit` stages bind the qualified recipe through
`--recipe-sha256`; a separate lead release is required for native execution.

The full-label route is opt-in: without `--full-label-region`, the online runner
keeps its existing paired row-credit contract. With the flag, the runner selects
the separate 570-label owner-region fit. `probes.full_label_fit.experiment`
owns snapshot preparation, qualification and annotation evaluation; `region`
owns CPU-testable geometry/ranking; `recipe` owns the fit recipe and LR profile;
`rollout` owns balanced generation assignment and verification. The snapshot and
qualification CLI is `python -m probes.full_label_fit.experiment --help`; exact
data, loss, metric, bounds and proposed argv belong to the
[full-label unit](../research/experiments/2026-10-02-full-label-self-rollout-fit/unit.md).
The owner CLI defaults to `--mode paired1` (six stages). The full-label packet
uses `--mode full-label --updates 2|16` and runs `run`, `readback`, then `offline`.

The former finite-panel producers, model-specific convenience loaders, stage
controllers and repair/closeout chains are no longer maintained. Their useful
results and exact historical recovery points are in the existing catalog.
Core training still retains its consumed DoRA/loss/packing mechanisms, regardless
of scientific novelty. New implementation requires an actual current question or
consumer, not another broad default directory.

## Scope-bound HF compatibility diagnostic

`python -m probes.runtime_compat --help` is import-safe and loads no model.
Pass explicit `--checkpoint`, `--inputs`, `--retained`, `--encodings`, `--policy`
and a fresh `--output` under this worktree's outputs. `--check-inputs` only checks
readability, row identities and input hashes; it does not qualify a checkpoint or
GPU execution. Actual execution requires clean source and separate GPU authority.
The compose/native-batch dependency still consumes `probes.iterative_positive.POLICY`;
the explicit policy must match it. This is the retained fixed-bank recipe, not a
new generic loader, training run, vLLM benchmark or evidence of numerical parity.
The saved 2026-09-29 result is relocated to
`/data/CoordExp/.worktrees/research-probes/outputs/runtime/vllm-dora-20260929/hf-compat/`; original source and
producer mappings live in the root OpenSpec migration receipt.

## Reusing resident acquisition

`src.qwen.vllm_rollout.VllmDoraRollout` owns the resident engine; the caller owns
HF forward/backward, optimizer state, and snapshot identity. Configure coordinate
normalization once, then complete each acquisition before updating HF. Await
`refresh(...)` before acquiring the next snapshot. Acknowledgement includes
derived factors and cache invalidation; engine startup is not repeated.

Use an explicit `NativeGenerationPolicy` with `use_model_defaults=False` and
aligned per-request seeds for full-support temperature-one sampling. Request
`trace=True` for both raw and normalized-policy selected-action likelihoods.
The returned `ContinuationResult` keeps literal action IDs, including EOS and
sampled PAD. The two likelihood channels come from the same vLLM acquisition.
Set `allow_pad_tokens=True` when greedy acquisition must preserve literal PAD too.
Paired trace capture supports the native V2 sampler with TP=1/PP=1 and ordinary
single-token decoding; speculative and batch-sharded sampling are rejected.

HF uses `src.qwen.coordinate_policy.MedianPolicy` for the same normalization
arithmetic and differentiable current factors. Literal multimodal suffix replay
uses `src.qwen.native.exact_history_inputs(..., prompt_only_media=True)` together
with `prompt_only_placeholder_masks` to keep original image placeholders bound
to the prompt. These interfaces preserve action identity; they do not promise
bitwise agreement between HF and vLLM or authorize changing a frozen experiment.
The bounded native qualification, measured costs, numerical differences, and
remaining integration boundary are recorded in
[`verification.md`](../openspec/changes/archive/2026-10-03-unify-resident-rollout-policy/verification.md).
