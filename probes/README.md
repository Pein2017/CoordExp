# Retained research operators

Scientific decisions belong to [research](../research/index.md), not these files.
The small implementation set is explicit; there is no universal trainer, profile
registry or compatibility alias for a retired experiment.

| Capability | Entry and contract |
|---|---|
| Output-QP | `python -m probes.output_qp --capture <six-array.npz> --output <fresh.json>`; selected-output-row minimum-Frobenius solve with exhaustive supplied-state FP32 certificate |
| Readout norm | `python -m probes.readout_norm --input <explicit-arrays.json> --output <fresh.json>`; effective OUTPUT-row lower-median norm scaling, selected logits only |
| Saved row evaluator | `src.eval.saved_rows`; explicit raw/case/reference-bank inputs, class-agnostic matching, validity and recurrence separately |
| Online row-credit owner | `python -m probes.online_row_credit_owner --root <released-pair-root> --release-sha256 <exact-digest>`; one source-bound six-stage sequential invocation, separately released by the lead |

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
