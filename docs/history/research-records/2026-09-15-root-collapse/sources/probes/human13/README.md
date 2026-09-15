# Human13 finite-panel interventions

These are separate scientific profiles over the frozen 13-image / 392-owner
panel, with nested N2, N4, and N13 subsets. Same-panel certificates do not establish
transfer or native deployment quality. No N256 solver profile is included.

| Entry | Preserved protocol | Original producer at archived Human13 `aa7cdccb9` |
| --- | --- | --- |
| `python -m probes.human13.output_qp` | Shared selected-output-row minimum-Frobenius QP; FP32 exhaustive certificate; separate fresh natural-greedy gate | `scripts/research/run_human13_output_qp_same_panel.py` |
| `python -m probes.human13.magnitude_qp` | Exact magnitude VJP/JVP operator and vocabulary-separation mechanics; no finite-difference fallback or claim of a completed QP solve | `scripts/research/run_human13_dora_magnitude_qp.py` |
| `python -m probes.human13.magnitude_finite` | One shared AdamW update across canonical routes; per-image CE sums divided by total decisions; exact Source restoration; materialization and fresh natural readback | `scripts/research/run_human13_dora_magnitude_finite_overfit.py` |

Scientific records are under `research/investigations/qwen3-vl-dense-enumeration/experiments/`:
`2026-08-31-human13-shared-output-qp-same-panel-overfit`,
`2026-09-01-human13-dora-magnitude-qp-bridge`, and
`2026-09-01-human13-dora-magnitude-finite-overfit`.
Original records/code remain recoverable at
`archive/research-restructure-20260909/human13-output-qp-identity-generalization`.

The package-local source config and its inherited base are copied byte-for-byte.
Panel/image, checkpoint, tokenizer, and special-token source-gate hashes remain
checked. The fixed research base's source-gate study and roundtrip receipt are
read as evidence; no sibling-worktree Python code is imported.

Run from the repository root. This CPU-only preflight reads the actual panel,
image bytes, checkpoint files, source-gate evidence, and tokenizer; it does not
load or forward a model:

```bash
python -m probes.human13.output_qp --check-bindings
python -m probes.human13.magnitude_qp --check-bindings
python -m pytest -q probes/human13/tests
```

Examples for a separately authorized model run (all destinations must be fresh):

```bash
python -m probes.human13.output_qp capture --stage N2 --image-id 6040 --output-dir /tmp/human13-new/output-qp/capture-6040
python -m probes.human13.magnitude_qp operator-image --image-id 6040 --receipt /tmp/human13-new/magnitude-operator.json
python -m probes.human13.magnitude_finite --stage n2 --output-dir /tmp/human13-new/finite --steps 1 --learning-rate 0.001 --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --check-every 1 --seed 0
```

The final example defines an explicit one-step execution, not a scientific
success threshold or an approved launch budget. Output-QP `solve` consumes the
stage's captured files. `verify` and `aggregate` retain the fresh-process Source
A/B/A and RP1.0 primary / RP1.10 monitor rules. Magnitude finite `materialize` and
`readback` take explicit candidate/adapter/output paths.

Historical N2 finite v1 receipts retain their exact pinned candidate SHA256,
including receipts that contain a stage field. Newly produced finite candidates
use `human13_dora_magnitude_finite_candidate.v2` and the explicit
`probes.human13.magnitude_finite` producer identity. Their receipt-content and
payload hashes, stage/order, owner count, and decision count are checked just as
for N4/N13. Original receipts are never rewritten. Source bindings also record
the current package producer independently of the historical scientific inputs.

CPU acceptance covers the original 23 tests, a native causal-history
forward/backward counterexample, strict token-span/dedup evidence consumption,
and new-versus-historical receipt binding failures. A copied independent root
with no `scripts/` directory passes these tests, actual preflight, saved N4
aggregate readback, and legacy N2 receipt loading with original Human13 worktree
access forbidden. The saved N4 aggregate exactly matches the original reader.
No new real-model forward, generation, derivative, or scientific run has been
performed; full-model semantic acceptance remains outstanding.
