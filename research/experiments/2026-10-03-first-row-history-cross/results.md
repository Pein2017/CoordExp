# Results

CPU candidate qualified; native execution remains unreleased. No model forward,
research checkpoint load, optimizer, replay, update or scientific result occurred.

The candidate composes maintained `native_components`, input preparation,
`MedianPolicy`, `generate_continuations`, full-label rendering, parser and IoU
helpers in `probes/first_row_history.py`. A small sequential processor records
the incoming median argmax and requested-token likelihood before forcing each
of the first9 actions into a fresh score tensor. It returns the incoming tensor
unchanged for all later steps. Raw selected likelihoods and forced-selection
likelihoods remain separate. The reader excludes the supplied row from free-row
target recovery and retains it as an earlier duplicate-overlap partner.

Preparation completed with exit0 at
`outputs/research/physical-fn-recovery/2026-10-03/first-row-history-cross/prepared-01/`.
It used the actual tokenizer/processor, verified the maintained first-person GT
row, original prompt/media/grid binding, all49 selected-image annotation
identities, both saved native prefixes, and the saved first73 actions. Saved A0
contains326 actions and A16 contains3084, so both natural short runs must stop at
length73. The prepared packet and unreleased proposal bind the small input,
saved-action and checkpoint-manifest JSON files; preparation did not read tensor
payloads. The preparation revision is context, while the later native release
must bind its own clean execution revision. Prepared bytes need no rewriting.

The first CPU slice crossed actual CLI parsing, request/frontend qualification,
shared median arithmetic, continuation generation, exclusive artifact
publication, reload and the report consumer. Only model/device computation was
substituted. Four fixture requests produced292 actions through two fixture
sessions, with zero actual checkpoint loads or model forwards. The output is
explicitly labeled `CPU_FIXTURE`; its teacher-shaped continuations establish no
scientific ability. A fresh subprocess reproduced the saved readback, exit0.
Raw evidence is under
`outputs/research/physical-fn-recovery/2026-10-03/first-row-history-cross/cpu-qualification-01/`
(`invocation.json`, `qualification.json`, four condition files, fidelity files,
`readback.json`, `terminal.json`). The terminal records0.826 seconds inside the
fixture producer and peak process RSS1,173,319,680 bytes, including the loaded
test/frontend runtime; these are not native resource estimates.

Focused CPU checks passed in three nonduplicated selections:

| Check | Exit | Result | Raw log |
|---|---:|---|---|
| `pytest ... -k real_cli` |0|1 passed,14.68s|`.local/scratch/first-row-history-cross/early-cli.log`|
| `pytest ... -k 'not real_cli'` |0|17 passed,12.98s|`.local/scratch/first-row-history-cross/negative-checks.log`|
| `pytest ... -k 'pre_force_observation or consumer_binds'` |0|2 passed,7.61s|`.local/scratch/first-row-history-cross/trace-boundary-checks.log`|

The counterexamples distinguish forced-prefix causal conditioning from a new
prefill, median-before-force observations from raw observations, raw-tensor
preservation from in-place corruption, and nine-step forcing from leakage into
the free suffix. A0 and A16 argmax/token-fidelity failures return technical HOLD2
before their dependent GT and remaining requests; an A16 HOLD retains A0/GT as
partial evidence. Compute exceptions, unreleased execution, frozen packet drift
and nonfinite likelihoods fail visibly. The reader rejects conditions published
after HOLD and binds the full condition bytes, including likelihood channels.
Target/forced-row accounting checks retain invalid and censored boundary rows,
same-category annotation candidates and class-agnostic duplicate partners.

The seven base files in the original invocation have the same expected policy
hash identities and unchanged sizes/mtimes at this read-only checkpoint. Native
entry will reuse that original immutable-base evidence and hash the two selected
checkpoint payloads once through the existing manifest checker. It will not
instantiate or load an optimizer. Native cache/numerical fidelity, checkpoint
restoration, actual CUDA memory/runtime, and all four scientific continuations
remain unproven until the exact one-invocation lead release.

Proposed command after the lead commits and binds a release:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -m probes.first_row_history run --config outputs/research/physical-fn-recovery/2026-10-03/first-row-history-cross/native-release-01.json --output outputs/research/physical-fn-recovery/2026-10-03/first-row-history-cross/native-01
```

The proposal caps execution at four serial singleton requests, two checkpoint
loads and292 emitted actions, with zero optimizer/backward/replay/export work.
It grants no launch or replacement execution. The implementation owner stops at
this CPU candidate; the lead owns acceptance, clean source and exact release.
