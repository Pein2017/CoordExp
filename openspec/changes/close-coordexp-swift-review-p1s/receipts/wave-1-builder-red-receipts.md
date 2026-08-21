# Wave-1 builder RED receipts (verbatim excerpts from builder reports, 2026-08-21)

Durable record of each builder's observed-RED evidence; green-only tests are
unverified evidence under this repo's rules, so the pre-fix failures are the
load-bearing half of every fix below.

## Builder A — P1-3 (src/losses/runner.py)

Defect probe PASSED against unfixed code, proving the raise precedes the gather:
```
tests/losses/test_zero_eligible_collective.py::test_defect_probe_zero_eligible_raises_before_the_gather PASSED [100%]
============================== 1 passed in 5.08s ===============================
```
Desired-behavior test FAILED for the right reason against unfixed code:
```
>       assert gatherer.called is True, (
E       AssertionError: the zero-eligible rank must enter the collective gather before raising; raising locally desyncs peers that are already inside the gather
E       assert False is True
E        +  where False = <test_zero_eligible_collective._RecordingGatherer object at 0x7f8c0b987a40>.called
tests/losses/test_zero_eligible_collective.py:64: AssertionError
1 failed in 4.65s
```
Whole-file RED before the fix: `3 failed, 4 passed in 5.31s`. Post-fix:
`tests/losses/` 206 passed.

## Builder B — P1-2 (src/runtime/{train_runtime,finite_gates}.py)

Launch layer, target-form tests against unfixed source:
```
__________ test_declared_fp16_without_scaler_is_refused_at_preflight ___________
        accelerator = _Accelerator(mixed_precision="fp16", scaler=None)
>       with pytest.raises(RuntimeContractError) as exc_info:
E       Failed: DID NOT RAISE <class 'src.common.errors.RuntimeContractError'>
___ test_declared_fp16_without_active_scaler_is_refused_before_prepare[None] ___
E       Failed: DID NOT RAISE <class 'src.common.errors.RuntimeContractError'>
_ test_declared_fp16_without_active_scaler_is_refused_before_prepare[scaler1] __
E       Failed: DID NOT RAISE <class 'src.common.errors.RuntimeContractError'>
3 failed in 6.15s
```
(`scaler1` = `is_enabled() -> False`.)

Consensus layer, ws=2, `scaler_active=False`, unfixed source (the documented
fail-open — both branches take the retained bf16 path):
```
[all-safe]   action='apply'          terminal_reason=None status='ready_to_step'                should_step=True
[one-unsafe] action='not_attempted'  terminal_reason=None status='skipped_gradient_or_overflow' should_step=False
```
Third RED from the same run: `TypeError: RankGradientFiniteReport.__init__()
got an unexpected keyword argument 'declared_fp16'` and `KeyError:
'declared_fp16'` on `rank_diagnostics[0]` — the field did not exist. Post-fix:
both branches converge `action=None,
terminal_reason='pre_wrapper_fp16_scaler_missing'`; `tests/runtime/` 193
passed + 19 new contract tests.

## Builder C — P1-1 characterization sensitivity (RED-equivalent)

Golden pinned against pre-extraction code:
`GOLDEN_PAYLOAD_SHA256 = 45c07ba45eaf902c61a86fc82c74b0468d0f3845488b5ed9b17e1b34a24286cd`,
`GOLDEN_PAYLOAD_BYTE_LENGTH = 2924`; green-before `4 passed in 5.44s`
(repeated in a second process, `4 passed in 5.89s`).

Sensitivity: one injected metadata key (`"sensitivity_probe": True`) fired
both guards, run against the pre-extraction loop AND repeated post-extraction
with the identical perturbed digest (assembler byte-equivalent even under
perturbation):
```
E       AssertionError: assert ('38c0f266a1d...8c1b84', 2945) == ('45c07ba45ea...4286cd', 2924)
E         At index 0 diff: '38c0f266a1d9cae74beb7bb52da2e6c84905b10eb808b33c704fe32a668c1b84' != '45c07ba45eaf902c61a86fc82c74b0468d0f3845488b5ed9b17e1b34a24286cd'
E       AssertionError: assert ['split', 'se..., 'pack_plan'] == ['split', 'pa..., 'pack_plan']
E         At index 1 diff: 'sensitivity_probe' != 'pack_id'
```
Perturbations reverted (git diff clean); green-after with zero fixture edits:
`4 passed in 5.55s` — the payload-byte-equality acceptance proof.

First-attempt failure kept honest: raw `pickle.dumps` of the payload is
byte-nondeterministic within one process (torch tensor storage keys are
address-derived; minimal repro `pickle.dumps(torch.arange(6).reshape(2,3))`
twice differs at byte 967), so the guard pickles with a `reducer_override`
projecting only `torch.Tensor` to `(dtype, shape, values)`.
