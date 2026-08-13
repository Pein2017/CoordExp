# Wave 3 Attempt 3 Pre-cost Review

Status: `HOLD`

Review scope: frozen attempt-3 launch materials only. This is a pre-launch
planning receipt, not an execution receipt and not evidence of exact resume.

## Frozen bindings

- Implementation commit: `037ab6683f9eeeb99157960f9fcf5bb3176a7044`
- Command manifest: `receipts/wave-3-attempt-3-command-manifest.json`
- Command-manifest SHA-256: `c3e4e93d997c87ad26379b0246f5536aec4f96afbc9a59be16985572a718cf42`
- Launch packet: `receipts/wave-3-attempt-3-launch-packet.md`
- Launch-packet SHA-256: `70b4f4cc7b235db0f21dcf5e3ade68d06b5db923a4876ceee6ba92ceed07ca02`
- Frozen world size: `2`
- Frozen artifact root:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r3`

The manifest and packet are immutable evidence. They were not edited as part
of this review and MUST NOT be edited to resolve the findings below.

## Findings

### P1: resumed-parent step-2 TOCTOU

The attempt-3 controller observes and authenticates the committed parent
step-1 boundary from outside the training process, then signals the process
group. The real checkpoint handler has already returned before that external
observation. The trainer can therefore enter step 2 in the check-to-signal
window, contradicting the packet's required one-step parent invariant.

Required bounded successor: in
`scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py`, route only the
resumed parent through a synchronous held-parent wrapper around the real
pipeline checkpoint handler. The wrapper calls the real handler first and,
only after successful committed step 1, blocks before control can return to
the trainer. The uninterrupted control and resumed child remain `src.train`.
The handler's failure propagates without entering the hold. No `src/` change,
admission weakening, or resume-compatibility change is permitted.

Acceptance nodes:

- `tests/training/test_reconcile_exact_resume_probe.py::test_resumed_parent_uses_held_parent_route_while_control_and_child_use_src_train`
- `tests/training/test_reconcile_exact_resume_probe.py::test_held_parent_calls_real_handler_then_blocks_before_step_two`
- `tests/training/test_reconcile_exact_resume_probe.py::test_held_parent_does_not_hold_when_step_one_publication_fails`
- `tests/training/test_exact_resume.py::test_resume_compatibility_keeps_max_steps_and_checkpoint_cadence_strict`

### P1: no outer attempt receipt

The attempt-3 manifest contains six direct commands but no single executor that
owns exclusive attempt admission, ordering, stop-on-failure, resource sampling,
and an attempt-level terminal receipt. In particular, an earlier command can
stop the chain without a signed outer receipt binding the frozen packet and
manifest to command observations, resource maxima, stop outcome, and the inner
verifier receipt.

Required bounded successor: create only the experiment-local
`scripts/probes/coordexp_swift/reconcile_exact_resume_packet_executor.py` and
`tests/training/test_reconcile_exact_resume_packet_executor.py`. The public
callable is:

```python
execute(
    *,
    manifest_path: Path,
    packet_path: Path,
    expected_manifest_sha256: str,
    expected_packet_sha256: str,
    attempt_marker_path: Path,
    terminal_receipt_path: Path,
    launch=...,
    gpu_sampler=...,
    process_sampler=...,
) -> dict
```

The CLI is `execute --manifest ... --packet ... --manifest-sha256 ...
--packet-sha256 ... --attempt-marker ... --terminal-receipt ...`.

The successor manifest schema v2 adds a machine-readable
`execution_contract` fixing exactly this order:

1. `setup`
2. `success.uninterrupted_control`
3. `success.resumed_child`
4. `rank_failure`
5. `interruption`
6. `verification`

The executor validates the frozen bindings and contract before atomically
creating an absent attempt marker with `O_EXCL`, executes each command at most
once, stops on the first failure with no retry, and writes one signed outer
receipt after marker ownership. The receipt binds implementation, manifest and
packet hashes, marker and argv observations, process/GPU resource maxima, stop
outcome, and the inner verifier receipt when verification is reached. This is
qualification tooling only, not a general orchestration framework.

## Execution and authority disposition

No attempt-3 setup, success, failure, interruption, or verification command
executed. No GPU/model work, cache preparation/publication, artifact-target
creation, or retry occurred under this review.

The two minimal TDD successors above are authorized. After they pass, freeze a
new commit-bound schema-v2 manifest and packet on new absent targets and submit
those exact hashes for pre-cost review. If that unchanged successor review is
`READY`, the lead-only executor proceeds under the current goal-level authority
without another user prompt. Any subsequent mutation requires re-freeze and
re-review, but not a repeated authorization prompt.

Historical attempt-1 and attempt-2 approvals and terminal facts remain
unchanged in their immutable receipts; this review neither rewrites nor extends
them.
