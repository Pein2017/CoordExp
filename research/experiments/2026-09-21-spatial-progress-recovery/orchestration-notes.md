# Spatial progress recovery orchestration record

Status: `candidate`; scientific acceptance belongs to lead
`01a0c1f3-dbef-7b63-b2da-8dc7072cea8d`.

## Binding and ownership

- Worker: `01a0ba19-8004-7400-b8af-98ce9b9df0cd`, Sol/xhigh.
- Reused child: `/root/spatial_gate`, dispatched as `gpt-5.6-luna` with
  `reasoning_effort=max`. It was reused because it was the original Lane A
  execution owner and retained the failed runtime context. The dispatch contract,
  child report, and terminal agent state agree on this binding; provider token
  usage is unavailable.
- Luna owned implementation, tests, qualification, all eight model cells,
  reduction, and candidate integration. The worker made no implementation edits.
  The worker wrote this orchestration record and performed read-only acceptance.
- The child did not delegate. No reviewer agent or second scheduler was used.

## Delivered prompts and feedback

### Initial recovery prompt

> Read the authoritative recovery unit and current AGENTS/storage policy. Keep
> the frozen admission and predecessor artifacts immutable. You are the sole
> Luna-max implementation and execution owner: reproduce the supplied
> `KeyError('tokens')` on frozen CPU data, fix the real caller boundary, add a
> caller-level RED/GREEN check, qualify one production-shaped entry, then run
> the same eight frozen cells across GPUs 0 through 7. Use the new recovery root
> only, preserve source with `preserve_source`, retain every attempt and cost,
> produce a deterministic reducer and one candidate, do not delegate, release,
> replay Lane B, commit, or edit lead-owned records. The known wrong-panel lookup,
> parser-metadata/token mismatch, and insufficient old selfcheck are supplied
> hints, not your discoveries.

### Corrective feedback before model entry

> Current `runtime.py` couples `ROOT` to both frozen predecessor inputs and new
> execution outputs, and it checks old qualification directories before honoring
> explicit `--output`. Separate immutable predecessor selection paths from the
> recovery run root. Old directories remain immutable evidence and are not a
> recovery gate. Keep the predecessor admission, budget, and wall bindings while
> every new attempt/result uses the recovery root.

This feedback changed the next action: the child introduced distinct predecessor
and recovery roots, resolved explicit output before campaign checks, and then ran
qualification. No model call preceded the corrected path.

### Terminal packaging feedback

> Qualification and all eight frozen cells are terminal. Make no more model
> calls. Preserve superseded reductions, finish one stable candidate and manifest,
> include the first-qualified evidence and full cost/job closure, then stop.

## Failures, fixes, and outcome

- Supplied historical failures remained immutable: the old first qualification
  had zero forwards and the old repair reached seven model/vision forwards before
  `KeyError('tokens')`.
- Luna reproduced that interface failure on frozen CPU data. `_actual_row`
  returns parser offsets rather than serialized tokens. The fix slices the exact
  row from the bound native sequence using those offsets and validates the slice.
- The prior selfcheck had exercised helpers without crossing the failing caller.
  The new caller-level test reproduces the old failure and then checks trace
  parity through `_trace_parity`.
- After the worker's path-coupling feedback, the recovery qualification passed:
  maximum full-vocabulary log-probability error
  `7.3909759521484375e-06` under tolerance `2e-4`.
- All eight frozen primary cells completed without retry. Luna also corrected
  reducer-only wall binding, fixed-candidate selection, crossing effects, and
  vision-forward accounting before producing candidate v2.

## Worker verification

The worker independently checked the current committed refactor and candidate:

- all 53 manifest bindings resolved and hashed correctly (31 unique files);
- all 126 original/capture pairs across nine source snapshots were byte-equal;
- no Python, Markdown, shell, or bytecode file exists under the recovery output;
- caller selfcheck and Python compilation passed;
- a fresh reducer replay was byte-identical to
  `reduction-candidate-v2.json` (SHA256
  `4bd9de08ea6a1780d3dc601eb4e6ba1775c17b85640af74f5f788539aee6e32f`);
- candidate regeneration check and both-root output-layout scan passed;
- nine retained recovery attempts are terminal and no owned model process is live.

The first reducer check used `mktemp`, which creates the target file; the reducer
correctly rejected overwrite via exclusive creation. Repeating with a new absent
path passed and produced the byte-identical result. This was a verification-command
mistake and caused no model call or evidence mutation.

## Cost and bounded lesson

Recovery used 69 model and 69 vision forwards, `494.9969075322151` GPU-seconds,
and 44,065,258 retained artifact bytes. Cumulative historical plus recovery cost
is 236 model and 236 vision forwards and `1028.4273607879877` GPU-seconds. Saved
dispatch wall elapsed is `6071.743703842163` seconds under the 7200-second limit.

This single assisted trace shows that Luna-max completed the bounded repair and
execution after the worker made the shared path/output coupling explicit. It does
not measure unaided Luna performance or support a general model ranking. The
missed interface was a caller contract hidden by a helper-only selfcheck; the
useful coaching named the real entry boundary and separated immutable inputs from
new outputs. Scientific interpretation remains lead-owned.
