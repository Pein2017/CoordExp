# Visual instance binding

Status: `candidate`; `qualification-01` is the sole primary evidence. The
optional release was skipped under the lead ruling, and no further model calls
were authorized.

## Qualification and execution

`qualification-01` completed four real entries on `cuda:4..7`: tied-885,
untied-885, tied-14038, and the tied-14038 matched control. All four receipts
are `candidate_complete`, giving 16 primary cells (four states times clean,
A, N, and unrelated conditions), 80 model forwards, 80 vision forwards, and
259.62838562577963 GPU-seconds. Native source replay included the row entry and
terminator in every check; the largest source trace error was
`5.340576171875e-05`. Clean PNG re-encoding had maximum full-logit error `0.0`
for every candidate row, and every primary state had a nonzero condition
difference.

The primary rows retain full-vocabulary log probabilities for the complete row,
`x1`, `y1 given candidate x1`, joint corner, and full-row paths. Candidate-set
accounting includes the opener and terminator and does not renormalize over the
finite candidate set.

`states/` contains the four `scientific-01` runs that completed before the
08:03:12Z milestone ruling. They are preserved byte-for-byte as redundant,
audit-only protocol debt: 80 model forwards, 80 vision forwards, and
251.7555357143283 GPU-seconds. They are excluded from primary evidence,
denominators, selection, aggregation, replication claims, and response
matrices. An independent CPU comparison covers all 64 state-condition-row
combinations across complete-row, `x1`, `y1 given candidate x1`, and joint-corner
values and finds exact equality (`max_abs_delta=0.0`). The chronology is bound to
retained launch and terminal records; it is not established from score receipts
alone and is not characterized as a violation of the later instruction.

The CPU-only acceptance verifies decoded RGB dimensions and grid identity, clean
compositor pixel equality, exact mask rectangles, nonempty changed masks,
unchanged complements, all primary receipts, and all row/candidate-set sums.

## Saved response evidence

`response-matrices-v2.json` reports the four columns side by side for each
region and labels each candidate prefix as `native_reachable` or
`supplied_candidate_under_native_prefix`.

The primary strata are three target-first-revisit states and one matched
control. The retained untied-885 matrix includes its cross-effects; they remain
visible as relative candidate-row effects without changing the primary
denominator.

For the tied-14038 failure, A removal changed the actual A row by
`x1=-0.0937`, `y1-given-candidate-x1=-0.7182`, joint corner `=-0.8119`, and
full row `=-1.4761`; the supplied N candidate changed by `x1=-0.0490`,
`y1-given-candidate-x1=+0.7230`, joint corner `=+0.6740`, and full row
`=+0.6699`. N removal changed the actual A row by `x1=+0.0798`,
`y1-given-candidate-x1=-0.1087`, joint corner `=-0.0290`, and full row
`=-0.0648`; the N row changed by `x1=-0.0682`,
`y1-given-candidate-x1=-0.5011`, joint corner `=-0.5693`, and full row
`=-1.2150`. These are relative ablated-minus-clean effects. The other three
primary states show the same broad selective direction: A removal lowers A
preference and N removal lowers N preference, while unrelated effects are small
relative to the target masks.

This finite evidence retains differentiated local visual support at the admitted
recurrence and matched control boundaries, so H2 is downgraded on this support.
Cross-effects remain, and the result does not establish distinct native `x1`
owner selection. Lane B does not test the spatial progression hypothesis H1,
physical-owner recovery, or training origin.

## Evidence and cost

- Integrated two-lane candidate: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate/candidate-manifest-integrated-v2.json` (`1001bdfb20eaddc9455d4c228c022b876c3ed7f074df834a5f66459698ca9ae4`).
- Candidate manifest: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding/candidate-manifest-v2.json` (`a680cb307ca6571900da583798aed6de14e269d8dbcac0c7c29b5b772d9445bd`)
- Response matrices: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding/response-matrices-v2.json` (`df2ad8c7c1a0677417d44bb58912ac7060cf19e544d4b8e6e58ba28679c449e3`)
- Reduction: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding/reduction-v2.json` (`7361b053e7adf5f84a92117f68baa70a4c0556ff87ea3cc7119cdf7a5fabd0ca`)
- CPU acceptance: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding/cpu-acceptance-v2.json` (`650f4692ce99c88e6b6efbb71a74ef0997a367fa2eb86fe367788844660780bf`)
- Combined executed cost, including audit-only duplicate runs: 160 model forwards, 160 vision forwards, 511.3839213401079 GPU-seconds, and 0 retained model tensor bytes.
- Run-root artifact bytes: 37,872,373 primary plus 37,870,787 audit-only, 75,743,160 combined.
- Bound admission SHA256: `8863a4eb6d00ed9cb27cf8dd41e32429e7081e3bf363907b23bf4a7a0ba1d7d8`.
- Bound budget SHA256: `ad027198065151367a9778171f7c109ab718e4ccf428658f4d4352899dc6401a`.
- Bound wall-limit amendment SHA256: `3e1fd8135882c3c144c877e3a47e4c9977eb8f651345514a571422637ed0b836`.
- Lead milestone ruling SHA256: `a069b4756b7e6bae6e8e8ac10b750e906c85665b56d878eaeb4ff270dc226b13`.
- Redundant replay ruling SHA256: `c686bcbc01fa44b42d80c5f900ffe97239305f4dbdc76227277c754d492cd9ea`.

The unversioned `candidate-manifest.json`, `reduction.json`,
`response-matrices.json`, and `cpu-acceptance.json` are preserved unchanged as
superseded audit records. Their hashes are recorded in
`candidate-manifest-v2.json`.

Reproduction checks (CPU only; no model calls):

```text
python -m probes.training_set_completion.visual_instance_binding.selfcheck
python -m probes.training_set_completion.visual_instance_binding.finalize_v2 --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding
```
