# First Matched StateBank Assembly Rejected Before Training

Source session: active July 22 2026 treatment screen.

Source handles:

* `scripts/research/assemble_source_preservation_multi_route_state_banks.py`
* `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/state-banks-v1/`
* `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/unit.md`

The immutable `state-banks-v1` artifacts must not be used for training. A
read-only fixed-point audit reconstructed semantic identity as image, exact
prefix, exact candidate row, and physical owner. The single-route arm had 992
event identifiers but only 987 unique semantic events; the multi-route arm had
988. These were authentic sampled and Source rows that happened to be exactly
identical, but counting each twice would give duplicated training credit and
would confound the arm contrast because the two arms had different overlap
counts.

The audit also found that the assembler discarded rollout-level
`model_identity` and stamped every event with the reference StateBank's
checkpoint identity. Current input artifacts appear to come from the expected
Source checkpoint, but the generated bank did not prove that binding and the
same implementation could silently accept another checkpoint later.

The repair contract is narrow: remove Source semantic identities from each
sampled candidate pool before final 496-event selection; preserve one sampled
route per image in the single arm and at most three in the multi arm; fail if
496 cannot be reached; assert 992 combined identities; and parse and compare
every rollout artifact's own checkpoint identity with the reference binding.
Regenerate under a new immutable run identifier and independently re-audit
before any GPU training.
