# Lead acceptance: technical qualification

2026-09-22. **Technical qualification: lead-accepted. Scientific outcome: pending.**
Continue the already-authorized fresh nominal fit and matched source evaluation
under [unit.md](unit.md), [runtime ruling 02](lead-ruling-02-runtime-reference.md)
and [repair ruling 03](lead-ruling-03-branch-localization.md). No new arm, budget,
runtime baseline or acceptance threshold is introduced.

The lead verified the complete-v1 manifest SHA256
`7d99eed4bb83fa7ed6bbc6e79584bf70295a36925fc96ba3ab5284461c81507c`
and all 23 bound files under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-alignment/`.
The frozen production-plan-v4 SHA256 is
`e598cc79d352c8b4b44a0b80a96f9916ab035d95f94a2ae47c77539970c78c01`;
its 512-update schedule SHA256 is
`1ba7cae0dbf96b459d8af9e76c3a9e433ee385fcea5114ab6e6a618602b8a82a`.

Source OFF parity and trained-checkpoint fresh reload each retain matching full
response/full-vocabulary logits hashes, maximum difference 0.0 and equal short
greedy outputs on all three cases. Saved single-rank and unequal two-rank
qualification receipts retain zero loss/logit-gradient reference differences,
the planned global segment denominator, detached/wrong-grid falsification,
frozen-base preservation and intended trainable gradients. The saved distributed
moment comparison passes its declared bounds and rejects the half-gradient
mutation. All 903 trained tensors reload exactly.

The bounded source/caller review found no mature or restored target receiving
the new-only pending initialization marker, no PEFT forward/backward bypass,
and no persistent device-move reset hook. The lead freshly ran
`python -m pytest -q tests/adapters/test_dora_setup.py::test_new_dora_finalization_is_once_and_restored_targets_are_untouched`:
one passed. The independent review used the bound production implementation;
the lead did not repeat GPU qualification.

Resume is numerically checked, not bitwise reproducible. The retained strict
comparison fails; the numerical comparison records adapter max 3.3587e-6,
embedding max 1.6494e-6, moment max 0.0001432681 and relative L2 0.0054508049.
The cause is unresolved. Exact saved-to-loaded preservation is a separate passed
check. Scientific fitting starts fresh from the declared source, not these
qualification updates. Historical Mixin BF16 versus current live FP32 source
drift is separately retained; this acceptance makes no historical equivalence
or address-alignment efficacy claim.

Qualification snapshot cost is 745.957516670227 allocated GPU-seconds, with 21
producers terminal and zero scientific updates at that snapshot. Original wall
start 1790062963.818157 and package limits remain. Production may now be active;
the qualification snapshot is not a current all-jobs-terminal statement.

The lead's feedback note was moved byte-exactly to
[lead-localization-feedback-01.md](lead-localization-feedback-01.md). Its SHA256
remains `399274f2082fe845f184d720825498b64e5c8baf47070f23c02ba8bed040d14f`;
`qualification/lead-feedback-relocation-v1.json` records the old/new paths.
The original dispatch and worker qualification records remain unchanged.
Fresh research knowledge, both-root output-layout and `git diff --check` checks
pass. The unrelated recurrence-transition-readback state is now registered by
its owner; no edit to that unit was needed here.

Worker: continue fitting and the fixed natural-generation evaluations as
authorized. Measure actual long-generation throughput and retain all failures
and costs. Monitor remains separate from fit selection. Report decision-bearing
conflicts or the stable candidate directly; no acknowledgement-only reply,
full-dataset launch or self-acceptance is needed or authorized.
