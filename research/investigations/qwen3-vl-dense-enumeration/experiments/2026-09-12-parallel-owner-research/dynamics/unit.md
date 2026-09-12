# Dynamics: coordinate branch escape and return

Status: finite panel and cold consumer verification complete; scientific candidate awaits root acceptance.
Scientific owner: `astra_high_greedy_dynamics`; root owns lead acceptance.

## Frozen question

From unchanged Stable50, do one-token legal x1 alternatives change immediate
and returning strict recurrence under the original three loop histories and
their saved-native nonrepeat controls, without hiding invalid or later output?

This distinguishes narrow greedy path capture from locally restoring
recurrence. It does not identify attention/KV circuitry or authorize training.
The strongest competing account is that forced off-policy coordinates cause
incoherence rather than a useful escape; invalid and malformed outputs remain
in the denominator and same-image nonrepeat controls are retained.

## Inputs and fixed selection

Use original `351017-c01`, `417044-c01`, `477415-c02` h from the accepted
positive-branch manifest, never h+c or a trained checkpoint's replacement h.
Repeat history token lengths are 19,79,63. Control is the latest complete
valid row in each h that is not a strict repeat of earlier rows; selection
uses only the saved Stable50 decode. Control history lengths are 9,69,54.

Anchor adapter:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/training/adapter`.
Adapter-model SHA256:
`94d6aa4183af437a63fcc1c407430c05f16f66df6d69887babf8395d80df653d`.
The packet seals source images, adapter, source embedding delta, exact prompt
and continuation IDs, source packet and candidate manifest. The inherited
source-config adapter is explicitly overridden with Stable50 by existing
`checkpoint_config`; the source embedding delta stays unchanged.

## Intervention and denominators

At the first coordinate x1 of the next row, preserve all earlier tokens and
force exactly one coordinate token. Alternatives are selected before their
free outcomes:

1. Native self token, a no-op replay control.
2. Three highest-logit non-native coordinate tokens. Rank all legal coordinate
   tokens by descending FP32 logit, ties by smaller token ID.
3. Three nearest non-native coordinate bins by absolute bin distance, ties by
   smaller token ID. These are geometric neighbors, not probability neighbors.

Legal means token in coord_0..coord_999; it does not mean the later completed
box is geometry-valid. Record native token, raw-softmax logprob, logit gap,
coordinate rank and bin displacement. No outcome-conditioned candidate change.
Overlapping family tokens execute once with both memberships retained.
Six boundaries yield exactly 6 self +18 probability +18 geometric memberships
and at most42 unique cells. Add3 natural anchors and6 scoring replays. No
temperature, coordinate-slot, donor, seed, or checkpoint sweep.

## Runtime and evidence surface

Physical GPU4 exclusively, one HF FP32/SDPA native model per case, existing
patch-embedding linearization and unmerged DoRA/source embedding route.
Reuse `load_policy`, `build_requests`, `prepare_native_inputs`, `prepare_replay`,
`generate_continuations`, `native_record`, native pixel projection and existing
global owner matching. Generation is greedy RP1, no model defaults; each cell
continues to native EOS or remaining 3084-total-action-token cap. No short
artificial cap is substituted to save cost.

Natural anchor must reproduce saved Stable50 action IDs. Native self must
reproduce that same full natural action, and first-row repeat/control admission
must hold. Score target must be live full-vocabulary argmax at exact causal x1.
First real slice uses351017 anchor, repeat-boundary score/selection, native-self
and one high-probability alternative through full cap, durable parser and cold
readback. This is mechanics evidence within the declared panel, not an extra
scientific arm. Reuse its complete cells when finishing the case if practical.

## Outcomes, credit and stop

Primary descriptive outcomes: strict recurrence at the mixed intervention row,
first two complete rows, and first fully autonomous complete row; returning
recurrence throughout the full suffix. Count each later valid box once under
class-blind native-pixel IoU>.95 against all earlier valid supplied/mixed/free
rows. Keep invalid rows and parser drops between valid rows in the ledger.

All supplied history and the row containing forced x1 receive **zero autonomous
owner credit**. That mixed row's recurrence is still a valid interventional
outcome. Autonomous owner sets/F1 at IoU50/60/80 exclude the entire mixed row
and compare with the matched native-self suffix. Full-action matching is
descriptive only. Report raw starts, valid/free counts, geometry-invalid and
other drops separately, EOS/caps, family denominators, exact gain/loss owners.
An early nonrepeat followed by a later repeat is not a safe escape. Nonrepeat
does not establish visual correctness or a missing owner; uncertain rows remain
annotation-relative/unknown-neutral.

Stop after this finite panel and cold reduction, regardless of sign. Technical
failure preserves its directory, logs and exit status; repair same-contract
mechanics, never backfill cases or change candidates to obtain a result.
No coordinate-frame transport launch: a possible later design is a separate
question and requires root decision after first-wave evidence.

## Estimated resources and acceptance

At most45 complete generation calls and6 score replays, <=138780 generated
tokens before dedup/EOS (loose bound ignoring supplied prefixes), three normal
full-case model loads. First-slice transport may add a load, not new scientific
cells. Estimated FP32 peak CUDA12–18GiB and total assigned GPU time2–4hours;
these are estimates, not a portfolio budget cap. Persist actual counters,
RSS/CUDA peaks, time, raw tokens, model/batch identities and failures.

CPU acceptance:
`python -m pytest -q probes/parallel_owner_research/test_dynamics.py`.
Packet:
`python -m probes.parallel_owner_research.dynamics prepare`.
Producer/consumer:
`python -m probes.parallel_owner_research.dynamics run --help` and `reduce --help`.
Root independently inspects packet, exact slice artifacts and final reduction;
worker completion is candidate, never self-declared lead-accepted.

## Slice acceptance and continuation

Root independently verified the first slice terminal, packet/record hashes,
full3084 native/self exactness and raw burden ledger on2026-09-12, then granted
the frozen remaining panel on GPU4. Authoritative slice receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/dynamics/slice-351017-v1/terminal.json`.
Its two cells/anchor/selection are reused rather than regenerated. The early
nonrepeat branch still had late recurrence and cap; interpretation remains
bounded to that case pending the frozen full reduction.
