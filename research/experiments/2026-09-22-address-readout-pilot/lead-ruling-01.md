# Lead ruling 01: Lane B source HOLD; continue Lane A qualification

Date: 2026-09-22. Lead: `01a0c1f3-dbef-7b63-b2da-8dc7072cea8d`.
Worker: `922-worker`, `01a0c726-ad7c-7cc0-89b7-d76ac6fcf027`, Astra/low.
This is a bounded amendment to [unit.md](unit.md), within the user's approved
pilot and budget. It is not model qualification acceptance or scientific success.

## Evidence and ruling

The worker's
[blocker manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot/qualification/blocker-v1/manifest.json)
reports zero eligible genuine-detail cases in 122,218 inspected COCO rows.
There were zero model/vision forwards, optimizer updates or allocated GPU-seconds;
the model-execution wall clock has not started.

The lead read the eligibility implementation, the bound baseline processor
configuration and exact admission. Independent readback checked all 192 admitted
images (128 training, 32 calibration, 32 evaluation), all 384 original/processed
file hashes, dimensions and disjoint cohort identities. Every original has both
axes no larger than its baseline; every image plan and source processor has
`do_resize=false`. Receipt:
[lead-lane-b-readback-v1.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot/qualification/lead-lane-b-readback-v1.json),
SHA256 `0bde84025314043e347eeda1410e0298c62597269bf7548070e99ccb2f44c3f4`.
The whole-source census remains worker evidence; the independent pass certifies
this admitted 192-image support, not a fresh repeat of that entire census.

**Lane B is source-inapplicable HOLD for this pilot.** The required additional
original detail is unavailable under the frozen baseline/source contrast.
Do not substitute interpolation, crops, artificial degradation as the main
contrast, a new source, or a different hypothesis. This does not establish that
the model has sufficient visual information and does not reject H_B. Preserve
the availability evidence and zero-cell denominator. No Lane B model run is
needed merely to verify an empty eligible population.

**Lane A proceeds independently through real-entry qualification.** Lane B's
item 6 in the unit's qualification list is waived as inapplicable for this
continuation; it must not block independent Lane A model qualification. All
Lane A correctness gates and the lead gate before broad training remain intact.
Do not interpret this ruling as permission to start the full 256-update fits.

## Next worker assignment

1. Reuse the prepared code and `selection-v3/selection/manifest.json` (SHA256
   `b9a4f088466c1874ffef15380ff4fab7931b752b4208cbb2abba0de9836834d7`).
   Its 128/32/32 counts, disjointness and image bindings passed the readback;
   full scientific launch/admission acceptance is still pending. Do not rebuild
   or resample it simply because Lane B is held.
2. Finalize the immutable Lane A qualification launch configuration, preserve
   current producer source and actual input bindings, and execute the tiny real
   training/save/fresh-reload/free-generation slice. The source config contains
   historical run metadata: all new writes must go to the new pilot root; never
   inherit an old artifact writer destination while reusing model identity.
3. Demonstrate every Lane A qualification gate in the unit: zero/off native
   parity, coordinate-family mass and slot isolation, real grid/address identity,
   causal alignment/no future-label leakage, actual gate then Q/K/role gradients,
   frozen-backbone identity, learning signal, save/reload and cached/full parity.
   CPU tests alone are not these model witnesses. Ordinary repairs are yours to
   supervise within budget; escalate scientific/architecture conflicts promptly.
4. Bound the tiny qualification schedule before the first call. Start and retain
   the shared four-hour model wall clock and all allocated GPU cost from that
   first call, including failures and reloads. Keep the original eight-GPU /
   32-allocated-GPU-hour ceiling. Use independent GPU jobs where useful; fewer
   applicable lanes do not justify duplicated evidence or a new research arm.
5. Return one stable qualification candidate with exact replay commands, bindings,
   parity/gradient/update/reload evidence, CPU checks, costs, job closure and a
   measured forecast for the fixed paired full pilot. The proposed learning
   rate/schedule are not scientifically accepted by this source-availability
   ruling. Preserve all prior CPU failures and the initial blocker.

Owned paths and model/effort remain as in the unit. The worker may supervise
Luna children or implement directly. No code changes by the lead are needed.
Do not edit this ruling, unit/state/frontier/catalog or old receipts. The
`preparation.md` / `candidate-results.md` records can advance; retain a snapshot
when replacing bytes already bound by the blocker manifest.

At candidate readiness or a material blocker, append one new `LEAD_REVIEW_READY`
or `LEAD_BLOCKED` JSON line to the existing append-only `lead-events.jsonl`, with
the exact saved manifest pointer, and end the turn. A new durable lead wait is
bound to this continuation. Stop before broad Lane A training; no successor,
new Lane B contrast, publication or commit is authorized.
