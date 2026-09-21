# Root paired-smoke grant — corrected final-readback candidate

Root accepted the exact margin table (5fresh tests + independent all56/6030
raw-source comparison), inspected the scorer/normalization/gradient boundary,
then checked the one final-readback correction and freshly passed7scorer tests
and persisted-input validation. A subsequent root diagnostic-print KeyError
was in root's assumed manifest shape, after validation; the clean validation
rerun passed and is not a producer failure.

One sequential paired2step smoke is now authorized for owner
`/root/sol_high_training_recovery` on GPUs0–7, with no other research GPU grant.
Use exact `trainer-preparation-v2/inputs.json` SHA256
`70aeea1e9cd7e6d7b4383df541d878f6b5e3cbf6e656208eedc7c03c39985b60`
and producer `probes/dora_owner_learning/margin_preserved_train.py` SHA256
`08341459d2713e3cfb0e496f8840d879925a8cdde324fe080b7edd949fc72427`.
Shared base engine and all prior candidate/failure artifacts remain untouched.

1. Run weight0 exactly once into rawchild `smoke-weight0`. Freshly verify its
   exact retained A2 adapter-state parity and frozen counters/resources. Stop
   and preserve failure on any mismatch; no automatic retry.
2. Only after weight0 passes that exact mechanical gate, run weight10 once
   into `smoke-weight10`. This conditional grant does not require another
   root permission round. Freshly verify receipt, zero initial margin force,
   actual active margin signal/update effects and final post-update readback.
3. Cold-reload weight10 once on GPU0 using the proposed exact cold-check;
   require the three saved positive scores to reproduce. Report bounded
   lifecycle cost/peaks, per-step margin stats, final-reference counters,
   hashes and owned-process release as candidate evidence.

Each smoke:2updates,8loads,356model/image forwards including56actual-final
normal reads, unchanged backward/sync counts, no sampling. Reconcile exact
processes/output targets first; ignore expected stress occupancy. One launcher
and long blocking wait per command, full logs and exits, no nested agents.
No full-C grant or endpoint execution is implied. Root reviews actual paired
smoke evidence before the one32step launch.
