# Production activation attempt 1: pre-model orchestration failure

Date: 2026-08-10 UTC

Verdict: infrastructure failure before run identity, model load, or optimizer
work. This record does not complete OpenSpec tasks 4.14 or 4.15 and does not
authorize a retry.

## Bound identity

- repository HEAD: `7f399a2f04bbea615e66c42531381c10bfb42fdb`
- resolved production config fingerprint:
  `770f705f5357fdfc08d1d6fb332d828874265d296c5f567ae4c64266cd3c056c`
- launch intent key:
  `02bdbc993c57f4e3b7f38506d40d80af9acbc2ab327a1330ee1e86789078d3ad`
- singleton claim fingerprint:
  `ca087ebb4d98472188d84b4bab0cb5dcb86e14b59290f470d9246b407a20eeae`
- activation fingerprint:
  `12d4d4b6893e862e31429b69634bccd05ef1ecc696cf66b7cca484e28d703bc0`
- activation nonce:
  `3b6f979d2b39474f29d88869e8af6837ebbde0d539c9e4e4b1e1fd830586185a`
- launcher PID: `3920046`
- activation created: `2026-08-10T19:24:23.318183+00:00`
- binding failure recorded: `2026-08-10T19:35:53.453239+00:00`

Canonical ledger root:
`/data/CoordExp/outputs/prod/coordexp_swift/.owner_bridge_stage1_launch_ledger`.
The claim, intent, activation, eight worker admission slots, logs, and binding
failure are immutable external authority. They must not be deleted, moved,
renamed, or overwritten.

## Executed outcome

All eight workers passed the process-bound admission contract. Their slots
were published between `19:24:34.298352Z` and `19:24:35.691972Z`. Accelerate
then exited with return code 1 before publishing any matching `run.json`.

The binding failure has receipt fingerprint
`2505ff27e404bffbd54b8a33262a9b86e8f8bb6a146310ec45540d8db52ed92f`.
Its file SHA-256 is
`cda4cd4b970d12697a98965cd0ea66f27291a93b548344334a70f6aa218dbc47`.
The stderr SHA-256 is
`8020c3ea8fd5be58c19af29a557386b4fe0f34d3bc5b33bb05a55b9b47827aa5`.

No production run directory or `run.json` was created. No model-load,
optimizer, checkpoint, training metric, or finite/applied heartbeat artifact
exists. After failure, all launcher/worker processes were absent and every GPU
reported no compute process.

## Root cause

The production workers created the default NCCL process group before the
production-sized owner-bridge cache view was materialized. Ranks 1 through 7
entered the pre-model status broadcast at collective sequence 3. Rank 0 stayed
in the full four-cache materialization and never entered that collective.
After 600 seconds, ranks 1 through 7 timed out in `BROADCAST NumelIn=1`; the
later EOF/communicator-abort errors were secondary.

One materialization traverses approximately 32.9 GiB of cache payloads roughly
three times, or about 98.8 GiB of hash/unpickle work. The guard's second full
preflight took about 2277.5 seconds. Raising the NCCL timeout would neither
bound that path nor fix the latent asymmetry in which seven peers repeat the
same work after rank 0 succeeds.

The corrective boundary is therefore:

1. consume and strictly validate the guard's nonce/HEAD/config/cache-bound
   attestation before creating Accelerator or a process group;
2. reconstruct only the typed bounded materialization identity and inspect
   current manifests without a payload-wide scan;
3. publish one durable success/error slot per worker and require exact W8
   filesystem consensus before entering Accelerator;
4. keep later rank-stream chunk loading checksum-validated; and
5. remove the production worker's full materialization and NCCL status
   broadcast from this pre-model boundary.

## Recovery authority boundary

The original singleton claim is consumed. Literal reuse of its nonce,
activation, or admission slots is invalid because they bind exited process
identities and were published with no-replace semantics.

A future activation requires explicit user authorization for exactly one
append-only, parent-linked recovery successor. Such a successor must bind this
claim, the binding-failure receipt, the stderr hash, old and repaired HEADs,
the unchanged production config/source/data identities, a fresh nonce, and a
maximum recovery count of one. A second uncertain or failed recovery must stop
without a third activation.

## Claim boundary

This record establishes a deterministic pre-model orchestration defect and
the absence of model/optimizer progress. It is not training success, model
quality evidence, permission to alter the singleton claim, or permission to
launch again.
