# Permanent owner bridge one-time recovery successor acceptance

Date: 2026-08-11 UTC

## Decision

The one-time, parent-linked production recovery mechanism required by OpenSpec
task 4.13a is mechanically accepted. The user authorized exactly one recovery
activation. This record accepts the implementation and its fixed-target
lifecycle audits; it does **not** say that the canonical successor has been
reserved or that production training has launched.

The original production claim and all parent evidence remain immutable. The
only permitted activation is attempt ordinal `1` of `1`, using the fixed
`stage1-production-recovery-attempt-1-claim.json`. If reservation, payload
completion, activation, run binding, or the first finite/applied heartbeat is
failed or uncertain, all evidence is retained and execution stops. There is no
attempt 2 and no third activation.

## Parent authority

The accepted parent launch identity is:

- intent key `02bdbc993c57f4e3b7f38506d40d80af9acbc2ab327a1330ee1e86789078d3ad`;
- original claim SHA-256
  `0b597d82b97855b44742b959d5e864149d9f68070701cc8ce2665e15ad82db73`;
- original claim receipt fingerprint
  `ca087ebb4d98472188d84b4bab0cb5dcb86e14b59290f470d9246b407a20eeae`;
- original repository HEAD `7f399a2f04bbea615e66c42531381c10bfb42fdb`;
- production config fingerprint
  `770f705f5357fdfc08d1d6fb332d828874265d296c5f567ae4c64266cd3c056c`;
- failed binding SHA-256
  `cda4cd4b970d12697a98965cd0ea66f27291a93b548344334a70f6aa218dbc47`;
- failed binding receipt fingerprint
  `2505ff27e404bffbd54b8a33262a9b86e8f8bb6a146310ec45540d8db52ed92f`;
- terminal error `training.owner_bridge_launch_guard_process_exited` with a
  nonzero return code.

The fixed implementation also binds the original intent, activation,
preflight, all eight worker admissions, stdout, and stderr by prescribed
lexical path and exact SHA-256. No parent binding or parent-nonce `run.json`
exists, and the exact parent launcher PID/start-time/boot identity is no longer
live.

## Implemented recovery boundary

- The fixed successor is reserved with `O_EXCL`, followed by parent-directory
  `fsync`, before time, nonce, or payload derivation. It is never unlinked.
- Empty, partial, malformed, symlinked, or already occupied successor paths are
  terminal and reject before nonce generation or process creation.
- The child intent identity binds current HEAD, production config fingerprint,
  parent claim SHA/receipt, parent intent key, recovery policy, and ordinal 1.
- The original v1 claim and worker-admission schemas remain unchanged. Recovery
  uses distinct claim and admission policies/schemas.
- Workers select exactly one claim from the closed original/successor set by
  nonce. Recovery admission records the selected fixed path and raw SHA-256;
  pre-model quorum v2 records and reaches consensus on the same path/SHA.
- Parent evidence and child intents are read as stable regular files through
  component-wise `openat` traversal with `O_NOFOLLOW`. Same-ledger relocation,
  final symlinks, intermediate-directory symlinks, nonregular files, and
  oversized records fail closed.
- Parent liveness is evidence-positive: only `ENOENT` or `ESRCH` proves process
  absence. Permission, malformed `/proc`, and other inspection failures reject.
- Attempt-scoped intent, activation, admissions, pre-model slots, logs,
  binding, and failure paths are checked for residue before reservation.
- The production pipeline retains the static ordering proof that durable
  `run.json` publication precedes its sole model/optimizer construction path.

## Verification receipts

Final production-file SHA-256 values audited by both independent reviewers:

```text
11af0f06d119c9d6628c44f4e59e45e636bfb9e215ba6afdd02eefe09d2ff1b1  src/common/launch_identity.py
782a5ab398b8965ac44b462f4c234f1b7dc5ca63662562af7cbfdbf637f7f8f0  src/training/owner_bridge_launch_guard.py
e95c7609490642b77a3ab9395318ae8422eaffb8b086a9fbcf28ecacab146b54  src/training/owner_bridge_pre_model.py
2317cac34087d7f0b7c72c46561c2c1fe1d044d26dce13e24fd76b525bf8badb  src/training/pipeline.py
0b4d2ec77d3fc4998e21c07634f76a60c6dd679a74d472d92f1f354d02a46fc8  scripts/coordexp_swift/launch_owner_bridge_stage1.py
```

Executed gates on the final fixed source packet:

- recovery/launch identity, guard, pipeline, and pre-model suites: `140 passed`;
- canonical pytest scope: `2093 passed`;
- Ruff check and format check: pass;
- Python compile check: pass;
- strict OpenSpec validation: pass;
- `git diff --check`: pass.

Independent fixed-target verdicts:

- Sol xhigh lifecycle/security audit: PASS, no P0/P1;
- Opus xhigh recovery-contract audit: PASS, no P0/P1.

Both reviewers independently reproduced the formerly failing liveness,
dangling-symlink, valid-target-symlink, same-ledger relocation, intermediate
symlink, v1/v2 admission, claim-reservation, and no-third boundaries before
issuing their final verdicts.

## Bounded operational notes

The largest immutable parent identity file is the 15,453,735-byte intent. The
parent-evidence reader has a 16 MiB bound; all fixed parent files were reread
successfully and cannot grow without violating their SHA anchors. Child intents
use a separate 32 MiB bound. At acceptance time no component above or within
the canonical ledger root was a symlink; this is rechecked operationally before
the one allowed activation.

## Claim boundary and stop rule

This acceptance proves launch-control mechanics against temporary ledgers and
the exact frozen source hashes. It does not prove a canonical successor exists,
that GPUs are safe, that Accelerate has launched, that eight ranks have bound,
or that any optimizer step has completed. It also does not add model-quality,
generalization, or benchmark claims.

Before activation, the lead must commit this implementation, verify a clean
tree, re-run the exact production preflight, recheck all parent hashes and
recovery surfaces, confirm cache/config/source/GPU identity, and then invoke the
guard once with `--recovery-attempt 1 --execute`. Any failure or uncertainty
after reservation is terminal; preserve evidence and stop without a third
activation.
