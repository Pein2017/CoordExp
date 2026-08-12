# Wave 2 prelaunch guard-only exit

Status: operational prelaunch event; not a GPU replacement attempt and not a
Wave 2 model or semantic observation.

On 2026-08-09, the shell intended to launch the single Wave 2 replacement
performed these operations under `set -e`, in order:

1. require the fixed GPU receipt target to be absent;
2. query whole-node compute applications;
3. require that query result to be empty;
4. print the GPU inventory;
5. invoke the immutable-plan probe.

The shell exited with code `1` after approximately `0.19 s`, produced no
output, and stopped before the GPU-inventory print that preceded the probe
argv. The fixed target
`/tmp/coordexp-wave2-v2-replacement.wnqm0Z/gpu-receipt.json` remained absent;
no probe process, model load, CUDA allocation, or terminal probe receipt
existed. Each guard was then rerun independently and returned success, with
zero compute applications and all eight GPUs idle. The exact transient cause
was not recoverable; a brief unrelated whole-node process race is possible but
not established.

Two independent contract reviews therefore ruled that the replacement budget
was not consumed: the write-once GPU receipt target is the materialized attempt
ledger, and the probe writes that receipt on every invoked success or failure
path. An absent target plus execution ending before the preceding statement is
evidence that the probe was not invoked.

The authorized replacement procedure is consequently narrowed to:

- a fresh plan created after this ledger entry;
- exact live repo/source/config/dependency identity revalidation in the launch
  process;
- an absent fixed write-once GPU receipt target;
- one explicit `cuda:0` probe invocation with no retry loop;
- the probe's own target-device idle samples and resource ceilings, rather than
  a stale whole-node empty-process guard.

Once the probe argv is invoked, any terminal receipt at its fixed target—pass
or fail—consumes the single replacement authorization. No automatic retry is
permitted.

## Launcher-shadow non-invocation

After the first event was recorded, a second prelaunch shell rebuilt and
byte-compared the authenticated plan, confirmed the fixed target absent, and
printed the intended launch line under `set -x`. That line began with
unqualified `env -u ...`. The shell returned `0` after approximately eleven
seconds of preceding plan work, again with no GPU receipt, Conda/Python output,
process, or CUDA use.

Read-only diagnosis established the exact deterministic cause: this host's
`PATH` resolves `env` first to `/root/.local/bin/env`, a 328-byte shell wrapper
that only ensures `$HOME/.local/bin` is present in `PATH` and never executes its
arguments. `env -u RANK conda --version` therefore emits nothing and returns
`0`, while `/usr/bin/env -u RANK /root/miniconda3/condabin/conda --version`
prints `conda 25.5.1`. The traced line invoked only the shadow wrapper; Conda,
Python, and `run_command` were never entered.

Independent contract review again ruled the replacement unconsumed. The only
authorized correction is transport-level: a plan prepared after this record,
the same fixed plan content and GPU receipt target, absolute `/usr/bin/env` and
absolute Conda paths (or equivalent explicit shell `unset`), one invocation,
and no retry wrapper. Any subsequently entered probe run consumes the
authorization regardless of its terminal result.

## Corrected transport attestation and one-shot authorization

The corrected packet pins `/usr/bin/env` and
`/root/miniconda3/condabin/conda`. A CPU-only transport smoke first printed the
resolved Conda-environment Python path, then invoked the real probe command
against plan SHA-256
`e5c2b1eb0c7ff99b7a66de8dc172f5af0331959d9b5f12a07e61096dd79b4197`
and a throwaway CPU receipt target. It entered `_preflight_cuda` and produced a
strict v2 failure receipt with code `qwen.parity.cuda_device`, requested device
`cpu`, and empty arm/comparison/proof/negative-control/measurement inventories.

- Corrected-packet CPU receipt:
  `/tmp/coordexp-wave2-v2-transport-fixed.snIcui/cpu-fail-receipt.json`
- Receipt SHA-256:
  `c7fd3f3570b12596111387b5c1f783a65ea7ea5ebdd1ba9b7e549afc2384eeb5`
- Plan file SHA-256:
  `aadcfe046938315d90df05633b1da00d5a86e368eee3304b9d822183c1cdb681`

This proves the absolute-path launch transport reaches the probe and preserves
the write-once GPU target. The user had explicitly released the eight GPUs and
authorized continued execution of all remaining OpenSpec waves; independent
internal and Opus 5/max reviews both found the replacement budget unspent and
the corrected packet technically safe. The lead therefore authorizes exactly
one post-ledger GPU invocation using the same plan content, sample, device,
comparison contract, and fixed absent target. Any entered probe terminal
result consumes that authorization; no automatic retry is allowed.
