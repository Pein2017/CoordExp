# S K10-H20 Natural Crossover Launch Gate

Status: `CONSUMED AND CLOSED — the one authorized preGPU-v5 launch executed exactly once; execution-v4/evidence-v4 are complete; no further launch authority exists`
Updated: 2026-08-07

The gate history below is retained in full as immutable technical lineage. The
consumption record at the end of this file supersedes every earlier status
line; no earlier line confers current authority.

Historical status at the moment of authorization: `OPEN FOR ONE FIXED NO-TRAINING LAUNCH — preGPU-v5 sealed and independently accepted; only logs-v6 → execution-v4/evidence-v4 on GPUs 0/1/7`

The preGPU-v1 receipt is a historical technical seal, not current launch
authority. Independent Sol fixed-tree review rejected launch because its test
binding omitted `tests/research/test_natural_boundary_attention_actuators.py`
and these owner documents still described runner/finalizer integration as
pending after integration had completed. The same review also rejected the
finalizer's failure to hold tau null for unqualified events. This owned
successor repair binds the fifth test and corrects the status. The finalizer
repair now holds source-unqualified tau and utility null. PreGPU-v2 seals that
repaired fixed tree and passes prelaunch validation. Independent Sol re-review
accepted that exact prelaunch tree, but runtime subsequently rejected all three
shards before model load because the runner recomputed the sealed `h0_root`
directory inventory in a different order.

The CPU-only root cause is bounded: the sealer sorts final inventory entries by
their `relative_path` strings, while the preGPU-v2 runner hashed entries in
`sorted(Path.rglob(...))` order. For the same 109 files and bytes, the sealer
bound `00d8cd190c...` and the runner observed `8788bff4f...`. Ordinary nohup
`logs-v1` files were zero bytes, so their exact process progress is not
recoverable; the deterministic pre-model gate and absent execution root prove
only that they contain no scientific result. The tmux `logs-v2` files are
three 97-byte pre-model technical failures under receipt-v2's reserved
`execution-v1`; neither `execution-v1` nor `evidence-v1` was created.

The minimal successor sorts the runner inventory by `relative_path` before
canonical hashing. The prefix-collision regression is red on the old runner
and green on the successor; runner and sealer now agree on `h0_root`, the
operative S `h0_dir`, and the base-model directory. Independent Opus/max review
confirmed the cause is unique, the fix is operator/estimand-neutral, and the
regression is discriminating. PreGPU-v3 seals the corrected tree and fresh
CPU probe with raw SHA-256 `85ff6bf2a9da48274c1992b8a85488be6be83ce0694f9d8bb98c529412e74175`
and self SHA-256 `1d4db08bd8c45a63345e2b82c678e58de73d23b8d387b17e3ba75854f6f0e8b2`.
Its launch mapping is exclusively `logs-v3` to `execution-v2/evidence-v2`.
Independent Sol re-review returned PASS with P0/P1/P2 all zero. The subsequent
three-shard attempt then failed closed before any endpoint because the runner
compared the plan's normalized event identity against fields absent from the
full manifest event (`prefix_sha256` is nested under `natural_boundary`) and
would next have compared the slim plan projection to the full execution
payload. The three logs are 84/85/85-byte technical failures; `execution-v2`
and `evidence-v2` remain absent. This is not model evidence.

Fable P0 review isolated one contract mismatch rather than an operator or
estimand defect: the plan intentionally stores a nine-field selection
projection, while the admitted manifest stores the full execution event. The
accepted repair recomputes the full event and nested geometry self-hashes,
resolves the nested natural prefix, geometry, target owner, and ordered
covered owners, compares only the canonical projection to the plan, and then
retains the full event for execution. The frozen source-qualification triple
remains plan authority and is not re-inferred. Fable also required the exact
source preflight to cross the live executor's `_validate_inputs` census seam
while never calling `_load_once`. A downstream P0 then tightened that boundary:
`_validate_inputs` alone does not cover the model-free production seams before
the first forward. The successor therefore also runs `_load_cpu_contract`, the
complete 11-event `_preflight_full_runtime_cohort`, and the exact seeded
ledger-history/natural-context/attention-factory/K14-geometry path for all
three selected events. It materializes K01, K10, H20, and C11 CPU masks while
retaining consumption as explicitly required and unattested; no preflight may
claim live all-layer consumption or no-op parity. The processor-only adapter
must retain `model=None` and `session=None`, and `_load_once` remains forbidden.
The v4 successor exposes this as a receipt-independent preseal operation via
`run_s_k10_h20_crossover_shard.py --preseal-source-preflight`. It consumes the
exact plan and ten source paths plus the newly reserved execution/final roots,
emits canonical JSON to stdout, binds the current conclusion-critical code and
tests, and contains no pre-GPU-receipt identity or placeholder. This ordering
breaks the receipt/evidence hash cycle: the sealer can bind and validate the
preseal evidence, while exact receipt-bound per-shard source preflights remain
post-seal acceptance evidence rather than receipt bootstrap evidence.
Plan-v1 remains immutable and this repair alone is not launch authority; a v4
receipt, post-seal three-shard source-preflight acceptance, and independent
fixed-tree review are still required.

The preceding v4 description is historical. PreGPU-v4 was subsequently sealed
with raw SHA-256
`93672aa1983da7b55865c9b505dac1c22dfe1f36e3bb9a316ef741e128ff8f67` and
semantic self SHA-256
`62eaafc1a453df96a4031c09447c85d70b1aa9bbd20ca2cbc1280168297917d7`. Its
exclusive mapping was `logs-v5` to `execution-v3/evidence-v3`. The `logs-v4`
files were zero-byte transport artifacts; `logs-v5` are the actual attempt.
All three `logs-v5` shards failed identically after model load with
`GateTechnicalInvalid: S gate K10 scalar release failed: C11 callback requires scalar input_ids`.
The immutable, consumed `execution-v3` has only per-shard `failure.json` and
`failure.stderr`, no scientific endpoint or result, and therefore cannot be
reinterpreted as model evidence. The mapped `evidence-v3` result was not
produced.

The user-authorized v5 successor is operator-neutral and fixes only C11
keyword gate dispatch plus exact construction/runtime consumption-receipt
semantics across the producer, preseal, runner, and finalizer. The bound
five-suite CPU run passed 129 tests; the installed-Qwen probe and all three
receipt-bound shard preflights passed without loading a model or creating an
execution root. Independent Opus/max repair review and final receipt-bound
review both returned PASS with zero P0 and zero P1.

PreGPU-v5 is sealed with raw SHA-256
`7e4adff6e272dfaad8ad5bb5656c9fc0baf7f2992d3561e37edfb613991e57ec` and
semantic self SHA-256
`c349258554a2e308c5a4242157eb59cbb6793cdc7476c2a3a65719b845ae4571`.
The final reviewer explicitly accepted exactly one frozen no-training launch:
`shard-000→GPU 0`, `shard-001→GPU 1`, and `shard-002→GPU 7`. The exclusive
mapping is `logs-v6` to `execution-v4/evidence-v4`; all three roots were absent
at acceptance. No reorder, reselection, sweep, A3, P4, training, production
behavior, or promotion is authorized.

## Pre-GPU gate

The gate is closed until all of the following are true:

1. the materializer recomputes the v3 K10∩H20 rule and obtains exactly
   `gt:2299:29`, `gt:13348:14`, `gt:16228:15` in that order;
2. the plan binds the complete source manifest, census-v3, original execution
   plan, gate result, event hashes, natural-prefix hashes, and geometry hashes;
3. the four-cell contract is exactly `C00,C10,C01,C11`, with C00's explicit
   K01-style no-op and hidden K00 parity control;
4. the fresh runner, shared attention/residual actuators, base gate/live/
   natural runner, finalizer, materializer, sealer, and all conclusion-
   critical tests are regular files at their exact expected paths and their
   hashes are sealed;
5. focused CPU evidence, the installed-Qwen composition/consumption probe,
   the 11-event processor-only cohort preflight, and all three selected
   production factory contexts are canonical, passing, CPU-only, and
   model-free;
6. runtime versions match the installed interpreter/packages, and physical
   devices are exactly `shard-000→0`, `shard-001→1`, `shard-002→7`; and
7. at seal time, the fresh execution and final roots are distinct, absent, and
   non-symlink paths.

If any referenced implementation or test file is absent or drifts, the sealer
fails cleanly and confers no launch authority. The preGPU-v1 through v4 trees
are historical and cannot be relaunched. PreGPU-v5 is the sole current launch
authority and is consumed by exactly one successful or failure-only attempt
under its exclusive fresh roots.

## Runtime and interpretation boundary

The receipt binds every current manifest/census/execution-plan/config/panel/
cohort/cohort-manifest/H0/base-model path and hash, plus the complete H0 and
base-model inventories. Each shard derives a runtime identity with the exact
receipt path/raw/self hashes, current code hashes (including the live executor),
runtime versions, and its `shard-NNN`/`CUDA_VISIBLE_DEVICES` assignment. A
valid, independently accepted receipt would authorize one fixed three-shard
no-training launch only. The
runner must preserve `pre_opener_natural`, `opener_injected=false`,
`opener_generated_by_model=true` if and only if the first generated token is
the opener; a native STOP or invalid first token must set `row_started=false`
and remain a scientific outcome, not technical invalidity merely because no
opener was generated. It must also preserve
`use_cache=false`, three rows, and 256 row tokens, with no reorder,
reselection, sweep, A3, P4, old `no_2x2`, training, or production behavior.
Every cell emits a full endpoint vector. Componentwise tau and charged utility
remain descriptive. If C00/C10/C01/C11 do not all have qualified
source-specific endpoints, the crossover is explicitly unqualified; unmatched
valid outcomes are retained. No result from this gate authorizes architecture,
objective, training, decoder, wrapper, token, or checkpoint promotion.

## Gate consumption record

The authorized launch executed exactly once. `shard-000 → GPU 0`,
`shard-001 → GPU 1`, `shard-002 → GPU 7`, under the exclusive `logs-v6` to
`execution-v4/evidence-v4` mapping. All three shards completed. PreGPU-v5,
`execution-v4`, and `evidence-v4` are consumed; the gate confers no further
authority and cannot be reopened by renaming or reusing a root.

Consumed identities:

- preGPU-v5 receipt raw SHA-256
  `7e4adff6e272dfaad8ad5bb5656c9fc0baf7f2992d3561e37edfb613991e57ec`,
  self SHA-256
  `c349258554a2e308c5a4242157eb59cbb6793cdc7476c2a3a65719b845ae4571`;
- post-execution finalization successor receipt raw SHA-256
  `71b305dab1b25f838ba65433902cafe91e5aaa26ed80fbab792f607b1257c15e`,
  self SHA-256
  `c0244fb8e35e5fa4e277972085e5dc273ac6a7cda539d96ee64ce966fe5ee3ed`;
- formal evidence raw SHA-256
  `29407f7cddd632999e3f720e9982fc4a52a31eea5398181f2f9d07dd28947d90`,
  self SHA-256
  `1dfa4144d094aa06f35c1cc3fb7f29b2db10efc55367f65b96c7244695d990f3`;
- formal evidence receipt raw SHA-256
  `9fc2fbb155bd3158f26d9085072d74e5c48b3c6dfe7496ebb4774296f359c398`,
  self SHA-256
  `db5649f7a9628f19b0445511d6e138be31f8dc8c04567080a81de0f230d08a85`;
- formal root
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-07-s-k10-h20-natural-crossover/evidence-v4`.

The `logs-v1` through `logs-v5` attempts and the consumed `execution-v3`
failure root remain immutable technical history with no scientific endpoint.

### Executed provenance versus sealed pre-GPU identity

Executed-GPU provenance for this launch comes from `logs-v6`, the per-shard
`result.json`, `terminal_summary.json`, and `aggregate.receipt.json`. Each
`execution-v4/shard-NNN/runtime_identity.json` additionally embeds the sealed
CPU-only preGPU-v5 runtime identity, whose `runtime` block carries
`gpu_used: false` and `model_loaded: false`. That block is a pre-launch
contract, not an attestation about the executed run, and must never be read as
executed-GPU provenance. The seam is documented; no artifact is rewritten.

### Post-consumption disposition

The result is artifact-valid with `source_specific_crossover_status =
unqualified` and formal tau and utilities null, not zero. No retry,
re-selection, repair launch, new crossover, or sweep is authorized. Training is
`HOLD`, A3 is `DO NOT RUN`, P4 is `DO NOT RUN`. The scientific result is owned
by [results.md](results.md) and the review ledger by [review.md](review.md),
where all four reviews are complete: Fable preseal and postseal `PASS`, the
formal scientific audit `PASS` on evidence integrity, and the Fable
post-remediation re-review `PASS`. Documentation is **certified**; the formal
scientific disposition is unchanged.
