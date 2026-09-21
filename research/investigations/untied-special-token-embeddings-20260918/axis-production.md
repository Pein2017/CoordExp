# Untied head + axis validity: production launch

## Current attempt: packing performance repair

The original preparation was deliberately interrupted with SIGINT under explicit
user authorization after ~78 minutes. Its traceback confirmed execution inside
`cache_workflow._materialize_pack_plan`, recomputing the full-plan digest for each
pack. No full cache was published and training had not begun. Original logs and
receipts remain under the first launch packet; its terminal monitor was cancelled.

The current packet is `/data/CoordExp/outputs/infra_base/untie-axis-20260918-restart1/`
and runs in the same tmux session, `coordexp-untied-axis-20260918`.
Its config is byte-identical to the first packet. Changes: compute the plan digest
once; index the full encoded corpus once; expose encode/plan/supervision/assembly/
publication timing and bounded progress counters. Keep independent 64-byte digest
strings because pickle memoization is part of the frozen payload contract.

240 tests pass. New regression tests fail on the old repeated-work paths. Fresh
real smoke caches (32 train packs, 2 eval packs) have identical canonical payload
bytes to the previous successful eight-GPU training caches. Old-cache comparisons
are offline and checksum-verified; production admission still requires current
source fingerprints. The original model/loss/checkpoint acceptance remains valid.
No packing policy, sample order, loss, model, LR or epoch changes were made.

Restart was chosen to remove the confirmed serial bottleneck and avoid its repeat
in validation. The unobservable remaining loop index prevents a precise old-run
ETA or counterfactual speedup claim; record actual restart phase durations from
`cache-preparation.log`. No in-memory partial preparation could be reused.


User-authorized objective: CE + 0.01 * raw_axis_validity_hinge, untied selected-token
input/output deltas; all baseline data, ordering, epochs, LRs, packing and prompt
unchanged. Margin 1/999; mean axes, mean boxes per segment, mean supervised
segments per global update. No-complete-box segments contribute zero. Gate weight
remains zero. Base remains Qwen3-VL-2B-Instruct-coordexp-natural-adjacent.

## Acceptance

- 232 tests passed; source sign-flip mutation rejected by the gradient/formula test.
- Real eight-rank two-step run completed, including forward eval and two saves.
- Axis raw/weighted losses were finite and nonzero on all four train/eval rows.
- Peak rank-zero CUDA allocation: 10,512,198,656 bytes (~9.8 GiB).
- Fresh resume: 590 inference tensors and model/optimizer state for all eight
  ranks match uninterrupted training bitwise.
- Fresh native HF load: both FP32 [1004,2048] delta tensors match checkpoint
  exactly; independent storage, both nonzero/distinct; perturbing each affects
  forward and restoring it restores logits bitwise.
- Evidence: `/data/CoordExp/outputs/infra_base/untie-20260918/axis-acceptance.json`,
  `axis-integration-tests.log`, `axis-mutation.log`, `axis-eight-gpu.log`,
  `axis-resume.log`, `axis-hf-reload-exact.json`.
- Scope: mechanism/distributed/persistence acceptance, not a model-quality result.
  The small weighted hinge (~1.7–1.9e-6 in training smoke) is deliberately unchanged;
  it constrains teacher-forced expectations, not guaranteed greedy geometry.

## Durable launch

- tmux: `coordexp-untied-axis-20260918`
- cwd: `/data/CoordExp/.worktrees/coordexp-infras`
- authored config: `configs/train/geo_sorted_xy/untied_axis.yaml`
- frozen packet: `/data/CoordExp/outputs/infra_base/untie-axis-20260918/`
- script: packet `launch.sh`, under Conda `ms`, bare Python inside environment.
- pipeline: full cache preparation -> require-all-hit admission -> eight-rank
  torchrun; fail closed with terminal status/exit code on any failure.
- source hashes checked before preparation and again before training; full source
  and config snapshot plus tracked patch preserved in packet.
- run output:
  `/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch`
- checkpoint every 0.4 of schedule plus final, same as baseline; exact same-world-
  size model/optimizer/RNG state enabled from the first save.
- Eight A100 80GB ranks, EBS24, max length12000, four epochs, no extra model forwards.
- Full train117266/eval4952 rows; single cache preparation owner and worker_count1.
  Prepared immutable cache is verified once before distributed startup. No automatic
  relaunch loop and no quality-dependent change of the training budget.

The tmux launch is evidence of a running pipeline; consult packet `terminal.log`
and `training.log` to distinguish cache preparation, GPU training and completion.

## Production completed (2026-09-18 14:21:34 UTC)

Exit code 0; run.json status completed, 2444 optimizer updates. All 2447 train/eval
rows report finite status. Checkpoints: step-978, step-1956, step-2444; final and
best (acc_top1) both point to step-2444. Final train total loss 1.40948935; final
eval total loss 1.43690237. This is training/forward-eval completion, not a claim
about autoregressive malformed-box rate or detection quality.

Cache preparation and standalone admission took 48m48s; torchrun entry to clean
exit took 7h05m58s including startup, evaluation and checkpoint publication.
The terminal monitor d9bdd0e2-a4e3-4d48-a3b7-f89ce59a5b2e delivered completion.
Final fresh-HF reload receipt is in the restart1 packet at final-hf-reload.json.
