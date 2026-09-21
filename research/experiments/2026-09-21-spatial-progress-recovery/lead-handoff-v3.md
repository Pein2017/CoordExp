# Candidate handoff: spatial progress recovery v3

Status: `candidate`; no worker self-acceptance.

Stable candidate:

- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-recovery/candidate-manifest-recovery-v3.json`
- SHA256 `45190d71e428904142a0f3eeb3a056d2ddcbd773bbc635a01ae4607d9901645b`
- Reduction SHA256 `4bd9de08ea6a1780d3dc601eb4e6ba1775c17b85640af74f5f788539aee6e32f`
- Orchestration record SHA256 `5f265d44155fcdb57299852a09ed2788c6ab63dc9eb13536780c8626636004a2`

Execution completed the frozen denominator: 11 source trajectories, one admitted
failure boundary, one admitted matched control, ten failure HOLDs, eight of eight
primary cells analyzable. Qualification passed with full-vocabulary log-probability
maximum error `7.3909759521484375e-06` under tolerance `2e-4`. There were no new
failed model attempts and no release cells.

For fixed crossed `N-annotation--128`, moving historical x1 from 788 (after N=786)
to 784 (before N) changed:

- failure: full row `+0.1013593421`; x1 `+0.0529394150`; y1 conditioned on the
  candidate x1 `+0.0349893570`; joint corner `+0.0879287720`. Full-row y2 shams
  were `-0.0014541260` and `-0.0037408608`; joint-corner shams were
  `-0.0015916824` and `-0.0001173019`.
- matched control: full row `+0.1288194019`; x1 `+0.0432040691`; conditional y1
  `+0.0682530403`; joint corner `+0.1114571095`. Full-row y2 shams were
  `+0.0002572859` and `+0.0139621463`; joint-corner shams were
  `+0.0075066090` and `+0.0138299465`.

The predicted directional ordering effect is present and materially exceeds both
shams in each admitted boundary. Its comparable, slightly larger matched-control
effect supports a local general ordering-sensitive policy and does not identify a
failure-specific burst cause. Natural reference error, actual owner reselection,
physical recovery, and broader H1 remain unresolved; no secondary release was
authorized.

The reused execution owner `/root/spatial_gate` was `gpt-5.6-luna/max`, used no
descendants, and is terminal. Supplied hints were kept separate from worker
coaching and Luna fixes in `orchestration-notes.md`. Luna fixed the parser-offset
token interface, predecessor/recovery root separation, real-entry selfcheck,
reducer wall binding, fixed candidate selection, crossing effects, and vision
accounting. The worker independently rechecked the current committed refactor:
54 candidate bindings (32 unique files), 126 source original/capture pairs, a
fresh byte-identical reducer replay, caller selfcheck, candidate regeneration,
Python compilation, `git diff --check`, and both-root output-layout validation.
All nine recovery attempts and all owned jobs are terminal; no model process is
live.

Continuation cost: 69 model and 69 vision forwards, `494.9969075322151`
GPU-seconds, 44,065,258 retained bytes. Cumulative historical plus continuation:
236 model and 236 vision forwards, `1028.4273607879877` GPU-seconds
(`0.2856742669` GPU-hours). Saved wall elapsed is `6071.743703842163` seconds
under the 7200-second limit.

Acceptance commands:

```sh
python -m probes.training_set_completion.spatial_progress_gate.selfcheck
python -m probes.training_set_completion.spatial_progress_gate.reduce \
  --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-recovery \
  --out <new-absent-path>
python -m probes.training_set_completion.spatial_progress_gate.integrate_candidate --check
python -B -m src.artifacts.output_layout \
  --root /data/CoordExp/outputs \
  --root /data/CoordExp/.worktrees/research-probes/outputs
```

Changed owned paths are the four maintained Lane A runtime/reducer/check/integrator
files plus this recovery unit's `orchestration-notes.md` and `lead-handoff-v3.md`.
Lead-owned state, protocol, catalog, frontier, questions, accepted predecessor,
and Lane B were not modified by this recovery.
