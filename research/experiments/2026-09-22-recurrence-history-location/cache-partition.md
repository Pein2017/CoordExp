# Stage B: source-token versus downstream historical state

2026-09-22. Prospective continuation under the user's autonomous-until-convergence
grant. [Stage A](unit.md) is independently accepted in
[/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/lead-checks/stage-A-readback.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/lead-checks/stage-A-readback.json).
Its fixed selection rule chooses row86: its d difference fromL87 is2.082852,
above the predeclared0.282976 material threshold. This is not selection of the
largest older response. Native, nearL87 and olderL86 full vectors are frozen.

From these same histories, does the near-versus-older response difference survive
when only the edited-token cache, only downstream historical cache, or both are
retained, while recomputing the entire identical current-row prefix?

## Exact partition

Retain Stage A checkpoint, effective rows/deltas, image, batch4/target2,
FP32/SDPA and actual MRoPE/causal inputs. The edit remains x2 38 to640.
Native history ends before S at raw792; physical prefix width2112. Current
six-token S is raw792:798, physical2112:2118. Near source row87 x2 is raw789 /
physical2109; older source row86 x2 is raw780 / physical2100.

Prefill three actual histories: nativeN, nearE87, olderE86. At all28 decoder
layers, save the real post-RoPE K and V for each history. For each edited history,
partition the target row's historical positions into:

- P: before the source token, always native;
- D: the single edited source token, physical p;
- H: subsequent historical tokens [p+1,2112), ending before S.

Companion rows stay native. D has one token; H has2 tokens forL87 and11 forL86.
That difference is part of the historical-propagation partition, not a calibrated
per-token comparison. The source row's y2/end tokens are inside H.

For each location score the four corners N=(nativeD,nativeH),
D=(editedD,nativeH), H=(nativeD,editedH), F=(editedD,editedH).
N is shared. Use each full edited prefill for its F baseline; causal equality of
P and companions must hold before mixing. Always recompute all six current S
tokens; therefore the D route includes mediation within current S. It is not
restricted to one direct attention edge at the final query.

Use installed DynamicCache layers' keys/values via direct model forward. Feed
all four rows' actual six-token tails, full2D attention mask width2118,
explicit3-axis positions at2112:2118 and cache_position2112:2118. Do not call
prepare_inputs_for_generation, which clears explicit positions. No image
payload is needed in the S-only forward. The final returned logit predicts798;
the first returned logit does not predict792.

## Qualification and evidence

Before any hybrid interpretation, cached N,F87,F86 must match the accepted
Stage A R,L87,L86 full-vocabulary vectors within2e-4 and exact global argmax.
Use three prefill calls and seven S calls (N,F87,F86,D87,H87,D86,H86):
10 model forwards and3 vision forwards planned. No extra qualification replay
is authorized merely for metadata. Retain full logits and FP64 absolute
coordinate probabilities for every S endpoint.

Verify exact pre-source/companion K/V equality between paired prefills. Check
all layers' partition membership and actual cache lengths, hashes and source
identities. Baseline source caches must stay immutable, or each altered cache
must be cropped to2112 and restored exactly in finally after an S score. Cache
mutation of inference tensors must occur under torch.inference_mode. Persist
proof of target-only changes, unchanged P/companions, consumed S/positions/masks,
and restoration or immutable-source identity. Do not retain full cache dumps
merely for archive completeness; hashes, bounded slices and executable checks
are sufficient for this declared partition.

A CPU sensitivity check must reject wrong target/source boundaries, source/H
overlap, or changed companion/pre-source values, and must verify no-op source
restoration or immutable baseline behavior. One baseline plus one edited cache
is about3.61GiB; process near and older sequentially and avoid unnecessary cache
copies. Measure actual peak reserved memory. Cap14 model forwards and15 minutes
model execution; any concrete failure returns to root with evidence before a
repair/relaunch. Preserve attempts and all cost.

## Frozen interpretation and convergence

Primary d=z38-z999. For each location report E_F=d_F-d_N, E_D=d_D-d_N,
E_H=d_H-d_N and interaction I=d_F-d_D-d_H+d_N. Define the near-minus-older
contrasts Delta_F=d_F87-d_F86, and likewise Delta_D,Delta_H. The identity
Delta_F=Delta_D+Delta_H+(I87-I86) is algebra, not a causal fraction explained.
Report absolute P38/P999 and the true global argmax/gap separately.

Use0.001 as the numerical difference guard. For the selected location contrast,
preservation means same sign asDelta_F and |Delta_path-Delta_F| within
max(0.01,0.1*|Delta_F|); negligible means |Delta_path| no larger than that same
tolerance. These are practical interpretation thresholds, not statistical
confidence intervals. Partial, reversed and interacting patterns remain visible.

- If D preserves the difference and H is negligible, rereading source-token
  state is sufficient to carry this location contrast in the hybrid context.
- If H preserves it and D is negligible, downstream historical states can
  carry it after the changed token's direct cached contribution is removed.
- If both contribute or neither preserves it, report distributed/interacting
  dependence. Do not force a single-path explanation or begin a layer/head scan.

These are retained-state causal tests using surgical mixed caches. Sufficiency
in a hybrid is not exclusive natural mediation or a unique circuit. No outcome
establishes physical-owner recovery, image-scale generality, a fitted temporal
kernel, an adaptive accumulator, or the cause of the natural long-run exit.

After qualified Stage A and this single partition, converge on the supported
computational boundary and remaining ambiguity. A further launch requires a
specific decision-changing contradiction in these checks/results; merely finding
another nonzero effect is insufficient. Root continues through technical repair
and acceptance without another user approval. Child stops at the stable candidate.

Producer: probes/training_set_completion/recurrence_history_cache_partition.py.
Artifacts: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/cache-partition-001/.
