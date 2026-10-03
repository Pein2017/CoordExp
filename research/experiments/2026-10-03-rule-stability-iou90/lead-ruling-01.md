# CPU consumer accepted within scope; native release held for two repairs

The lead reviewed candidate `9c1b3d11d04026dd93133f0d5f4912a0be62aa1d`.
The saved CPU consumer boundary is **lead-accepted within its synthetic scope**:
18 image identities, 570 distinct annotation identities, versions 0/1,
per-image versus aggregate counts, transition list/count agreement, arm-specific
baseline ledgers and independent B-minus-A arithmetic agree. No acquisition,
model call or checkpoint readback was repeated. Evidence:
`outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/lead-consumer-check-01.json`.

A focused read-only review found no concrete blocker in unequal-rank SUM/update,
checkpoint/optimizer export, fresh-load sequencing or the differentiable median
transform. Actual CUDA/NCCL, model allocation, forward/backward, numeric replay
differences and behavioral reload remain native qualification targets. Historical
CPU lifecycle source identities remain historical; they are not resealed as the
final source. The recorded tiny-CPU-fixture exception remains an exception; exact
collection excludes both real-model fixture tests. Three unrelated existing
`_RopeOwner` fixture failures remain disclosed, without a full-suite pass claim.

Two reproducible full-support cases block release:

1. Native acquisition leaves `allow_pad_tokens=False`. An actual generated
   suffix `[151643, 151645]` contains PAD followed by EOS and is rejected after
   sampling, although the frozen policy permits both actions. Preserve generated
   PAD actions and their likelihoods through the maintained opt-in caller seam;
   do not mask, remove or reinterpret them as EOS.
2. An actual suffix `[151655, 151645]` contains an image-token action. Whole-history
   replay incorrectly treats it as another input image in position derivation
   and placeholder/feature matching. Bind multimodal handling to the original
   prompt media positions while preserving original action IDs, likelihoods,
   gradient flow and cached-generation semantics. Do not invent media or narrow
   policy support.

The worker owns these repairs and affected CPU falsifications. The lead owns the
narrow replay design and exact subsequent release. Preserve completed evidence;
return a corrected source identity and fresh unreleased proposals. No native
model/GPU work is released by this ruling, and no scientific result is accepted.

The replay repair keeps upstream forward/scatter/DeepStack computation and original
actions intact. Explicit prompt modality types plus text-only suffix types govern
positions; a scoped placeholder-mask adapter calls upstream validation on the
original prompt and extends its masks with false over the suffix, restoring the
method on exit. The qualification proposal will add one rank-0/image-1584,
version-0 technical cached-generation/replay pair on six forced special actions
(PAD, image, video, vision start/end and EOS), with raw and unforced-normalized
conditional scores retained before action forcing. It supplies no training loss
or scientific metric. This is one extra six-action request and one diagnostic
replay, making 73 qualification requests and at most 222,054 generated actions.
Its purpose is to exercise the repaired real-model path directly; it is not a
warmup or a policy change in either primary arm. The exact release is still pending.
