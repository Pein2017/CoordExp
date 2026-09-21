# Mature geo_sorted_xy HF/vLLM systematic diagnosis

Frozen question: from the mature `four-coordinate-xy/step-2444` dynamic-HF
anchor, does changing only (a) BF16 component materialization or (b) the runtime
from HF to vLLM change detection behavior on the frozen 32-row cohort, and can
an available precision/composition intervention reduce that change?

- Anchor: dynamic HF BF16 with the mature DoRA adapter and paired selected-token
  embedding delta from
  `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`.
- Input: the existing 32-row, 296-GT cohort with SHA256
  `2e97810c11fc3935b96196767378d1ab68a7571f555919ee0e5960240d764969`.
- Preserved contract: exact image plan, prompt/template/tokenizer, BF16 model
  input policy, greedy decoding, repetition penalty 1.1, max 512 new tokens,
  parser/scorer, and evaluator. No row may be filtered asymmetrically.
- Primary contrast 1: the already authenticated BF16 dense snapshot in HF
  versus that exact snapshot in vLLM. This isolates the engine/runtime after
  materialization; the production qualification gate is bypassed only inside a
  clearly labeled diagnostic runner and no admission receipt is written.
- Primary contrast 2: dynamic HF BF16 versus an FP32 dense materialization in
  HF on the same cohort. This tests whether higher-precision materialization is
  a practical escape from the BF16 operation-order loss; it does not authorize
  a production precision change.
- Primary evidence: terminal and artifact validity, per-row raw generations,
  parser/drop/truncation counters, prediction-set agreement, and diagnostic
  COCO AP/AP50/AP75. Fixed-prefix logits are secondary mechanism evidence.
- Strong alternatives: prompt/media/config drift, nondeterministic tie-breaking,
  vLLM multimodal preprocessing drift, score-channel differences, or the small
  cohort itself could masquerade as a backend effect.
- Attempt bound: at most one 32-row decode per new arm (BF16 vLLM and FP32 HF),
  one materialization per required dtype, and one bounded fixed-prefix follow-up
  only if those arms do not separate the leading hypotheses. One GPU per arm;
  no training, no full validation set, no code change, no qualification/admission
  mutation.
- Stop rule: stop once materialization loss and engine loss are separately
  measured on valid matched artifacts, or when a concrete runtime/identity
  failure makes one contrast technically invalid. A technical-invalid arm is
  not a scientific null.

This diagnostic does not change the OpenSpec acceptance threshold or authorize
archival. The mature dynamic-HF result remains the semantic reference.
