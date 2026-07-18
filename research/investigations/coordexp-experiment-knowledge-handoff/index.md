# CoordExp Experiment Knowledge Handoff

This package is a curated import of the historical experiment handoff. It is a
navigation and provenance layer, not a new scientific authority. Read the
[intake audit and current routes](14_intake_audit_and_current_routes.md) before
using a historical result.

## Authority and Scope

- Current research belief remains owned by the active `research/decisions/`
  files on the current research branch.
- Runtime, evaluation, artifact, and compatibility contracts remain owned by
  `docs/` and `openspec/` on their current canonical branch.
- This package compresses historical and cross-branch research evidence. Its
  documents are `non_normative_research` and must not silently promote a
  hypothesis, checkpoint, or intervention into a current decision.
- The source synthesis is commit `93deed826` in the
  `codex/historical-markdown-recovery` branch. Its claim and source ledgers
  were imported byte-for-byte from commit `1c4d79774`.
- The 326-megabyte raw snapshot unions were deliberately not copied into this
  worktree. They remain source provenance in commits `12467ca8e` and
  `816fbb0a1`; the imported ledgers retain their paths and content hashes.

## Minimal Terminology

- **Coordinate basin**: a competing local token sequence that can become the
  emitted box after the first coordinate choice.
- **Coverage-like state**: a history-conditioned signal that changes an
  available-versus-duplicate readout or continuation; the term does not imply
  a durable object ledger.
- **Causal route**: an intervention site and source definition whose change is
  tested against controls and a declared output, rather than an attention
  correlation alone.
- **Current owner**: the branch and path whose live document or contract should
  be read before proposing a new experiment.

The following declarations apply to every document in this package:

- **Qwen3 Vision-Language (`Qwen3-VL`)**: the pretrained multimodal model family
  used by the historical experiments.
- **cross-entropy (`CE`)** and **supervised fine-tuning (`SFT`)**: token-level
  training loss and the training phase that applies it.
- **Stage 1**: teacher-forced supervised fine-tuning. **Stage 2**: historical
  rollout-conditioned post-training; old `stage2_ab` names are provenance only.
- **Ranked Probability Score (`RPS`)**: an ordered-distribution loss computed
  from cumulative coordinate-bin probabilities. The historical name
  `gaussian_rps` means Gaussian coordinate soft targets plus this score.
- **repetition penalty (`RP`)**: the decode-time logit penalty recorded in
  historical run names such as `rp=1.10`; it is not a training mechanism.
- **Average Precision (`AP`)**, **mean Average Precision (`mAP`)**, and
  **F1 score**: historical detection metrics whose evaluator and denominator
  must be checked before comparison. `AP50` and `AP75` mean AP at 0.50 and 0.75
  Intersection over Union thresholds.
- **Intersection over Union (`IoU`)**: overlap between two regions. **Mean
  Absolute Error (`MAE`)**: average absolute coordinate-bin error.
- **Oracle-K**: a diagnostic that checks whether any of K sampled rollouts
  contains a recoverable object; it is not a deployable policy.
- **Low-Rank Adaptation (`LoRA`)** and **Weight-Decomposed Low-Rank Adaptation
  (`DoRA`)**: parameter-efficient adapter families named by historical runs.
- **multilayer perceptron (`MLP`)**, **query/key (`Q/K`)**, **Kullback-Leibler
  divergence (`KL`)**, **language-model output head (`LM head`)**, and
  **end-of-sequence (`EOS`)**: model components or measurements used by the
  mechanism capsules.
- **ground truth (`GT`)**, **false negative (`FN`)**, and **unlikelihood (`UL`)**:
  reference annotation, missed annotated object, and a historical anti-output
  objective respectively.
- **JavaScript Object Notation (`JSON`)**, **JSON Lines (`JSONL`)**,
  **command-line interface (`CLI`)**, and **parameter-efficient fine-tuning
  (`PEFT`)**: historical data, execution, and adapter terms.
- **Wasserstein-1 distance (`W1`)**, **95-percent Gaussian radius (`R95`)**, and
  **cumulative distribution function (`CDF`)**: historical coordinate-loss
  terms. `top_p` is the cumulative probability threshold used by nucleus
  sampling.
- **effective batch size (`EBS`)**: the total examples contributing to one
  optimizer update. **bounding box (`bbox`)** and `xyxy` mean a rectangle
  represented by left x, top y, right x, and bottom y coordinates.
- **Common Objects in Context (`COCO`)** and **Large Vocabulary Instance
  Segmentation (`LVIS`)**: datasets or proxy label surfaces named by historical
  evaluations.
- `L17`, `L18`, and similar labels mean decoder layer 17, decoder layer 18, and
  so on; `L17H1` means decoder layer 17, attention head 1. `A5`, `A6`, `v7`,
  `v9`, and `H1` through `H5` are source-local legacy arm or hypothesis labels;
  they have no portable meaning outside their owning record.
- Claim identifiers use plain namespaces: `BIND` for binding, `COORD` for
  coordinate objectives, `COV` for coverage-like state, `DUP` for duplication,
  `GRPS` for the Gaussian-plus-Ranked-Probability-Score round, `INFER` for
  inference/evaluation lessons, `PFX` for prefix history, `S2` for Stage 2,
  `TRAIN` for training/runtime lessons, `RESULT` for metric records, and
  `LEGACY` for design lineage. The numeric suffix is only a stable row identity.

## Preflight Before Any New Probe

1. State the exact question, target token/slot, checkpoint, prompt/template,
   decode policy, and artifact root.
2. Read the matching row in
   [01 cross-branch landscape](01_cross_branch_landscape.md).
3. Check the owner and claim handles in the evidence atlas before designing a
   nominally identical arm.
4. Treat a prior null, inert patch, or failed safety gate as a no-repeat
   constraint unless the new arm changes the declared discriminator.
5. Preserve validity, duplicate, prediction-count, closure, and stop counters;
   do not compare free decode, teacher forcing, and official evaluation as one
   metric.
6. Record whether the outcome confirms, narrows, retires, or leaves unchanged
   the mapped hypothesis and add the next seed to the owning investigation.

## Reading Order

Read this router and the [intake audit](14_intake_audit_and_current_routes.md)
first, then [01 cross-branch landscape](01_cross_branch_landscape.md) to select
one route. Read only the relevant synthesis (`02`–`12`), and use
[13 evidence atlas](13_evidence_atlas.md) plus [claims.tsv](claims.tsv) when a
claim or artifact needs verification. Do not read the raw recovery snapshots
as a default path.

## Package Documents

- [01 Cross-Branch Landscape](01_cross_branch_landscape.md)
- [02 Coverage-Ledger Mechanism](02_coverage_ledger_mechanism.md)
- [03 Gaussian Coordinate Soft-Target and Ranked Probability Score Round](03_gaussian_rps_mechanistic_round.md)
- [04 Autoregressive Duplication Capsule](04_autoregressive_duplication_capsule.md)
- [05 Binding and Formation Capsule](05_binding_and_formation_capsule.md)
- [06 Prefix History and Route Capsule](06_prefix_history_and_route_capsule.md)
- [07 Stage-2 Rollout Failure Registry](07_stage2_rollout_failure_registry.md)
- [08 Coordinate Objective and Decode Negatives](08_coordinate_objective_and_decode_negatives.md)
- [09 Training and Runtime Lessons](09_training_and_runtime_lessons.md)
- [10 Inference, Evaluation, and Artifact Lessons](10_inference_evaluation_and_artifact_lessons.md)
- [11 Historical Result Registry](11_historical_result_registry.md)
- [12 Legacy Design Lineage](12_legacy_design_lineage.md)
- [13 Evidence Atlas](13_evidence_atlas.md)
- [14 Intake Audit and Current Routes](14_intake_audit_and_current_routes.md)
- [Source coverage ledger](source_coverage.tsv)
- [Claim ledger](claims.tsv)
