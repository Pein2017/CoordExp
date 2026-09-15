# CoordExp-Swift Wave 1A Source Study Plan

## Purpose

Wave 1A is the read-only source-study kickoff for the CoordExp-swift rebuild.
It turns the most fragile external and legacy assumptions into auditable
evidence before any production `src/` implementation starts.

This plan is subordinate to:

- `docs/superpowers/plans/2026-06-30-coordexp-swift-execution-charter.md`
- `docs/superpowers/plans/2026-06-30-coordexp-swift-src-rebuild-roadmap.md`
- `openspec/changes/rebuild-coordexp-swift-training-infra/tasks.md`
- `openspec/changes/rebuild-coordexp-swift-training-infra/specs/**/spec.md`
- `docs/architecture/proposals/2026-06-27-coordexp-swift/DECISIONS.md`
- `docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md`

## Scope

Allowed in Wave 1A:

- read local repo files, legacy `src/`, tests, docs, OpenSpec, and source-study
  references;
- inspect installed package source for Transformers, PEFT, Accelerate,
  DeepSpeed, flash-attn, and MS-Swift if available;
- use CodeGraph, Serena, and raw shell for read-only
  exploration;
- perform lightweight package/config/tokenizer/processor introspection that
  does not mutate repo files or launch training;
- synthesize findings into source-study docs after review.

Not allowed in Wave 1A:

- moving or archiving old `src/`;
- writing production `src/`;
- creating probe scripts under `scripts/probes/coordexp_swift/`;
- marking OpenSpec implementation tasks complete;
- running GPU-heavy probes or smoke runs;
- choosing a final DoRA or special-token embedding mechanism when evidence
  leaves multiple plausible options.

## Source-Study Outputs

Wave 1A prepares the evidence needed to create these durable docs:

```text
docs/architecture/proposals/2026-06-27-coordexp-swift/source-studies/
  dora.md
  special-token-embeddings.md
  qwen-noresize-mrope-fa2.md
  legacy-invariant-inventory.md
```

If the directory does not exist when synthesis begins, create it then. Do not
create empty study files before evidence exists.

## Lanes

### Lane A: DoRA / Adapter Targets

OpenSpec tasks covered: 2.1, 2.2, and planning evidence for 2.3.

Questions:

- Does `adapter.type: dora` honestly map to PEFT DoRA/`use_dora`?
- Should `adapter.type: dlora` be rejected as a legacy/provisional spelling?
- How do PEFT and MS-Swift expose DoRA/LoRA target discovery?
- What trainable parameters and checkpoint payloads must a round-trip probe
  verify, especially DoRA magnitude-vector parameters?
- Does the public config name require any user approval before Wave 1B probes?

Required evidence:

- package versions;
- local package file paths and symbols;
- legacy CoordExp adapter/config references if any;
- exact probe assertions for Wave 1B.

Stop gate:

- If DoRA cannot be defined as PEFT `use_dora=True` without inventing
  semantics, stop and ask whether to use standard LoRA for a pre-smoke or keep
  DoRA-first blocked.

### Lane B: Special-Token Embedding Mechanism

OpenSpec tasks covered: 2.4 and 2.5.

Questions:

- How do custom Qwen wrappers, PEFT `TrainableTokens`, and LoRA
  `trainable_token_indices` differ for selected full embedding training?
- Is the local Qwen3-VL input embedding tied to the output head?
- Should compact payloads store additive deltas or absolute selected values?
- How should base-plus-adapter-plus-embedding-delta loading be verified?

Required evidence:

- local model/tokenizer identity and selected token ids;
- package source references for candidate mechanisms;
- tied/untied behavior evidence;
- exact round-trip probe assertions for Wave 1B.

Stop gate:

- If tied-head or checkpoint semantics remain ambiguous, stop and ask before
  implementation.

### Lane C: Qwen No-Resize, MRoPE, And FlashAttention

OpenSpec tasks covered: 2.6, 2.7, and 2.8.

Questions:

- What processor class and `patch_size`, `merge_size`, and
  `temporal_patch_size` does the local model use?
- What no-resize image dimensions are admissible before processor reshape?
- How is `image_grid_thw` produced and how does it determine visual-token
  counts?
- What does installed Qwen3-VL expect for position-id row shape and row meaning?
- How must packed training differ from whole-row upstream helper inference?
- What evidence proves the FlashAttention varlen path is actually used?

Required evidence:

- installed Transformers file paths and symbols;
- local model/processor config facts;
- MRoPE reset requirements;
- FA2 attention implementation and dtype constraints;
- exact probe assertions for Wave 1B.

Stop gate:

- If installed Qwen behavior contradicts OpenSpec, patch OpenSpec before
  implementation.

### Lane D: Legacy Correctness-Invariant Inventory

OpenSpec task covered: 2.9.

Questions:

- Which legacy tests/modules protect invariants that should be ported into the
  new test suite?
- Which invariants are already expressed in OpenSpec, and which remain weak or
  absent?
- Which new test names or fixture checks should carry those invariants forward?

Required evidence:

- concrete legacy file/test references;
- grouped invariants by rebuild wave;
- proposed new tests or fixture checks;
- gaps that should become P0/P1/P2 findings.

Stop gate:

- Do not archive old `src/` until this inventory exists and Wave 2 can point to
  it.

## Review Gate

After the four lanes report:

1. synthesize findings into the four source-study docs;
2. request an independent review of the study docs;
3. patch accepted P0/P1 issues;
4. only then mark OpenSpec source-study tasks complete where evidence exists.

Wave 1B probe scripting begins only after Wave 1A study docs and review are
accepted.
