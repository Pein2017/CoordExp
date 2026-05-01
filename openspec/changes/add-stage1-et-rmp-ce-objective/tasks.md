# Tasks

## 1. Governance

- [x] Add OpenSpec delta for Stage-1 ET-RMP-CE objective semantics.
- [x] Add superpower design and implementation plan artifacts.
- [x] Validate with `openspec validate add-stage1-et-rmp-ce-objective --strict`.

## 2. Entry Trie

- [x] Add red tests for desc divergence, coord divergence, shared-coordinate
      unique path, later-coordinate divergence, object-uniform probabilities,
      duplicate serialized-entry multiplicity, and teacher-forced path steps.
- [x] Implement pure entry-trie helpers and target-step dataclasses.
- [x] Verify pure trie tests pass.

## 3. Full-Suffix Objective

- [x] Add red tests for full-suffix rendering, recursive remaining-set update,
      boundary tokens outside trie MP, final close/EOS ordinary CE, and
      full-suffix CE without trie MP.
- [x] Implement full-suffix row builder and loss computation.
- [x] Add smart-batched full-suffix row scoring with serial parity tests.

## 4. Config, Trainer, Metrics

- [x] Add strict objective config under `custom.stage1_set_continuation`.
- [x] Branch trainer `compute_loss` to full-suffix batch scoring for
      `full_suffix_ce` and `entry_trie_rmp_ce`.
- [x] Add compact emitted ET-RMP metrics.
- [x] Preserve existing candidate-balanced default behavior and tests.

## 5. Profile And Docs

- [x] Fold the ET-RMP-CE experiment profile into
      `configs/stage1/set_continuation/production.yaml`.
- [x] Update Stage-1 objective docs.
- [x] Update metrics docs.
- [x] Run focused pytest and OpenSpec validation.
