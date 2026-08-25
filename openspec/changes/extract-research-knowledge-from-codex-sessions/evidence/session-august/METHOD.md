# August session intake method

## Frozen source boundary

- Capture contract: `/data/CoordExp/openspec/changes/extract-research-knowledge-from-codex-sessions/evidence/CONTRACT.md`.
- Input root: `/data/CoordExp/.codex/sessions/2026/08/**/*.jsonl`.
- Capture point: `2026-08-25T12:35:18Z`.
- The frozen input list contains 1,462 files. Four files whose rollout start is `2026-08-25T12:37`–`12:39` were excluded. The two files started earlier but modified after capture were retained because their source sessions existed before the cutoff.
- `manifest.tsv` has one row per frozen source path, preserves the raw path, rollout id, root/child group, date, and cwd/worktree, and contains no post-capture source.

## Classification and deep-read procedure

Visible user/assistant/agent-message text was extracted from each JSONL after removing bootstrap blocks (`recommended_plugins`, AGENTS/environment, skills, permissions, and app-context). Strong research signals (Human13/all-HF, Image2299, OwnerBridge, Qwen3-VL, mechanism/causal/ordering/serialization, greedy/retrieval, K16, and research-unit/claim language) were classified `high`. Indirect artifact, evaluation, checkpoint, geometry, inference, or research-worktree signals were classified `medium`; generic plugin, UI, routing, maintenance, and unrelated developer traffic was `skip`. Raw JSONL remains authoritative; keyword triage is only a reading aid.

Research-relevant rows are `needs-summary` unless the visible record explicitly says planned, unexecuted, proposal-only, discussion-only, held, or no training/model run; those are `unexecuted`. Generic rows are `development-skip`. Root/child identity is retained using `parent_thread_id` when present, otherwise the session id. No byte-identical duplicates were found; `duplicate_key=group:<root id>` is a grouping key, not a duplicate claim.

Correction round: every prior `skip` row was re-audited against its visible task/result text and cwd. Current research worktrees (`research-probes`, `research-probe-infras`, `permutation-bundle-coordinate-noise-pilot`, `image2299-mechanism-microscope`, `owner-commit-binding`, and `permanent-owner-bridge`) were retained as research-relevant; CoordExp-Swift rows were promoted when their Wave/cache/provenance/model/evidence signals affected reproducibility or evidence validity. This reclassified 94 rows to `high` and 440 to `medium`; 38 generic tool/plugin/UI/maintenance rows remain skipped. Image2299 identity overrides incidental Human13 wording, and OwnerBridge V1/source-preserving HOLD is owned by `/data/CoordExp/.worktrees/permanent-owner-bridge`, while the owner-commit-binding successor proposal remains separately owned by `/data/CoordExp/.worktrees/owner-commit-binding`.

## Counts and verification

- Rows: 1,462; unique source paths: 1,462.
- Relevance: 872 high, 552 medium, 38 skip.
- Disposition: 642 needs-summary, 782 unexecuted, 38 development-skip.
- Sampled 12 high rows, 12 medium rows, and 12 skip rows by replaying their raw JSONL paths; the samples are listed below. High/medium samples were deep-read for identity, lifecycle, and evidence boundary; skip samples were checked for generic-only content.

High sample ids: `019fbe7b-932a-7e41-97b6-14f3df0d527b`, `019fbe7b-c627-7b20-8e8b-8d28347565ae`, `019fbe7e-4db8-7273-8187-4b3e4cb73cea`, `019fbea3-8003-7fb0-aaa6-a2ce4617b067`, `019fbe7f-4d22-7363-bcbd-9818cb57c66f`, `019fbe8d-6c48-7513-b305-66db058b39c8`, `019fbea9-a7bd-7d51-9953-31278f0432f1`, `019fbe8c-1261-7f32-8b35-fdf5d536716c`, `019fbea4-f986-7642-94cc-b68e581d35e1`, `019fbea8-12e9-75d0-a024-ac7986d40bb5`, `019fbebf-ea48-7470-8abb-89951372b047`, `019fbebe-130a-72d0-a802-0553b1651c07`.

Medium sample ids: `019fbf0e-47ff-7db1-bad0-7f40776c364e`, `019fc311-0bf9-7ef1-91de-2d19c819f662`, `019fc41b-b83e-7652-9940-105d2c9ee865`, `019fc896-2748-7ba2-bd9b-a64cfea2e4d0`, `019fca7a-aad0-7b30-a153-baab11538207`, `019fd103-2bf6-7690-9ce4-ae8c506b83a1`, `019fd303-216d-7913-8269-54a0b1ff121e`, `019fd303-75db-77b3-9ecb-8e7f252fe0ed`, `019fd327-658a-7183-b1be-1a3212cd12aa`, `019fd4a6-a376-7240-8461-e916db7fb0bc`, `019fd4bd-ce60-7af3-9bbb-9fa28eb30dc1`, `019fd4e6-0369-71c0-b749-58d8d27c780d`.

Skip sample ids: `019fc2e1-1622-7482-a03c-49df4f12f2e1`, `019fc2e1-d349-73f3-9ad1-36ccf0be7a45`, `019fc1f6-97f7-7d80-a3e6-9cc33241c911`, `019fc2e7-5355-7b31-b6fc-21cdeecf335a`, `019fe045-7cf0-7b10-8cb1-733a0b5dddde`, `019fe1ed-578f-7b32-aeff-92a18a7fe5c0`, `019fe1ed-8035-7700-8997-61544f86b793`, `019fe23d-d341-7820-af62-daa14efd4dd7`, `019fe256-41a7-7700-a899-ec58e5cf980c`, `019fe557-bb08-7c53-85f5-7c9213e03b60`, `019fe57a-8e14-7f31-8902-5611743d79e9`, `019fe9cd-3d61-7970-ba9e-3b25d6423ead`.

## Reproduction

The manifest was produced by a deterministic shell/`jq` pass over the frozen list, extracting the first `session_meta` record and visible message text, then emitting the contract header and one TSV row per path. The exact temporary inputs were `/tmp/aug-files.txt`, `/tmp/aug_meta.tsv`, and `/tmp/aug_signal.tsv`; they are not evidence artifacts. No network, Notion write, experiment, GPU, install, deletion, or repository mutation was performed.
