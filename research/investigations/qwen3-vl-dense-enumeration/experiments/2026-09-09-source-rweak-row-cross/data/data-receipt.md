# CPU data preparation receipt

Status: candidate; root owns freeze and acceptance. No GPU allocation, model load,
training, confirmation512 read, annotation edit, or candidate implementation read.

Manifest:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/data-v1/manifest.json`

SHA256: `af3fd69e05bea3c148c293e528a2ea738ac49e45631bb0b71a94d7eb56d3fcd8`.

## Verified population and selection

All original 512 image records and 3759 unique GT annotation IDs retained before
selection. Original category-consistent global cardinality-first maximum-IoU
matcher replay gives:

| Arm | IoU50 | IoU60 | IoU80 |
| --- | ---: | ---: | ---: |
| Source | 2225 | 2055 | 1534 |
| Rweak64 | 2310 | 2102 | 1540 |

| Stratum | Population | Eligible | Selected |
| --- | ---: | ---: | ---: |
| Any Source owner loss | 71 | 69 | 16 |
| Gains with no Source loss | 73 | 71 | 8 |
| Identical owner sets, changed tokens | 343 | 343 | 8 |

Exclusions: 25 identical-owner-set and identical-token images; 4 illegal first
divergent rows; 451 eligible cases beyond their stratum's deterministic cap.
All 480 exclusions are individually recorded. No cross-stratum refill.
Illegal-action exclusions: `coco2017_val_000000130826`,
`coco2017_val_000000172547`, `coco2017_val_000000238410`,
`coco2017_val_000000484351`. Suffix parser drops/caps are not exclusions.

The selected panel has no divergent EOS action; EOS was not categorically
excluded. A focused synthetic terminal-EOS test verifies eligibility and rejects
reopening after EOS. Complete-row legality uses the original parser, requiring
exactly one accepted prediction, no parser drops, and exact span coverage.

Qualification IDs, in deterministic rule order:

1. `coco2017_val_000000211674`: prefix 0 tokens; Source/Rweak action 9/9.
2. `coco2017_val_000000466256`: prefix 20 tokens; actions 10/10.
3. `coco2017_val_000000322574`: prefix 0 tokens; actions 10/11.
4. `coco2017_val_000000279887`: prefix 10 tokens; actions 10/10.

Rule: EOS first by selection hash if available; then shortest prefix, longest
prefix, longest single action, resolving ties by selection hash. Deduplicate,
then fill ascending selection hash to at most four. No new outcomes used.

## Exact schema and runtime bindings

Top level: `schema`, `sources`, `policy`, `selection`,
`qualification_case_ids`, `cases`; schema value `row_cross_manifest_v1`.

Case keys: `row_id`, `stratum`, `selection_sha256`, `row_index`, `input_record`,
`image_path`, `image_width`, `image_height`, `image_plan`, `gt`,
`common_prefix_token_ids`, `common_prefix_text`, `actions`, `diagonals`,
`remaining_token_budgets`, `continuation_token_budgets`.

`actions.source/rweak`: `kind` (`row` or `eos`), `token_ids`, `text`,
`start_token_index`, `end_token_index` (exclusive).

`diagonals.source/rweak`: exact `generated_token_ids`, original `raw_record`,
`matched_gt_ids` keyed `iou_0.50`, `iou_0.60`, `iou_0.80`, and `stop_reason`.

Both budget dictionaries are keyed by **donor action arm**, not recipient.
`remaining_token_budgets` is always 3084 minus prefix/action length;
`continuation_token_budgets` is zero for EOS and otherwise that remainder.

`sources.source/rweak` includes original run roots, artifact hashes, full resolved
config, persisted resolved YAML binding, historical entry YAML path/hash,
model/tokenizer identity, runtime settings, prompt/processor fingerprints.
`sources` also binds exact base tensor/tokenizer files, adapter/embedding files,
Rweak checkpoint and bank identity, original loader/matcher/parser source files.

Original entry YAML files under the provider checkout were removed after the old
run. This is explicitly `historical_entry_yaml.available=false`; their historical
SHA256 remains available in resolved-config provenance. The persisted original
run's `configs/resolved.yaml` and `configs/resolved.json` agree exactly and are
both hash-bound. `original_yaml` points to that persisted resolved YAML, whose
top level wraps the runtime config under `config`.

Cold-load bindings use the read-only provider checkout
`/data/CoordExp/.worktrees/coco-gt-correction-portfolio`, not later A16 code/config.
Source opener: `src.inference.hf_backend.open_hf_backend_session`.
Rweak opener: `scripts.research.eval_coco_owner_focus.session_opener`, with exact
checkpoint path, bank path, arm `Rweak`, completed update 64 in the manifest.
Rweak retains Source selected additive embeddings. Original observed parameters
are all float32; SDPA, patch-embed linearization enabled, no image resize, left
padding, RP1.0, temperature0/top_p1, native im_end, cap3084.

## Verification and reproduction

Run from `/data/CoordExp/.worktrees/self-rollout-behavior`:

```bash
python -m unittest discover -s research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-09-source-rweak-row-cross/data -p 'test_prepare.py' -v
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-09-source-rweak-row-cross/data/prepare.py --verify
```

Preparation reconstructs original generated IDs using only `generated_token`
trace records (not `selected_token_replay`), sorted by generated step, excludes
only pad tokens, and validates contiguous steps, terminal stop semantics, exact
full decoded text and individual token text. It replays the original parser on
all 1024 diagonal outputs and matches complete prediction/drop records, verifies
input/GT/image/dimension identities, image-content hashes, paired image plans,
and exact common-prefix/action token identities and budgets. Historical capped
and malformed suffixes remain as original evidence. Full model files are hashed
but never loaded; only the tokenizer is CPU-loaded.

`--verify` independently reconstructs the entire manifest from original sources
and compares exact bytes to the frozen artifact. Initial creation is exclusive
(`open('x')`), preventing accidental overwrite.

Focused tests reject corrupted forced tokens, trace tokens, row identity, GT
denominator, and budgets. The direct-row B-versus-C terminal counterexample uses
the original global matcher: replacing B with C loses B and gains C with an empty
unchanged suffix and equal owner count. This proves why suffix-only accounting
cannot replace complete-output primary owner sets.

No unresolved data blocker. This receipt does not establish GPU implementation
qualification or scientific validity of any new cross continuation.
