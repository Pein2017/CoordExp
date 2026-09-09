# Source/Rweak row crossing

The reducer is an offline consumer of the frozen selected-panel manifest and saved
cross outputs. It preserves complete-output global assignment, category handling,
strict later-repeat counting, and direct/matching-ambiguous versus pure-tail
attribution. It does not establish causal shares or native deployment quality.
Scientific owner: `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-09-source-rweak-row-cross/readout.md`
(originally in the preserved self-rollout tree).

From the repository root, replay the four saved engineering cases without any
sibling-worktree Python imports or original provider filesystem access:

```bash
python -m probes.source_rweak_row_cross.reduce \
  --manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/data-v1/manifest.json \
  --cross-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/engineering/eng_beta/source-cross-b4-v1 \
  --cross-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/engineering/eng_beta/rweak-cross-b4-v1 \
  --qualification-case-ids coco2017_val_000000211674,coco2017_val_000000322574,coco2017_val_000000279887,coco2017_val_000000466256 \
  --original-code-root /data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/coco-original-code \
  --output-dir /tmp/row-cross-replay
```

The output directory must not already contain `reduction.json`. This command uses
a locally cached tokenizer, no model forwards. Its result is explicitly a
`qualification_subset` (4 cases, 16 complete outputs), not final 32-case acceptance.
The original manifest is never rewritten. `--original-code-root` maps its frozen
COCO-relative source paths to preserved historical bytes; every original SHA256
is checked. Those files are never imported. The output's `execution` field records
the current reducer path/hash separately from verified historical source bindings.
Omitting this option checks the manifest's original paths.

Portable checks:

```bash
python -m pytest -q probes/source_rweak_row_cross/tests/test_reduce.py tests/eval/test_assignment.py
```

Saved-input checks additionally set `ROW_CROSS_MANIFEST` to the manifest above and
`ROW_CROSS_ORIGINAL_CODE_ROOT` to the preserved code root, then run
`probes/source_rweak_row_cross/tests/test_reduce_saved.py`. Without an explicit
manifest these integration checks skip; portable scientific counterexamples still run.

## Preparation and native execution

The frozen manifest contains the original Source/Rweak model, adapter, embedding,
prompt and image settings. Batch size, output path and wall time are CLI arguments;
there is no additional configuration format.

Reconstruct the panel from the original 512-image evidence and verify identical
manifest bytes (cached tokenizer only, no model):

```bash
python -m probes.source_rweak_row_cross.prepare \
  --manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/data-v1/manifest.json \
  --original-code-root /data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/coco-original-code
```

Plan one qualification case without loading a model:

```bash
python -m probes.source_rweak_row_cross.run \
  --manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/data-v1/manifest.json \
  --original-code-root /data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/coco-original-code \
  --recipient source --mode qualify --case-ids coco2017_val_000000322574 \
  --batch-size 1 --device cuda --max-wall-seconds 300 \
  --output-dir /tmp/row-cross-native-plan
```

For an authorized model run, add `--execute` and select a fresh output directory.
`--mode cross` swaps the donor action; `--recipient` selects the model generating
the suffix. The whole trajectory cap remains 3084 and each suffix keeps its own
remaining budget. `--compare-to` checks complete tokens, parser output and owner
identity against a prior B1 output directory.

The runner loads Qwen components once, attaches the declared payloads, and calls
public native preparation/generation. Rweak must validate as completed update 64
with the original bank and recipe. `receipt.json` separates current `execution`
and observed `native_execution` from historical source bindings; it does not
create an HF session receipt for this untraced route. The reducer consumes the
preserved row schema. An occupied output directory is rejected.

Portable checks exercise the actual CLI, native preparation, a tiny CPU model
through Transformers generation, and the disk parser/matcher:

```bash
python -m pytest -q probes/source_rweak_row_cross/tests/test_run.py
ROW_CROSS_MANIFEST=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/data-v1/manifest.json \
  python -m pytest -q probes/source_rweak_row_cross/tests/test_prepare_saved.py
```

The bounded real Source check passed on `coco2017_val_000000322574`, B1,
FP32/SDPA with patch linearization: all 21 frozen diagonal tokens matched,
including the 11-token suffix. It completed in 11.11 seconds under the
300-second deadline, with 9,361,506,304 peak allocated GPU bytes and
9,698,369,536 host RSS bytes. The receipt is at
`/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/source-native-smoke-67d305d57/receipt.json`.
This is one-case engineering acceptance; it does not establish full-panel or
Rweak model parity or a new scientific result.
