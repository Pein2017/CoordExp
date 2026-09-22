# Lane A candidate: recurrence distribution census

Status: `candidate`; root owns scientific acceptance.

The frozen processed length-12000 source had 122,218 records. After excluding 525 prior split-qualified identities, 121,693 records remained; deterministic seed 19 selected 128 new images (122 train and 6 val). The identity audit resolved image paths and SHA-256 values and found no aliases or conflicts. The existing mature source contributes 145 unique images and 580 outputs across tied/untied original and normalized policies. The new natural source contributes 256 original-policy outputs, 128 per model.

The new cohort had 886 complete rows and 9 literal exact repeat rows for tied-original, and 860 complete rows and 5 literal exact repeat rows for untied-original. The corresponding <=8-bin same-description repeat-row counts were 38 and 35. Exact and near pair edges, invalid geometry, malformed rows, EOS/cap and endpoint occupancy are retained in `new-census.json`; category exposure denominators and image-unit rare-category bootstrap intervals are stored for both mature and new strata.

The final shared mechanism panel has 45 boundaries: 25 carried accepted numerical-feedback boundaries and 20 deterministic prospective additions. It contains 21 failure boundaries and 24 non-recurrent proxies. The prospective candidate pool had 10 failure candidates and 210 proxy candidates; 10 of each were selected by the frozen metadata round-robin rule. The exact executed selector source and hash are bound in `shared-panel.json`. The panel is a fixed mechanism sample and is not a census of all recurrence.

The two native producers completed 32 groups each with no invalid group receipts. Tied and untied qualification passed with maximum selected-row reconstruction errors `5.72e-6` and `7.63e-6`; no-op tokens, coefficient identity and temporary delta restoration checks passed. The run recorded 8,971 model forwards, 70 vision forwards, 1,131.02 GPU-seconds (`0.314 GPU-hours`) and 22,181,553 runtime bytes. Both worker processes exited with `status=complete`; no owned processes remain.

CPU acceptance:

```bash
PYTHONPATH=. python probes/training_set_completion/recurrence_census/accept.py
```

Key artifacts: `shared-sources.json`, `selection-rule.json`, `prelaunch-panel.json`, `eligible-manifest.json`, `mature-census.json`, `new-census.json`, `shared-panel.json`, `finalization-receipt.json`, and `result.json` under the lane output root. The standalone producer and reducers are under `probes/training_set_completion/recurrence_census/`.

The detailed scientific synthesis, including image-unit uncertainty, category
exposure denominators, spatial/size/onset and proxy associations, scheduling
deviation and the read-only replay command, is in `results.md` and
`scientific-synthesis.json`. The lane remains `candidate` pending root
acceptance.
