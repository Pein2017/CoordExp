# Lane B launch milestone

Transform and admission schema is frozen in `unit.md`. Source/panel dependency:
mature panel
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/panel.json` plus the accepted numerical-feedback selection
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/selection.json`.
The mechanical qualification cases are the exact `tied-417044-failure` and
`untied-417044-failure` records from `refined-03`: source row 11/prefix 119
tokens for tied and source row 16/prefix 169 tokens for untied. Each prefix
hash is checked against the selection record before model work. Final
scientific cells wait for Lane A frozen panel and consume it unchanged.

Geometry: for each source `(W,H)`, lossless RGB copy into `(W+256,H)` canvas, fixed black fill, shift `d=128` px = four processor grid cells. Centered scene/history is `00` at x=128; negative/positive cells use x=0/256. History boxes use exact canvas pixel map and round-half-even to bins; inverse drift and token counts are checked. Seven cells: `00`, `10-`, `10+`, `01-`, `01+`, `11-`, `11+`; mismatch cells diagnostic.

Native seam: build original prompt with `build_bound_native_requests`, replace only image path/size/hash in `NativeRequest`, prepare each transformed image via `prepare_native_inputs`, then pass transformed literal history as `extensions` to `generate_continuations` with original RP1 and 512-token budget. Every cell gets a fresh `prepare_native_inputs` call; no cache transplant. Centered `00` must retain valid rows and the frozen failure/proxy predicate or the state is admission-HOLD. A mapped known-bank match is retained as an optional grounding witness; zero witnesses limit physical interpretation but do not discard numerical recurrence. Forced description/wrapper x1 logits are a separate full-vocabulary capture at the same row boundary.

The earlier rows0-4 pilot is preserved under
`preselection-pilot-rows0-4/`; it is selection-mismatched and is not a
scientific result. The corrected tied/untied `00` calls are the first admitted
qualification evidence for this launch packet. Their checked prefix hashes are
`93030c774efa0d694fa3b4f5c49ea1d4dffbe159d888bb3b2cb86aab4bee9fcf` (tied)
and `d0b07937c8a9ec44c376b68b2bb336af6ac00626a93946ef4d0630377841a481`
(untied). Both `00` states retained recurrence under the corrected parser and
the six remaining cells were then run for each model. This is retained pilot
qualification only: 14 pilot cells, 4,256 model forwards and 42 vision
forwards; prior invalid-bound or
parser receipts remain preserved under `uncapped-prebound-pilot/` and
`parser-preflush-pilot/`.

The producer and CPU reducer consume image, group, model, panel and source
bindings from each transform manifest at the existing native seam; the
417044/refined-03 values are only the frozen qualification manifest, not a
runtime donut selector. The retained pilot receipts are `runtime-result-tied.json`,
`runtime-result-tied-transforms.json`, `runtime-result-untied.json`, and
`runtime-result-untied-transforms.json`, with matching CPU reductions. The
Lane A has now frozen `shared-panel.json` with 45 states (21 failures and 24
proxies). The CPU state-entry receipt is `generic-entry-readiness.json` with
45/45 resolved, zero binding errors, and zero model calls. Final model work is
pending parent device release; no new model call has been made since this
pilot and no owned process remains.
