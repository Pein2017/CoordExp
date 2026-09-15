# Research root collapse: source recovery and retirement

## Scope and authority

User-approved 2026-09-15 knowledge restructuring in the registered `research-probes` checkout, `/data/CoordExp/.worktrees/research-probes`, branch `research-probes`, baseline `f39066f6948c799827d706eb88876b931112ade8`. The user chose one flat research root for the whole topic, retirement of OKF/type-based and topic-wrapper layers, explicit idea distillation, and removal of the old compatibility alias after consumer verification.

This is historical migration provenance, not current research authorization. Read [the research entry](../../../../research/index.md) and [conventions](../../../../research/CONVENTIONS.md) for current knowledge. No GPU work, scientific experiment, training/evaluation resumption, model/artifact-root rename, Git publication or durable Project Memory write is part of this migration.

## Preserved sources

[manifest.json](manifest.json) binds 119 pre-edit research files and six directly affected implementation/router files. [consumer-sources.json](consumer-sources.json) binds eleven additional directly affected consumer sources. Each source snapshot records its original repository-relative path, exact SHA-256, byte length and baseline Git identity. Original bytes are stored under `sources/<original-path>`; they are not rewritten when active links change.

The prior [2026-09-15 capture](../2026-09-15/manifest.json) remains immutable and is bound by its own file hash in this capture. It contains 674 source-version entries. Three of its archived operational-pilot files had already been removed by the baseline commit before this task. They are individually named in `preexisting_retirements`, with verified `f39066f...^:<path>` recovery specs and SHA-256. They were not restored, silently ignored, or attributed to this migration. The checker verifies both the exact deletion commit and the recovered Git blob.

Together the manifests account for 810 source versions: 807 materialized snapshots and three pre-existing Git-recoverable retirements. A source version is not a distinct scientific experiment. [path-map.json](path-map.json) records active moves versus frozen snapshots. [distillation.jsonl](distillation.jsonl) classifies every pre-edit research file's reading/retirement role and its new knowledge destination. These maps do not claim every old sentence was promoted into a current finding.

## Knowledge extraction

The new [story](../../../../research/story.md) restores the June/July ancestry. [Visual designation and causal use](../../../../research/questions/visual-designation-and-causal-use.md) separates painted information, exact replay, endogenous synthesis, held-out representation and actual behavioral consumption. [History](../../../../research/questions/history-repetition-stopping.md) preserves the tested progression from coordinate write/read through source-swap and selection-aligned controls to order-conditioned transitions. [Greedy compilation](../../../../research/questions/greedy-compilation.md) retains the missing matched denoising-OFF question without inheriting an obsolete compact-checkpoint diagnosis.

[Alternatives](../../../../research/alternatives.md) retains genuinely unresolved sparks: matched denoising, structured spatial rendering, geometric-transform transfer, specialization composition, prospective pre-onset signals and matched real-object versus synthetic coordinate guidance, alongside later unresolved directions. Already tested anti-copy, replay, layer, commit and calibration ideas feed question pages and the [complete catalog](../../../../research/experiments/catalog.jsonl), not a parallel idea backlog. Related later work is not automatically the exact control an old idea lacked.

All legacy records were inventoried and byte-preserved. Scientific extraction read the relevant synthesis, result, control, interpretation and next-discriminator sections; source tables, historical receipts and reviewer prose remain available in full. This was not a new raw-artifact or visual re-adjudication of every scientific claim. Operational lessons are bounded examples, not current harness rankings or revived agent prompts.

## Removed active surfaces

The old `research/ideas`, `research/decisions`, `research/mechanisms`, `research/archive`, `research/qwen3-vl-dense-enumeration` and consumed September-9 root handoff are retired. The latter's original bytes remain here, but it is not an active transfer. The old decision-graph checker is preserved as source and retired instead of reporting a misleading zero-node pass.

The `research/investigations` alias is removed after the focused consumer gate. Current maintained readers use explicit frozen record paths. Existing output roots keep their original `qwen3-vl-dense-enumeration` identifiers: those are evidence identities, not a surviving knowledge classification layer.

## Reading old links without old folders

Archived Markdown keeps its original relative links to preserve its bytes. Ordinary filesystem interpretation from the new archive depth can be wrong. The read-only resolver uses the document's original logical coordinates and maps the result to the correct capture:

```sh
python -B scripts/research/check_research_knowledge.py resolve \
  docs/history/research-records/2026-09-15-root-collapse/sources/research/ideas/prefix-denoising-sft/overview.md \
  experiments/2026-06-17-inert-objective-root-cause/unit.md
```

Live links do not get this fallback: a broken active link fails validation. External worktree/output handles remain explicitly unverified. Old source-time missing links remain reported; no replacement evidence is invented. No symlinks are recreated by this reader.

## Consumer and replay boundary

`probes.parallel_owner_research.transfer.research_exposure_sources()` reads the exact preserved research-side JSON corpus. The frozen baseline is 106 files and 135 distinct image IDs. Tests compare each file hash and image-ID list, exercise missing/empty/malformed corpus rejection, and run the real `freeze_selection()` entry on synthetic inputs with publication confined to a temporary directory. The synthetic fixture verifies selected IDs, attributed source paths and occupied-path rejection; it is not a new scientific cohort or GPU run.

Three remaining script constants point directly to preserved protocol files instead of the removed alias. Their frozen protocol hashes and scientific logic are not silently changed. Current code has a new source identity; old source bytes remain in the capture. Complete original-context replay of hash-bound historical producers is not claimed. Restoring one requires its original runtime/source closure under separate authorization, not editing a receipt to make new code match old evidence.

## Validation

The single `scripts/research/check_research_knowledge.py check` verifies the flat tree, live local links, all indexed protocol roots, unique current state, local source hashes, baseline Git blob identities and exact existing retirements. Its historical-link gaps are separate from active broken links. Focused contract and transfer tests supply negative cases and the actual data-consumer boundary. Final command results and scope observations are recorded in `validation.json` after execution, not inferred from this guide.
