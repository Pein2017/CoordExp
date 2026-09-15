# Frozen research source archive: 2026-09-15

This is a byte-preserving provenance collection, **not the current research workspace or launch authority**. Begin new research at [the live program](../../../../research/qwen3-vl-dense-enumeration/index.md); maintenance rules live in [the convention](../../../../research/CONVENTIONS.md).

The [manifest](manifest.json) binds every relocated non-bytecode file to its original logical path, byte count, SHA-256 and baseline Git commit. It also preserves the previous routers before their authorized edits. All original Markdown, JSON, TSV, scripts, reviews, proposals and handoffs remain available. Generated Python bytecode was removed by an exact inventoried, hash-guarded list; it was already Git-ignored. Empty log files were not treated as scientific duplicates and deleted.

## How to read an old source

`current`, `running`, `next`, `authorized`, and `latest` inside these documents are **historical source statements**. They do not describe the present Project or authorize a repeat. Old transport documents, including post-Pro discussion and N256 handoffs, are preserved here because their ideas and corrections matter, not because their once-current instructions remain live.

Most within-experiment relative links still identify nearby preserved files. Cross-tree links must be interpreted relative to the original source path in the manifest, not relative to this archive's deeper physical location. Use:

```sh
python scripts/research/check_research_knowledge.py resolve <source-document-path> <link-target>
```

The resolver accepts an original or archived source-document path, maps the logical local target through the manifest, and reports whether its local file exists. It does not fetch external artifacts or run commands. An absolute historical path outside this registered checkout remains an external handle, not locally verified evidence.

## Compatibility and executable provenance

`research/investigations` is one explicit compatibility symlink to the archived investigation tree. It is retained for actual old-path JSON consumers and evidence references, **not for new writing or current-state discovery**. At migration the existing transfer selector's filtered JSON walk retained every literal path and file hash. The checker repeats this invariant.

This alias does not restore `Path(__file__).resolve()` identity, fixed-depth repository discovery, or exact original-path bindings in sealed historical producers. **No historical GPU run or sealed-producer replay is claimed.** Do not patch old receipts or rewrite frozen source to make such a replay look valid. Original tracked bytes can be recovered from baseline `78bc27d8e5e8639450e01990908fcd3b417bbfa2` at the manifest's original paths; execution requires a separately authorized original-context restoration or a new implementation/execution identity. No clone or worktree was created for this migration.

## What the intake proves

It proves local source-byte preservation, complete catalog coverage and the checked data-path behavior, not independent scientific replication or visual re-adjudication. Source assertions about historical artifacts remain source assertions until the corresponding raw evidence is deliberately checked. Historical link gaps are recorded separately from new live-link failures.

The top-level operational harness benchmark and the historical knowledge handoff have their own preserved subtrees under `investigations/`. They are not classified as new dense-enumeration experiments. The imported claim/source ledgers retain their original identity and do not become a cross-checkpoint leaderboard.
