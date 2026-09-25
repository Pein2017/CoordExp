# Temporary legacy salvage

This is a temporary recycle/extraction queue, not the archive for completed
research and not a date-based warehouse. A useful result, protocol, negative
finding, research asset or source observation belongs with its research owner,
even when it is old. Implementation files and source snapshots never belong
under `docs/`.

The current research layout is owned by
[research/CONVENTIONS.md](../../research/CONVENTIONS.md). Global behavior and
engineering documentation remain in [docs/](../README.md); current-run source
captures may remain local under [reference/](../../reference/README.md).

## Remaining salvage and exit conditions

| Material | Residual use | Extraction or removal condition |
|---|---|---|
| `architecture/`, `engineering/`, `evaluation/` | Superseded global interfaces and historical behavior needed for contract comparisons. | Integrate a decision-relevant distinction into the current global owner, or remove the obsolete source document. |
| `research-records/` | Unresolved alternatives, old synthesis context, immutable preservation manifests and frozen exposure JSON. Primary protocols/results now live under `research/experiments/`. | Extract only a genuinely missing scientific distinction into its question/unit. Remove consumed context rather than retaining another router. |
| `worktree-retirements/`, `worktree-cleanup/`, `worktree-union/` | Unique branch-specific designs and retirement provenance not yet reconciled with current owners. Named research records and superseded global copies were separated from this material. | Resolve a concrete original-branch question, incorporate the remaining useful distinction, and delete the unneeded transport or snapshot. Do not regenerate these bulk intakes. |
| `superpowers/`, `root-orphans/` | Unmatched legacy implementation plans and design context whose useful remainder is not yet established. They are not current execution instructions. | Find a real current consumer or scientific owner and integrate the useful portion; otherwise discard after provenance/reference checks. |

No file earns permanent retention merely because it is unique. The remaining
messy material is explicitly pending extraction or deletion, not promoted
knowledge. Do not add routine run notes, complete source trees or copies of
current documentation here.

## Frozen exposure compatibility boundary

The 106 JSON records below
`research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/`
remain byte-exact at their original paths because the unchanged
`probes/parallel_owner_research/transfer.py` still reads that corpus directly.
Their union of 135 image IDs is an exclusion boundary, not a Markdown reading
path. Move them only with the consumer's owner and a source-hash/image-ID parity
check. This exception neither permits code in docs nor makes history a permanent
data store.

## Recovery and checks

`manifests/documentation-layout.json` now contains path routes only. It no longer
claims recovery of old source bytes or verifies source-code hashes. The frozen
exposure JSON files remain because a live consumer reads them; their data
identity checks are separate from source-code snapshot retention.

From the canonical research checkout, run:

```sh
python -B scripts/research/check_research_knowledge.py check
python -B -m pytest -q -p no:cacheprovider tests/knowledge/test_documentation_ownership.py probes/parallel_owner_research/tests/test_transfer_exposure.py
```

These are layout, identity and consumer checks, not scientific revalidation or
permission to resume experiments. The root checkout and ongoing probe changes
are outside this cleanup's mutation scope.
