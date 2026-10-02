# Output ownership migration — current acceptance

Verified on 2026-10-02. The authorized source/document/metadata package is
**lead-accepted**. All four output roots pass placement checks. Global migration
is still open for explicitly deferred annotation relocation. The separate v2/hard
compatibility ruling has now been executed.
No GPU research, remote publication or checkpoint repacking was performed.

## Accepted source and document changes

- `46cc16c0c`: current docs route physical outputs through the single storage
  policy and current research through the canonical Research Probes owner.
- `818fba0b8`: retire v1-backed proxy constructors and the unused dependent v2
  constructor; restore the base COCO builders using current typed config,
  renderer and image planner. Real token-budget builds require an explicit
  config; no replacement profile is silently selected. Existing base COCO
  data and shared images were not rebuilt or changed.
- `25e8d95d7`: retire 395 copied Python files, 277 copied YAML files, one copied
  shell entrypoint and 23 exclusive obsolete callers/tests. Keep historical
  Markdown, inventory tables and small recorded results. Source recovery is
  Git history, not an executable archive. Current loss contracts are retained.
- `04aa7dbdf` (main), `dd67e331e` (Research Probes), `426b7f044` (Web): qualify
  exact frozen checkpoint-card inventories without rewriting payloads or
  receipts. The actual Research Probes checkpoint producer packages future
  cards with the existing lossless JSON helper before identity inventory.
  The two visual descriptions now live in the existing preservation-and-credit
  question; their redundant output README files were removed after coverage,
  identity, manifest and holder checks.

The user-authorized v1 processed directory
`public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000` is absent: nine files,
695,894,025 logical bytes retired after fresh holder checks and unchanged
path/inode/size checks. Its original provenance manifest and recorded checksums
are unchanged. This does not claim physical disk savings or regeneration parity.

The user also retired weighted-v2/hard-v2: eleven files, 574,087,074 logical
bytes. Seven unchanged provenance/validation/consumer receipts (7,804 bytes)
were verified before original retirement and kept at the existing COCO/LVIS
research artifact owner. Full-export replay from those data files is retired.

All 86 historical model cards remain byte-identical. The scan also recognized
34 newer cards outside that historical set; none were modified. The exact
five-key checkpoint identity and README hash qualify placement, not tensor
integrity, scientific validity or model behavior. A present native payload
manifest prevents legacy fallback; loose prose and symlink escapes still fail.

The visual interpretation preserves selected-case scope, visible GT provenance,
greedy visual IoU matching versus research assignment, accepted-list versus
parser-order IDs, disjoint/other-owner cases and unresolved physical identity.
It does not turn six selected images or three details into prevalence evidence.

## Verification

Package checks are reused for unchanged source, inputs and targets; the lead
checked the final routing, consumer and deletion boundaries rather than repeating
all package tests.

| Check | Result |
|---|---|
| Base COCO builder package | 27 CPU tests passed; both surviving `--help` entries, py_compile and scoped diff check exited 0. Initial RED was missing archived codec/chat imports. |
| Metadata package | 32 CPU tests passed; strict identity/native-manifest, malformed hash/path and symlink cases plus actual producer-card inventory coverage. |
| Preserved current loss contracts | 32 tests collected, exit 0. No production imports of the retired objective/analysis modules remain. |
| Current document routes | 24 worker-checked added links and seven lead-checked owner/root paths resolve. |
| Four-root output checker | Exit 0; main shared root 559 files / 90 untraversed symlinks, infrastructure 505 / 0, Research Probes 39,063 / 0, Web 1 / 0; zero findings in every root. Counts are a live scan snapshot. |
| Research knowledge | Exit 0; 330 catalog entries, 324 distilled, six current, 166 claim references, no errors. Five external links were not revalidated. |
| OpenSpec strict validation | One change valid, zero errors; `skip_specs: true` remains applicable. Validation does not close the deferred data/annotation decisions. |
| Scoped Git checks | Diff checks pass; commits contain only owned paths. Shared guidance and the separate pytest discovery fix were not included. |

The current consumer command is:

```sh
python -B -m src.artifacts.output_layout \
  --root /data/CoordExp/outputs \
  --root /data/CoordExp/.worktrees/coordexp-infras/outputs \
  --root /data/CoordExp/.worktrees/research-probes/outputs \
  --root /data/CoordExp/.worktrees/research-probes-web-codex/outputs
```

Structured consumer evidence and exact retirement records are at the existing
maintenance owner:
`/data/CoordExp/.worktrees/coordexp-infras/outputs/maintenance/output-storage-closeout-20261002/source-and-metadata-closeout.json`.
The full historical machine inventory remains unchanged and hash-bound by
`inventory.json`; older verification snapshots are recoverable from Git history.

## Remaining boundary

- Gate A, the annotation review UI and exported retired Label Studio drafts
  retain their existing authority and paths. Annotation relocation is deferred
  to the later public-data migration; nothing was relocated or restarted here.
- The separate v2/hard compatibility decision is executed: both dataset paths
  are absent; original small records retain their hashes at the existing
  research artifact owner. No raw images or base COCO data were changed.
- The older `rescale_32_1024_bbox_max60_lvis_proxy` input remains referenced by
  existing inference/analysis configs. It was outside the selected length-budget
  v1 retirement; those configs and that data were not broadly pruned.
- Full public-data preparation was not qualified. The pre-existing bbox-format
  test still cannot collect because `rescale_jsonl.py` imports absent
  `src.datasets.preprocessors.resize`. Other unrelated old-runtime tests also
  remain outside this scoped retirement. These are not successful checks or
  newly introduced failures.

Tasks 4.2/4.4 remain open for the global annotation migration boundary above. The change is
not archived and global migration is not reported as complete. The bounded
source and metadata acceptance does not authorize a research launch or expand
scientific claims.
