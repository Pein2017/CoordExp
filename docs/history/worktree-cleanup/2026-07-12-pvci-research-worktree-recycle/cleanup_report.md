# PVCI Temporary Worktree Reclamation Report

## Preserved

- 43 files in the combined live PVCI research topic.
- 53 immutable source snapshots recorded in `manifest.tsv`:
  - 46 source records promoted into the live research reading path;
  - 7 planning/design records retained as raw history only.
- The CoordExp-Swift physical-length `summary.md` missing from the target.
- Six lightweight JSON artifact snapshots plus the manifest entry for one
  byte-identical 4.1 MB safetensors payload already present in Swift outputs.
- Four previously absent proposal-bridge request/result JSON files copied into
  Swift's ignored `outputs/probes/coordexp_swift/` artifact tree.

## Deliberately Not Promoted

- Painted-GT and PVCI implementation code, configs, scripts, tests, and active
  OpenSpec changes.
- `69ed/.cache` (about 2.5 GB), `eb1c/.cache` (about 986 MB), temporary adapters,
  debug images, Python caches, and other generated scratch data.
- Historical `docs/history/` already present byte-for-byte in Swift.
- Old prefix-denoising note paths already represented by Swift research units.

## Evidence Boundary

Many experiment units cite external `/data/CoordExp/outputs/...` roots. This
cleanup verifies only the seven artifacts listed in `local_artifacts.tsv`.
Other cited roots remain provenance handles and are not claimed as retained by
this bundle.

The temporary branches remain useful only as Git history. Their implementation
must not be treated as compatible with current CoordExp-Swift infrastructure.
Future experiments should reimplement the smallest required mechanism on the
canonical branch and cite the corresponding research unit.

## Cleanup Gate

- `eb1c` was clean before collection and was removed after the preservation
  commit passed its checks. Its local branch was intentionally retained.
- `69ed` contains disposable `.codex` synchronization dirt and is the active
  session worktree. Remove it only after this session moves elsewhere and the
  committed preservation bundle is confirmed.
- Branch deletion and remote deletion are separate, intentionally deferred
  decisions.
