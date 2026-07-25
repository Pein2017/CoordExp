---
name: baidu-netdisk-transfer
description: Transfer data with BaiduPCS-Go for one-off upload or download, browser-cookie login, durable tmux execution, bypy recovery, or append-only union sync with conflict detection.
---

# Baidu Netdisk Transfer

Choose the mode before launching because their overwrite semantics differ.

- **One-off**: deliberate upload, download, replacement, login recovery, or a
  single large transfer.
- **Union sync**: conservative multi-machine sharing that adds missing files,
  detects same-path/different-content conflicts, and never deletes or
  overwrites.

Ask before switching modes when overwrite behavior, deletion, or remote layout
is ambiguous.

## One-Off

Use the bundled `scripts/install_baidupcsgo.sh`, `scripts/upload_dir.sh`, and
`scripts/download_dir.sh`, resolving them relative to this skill directory.

- BaiduPCS-Go sees the Netdisk root; bypy may expose only an app sandbox.
- Use browser cookies when available, but never print or persist them in tracked
  output.
- Preserve the requested remote layout and verify account, quota, and target
  before a large transfer.
- Prefer conservative upload concurrency and retries. The upload helper is for
  deliberate replacement; select its skip policy when remote preservation
  matters. The download helper overwrites locally unless its preservation
  option is selected.
- Run long transfers in `tmux` and verify remote visibility rather than relying
  only on pane output.
- Scan generated names for characters rejected by Baidu or Windows. If a small
  blocking set is renamed, keep a mapping manifest beside the transferred
  artifact.

One-off completion requires source and destination identities, transfer exit,
representative remote or local visibility, overwrite policy, and any rename
mapping.

### Login

1. Resolve the installed BaiduPCS-Go binary and inspect its current `login
   --help` before choosing a login method.
2. Prefer the interactive or QR flow on a shared host. For browser-cookie login,
   require a user-named local cookie file that is untracked, ignored, and
   readable only by its owner.
3. In a private shell with tracing disabled, read the cookie into a
   task-specific variable, pass it through the installed version's supported
   cookie flag, then immediately unset the variable. Never echo the command,
   cookie, or expanded arguments. Warn that command-line cookie flags can be
   visible to other users through process inspection.
4. Verify the authenticated account with quota and root-listing operations
   before transferring data. Stop on an unexpected account or root.

Login completes only when account, quota, and root are verified without cookie
content appearing in tracked files or captured output.

## Union Sync

Use the bundled `scripts/baidu_union_sync.py` and
`references/config-template.json`. Read `references/semantics.md` before
changing conflict or deletion behavior.

The state transition is:

```text
node A + node B + remote => union of known files
```

1. Run the helper's doctor and status operations with the selected config.
2. Use SHA-256 manifest signatures when claiming content-conflict detection;
   size-only signatures detect only size differences and must be reported as a
   weaker inventory check.
3. Confirm the remote manifest set is fresh enough to represent every writer.
   For a pre-existing remote tree with no trustworthy manifest, inventory or
   seed it from a verified node before claiming union safety.
4. Review the dry run; mutation requires its explicit apply flag.
5. Pull remote-only files into staging and merge with ignore-existing behavior.
6. Push local-only files with skip-existing behavior.
7. Stop on conflicts unless the user explicitly authorizes manual recovery.

Preserve configured legacy state names so existing nodes retain manifest
continuity. Reject symlinks, special files, and unsafe names by default. Never
add mirror, delete, or overwrite semantics to an automated sync. For intentional
deletion, stop every sync loop, remove the item from each relevant surface, add
a deny rule for old nodes, then restart.

Union-sync completion requires the pre/post status, files added in each
direction, conflicts, skipped unsafe entries, node identity, and whether any
mutation was applied. State the signature mode and remote-manifest freshness so
the conflict-detection claim is bounded honestly.

## Failure Triage

- Authorization or slice-hash failure: verify login, quota, real-root target,
  conservative concurrency, retries, and rapid-upload settings.
- Missing downloaded files: inspect the tool's account-prefixed staging area.
- bypy-visible files missing in BaiduPCS-Go: recreate or locate the intended
  path under the real Netdisk root.
- ETA questions: combine completed bytes, current shard, local sizes, and
  verified remote visibility.
