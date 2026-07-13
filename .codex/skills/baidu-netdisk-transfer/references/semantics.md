# BaiduDisk Union Sync Semantics

## Contract

This workflow makes Baidu Netdisk act like a large-asset union remote, not a
byte-exact mirror.

Automated sync may:

- upload files that exist locally and are not known remotely
- download files that are known remotely and do not exist locally
- write append-only manifests
- write reports and local staging files

Automated sync must not:

- delete local files
- delete remote files
- overwrite local files
- overwrite remote files
- resolve same-path different-content conflicts by guessing

## Commands

The helper preserves legacy state names such as `temp/baidudisk-union-sync`
and `.baidudisk-union-sync-node-id`; changing them can sever manifest
continuity for existing nodes.

`doctor`
: Verifies that required tools are present and BaiduPCS-Go can talk to the
  account.

`scan ROOT`
: Scans the local tree and writes a local manifest.

`status ROOT`
: Downloads remote manifests, scans local files, and reports likely uploads,
  downloads, and conflicts. It does not transfer asset files.

`push ROOT --apply`
: Uploads the local root to the remote parent with skip-existing semantics,
  then uploads a unique local manifest for this node.

`pull ROOT --apply`
: Downloads the remote root to local staging without overwrite, then merges
  from staging into the local root with ignore-existing semantics.

`sync ROOT --apply`
: Runs pull, then push.

Transfers are dry-run unless `--apply` is present. For a long authorized sync:

```bash
tmux new -s baidudisk_union_sync_outputs
python .codex/skills/baidu-netdisk-transfer/scripts/baidu_union_sync.py \
  --config temp/baidu-netdisk-transfer/config.json sync outputs --apply
```

Automated upload uses `--policy skip`; pull merges staging with
`rsync --ignore-existing`. Stop on conflicts. Reject symlinks, special files,
and unsafe cross-platform names. Timers never use `--delete`, `--ow`, overwrite,
or mirror semantics.

## Conflict

A conflict is a same relative path where local and remote manifests disagree on
the recorded identity. With `hash_mode: "size"` this detects size mismatches
only; use a content-hash mode when same-size/different-content conflicts must be
detected. The safe default is to stop and ask for a manual decision.

Common manual resolutions:

- rename one copy and rerun sync
- delete the unwanted copy locally and remotely
- exclude mutable files that should not be synced

## Deletes

Deletes are manual. If old nodes may re-upload deleted files, place the path in
a denylist before restarting automated sync.

Stop every relevant sync loop, delete manually on each relevant local and
remote surface, add the denylist entry, then restart old nodes.

The denylist only blocks upload. It is not a deletion mechanism.

## Filename Safety

For cross-platform and Baidu compatibility, reject names containing characters
that frequently break cloud tools or Windows clients:

```text
< > : " | ? *
```

If the user wants automatic renaming, perform it as a separate explicit
maintenance operation and write a mapping manifest.
