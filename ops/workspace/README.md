# Workspace Ops

Local workstation and repository hygiene helpers live here when they are not
CoordExp training, inference, evaluation, or artifact-production entrypoints.

- `workspace_gc.sh`: dry-run-first cleanup for local caches and temporary
  workspace outputs. It refuses to delete checkpoint/output directories.
