# Infrastructure output ownership

Infrastructure runs use this physical checkout's `outputs/`, at
`/data/CoordExp/.worktrees/coordexp-infras/outputs/`. Root outputs is reserved
for selected shared assets, not infrastructure execution. The cross-checkout
policy is `/data/CoordExp/docs/OUTPUT_STORAGE_POLICY.md`.

Maintained implementation belongs to its source owner, scientific interpretation
to its research owner, and raw receipts/logs/results to the run owner. Copying
or integrating source does not qualify a new model launch. Preserve original
producer/config/data identity and verified bytes when an asset is promoted.
Historical docs use [closed Git recovery](RETENTION.md), not this worktree as
a generic legacy store.
