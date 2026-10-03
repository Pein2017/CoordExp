# Greedy-prefix temporary execution checkout retirement

The user requested retirement of session `01a0fe03-4ce4-7490-b937-5285451b0b11`
and `/data/CoordExp/.worktrees/greedy-prefix-native-01` on 2026-10-03.
This record owns artifact access and checkout retirement only. All scientific
statuses, accepted measurements, historical receipts and execution identities
remain unchanged. It grants no new research or execution authority.

## Retention and current access

Retirement is complete and lead-accepted. All nine retained roots passed
copy/readback and current inspection checks. The temporary checkout and its Git
worktree metadata were removed; the specified session was archived.

The original output parent was
`/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03`.
The retained parent is
`/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-03`.
For each row, append the unit directory to the old parent; append the unit
directory and `/retained-execution-01` to the retained parent. Preserve the
remaining relative file path exactly.

| Unit directory | Current retained evidence | Files |
|---|---|---:|
| `greedy-prefix-branching-01` | [Round01](../../../outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/retained-execution-01/) | 30 |
| `matched-coordinate-branches-02` | [Round02](../../../outputs/research/physical-fn-recovery/2026-10-03/matched-coordinate-branches-02/retained-execution-01/) | 49 |
| `completed-row-crossover-03` | [Round03](../../../outputs/research/physical-fn-recovery/2026-10-03/completed-row-crossover-03/retained-execution-01/) | 27 |
| `prefix-exposure-ranking-04` | [Round04](../../../outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/retained-execution-01/) | 181 |
| `short-dose-ranking-06` | [Round06](../../../outputs/research/physical-fn-recovery/2026-10-03/short-dose-ranking-06/retained-execution-01/) | 144 |
| `mass-versus-ranking-07` | [Round07](../../../outputs/research/physical-fn-recovery/2026-10-03/mass-versus-ranking-07/retained-execution-01/) | 135 |
| `output-delta-crossover-08` | [Round08](../../../outputs/research/physical-fn-recovery/2026-10-03/output-delta-crossover-08/retained-execution-01/) | 155 |
| `endpoint-block-ablation-09` | [Round09](../../../outputs/research/physical-fn-recovery/2026-10-03/endpoint-block-ablation-09/retained-execution-01/) | 145 |
| `dora-input-ablation-10` | [Round10](../../../outputs/research/physical-fn-recovery/2026-10-03/dora-input-ablation-10/retained-execution-01/) | 148 |

The [copy manifest](../../../outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/worktree-retirement-01/copy-manifest.json)
records every relative path, original and retained root, size and SHA256.
The verified inventory is 1,014 files and 2,100,079,252 logical bytes,
including 18 safetensors files. Each source and destination file was hashed once;
all bytes matched, and both inventories/stat records remained unchanged during
verification. The manifest SHA256 is
`b320eabf27b614368090376439e2b87f18b9ec1d51f37d254a325ac4350147b3`.
Retention is local under the canonical research
output owner; this is not an independent off-host backup. The existing canonical
unit outputs are separate from the retained execution subdirectories.

Each affected `state.json` adds `artifact_retention.current_artifacts` for its
historical artifact fields. Existing fields retain producer identity. To inspect
a sealed JSON path, use the explicit unit mapping above only when opening the
file; retain the original path when looking up its hash or provenance key.
No symlink, obsolete-checkout fallback, receipt rewrite or requalification is
introduced.

## Downstream inspection

- Round05 [attribution](../../../outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05/attribution.json)
  retains 114 entries. Its 58 old-path input bindings (54 raw files and four
  receipts/contracts) resolve through the Round04 mapping. Its 228 side raw
  locators refer to 43 unique files within that retained set. Preserve those
  logical locators and their accepted hash keys.
- Round11 [gallery](../../../outputs/research/physical-fn-recovery/2026-10-03/known-owner-visual-audit-11/index.html)
  and [ledger](../../../outputs/research/physical-fn-recovery/2026-10-03/known-owner-visual-audit-11/ledger.json)
  already use local presentation paths. The existing
  [receipt](../../../outputs/research/physical-fn-recovery/2026-10-03/known-owner-visual-audit-11/receipt.json)
  field `raw_identities` maps all 20 original raw paths to `copied_raw` files
  with hashes and raw identities. Use that map to expand case/pool provenance;
  use retained Round10 for its original acceptance and phase receipts.

Historical execution and readback commands that require the retired checkout
are archived. In particular, the old attribution readback binds its historical
writer source, and the visual-audit command is a builder, not a readback.
Neither is rerun for retirement. Retained inspection does not establish that
these historical commands are portable or authorize another model call.

## Source recovery and finalization

The retired checkout was detached at
`c36dec2648151408b4fb6080e82c3c6a87d6bb27`. At removal it had no unique commits or
uncommitted tracked/untracked work. That commit and all recorded execution
source commits remain ancestors of canonical research commit
`a7f826d9a73922a55e6611f69b0e950577b797ee`. Recover historical source with
`git show <execution_source_commit>:<repository-relative-path>` from the
canonical repository. There is no `greedy-prefix-native-01` branch ref to delete.

The [consumer check](../../../outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/worktree-retirement-01/consumer-check-01.json)
passed with old-checkout file opens prohibited: 85 current artifact links,
55 state hash bindings, all 58 Round05 input bindings and 228 raw references,
all 25 Round11 input bindings and 66 raw references, 186 gallery HTML links,
and 25 added results links. This is evidence-access verification, not research
recomputation. The existing research-knowledge check exited 0 with no errors.

Immediately before removal, all source/destination inventory and stat records
still matched the verified copy; no process held an old-checkout cwd or open
file. The exact `git worktree remove --force` command exited 0. Its force flag
covered only the already-retained ignored outputs and 74 disposable Python
cache files; source was clean. Both the directory and its Git metadata are
absent. No branch ref remained to remove.

The app acknowledged `archived: true` for session
`01a0fe03-4ce4-7490-b937-5285451b0b11` on host
`remote-ssh-discovered:pein-train`. Conversation history is preserved. The
[final receipt](../../../outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/worktree-retirement-01/retirement-complete-01.json)
records the actual outcomes and evidence pointers. No unrelated worktree,
branch, process or external reference checkout was included in this retirement.
