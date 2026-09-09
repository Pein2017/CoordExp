# Preservation and source disposition

All source tips below were clean when captured, except the research base with the known out-of-scope shared configuration deletions and this change. No remote refs or excluded worktrees were touched.

Output locator record: `/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/sources-and-outputs.json`. Small worktree-local output files were copied and byte-compared; relative output symlinks in the saved copy point to their original resolved targets. Their original link text is recorded. Root research-base outputs remain in place.

| Worktree | Saved commit | Recoverable ref |
| --- | --- | --- |
| research-probes | `73b8b3cc2614db052055c7c49fc33d6207b0f80a` | `archive/research-restructure-20260909/research-probes` |
| research-probe-infras | `2daa48b22fa4a2f7779ca855f2a736a0b69c776d` | `archive/research-restructure-20260909/research-probe-infras` |
| c-anchored-owner-mechanism-audit | `9eb1372a3b53cb1fae691419490388d494f2837d` | `archive/research-restructure-20260909/c-anchored-owner-mechanism-audit` |
| coco-gt-correction-portfolio | `c00043abc5b538cf62aedb143e983fb0a386b40a` | `archive/research-restructure-20260909/coco-gt-correction-portfolio` |
| dora-prox-linear-n2 | `ba801de514143d8fd25822180323e3e2b101bcd4` | `archive/research-restructure-20260909/dora-prox-linear-n2` |
| human13-output-qp-identity-generalization | `aa7cdccb9a4d10631cda22fcd61bcb9660813918` | `archive/research-restructure-20260909/human13-output-qp-identity-generalization` |
| image2299-logit-lens | `269477a3a78585dc07c11966daa07df23d05f473` | `archive/research-restructure-20260909/image2299-logit-lens` |
| n256-shared-output-qp-norm-scaling | `6b846c7c4488638b57d357b274ac3885e0256d55` | `archive/research-restructure-20260909/n256-shared-output-qp-norm-scaling` |
| self-rollout-behavior | `d6de155fb5f80b178e438e3036462b58eda6420c` | `archive/research-restructure-20260909/self-rollout-behavior` |

Source recovery: `git show <archive-ref>:<repo-relative-path>` or an isolated checkout at that ref. These refs preserve full committed sources, including the former DORA dirty content already committed by the user. This is source recovery, not a model replay claim.

Historical absolute code bindings required by retained reducers will be materialized from these refs into the preservation root as needed, before retiring their original provider directories. No old manifest/hash will be rewritten.

The inspected source-directory holders were editor/index tooling, including this team. No process has been stopped. Relevant execution/write holders will be checked again immediately before retirement.

Row-cross frozen COCO code bindings: seven files materialized at `/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/coco-original-code` from the saved COCO ref, with byte counts and existing manifest hashes verified. Use the reducer's explicit original-code-root option; the historical manifest is unchanged.

Other open changes remain unfinished: the Human13 all-HF shared-surface vertical still has five baseline/prelaunch/update tasks; target-binding hardening still has one separately authorized GPU task; Human13 K-trajectory crossover still has nine dose/vertical/matrix tasks. This refactor preserves their contracts and does not claim those experiments complete. The initial four-package maintenance selection and shared implementation ownership are specified in design section8.

## Completed retirement (2026-09-09)

All eight eligible source worktrees and their local branches were removed after a fresh clean-tip/ref check and a `/proc` cwd/open-file scan found no holders. The fixed `research-probes` remains locked. Production/Swift, excluded bridge-cache content, remote refs and shared configuration were not changed.

The preservation root additionally contains `ignored-local/` (54 files, 547,780,664 bytes), copied and byte-compared before retirement; `ignored-local-preservation.json` records counts. These include packing caches and local research/index material, beyond the separately preserved outputs. `retirement.json` records exact removed paths, local branches and archived commits. `pre-retirement-holders.json` records the empty holder scan. Historical Markdown targets are materialized under `historical-links/`; source recovery refs remain the authority for complete trees.

Recovered outputs and maintained package consumers are checked after retirement. The complete original source/hash identity is preserved; historical absolute paths written inside old receipts are not rewritten or promised as still existing directories.
