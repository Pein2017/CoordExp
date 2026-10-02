# Documentation as a long-lived asset

Keep an explanation when its value survives a change of function names, defaults,
backend versions or experiment stage: design reasons, semantic distinctions,
ownership boundaries and important failure modes. Keep one small navigation map.
Do not maintain parallel call graphs, parameter inventories, supported-version
matrices, run-status pages or narrated copies of source code in this directory.

## One owner for each kind of knowledge

`docs/` is checkout-local unless a page explicitly owns a cross-checkout policy.
Use the exact checkout's source, typed configs, CLI help and tests for executable
behavior. CodeGraph is a navigation aid bound to that same absolute checkout;
confirm important results against source because an index can be stale.
Compatibility-sensitive requirements belong to the owning OpenSpec, not a second
normative copy in docs. A code change does not automatically require a new doc.
A changed long-lived decision updates its existing owner rather than adding a
second design file. Small implementation explanations belong next to the code.

Scientific observations, populations, losses, evidence and current decisions
belong to the owning research question/unit. Temporary implementation planning
uses the existing bounded change, not a permanent docs diary. At closeout, distill
useful reasoning into its real owner and recover obsolete detail through Git.
Do not create `docs/history/`, `docs/archive/`, `docs/superpowers/`, `docs/handoffs/`
or per-run documentation trees. No maintained runtime reads docs as admission
data; machine-readable decisions and measured receipts have separate owners.

## Closed retirement, not a new archive inbox

The finite [retirement seal](../manifests/documentation/retirement-20261003.json)
binds the original Git commit and docs tree. Removed files are recoverable from
that exact snapshot, including useful material that has not been scientifically
reinterpreted. Sealing is not a claim that every old conclusion is accepted or
fully distilled. Preserve original provenance and distinguish invalid runs from
scientific null results. Never re-execute recovered historical code by default.

Use `python -B -m scripts.check_documentation --verify-recovery` to verify the
snapshot and any sealed non-Git bytes. Use its `--show-original <old-doc-path>`
option to read one original, or `git show <sealed-commit>:<old-doc-path>` for a
Git-backed file. Neither operation restores a tree or authorizes execution.
Git ancestry preserves the snapshot in ordinary descendants; it is not an
independent off-machine backup. Keep the sealed commit reachable when explicitly
rewriting history or exporting a repository. Do not append new history to this
closed retirement package.

The current `docs/README.md` is the only live inventory. The checker validates
its linked assets, local docs links and absence of runtime prose dependencies.
A new lasting asset requires an intentional README update, not a change to the
closed historical seal or a second catalog. Boilerplate generation and
indiscriminate archival growth are not default workflows.
