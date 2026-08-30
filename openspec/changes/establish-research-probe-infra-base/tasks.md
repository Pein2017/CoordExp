## 1. Freeze the public composition seam

- [ ] 1.1 Add an interface-level test that imports the complete minimal
  infra-base surface from `src.artifacts`, and update the existing intentional
  package-export contract in `tests/artifacts/test_research_probe_admission.py`
  for the same approved additions; observe the pre-change failure for the
  currently missing public exports before changing the facade.
- [ ] 1.2 Add a one-shot producer characterization proving strict exclusive
  publication refuses an occupied result path and preserves the original bytes
  and hash.
- [ ] 1.3 Add a two-producer journal characterization using a common immutable
  input but separate roots and opposite append orders; prove there is no shared
  lock, sequence, terminal barrier, cross-mutation, or scientific
  interpretation.
- [ ] 1.4 Record the Image2299 and historical Human13 mapping used by the guide:
  identify only their shared identity/durability mechanics and explicitly keep
  distributed stages, direct differentiable runtime, optimizer/rollback,
  checkpoints, metrics, and stop rules outside the base.  Do not read or edit
  active dirty Image2299 files during implementation and do not restore a
  Human13 worktree.

## 2. Complete the existing public artifact surface

- [ ] 2.1 Extend the existing lazy `src.artifacts` facade with only
  `publish_json_exclusive`, `BindingManifest`, `TargetTreeBinding`,
  `TargetTreeIdentity`, `AdmissionInspection`,
  `capture_target_tree_binding`, `revalidate_binding_manifest`, and
  `revalidate_target_tree_binding`; keep every implementation in its current
  deep owner module.
- [ ] 2.2 Replay the import regression and prove the public surface remains
  lazy, CPU-only, and behavior-identical to direct imports from the deep
  owners.
- [ ] 2.3 Replay the one-shot and independent-journal characterizations through
  the public surface.  Do not add a wrapper, registry, configuration object,
  generated runner, compatibility alias, overwrite fallback, or new
  dependency to make the tests pass.

## 3. Publish one canonical operator route

- [ ] 3.1 Add `docs/RESEARCH_PROBE_INFRA_BASE.md` with the one-shot,
  journaled, admitted-launch, and inference-backed selection table; include
  exact public imports, caller obligations, failure meaning, minimal examples,
  per-producer topology, and mechanics-only status language.
- [ ] 3.2 Add one routing link from `docs/AGENT_INDEX.md` and one link at the
  existing infra-lane paragraph in `docs/BRANCH_AND_WORKTREE_POLICY.md`.
  Preserve that policy as the sole lifecycle authority.
- [ ] 3.3 Add the promotion checklist requiring two live cross-direction
  consumers, semantic/topology agreement, evidence that direct composition is
  insufficient, and one cheapest falsifier.  Mark the generic coordinator,
  phase DSL, trainable-HF session, optimizer/RNG transaction, checkpoint API,
  monitor, and lifecycle automation as deferred rather than planned features.
- [ ] 3.4 Run a documentation residue check and remove any wording that
  duplicates lifecycle commands, assigns scientific meaning, treats historical
  Human13 as live, or presents Image2299-local helpers as stable owners.

## 4. Verify and submit the planning-owned change

- [ ] 4.1 Run
  `CUDA_VISIBLE_DEVICES=-1 conda run -n ms pytest -q -p no:cacheprovider tests/artifacts/test_research_probe_infra_base.py tests/artifacts/test_evidence_journal.py tests/artifacts/test_research_probe_admission.py`
  and report exact counts without interpreting opaque research payloads.
- [ ] 4.2 Run targeted import/compile/style checks for `src/artifacts`, the new
  test, and the changed docs; inspect the exact diff for accidental source,
  worktree-lifecycle, or active-probe edits.
- [ ] 4.3 Run `openspec validate establish-research-probe-infra-base --strict`,
  `git diff --check -- openspec/changes/establish-research-probe-infra-base src/artifacts tests/artifacts docs`,
  and a standards audit against the stable journal, admission, inference, and
  branch/worktree contracts.
- [ ] 4.4 Freeze the resulting diff and obtain one bounded read-only
  intent-contract review targeting only these blocking failure modes: a hidden
  coordinator, weakened exclusive publication, global producer ordering,
  scientific-semantic leakage, or duplicated lifecycle authority.  Apply at
  most one bundled correction round and have the lead replay the decisive
  checks.
- [ ] 4.5 Submit the verified infra-base change for user review.  Do not merge
  it into `research-probes`, cut a baseline tag, update Image2299, revive
  Human13, run a GPU smoke, commit, push, or mutate either fixed worktree
  without the corresponding later authorization.
