## 1. Freeze the fail-closed contract

- [x] 1.1 Add an interface-level regression that admits CPU preflight, mutates
  one bound target input before the vertical launch boundary, and proves the
  current launcher/model sentinel remains untouched; observe the test fail for
  the pre-change behavior.
- [x] 1.2 Add deterministic target-binding fixtures for clean capture and each
  typed rejection: non-worktree, symlink/unresolved root, conflict, staged,
  modified, deleted, renamed, untracked, changed commit, and changed declared
  effective input.
- [x] 1.3 Freeze the two current consumers' declared target roots and exact
  effective source/runtime/config/producer/validator input sets; record their
  current CPU round-trip expectations without moving scientific fields into the
  shared layer. Cover both their present fixed target and a clean dedicated
  probe-worktree fixture without introducing target-routing infrastructure.

## 2. Implement the narrow target-binding seam

- [x] 2.1 Implement canonical target-tree capture and live revalidation in the
  admission owner, preserving existing typed binding, canonical JSON,
  write-once, and two-stage journal invariants.
- [x] 2.2 Extend the natural-boundary support and K10-H20 crossover adapters to
  supply the frozen target/effective-input declarations and preserve their
  consumer-owned plans, projections, validators, finalizers, and schemas.
- [x] 2.3 Place one admission-owned pre-model revalidation choke point in the
  production-shaped support vertical immediately before its launcher boundary;
  a failed guard must leave CPU evidence durable and create no worker, model,
  GPU, vertical evidence, or final receipt.
- [x] 2.4 Remove or narrow any superseded post-launch-only caller knowledge so
  no alternate vertical path can treat `append_stage` revalidation as its sole
  target-drift guard.

## 3. Verify mechanics and compatibility

- [x] 3.1 Replay the new regression from 1.1 and demonstrate that the stale
  target now fails typed before launcher/model action while exact clean targets
  preserve the existing launcher path.
- [x] 3.2 Run deterministic binding/failure tests and production-path CPU round
  trips through both current consumers; compare their consumer-owned outputs
  with the pre-change fixtures without interpreting scientific fields.
- [x] 3.3 Run targeted formatting, type, compile, and relevant admission/journal
  tests; run strict OpenSpec validation and residue checks for stale
  post-launch-only target guards.
- [x] 3.4 Freeze the resulting infra diff and obtain a `claude-sonnet-5` leaf,
  `write:false` contract review on the exact target hash; lead disposition owns
  one bundled correction round and independently replays decisive checks.

## 4. Integration gate

- [x] 4.1 Submit the verified infra change for user review; do not merge, tag,
  push, or alter either fixed worktree directory without the next explicit
  authorization. User authorization was then received for the actual merge;
  no tag, push, or fixed-directory alteration occurred.
- [ ] 4.2 After an explicit bounded-GPU authorization, run the existing
  single-GPU mechanics smoke from the merged `research-probes` target with a
  fresh external output root; bind its exact target/runtime/device identities
  and keep its result mechanics-only.
- [x] 4.3 Revalidate the accepted merged target tree, sync the approved delta
  spec, and hand the actual merged identity to
  `establish-research-probes-baseline-v1` for its post-infra cutover ledger.
  The actual target was clean `f337de5d0bd016b79aa012acfc491544e6313333`
  with identity fingerprint
  `abe39a84025bc08e0a6249fe5415f5688d6a25e982dad58082bbbfddc028ded8`;
  approved delta sync is `30c29ed69d9ec3a74e8b96a7c8ec8a509146a765`.
