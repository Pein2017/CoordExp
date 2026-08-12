## ADDED Requirements

### Requirement: Zero-Weight Protected Loss Diagnostics Are Gradient-Free

When a protected auxiliary loss has effective weight zero, the trainer MUST
preserve its accepted raw diagnostic value without retaining an autograd graph
or contributing gradients for that term. The zero-weight optimized path SHALL
produce the same total training loss and trainable gradients as a reference run
that omits the term from the objective, while nonzero-weight behavior remains
unchanged. The strict FP32 raw-diagnostic comparison SHALL reuse the exact same
frozen FP32 logits, targets, semantic-atom inventory, denominators, and weights
for the reference and detached/no-grad paths; it MUST NOT compare diagnostics
produced by two independent model forwards.

#### Scenario: A zero-weight token-type term emits diagnostics

- **WHEN** the token-type auxiliary term has effective weight zero and raw
diagnostics are enabled
- **THEN** its raw FP32 diagnostic is finite and numerically equal to the
  accepted reference, but the diagnostic has no autograd dependency and makes
  no contribution to total-loss gradients
- **AND** both diagnostic paths consume one content-bound frozen FP32 logits
  tensor and identical targets, atoms, denominators, and weights

#### Scenario: The frozen Wave 3 plan crosses a later config default

- **WHEN** the current resolved config differs from the frozen Wave 3 parent
  only by the exact later strict defaults enumerated by
  `coordexp-swift-wave3-config-compatibility-projection-v2`
- **THEN** the plan retains full current fingerprint
  `da2a010eaacc6970c616e39a790372db43e6089357b9501d3b40f60f157fb5a9`
  and provider mode `synchronous`
- **AND** the projection removes exactly the provider, source-order next-fit
  policy, `window_size: null`, `lookahead: null`, `seed: 0`, `worker_count: 1`,
  `fragment_item_budget: 1024`, `fragment_byte_budget: 4194304`,
  `cursor_byte_budget: 65536`, `max_packs_per_fragment: null`, and complete
  `{checkpoint_dir: null, mode: disabled}` resume object
  before reproducing frozen parent fingerprint
  `de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`
- **AND** a missing, extra, reordered, or changed path/value row, a wrong
  current or projected digest, or any other resolved-config drift fails closed
- **AND** the live full identity and synchronous provider are reasserted
  immediately before attempt-marker publication and GPU setup and retained in
  the marker and terminal receipt

#### Scenario: A Wave 3 JSON artifact is published

- **WHEN** the plan, attempt marker, terminal receipt, or receipt-publication
  sidecar is committed
- **THEN** publication uses strict JSON and a same-filesystem atomic
  absent-target/no-replace link, fsyncs the file and containing directory, and
  strictly reloads and hash-validates the final target
- **AND** a post-link failure is recoverable only when that exact call observed
  its own successful link and the final bytes strictly reload as the intended
  artifact
- **AND** a pre-link failure exposes no partial final file, while an identical
  or foreign pre-existing target remains an immutable collision and is never
  claimed as this call's persistence

#### Scenario: The authorized Wave 3 v4 successor shares a selected GPU

- **WHEN** the one fresh-root Wave 3 v4 correctness/plumbing successor is run
  under the user's 2026-08-11 shared-GPU authorization
- **THEN** two pre-marker samples at least two seconds apart bind the exact
  selected index/UUID, exact 81920 MiB total memory, no more than 49152 MiB
  pre-existing use, at least 32768 MiB headroom, and the stable pre-existing
  `(gpu_uuid, driver_pid)` set
- **AND** after every spawned-worker outcome the controller performs bounded
  session-wide cleanup followed by exactly two post-run subset samples at least
  two seconds apart, and fails on any survivor, new GPU row, or incomplete
  cleanup or sampling evidence
- **AND** the controller never signals the pre-existing baseline jobs, the
  immutable v2/v3 artifacts remain non-executable history, and shared-load
  timing, memory, utilization, and efficiency observations are not promotion
  evidence

#### Scenario: Backward is compared with a base-only objective

- **WHEN** matched model state and micro-step inputs are backpropagated through
  the zero-weight optimized path and through the base-only reference
- **THEN** total loss and every trainable gradient agree within the declared
  dtype-specific tolerance

#### Scenario: The auxiliary weight becomes nonzero

- **WHEN** the same protected loss is configured with a nonzero effective
  weight
- **THEN** the accepted differentiable FP32 loss path is used and the resulting
  weighted term contributes to total-loss gradients as specified
