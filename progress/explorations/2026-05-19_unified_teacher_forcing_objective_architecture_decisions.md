---
title: Unified Teacher-Forcing Objective Architecture Decisions
date: 2026-05-19
status: discussion-decisions
scope: none-yet
---

# Unified Teacher-Forcing Objective Architecture Decisions

This note records resolved design decisions from the `grill-me-with-docs`
discussion. It is not an implementation report and does not describe current
runtime behavior.

## Supersession Note

This file preserves the chronological discussion trail. When earlier bullets use
draft names such as `typed_valid_set_marginal`, `TypedTargetAtom`,
`TargetGroup`, `struct`, `desc`, `eos`, or list-style module config examples,
the later OpenSpec change supersedes them. The active vocabulary for the
implementation plan is:

- `objective.id: teacher_forcing`
- `objective.profile: hard_sft | pure_valid_set_marginal |
  coverage_regularized_valid_set_marginal`
- `TeacherForcingTargetIR` with `SupervisionAtom`
- token roles `SCHEMA`, `TEXT`, `COORD`, and `STOP`
- `objective.modules.within_valid_coverage.coverage_strength`
- metric namespace `teacher_forcing/...`
- marginal scope `sampled_path_next_token`

## Decision

The planned objective redesign is a comprehensive architecture refactor, not a
small `ET_RMP_CE` replacement.

Resolved decisions:

1. Create a shared teacher-forcing target IR.
2. Prepare the abstraction for both Stage-1 and Stage-2 from the start.
3. Refactor the module hierarchy aggressively instead of preserving the current
   `src/detection` and `src/trainers/teacher_forcing` ownership shape.
4. Minimize backward compatibility. Preserve research/config comparability only
   where it protects explicit baselines and historical interpretation.
5. Migrate the active config surface aggressively. Old Stage-1 variant names
   such as `random_permutation_et_rmp_ce` and `prefix_rollin_et_rmp_ce` should
   stop being long-lived canonical identities.
6. Keep old ET-RMP behavior runnable only as explicit migrated comparator
   profiles under the new unified schema, not as broad aliases.
7. Make `src/objectives/` the canonical home for reusable objective math,
   target IR, objective atoms, aggregation, and diagnostics.
8. Keep `src/detection/` focused on detection data, templates, tokenization,
   residual-set branch construction, and Stage-1 adapters into the shared IR.
9. Keep `src/trainers/` focused on trainer orchestration, rollout construction,
   runtime/DDP concerns, and Stage-2 adapters into the shared IR.
10. Do not preserve old Python import paths by default. Temporary wrappers are
    acceptable only when thin, explicit, and scheduled for deletion.
11. Use one next-token `TypedTargetAtom` as the core IR unit, with separate
    `TargetGroup` metadata for semantic aggregation and provenance.
12. Use canonical mutually exclusive supervised token types:
    `struct`, `desc`, `coord`, and `eos`. Use `ignore` only for masked or
    out-of-objective positions.
13. Compute token-type loss over the full vocabulary, then compute inner
    valid-set, coverage, and hard-commitment losses over the conditional
    distribution inside the correct token type.
14. Represent candidate filtering and object coherence through a separate
    branch graph rather than fat token atoms. Token atoms should reference
    branch states/transitions by id.
15. Use a module-list config surface rooted at `objective.id:
    teacher_forcing`, with reusable target IR construction and explicit
    objective modules.
16. Name the within-valid regularizer inputs `coverage_target_weights` and
    `coverage_policy`, not `q_distribution`, so reviewers read the term as
    optional coverage pressure rather than a hidden ground-truth probability
    distribution.
17. Configure coverage through a module-level `coverage_strength`, with
    optional atom-level `coverage_strength_override` only when it is produced
    by an explicit named policy or schedule.
18. Implement token-type exclusivity as a standalone `token_type_mass` module.
    Inner modules consume conditional probabilities inside the correct token
    type but do not own or duplicate the type loss.
19. Keep the inner token likelihood unified as
    `conditional_valid_set_likelihood`. Do not create a separate
    `hard_token_commitment` loss module. Hard CE is the singleton-valid-set
    case of the same likelihood:
    `-log sum_{token in valid_token_ids} conditional_probability(token)`.
20. Use explicit atom-level `loss_tags` as the shared module-applicability
    mechanism. Target builders decide semantic truth, such as ambiguous branch,
    coordinate onset, schema exactness, and object-boundary-with-remaining.
    Module config filters may include or exclude tags for ablations, but loss
    modules should not rediscover target semantics from token ids.
21. Keep `BranchGraph` ID-only. Branch states and transitions store candidate
    ids, token labels, slots, and matched candidate ids. Full object or bbox
    payloads live in target groups or metadata and are referenced by stable ids.
22. Materialize `valid_token_ids` on every likelihood atom. Loss modules should
    consume atom-local valid sets directly; `BranchGraph` should explain,
    validate, and diagnose candidate filtering and coherence rather than being
    traversed in the runtime loss path.
23. Treat `eos` as its own mutually exclusive supervised token type everywhere
    in the shared IR. EOS/stop tokens must not be double-counted as `struct`.
24. Compute continuation-vs-EOS calibration over full-vocabulary probability
    mass, not token-type-conditional probabilities. This module intentionally
    compares valid continuation actions against EOS/stop actions across token
    types and should remain separate from conditional inner likelihood.
25. Use `candidate_uniform_aggregate_by_token` as the default
    `coverage_policy`. Coverage weights are assigned uniformly to compatible
    branch candidates first, then aggregated onto emitted token ids. A
    `token_uniform` policy may exist only as an explicit ablation.
26. Use exact coordinate tokens as the default coordinate valid-token policy.
    Optional coordinate tolerance may exist only as explicit valid-set expansion
    such as `radius_neighborhood`; it must not become Gaussian, Gibbs, IoU,
    CIoU, regression-like, or soft-coordinate CE in the new core objective.
27. Model coordinate-onset ambiguity through the general branch-transition
    mechanism, not special `x1`-only code. `x1` should be a major diagnostic
    role, but any description or coordinate slot can be a branchable atom when
    multiple candidate ids remain compatible.
28. Validate target IR strictly by default for training. Invalid atom fields,
    token-type mismatches, coverage-weight inconsistencies, and branch graph
    contradictions should fail before loss computation. A warning-collection
    mode may exist only for dataset exploration and diagnostics.
29. Use module-first aggregation for the new objective. Objective modules own
    per-atom math, while a shared reduction policy owns sample and batch
    aggregation. Semantic bucket balancing may exist only as an explicit
    reduction policy, not as hidden Stage-1 adapter logic.
30. During implementation, remove or deprecate legacy code/modules when doing
    so keeps the new hierarchy cleaner. Preserve only explicit migrated
    comparator behavior needed for research interpretation.
31. Keep diagnostics as a separate layer over shared IR plus compact module
    outputs. Objective modules compute differentiable math and may return small
    reusable statistic atoms; diagnostics own grouping, aggregation, and metric
    key emission.
32. Replace the current Stage-2 `src/trainers/teacher_forcing` objective
    pipeline as the canonical core. Shared objective logic should move to
    `src/objectives/`; Stage-2 trainer code should become thin adapter and
    wiring code only.
33. Use a one-shot, end-to-end cutover for active Stage-1 and Stage-2 objective
    code and configs. Do not introduce a long-lived compatibility wrapper
    phase. Preserve only explicit legacy comparator profiles where they are
    needed for research comparison.
34. Use `typed_valid_set_marginal` as the canonical objective family/name in
    configs and metrics. Use precise prose such as sampled-path typed valid-set
    marginal when the scope matters.
35. Rename active training metrics aggressively under
    `objective/typed_valid_set_marginal/...`. Do not preserve old
    `recursive_detection_ce/...` keys as first-class active metrics; map them
    only in legacy comparator reports when needed.
36. Do not keep ET-RMP-like endpoint shorthand as a canonical profile alias.
    The unified endpoint should be described as `coverage_strength: 1.0` under
    `typed_valid_set_marginal`. Exact old behavior remains quarantined as
    `legacy_recursive_detection_ce_exact` when historical comparison is
    required.
37. Keep exact legacy reproduction outside the clean `typed_valid_set_marginal`
    module system. `legacy_recursive_detection_ce_exact` should live as a
    quarantined comparator adapter rather than as normal modules that can be
    composed with the new objective.
38. Let `src/objectives/teacher_forcing` own target IR dataclasses, validation,
    token-type contracts, and generic builder utilities. Domain adapters such as
    Stage-1 detection and Stage-2 rollout training should construct concrete
    IR instances from their data; the shared objective package should not own
    COCO/template/rollout-specific target construction.
39. Define the `TokenTypeVocab` interface and validation in the shared objective
    package, but build concrete token-type vocabularies in adapters from the
    active tokenizer and template. Include the resulting vocab contract in each
    `TeacherForcingTargetIR`.
40. Derive `desc_token_ids` as the complement of explicit `struct`, `coord`,
    `eos`, and `ignored` token sets by default. Adapters may add tokenizer- or
    template-specific exclusions to `ignored`.
41. Represent schema/control exactness as the singleton-valid-set case of
    `conditional_valid_set_likelihood`. Do not add a separate schema CE module
    by default; use explicit reduction or module weighting if schema atoms need
    extra pressure.
42. Use one `BranchGraph` per residual-set enumeration episode, usually the
    whole assistant object-list sequence for one image/sample. Object-entry
    spans remain target groups, but branch graphs should model the residual set
    across entries rather than isolated per-entry branch structures.
43. Make `BranchState` distinguish residual enumeration from current-entry
    compatibility by storing both `remaining_candidate_ids` and
    `active_candidate_ids`. Remaining candidates drive EOS/continuation and
    future entries; active candidates drive valid tokens and within-entry
    filtering.
44. Remove a selected candidate from `remaining_candidate_ids` only at full
    object-entry completion, not when the current entry first becomes singleton
    or branch-committed. Branch commitment and emitted-object completion are
    distinct states.
45. Keep distinct candidate ids even if two candidates have identical
    serialized entries. Such cases are expected to be effectively absent in GT
    and self-rollout data, but the IR should preserve count identity: token
    branches may remain merged, while object-entry completion consumes the
    sampled teacher candidate id.
46. Require `selected_token_id` on supervised likelihood atoms, including
    ambiguous atoms. It records the teacher roll-in/control-flow token and is
    used for branch transitions and diagnostics, but it must not add selected
    hard CE at ambiguous atoms beyond the singleton-valid-set likelihood case.
47. Materialize `coverage_target_weights` for ambiguous coverage-capable atoms
    whenever the builder can define a meaningful coverage policy, even if
    `coverage_strength` is zero. Coverage weights are diagnostic data unless
    the coverage module is enabled with positive coverage strength.
48. Apply `within_valid_coverage` only to atoms tagged `coverage_candidate`.
    Ambiguity alone is not sufficient; the regularizer should mean residual
    candidate/object coverage under a named coverage policy.
49. Use constant `coverage_strength` for the initial implementation. Reserve a
    schedule-compatible config shape for later, but do not require scheduler
    plumbing in the first end-to-end refactor.
50. Enable `token_type_mass` by default in all new-framework
    `typed_valid_set_marginal` profiles. Use an explicit `no_type_loss`
    ablation when isolating the contribution of token-type exclusivity.
51. Require continuation/EOS diagnostics, but keep trainable
    `continuation_margin` default-off. A trainable margin may exist as an
    explicit ablation if cheap to implement after diagnostics. Prior discussion
    notes that pure EOS loosening has already looked unhelpful or harmful for
    decoding, so EOS intervention should be conservative and measured.
52. Treat true terminal EOS/stop positions as ordinary singleton likelihood
    atoms with token type `eos`. EOS is excluded from valid continuation atoms
    when objects remain, but it remains positively supervised when the sequence
    is actually complete.
53. Use identical core objective module names and config schema across Stage-1
    and Stage-2. Stage-specific differences belong in target IR construction,
    provenance metadata, or clearly separate non-core modules, not renamed core
    losses.
54. Shrink the Stage-2 auxiliary objective surface during the one-shot cutover.
    Delete or do not migrate duplicate unlikelihood, `coord_reg`, and
    coordinate/regression-style geometry regularizers. Also delete or do not
    migrate `bbox_*` auxiliary objective modules. Keep coordinate SoftCE only as
    an explicit ablation/comparator surface rather than core typed valid-set
    objective behavior.
55. Keep coordinate SoftCE only as a quarantined explicit ablation/comparator,
    outside the normal `typed_valid_set_marginal` module list. It should not be
    framed as core coordinate ambiguity handling in the new objective.
56. Keep the active new objective surface focused on typed valid-set core modules:
    `token_type_mass`, `conditional_valid_set_likelihood`,
    `within_valid_coverage`, and continuation/EOS diagnostics with default-off
    trainable margin. Do not migrate duplicate unlikelihood, bbox auxiliary
    losses, `coord_reg`, coordinate gates/text gates from `coord_reg`, W1,
    adjacent repulsion, or geometry regularizers into the active surface.
57. Preserve standard hard SFT as a first-class comparator profile inside the
    new `teacher_forcing` / `typed_valid_set_marginal` framework. Implement SFT as
    singleton `valid_token_ids` for every supervised atom, not as a separate
    old objective engine.
58. For hard SFT comparator profiles, still build residual-set branch graphs and
    diagnostic latent valid alternatives. The training `valid_token_ids` remain
    singleton selected tokens, while separate diagnostic valid sets expose the
    latent legal alternatives that hard SFT treats as negatives.
59. Name the diagnostic-only legal alternatives field
    `latent_valid_token_ids`. In typed valid-set profiles it usually matches
    `valid_token_ids`; in hard SFT profiles, `valid_token_ids` are singleton
    selected targets while `latent_valid_token_ids` record all semantically
    valid alternatives under the residual-set branch graph.
60. Build canonical latent target IR first, then apply explicit comparator or
    training profile transforms. Domain adapters should emit latent semantic
    truth once; profile transforms decide which tokens are trained as positives
    and which modules are enabled.
61. Treat `coverage_target_weights` as canonical latent metadata. Profile
    transforms may enable, disable, or weight the coverage module, but should
    not mutate the coverage weights except through explicitly named
    coverage-policy ablations.
62. Put profile transforms in the shared objective package, such as
    `src/objectives/teacher_forcing/profiles.py`. Domain adapters build
    canonical latent IR; config loading parses settings; shared profiles apply
    objective semantics to produce trainable valid sets, tags, and metadata.
63. Decide teacher roll-in ordering before canonical latent IR construction, as
    an adapter-level target construction policy. Roll-in policy chooses the
    teacher path; objective profile decides which latent alternatives receive
    training mass.
64. Use `rollin_policy.name: random_permutation` as the default target
    construction policy for typed valid-set profiles. Sorted/canonical roll-in
    may remain a sanity or historical comparator, but the main ablation axis
    should use random permutation.
65. Make random roll-in deterministic and reproducible by default, resampled by
    `(sample_id, epoch, global_seed)`. Provide a fixed-per-sample mode for
    tests, debugging, and exact target parity checks.
66. Run permutation-sensitivity evaluation as a separate analysis/eval job, not
    inside every training step. Training should emit cheap roll-in and latent
    valid-set summaries; multi-permutation NLL variance belongs in an explicit
    analysis workflow.
67. Split object-coherence diagnostics into target-side branch-graph coherence
    and decode-time object coherence. Branch-graph coherence belongs in IR
    validation/diagnostics; decoded object coherence belongs in eval/inference
    artifact analysis.
68. Implement sampled-path next-token valid-set marginal over teacher-forced
    roll-in prefixes, not full autoregressive recursive subtree DP. Full subtree
    marginal is out of scope for the first redesign because alternative
    branches create different autoregressive prefixes and hidden-state contexts.
69. Keep `typed_valid_set_marginal` as the concise config and metric family
    name, while recording `marginal_scope: sampled_path_next_token` and using
    `sampled-path typed valid-set marginal` in precise prose.
70. Supersede the untracked OpenSpec draft
    `openspec/changes/add-stage1-trie-marginal-objective/` rather than reusing
    it as the active change. Its Stage-1-only `recursive_detection_ce`
    framing conflicts with the approved unified
    `teacher_forcing` / `typed_valid_set_marginal` redesign. The draft files were
    removed locally after approval.
71. Create a new OpenSpec capability spec as the normative home for the
    redesign, tentatively
    `openspec/specs/teacher-forcing-typed-valid-set-marginal/spec.md`.
    Existing Stage-1 and teacher-forcing registry specs should be updated only
    as integration/supersession surfaces rather than remaining the primary
    owner of the new objective contract.
72. Avoid redundant planning artifacts. OpenSpec should own stable normative
    contracts; Superpowers should own tactical implementation planning and
    execution checklists; `progress/` should record discussion decisions and
    historical reasoning. Do not duplicate the same architecture prose across
    all three surfaces.
73. Use OpenSpec proposal/design/spec artifacts as the design and specification
    source of truth for this refactor. Use Superpowers only for the tactical
    implementation plan, with a short pointer back to the OpenSpec change
    instead of a duplicate long design spec.
74. The new OpenSpec change should rewrite stale normative portions of existing
    specs rather than leaving contradictory `SHALL` requirements until
    implementation. Historical context may remain prose, but deleted or
    demoted modules must not remain live normative requirements.
75. Defer current-behavior docs updates such as `docs/training/STAGE1_OBJECTIVE.md`
    and `docs/catalog.yaml` until implementation and smoke verification. During
    design/spec work, update OpenSpec and progress records only, unless a small
    routing pointer is explicitly needed.
76. Keep the first new-run ablation matrix minimal: initially require only
    `pure_valid_set_marginal` and
    `coverage_regularized_valid_set_marginal` with `coverage_strength: 0.1`.
    Prior hard SFT and ET-RMP experiments can serve as historical comparators
    rather than mandatory new runs in the first typed valid-set cutover.
77. Still implement `hard_sft` as a lightweight profile transform for unit
    tests and future same-codepath comparisons, but do not include it in the
    first new training-run matrix.
78. Control terminology strictly. The canonical family is
    `typed_valid_set_marginal`; the scalar for the coverage regularizer is
    `coverage_strength`; random ordering is expressed only as
    `target_ir.rollin_policy.name: random_permutation`, not in profile names.
    Canonical profile names should be short semantic labels such as
    `pure_valid_set_marginal`, `coverage_regularized_valid_set_marginal`, and
    `hard_sft`.
79. Treat `pure_valid_set_marginal` and
    `coverage_regularized_valid_set_marginal` as config preset names, not
    low-level profile transform names. The low-level training profile is
    `valid_set_marginal`; coverage strength is a module setting.
80. Keep the executable `coverage_strength` value only inside the
    `within_valid_coverage` module config. Presets may mention coverage
    strength in descriptive metadata, but there must not be a second executable
    top-level objective value.
81. Keep the `within_valid_coverage` module present in canonical
    `pure_valid_set_marginal` research configs with `coverage_strength: 0.0`.
    It must contribute zero loss but may emit coverage diagnostics when
    `coverage_target_weights` are available. Omission may remain valid for
    ultra-minimal tests.
82. Let coverage loss scale as `module.weight * coverage_strength`, but keep
    canonical configs simple: `within_valid_coverage.weight` should be `1.0`
    and users tune `coverage_strength`.

## Rationale

The current Stage-1 latest compact recursive detection stack entangles target
construction and loss semantics in `src/detection/objective.py` and
`src/detection/loss.py`. The current Stage-2 stack already has a teacher-forcing
registry under `src/trainers/teacher_forcing`, but that package is trainer-owned
and not a clean canonical home for shared objective math.

The new design should make the research objective auditable in one place:
token-type exclusivity, valid-set marginal likelihood,
coverage-strength-weighted coverage, hard branch commitment, continuation/EOS
calibration, and diagnostics should be defined once and reused by Stage-1 and
Stage-2 adapters.

## Consequence

The expected future hierarchy is a new objective-centered architecture:

```text
src/objectives/
  core/
  teacher_forcing/
  detection/

src/detection/
  data/template/tokenization/runtime/residual-set adapters

src/trainers/
  trainer loops, rollout construction, execution, and adapter wiring
```

The new config surface should express the unified objective directly rather
than relying on old variant names. A future config may use a shape like:

```yaml
objective:
  id: teacher_forcing
  family: typed_valid_set_marginal
  marginal_scope: sampled_path_next_token
  target_ir:
    name: typed_residual_set
  modules:
    - name: token_type_mass
      weight: 1.0
    - name: conditional_valid_set_likelihood
      weight: 1.0
    - name: within_valid_coverage
      weight: 1.0
      config:
        coverage_strength: 0.1
```

with explicit modules for token type supervision, conditional valid-set
likelihood, coverage, coordinate branching diagnostics, and continuation/EOS
calibration.

The controlled vocabulary is:

```text
typed_valid_set_marginal:
  canonical objective family
sampled_path_next_token:
  marginal scope
coverage_strength:
  scalar strength for within-valid coverage
rollin_policy.name:
  target-construction path policy such as random_permutation
pure_valid_set_marginal:
  config preset using profile valid_set_marginal and coverage_strength zero
coverage_regularized_valid_set_marginal:
  config preset using profile valid_set_marginal and positive coverage_strength
hard_sft:
  low-level singleton-valid-set comparator profile
valid_set_marginal:
  low-level profile where trainable valid tokens equal latent valid tokens
```

The shared target IR should be shaped around compact atoms plus explicit
structure:

```text
TeacherForcingTargetIR
  atoms: tuple[TypedTargetAtom, ...]
  groups: tuple[TargetGroup, ...]
  branch_graphs: tuple[BranchGraph, ...]
  token_type_vocab: TokenTypeVocab
  metadata: TargetIRMetadata

TypedTargetAtom
  atom_id
  position
  token_type: struct | desc | coord | eos
  selected_token_id
  valid_token_ids
  coverage_target_weights
  coverage_strength_override
  loss_tags
  branch_state_id
  selected_transition_id
  group_id
  provenance

BranchGraph
  states: branch state ids with candidate id sets and roles
  transitions: token-labeled candidate filtering transitions with matched
    candidate ids
```

The branch graph is the source of truth for candidate filtering. This keeps
token atoms small while preserving the ability to prove that coordinate tokens
belong to a coherent object branch.
The graph should not duplicate full object payloads; it should carry stable ids
that point into target groups or metadata. This makes the same graph mechanism
usable for Stage-1 GT objects and Stage-2 rollout-aligned or residual-set
hypotheses.

Likelihood atoms are the tensor-facing training contract. They should be
immediately usable by the objective runner through `valid_token_ids`,
`selected_token_id`, token type, coverage fields, and tags. Branch graphs should
be cross-checked against those atoms, for example by verifying that an atom's
valid tokens match the outgoing transition labels from its branch state and
that the selected transition token matches the selected token.

EOS is separated from `struct` so schema stability and stop-vs-continue
behavior can be diagnosed independently. Structural tokens such as object-entry
and bbox-start markers are `struct`; EOS or stop tokens are `eos` and should not
contribute to `P_struct`.

Continuation-vs-EOS calibration is the exception to the within-type inner-loss
rule. It should compare full-vocabulary valid-continuation mass with
full-vocabulary EOS mass at boundary atoms tagged
`object_boundary_with_remaining`. The default can be diagnostics-only, with a
trainable margin enabled by module weight when needed.

The default coverage policy should preserve the infinite-random-shuffle
interpretation: uniformly sampled residual candidates define the target mass,
and candidates that emit the same token aggregate their mass onto that token.
For example, two compatible `person` candidates and one `dog` candidate produce
coverage weights of `2/3` for `person` and `1/3` for `dog` when coverage is
enabled.

The new core objective should treat coordinate ambiguity as valid branch
coordinates, not as single-object coordinate uncertainty. Exact coordinate
tokens are the default. Coordinate neighborhoods, if enabled, expand
`valid_token_ids` under an explicit policy and remain a valid-set marginal. The
current IoU/Gibbs/gaussian-style coordinate soft-target machinery should be
left out of the new core and only considered later as historical or experimental
comparators if needed.

Branchability is slot-agnostic. A branchable atom references a current branch
state, valid outgoing transition tokens, a selected teacher transition, and the
next filtered state. Diagnostics should still record the first divergence role,
especially `x1`, but implementation should use the same state-transition logic
for `desc`, `x1`, `y1`, `x2`, `y2`, and any future boundary branch points.

Training should use fail-fast IR validation. Likelihood atoms require non-empty
valid tokens, selected tokens inside the valid set, token ids compatible with
the supervised token type, finite positive coverage weights when coverage is
enabled, and coherent branch references when branch tags are present. A
`warn_collect` validation mode can be useful for target-builder audits, but it
should not be the training default.

Aggregation should be visible and shared. Modules compute losses over
applicable atoms; reduction policies aggregate over atoms, samples, and batches.
If length-insensitive or semantic-bucket behavior is needed, it should be named
in config as a reduction policy. The refactor may remove legacy hidden
normalization or loss modules instead of wrapping them indefinitely, as long as
explicit baseline comparators remain recoverable under the new schema where
research comparison requires them.

Diagnostics should not bloat the loss modules. Modules can expose reusable
values such as valid-set mass, selected-token probability, type masses,
coverage KL, and continuation margin. A diagnostics layer should group those
values by token type, loss tags, branch role, coordinate slot, remaining-count,
and target group metadata, then emit the configured metric namespace.

The new `src/objectives/` runner should be the canonical teacher-forcing core,
not a wrapper around the current trainer-owned Stage-2 pipeline. Useful Stage-2
ideas such as module lists, projected atoms, provenance, and registry contexts
can be migrated into the shared package, but trainer-owned code should not
remain the owner of reusable objective semantics.

The implementation migration should be one-shot for active objective surfaces:
active Stage-1 and Stage-2 configs should move to the new schema, tests and
docs should use the new names, and old objective-core modules should be deleted
or reduced aggressively once their behavior is migrated. Compatibility should
exist only as named legacy comparator profiles, not as broad aliases or parallel
engines.

The canonical config family and metric namespace should be
`typed_valid_set_marginal`. The name captures the first-class token-type gate, the
residual-branch source of valid sets, and the valid-set marginal likelihood
without baking a particular coverage strength into the family name.

Active metrics should use the new objective namespace, for example
`objective/typed_valid_set_marginal/module/token_type_mass/loss`,
`objective/typed_valid_set_marginal/role/x1/valid_set_mass`, and
`objective/typed_valid_set_marginal/boundary/valid_vs_eos_margin`. Old metric names
that encode the previous `recursive_detection_ce` support/balance framing
should not remain canonical training keys.

Comparator naming should avoid conflating clean math with historical behavior.
The `coverage_strength: 1.0` endpoint is a unified-objective setting, not an
ET-RMP alias. `legacy_recursive_detection_ce_exact` means reproducing old
behavior as closely as possible, including historical selected-child and
reduction quirks, only when needed for run-to-run interpretation.

Exact legacy behavior should not be represented as ordinary
`typed_valid_set_marginal` modules. In particular, selected-child CE at ambiguous
nodes should not become a reusable module that can be accidentally mixed into
the new objective. If exact reproduction is needed, it should be exposed under
an explicit legacy comparator objective id and kept out of the clean runner.

Target IR construction should follow a contract/adapters split:
`src/objectives/teacher_forcing` defines `TypedTargetAtom`, `TargetGroup`,
`BranchGraph`, token-type vocab contracts, validation, and generic helpers.
Stage-1 detection and Stage-2 rollout/residual-set code instantiate those
contracts from their domain-specific data and templates.

Token-type vocabularies are adapter-built because tokenizer/template semantics
are domain-specific. The shared interface should require mutually exclusive
`struct`, `desc`, `coord`, `eos`, and `ignored` sets, expose
`token_type_of(token_id)`, and validate that atoms and valid sets are compatible
with the type contract used to build the target IR.

Description tokens should be computed as the normal-text complement rather than
maintained as a brittle explicit allowlist. Adapters should explicitly identify
schema/control tokens, coordinate tokens, EOS/stop tokens, and ignored or
never-generate tokens; the remaining vocabulary becomes `desc`.

Schema/control tokens are ordinary typed likelihood atoms with singleton valid
sets and a `schema_exact` tag. Token-type mass pushes probability into
`struct`; conditional valid-set likelihood pushes probability to the exact
struct token. Extra schema pressure, if needed, should be visible as role or
module weighting rather than hidden in a dedicated schema-loss module.

Branch graphs should model the latent-order residual-set episode. A graph can
move from all remaining candidates before entry zero, through filtered
same-entry candidate states after description or coordinate choices, to the
residual set after a selected object entry is emitted. Target groups still mark
object-entry or bbox spans for aggregation and diagnostics.

Branch states should not collapse residual-set membership and current-entry
compatibility into one set. For example, after the description token `person`,
the residual set may still be `{A, B, C}` while the active compatible entry
candidates are `{A, B}`. After selecting `x1` for object `B`, the active set
may become `{B}` while the residual set remains `{A, B, C}` until the entry is
emitted.

Candidate removal happens at object-entry completion. A singleton active set
means the current entry is branch-committed, but the selected object is still in
the residual set until the full entry is validly emitted. This keeps
boundary/EOS semantics and malformed-entry diagnostics clear.

Identical serialized candidates should not be collapsed at the candidate-id
level. If they occur, they produce merged singleton token supervision throughout
the entry and are disambiguated only at completion by the sampled teacher
identity. This preserves residual object counts without adding special token
loss behavior.

`selected_token_id` is roll-in identity, not an extra hard target while
alternatives are valid. The likelihood module uses `valid_token_ids`; if that
set is singleton, the result is hard CE naturally. Diagnostics and branch graph
transitions can still use the selected token to report selected probability and
to advance the teacher-forced branch state.

Coverage target weights should be built consistently across coverage-strength
settings so pure valid-set marginal and coverage-regularized runs differ by
module configuration rather than target shape. For singleton atoms, coverage
weights can be omitted; for ambiguous atoms with a meaningful policy, they
support diagnostics even when no coverage loss is added.

Coverage is not a generic ambiguity-balancing term. It should apply only where
the target builder can tie ambiguity to residual candidate coverage, usually
object branches or coordinate branches among compatible candidates. Other
ambiguous valid sets should use marginal likelihood without the coverage
regularizer unless explicitly justified.

Coverage-strength scheduling is a future training policy, not initial core
semantics. Initial ablations should use constant `coverage_strength` values
such as `0.0`, `0.1`, `0.25`, and `1.0`. The config can reserve a future
schedule shape, but Stage-1 should not need scheduler implementation before the
objective cutover.

Token-type loss is a core redesign component and should be default-on for all
new-framework profiles. To isolate its effect, create a named ablation with the
`token_type_mass` module disabled rather than treating type loss as optional in
ordinary configs.

Continuation diagnostics should report valid continuation mass, EOS
probability, valid-vs-EOS margin, and remaining-count slices. The trainable
margin should not be part of the default objective because previous evidence
suggests simply loosening EOS does not necessarily improve decoding and may
harm it. Any EOS intervention should be an explicit ablation after the
diagnostic behavior is visible.

Terminal stop supervision is not an EOS-loosening intervention. When no objects
remain, EOS atoms should use the normal token-type mass plus singleton
conditional valid-set likelihood. When objects remain, EOS is not part of the
valid continuation set and is only compared through continuation diagnostics or
an explicit default-off margin.

The shared core module vocabulary is `token_type_mass`,
`conditional_valid_set_likelihood`, `within_valid_coverage`, and optional
`continuation_margin`. Stage-1 and Stage-2 should use those names. Rollout,
matching, pseudo-target, or auxiliary geometry differences should be expressed
through IR provenance, adapter-built target groups, or separate non-core
modules.

The current Stage-2 registry names include `loss_duplicate_burst_unlikelihood`,
`bbox_geo`, `bbox_size_aux`, and `coord_reg`. The cutover should not preserve
the stale auxiliary surface. Duplicate unlikelihood, `coord_reg`, bbox
auxiliary modules, and regression-like coordinate regularizers should be
removed or quarantined rather than migrated. SoftCE ablations need separate
explicit names so they cannot be confused with typed valid-set core losses.

SoftCE should be retained only under an explicit comparator or ablation name
such as `legacy_coord_soft_ce_ablation` or
`coord_distribution_matching_ablation`. It should not appear in the ordinary
typed valid-set core module list, because it represents coordinate distribution
matching rather than valid-branch marginal likelihood.

The active objective surface should stay intentionally narrow so the first
experiments test the core research claim: token-type exclusivity plus
residual-set valid-branch likelihood and optional coverage. Legacy SoftCE and
exact legacy recursive CE may remain comparator paths; other auxiliary losses
should be removed or left behind.

Hard SFT should be represented by the same shared IR and runner. With
`token_type_mass` weight `1.0` and `conditional_valid_set_likelihood` weight
`1.0`, a singleton selected-token valid set decomposes ordinary full-vocabulary
hard CE into type mass plus conditional within-type likelihood. The primary SFT
comparator should be `hard_sft` with `rollin_policy.name: random_permutation`;
sorted-order SFT can remain a sanity or historical comparator when useful.

SFT target IR should retain the same branch graph used by typed valid-set profiles.
For SFT, `valid_token_ids` are singleton training targets, while diagnostic
latent valid sets come from the branch graph. This allows reports such as
latent valid-set mass, other-latent-valid mass, and false-negative mass without
changing the SFT loss.

Use `latent_valid_token_ids` for the legal-alternative field so the name
captures research semantics rather than only diagnostic usage. It supports
false-negative diagnostics for SFT and consistency checks for typed valid-set
profiles.

The target pipeline should separate canonical latent truth from experimental
profile semantics. Builders produce latent valid alternatives, selected
roll-in tokens, branch graphs, groups, and coverage weights. Profile transforms
then produce the trainable `valid_token_ids`, tags, and module settings for
typed valid-set marginal, hard SFT, no-type-loss, and related profiles.

Coverage weights describe the canonical residual-candidate coverage policy.
Keeping them stable across profiles lets coverage-strength sweeps and SFT
false-negative diagnostics compare against the same latent target. Alternative
policies such as token-uniform coverage should be explicit named ablations, not
silent profile mutations.

Profile transforms are shared objective semantics and should not live in
Stage-1 or Stage-2 adapters. They should transform canonical latent IR into
profile-specific training IR for hard SFT, typed valid-set marginal, and
related ablations.

Roll-in ordering is distinct from objective semantics. The adapter should apply
policies such as random permutation or sorted order before building canonical
latent IR, because selected tokens and selected transitions depend on the
teacher path. Convenience presets may combine roll-in and objective profile,
but the internal config should keep them separate.

Typed valid-set profiles should use `rollin_policy.name: random_permutation` by
default so branch commitment, coordinate filtering, and residual-set states are
sampled across valid object orders. The selected roll-in path remains control
flow; ambiguous-node loss still uses valid-set marginal rather than
selected-token hard CE.

Roll-in metadata should be recorded in the target IR or batch metadata:
roll-in policy name, seed, epoch, and selected candidate order or permutation
identifier. The default should resample per epoch deterministically; fixed
roll-in should exist for reproducible tests and target snapshots.

Permutation-sensitivity analysis should reuse the same IR builders and profile
transforms to evaluate multiple roll-in permutations for fixed images. It
should report NLL mean/std/min/max and latent-set diagnostics, but it should not
increase ordinary training-step cost.

Branch-graph coherence verifies that target construction does not allow
coordinate mixing along the teacher-forced path. Decoded object coherence checks
free-generation outputs for mixed coordinates, duplicates, and missed objects.
They should share candidate/coherence concepts but remain separate diagnostic
surfaces.

The core objective is a sampled-path next-token valid-set marginal over
teacher-forced roll-in prefixes. It is not a full recursive subtree marginal
over all alternative prefixes. Full subtree marginal would require evaluating
model probabilities under alternative prefixes, because each autoregressive
branch creates a different hidden-state context.

The short family name remains `typed_valid_set_marginal`, but configs/specs should
carry the precise scope `sampled_path_next_token`. This keeps metric keys short
while preventing confusion with full autoregressive subtree marginal DP.

The previous untracked OpenSpec change for `add-stage1-trie-marginal-objective`
is treated as superseded source material only. Future OpenSpec work should use
a fresh change, likely `refactor-teacher-forcing-typed-valid-set-marginal`, and
may copy only the still-valid conceptual language about valid alternatives,
coordinate-onset ambiguity, candidate filtering, and EOS diagnostics.

The new OpenSpec owner should be a dedicated
`teacher-forcing-typed-valid-set-marginal` capability. It should own the shared
target IR, token-type vocabulary, branch graph, canonical latent IR, profile
transforms, typed valid-set modules, sampled-path marginal scope, comparator
profiles, and metric namespace. Existing specs should point to or defer to this
new capability where they previously embedded stale Stage-1-only or Stage-2
registry objective semantics.

OpenSpec and Superpowers should be complementary rather than overlapping.
OpenSpec should define stable behavior, config/schema contracts, metric
semantics, and deletion/quarantine commitments. Superpowers planning should
translate that accepted contract into file ordering, test slices, migration
phases, and smoke commands. The progress note remains the compact discussion
record while the design is still being grilled.

When artifact writing begins, the OpenSpec change should carry the architecture
and normative requirements. A Superpowers plan may execute it, but should not
restate the whole design. If a Superpowers design bridge is required by
workflow, keep it short and point to the OpenSpec change as source of truth.

Existing specs such as `teacher-forcing-unified-loss-registry` should be
updated in the OpenSpec change to remove or supersede stale active requirements
for duplicate unlikelihood, `coord_reg`, bbox/geo auxiliary modules, coord/text
gates, and conflicting EOS handling. The accepted contract should not contain
contradictory normative requirements while the implementation plan is being
executed.

Current-behavior docs should not claim the future objective before it exists.
OpenSpec owns the future contract during planning. Training docs, catalog
canonical config paths, metrics docs, and runbooks should update after the code
path resolves and at least smoke verification shows the new behavior is real.

The initial experiment launch surface should focus on the new research delta:
pure sampled-path valid-set marginal and the intended small-coverage hybrid.
Hard SFT and old ET-RMP evidence already exists from previous experiments, so
they do not need to be part of the first new-run matrix unless later comparison
requires same-codepath reproduction.

The `hard_sft` profile remains valuable for tests because it sets
`valid_token_ids` to singleton selected tokens while preserving
`latent_valid_token_ids` for false-negative diagnostics. This profile can prove
the type-plus-conditional decomposition recovers hard CE without requiring an
immediate new training run.

The `token_type_mass` module owns the full-vocabulary type-exclusivity loss and
its diagnostics. Valid-set marginal, within-valid coverage, and hard
commitment modules should operate on conditional probabilities normalized
inside the atom's supervised token type. This makes type supervision ablatable,
keeps diagnostics single-sourced, and avoids burying token-type pressure inside
each objective atom.

`conditional_valid_set_likelihood` owns both ambiguous-prefix support mass and
committed singleton-token supervision. When `valid_token_ids` contains one
token, it is ordinary hard CE inside the supervised token type. When it
contains multiple tokens, it is the valid-set marginal. This avoids a separate
hard-commitment module that could accidentally fire on ambiguous atoms and
reintroduce selected-path false negatives.

Coverage strength has one executable source of truth: the
`within_valid_coverage` module config. Preset names and experiment metadata can
summarize it for readability, but config loading should not accept parallel
top-level coverage-strength fields that could drift from the module setting.

The canonical pure preset should keep the coverage module in the module list
with zero strength so coverage diagnostics align across pure and
coverage-regularized runs. This should not add a gradient term.

Module weight remains part of the generic objective runner contract for all
modules. For coverage, `coverage_strength` is the semantic regularization knob.
Canonical presets should not tune both simultaneously.

Atom-level `loss_tags` are the shared applicability contract. For example,
`conditional_valid_set_likelihood` applies to atoms tagged `likelihood`,
`within_valid_coverage` applies to atoms tagged `coverage_candidate`, and a
future continuation module applies to atoms tagged
`object_boundary_with_remaining`. Config filters may include or exclude tags
for ablations, but the builder remains responsible for assigning the semantic
tags.

`token_type_mass` should start with a uniform module weight of 1.0 across
schema, text, coordinate, and stop/control roles. Per-role type weights are not
part of the first implementation surface. The first implementation should make
wrong-type mass measurable before adding knobs that can obscure whether the
basic type gate is doing its job.

The upstream integration seam is approved: use a CoordExp-owned
sidecar-aware `compute_loss` mixin plus a shared teacher-forcing objective
runner, not the plain upstream `compute_loss_func` as the core path. The
upstream `compute_loss_func` hook is useful for labels-only losses, but is too
narrow for target-IR sidecars because it receives model outputs and labels
after the model forward rather than owning sidecar stripping before
`model(**inputs)`.

The shared runner should preserve upstream ownership of launcher, model and
template loading, Qwen-VL preprocessing, LoRA/tuner plumbing, distributed
training mechanics, optimizer/scheduler/checkpointing, and callback/log timing.
CoordExp should own the target IR, batch sidecars, non-model-field stripping,
full-logit assertions, causal target-position convention, typed valid-set
objective math, diagnostics, and metric atoms.

The first upstream-aligned trainer contract should keep the standard
`compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None)`
signature. It should pop/stash target IR sidecars, strip non-model fields, pop
`labels` when the custom objective owns token supervision, reject or remove
`logits_to_keep` for full-vocabulary objectives, run `model(**model_inputs)`,
assert unsliced `[batch, seq, vocab]` logits, and then call the shared
teacher-forcing objective runner. Additive label-only losses may still use the
ordinary upstream path, but the typed valid-set objective should not.

The implementation should reuse the existing CoordExp batch-extra pattern
rather than inventing a second sidecar transport. Any target IR sidecar must be
preserved through collation with `remove_unused_columns=false`, then removed
before the model forward. Sidecar fields must never be forwarded into the
Qwen-VL model.

Generation/eval paths are a separate concern. Hugging Face Seq2Seq generation
evaluation may bypass `compute_loss`, so teacher-forced objective diagnostics
and rollout/generation metrics should remain separate until an explicit eval
adapter is designed.

Initial safety settings for the new target IR should be unpacked, non
padding-free, and full logits. Packing and `logits_to_keep` remapping are
second-stage features after the un-packed target-position convention is tested.

The implementation sequence should extract the Stage-2 teacher-forcing runner
first, then integrate Stage-1 recursive detection into the same runner. Stage-2
already has explicit pipeline manifests, objective atoms, forward helpers, and
duplicated runner code, so it is the lowest-risk place to prove the shared
execution interface with parity tests. Stage-1 recursive detection should join
after that interface is real, using the new shared target IR rather than
forcing the runner to be proven first on the most legacy sidecar path.

The shared runner should expose a sidecar/IR-first interface rather than a
generic `(outputs, labels)` loss-function interface. The request object should
carry model outputs or logits, input ids, labels for validation/metrics,
attention masks, the shared target IR, the resolved pipeline manifest,
objective context, `num_items_in_batch`, and train/eval mode. The result object
should carry the scalar loss, per-module losses, metric atoms, diagnostics, and
objective atoms.

Trainer-owned responsibilities remain outside the runner: model forward,
batch-sidecar popping, non-model-field stripping, DDP/log timing, pending
metric buffering, Hugging Face/ms-swift tuple-shape compatibility, and any
trainer-specific rollout orchestration. Runner-owned responsibilities include
full-logit shape assertions, causal target-position validation, target IR
validation, module dispatch, loss reduction, diagnostic atom construction, and
canonical objective-level metric payload construction.

This interface should intentionally not look like a plain CE loss function.
The research object is target-IR-driven teacher forcing, and hiding that behind
`(outputs, labels)` would push the hard semantics back into trainers.

Teacher-forcing position alignment is a high-risk correctness contract and
must be explicit rather than inferred inside modules. Each atom should carry
both `logit_position` and `target_position`, with the initial ordinary causal
LM invariant `target_position == logit_position + 1`. The runner should
validate that the selected token, when present, matches
`input_ids[batch_index, target_position]`, that supervised labels are
compatible with the selected-path target when labels are relevant, that the
logit row has a valid attention context, and that logits are unsliced against
the original input sequence.

Loss modules must not shift labels internally. Any causal shift should happen
exactly once in the target builder or runner validation layer, and tests should
include synthetic off-by-one traps where only the correct logit row gives the
expected loss.

The shared forward contract must stay compatible with flash-attention and
Qwen-VL packed-position metadata. Existing behavior that combines
`text_position_ids` with Qwen 3-row `position_ids` into 4-row position ids must
remain trainer/forward-helper owned, and the runner should consume already
validated logits rather than rewriting attention metadata. Initial typed-target
IR support should remain unpacked/non-padding-free; later packing support must
rewrite atom positions and validate them against packed sequence metadata.

The design should leave a future extension point for reusable forward
artifacts, including embeddings when the active model/template path exposes
them cleanly. This is not a first implementation requirement. The first
contract should allow an optional artifact container to be threaded through the
runner without requiring loss modules to depend on cached embeddings. Any such
reuse must preserve autograd correctness and flash-attention position semantics.

The first implementation of the typed valid-set target IR should explicitly
forbid packed and padding-free examples. It should require one dataset sample
per training sequence, full logits with shape `[batch, seq, vocab]`, and atom
positions that index directly into the un-packed `input_ids` and logits.
Packed-sequence atom rewriting, padding-free position remapping, and
`logits_to_keep` remapping are second-stage adapters with their own tests, not
part of the first correctness surface.

The canonical batch sidecar key for the shared target IR should be
`teacher_forcing_target_ir`. It should be registered in the shared batch-extra
surface as a non-model field that must be removed before `model(**inputs)`.
The legacy `recursive_detection_targets` sidecar is not the new shared
contract. It may be used only as an adapter input during migration, or replaced
directly by a detection builder that emits `TeacherForcingTargetIR`.

This naming is intentional: the new framework should not appear to be
recursive detection CE with a renamed loss. Stage-1 recursive detection is one
builder/adapter into the shared teacher-forcing target IR, not the owner of the
shared objective contract.

The final merged implementation should completely remove the active legacy
recursive-detection objective APIs once the new target IR path covers their
responsibilities. `recursive_detection_targets`, `RecursiveDetectionTargets`,
`TokenTarget`, `RecursiveDetectionCEMixin`, `RecursiveDetectionTargetsEnricher`,
`recursive_detection_ce` active config naming, and
`compute_recursive_detection_ce_batch_loss` should not remain active training
APIs. Temporary migration code may exist within the branch, but the final state
should expose the shared teacher-forcing target IR and runner instead.

Useful detection serialization knowledge should be moved or rewritten under the
new target IR builder: template alignment, object-entry spans, coordinate slot
metadata, object instance ids, candidate filtering, and tokenized assistant
span alignment. The old sidecar and monolithic loss function should not be
kept as permanent comparator APIs merely for backward compatibility.

The new IR should deliberately avoid old `TokenTarget` field names when the old
names encode legacy semantics. Use `target_position` instead of `position`,
`logit_position` as the explicit causal predictor row, `selected_token_id`
instead of `teacher_token_id`, `valid_token_ids` and
`coverage_target_weights` instead of `trie_branch_targets` and
`child_probabilities`, and `coord_role` instead of `coord_slot_name`. Avoid a
new `kind = trie_multi_positive` field. Ambiguity and coverage should be
expressed through `valid_token_ids`, `coverage_target_weights`, and
`loss_tags`.

Supervision atoms should store `token_role`, not atom-local token-type id
sets. The actual vocabulary partitions should be owned by a central
`TokenTypeVocabulary` or equivalent object. `token_type_mass` should resolve
role ids from that object, e.g. `ids_for(TokenRole.COORD)`, rather than reading
`type_gate_token_ids` from every atom. Token-type groups are global vocabulary
facts, and central ownership avoids bloated sidecars and inconsistent role-id
sets across atoms.

EOS/stop should be represented as its own supervised role,
`TokenRole.STOP`, rather than folded into schema/control. Schema/control tokens
and stop decisions answer different research questions. Keeping stop separate
makes type diagnostics and continuation diagnostics cleaner: `P_schema`,
`P_text`, `P_coord`, `P_stop`, valid-continuation mass, stop probability, and
continue-vs-stop margin can be reported without disentangling EOS from general
structural-token behavior.

Object-boundary continuation and EOS calibration should use the same
`SupervisionAtom` type with semantic loss tags rather than a separate
continuation atom class. For example, a boundary atom whose valid continuation
is `<|object_ref_start|>` can be tagged with `LIKELIHOOD`,
`OBJECT_BOUNDARY_WITH_REMAINING`, and `CONTINUATION_CANDIDATE`. Continuation
diagnostics or margin loss can compare valid continuation mass against a
competing stop token referenced by diagnostic context or a small optional
field, without creating a second atom hierarchy.

`valid_token_ids` should always be deduplicated vocabulary token ids. Candidate
or object multiplicity must not duplicate a token inside the valid set, because
valid-set likelihood is a probability mass over vocabulary tokens. Multiplicity
belongs in `coverage_target_weights` and diagnostic context. For example, if
one residual object maps to `left` and two residual objects map to `right`,
the atom should contain `valid_token_ids=[left_id, right_id]` and
`coverage_target_weights=[1.0, 2.0]`.

Coordinate tolerance, if enabled later, should be represented by expanding the
deduplicated valid set into a valid-neighborhood token set, not by introducing
a coordinate soft-CE distribution. With radius 1 around coordinate candidates
120 and 640, the atom's valid coordinate set would contain coordinate tokens
for 119, 120, 121, 639, 640, and 641, and the marginal module would still
compute `-log sum p(valid_token_ids)`. The first implementation should keep
`coordinate_neighborhood_radius=0`; nonzero radius is a later tested extension
only if it remains valid-set semantics.

The final metric namespace should use `teacher_forcing/...` as the canonical
prefix, with module or diagnostic names below it. Examples include
`teacher_forcing/loss/total`, `teacher_forcing/loss/token_type_mass`,
`teacher_forcing/loss/conditional_valid_set_likelihood`,
`teacher_forcing/loss/within_valid_coverage`, `teacher_forcing/type_mass/P_coord`,
`teacher_forcing/valid_set/mass`,
`teacher_forcing/coverage/within_valid_entropy`, and
`teacher_forcing/continuation/continue_vs_stop_margin`. The objective family
is carried by config and run metadata, so metric keys should not repeat
`typed_valid_set_marginal` everywhere.

Legacy metric keys such as `recursive_detection_ce/...` and
`loss/recursive_detection_ce` should not remain active final keys. Temporary
parity logs may exist within the migration branch, but the final merged
implementation should publish the new namespace.

Runner validation should fail fast for target IR invariants in the first
implementation. Hard failures include off-by-one causal-position violations,
out-of-range positions, selected-token mismatch against `input_ids`,
empty/duplicate/out-of-vocabulary valid sets, selected token missing from the
valid set, malformed coverage weights, missing or inconsistent token/coord
roles, packed or padding-free metadata on an unsupported path, and logits shape
mismatch against `input_ids`. The runner should not silently repair or drop
invalid atoms.

Masking is allowed only when it is explicit semantic intent, such as a module
filter excluding a tag, a diagnostic-only atom, or an atom without the tag a
module consumes. Invalid target construction is a correctness bug, not a
training example to skip silently.

Token-position diagnostics should use normal `SupervisionAtom`s tagged with
`DIAGNOSTIC_ONLY` and any relevant probe tags. Such atoms may report valid-set
mass, selected-token probability, same-description legal alternative mass,
continuation-vs-stop margin, or wrong-type mass at a concrete causal position,
but they must contribute no gradient. Sample-level or decode-level diagnostics
that are not naturally tied to one token position should live in
`TeacherForcingTargetIR.sample_diagnostics` or in separate eval artifacts.

`coverage_target_weights` should be unnormalized nonnegative masses aligned
with deduplicated `valid_token_ids`. The `within_valid_coverage` module owns
normalization to a distribution over the valid set. Validation should require
finite nonnegative weights, matching length, and positive total mass for atoms
consumed by coverage, but builders should not be required to pre-normalize.

`selected_token_id` should be present on all non-diagnostic supervision atoms
as teacher-forced provenance, even when the likelihood loss is valid-set
marginal and must not use the selected token as hard CE. The selected token is
needed for causal-position validation, candidate-filtering provenance,
selected-token probability diagnostics, alternative-mass diagnostics, sampled
roll-in reconstruction, and the `hard_sft` comparator transform. Its presence
must not imply selected-token CE at ambiguous atoms.

For every non-diagnostic atom, `selected_token_id` must belong to
`valid_token_ids`. If the teacher-forced selected token is not valid, then the
valid-set builder, roll-in path, or causal-position alignment is wrong and the
runner should fail fast. Only explicit diagnostic-only probes may compare
against an illegal or competing token outside the valid set.

The live batch `input_ids` tensor should remain the source of truth for target
alignment. The target IR should not duplicate the full input sequence by
default. It should carry target positions, selected token ids, sample ids, and
optionally a lightweight sequence fingerprint such as sequence length plus the
selected position/token pairs. The runner validates the IR against live
`input_ids` at execution time.

Every supervision atom should carry an explicit `loss_weight` with default
`1.0`. Module reduction should use a weighted mean over consumed non-diagnostic
atoms, `sum(loss_i * loss_weight_i) / sum(loss_weight_i)`, and the runner
should combine module losses through module weights. Diagnostic-only atoms must
not contribute gradients regardless of weight. Do not reintroduce old
state-weighting policies in stage 1; later weighting policies should transform
atom weights explicitly and report diagnostics.

Stage 1 should use plain atom-weighted means with default `loss_weight=1.0`
for normal atoms. Sample-balanced, object-balanced, role-balanced, or
state-exposure normalization should be deferred to explicit future atom-weight
transforms, not hidden inside the first implementation. The IR and runner
should leave this extension point open, but first-run configs should avoid the
extra weighting knob.

Objective profiles should be transforms over canonical latent IR rather than
separate target builders. The builder should emit the richest truthful atom
with all legal `valid_token_ids`, the sampled `selected_token_id`,
`coverage_target_weights`, semantic tags, and diagnostic context. Profiles such
as `hard_sft`, `pure_valid_set_marginal`, and
`coverage_regularized_valid_set_marginal` should transform or filter how
modules consume the same latent truth. For example, `hard_sft` can replace the
training valid set with `[selected_token_id]`, while preserving latent
alternatives for diagnostics; pure marginal keeps the valid set and sets
coverage strength to zero; coverage-regularized marginal keeps both valid set
and coverage masses.

Every transformed atom should preserve canonical alternatives in formal latent
fields. `valid_token_ids` is the active training valid set after profile
transform. `latent_valid_token_ids` is the canonical legal set before profile
transform. `coverage_target_weights` is the active coverage mass vector when
coverage applies, while optional `latent_coverage_target_weights` may preserve
the canonical coverage masses for diagnostics. This lets hard SFT diagnostics
still measure legal alternative mass and false-negative pressure.

Profile transforms should be pure functions that return a new transformed
`TeacherForcingTargetIR` without mutating the canonical latent IR. The intended
flow is `canonical_ir = builder.build(...)`, then
`active_ir = profile_transform.apply(canonical_ir, profile_config)`, then the
runner consumes `active_ir`. The transformed IR should retain profile
provenance so tests and diagnostics can compare canonical truth against the
active training surface.

Profile transforms should run in the trainer/runner layer, not in the dataset
builder. The `teacher_forcing_target_ir` sidecar should carry canonical latent
IR so the same prepared sample can support hard SFT, pure valid-set marginal,
hybrid coverage, and diagnostic probes without rebuilding or invalidating
caches. The active profile is selected by run config and applied just before
runner execution.

`coverage_strength` should remain executable only inside the
`within_valid_coverage` module config. Profile names and preset YAMLs may
select or summarize that setting, but config loading should not accept parallel
top-level coverage-strength fields. This keeps the active coverage value in one
place and avoids profile/module drift.

Profile names should be preset labels and metadata only, never parsed for
behavior. Execution should read resolved module/profile-transform config, not
substrings such as `alpha0`, `coverage`, or `random_rollin` from a profile
name. Presets may expand to explicit config, but the profile string itself
should not encode executable semantics.

Roll-in policy belongs to target IR builder config and metadata, not objective
profile semantics. It changes the teacher-forced prefix and therefore the
canonical latent IR itself. The config should live under a target-builder
surface such as `target_ir.rollin_policy.name: random_permutation`, and the IR
should record roll-in metadata such as policy name, seed, emitted object ids,
and remaining object ids. Objective profiles then decide how to train on the
atoms produced under that roll-in world.

Candidate filtering state should be represented as a sample-level branch trace
with compact per-atom references. The IR may contain a `BranchTrace` with
`BranchTraceStep`s recording selected token, token/coord role,
candidates-before, candidates-after, and branch-step ids. Atoms should carry a
`branch_step_id` or compact diagnostic counts rather than duplicating full
candidate lists. This supports auditability and object-coherence diagnostics
without bloating every atom.

Branch traces and atoms should use stable internal `object_instance_id`s as the
primary object identity. Original source indices or record ids should be kept
as metadata for audit, but loss modules should not depend on them. This keeps
identity stable across normalization, ordering, repeated descriptions, and
future augmentation while preserving links back to source objects for
diagnostics.

Description ambiguity should be decided from tokenized description paths, not
from raw or normalized string equality. The objective must follow next-token
reality under the tokenizer. Normalized description strings, original text, and
same-description group ids should be retained only for diagnostics and audits.

Schema/control tokens should be hard singleton atoms by default. Structural
tokens such as `<|im_start|>`, `<|object_ref_start|>`, `<|box_start|>`, and
separators should normally have `valid_token_ids=[selected_token_id]` under
`TokenRole.SCHEMA`. Ambiguous valid sets are reserved for object-entry text,
coordinate branch choices, and explicit continuation-vs-stop comparisons.
Broad schema ambiguity such as "any structural token is fine" is out of scope
for the first implementation because schema stability is a primary goal.

For Qwen3-VL chat training, `<|im_end|>` is the only semantic EOS/STOP token.
It should be the only token in `TokenRole.STOP`, the only positive stop target,
and the stop token used by continuation diagnostics. The text-level terminator
token used for padding must remain outside STOP/EOS supervision and outside
description tokens. The intended padding token is `<|endoftext|>`, and only
`<|im_end|>` is EOS. Implementation should resolve and validate both tokens
from the active tokenizer instead of guessing inside objective modules.

The checked Qwen3-VL coordexp tokenizer files confirm the relevant special
tokens. In the base tokenizer under
`model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`,
`special_tokens_map.json` and `tokenizer_config.json` set
`eos_token="<|im_end|>"` and `pad_token="<|endoftext|>"`. The important token
ids resolve as single tokens: `<|endoftext|>` 151643, `<|im_start|>` 151644,
`<|im_end|>` 151645, `<|object_ref_start|>` 151646,
`<|object_ref_end|>` 151647, `<|box_start|>` 151648, `<|box_end|>` 151649,
`<|quad_start|>` 151650, `<|quad_end|>` 151651, `<|vision_start|>` 151652,
`<|vision_end|>` 151653, `<|vision_pad|>` 151654, `<|image_pad|>` 151655,
`<|video_pad|>` 151656, `<|coord_*|>` 151669, `<|coord_0|>` 151670, and
`<|coord_999|>` 152669.

For first-stage typed teacher forcing, schema vocabulary should contain only
structural tokens the assistant detection payload may be supervised to emit,
such as `<|object_ref_start|>`, `<|box_start|>`, and explicit row separators.
Unused or currently unsupervised special/control tokens such as
`<|im_start|>`, `<|object_ref_end|>`, `<|box_end|>`, `<|quad_start|>`,
`<|quad_end|>`, `<|vision_start|>`, `<|vision_end|>`, `<|vision_pad|>`,
`<|image_pad|>`, `<|video_pad|>`, and `<|coord_*|>` should be excluded control
unless a concrete supervised target surface later uses them. `<|im_end|>` is
STOP only, and `<|endoftext|>` is PAD/excluded control only.

`TokenTypeVocabulary` should be built from an explicit active payload schema
token set rather than by treating all structural-looking special tokens as
schema. The compact payload schema is deliberately small. Tokens like
`<|image_pad|>`, `<|object_ref_end|>`, `<|box_end|>`, and `<|coord_*|>` must not
be valid schema merely because they exist in the tokenizer.

Active payload schema token sets should be declared by serialization-format
adapters, not free-form YAML token lists. Config may select a payload format
such as `compact_full`, but the format adapter owns which schema tokens are
valid for that format. Schema tokens are part of serialization semantics rather
than a tuning knob.

Stage 1 of the typed valid-set objective should support only
`payload_format: compact_full`. Older coordjson/dense formats and other compact
variants should not share the first implementation path. Unsupported formats
should fail fast until a dedicated payload adapter with its own schema tokens,
span builder, and tests is added.

The first implementation should explicitly reject compact variants such as
`compact_no_desc`, `compact_no_bbox`, and `compact_min`. They change or remove
the text and coordinate ambiguity surfaces that the typed valid-set objective
is meant to test. They may become later adapters or ablations, but not implicit
stage-1 inputs.

Compact row separators/newlines should be removed from the new
`compact_full` typed-objective serialization rather than represented as schema
atoms. The two compact schema markers, `<|object_ref_start|>` and
`<|box_start|>`, should be sufficient to delimit entries and fields. This is a
serialization/template/parser contract change, not only a token-type-vocabulary
change: current code joins compact rows with `"\n"` and parses by splitting on
`"\n"`, so the implementation must update rendering, parsing, template span
projection, target IR construction, prompt examples if necessary, and tests to
use marker-delimited adjacent compact entries.

Do not keep newline in the teacher-forced sequence as a masked or
diagnostic-only formatting token for stage 1. If the newline stays in the
serialized target, it still becomes part of the autoregressive context and
would complicate position alignment and decoding semantics. The cleaner
contract is no row-separator token in the new compact_full target surface.

Qwen chat EOS/PAD handling should be centralized in a shared token contract
used by both training and inference. The contract should resolve
`stop_token_text="<|im_end|>"`, `pad_token_text="<|endoftext|>"`, their token
ids, and enforce that they are distinct single tokenizer tokens. STOP role
contains only `<|im_end|>`; the pad token is excluded from all supervised
schema/text/coord/stop groups; generation uses `<|im_end|>` for
`eos_token_id` and `<|endoftext|>` for `pad_token_id`; training targets never
use the pad token as EOS.

The shared Qwen chat EOS/PAD contract should live in `src/common`, e.g.
`src/common/qwen_chat_tokens.py`, because inference, training, artifact
metadata, and objective builders all need the same token resolution. The
teacher-forcing token-role vocabulary should live under
`src/objectives/teacher_forcing/token_types.py` because schema/text/coord/stop
partitioning is objective-specific. Objective token-type code may import the
common Qwen token contract; common code should not import objective modules.

`src/common/qwen_generation.py` should be replaced by the new shared Qwen chat
contract module in the final state. Its current responsibilities are broader
than generation and include EOS/PAD resolution plus Qwen processor geometry
helpers. The implementation should update imports to the new module and delete
`qwen_generation.py` if the usage surface is manageable. A temporary wrapper is
acceptable only during migration, not as a permanent active API.

Qwen chat token contract and Qwen processor geometry helpers should be split.
`src/common/qwen_chat_tokens.py` should own EOS/PAD contract resolution and
generation-token application. `src/common/qwen_geometry.py` should own
`do_resize=false` processor helper functions. This keeps token semantics and
image-geometry invariants separate while still making both reusable by training
and inference.

`TokenTypeVocabulary` should be derived from the active tokenizer and Qwen chat
token contract at runtime, not stored inside every `TeacherForcingTargetIR`.
The IR stores `allowed_token_roles` on atoms, normally as a singleton role set,
and may carry a lightweight tokenizer/token contract fingerprint, but the full
schema/text/coord/stop id sets are runtime-owned and reused by the objective
modules.

Runner validation should require every atom's `valid_token_ids` to be a subset
of the union of `TokenTypeVocabulary.ids_for(role)` for
`atom.allowed_token_roles`. This prevents self-contradictory losses such as a
coordinate-only allowed role set paired with text valid tokens while still
allowing rare mixed-role ambiguity at trie prefix boundaries. STOP atoms should
use `<|im_end|>` only; schema atoms should exclude STOP unless STOP is
explicitly allowed in the role set; text atoms should exclude
coordinate/schema/stop/pad/control ids; coord atoms should contain only
coordinate tokens.

`TokenRole.TEXT` should represent all normal free-text vocabulary ids, not only
tokens observed in the current object descriptions. It should be computed as
the complement of reserved schema/control, coordinate, stop, pad, unknown, and
special/image/control token ids. The description trie or valid-set builder then
chooses the prefix-specific valid text continuation set. This keeps
type-exclusivity separate from target validity.

The shared target IR should be atom-centered. The top-level dataclass should be
`TeacherForcingTargetIR`, and its central payload should be
`supervision_atoms: list[SupervisionAtom]`. Each `SupervisionAtom` represents
one causal next-token supervision event after roll-in has selected the teacher
prefix.

Each atom should carry at least: `batch_index`, `logit_position`,
`target_position`, `allowed_token_roles`, `selected_token_role`, optional
`coord_role`, `valid_token_ids`, optional `selected_token_id`,
`latent_valid_token_ids`, optional `coverage_target_weights`, `loss_tags`,
optional branch/candidate state, and diagnostic context. The exact field names
can be refined during implementation, but the abstraction should stay flat at
loss-module time.

Trie/residual-object machinery is a builder concern rather than a module
concern. Builders may use tries, residual object sets, coordinate-onset
candidate filtering, and deterministic random-permutation roll-in. Loss
modules should consume atoms that already state the allowed supervised token
role set, the valid token set, the selected teacher token if relevant, coverage
weights, and diagnostic metadata.

This makes Stage-1 and Stage-2 share the loss modules even if they construct
atoms through different builders.

Branch and candidate-object state should be diagnostic-only metadata after
atom construction. Candidate filtering, branch commitment, coordinate-onset
ambiguity handling, selected teacher branch choice, valid-token construction,
and loss-tag assignment are builder responsibilities. Loss modules should not
inspect object identities to decide whether an atom is ambiguous or committed.

Loss behavior should be driven by explicit atom fields such as
`allowed_token_roles`, `selected_token_role`, `coord_role`, `valid_token_ids`,
`coverage_target_weights`, and `loss_tags`.
Branch metadata may record same-description group id, candidate object ids,
selected object id, first divergence role, remaining object count, and
coordinate branch depth for diagnostics and audits, but gradient calculation
must not require reconstructing object tries.

Mixed-role atoms are required for shared trie prefix cases where a valid
description can either end at a schema marker or continue as text. COCO-80 has a
concrete case: `car` tokenizes as `["car"]`, while `carrot` tokenizes as
`["car", "rot"]`; images containing both require `<|box_start|>` and `rot` to
be valid after the prefix `<|object_ref_start|>car`. The design should support
many such shared-prefix cases for future larger or open-vocabulary datasets,
not just this single COCO pair.

For mixed-role atoms, token-type mass should be computed over the allowed role
union, for example `P(TEXT or SCHEMA)`, and the conditional inner valid-set
likelihood should normalize within that same union. Most atoms remain singleton
role atoms; mixed text/schema role sets should be explicit diagnostics such as
`mixed_role_atom_rate`, `text_schema_boundary_atom_rate`, and
`prefix_description_conflict_count`.

The shared target IR may allow arbitrary role sets for future extensibility, but
the Stage-1 marker-delimited compact-full builder v1 should emit only singleton
role sets plus `{TEXT, SCHEMA}` for description-prefix boundary ambiguity. Other
mixed role sets such as `{TEXT, COORD}`, `{SCHEMA, COORD}`, `{COORD, STOP}`, or
`{TEXT, STOP}` should be rejected by builder/runtime validation unless a future
template state explicitly requires them.

Coverage weights for mixed-role shared-prefix atoms should represent
residual-object branch mass, not token-role balancing. Builders should assign
raw unnormalized `coverage_target_weights` by compatible object count and then
aggregate by next token id, independent of whether the token is text or schema.
For example, if two `car` objects and one `carrot` object remain after prefix
`<|object_ref_start|>car`, then `<|box_start|>` gets raw mass `2.0` and `rot`
gets raw mass `1.0`. The coverage module normalizes internally.

For mixed-role atoms, inner valid-set likelihood and within-valid coverage
should normalize over the allowed role union, not over the whole vocabulary and
not over only the selected teacher role. Full-vocabulary wrong-role competition
belongs to `token_type_mass`; inner modules answer whether mass inside the
allowed role union lands on valid branch tokens and how that valid mass is
distributed when coverage is enabled.

Mixed-role token-type diagnostics should report both allowed-union behavior and
selected-teacher-role provenance. The type loss is `-log P(allowed role union)`.
Diagnostics should include allowed union mass, selected role mass, per-role mass,
wrong-type mass, allowed-union top-1 role accuracy, and selected-role top-1
accuracy. Selected-role diagnostics are useful for teacher-path provenance but
must not become a loss before branch commitment.

After a mixed-role description-boundary atom, candidate filtering should follow
the selected teacher token through the description trie. If the selected token
is a text continuation, keep candidates whose description token path continues
with that token. If the selected token is `<|box_start|>`, keep candidates whose
description token path is complete at the current prefix. The branch becomes
committed only if the filtered candidate set is singleton; otherwise the builder
continues valid-set marginal construction, including coordinate-onset ambiguity
when repeated descriptions remain.

Description-boundary ambiguity should have its own diagnostic role distinct from
ordinary description-token ambiguity. Use roles such as `DESC_TOKEN`,
`DESC_BOUNDARY`, `x1`, `y1`, `x2`, `y2`, `OBJECT_BOUNDARY`, and `STOP`.
`DESC_TOKEN` means all alternatives are text continuations; `DESC_BOUNDARY`
means the valid alternatives cross a text/schema boundary, such as `rot` versus
`<|box_start|>` for `car`/`carrot`.

Description tokenization should be cached globally by tokenizer fingerprint,
normalization policy, and normalized description string, while active residual
description tries should be built per sample or per roll-in state. Tokenization
is reusable across objects and datasets; candidate sets, duplicates, branch
commitment, and coordinate-onset ambiguity are image/state-specific and should
not be forced into a global trie.

The description token cache should remain runtime/in-memory for v1. Disk
persistence should be deferred until there is evidence that tokenization is a
bottleneck. If persistence is added later, it must be keyed by tokenizer
fingerprint, normalization policy, and description string to avoid stale target
construction.

Description tokenization should be context-aware and validated against rendered
compact entries. The canonical description token ids should be extracted from
the tokenization of `<|object_ref_start|>{desc}<|box_start|>` by locating exact
single-token marker boundaries and taking the token span between them. The cache
key should include tokenizer fingerprint, normalization policy, serialization
policy, left marker, right marker, and normalized description string.

The builder must guarantee exact and unique mapping between rendered text spans
and token spans. Marker ids must appear as exact boundaries, description token
spans must be uniquely extractable, and extracted token ids must align with the
actual training sample positions used by the target IR.

Mapping extraction and span alignment failures should be fail-fast errors, not
recoverable warnings. The builder and runner should hard-fail when marker ids
are not exact single-token boundaries, rendered entry tokenization has missing
or ambiguous markers, description token spans are not uniquely extractable,
extracted spans do not match full assistant sample positions, selected token ids
do not match `input_ids[target_position]`, causal shift invariants fail, or a
supervised token is unexpectedly truncated or masked.

Teacher-forcing target IR construction should not allow partial supervised
object truncation in v1. The rendered full sample must fit the configured max
length with prompt/image placeholders, assistant compact payload, and `<|im_end|>`.
If truncation would remove or partially cut any supervised marker, description,
coordinate, STOP token, or target position, the sample is invalid for this
objective. Length overflow should be handled before target construction through
an explicit reject/drop policy with counters, not by silent tokenizer/model
truncation after targets are built.

Object subsampling for overlength samples is out of scope for v1. V1 should use
`length_overflow_policy: reject_sample` rather than silently subsampling objects
to fit max length. Future object-subsampling support would need its own explicit
semantics for seed, selection policy, dropped-object interpretation, EOS/continue
meaning, and eval comparability.

Sample rejection should have explicit counters and fail thresholds. Overlength
rejection should count rejected samples, rejected objects, rejected fraction,
max observed token length, and length percentiles in dataset/run manifests.
Production configs should fail if the rejection fraction exceeds an explicit
threshold. Mapping/alignment failures remain unconditional hard errors; the
threshold applies only to expected overlength rejection.

Canonical `TeacherForcingTargetIR` construction should happen in the dataset or
detection item builder, not in the trainer. The builder owns compact-full
rendering, chat-template/tokenization, exact span validation, canonical target
IR construction, roll-in selection, and overlength rejection. The collator only
pads tensors and preserves/attaches per-sample IR sidecars. The trainer strips
sidecars before model forward, validates live `input_ids` against IR positions,
and computes losses/diagnostics from logits and IR.

Dataset-time random roll-in should be deterministic and reproducible. The base
roll-in seed should be `17`. Per-sample roll-in randomness should be derived
from stable inputs such as base seed, epoch, sample stable id, roll-in policy
name, and roll-in policy version rather than Python process-global random state
or dataloader worker order. IR metadata should record roll-in policy, version,
epoch, and derived seed provenance.

Random roll-in should vary by epoch deterministically. Use the base seed `17`
plus epoch, stable sample id, policy name, and policy version to derive the
per-sample roll-in seed. This improves order coverage while preserving
reproducibility across distributed workers and dataloader order. Encoded-sample
caches that store full input ids and target IR must either include epoch/roll-in
seed in the cache key or be disabled for teacher-forcing v1; a static cache must
not silently freeze roll-in.

Teacher-forced validation and probe roll-in should use fixed evaluation seeds
rather than training epoch-varying roll-in. Routine validation should use a
stable evaluation roll-in epoch/seed such as base seed `17` with
`eval_rollin_epoch: 0`, so curves are comparable across checkpoints.
Permutation-sensitivity analysis may intentionally evaluate multiple fixed
roll-in epochs/seeds such as `[0, 1, 2, 3, 4]`, and reports must record the
roll-in seed/epoch set.

Training v1 should disable encoded-sample cache because roll-in changes each
epoch. Fixed eval/probe teacher-forcing examples may use encoded-sample cache
only if the cache key includes target-IR-relevant fields such as tokenizer and
chat-template fingerprints, compact-full serialization policy, description
normalization policy, roll-in policy/version/seed/epoch, target IR schema
version, and max length. Cached payloads must include both `input_ids` and
`teacher_forcing_target_ir` and be validated on load. If the existing cache
cannot key these fields cleanly, disable it for eval/probe too.

Old recursive-detection training implementation files and active config surfaces
should be removed outright rather than temporarily wrapped or deprecated. This
includes old recursive target builders, recursive CE losses, recursive target
enrichers, recursive trainer mixins, recursive metric emitters, config schema
variants, and active Stage-1 recursive-detection training configs. Historical
inference/eval compatibility for already-trained checkpoints may remain, but
new training configs using old objective ids should not run through wrappers.

Schema validation should keep explicit rejection/migration errors for deleted
old objective ids and fields, but only as failure logic. Configs using
`objective.id: recursive_detection_ce`, old recursive variants, or old
support/balance fields should fail with a clear message pointing to
`objective.id: teacher_forcing` profiles and the historical inference namespace.
The rejection path must not instantiate old config dataclasses or compatibility
wrappers.

Old terminology should be removed aggressively from active docs and config
names. Terms such as `recursive_detection_ce`, `prefix_rollin_et_rmp_ce`,
`ET_RMP_CE`, `et_rmp_like`, `support_balance`, `trie_balance`, `trie_support`,
and objective-identity names such as `alpha0`/`alpha1` should not appear in
active runnable surfaces. They may appear only in clearly historical or
migration context, such as old-checkpoint inference configs or a brief
historical comparator explanation. Current docs should route users to
`teacher_forcing` and the new package/config paths.

OpenSpec should use a fresh stable capability for the new objective, preferably
`openspec/specs/teacher-forcing-objective/spec.md`. The spec should define the
stable contract for `objective.id: teacher_forcing`, profiles, target IR
sidecar, allowed-role-set semantics, token-type mass, conditional valid-set
likelihood, within-valid coverage, compact-full marker-delimited training
serialization, parser/grammar compatibility boundaries, and migration away from
old recursive-detection training APIs. Old recursive-detection specs should be
treated as superseded history rather than mutated to look like they always meant
the new design.

OpenSpec should contain stable externally visible contracts, not the
implementation checklist. It should specify config schema behavior, target IR
sidecar fields and invariants, loss semantics, serialization/parser/grammar
contracts, metric/artifact namespaces, cache/packing/logit constraints, and
migration/deletion behavior. The implementation plan should own file-level
moves/deletions, sequencing, tests, smoke commands, and temporary mechanics.

Empty-object handling should distinguish missing detection lists from explicit
empty detection lists. If no detection list is available for an image/sample,
the runtime guardian should drop the sample as invalid or unsupported input for
the detection objective. If a dataset intentionally provides an explicit empty
detection list, it may be represented as a STOP-only teacher-forcing sample:
empty assistant payload plus chat-template `<|im_end|>`, with a singleton STOP
atom targeting `<|im_end|>`. This is hard STOP supervision, not continuation
margin. Parser/eval may parse empty output before `<|im_end|>` as
`{"objects": []}` when empty lists are in scope.

For Stage-1 COCO teacher-forcing v1, explicit empty detection lists should be
dropped too. The default policy is to drop both missing detection lists and
explicit empty object lists with counters. STOP-only empty-object samples are a
future optional policy for intentionally constructed negative-image datasets,
not part of the initial COCO objective.

Each supervision atom should store both `logit_position` and
`target_position`. For the first implementation, the canonical causal
teacher-forcing invariant is `target_position = logit_position + 1`, meaning
`logits[batch_index, logit_position]` predicts
`input_ids[batch_index, target_position]`.

The runner should validate this convention centrally rather than letting loss
modules shift labels internally. When `selected_token_id` is present, it should
match `input_ids[batch_index, target_position]`. The runner should also verify
that supervised target positions are compatible with labels/attention masks
where applicable, and that logits are unsliced with `logits.shape[:2] ==
input_ids.shape[:2]`.

The new `compact_full` training serialization should be marker-delimited with
no row-separator newline. Object entries should be delimited by
`<|object_ref_start|>` and fields by `<|box_start|>`; newline should not be a
schema/control target, a diagnostic-only formatting token, or an allowed token
in the new training/template/target-IR path.

This is a serialization/template/parser contract change, not merely a token
vocabulary tweak. The implementation must update rendering, strict template
parsing, span projection, target-IR construction, constrained decode grammar,
prompt examples if needed, and tests together.

Backward compatibility is allowed only at the inference/eval parser boundary
for previously trained models and ablation studies. Compatibility parsing may
accept legacy newline-delimited `compact_full` predictions, but new training
examples and new strict compact-full target construction must reject that
format. New artifacts should record the active serialization policy, for
example `compact_row_separator: none` or an equivalent compact-full payload
version, so legacy newline outputs and new marker-delimited outputs are not
silently conflated.

The legacy `recursive_detection_ce` / `prefix_rollin_et_rmp_ce` training API
should be hard-deleted from active training rather than preserved as an alias
or compatibility mode. New training configs should use the new
`objective.id: teacher_forcing` surface and explicit profiles/modules. Standard
SFT should remain available as a baseline, but it should be represented through
the new teacher-forcing surface as a hard selected-token profile rather than
through the legacy recursive-detection path.

Old names such as `recursive_detection_ce`, `prefix_rollin_et_rmp_ce`,
`ET_RMP_CE`, `support_balance`, and profile-like `alpha1` terminology should
not remain runnable training vocabulary. Existing implementation knowledge can
be mined during migration, but active legacy sidecars, mixins, config schema
variants, and target builders should be removed once the new target IR path
covers their responsibilities. Historical checkpoints may still be evaluated
through inference/eval compatibility paths.

Old Stage-1 recursive-detection training configs should be removed from the
active config tree. This includes the active `configs/stage1/recursive_detection_ce.yaml`
surface and the `configs/stage1/recursive_detection_ce_latest/{prod,smoke,ablation,negative}`
launch surfaces. New active Stage-1 training configs should live under a clean
teacher-forcing namespace, for example `configs/stage1/teacher_forcing/**`,
with explicit profiles such as `hard_sft`, `pure_valid_set_marginal`, and
`coverage_regularized_valid_set_marginal`.

Legacy inference/eval configs for already-trained checkpoints may remain, but
they should be separated into an explicitly historical namespace such as
`configs/infer/legacy_recursive_detection_ce/**` or
`configs/infer/historical/recursive_detection_ce/**`. Those configs may carry
legacy metadata and newline-or-marker parser compatibility, but they must not
define or imply a runnable training objective.

Old recursive-detection / ET-RMP training tests should be deleted or rewritten
against the new teacher-forcing IR and module contracts. Tests should not keep
old API names, old sidecar names, old config variants, old mixins, or old
support/balance terminology alive as active training behavior. Behaviorally
valuable coverage should move to new tests for target IR construction, causal
positions, token-type vocabulary, valid-set likelihood, coverage regularization,
marker-delimited compact-full serialization, and Stage-1 wiring.

A small legacy-parser-only test island may remain for old checkpoint inference
and eval compatibility. That coverage may assert that legacy newline-delimited
compact-full predictions are parsed for historical scoring, but it must not
construct training targets or import old recursive-detection training builders.

The old ET-RMP comparator semantics should be represented structurally under
the new objective schema rather than by reviving old names. Do not create active
profiles or aliases named `et_rmp_like`, `alpha1`, `support_balance`,
`prefix_rollin_et_rmp_ce_compat`, or `typed_trie_alpha1`.

The structural comparator should be:

```yaml
objective:
  id: teacher_forcing
  profile: coverage_regularized_valid_set_marginal
  target_ir:
    rollin_policy:
      name: random_permutation
  modules:
    token_type_mass:
      enabled: true
    conditional_valid_set_likelihood:
      enabled: true
    within_valid_coverage:
      enabled: true
      coverage_strength: 1.0
    continuation_margin:
      enabled: false
```

Documentation and result tables may describe this as
“coverage-regularized valid-set marginal with `coverage_strength=1.0`, matching
the within-valid distribution behavior of the prior ET-RMP comparator.” The old
ET-RMP name may appear only as historical explanation, not as executable config
vocabulary.

`coverage_strength` should live only inside the `within_valid_coverage` module
config. It should not appear as a top-level `objective.alpha`, top-level
`objective.coverage_strength`, or profile-name encoding.
`objective.target_ir.rollin_policy.name` is a separate target-builder decision,
and `profile` should select a clean module bundle rather than encode
hyperparameters.

The allowed shape is:

```yaml
objective:
  id: teacher_forcing
  profile: coverage_regularized_valid_set_marginal
  modules:
    within_valid_coverage:
      enabled: true
      coverage_strength: 0.1
```

Profiles such as `typed_trie_alpha0p1_random_rollin` are forbidden terminology.

For `profile: coverage_regularized_valid_set_marginal`, production and ablation
configs should require an explicit finite nonnegative
`objective.modules.within_valid_coverage.coverage_strength`. Real experiment
configs should not rely on hidden profile defaults for this value. Test and
smoke helpers may fill local defaults only when the default is visible in the
test.

`hard_sft` should require coverage disabled or absent. `pure_valid_set_marginal`
should require coverage disabled or `coverage_strength: 0.0`.
`coverage_regularized_valid_set_marginal` should require coverage enabled and
an explicit positive `coverage_strength` in production; `coverage_strength: 0.0`
is semantically the pure profile and should be rejected in production configs
for this profile.

For the first implementation, module weights should be fixed rather than exposed
as an independent tuning surface. `token_type_mass`,
`conditional_valid_set_likelihood`, and `within_valid_coverage` should each use
their fixed default contribution, with `coverage_strength` as the semantic
within-valid coverage knob. This avoids ambiguous effective weights such as
`within_valid_coverage.weight * coverage_strength`.

Continuation margin is the exception: if it is enabled later, a margin weight
may be exposed because continuation calibration is conceptually separate from
the main likelihood and coverage decomposition. Top-level lambda knobs such as
`lambda_type`, `lambda_inner`, or `lambda_schema` should not be introduced in
the first implementation.

Token-type exclusivity should be implemented as a loss-only full-vocabulary mass
objective during training, not as a hard pre-softmax token-type mask. Training
should use full-vocabulary logits and full-vocabulary softmax, compute
`L_type = -log P(correct_role)`, and then let inner modules consume conditional
probabilities normalized within the correct token role. This preserves
wrong-type mass as both a gradient source and a diagnostic.

Hard grammar or token-type masks may be used as decode-time policy, but they are
not part of the training loss semantics.

Decode-time compact grammar should remain optional rather than required for
primary evaluation of the new objective. The intended default is to rely on the
trained grammar: the model should learn schema/text/coordinate/STOP stability
from the token-type and inner objective rather than depending on an inference
mask for schema validity.

Raw or minimally constrained decoding should therefore be a first-class
diagnostic and the preferred objective-development view. Grammar-constrained
decoding may remain available as an explicitly labeled production-stability or
comparison mode, but it must not hide failures in learned schema stability.

New teacher-forcing inference configs should keep
`infer.generation.compact_grammar.enabled: false` by default. Grammar-enabled
runs are allowed only as explicitly labeled secondary decode modes, such as
`*_grammar_decode`, and result reports should separate `raw_decode` from
`grammar_decode`.

Historical old-checkpoint inference configs may keep compact grammar enabled in
the legacy inference namespace. For new marker-delimited models, the compact
grammar processor must be updated so that after four coordinate tokens it
allows `<|object_ref_start|>` or `<|im_end|>`, not newline. Legacy newline-based
grammar behavior should not be the default for new models.

Compact grammar should expose an explicit serialization policy so legacy
newline grammar and new marker-delimited grammar cannot be confused. The new
policy should be `marker_delimited`: after four coordinate tokens, grammar may
allow `<|object_ref_start|>` or `<|im_end|>`, and newline is not structural.
The legacy policy should be `legacy_newline_delimited`: after four coordinate
tokens, grammar may allow newline or `<|im_end|>`, and after newline it may
allow `<|object_ref_start|>` or `<|im_end|>`.

New teacher-forcing inference configs may only use `marker_delimited`.
Historical inference configs must explicitly request `legacy_newline_delimited`
if they need the old behavior. If compact grammar is enabled without an
explicit policy in a new config, it should default to `marker_delimited`.

Parser compatibility should use a separate parser mode rather than reusing the
decode grammar serialization policy. Grammar policy controls generation-time
allowed tokens; parser mode controls what already-generated text is accepted
for scoring.

New teacher-forcing outputs should default to a strict marker-delimited parser
mode such as `marker_delimited_strict`, which accepts marker-delimited entries
and rejects newline-separated rows as malformed new-format output. Historical
checkpoint scoring may use a `legacy_compatible` parser mode that accepts both
legacy newline-delimited and marker-delimited compact-full predictions if they
are otherwise parseable. Parse artifacts should record the active parse mode
and observed separator style so compatibility does not become invisible.

The strict marker-delimited parser should be all-or-nothing for primary
evaluation. If the generated compact payload violates the marker-delimited
grammar, the prediction should fail parsing rather than silently salvaging a
prefix or subset of valid objects. Optional salvage may exist only as
diagnostic/debug extraction and must not feed headline AP/AR unless explicitly
labeled as salvage-based analysis.

Strict compact parse diagnostics should use a small stable taxonomy. Initial
stable codes should include `empty_output`, `legacy_separator_in_new_format`,
`missing_object_ref_start`, `missing_box_start`, `empty_description`,
`forbidden_description_token`, `wrong_coord_arity`, `invalid_coord_token`,
`trailing_garbage`, and `invalid_geometry`. Artifacts may also keep broad
compatibility errors such as `malformed_compact_full`, but they should expose a
specific parse error code, active parse mode, and observed separator style.

Compact parse diagnostics should live both in per-sample inference artifacts and
aggregate metrics. Per-sample artifacts should keep strict `pred` as the scored
prediction while adding a `parse_diagnostics` object with fields such as
`format`, `mode`, `observed_separator`, `error_code`, optional `error_offset`,
optional `entry_index`, and diagnostic-only `salvaged_object_count`. Aggregate
metrics should report parse failure rates, error-code counts, observed
separator counts, and salvage counts under an inference parse namespace such as
`infer/parse/compact_full/...`. These parse metrics should stay separate from
training-time `teacher_forcing/...` objective metrics.

New training metrics should use the `teacher_forcing/...` namespace with no
compatibility aliases for old `recursive_detection_ce/*` keys. New metrics
should include module losses, token-type mass diagnostics, valid-set mass
diagnostics, coverage diagnostics, continuation diagnostics, coordinate-onset
diagnostics, and ambiguity rates. Old keys such as `recursive_detection_ce/*`,
`loss/recursive_detection_ce`, `trie_support/*`, `trie_balance/*`, and `et_rmp/*`
should not be emitted by new training runs. Historical artifacts may still
contain old keys, but dashboards and new configs should use the new namespace.

Legacy analysis scripts that read old metric keys should remain historical and
read-only. Do not build a generic migration adapter that maps old
`recursive_detection_ce/*` or support/balance keys into new `teacher_forcing/*`
keys, because the semantics are not necessarily one-to-one after the redesign.
Valuable analysis routines should be replaced by new scripts that read the new
teacher-forcing metrics and use the new terminology directly.

The minimal new analysis/report surface should consist of three focused tools:
`src/analysis/teacher_forcing_objective_report.py` for scalar run summaries and
AP/AR links, `src/analysis/teacher_forcing_atom_probe.py` for teacher-forced
mechanistic atom diagnostics, and `src/analysis/compact_full_parse_report.py`
for inference parse diagnostics. Old analysis scripts should not be migrated en
masse; historical ones may remain for old artifacts, but new reports should use
the new metrics and terminology.

The current `src/trainers/teacher_forcing/` package should not be preserved as
the canonical objective implementation. Its useful scaffolding ideas can be
mined, but old auxiliary-loss modules and registry semantics should be deleted
or rewritten. Active modules such as adjacent repulsion, bbox geometry aux,
bbox size aux, coordinate regularization, duplicate unlikelihood, and old token
CE should not remain in the new core package.

The new canonical implementation should expose target IR contracts, role/type
vocabulary, profile transforms, reductions, diagnostics, runner logic, and the
new modules: token-type mass, conditional valid-set likelihood, within-valid
coverage, and optional continuation margin. Trainer-side code should integrate
with that package rather than owning the objective semantics.

The canonical objective implementation should live under
`src/objectives/teacher_forcing/`, while detection-specific target construction
should live under `src/detection/teacher_forcing/`, and `src/trainers/` should
own only integration glue. Objective math, target IR contracts, role/type
vocabulary, profile transforms, reductions, diagnostics, validation, runner
logic, and objective modules belong under `src/objectives/teacher_forcing/`.
Compact-full serialization, residual-object branching, coordinate-onset
candidate filtering, and detection target builders belong under
`src/detection/teacher_forcing/`. Trainer mixins or integration helpers should
handle sidecar stripping, model forward, unsliced logits validation, metric
logging, and ms-swift/HF integration without owning objective semantics.

The high-level detection template id should remain `compact_full`; do not rename
it to `compact_full_v2` or similar. The changed behavior should be represented
as serialization policy/version metadata, not a new format name. The new policy
is marker-delimited compact-full with no row separator; legacy newline-delimited
compact-full remains only for historical inference/eval compatibility.

Training serialization policy should be configured and validated under
`detection_template`, not under `objective`. Recommended shape:

```yaml
detection_template:
  id: compact_full
  serialization_policy: marker_delimited
```

The objective consumes target atoms and should not own assistant text rendering.
For new `objective.id: teacher_forcing` compact-full training, the
serialization policy must be `marker_delimited`; legacy newline-delimited
serialization is rejected for training. Inference parsing and optional compact
grammar use separate config surfaces even when they reference related policy
concepts.

Production and ablation training configs should explicitly set
`detection_template.serialization_policy: marker_delimited`. This should not be
an invisible production default, because the separator policy is
research-significant. Tiny tests may use local defaults only when the resolved
policy is still visible in the resulting config or fixture.

Prompt examples and docs for the new compact-full training contract should show
literal adjacent marker-delimited entries without human-readable row separators.
Examples with newline-separated rows should appear only when explicitly labeled
as legacy newline-delimited output. If readability is needed, use prose or
annotations outside the literal serialized payload rather than inserting
formatting characters into the payload example.

Description text normalization for marker-delimited compact-full should be
strict. Stage-1 teacher-forcing rendering should strip leading/trailing
whitespace, reject empty descriptions, reject newline/tab/control characters,
reject reserved schema/control/image/stop marker text, reject coordinate-token
text inside descriptions, preserve normal internal spaces when needed, and avoid
adding any synthetic trailing separator before `<|box_start|>`. Markers are the
only reliable field and entry boundaries.

## Evidence

- Scope: `none-yet`
- Discussion handles:
  - `docs/AGENT_INDEX.md`
  - `docs/catalog.yaml`
  - `docs/training/STAGE1_OBJECTIVE.md`
  - `openspec/specs/stage1-latest-detection-objectives/spec.md`
  - `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`
  - `openspec/specs/runtime-architecture-refactor-program/spec.md`
- Current Stage-1 friction handles:
  - `src/detection/objective.py`
  - `src/detection/loss.py`
  - `src/detection/token_types.py`
  - `src/detection/tokenizer_contract.py`
  - `src/detection/runtime.py`
  - `src/config/schema.py`
- Current Stage-2 friction handles:
  - `src/trainers/teacher_forcing/contracts.py`
  - `src/trainers/teacher_forcing/module_registry.py`
  - `src/trainers/teacher_forcing/objective_pipeline.py`
  - `src/trainers/teacher_forcing/objective_atoms.py`
- Upstream integration handles:
  - `/data/ms-swift/swift/trainers/trainers.py`
  - `/data/ms-swift/swift/trainers/trainer_factory.py`
  - `/data/ms-swift/swift/llm/template/base.py`
  - `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/trainer.py`
  - `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/trainer_seq2seq.py`
- Tokenizer artifact handles:
  - `model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp/special_tokens_map.json`
  - `model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp/tokenizer_config.json`
  - `model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp/added_tokens.json`

## Open Follow-Up Questions

The grilling loop should next resolve:

1. whether to stop grilling and prepare the stage1/stage2 implementation plan.


## 2026-05-20 Stage-2 Residual-Set Self-Prefix / UL Redesign

Detailed decisions were moved to [2026-05-20_stage2_residual_set_self_prefix_ul_redesign.md](2026-05-20_stage2_residual_set_self_prefix_ul_redesign.md).

Summary:

- Stage-2 correction should be residual-set conditioned at grammar-valid self-prefixes.
- K rollout attempts are independent self-prefix correction samples, not merged target distributions.
- Correction samples anchor at the earliest actionable error boundary.
- Repeated-object boundaries use positive-only residual-set correction, not duplicate unlikelihood.
- UL mining can promote strict K-valid consensus clusters as rollout-local TP-like positives with separate provenance, weight, metrics, and review artifacts.
