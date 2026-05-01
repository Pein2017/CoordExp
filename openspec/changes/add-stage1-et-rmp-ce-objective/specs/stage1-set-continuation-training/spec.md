## MODIFIED Requirements

### Requirement: Stage-1 set-continuation supports recursive full-suffix objectives
The trainer SHALL keep `candidate_balanced` as the default objective and expose
off-by-default recursive full-suffix objective modes through
`custom.stage1_set_continuation.objective`.

Normative behavior:
- `objective.mode=candidate_balanced` SHALL preserve the current production
  candidate-balanced behavior,
- `objective.mode=full_suffix_ce` SHALL train one full remaining suffix with
  ordinary full-vocabulary CE,
- `objective.mode=entry_trie_rmp_ce` SHALL train one full remaining suffix with
  entry-trie multi-positive CE at all object-entry divergence nodes,
- full-suffix modes SHALL reuse the existing subset-prefix sampler and
  canonical object-entry serialization,
- full-suffix modes SHALL train recursive closure through object entries,
  comma boundaries, global close, and EOS/chat-template end labels.

#### Scenario: Default objective is unchanged
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** no objective mode is authored
- **WHEN** training computes the Stage-1 set-continuation loss
- **THEN** the trainer uses the existing candidate-balanced objective.

#### Scenario: Full-suffix objective trains recursive closure
- **GIVEN** a sampled prefix subset `S`
- **AND** remaining objects `R = O - S`
- **WHEN** `objective.mode=entry_trie_rmp_ce`
- **THEN** the trainer samples one full suffix permutation over `R`
- **AND** trains the continuation through every remaining object entry
- **AND** trains inter-object comma boundaries with ordinary CE
- **AND** trains final global close and EOS/chat-template end labels with
  ordinary CE.

### Requirement: Entry-trie RMP applies MP at every object-entry divergence node
For `objective.mode=entry_trie_rmp_ce`, the trainer SHALL build a trie over the
serialized object entries of the currently remaining object multiset at each
recursive object-selection state.

Normative behavior:
- trie entries SHALL exclude schema opener, inter-object comma, final global
  close, EOS, and chat-template stop tokens,
- implementations MAY tokenize entries in their current autoregressive context
  to align with chat-template label spans, but boundary tokens SHALL remain hard
  CE control-flow labels rather than trie positives,
- every trie node with multiple child tokens SHALL use an object-uniform
  valid-balance target over valid child tokens,
- branch nodes SHALL also include a full-vocabulary valid-support term
  `-log(sum_{v in valid children} p_theta(v | context))`,
- `branch_support_weight=1.0` and `branch_balance_weight=1.0` SHALL be
  equivalent to the prior full-vocabulary object-uniform soft CE,
- implementations MAY increase `branch_support_weight` relative to
  `branch_balance_weight` for controlled ET-RMP support-mass experiments,
- every trie node with exactly one child token SHALL use ordinary hard CE,
- object-uniform probabilities SHALL be proportional to active object
  multiplicity under each child token,
- coord tokens SHALL follow the same trie rule as text tokens,
- exact duplicate serialized entries SHALL remain multiplicities and SHALL NOT
  be artificially separated.

#### Scenario: Same-description objects branch at bbox tokens
- **GIVEN** remaining entries for two `cat` objects with different bboxes
- **WHEN** their desc tokens are shared
- **THEN** desc-prefix tokens use ordinary CE
- **AND** the first bbox token where the entries diverge uses object-uniform
  multi-positive CE.

#### Scenario: Shared first coordinate branches later
- **GIVEN** two remaining entries with the same desc and first bbox coordinate
- **WHEN** their second coordinate differs
- **THEN** the first coordinate token uses ordinary CE
- **AND** the second coordinate token uses object-uniform multi-positive CE.

### Requirement: Full-suffix rows are compatible with smart-batched exact runtime
Recursive full-suffix objectives SHALL support the current smart-batched exact
runtime contract by treating each original sample's full suffix as one
independent padded row.

Normative behavior:
- full-suffix rows SHALL NOT attend to one another,
- smart batching MAY group full-suffix rows by sequence length,
- supervised-suffix logits SHALL crop only unsupervised prefix logits,
- row losses SHALL be scattered back to original samples before batch
  reduction,
- the implementation SHALL NOT introduce true packed-varlen multimodal
  attention, GPU KV prefix sharing, or branch attention masks in this change.

#### Scenario: Smart-batched full-suffix rows match serial rows
- **GIVEN** two full-suffix rows with different prefix lengths
- **WHEN** the trainer scores them with `smart_batched_exact`
- **THEN** each row's loss and metrics match serial retained scoring for the
  same row.
