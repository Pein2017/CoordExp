# Stage-1 ET-RMP-CE Design

## Problem

The current Stage-1 set-continuation trainer is a candidate-branch objective.
For a sampled prefix `S`, it scores independent one-entry continuations
`prefix + entry(o) + boundary(o)` for selected remaining objects. The optimized
production loss is candidate-balanced CE over those chunks; MP/logZ quantities
are diagnostic unless PEM is enabled.

That objective can improve local object evidence while failing to train the
closed autoregressive process:

```text
object -> comma -> object -> comma -> ... -> global close -> EOS
```

It also mixes object identity with object serialization and boundary tokens
inside a chunk score. This is especially brittle for crowded scenes with
same-description objects, where the valid next object may remain ambiguous
until bbox or later coordinate tokens.

## Goal

Add an off-by-default objective mode inside the existing
`custom.trainer_variant: stage1_set_continuation` setup:

```yaml
custom:
  stage1_set_continuation:
    objective:
      mode: entry_trie_rmp_ce
      suffix_order: random
```

The new mode is called **Entry-Trie Recursive Multi-Positive CE**,
abbreviated **ET-RMP-CE**.

## Non-Goals

- Do not change the default candidate-balanced production profile.
- Do not introduce RL, beam search, rollout replay, or grammar-constrained decoding.
- Do not reuse coord-vocab-normalized logprob as the main probability space.
- Do not implement true padding-free packed multimodal attention.
- Do not include schema opener, comma, final close, or EOS inside the logical
  object-entry trie. Tokenization may include surrounding context or the next
  boundary only to align with production chat-template labels.

## Objective

For each sample:

1. Sample a prefix subset `S0` using the current subset sampler.
2. Let `R0 = O - S0`.
3. Sample one full suffix permutation `tau` over `R0`.
4. Render one teacher-forced continuation row:

```text
prefix + entry(tau_1), entry(tau_2), ... entry(tau_m)]} + assistant end/EOS
```

If `S0` is empty, the existing set-continuation convention is preserved:
the generated schema opener `{"objects": [` is part of the supervised objective
span, but it is not part of the entry trie.

For each recursive state `Sk`, build a trie over the full tokenized object
entries of the current remaining multiset `Rk = O - Sk`. The entry token
sequence is exactly the serialized object dictionary entry and excludes:

- inter-object comma,
- global close `]}`,
- EOS or chat-template stop tokens,
- schema opener.

At each token inside the teacher-forced entry `entry(tau_k)`:

```text
if current trie node has multiple child tokens:
    use object-uniform soft CE over all child tokens
else:
    use ordinary hard CE on the teacher-forced token
```

At branch nodes:

```text
q(v) = number of active remaining objects under child token v
       / number of active remaining objects under current node
L = -sum_v q(v) log p_theta(v | context)
```

After each entry:

- if objects remain, comma/separator tokens use ordinary full-vocab CE;
- if no objects remain, global close and assistant end/EOS use ordinary full-vocab CE.

## Duplicate Entries

Exact duplicate serialized object entries are represented as multiplicity in
the trie. They never require artificial divergence. When a duplicate entry is
emitted once along the sampled suffix, exactly one object instance is removed
from the remaining multiset.

## Smart-Batch Fit

Current `smart_batched_exact` batches independent candidate branch rows with
length-aware padded-row grouping. ET-RMP-CE uses the same physical idea but
changes the row unit:

```text
candidate-balanced row: prefix + one candidate + boundary
ET-RMP-CE row:          prefix + full remaining suffix + close/EOS
```

Rows remain independent. They do not attend to one another. The smart-batch
scheduler groups full-suffix rows by sequence length, pads them as ordinary
batch rows, requests supervised-suffix logits when configured, and scatters
losses back to the original samples. This keeps compatibility with the current
smart-batched bridge without introducing true packed-varlen attention or KV
prefix sharing.

## Diagnostics

Trainer-native ET-RMP-CE metrics:

- `rmp/branch_nodes`
- `rmp/branch_nodes_desc_text`
- `rmp/branch_nodes_coord`
- `rmp/branch_nodes_structural`
- `rmp/branch_nodes_other`
- `rmp/valid_children_mean`
- `rmp/target_entropy_mean`
- `rmp/valid_child_mass_mean`
- `rmp/teacher_branch_top1_acc`
- `rmp/valid_child_top1_acc`
- `loss/rmp_branch_ce`
- `loss/rmp_unique_ce`
- `loss/rmp_coord_branch_ce`
- `loss/rmp_desc_text_branch_ce`
- `loss/rmp_boundary_ce`
- `loss/rmp_close_ce`
- `loss/rmp_eos_ce`
- `rmp/gt_count_ge7_samples`

Rollout hygiene remains eval-side and should be reported through detection eval
artifacts or callback metrics:

- parse-valid rate,
- empty predictions,
- hit max tokens,
- object after global close,
- extra top-level key,
- missing separator,
- GT-count buckets.

## Ablation Plan

A. Reference checkpoint only.

B. Current candidate-balanced set-continuation objective.

C. Random subset prefix plus full remaining suffix CE, no trie MP:
`objective.mode: full_suffix_ce`.

D. RMP-CE with MP only at the first entry divergence:
kept as a design comparison; not required for the first code path unless the
config grows `first_divergence_rmp_ce`.

E. ET-RMP-CE with MP at all entry-trie divergence nodes:
`objective.mode: entry_trie_rmp_ce`.

The key comparison is D vs E. In the first implementation, C vs E is the
minimum useful code-backed comparison because it separates recursive closure
from trie multi-positive labels.
