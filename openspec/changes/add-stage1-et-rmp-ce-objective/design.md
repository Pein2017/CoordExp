# Design: Stage-1 ET-RMP-CE Objective

## Objective Boundary

ET-RMP-CE is added as an objective selector inside
`custom.stage1_set_continuation`, not as a new trainer variant. This keeps the
existing setup-path guardrails: raw metadata collation, coord-token-only
validation, packing rejection, encoded-cache bypass, and benchmark metadata.

The default remains `candidate_balanced`. The new experimental modes are:

- `full_suffix_ce`: recursive full suffix with ordinary hard CE everywhere;
- `entry_trie_rmp_ce`: recursive full suffix with entry-trie multi-positive CE
  at all divergence nodes.

## Full-Suffix Rows

For each original sample, the trainer builds one row:

```text
prefix + entry(tau_1), entry(tau_2), ... entry(tau_m)]} + assistant end/EOS
```

If the prefix is empty, the generated schema opener is included in the
supervised objective span to preserve free-generation alignment. It is not
included in the entry trie.

## Entry Trie

At recursive state `Sk`, the remaining multiset is `Rk = O - Sk`. The logical
entry trie covers serialized entries `E(o)` for every `o in Rk`, excluding
comma, global close, schema opener, and EOS. The implementation may tokenize
each candidate entry in the current autoregressive context, with the next
boundary text present only to recover the same object-entry label tokens that
the chat template produces.

During teacher forcing, the sampled object `tau_k` advances the context through
one path in that trie. At every node:

- one child token: hard CE on the teacher token;
- multiple child tokens: soft CE with object-uniform child probabilities.

Exact duplicate serialized entries remain multiplicities under the same path.
Emitting one duplicate removes one object instance from the remaining multiset.

## Probability Space

The main objective uses full-vocabulary log probabilities for all tokens,
including coord tokens. Coord-vocab-normalized quantities from the current
candidate scorer remain out of the main ET-RMP objective and may be added later
only as named diagnostics or auxiliaries.

## Smart-Batch Compatibility

The current `smart_batched_exact` bridge batches independent padded branch rows.
ET-RMP-CE keeps that contract and introduces a sibling full-suffix scorer:

- rows are independent full sequences;
- rows may be grouped by the existing branch-batcher scheduler;
- `logits_to_keep` uses the same supervised-suffix idea, cropping only
  unsupervised prefix logits;
- no true packed-varlen attention, branch attention mask, or GPU KV prefix
  sharing is introduced.

## Metrics

Trainer-native metrics cover trie behavior, branch CE, unique CE, boundary CE,
close/EOS CE, valid-child mass, top-1 validity, and GT-count buckets. Rollout
hygiene metrics remain eval-side because they require generation.
