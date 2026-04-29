# Add Stage-1 ET-RMP-CE Objective

## Why

The current Stage-1 set-continuation objective scores independent one-entry
candidate continuations. This does not train the full recursive generation
process that rollout uses, and it can entangle object evidence with schema,
boundary, close, and chunk-length effects.

Detection examples with repeated classes make the gap sharper. A desc token may
only narrow the remaining set from many objects to several same-class instances;
the actual valid choice can remain ambiguous until bbox or later coordinate
tokens. A one-token or first-divergence-only MP objective would still treat
valid same-class continuations as negatives after the first shared tokens.

## What Changes

Add an off-by-default objective mode under the existing Stage-1
set-continuation trainer:

```yaml
custom:
  trainer_variant: stage1_set_continuation
  stage1_set_continuation:
    objective:
      mode: entry_trie_rmp_ce
      suffix_order: random
```

The mode trains one full remaining suffix per sampled prefix. At every object
entry token, it builds a trie over all currently remaining serialized object
entries and applies object-uniform multi-positive CE at every branching trie
node. Unique trie nodes, comma boundaries, global close, and EOS/chat-template
end labels use ordinary full-vocabulary CE.

The current candidate-balanced objective remains the default and keeps its
existing production profile.

## Impact

- Changes Stage-1 loss semantics only when the new objective mode is selected.
- Reuses existing set-continuation setup, collator, prefix sampling, object
  serialization, packing rejection, and train-forward runtime controls.
- Adds a sibling full-suffix smart-batched row scorer. The row unit changes from
  `prefix + candidate + boundary` to `prefix + full_suffix + close/EOS`, but
  rows remain independent padded batch rows.
- Does not change the offline JSONL data contract, image geometry, inference
  artifacts, or default candidate-balanced profile.
