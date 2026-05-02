# Add Stage-1 ET-RMP-CE Objective

Status update, 2026-05-02: candidate-balanced / energy-style
set-continuation has been retired as a production training direction. This
change should now be read as the promotion path for ET-RMP-CE: full-suffix
teacher-forced token CE, prefix-conditioned sampling, entry-trie
multi-positive token CE, support/balance reweighting, and hard CE for
schema/control/separator/stop tokens.

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

Candidate-balanced one-step continuation, candidate energy/logZ objectives,
chunk-level MP, candidate branch CE as production objective, and PEM/margin
losses tied to candidate energy ranking are legacy compatibility surfaces only.

## Impact

- Changes Stage-1 loss semantics only when the new objective mode is selected.
- Reuses existing set-continuation setup, collator, prefix sampling, object
  serialization, packing rejection, and train-forward runtime controls.
- Adds a sibling full-suffix smart-batched row scorer. The row unit changes from
  `prefix + candidate + boundary` to `prefix + full_suffix + close/EOS`, but
  rows remain independent padded batch rows.
- Does not change the offline JSONL data contract, image geometry, or inference
  artifacts.
- Retires candidate-balanced production training in favor of the full-suffix
  ET-RMP-CE objective family.
