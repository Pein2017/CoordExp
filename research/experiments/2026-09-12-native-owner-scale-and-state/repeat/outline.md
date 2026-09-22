# Lane B: deployment-greedy repeat supply and physical drift audit

## Frozen question

Can unchanged Stable50 greedy complete outputs supply later-row, class-blind
native-pixel IoU `>0.95` repeat rows that could be used as a future negative
source, and what visually same-instance drift does that strict predicate miss?

## Frozen source and predicate

- Initial source is the 384-record Stable50 endpoint packet used by the
  positive-progress matched control. A later raw128 source may be consumed
  only when every record is a member of that frozen 384 universe.
- Output allowance remains exactly `3084`; existing cap/EOS outcomes stay in
  the denominator. No model call, training, threshold sweep, or sample
  expansion is part of this lane.
- A strict repeat is one later valid complete row whose native-pixel box has
  IoU `>0.95` with any earlier valid complete row, regardless of class. The
  later row is counted once, not once per matching pair.
- Same-category sub-threshold overlaps are review candidates only. GT-unmatched
  seeds are `unknown_or_unmatched`, not hallucination labels or negatives.

## Review and acceptance

The consumer freezes at most two deterministic cards per stratum: exact versus
sub-threshold drift, supported versus unknown seed, and (for drift) near/mid/
low IoU bins. Cards show the full original image and both row boxes; they are
not cropped quality proxies. `replay-packet.json` records the source hashes and
the cold consumer command. `consumer.json` must pass the exclusive threshold
boundary check (`.95` itself is not a repeat) and reproduce the census from the
source hash.

The lane reports supply and concrete visual drift counterexamples only. It does
not rerun the empty 0/768 raw-softmax comparison and does not claim negative
learning efficacy.
