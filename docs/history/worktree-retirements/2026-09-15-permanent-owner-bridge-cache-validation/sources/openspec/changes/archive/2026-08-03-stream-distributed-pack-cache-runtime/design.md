## Context

The v2 packing cache is prepared and fully validated in a distinct
single-process phase.  Distributed train assembly nevertheless calls
`load_cache_manifest`, which currently verifies and decodes every payload, and
then calls `load_rank_micro_steps_from_cache`, which verifies and decodes every
payload again while selecting the canonical rank-local tuple.  The second call
is required; the first full payload pass is redundant.

The observed production launch spent about 10.5 minutes before training.  The
first proposed lazy reader was audited against the real 13.06 GiB cache: 33
chunks, about 431 MB per full chunk, 7.39 seconds per warm-cache verified decode,
and about 259 aligned transitions per rank across eight epochs.  Without proven
overlap it would add about 32 minutes to final wall clock, so it is rejected.

## Goals / Non-Goals

**Goals:**

- Minimize final end-to-end training wall clock with a deterministic, low-risk
  removal of one redundant full payload pass.
- Preserve exact cache integrity, rank-local order, eager tuple consumption,
  eval behavior, and all research semantics.
- Make every verification boundary explicit so a future omitted argument cannot
  silently reintroduce the expensive pass.

**Non-Goals:**

- Reducing resident rank memory.
- Lazy chunk decoding, background prefetch, threads, process pools, shared
  memory, mmap, or cache sharding.
- Changing cache v2, fingerprint determinants, chunk size, 16-worker
  preparation, data, prompts, geometry, configs, eval, or resume behavior.

## Decisions

### 1. Use two explicit verification levels

`manifest` validates version, status, fingerprint-to-determinant binding,
contiguous chunk plan, counts, safe existing paths, and digest syntax without
reading payload bytes.  `payloads` additionally verifies every declared SHA256,
restricted-unpickles each chunk, and validates tuple/count/micro-step types.
There is no default level.

Preparation, post-publication self-check, completeness checks, existing-cache
reuse, and eager eval use `payloads`.  Distributed train resolution uses
`manifest` only.

### 2. Keep the existing eager rank loader as the single payload pass

After manifest admission, `load_rank_micro_steps_from_cache` remains unchanged.
It verifies every chunk digest and decoded payload while selecting the canonical
rank-local sequence, and it completes before the first forward.  Corruption
after preparation therefore still fails closed before training.  No iterator,
thread, new lifetime owner, or schedule implementation is introduced.

### 3. Keep the fallback surface small

Configured eval and the no-eval-cache alias remain unchanged because the train
tuple stays eager and reiterable.  Cache files remain reusable.  Rollback is the
single verification-level call at distributed train resolution.

## Verification

- A corrupted payload is accepted by `manifest` structural admission but is
  rejected by the immediately following eager rank load before forward.
- Unknown or omitted verification intent fails at the call boundary.
- Preparation tests pin `payloads`; initialized distributed training tests pin
  `manifest`; eager eval stays on `payloads`.
- Focused pack-cache, pipeline assembly/rebuild, trainer, runtime, and eval tests
  must remain green in source and independently in the clean target.
- The final target diff must contain no lazy-reader, prefetch, thread, geometry,
  prompt, config, evaluator, or resumable-checkpoint changes.

## Performance Decision

This route removes one of the two full checksum-plus-unpickle passes and adds no
training-loop work.  Based on the observed startup breakdown it is expected to
save roughly four minutes per launch.  That estimate is not presented as a new
production measurement; the next authorized launch may record actual admission
time.  The stronger claim is structural: final training cannot be lengthened by
new in-loop cache work because none is added.
