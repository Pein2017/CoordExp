# Fresh self-prefix known-FN bridge

Lead: 01a0dd7c-0899-7b81-90a2-2f50da3476d1. Execution owner: existing 926-worker, 01a0de41-cc56-7a62-8c56-c2d9850b95b5, actual gpt-6-astra/low. The user explicitly resumed experiments and authorized eight GPUs on 2026-09-29. No goal is created. Lead owns interpretation and finite runtime releases; worker returns candidates at assigned boundaries.

## Decision and predecessors

From the named margin checkpoint, does relocating matched C credit from its actual prefix h to h+B improve natural acquisition of a missing retained B while preserving incumbents and stable output, compared with the same B credit leaving C at h?

The [previous online unit](../2026-09-27-online-row-credit/results.md) established useful supervised learning and intermittent error suppression, but no durable hidden-label recovery. Its fixed geometry witnesses improved conditional margins without natural stability transfer. The old insertion experiment rewarded B and C at competing prefixes; its two selected acquisitions did not establish net recovery. This experiment tests that specific credit inconsistency with equal positive row weights. It does not assume every previous failure had that cause.

The user-referenced margin benchmark reduced parsed literal repeats in two data orders, but used a median coordinate logits processor and a different ancestor. Its checkpoint is a candidate input, not qualified no-processor performance here. All next-stage natural generation uses our ordinary policy. Masking an unknown row does not make the optimization omission-neutral. The ultimate objective remains physical-owner recovery; annotation-ID recovery below is its bounded proxy.

## Anchor and information boundary

Candidate anchor: `/data/CoordExp/outputs/infra_base/start-loss-benchmark-20260928/train/instance_margin-order17/checkpoints/step-256`. This fixed data-order candidate is not selected by our hidden results. Bind its own `inference_payload_manifest.json`, every adapter/delta/config artifact, base/config/tokenizer identity, and all actual loader copy receipts. Do not create a fake historical `identity.json` or mutate the checkpoint. CPU header inspection finds 588 FP32 DoRA tensors and two independent FP32 [1004,2048] input/output deltas. Verify exact keys/dtypes/values of our exported zero against the explicit anchor, using maintained normalization. It descends from illegal-mass001 step2444, not the old axis001 teacher.

Keep the original 18 images and whitelisted prompt/media/grid identities, retained513/hidden57 partition, COCO80/cans-as-bottle scope, Human13 and refined5 membership. Retained data: `/data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-27/rollout-row-credit-01/cpu-04/retained-10.json`. Input-only metadata: the sibling `retained-sft-01/inputs.json`. All18 are exposed adaptation data; refined5 has redraw uncertainty. Earlier checkpoint exposure to these annotations is not ruled out, so hidden means withheld from this stage, not never seen historically.

Only retained labels, frozen input identities and current raw predictions enter runtime selection, training and startup hash guards. Hidden truth and evaluator partition hashes stay in a separate offline binding checked only after whole-output freeze. Unknown predictions are context, not physical negatives. No fixed old pseudo bank, hidden-derived target, full GT sequence, dataset change or 20% experiment.

## Current-rollout bridge selection

Every update first obtains one fresh empty-history rollout for every image with the same current parameters. Reuse the maintained first-literal-valid, class-agnostic one-to-one IoU .5 assignment, followed by category agreement, to identify M. Never replace it with category-constrained assignment.

Consider retained annotations absent from M; additionally withhold any B having same-category IoU>=.5 overlap with ANY valid prediction anywhere in that whole rollout. Keep every disposition. Sort remaining B by (x1,y1,numeric annotation ID). For each B find the first generated first-literal-valid M row C whose OUTPUT anchor (x1,y1) is lexicographically greater than B's retained anchor. Select the first B with such a C, at most one pair/image. This is a frozen placement convention, not certified semantic order. No selective image exclusions, quota filling or fabricated EOS. A capped trajectory may supply a certified complete C before its incomplete tail.

h is the exact saved prompt and generated tokens strictly before C's opener, including unknown context. Render B from the retained label with the maintained renderer; C is its exact original emitted row, not rewritten to GT. Construct the identical short h+B+C input in both arms. Control selects B's complete atoms; treatment selects B and shifted C atoms. Row opener and local closure are credited; EOS and later suffix are absent/uncredited. C's own-box coordinate targets move with C.

## Frozen losses and synchronization

Let m be the ORIGINAL number of M rows for an image and ell the existing complete-row CE + .1 type gate + .01 own-prefix order gate, normalized within each row. Selected C implies m>=1.

- Control: sum(M ell at original prefixes)/m + ell(B|h)/m.
- Treatment: sum(M except C ell at original prefixes)/m + ell(B|h)/m + ell(C|h+B)/m.

All other M prefixes, tokens and coefficients remain exact. Do not renormalize by m-1, average B and C together, pool their unequal token counts, or leave C's original positive credit in treatment. Images without a pair retain all M and have zero bridge credit. Empty M is differentiable zero. Every original image remains in the fixed18 mean.

Both arms add the same maintained legal-mass loss on ALL certified actual trace coordinate contexts and .1 current-error Gmax. These keep their original own-start legal sets, full-vocabulary illegal complement, row/error normalization and tie gradients. C's geometry-only credit on the original trace is not removed. There is NO full GT .25R, old duplicate redirect, P0 or fixed witness branch. The explicit B term is the only new retained-target injection. Do not simultaneously adapt the benchmark's instance margin; this experiment first borrows its checkpoint. Error recurrence is measured and can reject the recipe.

Each image has one trace forward and at most one short bridge forward. Reuse uneven-rank accumulation: rank counts3/3/2/2/2/2/2/2, every contribution8/18 before DDP average, only final local backward synchronizes. The final branch is no longer R. Empty/absent bridge cases must not skip the rank's final synchronized graph. All ranks make one optimizer step per round.

## Execution and evidence

Use one resident BF16-base/FP32 trainable untied DoRA model, FA2 and BF16 autocast for acquisition/replay, actual attention/LoRA dropout zero. Ordinary empty-history greedy max3084, temperature0/top_p1/top_k0/RP1, no model defaults or decode processors. Seed92711; fresh AdamW per arm, language/delta LR1e-5/5e-6, betas(.9,.999),eps1e-8,WD0,clip1, continuous moments through all assigned updates. Reuse qualified source/payload/producer/freshness/export paths. No exact cross-run gradient parity claim.

CPU package implements the bounded consumer and prepares one treatment update plus proposed paired16 commands. Historical geometry-treatment16 records and the local census are CPU fixtures only; fresh targets always come from the new arm's current rollout. No GPU/model call during CPU preparation. Reuse native imports, not output-captured executable code or a generic replay framework.

Nearest actual-consumer falsifiers must detect: C credited twice or at the wrong prefix; m-1 or pooled normalization; B/C dose mismatch; extra R/redirect targets; wrong saved tokens, causal t-1, own-box targets, opener/closure/EOS masks; stale/mixed producer or swapped anchor/arm; hidden-data/hash dependency; incorrect final-sync/uneven-rank/empty behavior. On a frozen fixture, positive-loss difference must equal [ell(C|h+B)-ell(C|h)]/m. Geometry reductions remain common and unchanged. Preserve existing default numerical paths. CPU checks do not qualify live DDP or this new anchor.

After lead CPU acceptance, the smallest real-entry release is one treatment18-rollout -> update1 ->18-rollout run, then fresh CPU artifact readback and separate offline scoring. Its incoming18 also establishes this checkpoint's ordinary FA2/BF16 baseline, avoiding a duplicate baseline query. Verify new loader/export0 identity, actual bridge/C relocation, no-R final synchronization, all-rank finite590 gradients, actual memory and every raw/input/producer identity. This single update is mechanics and descriptive evidence, not a treatment effect. Saved artifact equality is not fresh model reload parity.

Conditional next release: two sequential arms16 updates each, exports0/1/2/4/8/16,17 fresh18-image versions/arm, both run/readback blocks before either offline truth stage. Each arm starts the explicit anchor with fresh optimizer, not slice1. Each arm306 natural requests/300220 visual tokens/max943704 generated tokens; replay at most576 forwards. One-update slice36 requests/35320 acquisition visual/max111024 generated tokens, at most36 replay forwards. CPU preparation must derive prompt+history+B length, selected-logit, token and visual bounds from actual retained rows/inputs; matrix bounds are not measured GPU peaks.

Primary paired evidence: endpoint16 AND all curves/late9..16 retained acquisition minus incumbent losses, hidden acquisition/preservation separately, raw/category and Human13/refined5/per-image, complete/valid literal repeats, invalid/malformed/near-overlap/caps/empty/length burdens and acquired-ID persistence. Same full570 class-agnostic matcher then category check. Track each selected B's subsequent natural coverage and C's survival; also preserve the incoming selected cohort for fixed-target continuity. Dynamic arm-specific selected rates are descriptive and cannot replace the common annotation population. Injected-prefix likelihood or conditional continuation is insufficient: final evaluation starts from the original image and empty history with no B supplied.

This first bridge uses an exact saved C after forced B; it is a local counterfactual, not fully on-policy. Downstream old M prefixes are unchanged. If it improves conditional B/C but not unforced recovery, the strongest alternative is distribution/continuation mismatch or interference beyond the local bridge. A later intervention may resample after B; it is not silently included in this contrast. Cleaner but lower-coverage output remains mixed; both arms improving does not prove relocation benefit. No favorable checkpoint selection, physical-precision claim or automatic extension.

## Ownership and stop

Worker owns only `probes/online_row_credit.py` and `tests/probes/test_online_row_credit.py`, plus new output artifacts under `/data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-29/self-prefix-known-fn-01`. Lead owns this protocol/state and research routing. Small reversible plumbing choices stay local; scientific conflict returns to lead. Return concrete failed checks promptly; lead may authorize repair, never silently relax contracts. No broad suite, new infrastructure, publication, production action, GPU occupancy inspection or mutation of historical artifacts.

Current assignment ends at a clean CPU candidate with exact future argv, immutable qualifier, focused checks, real fixture counts/bounds and remaining live risks. Lead subsequently accepts and releases the bounded slice/finite pair under existing user authority; no further user approval is pending.
