# Shared literal-row execution seam

Status: implemented; 44 focused/legacy tests pass and real two-rank C2 plus
cold qualification passes. Candidate for root acceptance; see `results.md`.
Owner: `/root/astra_xhigh_shared_training`. Scientific packet values belong to
the lane owner and root, not this module.

The public preparation call is `probes.parallel_owner_research.training.prepare_packet`.
All arguments after `output_path` are required keyword arguments:

- `lane`, `anchor_input_path`, `margin_input_path`, `normal_keys`
- `positive_records`, `conditional_records`, `arms`
- `weights`, `denominators`, `optimizer`, `clip_gradient_norm`, `runtime`

Each literal record contains `record_id`, `example_id`, the old complete `image`
identity dictionary, `prompt_token_ids`, `prompt_token_ids_sha256`,
`prefix_token_ids`, and `target_token_ids`. Metadata such as owner/role/stratum
may be retained. A positive target is exactly one complete non-EOS row, and its
loss is the SUM over every literal target token. A conditional record additionally
contains explicit `kl_positions` relative to its own `target_token_ids`, and
`unknown_mask_policy: literal_positions_only`. Conditional prefixes are complete
literal histories; the engine does not append a positive or infer a history.

`arms = {NAME: {steps: [[{record_id: ID, weight: NUMBER}, ...], ...]}}`.
Every step is explicit. The list is replayed on every rank. Repeated entries
are exposures; the engine neither samples nor fills missing steps. All conditional
records are replayed once every step in every arm, independently of this list.

Both `weights` and `denominators` have exactly the keys `positive`,
`conditional_kl`, `normal_kl`, `margin`. Values are explicit finite numbers;
denominators are strictly positive. Positive NLL is a complete-row SUM;
each conditional/normal KL is a token mean of forward KL from detached Stable50
full-vocabulary references. Fixed normal keys are replayed once globally each
step. Every image margin is the maximum one-sided floor violation using the
CURRENT full-vocabulary non-target competitor, with the accepted source floor
table (including the 17 retained KL-only near ties). The known nine invalid
geometry tokens stay excluded by the bound original normal mask. No mask is
inferred from an unfamiliar/new row.

For replicated positives/conditional records, local scale is weight/denominator.
For sharded normals/margins, local scale is world_size*weight/denominator.
DDP's final average therefore yields the same explicitly declared global sum
on two or eight ranks, including uneven shards. `normal_keys` may be a subset
of the bound old bank; active lane images MUST be excluded. Both normal and
margin denominators must equal this selected image count. There is exactly one synchronized backward per step;
the preceding replays use no_sync. Normal partitions are `normal_keys[rank::world]`.

`optimizer` is the complete AdamW keyword dictionary (current proposal:
`lr=1e-5, betas=[0.9,0.999], eps=1e-8, weight_decay=0, foreach=false`).
`clip_gradient_norm` is explicit. `runtime` requires `world_sizes` and the
limits `max_rank_seconds`, `max_cuda_allocated_bytes`, `max_cuda_reserved_bytes`,
`max_rss_bytes`, `max_model_forwards_per_rank`, `max_image_forwards_per_rank`.
No limit is a portfolio spending cap; these are individual invocation bounds.

Root ruled common margin10 for both lanes, with history conditional10 on original
P+c→w states only; composition uses identical admitted incumbent successor
preservation states across arms, never A/B targets themselves. Composition with
no usable such successor must return to root for a common conditional0 ruling.

The anchor is the accepted old repeat-recovery input packet, validated read-only;
it binds unchanged Stable50, original base, source special embeddings, prompt
construction, normal56, and masks. The margin input is separately hash-bound.
The new engine and imported execution sources are hash-bound at preparation.
Execution remains fp32 SDPA, unmerged, model.eval(), language-only DoRA exactly
588 tensors / 18,006,016 scalars, fresh AdamW per arm. No event/negative sampling.

CLI: `python -m probes.parallel_owner_research.training verify --input PATH`;
`CUDA_VISIBLE_DEVICES=0,1 python -m probes.parallel_owner_research.training launch
--input PATH --arm NAME --world-size 2 --output-root PATH`;
then a separate cold check with one explicitly reserved visible GPU.

## Mechanical qualification request

Freeze two updates using old three literal positives, old three conditional
records, and unchanged normal56 + margin10. Run the new two-rank route once and
compare its global loss components, gradient/update norms, and saved adapter to
the existing eight-rank C2 oracle; byte equality is not assumed across reduction
trees. Replay all positive scores after a cold load and require exact discrete
counts plus <=1e-5 score deltas. If the old oracle lacks a decision-bearing
tensor, root may grant one eight-rank run of this SAME packet rather than infer
gradient parity from score equality. Preserve all raw failures and never use
the plumbing run as a scientific result.

Named risks: replicated-vs-sharded DDP scaling; variable histories and literal
causal positions; conditional mask/history leakage owned by lanes; four-times
larger per-rank normal reference cache at two ranks; Stable50 override and
saved/cold identity; current rather than frozen margin competitor; one-sync
DDP choreography; runtime/source fingerprint changes between prepare and launch.
