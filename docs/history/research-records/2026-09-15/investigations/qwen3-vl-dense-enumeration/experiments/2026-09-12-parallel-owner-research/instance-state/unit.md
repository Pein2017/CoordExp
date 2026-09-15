# Owner-grounding cached-carrier intervention

Status: **first panel complete, lead-accepted and closed**; see`results.md`.
The separately authorized amplitude-control successor has its own unit and
does not reopen this panel. Owner:
`astra_high_instance_state`. Owning portfolio: `../portfolio.md`.

## Question and contrast

At a fixed emitted row, can its cached state carry visually grounded owner
information that causally changes subsequent free owner coverage independently
of its literal category, coordinate tokens and position?

Keep Stable50, the original image, exact text, sequence length and native MRoPE
unchanged. During donor prefill, block only the completed row's queries from
reading image keys in A's predicted region, a same-class wrong-owner B region,
or an empty background region. All three masks remove the same key count.
Transplant only a named row-carrier slice of the donor cache into a fresh
untouched native recipient cache. Two fixed slices across all decoder layers:
the one `<|box_end|>` token, or the four coordinate tokens. No layer/head scan.

No optimization, new token, alternative checkpoint, future-answer donor,
image edit, spatial target answer, or GT-derived intervention box is allowed.
The B-control may use a different saved model-predicted box, verified visually;
its textual row is never added to the recipient history.

## Actual continuation seam

Native prefill ends at the completed row. A common already-natural next-row
opener is supplied to all arms, including untouched native. Native
`model.generate` consumes this single uncached opener with the patched cache
and then freely generates the suffix. This common structural token is not a
category/coordinate answer. It is needed to ensure the returned cache affects
fresh computation instead of attempting to alter already-computed logits.

The baseline is the existing `generate_continuations` caller with the identical
history plus opener. Exact suffix equality is required for both self-transplant
carriers. The cached route must observe input_length1 and the exact prefill
cache length at its first forward, and perform no new image encoding. Every
masked donor must consume the row-query/image-key mask exactly once per native
text-attention layer. Image features are recomputed identically for donor
prefills, not borrowed from a differently painted/resized image.

## First real slice

One exposed clean-control image9813, original832x1248 photo: A is the seated
driver on the left; B is the foreground woman beside the horse; background is
empty lower sand. Both A/B boxes are saved Stable50 predictions and were
visually inspected on the original image on2026-09-12. A's first row is9tokens.
Prefix includes1362native prompt tokens. Selected key counts are24/24/24.
Closing carrier is absolute position1370; coordinates1366..1369. Row-query
positions1362..1370. The supplied opener occupies position1371.

Inputs: source packet
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/packet-repair-01.json`.
Anchor: unchanged Stable50
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/training/adapter`.
The packet freezes native base, FP32/unmerged DoRA, special-token payload,
SDPA, no resize, native grammar, RP1 and all input identities.

32free tokens per smoke branch:1untouched +2self +6masked branches =9
continuations, at most288decode forwards plus4standalone prefills;5image
forwards;1model load. Peak estimates, not measurements: below30GiB GPU
allocated,24GiB hostRSS, under20minutes after model load. These are feasibility
estimates, not an overall portfolio cost limit. PhysicalGPU5 only, one producer.
Records, mask consumption, exact suffix IDs, cache delta magnitude, model and
input identities, full log/exit, wall time and lifecycle resource peaks persist.

First launch command, after root grant:
`CUDA_VISIBLE_DEVICES=5 python -m probes.parallel_owner_research.instance_state smoke --out-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/instance-state/smoke-v1`.

## Panel admission and stop

Up to4exposed histories total; admit current history-dominant/mixed/clean
examples only when emitted A and same-class B are real visually separable
owners and image-key sets are nonempty and count-matched. Unknown corner
bottles or degenerate chair boxes are not owner truth. No replacement search
outside the named exposed history bank. The final panel packet is frozen
before panel GPU work; root accepts the real slice and admission first.

Root's2026-09-12 ruling changes the final primary horizon to the full remaining
3084-token action limit; first512free tokens are secondary. This is fixed
before final panel launch and does not alter the32-token smoke. Preserving
literal history and common opener means no EOS/admission inference. Every
cell records free physical-owner A/B behavior, annotated-owner sets at
IoU50/60/80, any-class strict later-row repeats at native-pixel IoU>.95,
geometry/malformed drops, raw starts, EOS and cap. Prefix owners are not free
recoveries. Merely changing coordinates, category, count, EOS or likelihood
does not establish owner-specific state. Effects shared by B/background are
nonspecific under this contrast. A null excludes only this row-local
visual-grounding/cache intervention, not distributed owner state or architecture.

Stop after this frozen panel's technically verified reduction and bounded
physical-owner review. Technical invalidity remains separate from null.
Changes to estimand, checkpoint, carrier family or panel require root ruling.

## Preparation verification

`python -m pytest probes/parallel_owner_research/tests/test_instance_state.py -q`:
3passed. Tests check exact row-query/key masking locality, equal-count spatial
controls, exact self-transplant and untouched-cache-position preservation.
`python -m probes.parallel_owner_research.instance_state prepare`: one-case
candidate packet, all region key counts24. No model or GPU was loaded.

## Real slice result and frozen panel candidate

Root granted smoke-v1 onGPU5. One invocation exited0:9continuations,
292model forwards,5vision forwards,31.68seconds,9,692,463,104bytes GPU allocated,
9,950,986,240bytes reserved,9,712,848,896bytes peakRSS. Both self-carrier
continuations exactly match the untouched native32-token suffix and the saved
native suffix. Every masked prefill consumes exactly layers0..27 once; all
three masks use9queries and24keys. Nonself slices contain55,292..221,183changed
scalars; all eight patched32-token suffixes happen to equal native. This is
successful actuation/parity with no short-horizon effect, not owner-state null.
Cold native-parser/global-matcher consumer reloaded all8cells successfully.

Smoke evidence: raw root `smoke-v1/receipt.json`, `9813-masks.json`,
`consumer.json`, outer `smoke-v1.log` and `smoke-v1.exit`. Exact executed source
is retained as `smoke-v1/runner.py`, SHA256
`fd70a568d64b50e0cc0dc7b2da1b26d9cf9497a5998aedc8151ce0204b0b3c9b`;
subsequent panel-only code additions do not rewrite it.

The final candidate panel is3histories:9813clean driver,417044first physically
supported clipped donut,477415first real left-edge chair. Keys per matched
region are24/2/25respectively. B boxes on the two loop images come from
retained positive32 model predictions; root accepted their use as spatial
controls only, with no B text or alternate model/cache donor. Actual A/B and
background coordinates and source proofs are frozen in `panel.json`.

39654is excluded: its bad-history bananaA region is on purple fruit, not a
banana.351017is excluded: its tiny upper-left bottleA lacks convincing
physical bottle support. Those exclusions mean the panel does not cover the
current strongest history-dominant/mixed failure types; no substitute owner
label is manufactured. Later narrow donut strips and collapsed chair boxes
are not equated with the physically supported initial owner.

Final design:3untouched +6self +18masked =27continuations,15vision prefills,
one load. Hard maximum82,839model forwards under full remaining3084rule,
observed short-slice throughput suggests several minutes to tens of minutes,
estimated under15GiBGPU/16GiBRSS. No changed layers, heads, masks or cases are
selected from final outcomes. Stop after frozen whole-suffix and first512
reduction, physical-owner review, and root acceptance.
