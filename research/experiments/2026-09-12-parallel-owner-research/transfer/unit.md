# Fresh transfer: fixed256 and separate exposed history cross

Status: Full256×3 and separate12-cell cross complete, cold-verified;
scientific stop reached. Root final acceptance pending; [results](results.md)
own interpretation. No further model work is running or requested.
Owner: `/root/astra_high_fresh_transfer`; root owns acceptance and first model grant.

From unchanged Stable50, does accepted C32 improve annotation-relative natural
owner coverage and burden on a fixed, previously unselected256-image panel?
The primary contrast is C32 minus Stable50; D17 minus Stable50 and C32 minus
D17 are explanatory. Strongest alternative: changes only repair the exposed
training/screening images or exchange coverage for false positives/repetition.

## Frozen population and identity

Raw authority: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/transfer/`.
`selection.json` contains exact IDs, SHA-ranked selection, complete projected
exclusion sets and source hashes. ID-list SHA256:
`bee7109771c431f88f6d27694c3beaa321152da80df33e6c3f16205a5f74a66a`.
Source: existing COCO `rescale_32_1024_bbox_len12000/val.coord.jsonl` and its kept
rescaled image bytes.4952 source images,1051 excluded,3901 eligible; select256
by SHA256(`parallel-owner-fresh256-2026-09-12:` + decimal image ID). No image
count, object density, model output, visual result or adaptive backfill filter.
Native serialization uses the existing `rescale_32_1024_bbox_len12000_xy_sorted/val.coord.jsonl`
required by the unchanged geo_sorted_xy renderer. Both source hashes are bound;
selected image paths/dimensions and complete annotation-object multisets must
match exactly. Only annotation order differs; selection is not reopened.

The focused reconciliation covers1144 identity-bearing investigation raw
input/selection/run/visual manifests and investigation-side JSON records.
It includes prior training, screens, candidate sets, visual adjudication and
earlier holdout/confirmation panels beyond latest384. This is **provenance-
qualified freshness**, not an exhaustive raw-log/outside-investigation census
or a claim of pretraining/SFT-disjointness. No other portfolio lane may consume
the selected panel for selection, training or adaptive diagnosis.

`packet.json` binds actual retained Stable50/C32/D17 adapter bytes, retained
root acceptance/endpoint packets, base, original embedding delta, prompts,
native media identity, GT and code hash. Stable50's historical receipt uses
`coordexp-swift` identity metadata while the inspector uses `coordexp-infras`;
all payload/semantic/tensor fields must match exactly, and both identities are
retained. This is not a Source/checkpoint substitution.

## Route, resource estimates and acceptance

Producer: `probes/parallel_owner_research/transfer.py`; accepted native
`runtime.build_request/load_policy`, `build_requests/native_record`, generation,
global matcher and strict-repeat scorer are reused unchanged. Explicit labels,
no global ARM monkeypatch. FP32 SDPA, unmerged DoRA, original embedding,
RP1/greedy/3084 cap, original prompts, no resize, no batch approximation.

Physical GPUs6,7; one independent worker per GPU and checkpoint, no NCCL or
eight-rank producer. Fixed full workload768 natural continuations,6 model loads;
2,368,512 maximum generated tokens follows the frozen cap, not a portfolio
spend ceiling. Initial estimate35–90minutes wall and16GiB CUDA/18GiB RSS per
worker, qualified from actual counters before full. Persist exact model/image
forwards, generated tokens, model loads, peak allocated/reserved CUDA, RSS,
elapsed/allocated GPU seconds and outer exit codes.

Qualification uses2 **old exposed** images on each of the3 checkpoints,
6 continuations and3 loads on GPU6. Require exact retained complete natural
token equality, live checkpoint identity, CPU/GPU prompt/media/grid parity,
native cold parser/scorer parity and complete outer exits. Fresh256 is not
read twice. Root grant precedes qualification; root accepts real slice before
full execution. One live producer per phase; preserve technical failures and
do not convert incomplete populations into scientific nulls.

Qualification receipt: raw `qualify/result.json`, SHA256
`770ede04f434e0d335ced689a79480329065dc45965b2ba2626b2100f6b174d7`.
All6 archived natural complete-token comparisons and native cold consumers
passed. Packet SHA256
`3ae84969cce7b60629ea185699e1ec63ebd1934ad1464c4c993e4225bce83739`.
The first qualification attempt is retained as `qualify-technical-invalid-01`:
one verified model load, zero forwards/outputs, then old-record absolute-path
rejection. The same-contract relative-path fix passed a native-loader RED/GREEN
regression and all5 old native request prompt checks; no image or ID changed.
The previous packet is preserved, not overwritten without history.

Executable commands, from the named worktree:

```bash
python -m pytest probes/parallel_owner_research/tests/test_transfer.py -q
python -m probes.parallel_owner_research.transfer launch --phase qualify
python -m probes.parallel_owner_research.transfer merge --phase qualify
python -m probes.parallel_owner_research.transfer launch --phase full
python -m probes.parallel_owner_research.transfer merge --phase full
python -m probes.parallel_owner_research.transfer launch --phase cross
python -m probes.parallel_owner_research.transfer merge --phase cross
```

## Primary evidence and fixed stop

Cold exact ordered256 per label and checkpoint fingerprint; complete token/
text/action/media identities. TP/FP/FN/F1 and paired owner gains/losses/retained
at IoU50/60/80; all parsed rows, geometry-invalid plus other malformed drops,
row starts, strict class-blind native-pixel IoU>.95 later-row-once repeats,
EOS/caps and generated lengths. Report paired-image deltas, concentration and
image-paired percentile bootstrap95% intervals (10000 draws, seed20260912),
not owner-independent resampling or a new complete-scene truth claim.

Stop after one frozen256 read across the three fixed checkpoints and the
separate12-cell history cross, with cold verification and interpretation.
No adaptive expansion, checkpoint selection from results, ensemble, matcher/
GT edits, architecture/training/promotion or new coefficient/dose branch.

## Secondary12-cell history cross (never in fresh denominator)

Root approved3 exposed images25274/511251/417044 × recipient C32/D17 × donor
C32/D17. Use archived natural C/D tokens. Locate first unequal token; each
donor prefix ends at the first complete-row end at/after that divergence.
Entire prefix is intervention; no GT-selected boundary. If donor EOS arrives
before a complete row, freeze that prefix as absorbing EOS with zero free
tokens; never invent a row. If neither row completion nor EOS exists, report
secondary infeasibility rather than changing the boundary.

Each diagonal self-history must exactly reproduce the archived remaining
tokens under3084 minus inherited-prefix length. Record full intervention
outcome, forced-prefix owner/localization score and **free suffix only** global
matching separately; supplied rows never receive free credit. Also report
new free owners after removing forced-prefix owners. Cross results measure
history-conditioned behavior, not autonomous fresh transfer or training.
