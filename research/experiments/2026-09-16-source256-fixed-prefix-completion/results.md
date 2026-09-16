# Source256 fixed Source-history completion: registered-dose closeout

## Decision

**No incremental value at the registered64-update dose; do not advance B.** Both arms completed mechanically valid training and all80 saved natural-readback shards were scored. B misses more known owners than canonical A on train and dev, loses more Source incumbents, and emits more malformed, strict-repeat and capped outputs. This closes this learning package at the frozen dose; no refresh, new seed, added dose or further visual review is authorized.

## Fixed evidence boundary

A is canonical SFT; B is50% canonical replay plus50% eligible fixed Source-history completion/canonical fallback. Both start from original Source step2444 with fresh AdamW and share train256/dev128, owner information, schedule and64 updates. Natural readback is empty-prefix greedy,RP1,cap3084,bs4. Source0,16 and64 were evaluated;64 is primary.

Processed data is `rescale_32_1024_bbox_len12000_xy_sorted`, not raw COCO. User stopped further view_images after120 of128 selected hypotheses:34 additions and one targeted seed-owner exclusion give1988 train owners; dev retains891. Eight selected but unreviewed hypotheses and other unreviewed hypotheses remain HOLD. Final143/256 eligible images produce1144/4096 correction presentations (27.9296875%), passing64/12.5% scarcity gates.

## Primary known-owner coverage

Global one-to-one class-agnostic IoU>=0.5; fixed bank, not exhaustive physical scene recall.

| Endpoint | Train matched /1988 | Train FN | Dev matched /891 | Dev FN |
|---|---:|---:|---:|---:|
| Source0 | 1300 (65.39%) | 688 | 620 (69.58%) | 271 |
| A16 | 1328 (66.80%) | 660 | 617 (69.25%) | 274 |
| B16 | 1303 (65.54%) | 685 | 603 (67.68%) | 288 |
| A64 | 1360 (68.41%) | 628 | 611 (68.57%) | 280 |
| B64 | 1321 (66.45%) | 667 | 607 (68.13%) | 284 |

B64 covers39 fewer train owners than A64, and4 fewer dev owners. Relative to Source, A64 nets+60 train/-9 dev; B64 nets+21 train/-13 dev. Both fail the dev non-regression criterion.

## Source incumbent accounting

| Endpoint | Train gained G | Train lost L | Dev gained G | Dev lost L |
|---|---:|---:|---:|---:|
| A64 | 95 | 35 | 17 | 26 |
| B64 | 96 | 75 | 24 | 37 |

Observed: train gains are almost identical (B96/A95), while B loses40 more Source incumbents (75/35). This supports a retention cost as the immediate accounting explanation for the lost net benefit; it does not isolate a causal prefix/EOS mechanism. The intervention changes both conditioning history and credit allocation.

## Separate output burdens

| Endpoint/split | Valid rows | Malformed rows | Strict repeat rows | Cap debt | Invalid geometry |
|---|---:|---:|---:|---:|---:|
| Source0/train | 2323 | 794 | 467 | 4 | 0 |
| Source0/dev | 1103 | 58 | 111 | 0 | 0 |
| A64/train | 2343 | 152 | 432 | 2 | 0 |
| A64/dev | 1065 | 18 | 96 | 0 | 0 |
| B64/train | 3178 | 1445 | 1153 | 8 | 0 |
| B64/dev | 1257 | 295 | 228 | 1 | 0 |

Strict repeat is the declared geometric proxy (class-agnostic IoU>0.95 to an earlier row), not a new visually confirmed identity verdict. Cap debt counts images; row counters count rows and can be concentrated in looping outputs. Annotation-unmatched predictions remain unknown, not confirmed false instances. Endpoint physical false-instance count is unresolved because the user stopped further visual review. This limits physical claims but cannot rescue B from its already-failed fixed-reference coverage/retention gates. Class-consistent IoU50/60/80 and full per-image records are retained in result.json.

## Mechanical acceptance and recovery

A/B completed64 applied updates each,4096 presentations/logical forwards and2048 model calls; checkpoints16/32/64 have distributed state consensus. Final bank hydration, masks, exact prefix/full suffix/EOS and bs4 token parity were qualified before main launch.

Original evaluation controller exited after61/80 saved shards, contemporaneous with Codex-wide task Shutdown. This is strong lifecycle evidence, not proof of an exact signal; kernel OOM evidence was unavailable. A separate tmux producer recovered only19 missing shards. Lead verified all61 retained hashes unchanged and all80 shards completed atbs4. No model retraining or successful-shard regeneration occurred.

The CPU reducer initially supplied tokenizer.json where AutoTokenizer requires its parent directory. A caller regression failed before the one-line fix and all7 evaluator tests passed afterward. A versioned packet rebinds only the corrected evaluation producer and reducer packet argument; original packet/failures remain. The actual reducer completed successfully and cold-admitted every shard; G/L arithmetic was independently checked.

## Decisive artifacts

- [Fixed result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/runtime/main-v1/evaluation/result.json)
- [Final preparation](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/preparation/source256-admitted-v1/preparation.json)
- [Visual admission](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/lead/visual-review-v1/acceptance.json)
- [Main release](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/lead/main-release.json)
- [Reducer repair binding](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/runtime/main-v1/evaluation/reducer-repair.json)
- [Interruption diagnosis](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/runtime/main-v1/evaluation-recovery-v1/interruption-diagnosis.json)

Scientific scope: one paired seed, this processed256/128 panel and this history/supervision package. No statistical significance, pure mechanism or population-wide generalization claim. Source exposure and historically used dev status remain as documented in unit.md.
