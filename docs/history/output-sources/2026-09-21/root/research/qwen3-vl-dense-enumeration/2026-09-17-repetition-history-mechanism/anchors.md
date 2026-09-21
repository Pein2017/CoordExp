# History-exposure anchor packet

Status: candidate ready for the first production-shaped paired measurement. This packet is CPU-prepared from the accepted 2026-09-17 row-branch evidence; it launches no model work. The executable details and SHA-256 bindings are in [`anchors.json`](./anchors.json).

## Frozen construction

For the accepted bird case, construct `H_base || X || Y || A || B`, with `X,Y ∈ {A,B}` and the four assignments `AA`, `AB`, `BA`, `BB`. Every complete row is exactly 9 tokens, has the same `bird` serialization, and contains no EOS. This fixes row length, anchor set, and final A/B positions while exposing relative history placement. The packet does not use a comparable-displacement gate.

`H_base` is the first four complete native rows of image 309264 (`token_offsets=[0,36]`, 36 tokens, digest `a405536c300235c00691cf910ff1a0dfdac26020457444b3d6f0c64ac58a9747`). The prefix is retained even though it may already expose the same lower-left A identity; that is reported exposure, not an exclusion rule.

The native onset reference is image 309264 native raw at `[45,54]`, generated row index 5: the first complete repeat of A. The preceding A row is at `[36,45]`. Both are bound to `runtime/309264/native/raw.json` (`ab6dfbff93d5b204cab7181b81e8efb35a060c42841381bbb578e41ad59fa172`).

| Anchor | Physical binding | Box | Exact row source and relative offsets |
| --- | --- | --- | --- |
| A | `physical:left_lower_cage_bird`, accepted lower-left cage bird | `[0,782,93,902]` | native raw `[36,45]`; 9 tokens |
| B | `known:367404`, accepted known bird | `[637,552,701,624]` | distinct raw `[45,54]`; 9 tokens |
| C (optional) | `known:42098`, accepted known bird | `[688,552,766,624]` | distinct raw `[54,63]`; 9 tokens |

B and C are both present in the accepted distinct branch. C is optional and does not change the four `X/Y` assignments. Supplied A/B/C owner identities remain excluded from autonomous post-release credit. The accepted same-owner alternate A expression is `[0,781,96,903]` at `[45,54]` in `runtime/309264/same/raw.json`; it is available for the one permitted alternate-expression check.

Physical evidence is bound to `review/309264-native-future-detail.png` (A), `review/309264-known-detail.png` (B/C), and `review/309264-right-detail.png` (C plus the unresolved neighboring/feeder review debt). These are review evidence, not new labels. The two free white-bird rows and the feeder-region row retain the existing one-owner/unknown ambiguity.

## Book case

`386313-book` is HOLD. Existing native onset is `[109,118]`, row index 12, in `runtime/386313/native/raw.json`; the accepted same-owner teal/green spine expression is `[554,629,566,692]` in `runtime/386313/same/raw.json`. The existing distinct route is a `clock`, not a same-class B. The accepted review does not bind a separate physical book spine with sufficient identity certainty; the right-shelf cluster has unresolved individual boundaries. No B is invented from IoU and no broad search follows.

## Historical pointers

- [`research/questions/history-repetition-stopping.md`](../../../../research/questions/history-repetition-stopping.md) — nearest current synthesis and boundary.
- [`2026-09-11-small-owner-repeat-origin/results.md`](../../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-11-small-owner-repeat-origin/results.md) — accepted random-image/history result: later supplied histories can sustain old-category repetition; coordinate translation is not a reliable repair.
- [`2026-09-11-checkpoint-history-cross/results.md`](../../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-11-checkpoint-history-cross/results.md) — accepted checkpoint/history cross: history and checkpoint can enter or escape repetition through distinct conditional routes; no single mechanism.

The complete JSON is the source of truth for exact token IDs, row text, offsets, raw artifact hashes, image hashes, and review-artifact hashes.
