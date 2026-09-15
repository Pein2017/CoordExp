# Direct C-D0 results

## Final disposition

- Mechanics: `MECHANICALLY_VALID`.
- Science: **`SCIENTIFIC_STOP_DIRECT_C_D0`**.
- Action: stop this registered objective.  Do not add the old G0 permutation,
  a refresh, another seed, checkpoint selection, or a larger cohort as a
  rescue inside this unit.

On the frozen 128-image, 891-observed-owner screen, D0 matched 24 fewer IoU50
owners than C.  It gained 19 C-missed owners but lost 43 C-covered owners,
retained only `586 / 614 = 95.44%` of Source-covered owners, lost twice as many
Source owners as C (`28` versus `14`), and produced one length-capped decode
where C produced none.  All four registered GO predicates are false.

The authoritative machine-readable verdict is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/verdict-v1.json`
(SHA-256
`7c8dbb95ba913b6d0d39cc9d4c1afb3b2a95b7be492682ccbbc02790b2a218c6`).
It contains the exact gain/loss/retention owner identities and validated
Source/C/D0 run provenance.

## Natural-greedy owner coverage

| Arm | IoU50 | IoU60 | IoU80 | IoU50 owner micro | IoU50 image macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 614 | 585 | 451 | 0.6891 | 0.7941 |
| C | 630 | 598 | 461 | 0.7071 | 0.8041 |
| D0 | 606 | 578 | 438 | 0.6801 | 0.7916 |

The paired D0-minus-C net matched-owner counts are `-24`, `-20`, and `-23` at
IoU50/60/80.  The descriptive paired-image-bootstrap 95% interval at IoU50 is
`[-0.0603, 0.0000]` owner-micro recall; it is not verdict-gating.

| Arm | Generated rows | Dropped predictions | Invalid predictions | Natural order violations | Cap/nontermination |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 1,103 | 58 | 0 | 34 | 0 |
| C | 1,388 | 37 | 1 | 39 | 0 |
| D0 | 1,404 | 321 | 0 | 45 | 1 |

Natural disorder is monitor-only.  Unmatched predictions remain unknown and
are not counted as hallucinations.  All 128 rows in every arm were parsed;
D0's single cap stopped by length and the other 127 stopped at natural
`im_end`.

## Full-run mechanics

Both arms completed 26 optimizer steps at world size eight with 208
microsteps per rank.  Every update passed the all-rank finite-gradient gate.
Only the shared 588-tensor language-DoRA adapter was saved; no merged model was
created.  Fresh cold-process unmerged readback exactly reproduced each saved
adapter.

| Arm | Loss, step 1 -> 26 | Launch wall | Peak GPU MiB | Final adapter fingerprint |
| --- | ---: | ---: | ---: | --- |
| C | 1.688076 -> 1.482401 | 222 s | 7,427 | `5eed9a1eeccbd8117ab7fec7c9c63fafc1e26b1f25233f6a4c56908c88d4bb95` |
| D0 | 3.468970 -> 3.280068 | 303 s | 9,317 | `947fc2e76a1b13efe55f2e009cd8b2beb14e81330129491de037242d9e37b4b1` |

D0's positive loss decreased from `2.320821` to `2.172327`; its preservation
loss decreased from `1.148066` to `1.105345`.  Thus the negative natural-decode
result is not an optimizer no-op.  The cold-readback receipt SHA-256 values are
`319e8b50ec46262d096d1a9b8fe9a2768e01e2a47dbf60f876bcd9c854abc1ff`
for C and
`247930ac93f89c9e45ed5a047f596ba24fff8d523512e96b10c242a51c799213`
for D0.

| Arm | Decode seconds | Generated tokens/s | Peak allocated GPU bytes |
| --- | ---: | ---: | ---: |
| Source | 995.5 | 11.12 | 10,952,750,080 |
| C | 1,233.2 | 11.06 | 11,310,063,616 |
| D0 | 1,549.9 | 10.50 | 14,924,035,072 |

## Claim boundary

This one-seed result rejects promotion or paired replication of the frozen D0
bundle at its registered dose.  It does not show that internal DoRA learning,
actual-prefix objectives in general, refreshed-prefix variants, or full COCO
training cannot work.  It also does not estimate precision or full-scene
recall under missing annotations.

## Prior smoke

The authoritative machine-readable smoke receipt is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/smoke-result-v1.json`
(SHA-256
`d78eac47f89af6c7ad515a2fd1e5d3ad2a42d4da147ecbdc65c145e933877b44`,
receipt ID
`ade46e6938bec03586ff69c03bbf2c5971105bb25a834d68cbffe5ed6f951e59`).

### Smoke mechanics evidence

Both arms completed one real optimizer update at world size eight and wrote
only a 72,111,984-byte unmerged language-DoRA adapter.  All 588 A/B/magnitude
tensors changed from Source.  Independent-process readback exactly matched
all saved tensors to the materialized adapter state, observed no merged
adapter, and produced deterministic cold decodes.

| Arm | Events/presentations | Total loss | Gradient norm | Peak GPU MiB | Final adapter fingerprint |
| --- | ---: | ---: | ---: | ---: | --- |
| C | 64 | 1.565119 | 1.687046 | 7,027 | `7f5f1bc081014372104c9bf44344580b56a43aacd3a5e388a36fb5d0f9a2cc38` |
| D0 | 32 positive + 32 preservation | 3.744413 | 5.432662 | 6,959 | `8893b3a1e9379682ad444efe4de3c8858cccbb3a28789e2ad75dbcfa49c3ac7f` |

These smoke values established execution, update, persistence, memory
headroom, and unmerged reload only; they did not enter the scientific verdict.
