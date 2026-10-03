# Round07 fixed observations: terminal candidate

The single released package completed with **exit 0**. Both endpoints repaired
**6/6** fresh baseline-illegal contexts and retained **4/4** baseline-legal
contexts. Relative to the shared fresh anchor, Gmass1 exchanged 10 gained / 12
lost category-correct owners; saved Gmax1 exchanged 15 gained / 22 lost.
Gmass1 had more invalid rows, repeats, generated tokens and capped outputs.
These are observations on the frozen cohort; lead scientific disposition is pending.

Worker `01a0ff2d-e098-74e2-aa11-c65e21e558fc` remained GPT-6.1-Sol/high.
Execution source stayed clean/detached at
`2876f43db0d81569937a4ff1142129039204d038` in `greedy-prefix-native-01`.
The [lead ruling](lead-ruling-01.md) owns CPU acceptance and exact release.
No execution source, runtime, predecessor or scientific protocol was edited.

One Gmass update used six equal 1/6 losses at original7511 then351017, each
repeated three times. Losses were 0.0477066040 three times and 0.0451068878
three times (mean 0.0464067459). All 590 gradients were finite. Preclip norm
was 5.2138676643, clip1 applied, coefficient 0.1917961622, one AdamW step
at LRs1e-5/5e-6. The ten anchor and ten Gmass HF diagnostics were no-grad;
checkpoint publication preserved optimizer/parameter state, restored training
mode and released HF resources before native execution. Accepted Gmax1 was
never retrained. Its historical preclip norm93.15590668 and full590-gradient
record are reused with provenance; its native readout here is fresh.

| Context split | Fresh illegal | Fresh legal | Gmax repaired / retained | Gmass repaired / retained |
|---|---:|---:|---:|---:|
| 7511-626/original | 1 | 0 | 1 / 0 | 1 / 0 |
| 7511-626/training_neighbor | 1 | 1 | 1 / 1 | 1 / 1 |
| 7511-626/held_out | 1 | 1 | 1 / 1 | 1 / 1 |
| 351017-1507/original | 0 | 1 | 0 / 1 | 0 / 1 |
| 351017-1507/training_neighbor | 2 | 0 | 2 / 0 | 2 / 0 |
| 351017-1507/held_out | 1 | 1 | 1 / 1 | 1 / 1 |

Original351017 was already legal at a literal zero-margin tie (bin966), so
its repair denominator is **0**. Its emitted tie is preserved; no manufactured
error contrast or retry was used. All native score margins/masses/bins,
including diagnostics at neighbors, remain in the immutable terminal.

| Original prefix / endpoint | Emitted bin | Legal mass | Native max margin |
|---|---:|---:|---:|
| 7511-626 / anchor | 999 | 0.953660023 | -0.125000000 |
| 7511-626 / gmax1 | 982 | 0.988774924 | 1.625000000 |
| 7511-626 / gmass1 | 972 | 0.989932413 | 1.500000238 |
| 351017-1507 / anchor | 966 | 0.953378064 | 0.000000000 |
| 351017-1507 / gmax1 | 966 | 0.980499946 | 1.125000000 |
| 351017-1507 / gmass1 | 966 | 0.981712761 | 1.000000238 |

Natural outcomes use the fixed18 images /570 labels and unchanged
class-agnostic cardinality-first IoU>=.5 assignment plus exact category
comparison. Geometry-only and category-correct known owners are separate.

| Natural outcome | Anchor | Gmax1 | Gmass1 |
|---|---:|---:|---:|
| Category-correct owners | 274 | 267 | 272 |
| Geometry-only owners | 275 | 269 | 274 |
| Valid rows | 652 | 665 | 838 |
| Invalid rows | 430 | 193 | 555 |
| Literal complete repeats | 502 | 284 | 739 |
| Literal valid repeats | 89 | 100 | 215 |
| Near-repeat occurrence pairs | 92 | 987 | 1026 |
| Malformed outputs | 2 | 1 | 3 |
| Unmatched rows | 377 | 396 | 564 |
| Category disagreements | 1 | 2 | 2 |
| Natural generated tokens | 9922 | 7924 | 12729 |
| EOS | 16 | 17 | 15 |
| Caps | 2 | 1 | 3 |

| Owner transition | Category gained / lost / retained | Geometry gained / lost / retained |
|---|---:|---:|
| gmax1 | 15 / 22 / 252 | 16 / 22 / 253 |
| gmass1 | 10 / 12 / 262 | 11 / 12 / 263 |
| gmax1-to-gmass1 | 18 / 13 / 254 | 18 / 13 / 256 |

Per-image gained/lost/retained IDs, all burden deltas, stop/length outcomes
and saved-token divergence/visitation are retained in `package-01/complete.json`
and the bound terminal candidate. Gmax1-to-Gmass1 is a direct endpoint comparison,
not a sequential training trajectory. Unmatched is neutral; repeat occurrence
pairs are not physical entity counts.

| Serialized component | Gmax L2 | Gmass L2 | Gmax relative L2 | Gmass relative L2 | Elements |
|---|---:|---:|---:|---:|---:|
| DoRA | 0.0422796393 | 0.042283972672 | 3.4239215243e-05 | 3.4242724527e-05 | 18,006,016 |
| input-delta | 0.0035903631812 | 0.0035887530921 | 0.00074643648466 | 0.00074610174714 | 2,056,192 |
| output-delta | 0.00045240082543 | 0.0035989677012 | 0.00013493928805 | 0.0010734775713 | 2,056,192 |
| combined | 0.042434222906 | 0.042588332466 | 3.436401394e-05 | 3.4488814695e-05 | 22,118,400 |

CPU subtraction and accumulation use FP64, component plus exact serialized
key/shape alignment, and equal original checkpoint0 values. This is parameter
displacement, not functional matching. No rescaling or gradient matching occurred.

All four phases exited0 and the package completed its successful final readback
once. Counters:1 optimizer step,6 training replays,20 new HF diagnostics,30 native
scores,54 natural continuations,84 requests and30,605 new native tokens
(cap166,566). Active phases totaled500.7666s; package wall575.2395s.
One GPU/rank/sequence,context4456,2GiB KV and1800+30 phase/cleanup bounds
were preserved. No worker-added warmup, request, forward, dose or launch occurred.

| Phase | Active seconds | Parent peak RSS KiB | Reaped-child peak RSS KiB | Reported artifact bytes |
|---|---:|---:|---:|---:|
| native-anchor | 154.986612 | 1,238,856 | 6,802,488 | 57,250,179 |
| native-gmass1 | 188.142975 | 1,234,576 | 6,803,840 | 57,919,477 |
| native-gmax1 | 133.992249 | 1,238,956 | 6,800,256 | 56,786,502 |
| train | 23.644799 | 13,483,728 | 13,483,728 | 179,246,352 |

HF CUDA peaks:20,819,803,648 allocated /22,097,690,624 reserved bytes.
Actual package payload356,100,663 bytes at terminal capture. Native child CUDA
memory and internal startup/capture forward counts remain explicitly unmeasured;
startup defaults are unchanged. Fresh checks found all11 observed package,
native-parent, engine-child and tracker PIDs absent and all observed groups empty.
The package reports all four groups drained. Training phase PID was not separately
captured; this limitation is recorded in the cleanup receipt. No additional cleanup
action was needed.

Execution-local immutable evidence root:
`/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/mass-versus-ranking-07`.

- `native-terminal-candidate-01.json` SHA256 `9f48e159178032324dcd954ac98bfa55da03c9e0533e0e0758805ad99bb16bd5`.
- Released contract SHA256 `67e98ce0c666c15a08087be7282b95397b2d751f77eefc7b783f6d9632b55243`.
- `package-01/complete.json` SHA256 `43577c01ff640ad187a4e86d7e35d836ff469a3c7f94c28d0dbff649e6d832d5`.
- Phase receipts, checkpoint identities, exact argv/PID/start ticks/PGID/SID/session,
  successful readback and fresh cleanup are bound in the terminal candidate.
- Gmass1 weight identity `9fc7b57baee2168412001b97c4ac547f03d276dde1216cc3c406ac6aa7548b08`.
- Gmax1 weight identity `b2fd7427a6f698a0b64a47aa0643cbd6b0dea0fbf3eb0029f11b1d352c8bf1c9`.
- First real update and first scheduled anchor readout were reported directly;
  existing successful readback was reused without another model call.

CPU source/checked preparation remains at commit2876f43 and the canonical
`cpu-terminal-candidate-01.json`/`cpu-checks-final-01.json` pointers in state.
No CPU suite or qualification was repeated.

This is one fixed-update operational contrast on a fitted cohort. Gmax1 is one
saved realized training outcome, not a fresh training replicate; training
variability is unmeasured. The unequal margins, preservation and recurrence
readouts remain separate. No global determinism, objective dominance, causal
owner-loss explanation or physical false-negative recovery is claimed.
Worker execution is complete; lead final acceptance and scientific disposition
remain pending. No next unit is scheduled.
