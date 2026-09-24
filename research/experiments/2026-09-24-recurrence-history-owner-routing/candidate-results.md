# Four-arm history-owner routing candidate

Status: **worker candidate, awaiting lead physical and technical acceptance**. The exact admitted four trajectories completed. No arm, sample, row, or model call was added. This tests local next-row routing under historical coordinate substitution; A0 remains a physically ambiguous replay baseline.

## Frozen binding and execution

- Protocol SHA-256 `3cef91a395beae853a1c1bd8d739ccb3ffc7126a99815707397a4d448ed0323e`; admission SHA-256 `41132cbefdbcd3eeb384b18a3b4c63c0a7e4f914c5050b605988e109b717049d`; preceding accepted row1 closeout SHA-256 `29ea9ff76c9128910f7bc57cea93a06036e166a9924acf7908797a4d4cf54c39`.
- Original fresh-18 full four-request batch, target index 3, train313465, untied step2444 FP32 SDPA. Source image SHA `bc28ebbcd6ad8015d7097959539c5069d4c7f92fa4431665d893b3d4b74c7f4b`; raw SHA `5c3758e2d1e677aa6af6b789d4850012f030c652bca8c826ca8fd57a7f14719d`; trace SHA `eea6a5b2d5ea16f9c3ce63cefdc31331b399c0e61ec7432b36256644fb83b5f7`; runtime receipt SHA `23e219bab7f87097f1f6342e723b32b2c2e03c8f57f5d0559fc4ffa55b9a5875`.
- CPU preflight SHA-256 `fd1ca3095e7a95874e4386462106ba4e314cfcc59768591184566ad4dfa7c12e`: 82 caller/receipt checks, 16 direct source captures, all five widths 1377–1381, 24,403,968 image tensor elements, native SDPA mask checked against installed consumer. The four-token historical write is confined to target raw offsets `[5,9)`; the original current header, companions including 4+t pad tail, positions and mask stay fixed. The accepted native five-step full-vector references and trace are bound.
- Raw root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-history-owner-routing/attempt-001`. Producer SHA-256 `4afda145679e1d1bb10827eadaaf622b90e783328e82c1c5f0d640d245175c13`. Child receipt SHA-256 `fe0a01f85f7ec34cfc1bd4123780e2bf8c7530b51eb3812dcb84344562b9e7d5`; independent CPU readback SHA-256 `614530d5de4213db98dbf228e0bec6a64feb59b4e5c7bd18301b7744a67b528d`; visual proxy record SHA-256 `24f6c653a9aedcc1ff5f42eefbd3f2caa051d06b57f3753e3574362cdfd65a52`.

## Complete generated rows

| Arm | Historical coordinate IDs at raw `[5,9)` | Greedy current-row IDs | Stop | Box bins | Worker visual proposal |
| --- | --- | --- | --- | --- | --- |
| Native A0 | 151670,151827,151947,152174 | 151675,151867,151887,152077,151649 | complete | [5,197,217,407] | inner bowl A1, previously lead accepted |
| Identity-write sham A0 | same four IDs, through write caller | 151675,151867,151887,152077,151649 | complete | [5,197,217,407] | same A1 |
| A1 donor, source row1 | 151675,151867,151887,152077 | 151670,151827,151947,152172,151649 | complete | [0,157,277,502] | broad A0-like extent covering inner bowl and outer vessel; physical owner **UNKNOWN/HOLD** |
| B donor, source row2 | 151827,151670,152085,151911 | 151847,152106,152130,152498,151649 | complete | [177,436,460,828] | central citrus juicer/other object proposal; lead review required |

All four rows have canonical syntax and ordered geometry. Native/sham boxes have IoU 0.883752 with reviewed A1 annotation716308 and 0.001785 with B annotation715278. The A1-donor box has IoU 0.503720 with A1 and 0.035537 with B, but the original image shows its broader pot/inner-bowl extent; IoU does not settle its physical owner. The B-donor box has IoU 0 with both reviewed bowl annotations. The source image and overlays are in the visual proxy record. The user's pot interpretation keeps A0 unassigned as an owner.

Both donor arms diverged from the native target at free token zero, so their later logits condition on different self-generated prefixes. The count-only prediction **fails exactly**: neither donor tail equals native. The proposed strict exclusion pair A1→B and B→A1 and strict anchoring pair A1→A1 and B→B are **not supported by these candidate visual outcomes**; lead physical rulings remain required, especially for the broad A1-donor result. No one-sided success, owner mass, literal copying, or abstract visited-owner ledger is inferred.

## Consumer and source gates

- Exactly 20 model, 20 vision forwards, 20 emitted target tokens, five per arm; zero free tokens beyond these rows. Native and write-sham completed and qualified before either fixed donor. No extra qualification forward.
- Every native full vector is **exactly equal** to the corresponding accepted row1 five-step vector (max absolute error 0), and every active native source trace passes. No after-EOS source trace is claimed for ended companion2. Sham target, all-batch vectors, row0 states, and greedy tokens are exactly equal to native (max error 0).
- Every arm's actual top-level input IDs, native mask, position IDs and cache position hash matches its independently reconstructed full-prefix input. All 28 text attention modules consumed the expected native 4D SDPA mask at every step. The observer confirms each donor's exact historical four-ID replacement and that the arm consumed only its own earlier greedy tokens. All companion layer states and full vectors match native at the same step (max error 0), while donor target historical states were allowed to change. CPU cold readback replays these checks from saved raw vectors and input tensors; it does not load the language model.
- At each donor's first decision, the global winner is the first ID shown above. Native first-step winner margin is 0.0322323 logit; A1-donor 0.1070824; B-donor 0.4593105. The full per-step logits, top two, gaps and FP64 probabilities are saved in raw tensors/readback; later distributions are descriptive only because prefixes differ.

## Cost and terminal state

- Planned outer charge 121.725604221 s, observed parent-monotonic outer charge **57.915783612 s** (outer-command SHA-256 `f8ecb94a58b9f961bbb0c1207d414da4b8cea0d3642d6336e9a455dceac87864`). Child internal interval 52.303539470 s. Conservative cumulative sequence charge: **0.29135763040168866 GPU-hours**, from accepted prior 0.27526991273168866 h. The user's removed elapsed-time ceiling was respected; time remains recorded.
- Peak CUDA allocated 9,934,806,528 bytes; peak reserved 10,659,823,616 bytes; peak RSS 9,497,304 KiB. Raw root after readback/overlays: 123,113,506 bytes across 69 files, below 512 MiB plan. Child exit 0; receipt status `candidate_complete`; terminal child PID 1812015 absent after completion.
- A first outer wrapper invocation failed **before child spawn** because the output directory for its log did not yet exist. It made zero model/vision/GPU calls and wrote no launch receipt. The corrected wrapper created that directory, then launched the only model attempt above. This prelaunch setup miss is disclosed for the lead's technical acceptance decision; there was no model retry.

Stop: finite four-arm package complete. Physical owner assignment and acceptance belong to the lead; no successor is authorized here.
