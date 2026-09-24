# Candidate: legal first x2 does not release these repeats

Status: worker candidate, lead acceptance pending. The exact [unit](unit.md) froze images 885 and 5586 at their original first illegal x2 query, plus native, forced-47 and forced-1 branches. Same promoted-live mature source, original image/prompt, RP1 greedy, 384 new tokens including the first choice. Only that first x2 token was forced. No free continuation beyond the fixed horizon, training, decoder mask, new arm or physical-owner review.

Both legal values make the first box valid, but exact complete-row repetition persists in **all four forced releases**. The correction does not robustly interrupt repetition at these two boundaries. This locally weakens numerical-invalidity as its sustaining cause; it does not rule out legality influencing other histories. All six releases hit the 384-token cap without EOS, so earlier stopping or natural completion cannot be counted as recovery.

| Image | First x2 | Complete rows | Valid geometry | Repeat prior prefix / within release | First repeat | Max contiguous run | Equality x / y | Parser drops |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 885 | native 0 | 43 | 0 | 0 / 40 | row 3, within | 30 | 43 / 12 | 44 |
| 885 | forced 47 | 43 | 2 | 2 / 39 | row 1, prior | 30 | 41 / 11 | 42 |
| 885 | forced 1 | 43 | 1 | 0 / 40 | row 3, within | 31 | 42 / 11 | 43 |
| 5586 | native 0 | 43 | 40 | 0 / 36 | row 3, within | 18 | 3 / 0 | 4 |
| 5586 | forced 47 | 43 | 22 | 2 / 35 | row 1, prior | 16 | 21 / 3 | 22 |
| 5586 | forced 1 | 43 | 21 | 0 / 34 | row 3, within | 14 | 22 / 5 | 23 |

No reversal-invalid complete boxes appear in these six windows. Each parser-drop count includes one malformed trailing object span at the cap; all other drops are geometry-invalid complete rows. Rows are counted **before** parser exclusion. In image 885, forced 47 first yields `person [0,0,47,22]`, which exactly repeats a prior-prefix row; forced 1 yields `[0,0,1,22]`, but both soon return to repeated `[0,0,0,38]` and later `[0,0,0,0]`. In image 5586, the native first three rows are `[0,0,0,47]`, `[0,0,0,38]`, `[0,0,0,38]`. Forced 1 first yields `[0,0,1,91]`, then `[0,0,0,93]` twice. Forced 47 first yields a prior-prefix repeat `[0,0,47,38]` twice. Its later behavior has more invalid geometry than matched native. These are exact row identities, not physical-owner assignments; annotation-unmatched outputs remain UNKNOWN. No separate known-annotation coverage calculation was needed for this local recurrence question.

The newly generated native first 64 tokens match the accepted source trace **exactly** for both images. At the first x2 query the loaded source's coordinate full-vocabulary log-probabilities match accepted readouts with maximum difference 0, the full-vocabulary winner is coordinate bin 0, and the cached generation's first full-vocabulary logits match exact prefill with maximum difference 0. The true generation caller verified every later emitted token equals argmax of unmodified raw logits. The forced branch uses a literal one-token extension followed by a fresh independent cached generation; it does not reuse another branch's cache. Actual prefix, media, prompt and grid checks passed. A focused CPU mutation rejects a persistent override and a force at the wrong token index.

All six planned cells exist, no HOLD. Each has every emitted token and raw/policy log-probability, full release text, complete raw rows, parser drops, cap and recurrence fields. Saved-only reduction and fresh replay are byte-identical, SHA256 `16346301f2e8acbda6b8ffc798e4f99f9ff2f27bb88e3c5bffca0656be6edb06`. The independent saved readback checks input/source bindings, seven current/captured code pairs, all cells, numerical finiteness, first-64 replay, first-token identity, recurrence arithmetic and terminal PID. The focused tests pass 4/4; research knowledge, both-root output-layout and diff checks pass. The accepted predecessor's missing historical preparation-source bytes remain disclosed; this execution captured current source directly and did not invent those bytes.

New package cost is 178.223715 allocated GPU-seconds and 2302 model forwards, including source loading and two prefill forwards; its model-wall duration is 178.223715 seconds. Common overnight cumulative GPU time is 216.201243 seconds and 2339 forwards from the unchanged start `1790182797.5338483`. One producer exited successfully and its PID is absent. No owned job remains. Retained output is under 1 MiB. Caps were not approached.

Reproduce the saved result with:

```sh
python -B -m probes.training_set_completion.coordinate_order_knowledge.first_illegal_release reduce --out /absolute/fresh/reduction.json
python -B -m pytest -q probes/training_set_completion/coordinate_order_knowledge/test_first_illegal_release.py probes/training_set_completion/coordinate_order_knowledge/test_zero_history_amended.py
python -B scripts/research/check_research_knowledge.py check
python -B -m src.artifacts.output_layout --root /data/CoordExp/outputs --root /data/CoordExp/.worktrees/research-probes/outputs
git diff --check
```

Producer commands were `python -B -m probes.training_set_completion.coordinate_order_knowledge.first_illegal_release prepare` and `... run`; the bound launch receipt has argv, source captures, source configuration, clock and PID. The output root is `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-first-illegal-coordinate-release/`. New maintained paths are this unit, `first_illegal_release.py` and its focused test. A transient candidate state file failed the knowledge check as an orphan because the catalog is peer-owned; it was removed, and the fresh check passes. Peer staged/dirty files and accepted predecessors were left unchanged. Lead owns scientific acceptance and the next decision.
