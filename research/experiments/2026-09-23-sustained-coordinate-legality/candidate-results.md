# Candidate: strict coordinate legality does not stop exact-row repetition

Status: worker candidate, lead acceptance pending. The [unit](unit.md) fixes one mature live-promoted source, the original first-illegal x2 prefixes of images 885 and 5586, 384 new tokens, and a probe-local **joint coordinate-family/order** policy. The two native cells are reused from the accepted first-illegal release with exact hashes; two constrained cells are new. No checkpoint, prompt, RoPE, image, sampling, repetition penalty, object-count policy or shared decoder changed.

Every one of the 43 complete raw boxes in **each constrained cell** satisfies x1<x2 and y1<y2 before parser drops. Exact-row repetition nevertheless persists. For image 885, the constrained release has 38 within-release repeats among 43 rows, versus 40 native. For 5586 it has 40, versus 36 native. Both constrained releases cap at 384 tokens without EOS; each has one malformed trailing partial object. Thus “valid and still repeats” is observed at both fixed histories. This locally weakens invalid coordinate order as the sustaining cause of their exact-row repetition, while leaving physical-owner recurrence, learned numerical knowledge, other histories and unrestricted completion unanswered.

| Image | Condition | Complete / strict-valid rows | Distinct exact rows | Repeat prior prefix / within release | First within-repeat | Longest contiguous run | Equality-invalid x/y | Parser drops | Stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 885 | reused native | 43 / 0 | 3 | 0 / 40 | row 3 | 30 | 43 / 12 | 44 | cap |
| 885 | joint policy | 43 / 43 | 5 | 3 / 38 | row 3 | 24 | 0 / 0 | 1 | cap |
| 5586 | reused native | 43 / 40 | 7 | 0 / 36 | row 3 | 18 | 3 / 0 | 4 | cap |
| 5586 | joint policy | 43 / 43 | 3 | 7 / 40 | row 2 | 20 | 0 / 0 | 1 | cap |

Both constrained cells' first row repeats a prior-prefix exact row. In image 885, the native first four boxes are `[0,0,0,22]`, `[0,0,0,38]`, `[0,0,0,38]`, `[0,0,0,38]`; constrained first four are `[0,0,47,22]`, `[0,0,47,36]`, `[0,0,47,36]`, `[0,0,13,38]`. In image 5586, native begins `[0,0,0,47]`, `[0,0,0,38]` twice, `[0,0,999,198]`; constrained begins with `[0,0,47,38]` four times. These are frozen-condition examples, not selected wins. Rows have unequal information content despite equal 384-token and 43-complete-row denominators. No physical-owner inference is made; annotation-unmatched is UNKNOWN. A cap with malformed trailing output is not natural recovery.

The explicit token-state scanner admits only the compact row sequence and detects malformed/ambiguous history as technical HOLD. At x1/y1 it admits coordinate bins0..998; at x2/y2 it admits only bins strictly greater than the same-row preceding corner. It masks all non-coordinate vocabulary at coordinate slots and leaves scores unchanged elsewhere, including descriptions and EOS. It does not enforce uniqueness, count, post-hoc sorting or stop. This bundle cannot isolate order from token-family enforcement.

CPU nearest-caller tests cover the x2 midbox start, equality/reversal, x1/y1 bin999 exclusion, singleton bin999 x2, new-row reset, outside-slot identity, malformed state and processor insertion. A wrong-mask, illegal-first-token and shifted-history saved mutation each fail. At model entry, an OFF-policy eight-token cached replay matches accepted native for both images. Exact prefill coordinate full-vocabulary log-probability maxima and OFF cached first full-logit maxima are each0 against2e-4. The maintained generation caller checks every emitted token against the actual masked argmax, exact legal mask at each coordinate slot and unmodified full logits outside slots within2e-4; outside-slot selected raw/policy log-probabilities are exactly equal in all428 such steps. Saved cells retain all384 tokens, raw and policy log-probabilities, raw/policy argmax, slot, same-row thresholds and legal ranges. The independent CPU saved-token verifier reconstructs the slot and mask for every step and all complete raw boxes before parser drops. It and the deterministic reducer each replay byte-identically.

Saved-only reduction SHA256 `78bfe68eb4c3a973c1ab4e4af149fb7541bf18a9ab94a40080000b64879afe7f`; saved-token verification SHA256 `1d165b9122dc415d299e30c1fa569c50a86e15820d0dd9697e69e6fc45d0ecf7`. Five focused tests, knowledge, both-root layout and diff checks pass. A first CPU caller test failed because its mock returned an object where the maintained caller returns a tuple; the mock was corrected before plan freeze/model entry, with no model cost or producer change.

New package cost: 786 forwards, 66.384702 allocated GPU-seconds and model-wall seconds, including two prefills,16 OFF-replay steps and768 constrained-generation steps. Shared overnight cumulative:3125 forwards and282.585945 allocated GPU-seconds from unchanged start `1790182797.5338483`. One producer exited0; its PID is absent. All2 planned new cells are complete,2 accepted native cells reused, no HOLD. No other owned job remains. All package/shared caps remain intact.

Reproduce saved evidence from the checkout with:

```sh
python -B -m probes.training_set_completion.coordinate_order_knowledge.sustained_legality reduce --out /absolute/fresh/reduction.json
python -B -m probes.training_set_completion.coordinate_order_knowledge.sustained_verify --out /absolute/fresh/verification.json
python -B -m pytest -q probes/training_set_completion/coordinate_order_knowledge/test_sustained_legality.py probes/training_set_completion/coordinate_order_knowledge/test_first_illegal_release.py
python -B scripts/research/check_research_knowledge.py check
python -B -m src.artifacts.output_layout --root /data/CoordExp/outputs --root /data/CoordExp/.worktrees/research-probes/outputs
git diff --check
```

The executed commands were `python -B -m probes.training_set_completion.coordinate_order_knowledge.sustained_legality prepare` and `... run`; the launch receipt binds effective source/config/current-captured code, image/prefix/media, argv, PID and clock. Root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-sustained-coordinate-legality/`. New maintained paths are this unit, `sustained_legality.py`, `sustained_verify.py` and the focused test. Accepted predecessors and peer staged/dirty work were not modified. No extra horizon, arm or successor was launched; lead owns acceptance and next research choice.
