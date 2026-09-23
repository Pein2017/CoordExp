# Accepted position-correct preceding-row K/V sufficiency

2026-09-22. Both prespecified native-winner predictions pass. Replacing the
newest repeated row by the preceding row's contextual pre-K at the new phase
and preceding-row V retains the native winner in both fixed late contexts.
This closes the previously untested joint K/V substitution, not whole-burst
origin or transition timing. Freshly computed last-row K/V is unnecessary for
these two categorical endpoints under this particular replacement.

| Case | Native winner / gap | Corrected-copy winner / gap | D minus NN full-vector maximum | D minus key-only ON full-vector maximum |
|---|---|---|---:|---:|
| val | 999 / 0.006084442 | 999 / 0.015739441 | 0.050738811 | 0.038114309 |
| train | 348 / 0.028825760 | 348 / 0.046014786 | 0.340334415 | 0.336269379 |

Val d=z38-z999 changes−0.00608444→−0.01573944; P999 rises0.0412647→0.0415995.
Train d=z350-z348 changes−0.02882576→−0.04601479; P348 rises0.0160270→0.0164069.
The vectors differ despite winner retention. Historical key-only ON here means
old pre-K/new phase, not the mass/profile experiment's ON cell.

The fixed val controls remain: masking the added row chooses38; literal old
post-RoPE K/V duplication chooses38. Correcting the key position allows the
preceding row's contextual cache to substitute successfully. This supports a
conditional addition effect in the late val context. Train has no matched
masked-row addition control, so only replacement sufficiency is claimed there.
Old K/V already encode context; neither result establishes context-free copying,
count-only dynamics, physical recovery or an absolute-position clock.

## Independent acceptance

Root verified63 bindings and all four full-vocabulary vectors. Both fresh NN
vectors reproduce accepted NN exactly. All28 layer source/destination K/V,
actual mask/phase/slots and unchanged history/companions checks pass. Independent
FP64 complex-pair rotation errors are4.875e-5/5.551e-5 against the accepted D
keys, below2e-4. Actual source V matches the original native observer hashes.
The cache selfcheck was rerun successfully by root; source review confirms
restoration uses layer references after append and verifies the native digest.

Six model/two vision calls,28.5122s package execution,35.6050s elapsed,
16,605,249,536 bytes peak reserved. The process is terminal; no retry.

- [Frozen protocol](unit.md)
- [Full result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-positioned-duplicate/attempt-001/result.json)
- [Independent readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-positioned-duplicate/lead-checks/positioned-readback.json)
- [Lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-positioned-duplicate/lead-acceptance.json)

The final distinct question in this local loop is whether a fixed FIRST-row
contextual template can replace the distributed repeated history. A matched
last-only use of that same donor is needed to distinguish donor age from
replacement extent. No layer/head/component scan follows this result.
