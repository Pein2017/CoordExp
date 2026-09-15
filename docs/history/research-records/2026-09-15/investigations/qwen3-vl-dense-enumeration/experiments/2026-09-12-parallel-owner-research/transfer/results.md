# Fixed fresh transfer and exposed history-cross results

Status: **candidate complete, cold-verified; root final acceptance pending**.
All scientific stops reached; no further model work requested or running.

## Decision-bearing answer

On this provenance-qualified fresh256 panel, **C32 does not establish a robust
owner-coverage gain or a benefit over matched-progress D17**. Both reduce
annotation-relative false positives, malformed geometry and repeat burden
relative to Stable50, predominantly by avoiding two catastrophic capped loops.
This is useful burden transfer, not evidence of broad new-owner recovery or
unique margin necessity.

Natural greedy, RP1, cap3084;256 exact paired images,1836 annotated owners.
No forced rows enter this primary denominator.

| Checkpoint | TP50 / FP50 / FN50 | F1@50 | TP60 | TP80 | Repeats | Parser drops | Caps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stable50 |1240 /804 /596|.639175|1160|851|226|820|3|
| C32 |1242 /617 /594|.672260|1158|851|61|301|1|
| D17 |1245 /588 /591|.678659|1156|853|57|299|1|

C−Stable owner gained/lost/retained counts:
IoU50 **44/42/1198**; IoU60 **38/40/1120**; IoU80 **35/35/816**.
Thus TP deltas are+2/−2/0, with18 improved,17 worsened and221 TP50-tied images.
C−D owner counts are23/26/1219,29/27/1129,30/32/821: TP−3/+2/−2.
D−Stable TP deltas are+5/−4/+2, not an established larger owner recovery.

Image-paired bootstrap95% intervals (10000 draws, seed20260912):

| Contrast | ΔTP50 (point; interval) | ΔF1@50 (point; interval) |
|---|---:|---:|
| C−Stable |+2;[−22,+27]|+.033085;[−.004533,+.081052]|
| D−Stable |+5;[−21,+34]|+.039484;[+.002017,+.086447]|
| C−D |−3;[−20,+14]|−.006399;[−.016430,+.003760]|

These resample images, not individual owners. They describe uncertainty for
the frozen eligible-source sample, not full-scene completeness or
pretraining/SFT-disjoint generalization. D's positive F1@50 interval does not
establish owner recovery, a unique mechanism, or superiority at all thresholds.

## Burden and concentration, without changing the denominator

Raw row starts: Stable2864, C2160, D2132; parsed rows2044/1859/1833.
Geometry-invalid drops817/300/298; other malformed drops3/1/1. All invalid
and malformed rows remain in this burden ledger although the unchanged native
matcher does not treat parser-dropped geometry as valid FP boxes. Direct raw
row-start counts exactly equal parsed-plus-dropped rows for all arms.
Generated tokens26788/20419/20157; EOS253/255/255.

C's FP50 reduction187 is dominated by images127270 and58111: their combined
FP decrease155 is83% of the net reduction. They contribute143 of165 fewer
repeats and525 fewer drops (other images add6 net drops), and both switch from
capped Stable loops to uncapped C/D outputs. Those two images contribute+7TP50
against the full-panel+2; the remaining254 contribute−5. This is a descriptive
concentration decomposition, not a new evaluation cohort or selection rule.
Image31296 contributes a further−28FP and−21repeats for C.
Image35326 still caps under all three checkpoints.

FP decreases/increases/ties occur on30/23/203 images for C and32/27/197 for D.
The full paired image ledger, all thresholds and bootstrap intervals are in
`full/result.json`; burden concentration is in `interpretation.json`.

## Separate fixed12-cell history cross

All six diagonal self-history cells exactly reproduce the archived remaining
tokens under inherited3084-token budgets. All literal prefixes end at complete
rows; none invokes the predeclared absorbing-EOS exception. The entire prefix
is supplied intervention and never receives free credit.

Each cell below is **free-suffix TP50 / FP50**, not full forced-output credit:

| Image | Recipient | C-history | D-history |
|---|---|---:|---:|
|25274|C32|12 /15|12 /13|
|25274|D17|0 /143 (cap)|0 /144 (cap)|
|511251|C32|7 /23|0 /137 (cap)|
|511251|D17|7 /24|0 /123 (cap)|
|417044|C32|9 /11|9 /11|
|417044|D17|10 /10|10 /10|

**Observation:**511251 follows the donor history: C's early complete row
supports ordinary continuation even in D, while D's row sends even C into a
capped loop.25274 instead follows the recipient checkpoint under both tested
histories; C-prefix alone does not rescue D.417044 preserves the recipient's
localization/owner difference under either prefix.

**Inference:** early-history path capture is sufficient to explain the tested
511251 contrast, but cannot universally explain all three cases. Different
images retain different conditioning sensitivity. This is not a new attention/
instance-state mechanism proof, and all three cases are previously exposed.
Full-output, forced-prefix, free-suffix and newly free owner ledgers at
IoU50/60/80 are preserved separately in `cross/consumer.json` and `cross/result.json`.

## Technical evidence, resources and stop

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/transfer/`.

- `packet.json`: SHA256`3ae84969cce7b60629ea185699e1ec63ebd1934ad1464c4c993e4225bce83739`.
- `full/result.json`: SHA256`8c363ee2f0a7e32f21a2365462b85461a22bc1e0d37b0872e4a40d9691c5857c`.
- `cross/result.json`: SHA256`ede97fef938a3b7bc76acc141dc6d14ff0c49219207ffb586355b813532d1ea2`.
- Both complete cold consumers pass exact population, label, adapter, prompt,
  media, action/text, native parser and scorer checks; outer exits are0.
- Full:768 image forwards,67364 model forwards/tokens,6 model loads,
  5400.38 allocated GPU-seconds. Cross:12 cells,1078.50 allocated GPU-seconds.
  Qualification:6 old calls,624 forwards/tokens,72.88 allocated GPU-seconds.
  The retained zero-forward technical-invalid first qualification cost8.02GPU-seconds.
- Full peak CUDA allocated9.27GiB, reserved9.79GiB, peak RSS11.15GiB. No OOM,
  source/model substitution, hidden retry, adaptive image expansion or promotion.
- Five focused CPU tests pass, including the real native-loader path failure
  and correction. Root accepted the qualification before full execution.

The one fresh256 read and separate12-cell cross are complete. A warranted
future question may be proposed separately; these results authorize no new
checkpoint selection, training, held-out expansion or architecture change.
