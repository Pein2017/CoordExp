# Missing-Arm Successor Verification — 2026-08-13

## Scope

This verification covers only the authorized A4/A6/A8-prime successor. It
does not promote a checkpoint, claim validation/generalization, authorize a
100-update run, supervise K-miss, sync stable specs, or archive the change.

## Frozen identities

- Manifest SHA-256:
  `a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`
- Census SHA-256:
  `82d3f87883121ca7e215dc9d919660c5e426e4fc190a5e69f5406b35fa27235d`
- Plans receipt SHA-256:
  `47f80acd63313eb80610fc1f5637b1b0710fab99c60b12599d87ab65e8cedc3b`
- A4 training receipt SHA-256:
  `749a42b187ddbdb72f188feaedae562983d5ea40758a5b48a1e16628c2da04d4`
- A6 training receipt SHA-256:
  `721562f9a2d2395679a9467de0cbaa2ff69c5b7fc22a363f77bbf4ca98012f53`
- Final analysis SHA-256:
  `2f2ac938ad9fe553c2cd87c3684a65afe2f4331d01d4b0e806fb923992b64e98`

## Conclusion-changing checks

1. The integrated Human-13 loss, segment, census, HF scorer, materializer,
   payload, runner, training, launcher, and evaluation matrix suite passed:
   `161 passed`.
2. Strict OpenSpec validation passes for
   `add-human13-k-union-greedy-overfit-probe`.
3. All 73 A6 donors reproduce their sealed duplicate-clean prefix binding.
4. A4 image 14038 completed one production-shaped fixed-theta score/replay
   pass, one finite update, and checkpoint write/read. The full A4 arm then
   completed sixteen updates with checkpoints `1,2,4,8,16`.
5. A6 completed sixteen updates with the same checkpoint schedule.
6. Every A4/A6 checkpoint was evaluated on all thirteen images through HF
   fp32/SDPA, physical batch size one, greedy decoding, repetition penalty
   1.0; the analyzer used the frozen cardinality-first, maximum-total-IoU
   one-to-one owner ledger.
7. The no-update census contains 680 finite aligned sites. A8-prime is omitted
   because required margin `1.4144247604` exceeds the frozen `0.5` limit.

## Runtime evidence

| Arm | Packs/exposure | Logical tokens/exposure | 16-update wall s | Peak bytes | Five-checkpoint eval wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A4 | 45 replay + 44 score forwards | 510853 replay tokens | 2483.3 | 9890385920 | 4272.3 |
| A6 | 13 | 141865 | 505.0 | 9933340672 | 1611.5 |

A4's `510853` counter covers replay packs; the extra 44 fixed-parameter score
forwards are reported separately and are not silently counted as cached or
reused computation.

## Disposition

`PASS` for the bounded implementation and execution contract. Scientifically,
A4 and A6 are `NARROW`: both show low-dose K-hit-to-greedy movement, both lose
Source owners, and repeated static-target training produces output burden.
A8-prime is `MECHANICALLY_BLOCKED`, not a scientific null. The owning result is
`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-12-human13-k-union-to-greedy-overfit-screen/results.md`.
