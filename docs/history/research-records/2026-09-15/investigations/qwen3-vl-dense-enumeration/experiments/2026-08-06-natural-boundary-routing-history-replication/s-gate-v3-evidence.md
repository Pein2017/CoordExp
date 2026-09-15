---
title: S gate-v3 natural-boundary routing and history evidence
type: investigation-evidence
role: evidence-note
authority: non_normative_research
unit_id: 2026-08-06-natural-boundary-routing-history-replication
event_id: S/gt:5001:15
status: complete_case_level
updated: 2026-08-06
---

# S gate-v3 evidence

## Disposition

The fresh S `gt:5001:15` gate-v3 completed mechanically with all 15 frozen
arms and terminal status `completed`. This is one case-level, no-training
observation. It does not open the support plan or establish a checkpoint,
cohort, controller, architecture, decoding, or training conclusion. S is the
sole decision-owning substrate in this unit; A3 is absent and remains held.

The natural boundary is exact: the prefix ends with token `151649`
(`<|box_end|>`), not the opener `151646`; every arm records
`admission_mode=pre_opener_natural`, `opener_injected=false`, and zero
synthetic opener injections. A model-generated opener was observed in 11/15
arms. N10/N20/H10/H20 are scientifically meaningful invalid-first-token or
no-row outcomes, not instrumentation failures.

## Mechanical PASS receipt

Primary immutable root:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/s-gt5001-live-gate-v3`

| Artifact | File SHA-256 | Embedded/self SHA-256 |
| --- | --- | --- |
| `result.json` | `81b6e9557fd7f0eed54b52ea2c34bee1d62056f4f2459d7f24bfee1c4fcef309` | `9719f2efe77f399ca66985659d902e33087be79bfdd29762643c6045cbed00fc` (`result_sha256`) |
| `runtime_identity.json` | `12e685b78d36bce35b7582d7246521f37a2276ce65248683383f6cdba43b85e8` | `c612564431c6bf0cee76335272730602f218b8b389042b6473e7a7b6905ad78f` (`identity_sha256`) |
| `terminal_summary.json` | `011b2fa4f806e0b0acb429a6bb39b21943c31a5cdea87abb7c0b2b126704cd3e` | `9719f2efe77f399ca66985659d902e33087be79bfdd29762643c6045cbed00fc` (`result_sha256`) |
| `s-gt5001-live-gate-v3.launch.log` | `cefd8eac7794c94422c43dc58edda232b51f1291610d03afbbfc8c32abdfd578` | terminal line reports the same `result_sha256` and `status=completed` |

The gate runtime identity binds
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/pre-gpu-receipt-v4/pre-gpu-receipt.json`
(file `ad09d12b0fb627bd3705d8b5906cb7168e0d66df67fb56ba64213d8dd0da6cee`,
self `4a875dd2af9805a035c60a174ac33b845c103a06814a7fc9404b4fc93100ee42`) and
forces `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`. The receipt records
`operator_semantics_unchanged=true`; this is a deterministic backend repair,
not an estimand or operator change. The v4 focused-test receipt at
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/pre-gpu-evidence-v4/focused-tests.json`
has SHA-256
`6c029381f1a5cd211efe876567fae8e8ce0f4960fb65afc0d4f7a410c8c8f0f7`; the
installed-Qwen probe at
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/pre-gpu-evidence-v4/installed-qwen-mask-probe.json`
has SHA-256
`0aa1c3f0587e7cafa8738dc3e62734a0e19f704f5a85056c19b5f72ed3a4974a`.

Conclusion-critical mechanical checks in the gate result are all satisfied:

- 15/15 arm IDs are present in the frozen order `K00,K01,K10,K11,K12,K13,K14T,K14B,N00,N01,N10,N20,H00,H10,H20`.
- K01 versus K00 and N01 versus N00 each have 27 measured/reference steps and maximum absolute drift `0.0` against tolerance `1e-4`; H00 also has 27-step parity at `0.0` against N00. K00/N00 are reference captures, so their `passed=true` is not a second comparison.
- Every attention-actuated arm with runtime consumption evidence (K01, K10--K14T/B, H00, H10, H20) attests one identical float32 mask through all 28 layers, with no missing or repeated layers and one call per layer.
- K14T and K14B each have 27/27 scalar receipts with dose `+2.0`; every block-23 mass receipt is `status=passed`, `passed=true`, and observes a positive selected-mass shift. Production GQA is `16` query heads / `8` KV heads / `2` query groups.
- Residual requests are non-persistent and explicit: native `N00` for K/H arms; `N01` terminal no-op replay at position `1328`; `N10` terminal replacement at `1328`; and `N20` whole-row replacement at positions `1320..1328`. No stale residual hook is recorded.

## Case-level arm behavior

Counts below are the three-row endpoint bookkeeping (`matched` means strict
physical-owner match; `unmatched` is a valid scientific outcome). Owner IDs
are raw endpoint IDs, not a claim of complete support.

| Arm | Admission / first token | Endpoint behavior | Focused interpretation |
| --- | --- | --- | --- |
| K00 | opener `151646`, model-generated | 2 matched (`gt:5001:19`, `gt:5001:9`), 1 unmatched; closure | Native baseline; no actuator. |
| K01 | opener `151646`, model-generated | Same 2 matched + 1 unmatched; closure; parity to K00 | Static mask no-op/actuation control; no endpoint change in this case. |
| K10 | opener `151646`, model-generated | 3 matched (`gt:5001:12`, `gt:5001:15`, `gt:5001:19`), 0 unmatched; closure | **Oracle-routing sufficiency at this case**, not a natural owner slot. Admission is unchanged from K01; the owner change is post-opener realization. |
| K11 | opener `151646`, model-generated | 2 matched (`gt:5001:12`, `gt:5001:19`), 1 unmatched; closure | Partial static routing effect; not a natural-admission conclusion. |
| K12 | opener `151646`, model-generated | 0 matched, 3 unmatched; closure | Valid unmatched outcome; static intervention did not yield a source-specific owner here. |
| K13 | opener `151646`, model-generated | 0 matched, 3 unmatched; closure | Valid unmatched outcome; do not relabel as technical failure. |
| K14T / K14B | opener `151646`, model-generated | Each 2 matched (`gt:5001:19`, `gt:5001:9`), 1 unmatched; closure | Soft finite-dose controls are mechanically actuated and mass-attested, but show no case-level endpoint change from K01. |
| N00 / N01 | opener `151646`, model-generated | Each 2 matched + 1 unmatched; closure; N01 parity to N00 | Native history baseline and terminal no-op replay; no endpoint change here. |
| N10 / N20 | first token `151670` (coordinate), opener not generated | One invalid row, 0 matched, no endpoint owners; invalid stop | Natural admission/grammar outcome under terminal replacement, not STOP evidence and not instrumentation failure. |
| H00 | opener `151646`, model-generated | 2 matched + 1 unmatched; closure; parity to N00 | Unconditional-history baseline; all-layer mask consumption attested. |
| H10 | first token `151649` (`<|box_end|>`), opener not generated | One invalid row, 0 matched; invalid stop | Latest-row history intervention changes admission at this case; it is not a post-opener identity result. |
| H20 | first token `12963`, opener not generated | One invalid row, 0 matched; invalid stop | Whole-latest-row history intervention changes admission at this case; scientific invalid outcome. |

The K10/K11/K12/K13 comparisons are therefore primarily post-opener owner
realization conditional on a naturally generated opener. N10/N20/H10/H20
probe the pre-opener admission boundary and must not be collapsed into the
same estimand. The data support neither a global static-routing claim nor a
global dynamic-history null from one event.

## Alternatives and required support/cohort decision

The strongest live alternatives are:

1. Static image routing is sufficient for some post-opener owner choices
   (the K10 case), while natural admission is a separate gate.
2. Latest-row history can alter the pre-opener token competition and produce
   invalid/no-row outcomes (N10/N20/H10/H20), but this could be case-specific
   grammar sensitivity rather than a general STOP or coverage controller.
3. K12/K13 unmatched rows reflect semantic safety cost under the declared
   intervention, not failed instrumentation; a source-specific effect cannot
   be inferred from them alone.

The already sealed S support completion must decide whether these patterns
recur: exactly 200 contexts, 77,428 planned scalar forwards, and eight fixed
shards, without rebuilding from gate outcomes. Cohort qualification requires
at least three static-eligible S events over at least two S images; dynamic-only
or unmatched/invalid events remain separately reported and do not satisfy the
static threshold. Only after that S decision can a minimal A3 contrast or any
2x2/crossover question be considered. No checkpoint or training route is
qualified by this case.

## Claim boundary

This note records a mechanically valid, case-level S gate-v3 result. It does
not promote K10 to a natural slot, does not turn N/H invalidity into a STOP
controller claim, and does not erase unmatched outcomes. This evidence note is
not itself an execution authority: the already-running support completion is
authorized by the user's explicit successor decision. A3, training, production
decoding, architecture changes, and checkpoint promotion remain unauthorized.
