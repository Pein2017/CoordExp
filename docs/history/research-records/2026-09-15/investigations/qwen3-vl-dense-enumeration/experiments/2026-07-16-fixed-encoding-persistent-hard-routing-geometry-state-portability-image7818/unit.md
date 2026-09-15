---
title: Fixed-Encoding Persistent Hard-Routing Geometry-State Portability on Image 7818
description: Bounded test of whether two eligible same-description hard-routing geometry states transfer through decoder layer 23 but not layer 13.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-16
---

# Fixed-Encoding Persistent Hard-Routing Geometry-State Portability on Image 7818

## Question

When persistent hard visual routing produces a valid instance-owned geometry
path, is the returned residual state at the shared pre-`x1` boundary a portable
causal geometry state?

The positive test replaces the unrestricted recipient's decoder layer-`23`
returned state with a donor state captured under hard routing. Decoder layer
`13` is the frozen negative control. The unit contains no layer sweep, training,
or architecture proposal.

## Frozen Parent Evidence

This unit is authorized only by the completed [geometry-donor eligibility
screen](../2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/results.md).
It binds both receipts:

```text
merged panel receipt SHA-256:
2046d5784030f255bc039b9252b2e63770e9f9b069219cfa9478a6e94c8f59b2

image-7818 split receipt SHA-256:
7c4d9ea4c6a82da103453de4189ad1041f46efde3cf0333a6b17442d322aa017
```

The frozen clean donors share the description `wine glass`:

| Donor role | Annotation | Persistent donor `IoU` | Persistent coordinate release |
|---|---:|---:|---:|
| target | `664730` | `0.840456` | `+0.437698` |
| paired | `661523` | `0.806652` | `+1.054995` |

Both donor supports, realized rows, feature provenance, model configuration,
source JavaScript Object Notation Lines (`JSONL`), annotation ledger, and
cohort digests must match the parent receipts before model execution.

## Intervention

Encode the unmodified full image once in full-model 32-bit floating-point
(`float32`) arithmetic with Scaled Dot Product Attention (`SDPA`). Target and
paired donors have an identical generic multi-token row prefix through
`BOX_START` and therefore the same pre-`x1` recipient boundary.

For each donor and each frozen layer:

1. in a separate no-cache forward pass, capture batch element zero at the exact
   shared boundary from the full zero-based decoder block's returned hidden
   state, after the entire block has executed, while the complete partial-row
   scoring-query range is hard-routed to that donor's frozen visual support;
2. run the same unrestricted recipient prefix;
3. discard every donor key-value cache, start a fresh unrestricted recipient
   prefill, replace only that block's returned state at batch element zero and
   the exact recipient boundary once, and remove all hooks before any later
   call;
4. score the donor's frozen realized four-coordinate suffix under unrestricted
   and replacement execution with raw no-cache logits, excluding `BOX_END`
   from the coordinate-release statistic;
5. separately continue only from the patched unrestricted recipient cache and
   greedily generate exactly four coordinate tokens plus `BOX_END`;
6. attribute the generated box against the intended donor, the paired object,
   all accepted objects, and the support envelope.

The hard-routing mask may remove non-eligible image keys only for the complete
partial-row scoring-query range, including the `BOX_START`-to-`x1` query.
Pre-row queries, non-image keys, and every later unrestricted recipient call
must retain their ordinary access.

Run an unrestricted baseline and a self-state replacement no-op at each layer.
The intervention tests portability into the same unrestricted recipient. It is
not continued hard routing or cross-image transport. The donor's frozen suffix
is used only as a common teacher-forced measurement path; successful free
continuation is independently required. Under cached continuation, the
replaced block's own key-value entries remain those of the unrestricted
recipient prefill; only downstream blocks can carry the one-time returned-state
intervention. This is the intended seam, not persistent rewriting.

## Trust Gates

Interpretation requires:

- exact parent and input digests;
- unchanged frozen supports, annotation identifiers, and realized rows;
- live reproduction, on the same four frozen coordinate slots, of the parent
  persistent-hard and unrestricted selected token identifiers, ranks, and
  per-slot raw log-probabilities within `1e-4`;
- identical target and paired prefix tokens, boundary index, position
  identifiers, and image encoding;
- exactly one donor-state capture and one recipient replacement;
- a structurally exact dynamic hard-routing mask;
- self-state replacement preserving top token identifiers and ranks with
  maximum absolute selected-token log-probability drift at most `1e-4`;
- an exact four-coordinate suffix and natural `BOX_END` closure.
- valid and trusted execution at both layer `23` and the corresponding layer
  `13` control; a missing or invalid layer-`13` arm invalidates promotion rather
  than counting as a negative result.

Any failure invalidates the corresponding arm; it is not negative mechanism
evidence.

## Portability Decision

For each donor, decoder layer `23` passes only if all conditions hold:

1. on the exact same frozen donor suffix used by the parent persistent path,
   replacement minus unrestricted raw no-cache coordinate log-probability is
   at least `+0.10` natural-log units per coordinate;
2. replacement recovers at least `50%` of that donor's persistent hard-routing
   coordinate release;
3. the effect is at least ten times the corresponding self-state no-op drift;
4. the independently generated path closes naturally, is strictly closer to
   the intended donor than the paired object, and has donor Intersection over
   Union (`IoU`) at least `0.30`;
5. donor `IoU` exceeds the strongest competing accepted-object `IoU` by at
   least `0.15`;
6. donor `IoU` exceeds the frozen support envelope's donor `IoU` by at least
   `0.05`, and generated normalized-coordinate Manhattan distance to the donor
   is at most `75%` of the support envelope's distance.

The corresponding donor at layer `13` must fail the same positive criterion.
Layer `13` is a veto, not a second search point. Its execution and trust gates
must first pass. A layer-`23` effect without a valid corresponding layer-`13`
failure is classified as invalid or non-localized portability.

For these gates, define persistent release and replacement release from the
same four frozen coordinate slots:

```text
R_persistent = mean(log p_hard(y_d) - log p_unrestricted(y_d))
R_replacement = mean(log p_replacement(y_d) - log p_unrestricted(y_d))
```

Define self-state no-op drift conservatively as the maximum absolute per-slot
difference between separate-forward self replacement and unrestricted raw
no-cache coordinate log-probabilities on that same suffix. The `10x` test uses
this maximum, not a mean that can cancel opposing drift.

Finally record the unrestricted free-continuation owner. A positive layer-`23`
arm establishes owner-specific geometry-state portability only if its intended
owner differs from the unrestricted baseline owner, or if both target and
paired donor states pass and independently generate their two distinct intended
owners. If a passing state only strengthens the already selected baseline
owner, classify it as bounded donor-path confidence portability.

## Stop Rules

- If neither donor passes layer `23`, close this exact residual-delivery line.
- If layer `23` and layer `13` behave alike, report broad state replacement,
  not a late geometry seam.
- If at least one donor passes layer `23` with the layer-`13` veto, report
  bounded geometry-state portability only when the owner-switch or two-donor
  specificity rule also passes; otherwise report donor-path confidence
  portability.
- In every outcome, stop. Do not automatically add layers, donors, images,
  trainable bridges, or architecture changes.

## Outcome

Execution is complete. The paired annotation-`661523` donor passes at decoder
block `23`, switches the unrestricted baseline owner from annotation `664730`
to `661523`, and fails at the valid decoder-block-`13` control. The positive
effect is concentrated at `x1`; see the [verified results](results.md) for the
bounded interpretation and evidence receipt.
