---
title: Duplication Collapse Final Analysis
date: 2026-04-13
status: consolidated-final
owner: codex
---

# Duplication Collapse Final Analysis

## Scope

This document is the consolidated final record of the duplication-collapse
investigation. It supersedes the scattered study-specific markdown that had
been living in:

- `openspec/changes/add-duplication-collapse-analysis-study/`
- `research/duplication_followup/`

It also integrates the key conclusions that were derived from:

- the fixed-decode, manifest-driven duplication-collapse study,
- the pure-CE matched replay and contrastive panels,
- the crowding and class-prior follow-up analyses,
- the prefix-perturbation and interpolation experiments,
- and the earlier historical training notes that helped interpret pure CE,
  soft-CE / W1, and mixed-objective behavior.

The goal of this document is to leave one durable, self-contained analysis
artifact under `progress/` while keeping the machine-readable runtime
artifacts in their existing research roots.

## Runtime Artifact Organization

The runtime artifacts were not moved across study roots because those paths are
already encoded into manifests, reports, and cross-check scripts. Instead, the
cleanup strategy was:

- keep the stable machine-readable result roots intact,
- remove temporary scratch files from `temp/`,
- remove superseded study-specific markdown after consolidation,
- and document the stable artifact map here.

The stable artifact groups are:

- Canonical pure-CE comparison:
  - `research/duplication_collapse_pure_ce/duplication-collapse-matched-replay-pure-ce-core/`
  - `research/duplication_collapse_pure_ce/duplication-collapse-contrastive-pure-ce-panel/`
- Follow-up causal probes:
  - `research/duplication_followup/duplication-followup-crowding-class-panel/`
  - `research/duplication_followup/duplication-followup-prefix-perturb-ce-ciou/`
  - `research/duplication_followup/duplication-followup-prefix-perturb-center-shard-a/`
  - `research/duplication_followup/duplication-followup-prefix-perturb-center-shard-b/`
  - `research/duplication_followup/duplication-followup-prefix-perturb-pure-ce-shard-a/`
  - `research/duplication_followup/duplication-followup-prefix-perturb-pure-ce-shard-b/`
  - `research/duplication_followup/duplication-followup-prefix-perturb-softce4b-shard-a/`
  - `research/duplication_followup/duplication-followup-prefix-perturb-softce4b-shard-b/`
  - `research/duplication_followup/duplication-followup-prefix-perturb-stage2/`
- Control-vs-failure summaries:
  - `research/duplication_collapse_control_compare/`
- Historical and supporting study roots that remain useful for deeper replay:
  - `research/duplication_collapse_bootstrap/`
  - `research/duplication_collapse_focus/`
  - `research/duplication_collapse_intervention/`
  - `research/duplication_collapse_center_param_cases_gpu6/`
  - `research/duplication_collapse_ce_ciou_cases_gpu1/`
  - `research/duplication_collapse_center_wine/`

The code surfaces that generated and interpreted these artifacts are:

- `src/analysis/duplication_collapse_analysis.py`
- `src/analysis/duplication_followup.py`
- `scripts/analysis/run_duplication_collapse_analysis.py`
- `scripts/analysis/run_duplication_followup.py`

## Sources Re-read Before Consolidation

The following markdown documents were re-read before consolidation to ensure
that the final synthesis remained consistent with the written study record:

- `openspec/changes/add-duplication-collapse-analysis-study/proposal.md`
- `openspec/changes/add-duplication-collapse-analysis-study/design.md`
- `openspec/changes/add-duplication-collapse-analysis-study/specs/duplication-collapse-analysis-study/spec.md`
- `openspec/changes/add-duplication-collapse-analysis-study/tasks.md`
- `research/duplication_followup/executive_task_list.md`
- `research/duplication_followup/findings_2026-04-13.md`
- `docs/training/STAGE1_OBJECTIVE.md`
- `docs/data/CONTRACT.md`
- `progress/pretrain/2026-01-26_stage1_ablation.md`
- `progress/benchmarks/2026-02-26_stage1_training_dynamics_4b.md`
- `progress/benchmarks/2026-02-26_stage1_coco80_4b_res_768_vs_1024.md`

No other study-specific markdown was present in the primary research output
roots. The rest of the study state lives in JSON, JSONL, manifests, and code.

## Study Contract And Method

The investigation was run under a deliberately narrow and reproducible study
contract:

- existing checkpoints only,
- no retraining inside the study loop,
- HuggingFace as the authoritative deep-probe rollout backend,
- fixed production-like decode settings,
- checkpoint discovery restricted to locally available Stage-1 and Stage-2
  artifacts whose names include `merged`,
- and a manifest-driven workflow that preserved checkpoint provenance and
  artifact provenance throughout the study.

The decode contract was fixed to:

- `temperature = 0.0`
- `top_p = 0.9`
- `repetition_penalty = 1.05`
- `max_new_tokens = 3084`
- `seed = 42`

Secondary decode diagnostics were allowed only after reproducing the behavior
under that fixed baseline, and they were never treated as the main explanatory
surface.

The final diagnostic workflow had five layers:

1. Historical artifact audit and high-recall failure mining.
2. Fixed-setting rollout reproduction on the selected case.
3. Deterministic re-forward surgery at the duplication onset.
4. Same-prefix controlled comparisons, with `predicted_object` vs
   `exact_duplicate` preferred when the rollout yielded a usable continuation.
5. Follow-up counterfactuals and local-prefix perturbations that edited the
   pre-collapse prefix without retraining.

The study also made two design choices that mattered a lot:

- it treated `coord_x1` / `coord_y1` escape from the previous or local
  neighborhood as the primary mechanism surface,
- and it treated late history-overwrite as a secondary amplifier unless a
  stronger controlled intervention could show primary causal force.

## Checkpoint Families Included

The family comparison was constrained by what actually existed on disk and by
what was compatible with the expanded coordinate-token pipeline.

The main families that anchored the conclusions were:

- `stage1_pure_ce_ckpt1932`
  - a clean and token-compatible pure-CE Stage-1 reference
- `stage1_softce_w1_4b`
  - a soft-coordinate-supervised Stage-1 family
- `stage1_2b_hard_soft_ce`
  - a mixed Stage-1 family with hard plus soft coordinate supervision
- `stage1_ce_ciou_ckpt1564`
  - a CE-like soft-disabled continuation proxy, not a clean from-scratch pure
    CE baseline
- `stage1_2b_center_param_ckpt1564`
  - a center-parameterization proxy branch that remained especially valuable as
    a failure anchor
- `stage1_adjrep_global_2b`
  - an adjacent-repulsion branch that helped expose what heuristic anti-copy
    shaping does and does not fix
- `stage2_pseudo_positive_ckpt300`
  - a Stage-2 rollout family used to test whether the later training regime
    creates a new failure or stabilizes an existing one

One important provenance note:

- the CE-like disabled branches were useful comparison families, but some of
  them were soft-disabled continuations from soft-trained parents
- they therefore could not be described honestly as clean pure-CE baselines

That provenance caveat is important because it affects what we can and cannot
claim about pure CE from observational comparisons alone.

## Core Causal Surfaces

The study converged on four high-value causal surfaces.

### 1. Early coord escape

The first major discriminator was whether the earliest coord decisions,
especially `coord_x1` and `coord_y1`, rapidly evacuated probability mass away
from the previous or local same-desc neighborhood.

This was more diagnostic than:

- whole-object aggregate scores,
- duplicate counts alone,
- or attention summaries alone.

### 2. `predicted_object` vs `exact_duplicate`

The most informative comparison was not teacher-aligned `gt_next` vs
`exact_duplicate`. It was:

- the actual generated next object under the same prefix,
- compared against the exact-duplicate candidate under that same prefix.

This comparison directly asks whether rollout escaped the basin it was at risk
of falling into.

### 3. Prefix geometry perturbation

The strongest controlled evidence came from local-prefix perturbation:

- drop the previous same-desc source object,
- replace it with the alternative same-desc object,
- interpolate its geometry toward the alternative,
- or redirect only `x1/y1`.

These experiments were the clearest tests of whether local geometry, rather
than merely repeated text, was controlling the collapse.

### 4. Late history-overwrite

The probes repeatedly showed late-layer shifts away from visual evidence and
toward recent generated history or prior coord spans. That effect was real and
repeatable, but it was not enough by itself to explain the duplicated branch.

Healthy same-desc controls could still show some overwrite while escaping
collapse. So overwrite remained a supporting mechanism rather than the primary
root cause.

## Converged Mechanism-Level Findings

### 1. The best current root-cause description is a local coordinate basin

The strongest explanation is not:

- "the model ignores the image,"
- nor simply "the model copies the last coord tokens,"
- nor generic language-model degeneration without geometry.

The strongest explanation is:

- Stage-1 coordinate supervision can leave a smooth local basin around the
  previous or nearby same-desc instance.
- During rollout, the next object begins from a history-biased state.
- If the first coord decisions do not escape that basin quickly, the model
  continues from a locally self-consistent but wrong geometry-conditioned
  prefix.
- Autoregressive continuation then makes that state sticky.

This is why the final preferred language is:

- `coordinate basin`
- `weak local escape barrier`

especially at:

- `coord_x1`
- `coord_y1`

### 2. Early-slot geometry is more important than late attention summaries

The current controls support a stronger and deeper story than the earlier
attention-only view.

Healthy same-desc controls can still show:

- some late history-overwrite,
- some prior-coord attention,
- and some vision-to-history handoff.

But they still escape because their early coord decisions become sharper and
move away from the previous or local neighborhood.

So the decisive question is not:

- did the model think about the previous object?

It is:

- did the earliest coord decisions successfully leave the previous/local basin?

### 3. Prefix geometry matters more than source deletion

The clearest causal pattern from the perturbation studies was:

- baseline often stays trapped,
- dropping the previous source object often still stays trapped or falls into
  another bad mode,
- replacing local geometry with the same-desc alternative often escapes,
- interpolating local geometry often escapes,
- and substituting only `x1/y1` can already be enough to escape.

That means the failure is not explained by the mere presence of repeated
same-desc text in the prefix. The more specific explanation is:

- the immediate local geometric anchor in the prefix keeps the rollout inside
  the basin.

### 4. Crowding is a strong trigger but not a sufficient explanation

The crowding audit supported a real upstream effect:

- duplicated cases often come from dense same-class neighborhoods,
- overlap and local center proximity are much stronger in duplicated cases than
  in healthy controls.

But the study also found:

- lower-cardinality same-desc failures,
- and healthy same-desc controls in repeated-object categories.

So the right interpretation is:

- crowding-driven instance aliasing is a strong trigger,
- but not a sufficient explanation on its own,
- and annotation noise is not currently the leading hypothesis.

### 5. Late history-overwrite is a secondary amplifier by default

The study did not find strong enough controlled evidence to promote late
history-overwrite to a primary-cause claim.

What is supported:

- the final layers often lean away from vision and toward recent text/history,
- this likely stabilizes a bad local state once it already exists.

What is not yet supported strongly enough:

- that late overwrite alone creates the basin in the first place.

## How Behavior Varies Across Checkpoint Families

The checkpoint families do not differ only by "more duplication" vs "less
duplication." They differ in how they enter, stabilize, or escape the same
local geometry-conditioned basin.

### Center-parameterization proxy

This family was one of the clearest same-desc duplicate-basin families.

Intuitively:

- it behaves like a model that has latched onto the local object cluster and
  keeps walking around inside that same cluster instead of stepping to a fresh
  instance.

Its anchored perturbation behavior made the mechanism especially clear:

- baseline stays trapped,
- source deletion stays trapped,
- geometric replacement, interpolation, and `x1/y1` redirection can escape.

This does not mean center-parameterization is "better." It means it was one of
the clearest diagnostic failure families.

### Hard+soft CE 2B family

This family often looked softer and less rigid:

- it sometimes stayed near the right local cluster,
- but drifted semantically instead of producing a perfectly repeated same-desc
  loop.

Intuition:

- it often knows the right local region but does not form a crisp enough
  instance boundary to continue cleanly.

### SoftCE+W1 4B family

This family also behaved like a broad local-posterior model:

- sometimes same-desc duplication,
- sometimes semantic wobble,
- sometimes healthier behavior than a fully trapped branch,
- but generally with weaker local separation than the pure-CE baseline.

### CE-CIoU continuation proxy

This family looked sharper but more brittle:

- once it tipped the wrong way, it could collapse hard,
- but some perturbations produced undercount or semantic escape instead of a
  healthy continuation.

That makes it different from the softer families:

- not necessarily more robust,
- simply more decisive once a branch has tipped.

### Pure CE baseline

This was the most useful mechanistic baseline.

What it did well:

- raised the local escape barrier,
- made early-slot redirection more effective in a meaningful number of cases,
- improved local instance separation relative to the softer families.

What it did not do:

- it did not make the system immune,
- it did not eliminate undercount,
- and it did not eliminate semantic escape.

So pure CE should be understood as:

- the cleanest current baseline for local instance separation,

not as:

- a completed solution to rollout quality.

### Adjacent-repulsion branch

This family was useful because it showed what heuristic anti-copy shaping looks
like when it does not address the deeper basin:

- it often changed the visible shape of the failure,
- but did not convincingly remove the underlying local geometry-conditioned
  trap.

### Stage-2 rollout family

Stage-2 looked more like a basin stabilizer than a brand-new failure generator.

The same kinds of local-prefix geometric perturbations that freed some Stage-1
anchors could also free Stage-2 anchors. That supports the interpretation that:

- Stage-2 often amplifies or stabilizes an already available basin rather than
  creating a fundamentally unrelated pathology.

## Hypotheses Revisited

The later stages of the investigation revisited three explicit hypotheses and
added a fourth, deeper one.

### Hypothesis 1: annotation noise or GT ambiguity is the main cause

Current status:

- weakened

Why:

- the evidence currently points more strongly to crowding-driven instance
  aliasing than to annotation noise,
- low-cardinality failures still exist,
- and no strong direct label-error evidence dominates the studied cohort.

### Hypothesis 2: autoregressive MLE continuation is the main cause

Current status:

- partly supported, but only in a narrowed form

Why:

- autoregressive continuation clearly amplifies the failure,
- but the prefix-geometry experiments show that the collapse is not just
  generic runaway continuation,
- it is continuation from a locally self-consistent geometry-conditioned state.

Best refined version:

- teacher-forced training never teaches recovery from self-emitted local
  aliasing,
- and rollout turns that weak local barrier into a self-reinforcing loop.

### Hypothesis 3: visual ambiguity plus language priors is the main cause

Current status:

- partly supported

Why:

- duplicate-prone categories are selective rather than uniform,
- repeated-object scene families matter,
- but class prior does not explain the failure in the absence of a weak local
  coord escape barrier.

Best refined version:

- class priors and repeated-scene structure are basin activators, not the sole
  cause.

### Hypothesis 4: Stage-1 soft/distributional coord supervision weakens local instance separation

Current status:

- strongly supported and currently the leading mechanism hypothesis

Why:

- the pure-CE baseline is the best current counterexample,
- soft-coordinate-supervised families are more tolerant of locally similar
  alternatives,
- and the earliest visible failure still concentrates at the first coord
  decisions rather than only later shape refinement.

## Training-Objective Interpretation

The current Stage-1 objective structure matters for the mechanism story.

The key ingredients are:

- autoregressive generation over coordinate tokens,
- teacher-forced optimization,
- soft / distributional coordinate supervision on the same logits used for
  generation,
- and expectation-style geometry signals that also reuse those coord logits.

The study’s current interpretation is:

- if the same logits receive both discrete generation pressure and local smooth
  geometry pressure, nearby same-desc solutions can remain too mutually
  compatible,
- and this lowers the early escape barrier at rollout time.

This does not imply that all soft geometry is bad. It implies something more
specific:

- shaping the main generative coord logits with soft local distributions appears
  to be dangerous for the early instance-disambiguation step.

## Implications For Retraining

If retraining from the base 2B or 4B checkpoints is acceptable, the strongest
current recommendation is:

- train the main generative coord logits with pure CE,
- do not reintroduce soft-CE / W1 onto those same logits early,
- recover geometric smoothness through a separate auxiliary path or detached
  branch,
- and add a targeted same-prefix anti-copy / instance-separation signal focused
  on early coord decisions.

The most important reason for that recommendation is that the study now points
to:

- weak early local escape,

not to:

- generic late attention pathology,
- and not to a simple decoding-temperature problem.

## Implications For Parameterization

One promising design direction is to change the model-facing geometry
parameterization from:

- `(x1, y1, x2, y2)`

to something center-first such as:

- `(cx, cy, w, h)`

The argument in favor is aligned with the current findings:

- the first discriminative spatial decision is the critical one,
- `x1/y1` is a boundary-anchored representation that seems especially prone to
  local same-desc collapse,
- `cx/cy` is closer to instance identity than top-left boundary alone,
- and `w/h` can potentially absorb some annotation tolerance without blurring
  the early center decision.

The best current recommendation is not to rewrite the raw data contract
permanently. Instead:

- keep raw data and eval canonical in `xyxy`,
- introduce a reversible model-facing transform,
- and test center-first parameterization at the model/training interface.

The most cautious staged plan is:

1. First test center-based tokenization cleanly.
2. Then, if the center-first effect looks real, consider making `w/h`
   regression-based while keeping `cx/cy` as the decisive discrete tokens.

The current center-parameterization proxy checkpoint does not falsify this
proposal because it does not fully test a true center-token autoregressive
contract in the stronger sense being proposed here.

## What The Study Supports With High Confidence

The following claims now have strong support:

- Duplication collapse is best understood as a local coordinate basin plus weak
  early escape barrier.
- `coord_x1` / `coord_y1` are the primary onset-local surfaces.
- `predicted_object` vs `exact_duplicate` is the right main causal probe.
- Prefix geometry matters more than source deletion.
- Late history-overwrite is secondary by default.
- Crowding is a strong trigger, but not sufficient on its own.
- Pure CE raises the local escape barrier meaningfully, but is not immune.
- Stage-2 tends to stabilize an available basin rather than inventing a wholly
  unrelated one.

## What The Study Still Does Not Support Strongly Enough

The following claims should remain conservative:

- that annotation noise is the main explanation,
- that late history-overwrite is the primary cause,
- that pure CE completely solves duplication collapse,
- that any CE-like disabled continuation is equivalent to a clean pure-CE
  baseline,
- or that crowding alone explains the whole phenomenon.

## Open Questions

Several questions remain open and important:

- How much of the remaining pure-CE failure is still a coord-basin problem
  versus stop/continue instability or semantic continuation instability?
- After tighter crowding matching, do class-prior effects survive strongly?
- Would a true center-token autoregressive parameterization reduce the basin
  more than pure CE in `xyxy` space?
- Can a targeted same-prefix anti-copy signal preserve healthy same-desc
  continuation while suppressing the duplicate basin?
- Can any intervention on late overwrite flip rollout behavior reliably enough
  to upgrade it from secondary amplifier to primary causal factor?

## Recommended Next Step

If the next work item is another model iteration rather than more diagnosis,
the strongest current bet is:

- fully retrain from the base model with pure CE on the main generative coord
  logits,
- keep any geometry smoothing off that main token path,
- and, if feasible, test a center-first representation with the external raw
  data contract preserved via reversible conversion.

If the next work item is still analysis, the highest-value additions would be:

- tighter matched crowding bins with larger per-class volume,
- more low-cardinality duplicate families under the same perturbation protocol,
- and a clean apples-to-apples comparison once any new center-based or retrained
  checkpoint becomes available.

## Cleanup Note

This final document supersedes the scattered markdown that previously lived in
the study-specific OpenSpec change and follow-up workspace. Those notes can be
removed once this document is committed because their content has been merged
here.
