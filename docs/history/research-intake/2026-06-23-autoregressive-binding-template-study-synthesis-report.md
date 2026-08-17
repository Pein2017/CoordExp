# Autoregressive Binding Template Study: Standalone Research Synthesis

## Executive Summary

This report synthesizes a recent multi-day mechanistic study of autoregressive object binding in a vision-language detection model that emits object spans as text-like sequences. The primary evidence window is June 21 through June 23, when the work moved from selected-case hidden patches into formation-position row building, train/validation population scaffolds, staged forcing, destination-family reduction, ownership-transition tomography, and failure-locus separation. Earlier work is included as background only where it explains why the late-window experiments were designed the way they were. Each object span contains an object descriptor, a bounding box expressed as coordinate tokens, and structural boundary tokens that close the object and move to the next object or terminate the sequence. Two template families were studied: a description-first family and a geometry-first family.

The central question was why the model duplicates objects, misses valid objects, drifts into wrong coordinate basins, or prematurely closes or restarts object spans. The study started with a small paired rollout cohort and expanded into teacher-forced train and validation panels, hidden-state readouts, attention and value-route probes, causal activation patches, continuation scoring, and staged forcing experiments.

The strongest current conclusion is that the failures are not well explained by a single cause such as visual non-perception, literal copying of previous coordinates, train-versus-validation memorization, a missing descriptor logit, or a globally broken output grammar. The better current picture is a multi-regime autoregressive state-machine failure:

- In many ordinary pre-x1 states, the model is already in coordinate-token mode, but it chooses the wrong coordinate basin or lacks a stable ownership/cursor state for the target object.
- After the first correct coordinate is supplied, many rows become locally coordinate-ready, so the main fault is often coordinate-basin onset rather than absence of usable object evidence.
- Later-slot hidden deltas can move the distribution, but they often import donor slot phase or donor coordinate ownership, so rank movement is not enough evidence of true receiver repair.
- Some rows that initially looked like pre-x1 coordinate failures are actually completed-box or wrapper-mode boundary states; these need boundary/router probes, not coordinate-basin probes.
- Descriptor repair, selected-instance geometry binding, current-box repair, and next-row routing are separable mechanisms.
- Trained-sequence failures are common under teacher-forced ground-truth prefixes, so the dominant mechanism cannot be dismissed as held-out visual generalization failure.
- Training remains premature. A training objective should not be launched until there is a replicated, span-aware, locus-specific causal handle that repairs full object behavior rather than only a token, rank, descriptor, or local margin.

The most mature positive causal handle is the crowded closure/router family: a large layer-24 boundary direction can flip crowded completed-box states from premature object-boundary tokens to the box-end decision, and every flipped case in the all-crowded candidate slice follows the expected next-object route. This handle is strong for that boundary family, but it should not be generalized to coordinate-onset failures.

The main frontier is now split into three tracks:

1. Clean pre-x1 coordinate-mode onset failures: localize ownership formation around layers 17 and 18, especially layer-17 self-attention and layer-18 resolver states.
2. Completed-box and wrapper-mode boundary/router failures: probe box-end versus object-reference boundary and termination competition.
3. Small-object, final-extent, and tail/delayed-evidence failures: distinguish coordinate locality, exact-token smoothness, visual ambiguity, and delayed resolver behavior.

## Background and Motivation

The model performs detection by generating a sequence of object records. Each record is an autoregressive text span that includes an object descriptor and a bounding box. This design creates a useful but fragile interface: object identity, spatial grounding, coordinate quantization, object closure, and next-object routing must all remain synchronized across many generated tokens.

The observed failure modes include:

- Duplicate or near-duplicate objects.
- Missed objects even when the object is visibly present.
- Correct-looking descriptors paired with wrong or repeated boxes.
- Wrong coordinate basins despite correct coordinate-token type.
- Premature boundary or restart tokens while a box should still be open.
- Valid local object spans that fail to transition cleanly to the next object.
- Cases where a generated row is syntactically valid but bound to the wrong same-class instance.

The motivation of the study was not simply to measure detection performance. The goal was to identify the latent autoregressive mechanism that decides which object is currently being described, which spatial basin the next coordinate should enter, when a box should close, and when the sequence should move to the next object or stop.

The study repeatedly tightened its evidence standard. Early observational patterns were not accepted as mechanism claims unless they survived repetition-penalty correction, split discipline, non-circular labels, held-out or reserve checks, behavior-level continuation, and causal perturbation where appropriate. This is why several attractive early explanations were demoted.

## Research Questions and Hypotheses

The work pursued the following core questions.

1. Does template order change when semantic identity or geometry becomes committed?

The description-first and geometry-first families expose different local events. A description-first prefix can settle on a descriptor before coordinates are emitted; a geometry-first prefix can expose spatial settlement earlier. The initial hypothesis was that template order might change whether the model binds semantically first, spatially first, or falls into duplicate/repetition basins at different local decision points.

2. Are duplicates caused by high-confidence previous-anchor reuse?

Early rollouts suggested that duplicate rows might repeat a previous object's coordinate basin. The study tested whether duplicate onsets had higher raw emitted-coordinate probability, lower entropy, or closer distance to previous anchors. This hypothesis was later weakened by repetition-penalty correction and non-circular null comparisons.

3. Are missed objects visually absent to the model, or does the autoregressive state fail to route to them?

A major aim was to separate visual non-perception from language-side or cursor-side routing failure. Teacher-forced prefixes, staged coordinate forcing, train-row failures, hidden readouts, and continuation probes all address this distinction.

4. What state chooses the coordinate basin before x1?

The pre-x1 position became a central locus. The model often knows it should emit a coordinate token, but the exact coordinate basin can be weak, wrong, donor-like, origin-like, or wrapper-mode. The central question became whether a stable object ownership state exists before x1 and where it forms.

5. Are descriptor repair and instance binding the same thing?

Post-box counterfactuals showed that descriptor logits can be repaired without selecting the intended instance geometry. This motivated separate readouts for descriptor, target-box overlap, same-class wrong-basin overlap, current-box repair, and next-row routing.

6. Is there a train-versus-validation mechanism split?

The study deliberately expanded from validation-only examples to trained-sequence rows. Trained rows still fail under teacher-forced ground-truth prefixes. Therefore train exposure does not guarantee local coordinate-basin ownership or boundary routing.

7. Can a causal handle justify training?

The study treated training as gated. A candidate intervention must replicate across a population, preserve span behavior, beat controls, repair full object behavior, and avoid mixing distinct failure loci. No current coordinate-onset handle satisfies this training gate yet.

## Recent Evidence Base and Experimental Process

This synthesis is anchored primarily in the late evidence window from June 21 through June 23. Earlier notes are used as background when they explain why the late-window experiments were needed, but the main interpretation follows the newer late-window revisions.

The central June 21 evidence family was the transition from post-box descriptor and coordinate readouts into causal patch semantics. The important products of that day were not merely results; they defined how later experiments had to be interpreted. Hidden and component patches were separated by site, projection basis, strength, first-step versus sustained application, and whether the output was only a next-token logit or a generated continuation. The late June 21 evidence also made post-box state a mechanism object: a row can carry enough state to repair a descriptor while still falling into a wrong same-class coordinate basin or a wrong next-row route.

The central June 22 evidence family was the population and formation redesign. The work stopped treating a vivid person/backpack-style case as the sampling prior and built deterministic row surfaces over train and validation examples. This included formation-position row builders, train/validation candidate banks, rollout-label attachment for validation rows, compact probe panels, teacher-forced formation rows, robust-position formation readouts, staged span continuations, staged-slot readout rows, launch filters, and guided-delta reducers. These were central, not supporting, because they changed the study's unit of evidence from selected case microscopes to row populations with explicit split, regime, position, and failure-locus metadata.

The central June 23 evidence family was reduction and stratification. The newer reducers split the broad pre-x1 failure population into mutually different mechanism families: coordinate-mode x1 onset failures, premature boundary or wrapper-mode failures, small-object extent or weak-evidence rows, tail or delayed-evidence rows, strict clean local repair rows, rank-only movements, intrusive slot transitions, donor capture, origin collapse, and worse-or-escape rows. The June 23 component passes then localized the clean coordinate-onset problem toward the layer-17 to layer-18 ownership transition while moving wrapper-mode rows to a separate boundary/router track.

Supporting documents include earlier cohort selection, duplicate audits, behavior bridges, descriptor-boundary probes, attention/value route probes, adapter and local-ridge analyses, and initial continuation taxonomies. These remain important because they explain why simple explanations were rejected, but later evidence supersedes their strongest interpretations.

The study used several evidence scopes. The scope labels matter because many results are narrow but mechanistically useful:

- Observational rollout analysis: free model outputs on a limited validation slice.
- Candidate-only analysis: target/candidate logits or identity margins without behavior perturbation.
- Teacher-forced readout: model scored at ground-truth or constructed prefixes.
- Prefix continuation: short deterministic generation from a selected prefix.
- Hidden readout: hidden states projected through output surfaces without intervention.
- Readout-only projection: hidden directions analyzed without changing the model.
- First-step causal patch: a hidden, residual, attention, MLP, or direction vector is applied only at the boundary that predicts the next token.
- Sustained or all-steps causal patch: the patch is repeatedly applied during generation; this is a stress test, not natural one-shot unfolding.
- Span-aware continuation: a patched or forced first token is followed by greedy unpatched continuation, with current-object and next-object spans scored separately.
- Population scaffold: larger train and validation candidate banks, usually teacher-forced.
- Tiny or smoke scope: one to a few states, used for mechanism discovery or implementation validation.
- Selected panel scope: curated states for a specific mechanism or failure family.
- Per96, per128, and per256 scopes: broader train and validation row banks used to reduce single-case and high-frequency-descriptor bias.
- Model-backed scope: a forward pass or generation was actually run through the model.
- Post-hoc reducer scope: existing artifact rows were reclassified, grouped, or summarized without rerunning the model.

The late-window experimental process had the following design logic.

1. Falsify descriptor-only and static-output explanations before expanding.

The hidden patch continuation and projection probes showed that descriptor repair is real but insufficient. A late residual or transported descriptor direction can flip a wrong descriptor into the target class and still generate a valid box for a different same-class instance. Geometry-basis falsifiers and coordinate score traces showed that the wrong coordinate family can already be top-ranked at generated coordinate slots. Therefore later experiments were designed around instance ownership and coordinate-basin selection, not just descriptor logits.

2. Separate immediate-token effects from generated-span effects.

The study repeatedly distinguished first-token rescue from span repair. A patch can make the next coordinate or boundary token look better while still failing to produce a valid target-overlapping object. Continuation reducers therefore scored whether the current open box was repaired, whether the next generated object was repaired, whether the descriptor matched, whether the box was complete and valid, and whether target-overlap thresholds were met. This prevented rank or descriptor improvements from being promoted into binding claims.

3. Build deterministic formation positions before asking causal questions.

The June 22 formation row design created eight meaningful cutpoints around an object span: descriptor onset, descriptor end, object-reference end, box start, pre-x1, post-x1, box close, and next-object onset. The key design constraint was suffix replay: when an earlier position is perturbed, later state must be recomputed through the remaining prefix rather than treating a stale hidden state as causal evidence. Early row builders were model-free by design so that row identity, target token, target coordinate, completed box, source row, object index, split, and regime could be audited before GPU work.

4. Escape the single-case microscope through candidate banks.

The train/validation candidate bank selected object rows by ground-truth structure regimes: repeated class, nearby duplicate basin, crowded scene, small object, termination tail, and simple control. Image caps prevented one extreme image from dominating a regime. Later descriptor caps prevented high-frequency categories from silently dominating expanded banks. Validation rows were joined to existing rollout labels where possible, so matched controls, relaxed IoU rows, and unmatched false negatives could be selected before hidden-state probing. Train rows did not have the same free-rollout labels, but they remained central because teacher-forced readout can still test whether a trained sequence has a strong local binding state.

5. Use teacher-forced readout to separate schema readiness from coordinate ownership.

The robust-position readouts deliberately focused on positions less confounded by multi-token descriptor scoring: pre-x1, post-x1, box close, and next-object onset. This made the main contrast crisp. The model usually had the correct token type and structural schema, but pre-x1 coordinate ownership was weak. After x1 was supplied, later coordinates became much more local. This process choice made the later mechanism question more precise: not "can the model output boxes?" but "why does the ownership state fail before the first coordinate anchor?"

6. Expand population size in stages and record why each expansion existed.

The per24 and per48 passes tested whether the broad pre-x1 pattern existed beyond the first train/validation panel. The per96 pass added descriptor caps and made the broad train/validation scaffold more robust. The per128 pass checked whether guided-delta and destination-family conclusions survived a larger row bank. The per256 pass deliberately emphasized trained-sequence failures and matched validation analogs, then warned when the first per256 destination taxonomy was train-only and needed a validation mirror before train-versus-validation claims. Each expansion had a role: first to avoid the original semantic pair, then to reduce frequency bias, then to test train-first severity, then to compare train and validation under launch filters.

7. Convert readout failures into staged forcing families.

Span continuation was used as a diagnostic ladder. Free pre-x1 continuation was compared with forcing x1, forcing x1-y1, forcing x1-y1-x2, and forcing the full target box. This created a failure-locus taxonomy: x1 onset rescue, y1 or anchor rescue, x2 extent rescue, y2 final-extent rescue, full-box closure or router failure, and already-successful rows. The design reason was to avoid calling every miss "visual non-perception." If a row recovers when x1 or x1-y1-x2 is supplied, then the model has some usable object evidence downstream of the missing anchor.

8. Treat donor-state patches as probes of ownership, not as candidate training targets.

Guided-delta patches moved hidden state from later slots back into pre-x1 receivers. Raw donor import and donor-minus-previous-slot were compared because the study needed to distinguish object information from slot phase and coordinate value. The result was diagnostic: target rank often moved, but donor-nearer or slot-intrusive outcomes were common. The design consequence was to create clean-versus-intrusive reducers and destination-family panels before any further localization. Whole donor-state matching was demoted as a training idea.

9. Add destination-family labels before component localization.

Reducers split patch outcomes into receiver repair, donor-slot capture, previous or control-slot pull, baseline inertia, coordinate-edge ambiguity, and off-basin escape. Later reducers split ownership into target geometry, target rank, donor, origin, worse-or-escape, and other. These labels changed the meaning of component patches: a rank improvement can still be donor ownership, origin collapse, wrapper-mode inertia, or rank-only movement. Component analysis after June 22 therefore used destination-pure or ownership-role panels instead of pooled pre-x1 failures.

10. Correct row semantics when newer notes exposed ambiguity.

The baseline-inertia and boundary-gate notes corrected an important semantic error. Some rows carried an x1 comparison label but the realized staged prefix already contained the target object's coordinates and was missing only box-end. These are completed-box or wrapper-routing states, not clean pre-x1 onset states. The report therefore separates ordinary coordinate-mode pre-x1 failures from premature boundary or wrapper-mode failures. This correction supersedes any older phrasing that treated all baseline-inertia rows as direct visual or coordinate-onset failures.

11. Localize the clean coordinate-onset transition to layers 17 and 18.

The clean pre-x1 ownership reducer showed a soft mixed state at layer 17 and a donor snap by layer 18 under compatible donor patches. Simple-control donors collapsed to an origin attractor and therefore were not repair controls. The component-output tomography panel then separated layer-17 self-attention, layer-18 self-attention, and MLP roles. Layer-17 self-attention exposed compatible basin identity with high gain; in bad rows this meant donor or origin, while in rare compatible rows it could look target-like. Layer-18 self-attention reduced donor-like landing and looked more like a resolver or redistribution point. MLP mostly shaped rank, value, or basin stabilization.

12. Route boundary failures separately from coordinate-onset failures.

The crowded closure/router direction-patch experiments provided the clearest current positive causal handle. A large boundary direction at layer 24 could flip all current crowded-router candidates at high strength, and when the first token flipped to box-end the next unpatched token followed the expected route. This is evidence for an under-margined boundary-router state, not for pre-x1 coordinate repair. The per256 failure-mode reducer reinforced the split by showing that premature boundary-mode rows had zero strict clean coordinate repair across layer 17 and 18 component sites, even when rank moved dramatically.

13. Keep train and validation as evidence roles, not a binary explanation.

Train rows are central because they show failures under exact teacher-forced trained sequences. Validation rows are central because they provide held-out analogs and free-rollout labels when available. The late-window evidence does not support a clean train-memorized versus validation-unseen mechanism split. Train rows can be more severe because of selection, while validation analogs can be similarly or more responsive under component patches. The recommended axis is locus and destination family, with train and validation kept paired or stratified.

14. Keep training blocked until the evidence is span-aware and locus-specific.

The promotion rule became stricter as the process matured. A coordinate-onset training objective is premature unless it repairs full object spans across a replicated population, preserves syntax, avoids donor and origin capture, passes controls, and is stratified by failure locus. Descriptor gain, target-rank movement, donor-state matching, or first-token improvements alone are not training-ready.

## Experiment Families and Their Purposes

The families below are organized by mechanism and evidence role, not as a chronological log. The late-window formation builders, population banks, staged forcing runs, guided-delta reducers, boundary-gate probes, and ownership-transition reducers are treated as central experiment families because they changed what later results mean.

### Initial Cohort and Deep Probe

The first cohort selected a range of validation examples where the two template families produced duplicates, low recall, overemission, or structurally interesting disagreements. Early deep probes showed that coordinate-token type was usually stable: the model tended to know it was in a coordinate slot. The fragile part was not the existence of a coordinate vocabulary mode, but which coordinate basin won and whether the object state remained bound to the intended target.

### Phase 3 Audit Gates

The audit-adjusted phase tested whether early duplicate evidence survived stronger controls. Repetition-penalty correction was especially important. Raw duplicate emitted-coordinate probability advantages flipped or collapsed after applying the rollout repetition penalty. Held-out null comparisons also showed that simple factors such as object position, same-class density, and phase could explain much of the apparent signal. The result was a demotion of the strong raw duplicate-probability and previous-anchor causal stories. What survived was weaker but important: rank and selection fragility around the coordinate basin.

### Realized Behavior Bridge

The next stage tested whether prefix-side target guidance actually changed generated behavior. It used deterministic short continuations and guidance arms such as coordinate seed, descriptor seed, and descriptor-plus-coordinate seed. The bridge produced real behavior movement, but not enough to promote to latent patching or training. Only 2 target-guided rows changed to the target, parse preservation was only 0.625 on target-guided arms, and the positive cases were not broad enough or outside enough of the discovery subset.

The bridge did establish that coordinate-bearing guidance can nudge emitted boxes and that local schema states are often well understood. It also exposed the separation between coordinate slots and boundary closure: the model can walk through the expected descriptor-to-box trajectory and still fail to close or route correctly.

### Boundary and Closure Probes

Hidden readouts over box-end states found late boundary overwrites. In schema-break open-box failures, an earlier late layer could still favor box-end, but the final layer flipped toward object-reference boundary tokens or termination. Raw decoder-layer boundary-direction patches rescued many such failures while preserving clean states. A joint boundary direction suppressing both object-reference boundary alternatives rescued all 25 selected schema-break states at high strength in that panel, while preserving 61 clean states.

Further attention and value-route probes showed that this was not a simple wrong-attention story. Some heads attending through object-reference anchors supported closure. A later head had a strong role in suppressing object-reference boundary logits, but suppressing its value contribution could also expose termination. The immediate boundary failure looked language-side and template-history-heavy rather than visually driven at the final boundary site.

### Coordinate Basin and Value-Route Probes

Coordinate probes examined whether wrong coordinate outputs were direct copies, local basin effects, adapter distortions, or component-specific value routes. Several findings converged:

- Single coordinate-bin directions often improved target rank but failed to produce the exact target.
- Some stubborn coordinates were recoverable only by local-simplex directions that suppressed a family of neighboring wrong anchors.
- The coordinate-token adapter could amplify an already-wrong local ridge, but adapter magnitude alone did not explain all failures.
- Attention/value routes through current partial box and coordinate tokens could perturb coordinate bins, especially later slots, but they did not produce reliable full object repair.
- Random residual controls weakened claims that generic residual-only patches were specific mechanism handles.

The coordinate surface is best viewed as a rugged local energy landscape. The model often remains type-correct and locally plausible, yet chooses the wrong coordinate anchor within or near a basin.

### Same-Description History and Lock-In Probes

Same-description history was tested because duplicate-like behavior suggested that prior objects with the same descriptor might shape the next coordinate basin. Prefix counterfactuals replaced same-description history coordinates with sentinel values or target-shaped coordinates.

The result was conditional:

- Sentinel history often damaged target-coordinate scoring.
- Target-shaped same-description history could improve duplicate or repeated states.
- The same target-history perturbation could harm new-object states by pushing them into a wrong anchor basin.
- Final-query attention did not simply shift to literal previous coordinate tokens.
- Wrong anchors were often already rank-visible before becoming top-1.

This demoted the literal copy story. Same-description history appears to perturb a broader coordinate-basin selection field. In some states that field helps target binding; in others it selects an already-plausible wrong anchor.

### Post-Box Descriptor and Instance-Binding Probes

Post-box probes tested whether, after a completed box, the model can start the correct next object. Exact self-next transitions were locally strong: if the model stays on its own generated trajectory, it often knows to emit the next object-reference start or descriptor.

Counterfactual completed boxes were more revealing. A completed box from a wrong same-description or next-generated object could flip a descriptor basin, such as from a backpack descriptor toward a person descriptor. Hidden readouts showed that the target descriptor trace could appear before the final layer and then be overwritten late. A natural same-case baseline-minus-current residual delta could rescue the descriptor with low strength. However, that natural delta was not mostly a simple descriptor-token direction; most of its norm was elsewhere.

Component localization found that the rescue was carried mainly by the incoming late-block residual stream, not by isolated self-attention or MLP output at the selected late block. Projection analysis then showed a layerwise rotation: early late-site correction was mostly descriptor-orthogonal and state-like, while final late-site correction became descriptor-readable.

The decisive follow-up was continuation. Descriptor repair generated valid object spans, but the geometry did not overlap the selected target. In the backpack case, repaired descriptors often landed on another real same-class backpack in the image. Score traces at the generated coordinate slots showed that the wrong same-class coordinate bins were already top-1, while selected target bins were far down the coordinate ranking. Thus descriptor repair is real but not instance-binding repair.

### Formation-Time and Population Readouts

The study then moved from selected post-box examples to broader formation-time panels. Deterministic row builders materialized meaningful formation positions:

- Descriptor onset.
- Descriptor end.
- Object-reference end.
- Box start.
- Pre-x1.
- Post-x1.
- Box close.
- Next-object onset.

Candidate banks sampled train and validation rows across repeated class, duplicate-nearby, crowded, small object, termination tail, and simple-control regimes. Validation candidates were attached to existing rollout labels where available, then expanded into teacher-forced formation rows.

Broad readouts found the central pattern:

- Descriptor and structural transitions are usually strong.
- Pre-x1 is weak.
- Post-x1 is much stronger.
- Train and validation both show pre-x1 weakness.
- Simple controls are much easier than crowded, repeated, small-object, and duplicate-nearby rows.

At per256 scale, ordinary pre-x1 rows were coordinate-mode in both train and validation, with coordinate mass near 0.998 and coordinate top-1 class rate 1.0. Yet target rank was often weak: rank at or below 10 was about 0.315 for train and 0.262 for validation. After x1 was supplied, rank at or below 10 improved to about 0.640 for train and 0.579 for validation, and distance within 16 bins rose above 0.82 in both splits.

This strongly argues against a blanket visual non-perception story. A large fraction of rows become locally coordinate-ready once the first coordinate anchor is supplied.

### Staged Span Forcing

Staged forcing asked how much of the object span can be recovered if selected target coordinates are supplied.

Over 178 broad pre-x1 failure states:

- Free generation produced complete valid boxes with IoU at least 0.5 in 18 of 178.
- Forcing x1 raised that to 82 of 178.
- Forcing x1 and y1 raised it to 94 of 178.
- Forcing x1, y1, and x2 raised it to 137 of 178.
- Forcing the full target box raised it to 164 of 178.
- The remaining full-box failures were mostly closure or router failures.

This split the population into multiple loci:

- x1 onset or cursor failure.
- y1 or row-anchor failure.
- x2 extent failure.
- y2 final-extent failure.
- Full-box closure/router failure.
- Already successful rows.

Small objects were the clearest exception to simple x1 rescue: forcing x1 alone rarely produced strong IoU on small objects. These rows require separate locality, extent, and coordinate-smoothness analysis.

### Guided Delta, Slot-Phase, and Destination Families

Later-slot hidden states were used as donors for pre-x1 receivers. The idea was to see whether information available after x1 or after x1-y1 could be projected backward to repair pre-x1.

The broad result was not a clean repair. In one 352-row train/validation guided-delta run, target rank improved in about 0.597 of rows, but exact target top-1 was only about 0.006, and slot intrusion was about 0.906. Later-slot donor states were powerful coordinate attractors, but they often imported the donor slot or donor coordinate basin rather than repairing the receiver's x1 ownership.

Previous-slot deltas reduced donor capture compared with raw donor import, but they still rarely solved the pre-x1 decision. Per128 expansion showed the same shape: target rank improvement about 0.452, exact top-1 about 0.003, distance within 16 about 0.101, and slot intrusion about 0.690.

This motivated destination-family labels:

- Receiver repair: the desired rare positive family.
- Donor-slot capture: the donor's coordinate or slot value dominates.
- Previous or control-slot pull: the patch pulls toward an adjacent or control slot.
- Baseline inertia: the state remains near the original baseline or boundary mode.
- Coordinate-edge ambiguity: receiver and donor are close or locally ambiguous.
- Off-basin escape: the patch leaves the expected local basin family.

True receiver repair remained rare. Baseline inertia became the largest train-only destination family in the per256 train-failure panel.

### Baseline Inertia Correction and Boundary-Mode Split

A crucial late correction revised the interpretation of baseline inertia. Representative baseline-inertia rows had assistant prefixes that already contained all four target coordinates and were missing only the box-end decision. They still carried an x1 comparison label because of staged counterfactual semantics, but the rendered prefix was a completed-box boundary state, not a clean pre-x1 onset state.

This correction matters. Baseline inertia is a real failure family, but much of it is completed-box or wrapper-mode routing, not direct evidence that the model cannot perceive x1. In baseline-inertia rows, the top full-vocabulary token was usually a wrapper or boundary token, often box-end. Ordinary broad pre-x1 rows, by contrast, were coordinate-mode with near-total coordinate mass.

Therefore the current failure population has at least two distinct surfaces:

1. Coordinate-mode wrong-basin or weak-rank failures, where the model is ready to emit a coordinate but chooses the wrong coordinate basin.
2. Boundary or wrapper-mode failures, where the model is not in coordinate-emission mode and instead prefers box closure or object-reference boundary tokens.

These must not be pooled.

### Ownership Transition and Component Tomography

The clean pre-x1 ownership reducer analyzed compatible donor patches across layers 17 through 21. It separated ownership into target geometry, target rank, donor, origin, worse-or-escape, and other.

Across 595 compatible donor patch rows covering train and validation:

- Layer 17 was mixed and comparatively soft.
- By layer 18, compatible donor patches mostly snapped into donor ownership.
- True target-geometry repair was rare.
- Simple-control donors were not repair controls; they collapsed almost deterministically to the origin basin.

The most common transition signatures were donor snap or origin-to-donor snap. A small stable target-geometry subset existed, but it was rare and often already geometrically compatible.

A component-output tomography panel over 85 train and validation sequences and 340 component patch rows found:

- Layer-17 self-attention is a high-gain exposure path for compatible basin identity. In bad rows this exposes donor or origin; in stable target-geometry rows it can appear target-like because target and donor are already close.
- Layer-18 self-attention sharply reduces donor-like landing and acts more like a resolver or redistribution point.
- MLP output is more consistent with rank/value shaping and basin stabilization than first exposure of ownership.

This suggests that layer 17 is a promising source/value localization point for coordinate-mode onset failures, while layer 18 should be treated as a resolver rather than a simple writer.

### Per256 Failure-Mode Split

The newest per256 staged failure-mode reducer analyzed 492 guided-delta rows across train and validation component sites. It split the broad cohort into:

- 348 coordinate-mode rows.
- 144 wrapper-mode rows.
- 138 rows with box-end as the top token.

Failure locus counts included:

- 213 x1 onset-anchor failures.
- 138 premature boundary-mode failures.
- 66 small-object extent or weak-evidence rows.
- 69 tail no-rescue or delayed-evidence rows.
- 6 wrapper-mode router-competition rows.

A bounded layer 17 and 18 component-site scan over the recommended panel showed:

- Premature boundary-mode rows had zero strict clean local repair at every layer and site, with intrusion rate 1.0. These are router or boundary failures, not coordinate-basin repair targets.
- Coordinate-mode x1 onset-anchor failures repaired best at layer 17, with MLP strict clean local repair about 0.294 and self-attention about 0.255.
- Tail or delayed-evidence rows had a plausible layer-18 self-attention resolver signal.

This is now the strongest routing instruction for the next GPU work: do not mix coordinate-onset rows with premature boundary-mode rows.

## Main Results and Evidence

### 1. Strong Raw Duplicate-Probability and Previous-Anchor Stories Were Demoted

Early duplicate rows looked close to previous object anchors, and raw logits sometimes suggested duplicate advantages. After repetition-penalty correction, the raw emitted-coordinate probability advantage for duplicates flipped or collapsed. Rank gaps survived only weakly and directionally. Held-out null leaderboards showed strong contributions from object index, class density, and family/event phase. Candidate identity margins were positive in candidate-only matched-target rows but did not constitute realized behavior repair.

Current interpretation: duplication and low recall involve rank and basin-selection fragility, but the strong story that duplicates are simply high-confidence previous-anchor reuse is not supported.

Evidence scope: val200, teacher-forced and candidate-only audit gates, repetition-penalty adjusted, non-circular null comparisons.

### 2. Local Schema and Token Type Are Usually Strong

Across many probes, descriptor starts, box starts, object-reference ends, and box-close positions were often locally high confidence. Coordinate slots usually had nearly all probability mass inside the coordinate vocabulary.

Examples:

- Broad robust-position readouts showed ordinary pre-x1 coordinate mass near 0.998 for train and validation.
- Post-x1 rows became substantially more localized than pre-x1 rows.
- Box-close and next-object-onset readouts were often near-saturated, though next-object onset should be interpreted as boundary pressure in tail or simple-control rows, not always as a natural demand for another object.

Current interpretation: the dominant failure is not a generic inability to follow the output grammar. It is object ownership, coordinate-basin selection, and boundary routing under specific local states.

Evidence scope: broad teacher-forced train/validation readouts, per96 to per256, hidden logit-lens rows, staged-slot readouts.

### 3. Behavior-Level Prefix Guidance Was Real but Not Promotion-Ready

The realized bridge produced 96 behavior rows: 64 controls and 32 target-guided rows. Only 4 rows changed next-object behavior, only 2 changed to the target, and both target successes were discovery cases. Parse preservation on target-guided arms was 0.625, below the promotion threshold. Controls were mostly null, but wrong-control families were incomplete.

Current interpretation: prefix guidance can reveal local recoverability, but current evidence is not sufficient for latent patching or training. Coordinate guidance nudges boxes and basin trajectories more than it supplies exact object identity.

Evidence scope: selected 16-case bridge, deterministic short continuation, strict closed-object parsing.

### 4. Boundary Closure Has a Real Causal Handle, but It Is a Separate Mechanism

Schema-break open-box states showed a late-layer overwrite: earlier late hidden states could favor box-end, while final states flipped toward object-reference boundaries. Raw decoder-layer boundary directions causally rescued many failures while preserving clean states. A joint object-reference-boundary direction rescued all 25 selected schema-break rows in one panel.

The later crowded closure/router population strengthened this: in all 17 crowded-router candidates, a large layer-24 box-end versus object-reference-boundary direction flipped the close-box decision by strength 128, and every flipped state then followed the expected next-object route.

Current interpretation: crowded closure/router rows often contain a coherent next-object continuation, but the natural boundary margin is too weak or routed through the wrong boundary mode. This is one of the clearest causal handles, but it applies to boundary/router rows, not to clean pre-x1 coordinate-onset failures.

Evidence scope: selected hidden causal patch panel, selected boundary-head/value probes, staged-slot crowded train/validation candidate panel, all-current crowded candidate strength curve.

### 5. Descriptor Repair Is Not Instance Geometry Repair

Post-box hidden-state patches and layer-input residual deltas could repair descriptor output. Transported readout gradients and path-averaged gradients confirmed that the live descriptor axis is real and can repair generated descriptors. But continuations showed that repaired descriptors often landed on a wrong same-class spatial basin rather than the selected target.

In the backpack example, descriptor repair produced valid backpack spans, but the generated boxes overlapped a different real backpack, not the intended target. Score traces at coordinate slots showed wrong same-class bins as top-1, while selected target bins were far down the coordinate ranking.

Current interpretation: descriptor state, instance pointer, and coordinate basin are separable. Training or intervention should not treat descriptor rescue as object-binding rescue.

Evidence scope: selected post-box descriptor flips, hidden-state readout, layer-input causal patching, projection subspace analysis, first-step continuation, coordinate score traces.

### 6. Pre-X1 Is a Major Coordinate-Basin Onset Bottleneck

The strongest population-level pattern is the pre-x1 to post-x1 transition. At pre-x1, target rank and local distance are often weak; after the correct x1 is supplied, y1 and subsequent coordinates are much more locally readable.

In broad staged forcing, forcing x1 alone increased strict complete-valid IoU at or above 0.5 from 18 of 178 to 82 of 178. Forcing more coordinates increased the count further to 137 of 178 after x1-y1-x2 and 164 of 178 after the full target box.

Current interpretation: for many rows, the problem is not missing object evidence. It is entering the correct coordinate basin at the first coordinate anchor.

Evidence scope: 178-state broad train/validation pre-x1 failure panel, staged continuation and forcing, broad teacher-forced readouts.

### 7. Small Objects Are a Distinct Coordinate Smoothness and Extent Family

Small objects are not solved by x1 forcing alone. In broad forcing, small-object rows had very low IoU rescue from x1 alone compared with crowded, duplicate-nearby, repeated-class, or termination-tail rows. Later staged-slot readouts showed that small objects can preserve coordinate locality while failing exact coordinate rank.

Current interpretation: small-object failures should be analyzed as coordinate locality, quantization, exact-token smoothness, and extent estimation issues, not simply as object absence or boundary failure.

Evidence scope: staged forcing, staged-slot readout, layer-24 to final surface bridge, per96 and per128 population scaffolds.

### 8. Later-Slot Donor Deltas Are Diagnostic, Not Clean Repairs

Donor hidden deltas from later guided slots can move receiver pre-x1 logits, but they usually import donor slot phase or donor coordinate ownership. The largest guided-delta runs showed target-rank movement with rare exact target top-1 and high intrusion. Previous-slot deltas reduce donor capture compared with raw donor import but still do not form a clean repair vector.

Current interpretation: later slots carry useful coordinate and slot-transition information, but this information is entangled with coordinate value, slot phase, and object ownership. Future probes should split these factors rather than import a whole donor state.

Evidence scope: 352-row broad train/validation guided-delta run, v6 train-first panel, per128 and per256 guided-delta replications, clean-versus-intrusive reducers.

### 9. Destination Families Matter More Than Train Versus Validation

The per256 matched train/validation component runs did not support a simple train-memorized versus validation-unseen split. Train rows were often more severe because they were selected as failures, but validation analogs responded similarly or sometimes better to component patches. Destination-family profiles were similar across splits.

The useful axis is not primarily train versus validation. It is failure locus and destination family:

- Coordinate-mode x1 onset-anchor failure.
- Premature boundary mode.
- Donor-slot capture.
- Control-slot pull.
- Baseline or wrapper-mode inertia.
- Off-basin escape.
- Small-object extent or weak evidence.
- Tail no-rescue or delayed evidence.

Evidence scope: per256 train and validation robust-position readout, matched launch panel, component-site guided deltas, destination-family reducers.

### 10. Ownership Formation Has a Layer-17 to Layer-18 Transition

Clean pre-x1 ownership analyses found a soft mixed state around layer 17 and a strong donor snap by layer 18 under compatible donor patches. Component tomography suggests layer-17 self-attention exposes compatible basin identity, while layer-18 self-attention reduces donor-like landing and behaves like a resolver or redistribution point. MLP is more associated with rank and value shaping than first exposure.

Current interpretation: the coordinate-onset ownership state is not written by a single late universal component. It is a residual-stream ownership decision shaped by self-attention, MLP, donor compatibility, and basin geometry. The next fine-grained attention/value tomography should be role-stratified rather than pooled.

Evidence scope: 595 clean pre-x1 compatible donor patch rows, simple-control origin scan, 85-sequence component tomography panel, layer 17 and 18 component scan.

## Mechanism-Level Synthesis

The model appears to operate with a latent object-binding or cursor state that must coordinate several decisions:

1. Which object is currently owned by the row.
2. Which descriptor basin should be emitted.
3. Which coordinate basin should x1 enter.
4. How later coordinates should follow from x1.
5. Whether the completed box should close.
6. Whether the sequence should begin another object or terminate.

Failures occur when these state variables desynchronize.

### Coordinate-Mode Onset Failure

In ordinary pre-x1 rows, the model is usually in coordinate mode. The failure is that the x1 coordinate basin is wrong, weak, donor-like, origin-like, or not target-local enough. Once x1 is supplied, many rows become much more readable. This suggests that x1 is an ownership and cursor anchor, not just a coordinate value.

### Slot-Phase Intrusion

Later-slot states contain useful coordinate information, but they also carry slot phase. When projected backward to pre-x1, they often drag the receiver toward donor y1, x2, y2, or previous-slot basins. This explains why target rank can improve while the visible top coordinate becomes donor-nearer or control-nearer.

### Boundary and Wrapper-Mode Failure

Some failures are not coordinate-mode failures at all. They prefer box-end, object-reference start or end, or termination tokens. These rows must be handled by router and boundary-margin probes. The crowded closure/router family shows that some boundary failures have a clear large-margin causal handle.

### Descriptor-Only Repair

Late residual state can repair a descriptor, and this descriptor signal rotates into descriptor-readable coordinates near the final block. But descriptor repair can select a wrong same-class object. It does not guarantee target geometry, current-box repair, or next-row routing.

### Local Coordinate Ridge and Adapter Effects

Exact coordinate selection can fail even when the target neighborhood is locally present. The coordinate-token adapter can amplify local ridges, sometimes toward wrong anchors. Some stubborn failures need local-simplex directions rather than single-bin target directions. This aligns with small-object and final-extent failures where locality and exact rank diverge.

### History-Conditioned Basin Selection

Same-description history can help or hurt. It is not literal copying. It reshapes a broader residual and coordinate-basin field. Target-shaped history helps some duplicate or repeated states but harms new-object states by making wrong anchors win. The mechanism is state-conditional.

### Train Rows as Mechanistic Evidence

Trained-sequence failures under teacher-forced ground-truth prefixes are highly informative. They show that local autoregressive state selection can fail even for objects seen during training. This weakens pure visual non-perception and pure held-out generalization explanations.

## What Later Evidence Revised or Contradicted

Several earlier interpretations were explicitly revised:

- Raw duplicate-probability and previous-anchor reuse were demoted after repetition-penalty correction and null comparisons.
- Candidate-only identity margins did not justify behavior claims; realized continuation was required.
- Descriptor repair was revised from a possible object-binding handle to a descriptor-only or same-class-basin handle unless geometry and span repair also occur.
- Static descriptor projection was revised because transported readout directions better capture the live readout surface.
- Late target-bbox output-embedding directions were falsified at the descriptor boundary for the selected backpack case.
- Same-description history was revised from possible literal copying to a conditional basin-field perturbation.
- Baseline inertia was revised from a broad pre-x1 coordinate failure to a mixed family, often completed-box or wrapper-mode boundary routing.
- Layer-24 pooled component results were revised by layer 17 and 18 scans: layer 17 is better for coordinate-mode x1 onset repair, while layer 18 often acts as resolver or produces large rank swings without clean repair.
- Train-versus-validation was demoted as the primary axis; locus and destination family became more important.

## Uncertainties and Limitations

The evidence is strong enough to define a mechanism map, but not enough for training or final causal closure.

Major limitations:

- Many causal probes are selected panels, tiny smokes, or mechanism-biased slices rather than full validation estimates.
- Teacher-forced prefixes test local model state under supplied ground-truth context; they do not by themselves explain all free-rollout failures.
- Some intervention strengths are artificial and should be interpreted as causal diagnostics, not naturalistic model behavior.
- First-token or rank repair is often not span repair. Several probes showed that rank movement, descriptor repair, or x1 improvement can fail to produce a correct object span.
- Analog matching between train and validation rows is still loose in area, center, tail length, and failure severity in several panels.
- Multi-token descriptor scoring remains less mature than coordinate and boundary scoring.
- Some artifacts had operational failures or semantic corrections; rejected runs should not be used as model evidence.
- Destination-family labels are improving but still coarse; off-basin escape and tail/delayed evidence likely need subdivision.
- The exact upstream source of clean target-geometry ownership remains unresolved.
- The coordinate-token adapter's role is important but not fully separated from hidden-state and local-basin effects.

## Recommended Next Directions

### 1. Split Future Work by Failure Locus

Do not pool all pre-x1 failures. Use separate panels for:

- Coordinate-mode x1 onset-anchor failures.
- Premature boundary or wrapper-mode failures.
- Small-object extent and exact-rank failures.
- Tail no-rescue or delayed-evidence failures.
- Strict clean local repair versus intrusive slot transition.
- Donor-slot capture versus control-slot pull versus baseline inertia.

### 2. Run Layer-17 Attention and Value Tomography for Coordinate-Mode Onset

The next coordinate-mode probe should focus on layer 17 self-attention and MLP source/value decomposition, stratified by ownership role:

- Stable target geometry.
- Donor snap.
- Origin collapse.
- Rank-only repair.
- Persistent worse or escape.
- Strict clean repair versus intrusive transition.

The question is which source regions and value components expose object identity, coordinate value, and slot phase before the model emits x1.

### 3. Treat Layer 18 as a Resolver

Layer 18 often produces large rank movement, but not necessarily clean ownership. It should be probed as a resolver or redistribution point rather than as a simple writer of coordinate ownership.

### 4. Expand Boundary/Router Probes Separately

For wrapper-mode and crowded closure rows, continue boundary/router-margin experiments:

- Box-end versus object-reference start.
- Box-end versus object-reference end.
- Box-end versus termination.
- Strength thresholds across train and validation rows.
- Span-aware continuation after the first patched token.

The crowded family has the strongest current causal handle and deserves broader replication outside capped launch panels.

### 5. Purify Slot-Phase and Donor-State Handles

Later-slot hidden deltas should be decomposed. Candidate approaches:

- Donor minus previous-slot controls.
- Same-slot donor controls.
- Projection that removes donor coordinate value while preserving transition information.
- Separate object identity, coordinate value, and slot phase bases.
- Destination-family-conditioned source/value tomography.

Do not optimize toward generic donor-state matching.

### 6. Study Small-Object Smoothness and Coordinate Locality

Small-object rows should be analyzed with local coordinate neighborhoods, exact-token rank, radius mass, adapter deltas, and extent slots. The key distinction is whether the model has weak visual evidence, preserved local evidence but poor exact-token smoothness, or a delayed slot resolver failure.

### 7. Keep Train-First Failure Mining Central

Mine additional trained-sequence rows that fail under teacher-forced ground-truth prefixes. Match validation analogs by:

- Candidate regime.
- Descriptor or category.
- Object area.
- Coordinate neighborhood.
- Object order.
- Repetition and same-description pressure.
- Remaining-object tail length.
- Post-x1 recovery or nonrecovery.

Train rows are not a nuisance; they are a powerful way to separate dataset exposure from local autoregressive state failure.

### 8. Require Span-Aware Evidence Before Training

Training remains blocked until a candidate handle:

- Replicates beyond a single vivid case.
- Applies to a clearly defined failure locus.
- Repairs a full valid object span, not only descriptor or coordinate rank.
- Preserves parse and clean controls.
- Reports target gains, target losses, malformed regressions, wrong-family substitutions, and boundary failures.
- Survives norm-matched random, wrong-region, wrong-head, noop, and too-late controls.
- Has a plausible rollback and artifact interpretation path.

The first plausible training candidates would be localized, state-conditioned pre-x1 basin objectives or small formation-layer corrector probes. Full hidden-state matching and descriptor-only gain objectives are not justified.

## Bottom Line

The study has moved from a duplicate/previous-anchor suspicion to a richer model of autoregressive binding as a state-machine problem. The model is usually able to emit the correct type of token and often contains local visual or coordinate evidence, but the row state can fall into the wrong ownership, coordinate, slot-phase, descriptor, or boundary regime.

The most important current split is:

- Coordinate-mode pre-x1 onset failures: the model is ready to emit coordinates but lacks the correct target ownership or basin anchor.
- Boundary or wrapper-mode failures: the model is not in the coordinate mode and instead chooses closure, object-reference boundary, or termination behavior.

The strongest positive causal boundary handle should be expanded, but it should not be confused with coordinate-onset repair. The coordinate-onset frontier now requires role-stratified layer-17 and layer-18 attention/value tomography over population-first train and validation panels.

The report's practical recommendation is simple: keep the work population-first, locus-stratified, and span-aware. The single vivid person/backpack-style examples remain useful microscopes, but they should no longer define the experiment distribution or justify training.
