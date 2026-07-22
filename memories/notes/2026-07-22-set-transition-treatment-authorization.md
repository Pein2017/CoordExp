# Set-Transition Treatment Authorization After the Completion Study

Source session: 019f4a19-d81c-75a2-84b0-2c20379e686e, July 22 2026.

Source handles:

* `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-human-refined-greedy-set-completion-conditions/results.md`
* `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/results.md`
* `openspec/changes/add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot/`

Captured: 2026-07-22.

## Closed diagnostic result

The July 22 human-refined completion study is complete. It executed 63
image-by-prefix-depth states. Native decoding produced 444 rows assigned to 248
trusted physical owners. Repeated terminal suppression produced 622 rows and
260 owner assignments but did not complete any additional image. Therefore,
terminal suppression is a diagnostic intervention, not a treatment.

For the same covered-object set, changing prefix order changed the future owner
set in 13 of 36 paired states and changed completion in 2 of 36. Prefix order
is therefore causally active; the model does not expose a demonstrated
order-free covered-set state.

Coordinate binding is conditional rather than governed by one universal
boundary token. The first horizontal coordinate can distinguish horizontally
separated instances, while vertically separated same-class objects may require
the first vertical coordinate. Later extent coordinates can still fail after
the earlier coordinates identify the correct local area. Treat entity discovery
and complete-box geometry as separate outcomes.

## Treatment authorization

The user authorized an autonomous 256-image treatment screen and authorized
promotion to 1,024 images if the 256-image evidence is promising. Eight GPUs
may be used. The twelve human-refined validation images remain development and
validation evidence only and must never contribute training gradients.

The current treatment question is whether training can move probability from
an actual harmful branch under a self-generated prefix toward at least one
verified, physically new uncovered owner while retaining ordinary Source-model
routes. It must not impose one canonical next owner, require uniform probability
over all remaining owners, or define success as longer output alone.

The minimum credible design must compare the unchanged Source checkpoint with
a treated model and must preserve exact prefix, tokenizer, checkpoint, image,
physical-owner, and candidate-row provenance. Unknown or annotation-ambiguous
predictions are neutral, not negative. The primary free-rollout judgment is
unique physical-owner coverage together with ordinary-owner retention,
duplicates, malformed rows, semantic stability, and geometry; fixed-prefix
training margins are explanatory only.

Do not scale the earlier coordinate-only objective or the earlier single-route
positive-imitation objective unchanged. The coordinate screen improved local
teacher-forced margins without clean-rollout benefit. Single-route imitation
shifted route families but exchanged newly gained owners for ordinary owners.
The next treatment therefore requires multiple safe positive alternatives when
available and explicit Source-route preservation.

Promotion from 256 to 1,024 is a lead-agent judgment rather than a fixed metric
threshold. It requires a real clean-rollout signal that looks like set expansion
or materially safer redistribution, not merely lower training loss, longer
sequences, selected-route imitation, or terminal suppression.
