# Implementation preparation: no new model result

User decisions are fixed in [the protocol](unit.md):18-image full-label positives,
class-agnostic strict IoU>0.9 overlap events, model-learned behavior, two16-update
arms and coverage observations without a fixed retention threshold. The requested
implementation worker is GPT-6.1-Sol/xhigh. Native execution is not yet released.

Read-only preparation inspected all570 boxes: no degenerate/out-of-range box or
exact within-image box collision;11,036 distinct within-image GT pairs,7,092
same-category and3,944 cross-category. Maximum IoU is0.7878787879, from image14038
book annotations-132/-130; cross-category maximum is0.6503208066. These are input
diagnostics, not prediction-error measurements or proof that0.9 is identity-safe.

Thirteen of18 images overlap the historical val200 image list. This is an exposed
training-internal study. The stronger anchor's historical median-normalized val200
mAP48.0328% selects the starting recipe, not a fresh native result for this unit.

Current status: protocol prepared; no implementation, new inference or training
has been accepted. Worker technical candidates and exact evidence locators will
be appended here; lead acceptance remains separate in state/rulings.
