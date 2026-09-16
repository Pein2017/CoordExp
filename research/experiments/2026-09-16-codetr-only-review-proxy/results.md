# Co-DETR-only proxy: accepted user adjudication

On 2026-09-16 the user inspected the source/crop/rollout/Co-DETR figures and ruled:

> person 的那个是 wrong,因为只有两个显著的 person,而 bbox 指向了左边 person 的左手小区域; 其他两张,我理解都是对的; 有chair和dog且定位准确

This supersedes the provisional extent judgments for these three cases. The
[historical retained replay](retained-replay.md), original labels and receipt remain preserved.
Do not continue citing its three defective cases as the current accepted truth.

| Case | Current user ruling |
|---|---|
|237954:p11 dog | Real depicted dog with acceptable, accurate localization; usable proxy support. |
|307814:p0 chair | Real chair with acceptable, accurate localization; usable proxy support. |
|134520:p0 person | Wrong independent-person prediction: a hand region belonging to the left person. |

With these three overrides, the 20 detector-supported candidates comprise 15 clean,
1 defective and 4 gray. Only these 3 cases are newly user-adjudicated; all other
labels remain inherited provisional visual references. The unchanged combined
proxy supports 14, now 12 clean, 0 defective and 2 gray. These are reference updates,
not new model performance measurements or an independent calibration.

The remaining error demonstrates part-versus-whole/owner identity ambiguity in
a tight context crop. It is not evidence that all high-overlap detector/rollout
agreement is bad. The dog/chair labels show that the earlier visual reference
was stricter than the user's acceptable extent criterion. Do not fit a proxy to
those superseded hard negatives or silently promote the new supported rows into
training. Depicted-object teacher scope is separate from whether these pixels
contain a correctly localized dog; no blanket teacher-scope override is inferred.

## Input route verified

These historical Co-DETR calls already used unmatched-centered 3x context crops,
minimum 128px/side, clipped to the source image, followed by native keep-ratio
Resize at (2048,1280). They were not full-image detector calls. Proposed full-image
context checks are an additional control, not the actual historical route.

## Research direction correction

The [Human13 pure-CE result](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-human13-pure-ce-replay/results.md)
already demonstrated 392/392 at IoU80 and zero unmatched after 140 magnitude-only
CE updates from Source. COCO22 adds only the narrower result of successful
continuation to 11 additional images with full old-data replay. Its larger image
count is not a larger owner count than Human13, nor evidence of rollout-refresh
value or a new SFT algorithm.

Preparation used trusted GT and multiple discovery policies, followed by review
and a frozen teacher. It was one offline acquisition stage, not one rollout per
image, and not an iterative correction loop. The lead should have distinguished
this limited incremental question from the already settled overfitting result
before recommending another fitting stage. Retain COCO22 as a baseline, not a
new capacity discovery or authorization for repeated larger-panel SFT studies.

Any next algorithmic contrast should first name what refreshed acquisition or
correction is expected to improve beyond fixed-teacher SFT, under matched update
and review budgets. No new research arm or training launch follows from this note.

## Durable evidence

- [User rulings and both updated counts](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-codetr-only-review-proxy/user-adjudication-v1/summary.json)
- [Per-case reference versions](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-codetr-only-review-proxy/user-adjudication-v1/cases.json)
- [Original figure bindings](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-codetr-only-review-proxy/visual-three-v1/manifest.json)
- [Figures with user-ruling captions](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-codetr-only-review-proxy/visual-three-user-v1/manifest.json)
