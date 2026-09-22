> LATEST USER RULING: dog and chair accepted with accurate localization; person rejected as the left person's hand fragment. This supersedes the provisional visual interpretation below. See ../user-adjudication-v1/summary.json. Historical references and images remain unchanged.

# Three historical extent-error examples

Each figure shows one full source image with the detector context window (yellow),
the exact retained Co-DETR input PNG with the rollout box (orange), and the same
PNG with the selected Co-DETR box (cyan). Neither prediction is ground truth.
The actual historical detector used context mode:3x candidate width/height,
minimum128px/side, clipped to the source image, then native keep-ratio Resize at
(2048,1280), no flip. Model inference was not rerun for these figures.

## Current lead observations, not replacements for the historical labels

-237954:p11: dog in a pasted photograph; historical label too_tight. Under the
 current real-original-scene rule, this depicted object is out of scope for a
 new teacher regardless of detector agreement. This is not proof of no dog pixels.
-307814:p0: chair is real and identifiable, with truncated/occluded extent.
 Historical label too_loose. The exact correct lower extent needs a stated
 visible-versus-amodal convention; the current visual alone does not independently
 certify this as a hard negative. Hold any changed geometry judgment for adjudication.
-134520:p0: the matched region is an arm fragment; the tight crop removes the
 people visible in the full source. Agreement does not establish an independent
 person owner or a correct person extent. Historical label too_tight is preserved.

The phrase 'three definite errors' refers to the previous provisional reference,
not three newly verified universal detector failures. No reference labels or
historical20/64 metrics were rewritten in this visualization turn.

## Rendering integrity

Source image bytes and saved crop PNGs pass their original hashes. All crop
windows and dimensions are preserved. The current JPEG decoder reproduces the
first crop exactly; color crops differ slightly from current source recropping
(mean channel differences are recorded in manifest.json). The displayed detector
panels always use the exact saved PNG, never a regenerated crop. The cause of
those decoder/output differences was not investigated or used to change labels.

## Research scope clarification

Human13 pure CE already attained392/392 with140 updates, including IoU80 and
zero unmatched outputs. COCO22 is a later fixed-teacher cumulative fitting test
with full old-data replay, not new evidence for the existence of small-panel
SFT fit and not an online rollout-refresh algorithm. Its preparation used
multiple discovery policies plus trusted GT; calling it single-rollout per image
would be inaccurate. Preserve it as a baseline and do not infer refresh value
from final training-set fit.
