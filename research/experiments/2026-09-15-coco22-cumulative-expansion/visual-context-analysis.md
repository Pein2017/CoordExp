# Source-bound analysis of tiny boxes and asymmetric localization

This is a read-only analysis of user-selected examples, not a new mechanism
experiment or a change to the frozen 376-owner teacher. Evidence packet:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/runtime-repair-v1/visual-context-evidence.json`. Raw discovery rows and per-policy packets are preserved under
the current output root's `discovery-v2/`.

## What the examples establish

- The user-supplied 510122 GT crops and 15-box overview are source annotation
  views, not model rollouts. The sky speck is COCO annotation 1855388 (kite),
  around 6x3 pixels. It was already HOLD and excluded from trusted teachers.
  GT index 2, annotation 1329432, was excluded as a duplicate of 1332524.
  The left occluded person GT index 0 is a verified distinct owner. These GT
  screenshots cannot establish either model binding success or model sky noise.
- All four actual discovery policies for 510122 output the obvious large kite
  and do not reproduce the tiny GT kite. They also recover the previously
  unlabeled checked-shirt person, a concrete positive model discovery signal.
  This is one image and four fixed trajectories, not a general binding score.
- In 335722 at T=.7, p17 is donut [502,217,548,247] on a blank wall; p22 is
  donut [548,328,616,351] on a chair back. Both are syntactically complete
  class/box rows and genuine model grounding errors. They are not identical
  sizes: about 53x26 and 79x20 pixels. The 301-token sequence ends with EOS, far
  below the 3084 cap. It also contains 8 malformed/raw-dropped spans, including
  missing coordinates and inserted text near these wrong boxes. Valid and
  wrong donuts are interleaved; neither a pure late-tail explanation nor a
  forced fixed-count explanation follows from this record.
- In 124185 at T=.7, p4's rendered box [451,129,681,816] extends from the book
  stack into the red bag. Horizontal placement can be useful while the lower
  edge is plainly excessive. But the raw row first emits three coordinates,
  inserts repetitive English, then reopens a box; its parsed description is
  not the clean COCO class book. The whole 3084-token sequence is capped.
  It is a mixed protocol/localization failure, not a clean four-coordinate
  instance-binding specimen. At T=.1 there are also clean book-labelled boxes
  over the red bag region; the broader book grounding problem therefore does
  not disappear merely by rejecting malformed syntax.

## Interpretation and boundaries

Separate annotation defects, object existence/category, instance identity,
localization, and sequence/termination validity. Small area is a screening cue,
not proof of falsehood. Tiny indistinguishable food items should remain HOLD
when their count/category or identity cannot be established; magnification
does not add source information. A clearly identifiable small instance may
use a reasonable coarse box. Do not turn a whole dish into one donut target
under a one-instance-per-row protocol. A future ignore/crowd rule or area
filter must be versioned separately; no denominator changes in this run.

The shown rectangles suggest class/shape priors surviving while location or
generation state becomes unreliable. This is a hypothesis, not a proven
mechanism. Bad GT can teach spurious mappings if actually exposed during
training, but neither exposure to this exact GT nor its causal contribution
to these model errors has been established. The newly frozen teacher excludes
the speck. Current shared axis-validity hinge only enforces ordered axes; a
wrong but well-ordered rectangle can satisfy it. Grounding is still learned
through the supervised coordinate targets; the hinge is not an object verifier.

## Visual-register hypothesis: plausible, untested here

[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
reports high-norm background patch tokens that carry global information and
harm local feature quality; dedicated registers help in the tested ViTs.
[Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010)
studies sparse register neurons and test-time interventions, including VLMs.
Neither establishes this mechanism in our Qwen checkpoint. A coordinate
pointing at a background patch does not prove the decoder attended to that
patch, and high attention alone does not establish causal grounding failure.

After the main arm, the cheapest discriminating study would freeze these
wrong donut rows and nearby valid same-class controls, trace the exact saved
prefixes, and inspect vision-token norm/attention at the encoder and language
interface. Only a reproducible association should motivate a matched
intervention on the suspected tokens versus ordinary background controls.
Require selective reduction of wrong boxes while retaining valid owners;
fewer boxes or earlier stopping alone is not evidence of repaired grounding.
Any prefix correction, register intervention or per-edge objective is a
separate future contrast, not added to the current main arm.

## Additional coordinate-context observation

The immediately preceding 124185 T=.7 handbag row has box
[368,520,681,804], whereas the rendered book box is [451,129,681,816].
The shared x2 and nearby y2 are compatible with coordinate carryover or
instance mixing. This correlation does not distinguish decoder-prefix
copying from a visual grounding error. Test on clean, source-bound rows
before attributing a cross-corner mechanism.

The subsequent explicit user visibility ruling is persisted separately in
`runtime-repair-v1/user-visibility-ruling.json`: prior-only, visually
unresolvable targets should be excluded from the next supervised/required
recall standard. The current tiny kite already satisfies that exclusion;
no current teacher changes are made.
