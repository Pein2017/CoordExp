# Sol versus Luna visual-review routing calibration

Status: SUPERSEDED by user-ruling-v2.json. The user accepts C09 as a valid single-target bbox with minor localization error despite overlapping another person. The original C09 reference and resulting 8/8 versus 7/8 comparison must not be used to rank reviewers. Raw reviewer outputs and the original frozen reference remain preserved. Luna-max is being assessed using the identical brief; all routes will be interpreted under corrected semantics.

Matched input: ten boxes, same original/full bbox/marked-context views and instructions; Sol-high and Luna-high were blind to each other and the frozen lead references. Two group boxes were constructed positive controls; C09 was a natural generated crowded-person box.

| Measure | Sol | Luna |
|---|---:|---:|
| Clear assessment agreement with lead reference | 8/8 | 7/8 |
| Group boxes labeled merged | 3/3 | 2/3 |
| Unsafe or ambiguous cases admitted | 0/5 | 0/5 |
| Coherent single-owner cases unnecessarily held | 1/4 | 1/4 |

C09: Luna described both people but reported one owner; Sol reported two merged owners. C04: Luna confidently rejected a windshield patch where Sol retained uncertainty. C05: Luna placed the neighboring adult head inside the box although it is above the top edge. C10: both over-withheld a coherent single-person box for incidental neighboring limbs/background.

Routing: Luna may perform bounded first-pass review and organize evidence. Dense same-category overlap, uncertain windshield/reflection, owner deduplication, geometry correction and all new owner admissions remain with Sol or lead. No automatic JSONL acceptance or splitting based on either model alone. An axis-aligned bbox may legitimately contain incidental parts of neighbors.

Actual monetary cost and end-to-end throughput were not measured. Ten targeted cases cannot support a universal accuracy or cost-effectiveness claim. See `lead-comparison.json` for bound inputs and computed results.
