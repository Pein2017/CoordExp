# Human Review Candidate Ledger

Open `unmatched-prediction-review.html` directly in a modern browser. The page
is self-contained and does not require a server.

## Scope

- Review set: `human-review-candidate-ledger-20260716a`
- Policy: Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`)
- Ontology: Common Objects in Context 80-category ontology (`COCO-80`)
- Images: `12576`, `19432`, `9400`, `7816`, and `15254`
- Unmatched candidates: 119
- Accepted reference-ledger boxes: 81
- Image `15254` is intentionally retained as a zero-unmatched negative case.

## Review Semantics

- **Approve** means a real, reportable `COCO-80` physical entity supports the
  prediction. It does not imply that both category and geometry are correct.
- **Reject** means no reportable entity supports the prediction, or the visible
  entity is outside `COCO-80`.
- **Unknown** means the pixels are insufficient for a reliable decision.

For an approved prediction, select an accepted-ledger entity identifier or use
`Assign new human entity`, then record semantic and geometry status. Repeated
predictions of the same physical object must share one entity identifier.

Keyboard shortcuts outside text fields are `A`, `R`, and `U` for the three
verdicts and left/right arrow keys for navigation.

The first-pass view hides other model predictions and does not display arm,
seed, frequency, or score. After the first pass, use `Consolidation` to revisit
approved candidates and unify physical-entity identifiers.

The browser autosaves into `localStorage`, which is local to that browser and
file origin. Use `Export JSON` regularly. A completed export passes its
annotation gate only when every candidate has a verdict and every approval has
a same-image entity identifier plus semantic and geometry status.

## Frozen Sources

- Manifest Secure Hash Algorithm 256-bit digest:
  `ac188ea1c09526660a86710fb8c99f632e99cdc53b5816cc870cf67c81dd0dc7`
- Unmatched queue Secure Hash Algorithm 256-bit digest:
  `0bdfe3a1cbbf31d0953dd71d64c4cc08d1ae036d793de423ce1a12e9ade17e78`
- Accepted ledger Secure Hash Algorithm 256-bit digest:
  `52e9f21eb32f7c3793d1356125931cb4c2a1fa5a12647d0d437dec217668d8df`

Do not begin causal branch discovery from this cohort until the completed
review export has been frozen and hashed. Before trajectory genealogy begins,
the 119 post-merge candidate identifiers must also pass the unit's source-call,
raw-rollout, parsed-row, seed, and decode-identity join gate. This join gate
does not block human annotation.
