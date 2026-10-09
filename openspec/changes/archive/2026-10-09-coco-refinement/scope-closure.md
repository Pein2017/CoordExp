# Scope closure: COCO refinement (2026-10-09)

**Disposition:** Archive the bounded implementation evidence without claiming production adoption. The user expressly cancelled the planned Gate A train/val human acceptance, ROI inference and final replacement rollout as non-current future ideas. These are **withdrawn, not completed**.

## Retained delivered source

Keep the existing `src/coco_refinement/` code, local browser/SVG bbox editor, native Draft and Commit persistence, Focus Queue, materializer and selected max_len12000 norm/coord training publisher, plus relevant tests and receipts. This closure does not delete source, rewrite COCO images/annotations, publish a training pair or start/stop any service. Prior tasks 3.2 and 5.1a were verified with the 22/22 targeted test subset on Node 22 and Python, as recorded in `tasks.md`; that is not a production or operator UAT pass.

## Withdrawn future requirements

- **3.8:** operator-launched Gate A experience and explicit train/val human acceptance, not performed.
- **4.1–4.7:** all planned resident-model, AI ROI, profile endpoint, atomic inferred-Draft insertion, UI controls, fake-engine/browser tests and real-model inference qualification, not implemented/qualified by this change.
- **5.1, 5.2, 5.3, 5.4:** representative human-plus-model train/val exercise, ROI/operator handoff material, legacy disposition/cleanup, final replacement approval and deactivation, not completed or adopted. The separate implemented 5.1a transactional publisher is preserved.

The old `proposal.md`, `design.md`, all three delta specs and unchecked checklist entries remain archived only for source provenance. The unimplemented `coco-roi-assistance` contract is **not** promoted to current `openspec/specs/`. To avoid falsely declaring a user-accepted standalone workspace or mandatory legacy replacement, **skip spec sync for this entire historical change**; current executable behavior remains owned by code and verified test boundaries.

This archive neither approves `localhost:53662` as a production service nor authorizes switching off the old port 8080, touching the legacy Draft state or carrying out any new annotation/ROI work. Future adoption would need a separately authorized change with contemporary evidence and operator decision.
