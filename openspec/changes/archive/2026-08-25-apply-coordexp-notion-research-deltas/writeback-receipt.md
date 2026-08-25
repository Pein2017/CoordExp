# Notion Writeback Receipt

Date: 2026-08-25 UTC

## Authority and source boundary

- Reviewed content router: `../extract-research-knowledge-from-codex-sessions/notion-update-plan.md`.
- Integrated evidence: `../extract-research-knowledge-from-codex-sessions/research-findings.md`, covering 6,278 session sources and 9,746 research-document paths.
- Current repository authority remains `/data/CoordExp` main except where the user explicitly selected `/data/CoordExp/.worktrees/CoordExp-swift` as the template reference.
- Raw transcripts, document mirrors, logs, images, checkpoints, and manifests stay local.

## Pre-write target inventory

All ten targets were fetched successfully on 2026-08-25 and retained their expected parent and stable ID.

| Owner | Page ID | Pre-write marker | Pre-write result |
| --- | --- | --- | --- |
| Global Research OS | `3c79d9ce-3f59-8148-96e7-c63c2a8e68ad` | `# Local knowledge projection` | PASS |
| Shared GPT-Web/Codex contract | `3c79d9ce-3f59-812a-977f-d25c95b7828d` | `# Conflict rule` | PASS |
| Serialization, Geometry & Ordering | `3c79d9ce-3f59-81c0-8a6c-dbac1a9f2254` | `supervised_response_text` | PASS |
| Image2299 | `3c79d9ce-3f59-81c0-91a6-cf798be94c96` | `Current frontier: HOLD` | PASS |
| Representation & Binding | `3c79d9ce-3f59-81ec-95f9-db7c27fe182b` | `# 2026-08-25 reconciliation` | PASS |
| Causal Evaluation | `3c79d9ce-3f59-8173-ba20-c5cade809d97` | `# Independent evidence descriptors` | PASS |
| Set Coverage | `3c79d9ce-3f59-814e-aeef-d9f755c10afe` | `all-HF shared trajectory-credit route` | PASS |
| Data Regimes | `3c79d9ce-3f59-81ff-83ef-d058775c28d9` | `Human13 N/K factorial is **planned only**` | PASS |
| Architecture & Explicit State | `3c79d9ce-3f59-810c-9fea-cc18e721270a` | `OwnerBridge V1 is **HOLD**` | PASS |
| Training & Optimization | `3c79d9ce-3f59-8186-838b-c775b98c787c` | `# 2026-08-25 reconciliation` | PASS |

## Terminal template adjudication

- Selected worktree: `/data/CoordExp/.worktrees/CoordExp-swift`, branch `coordexp-swift`, HEAD `22f2fc9e0db0931444981d9ced01af41e5cc843a` before the authorized explanatory comment.
- Main fixed point: `/data/CoordExp`, branch `main`, HEAD `b274d596ba3ce98b4f5bb64b0247a773637f5d9c`.
- Both render the canonical suffix as `<|im_end|>\n`.
- Local tokenizer receipt: `<|im_end|>` → `[151645]`; newline → `[198]`; combined suffix → `[151645, 198]`; PAD is distinct at `151643`.
- Supervision receipt: `<|im_end|>` is `eos_transition` and supervised; the following newline is `ignored_text` and masked. There is no separator between object rows.
- Decision: retain the canonical rendered newline for Qwen template fidelity; keep `<|im_end|>` as the final supervised token. This is intentional and does not change production behavior.

## Mutation verification

| Page ID | Applied delta | Distinctive post-write marker | Result |
| --- | --- | --- | --- |
| `3c79d9ce-3f59-8148-96e7-c63c2a8e68ad` | Added 6,278 + 9,746 source census, 16-finding router, and research-graph authority | `9,746 research-document paths` | PASS |
| `3c79d9ce-3f59-812a-977f-d25c95b7828d` | Added bootstrap false-positive, inherited-child false-negative, and triage boundary | `# Source-census and classifier boundary` | PASS |
| `3c79d9ce-3f59-81c0-8a6c-dbac1a9f2254` | Resolved canonical render/EOS-only supervision and added bounded online-permutation result | `final supervised token`; `online-permutation result` | PASS |
| `3c79d9ce-3f59-81c0-91a6-cf798be94c96` | Added near-policy dose response, GT-prefix non-decoder, natural insertion, and same-owner zero-debt boundary | `up-10`; `0/10`; `Current frontier: HOLD` | PASS |
| `3c79d9ce-3f59-81ec-95f9-db7c27fe182b` | Added visual designation/failed bridge controls and prerequisite-only mechanics | `proposal bridge`; `mechanics prerequisites` | PASS |
| `3c79d9ce-3f59-8173-ba20-c5cade809d97` | Added capability split, frozen FN cohorts, shared-prefix impossibility, and unresolved crossover | `114 supported native false negatives`; `needs-adjudication` | PASS |
| `3c79d9ce-3f59-814e-aeef-d9f755c10afe` | Added bagging boundary and separated failed cross-engine parity from later all-HF no-update mechanics | `22/1573`; `463 sampling`; `retired partial` | PASS |
| `3c79d9ce-3f59-81ff-83ef-d058775c28d9` | Added literal partial-execution status and standalone owner-credit reporting contract | `24 checked / 5 unchecked`; `H gain` | PASS |
| `3c79d9ce-3f59-810c-9fea-cc18e721270a` | Separated V1 runtime repair from V1 science and successor proposal | `W8 collective failure`; `fresh immutable scientific run` | PASS |
| `3c79d9ce-3f59-8186-838b-c775b98c787c` | Added HF/vLLM fail-closed semantics, launch bottleneck, validity gates, and bounded negatives | `use_dora=true`; `K=8 materialization`; `prefix-denoising objective` | PASS |

Post-write verification: 10/10 mutated pages refetched successfully; every distinctive marker was present and each pre-existing lifecycle/authority boundary checked above remained visible.

## Safety and residue

- Notion operations were limited to exact `update_content` replacements on the ten existing page IDs above. No page/database create, duplicate, move, delete, archive, attachment, or `allow_deleting_content` operation was used.
- No raw session, research-document mirror, log, image, checkpoint, or bulky artifact was imported. The local change directory is 28 KiB and contains only proposal/design/tasks/receipt metadata.
- Production code behavior was not changed. The only source edit is the separately authorized explanatory comment beside `IM_END_SUFFIX` in `/data/CoordExp/.worktrees/CoordExp-swift/src/templates/renderer.py`.
- `git diff --check` passed in both `/data/CoordExp` and the selected CoordExp-Swift worktree.
- `openspec validate apply-coordexp-notion-research-deltas --strict` passed with 10/11 tasks complete immediately before reconciling this literal validation task.
