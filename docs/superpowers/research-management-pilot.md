# Research Management Pilot

This note defines the first lightweight CoordExp Notion pilot.

The Notion project root is:

- `CoordExp`
  - `Experiments & Evidence`
  - `Compact Detection Sequence Pilot`
  - `Research Units`
  - `Decision Log`
  - `Methods & Protocols`
  - `Claims Ledger`

Notion URL roots:

- `CoordExp`: https://app.notion.com/p/3539d9ce3f59814fad41ce04ae1e42a9
- `Experiments & Evidence`: https://app.notion.com/p/3539d9ce3f59813dbff8f439549b92cc
- `Compact Detection Sequence Pilot`: https://app.notion.com/p/3539d9ce3f5981bba7acfc34ea12441a
- `Research Units`: https://app.notion.com/p/cb11525fc5964446aa0b69d411c6af4f
- `Decision Log`: https://app.notion.com/p/9556e75c8d1546e7b962d63bdd0ee923
- `Methods & Protocols`: https://app.notion.com/p/06fb33bb06e54e7bae84d09f674964d8
- `Claims Ledger`: https://app.notion.com/p/be257eccdf82488b8bff870e6b322e80

Linear roots:

- `PEI-1`: CoordExp Research Dashboard
- `PEI-2`: Compact Detection Sequence Ablation
- `PEI-3`: Gate 0 - Compact token/checkpoint contract
- `PEI-4`: Experiment gates - compact smoke, val200, and memo

## First-Experiment Rule

For the first compact detection sequence experiment, keep the workflow simple:

- `repo`: executable truth, exact configs, commands, tests, artifacts, and
  checked-in evidence.
- Notion: research memory, experiment interpretation, decision log, claim log,
  methods/protocols, and `progress/` migration teaching surface.
- Linear: overall process manager. Track phase boundaries, gate status,
  production-training launch, blockers, and final outcomes; do not mirror
  detailed super-power implementation checklists.
- `progress/`: final benchmark or diagnostic evidence after a real run exists.
- `docs/`: promoted stable current behavior only.
- OpenSpec: only when stable config, artifact, evaluation, compatibility, or
  behavior contracts change.
- super-power: branch-specific code implementation, tests, and smoke
  verification.

## Operating Boundary

Use Linear for the durable research process:

- Phase 1: training infrastructure merge.
- Phase 1.5: production training launch and monitoring.
- Phase 2: inference/eval integration after checkpoints exist.
- Phase 3: `val200`, benchmark note, and research interpretation.

Use super-power plans for the local engineering slice:

- files to edit,
- config contracts,
- tests to write,
- smoke command and artifacts,
- merge-readiness evidence.

Do not put file-by-file implementation checklists in Linear. Do not make a
super-power plan responsible for future research gates that require artifacts
from a later Linear phase.

## Active Decision: Token Roles Before Smoke

The compact sequence pilot must complete a role-separated token refactor before
the first GPU smoke run. The approved boundary is:

- `<|object_ref_start|>` and `<|box_start|>` are native Qwen structural markers.
- These marker rows may be included in trainable embedding/logit-row offsets.
- These marker rows are ordinary assistant CE targets only.
- These marker rows must not enter coord-softCE/W1, bbox-regression,
  coordinate diagnostics, or coordinate validity masks.
- `<|coord_0|>`..`<|coord_999|>` remain the only coordinate/regression-family
  token IDs.

Use `custom.trainable_token_rows` for the role-separated trainable-row contract.
Keep `custom.coord_offset` as a backward-compatible coordinate-only alias.
The code owner for this boundary is the new `src/tokens/` namespace:
`src/tokens/coord/` owns coordinate geometry tokens, `src/tokens/structural/`
owns compact structural marker identity, `src/tokens/roles.py` owns role sets
and loss-boundary separation, and `src/tokens/row_offsets.py` owns generic
trainable row adaptation. `src/coord_tokens/` remains compatibility-only during
the pilot.
Do not launch the compact smoke gate until the repo audit can report 1002
trainable row IDs but exactly 1000 coordinate loss IDs.

## Progress Migration Naming

In Notion, use clearer research-management names instead of copying the repo
folder labels directly:

| Repo term | Notion term |
| --- | --- |
| `progress/` | Experiments & Evidence |
| progress note | evidence card |
| benchmark note | result record |
| diagnostic note | mechanism record |
| direction note | research direction |

Each migrated evidence card should stay compact:

```text
Source path:
Optimized name:
Kind:
Status:
Scope:
One-sentence summary:
Main claim:
Evidence artifacts:
Related run/config/checkpoint:
Decision or next action:
Promotion state:
```

Do not copy large artifact contents into Notion. Link artifact paths and keep
durable summaries, manifests, configs, and metrics in the repo/output roots.

## Pilot Rule

For a research ablation such as compact detection sequence:

1. Start from the repo-local super-power spec/plan when implementation or
   verification details matter.
2. Use the Notion `Research Units` record as the canonical research-memory
   entry. The `Compact Detection Sequence Pilot` page is a supporting brief.
3. Use the Notion `Experiments & Evidence` page as the renamed/optimized
   migration surface for curated `progress/` records.
4. Use Linear `PEI-1` through `PEI-4` for the overall process and phase gates.
   The fine-grained implementation checklist belongs in the super-power plan.
5. Promote measured results into `progress/benchmarks/` only when a real run
   result exists.
6. Promote stable current behavior into `docs/` only after the experiment stops
   being a one-off dated result.
7. Create an OpenSpec delta only if the implementation changes stable contracts
   or compatibility expectations.

## Success Criteria

- The executable truth remains in the repo.
- The current `progress/` history has a readable Notion migration surface.
- The first experiment has a single, easy-to-update Notion control page.
- Notion improves scanning and review without replacing checked-in evidence.
- No agent-only hidden persistence layer is introduced.
