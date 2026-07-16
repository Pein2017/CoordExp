# Delegated Agent Model Routing

This file is the human-readable routing policy for delegated CoordExp agents.
The executable role defaults remain in `.codex/agents/*.toml`; when the two
surfaces disagree, the tracked role profile is authoritative until the policy
and profile are deliberately reconciled.

`GPT` in a model-family name means Generative Pre-trained Transformer.

Historical research notes and session records may mention retired models as
provenance, but they are not valid routing guidance.

## Active Model Families

- `gpt-5.6-luna`: cost-efficient discovery and bounded execution work with a
  mechanical acceptance criterion.
- `gpt-5.6-sol`: scientific judgment, conclusion-critical implementation,
  causal diagnosis, and evidence-critical audit.

## Retired Model Families

Do not assign these families to new CoordExp agents:

- `gpt-5.6-terra`
- `gpt-5.5`

Their names may remain in immutable traces and historical provenance.

## Default Routing

| Work type | Default route | Required evidence gate |
|---|---|---|
| Repository scouting, artifact discovery, and preflight | `gpt-5.6-luna`, medium reasoning | Exact paths, symbols, or artifact handles |
| Small executable attestation probe | `gpt-5.6-luna`, medium reasoning | Command, exit status, and compact receipt |
| Thin deterministic implementation | `gpt-5.6-luna`, high reasoning | Narrow ownership plus deterministic tests or a real smoke |
| Bounded mechanical implementation after a local failure | `gpt-5.6-luna`, maximum reasoning | Explicit acceptance criterion; no unresolved scientific judgment |
| Research synthesis and contract audit | `gpt-5.6-sol`, medium reasoning | Evidence-linked verdict and claim boundary |
| Novel hook, cache, row-state, precision, or cross-layer implementation | `gpt-5.6-sol`, high reasoning | Real runtime smoke and conclusion-owning verification |
| Model-behavior diagnosis or cross-root causal tracing | `gpt-5.6-sol`, extra-high reasoning | Competing explanations and discriminating evidence |

`gpt-5.6-luna` with maximum reasoning remains a bounded implementation worker,
not a scientific or architecture judge. Current evidence supports this role
qualitatively; it does not constitute a controlled cost-quality benchmark.

## Escalation And Ownership

1. Assign one implementation owner to one semantic surface. Do not run parallel
   implementations of the same mechanism merely to create a model race.
2. Run the smallest real smoke as soon as the skeleton can exercise the
   conclusion-critical seam. Do not wait for a broad framework to be complete.
3. Give a local defect one focused follow-up using the existing agent context.
4. Escalate to `gpt-5.6-sol`, high reasoning, when the failure indicates a
   semantic misunderstanding, repeated stall, cache or hook ambiguity, runtime
   composition mismatch, or absence of an executable receipt. Do not issue
   duplicate blind retries to `gpt-5.6-luna`.
5. Reserve `gpt-5.6-sol`, extra-high reasoning, for unresolved Priority 0 or
   Priority 1 causal contradictions, not routine review.
6. Adjudicate disagreements from executed evidence, not model prestige,
   majority vote, prose length, or raw token expenditure.

Use minimal context inheritance: `none` or one recent turn for self-contained
discovery, one or two recent turns for bounded implementation and review, and
full history only for a task that genuinely requires complete-thread synthesis.

## Active Custom-Agent Assignments

- `repo_scout` and `probe_runner`: `gpt-5.6-luna`, medium reasoning.
- `contract_auditor` and `research_synthesizer`: `gpt-5.6-sol`, medium reasoning.
- `implementation_worker`: `gpt-5.6-sol`, high reasoning.
- `model_diagnostician` and `upstream_relation_tracer`: `gpt-5.6-sol`,
  extra-high reasoning.

Change these executable assignments only after repeated task evidence shows a
stable routing improvement; one anecdotal success or failure is insufficient.
