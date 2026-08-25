## Context

See [proposal.md](proposal.md). The corpus contains thousands of large JSONL sessions and many duplicated/divergent research trees. Keyword-only inclusion would admit generic development noise; summary-only inclusion would miss negative results and authorization changes. The exploration therefore needs a cheap mechanical census followed by selective semantic reading and cross-source reconciliation.

## Goals / Non-Goals

**Goals:**

- Give every inspected source a coverage disposition, including explicit skip reasons.
- Recover decision-bearing research knowledge with exact provenance and evidence boundaries.
- Let Luna workers parallelize independent read surfaces while one package lead owns deduplication and acceptance.

**Non-Goals:**

- No raw transcript publication, hidden-reasoning extraction, generic development history, experiment execution, or direct Notion mutation in this wave.

## Decisions

### 1. Use a two-stage relevance classifier

Stage A reads cheap signals: session metadata/date, cwd/worktree, user turns, outer assistant summaries, named artifacts, research paths, experiment vocabulary, and task outcome. Stage B deep-reads only high/medium candidates and any borderline session whose outcome changed research evidence, methodology, claim validity, continuation, or authorization.

Include sessions about hypotheses, experiments, artifacts, metrics, model behavior, causal interpretation, data/eval semantics, negative results, route decisions, or research-governance defects that changed evidence validity. Exclude generic plugin/tool/UI/refactor/install sessions unless the session directly altered a research evidence path or a durable research decision.

Alternative: read every JSONL fully. Rejected because it would spend most effort on development traffic and produce an unauditable volume of repeated text.

### 2. Partition sessions by date and documents by authority surface

Three session packages own May–June, July, and August 1–25. A fourth package owns research documents across root/canonical/secondary worktrees. Each Luna L1 may create at most two Luna L2 workers for independent subranges; L1 integrates and verifies its own package. Package write roots are disjoint.

### 3. Require both a row manifest and a synthesis

Each package writes `manifest.tsv` for coverage accounting and `synthesis.md` for unique findings. Required manifest fields are source/session identity, date/cwd/worktree, relevance/disposition, topics, evidence surface, execution/lifecycle, current owner, duplicate key, confidence, and reason. Synthesis entries state question, observed surface, scientific disposition, technical disposition, decision impact, not-claimed boundary, exact sources, and likely Notion owner.

Alternative: accept agent prose only. Rejected because it cannot prove source coverage or support cross-package deduplication.

### 4. Integrate from current owners outward

Repository artifacts and current research units outrank session narration. Integration groups identical or substantively duplicate findings, preserves same-path divergence, separates planned/executed/invalid/retired work, and flags contradictions or missing authority as `needs-adjudication`. A session can contribute provenance without becoming a current owner.

### 5. Defer Notion writes

The exploration produces a reviewed `coverage.tsv`, `research-findings.md`, and `notion-update-plan.md`. Only a later integration wave may mutate existing Notion owners after duplicate and claim-boundary review.

## Risks / Trade-offs

- [Mechanical classifier misses research embedded in implementation sessions] → Admit borderline sessions when user intent, cwd, referenced research paths, or final decision affects research; sample the skip boundary.
- [Luna overstates scientific conclusions] → Require exact source handles, evidence surface, technical/scientific split, and not-claimed text; L1 verifies L2 rows and L0 performs cross-package acceptance.
- [Duplicate sessions inflate findings] → Use session/thread IDs, content/provenance keys, artifact paths, and semantic duplicate groups before synthesis.
- [Dirty worktrees make “current” ambiguous] → Record checkout, branch/HEAD, dirty/uncommitted state, and treat the current research owner separately from candidate evidence.
- [Corpus grows during the run] → Freeze a capture timestamp and source list; later files are a new delta.

## Migration Plan

1. Freeze session/document source lists and counts.
2. Run four independent Luna packages with disjoint output directories.
3. L0 integrates package manifests, samples skips and duplicates, and resolves only evidence-backed owner mappings.
4. Produce the Notion update plan without writing Notion.
5. After review, apply only `needs-summary`/`needs-adjudication` outcomes in a separate write wave.
