---
name: ask-pro
description: Prepare self-contained, copy-ready prompts for the user to consult web GPT-pro on consequential decisions or difficult research and engineering reasoning. Use when explicitly requested or external reasoning could change the next step; not for routine lookup or automated model calls.
---

# Ask Pro

Prepare a consultation for the user to paste into the GPT-pro web UI; the user returns Pro’s answer for assessment. Pro may know the user’s broader context and access shared Notion, but cannot inspect this machine, repository, local artifacts, or Codex’s private conversation. Never invoke a model or submit the prompt automatically.

## When to use

Use or suggest this skill when difficult inference, mathematics, competing explanations, design tradeoffs, or consequential uncertainty could benefit from external reasoning. Do not require more local models, experiments, or a solved question first. Treat the user’s message as both a request and candidate ideas: preserve tentative ideas as hypotheses or branches, and let Pro reframe exploratory questions.

## Build a self-contained prompt

- State the objective, decision or open question, current bottleneck, and constraints that could change the answer. Define specialized terms and notation.
- Separate the user’s relevant original ideas (quote them faithfully and label them as the user’s) from Codex’s evidence-based synthesis. Preserve uncertainty; do not upgrade suggestions into facts.
- Extract decision-bearing substance from authoritative local evidence: relevant computations or code, assumptions, data and metric definitions, contrasts, results, failures, counterexamples, quantities, denominators, and limits. Include negative evidence. Paths, hashes, commits, and receipt IDs are provenance, not evidence; inline what Pro needs and never ask it to inspect local files or recall the private conversation.
- Distinguish observation, hypothesis, interpretation, and unknown. State what was tested, what was only proposed or mechanically checked, Codex’s current judgment, and the strongest alternative. Do not invent missing evidence.
- For mechanisms, show computations, assumptions, conditional consequences, and predictions beyond those used to construct the explanation. Compare the strongest alternative and name the cheapest discriminator. A toy model establishes possibility or derivation only; transfer to the real system needs evidence. Ask which causal links are supported versus assumed, including whether a representation is merely present or probe-readable versus used and updated during generation.
- Invite Pro to disagree, offer counterexamples or derivations, and say when evidence cannot decide. Seek a checkable conclusion or prioritized discriminator, not a generic survey or unranked experiment list. Use equations only when they clarify the reasoning.

Shared Notion is optional. If the reasoning relies on it, verify access and freshness and identify the page title, URL, section/version, and reading order. Keep the core question and findings in the prompt; inline needed facts when access cannot be verified. Do not claim local material is in Notion or create/update a page unless asked. Omit credentials and unrelated private data.

## Deliver

The normal handoff is response-only: provide exactly one ready-to-copy Markdown prompt in the user’s language, with no extra preface or duplicate format. This restriction applies to the final deliverable, not host-required progress updates. Do not create or save a local prompt unless the caller requests a file artifact. Organize the prompt for the question; it may include headings or tables, with no arbitrary word limit that drops necessary context. Remove local paths and unsupplied attachments before delivery and ensure the prompt plus any required, verified Notion reading is sufficient. Ask Pro to flag inaccessible evidence rather than infer it from filenames or memory.

Stop after preparing the consultation. Do not run research, begin experiments, or treat external advice as acceptance, authorization to execute, a change in research meaning, or permission to publish. When the user returns Pro’s answer, check decision-bearing claims against supplied evidence and locally where possible, then say what changes the next step.
