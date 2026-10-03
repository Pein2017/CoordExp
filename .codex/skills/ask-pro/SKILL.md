---
name: ask-pro
description: Prepare copy-ready prompts and identified shared Notion readings for the user to consult web GPT-pro on consequential decisions or difficult research and engineering reasoning. Use when explicitly requested or external reasoning could change the next step; not for routine lookup or automated model calls.
---

# Ask Pro

Prepare a consultation for the user to paste into the GPT-pro web UI; the user returns Pro’s answer for assessment. The user has confirmed web GPT-pro can access shared Notion; use that access without re-proving it for each consultation. This does not imply access to every page, this machine, repository, local artifacts, or Codex’s private conversation. Never invoke a model or submit the prompt automatically.

## When to use

Use or suggest this skill when difficult inference, mathematics, competing explanations, design tradeoffs, or consequential uncertainty could benefit from external reasoning. Do not require more local models, experiments, or a solved question first. Treat the user’s message as both a request and candidate ideas: preserve tentative ideas as hypotheses or branches, and let Pro reframe exploratory questions.

## Build the prompt and reading context

- State the objective, decision or open question, current bottleneck, and constraints that could change the answer. Define specialized terms and notation.
- Separate the user’s relevant original ideas (quote them faithfully and label them as the user’s) from Codex’s evidence-based synthesis. Preserve uncertainty; do not upgrade suggestions into facts.
- Supply decision-bearing substance from authoritative evidence: relevant computations or code, assumptions, data and metric definitions, contrasts, results, failures, counterexamples, quantities, denominators, and limits. Include negative evidence. The prompt and explicitly identified Notion readings may jointly supply this context; focus the prompt on the actual question, critical constraints, and material updates not yet on Notion instead of duplicating entire pages. Paths, hashes, commits, and receipt IDs are provenance, not evidence; inline needed local evidence absent from the readings and never ask Pro to inspect local files or recall the private conversation.
- For an ongoing investigation with prior results, consult its maintained catalog and current evidence owners for the closest supporting and counterexample results, not just recent updates. In the prompt or explicitly assigned Notion readings, identify what is already answered, what differs now, and the incremental question.
- Distinguish observation, hypothesis, interpretation, and unknown. State what was tested, what was only proposed or mechanically checked, Codex’s current judgment, and the strongest alternative. Do not invent missing evidence.
- For mechanisms, show computations, assumptions, conditional consequences, and predictions beyond those used to construct the explanation. Compare the strongest alternative and name the cheapest discriminator. A toy model establishes possibility or derivation only; transfer to the real system needs evidence. Ask which causal links are supported versus assumed, including whether a representation is merely present or probe-readable versus used and updated during generation.
- Invite Pro to disagree, offer counterexamples or derivations, and say when evidence cannot decide. Seek a checkable conclusion or prioritized discriminator, not a generic survey or unranked experiment list. Use equations only when they clarify the reasoning.

For each required Notion reading, identify the page title, URL, relevant section/date/version, and reading order. When a Notion tool is available, read current relevant content during preparation. If a particular page fails, is inaccessible, or is stale, identify the specific gap and inline only the needed missing evidence that is available; flag any unresolved gap or unverified freshness without inventing content. Do not default to a fully self-contained prompt or duplicate available page evidence. Do not claim local material is in Notion or create/update a page unless asked. Omit credentials and unrelated private data.

## Deliver

The normal handoff is response-only: provide exactly one ready-to-copy Markdown prompt in the user’s language, including any required Notion reading instructions, with no extra preface or duplicate format. This restriction applies to the final deliverable, not host-required progress updates. Do not create or save a local prompt unless the caller requests a file artifact. Organize the prompt for the question; it may include headings or tables, with no arbitrary word limit that drops necessary context. Remove local paths and unsupplied attachments before delivery and ensure the prompt plus identified required Notion readings supplies the needed context and evidence, with any unresolved gaps explicit. Ask Pro to flag inaccessible evidence rather than infer it from filenames or memory.

Stop after preparing the consultation. Do not run research, begin experiments, or treat external advice as acceptance, authorization to execute, a change in research meaning, or permission to publish. When the user returns Pro’s answer, check decision-bearing claims against supplied evidence and locally where possible. Check whether proposed experiments already exist and state what is new or which conditions differ, then say what changes the next step.
