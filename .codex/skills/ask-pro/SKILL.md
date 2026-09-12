---
name: ask-pro
description: "Prepare self-contained, copy-ready prompts for the user to consult web GPT-pro on consequential decisions, difficult reasoning or mathematical questions, and research or engineering bottlenecks. Use for explicit ask-pro requests or when external expert reasoning could materially change the next step; not routine status reporting or automated model calls."
---

# Ask Pro

Prepare an expert consultation, not an automated invocation. The user copies
the prompt into the GPT-pro web UI and brings its answer back. Pro has its own
memory of the user and can access the user's shared Notion, but cannot inspect
this machine, repository, local artifacts, or Codex's private conversation.

## When to consult

Use or proactively suggest this skill when a difficult inference, mathematical
issue, conflicting explanation, design tradeoff, or consequential uncertainty
would benefit from external reasoning. It is not a last-resort escalation:
do not require exhausting local models, running more experiments, or solving
the hard question before asking. Under the user's stated setup, consultation
has no marginal monetary cost; do not impose low-frequency rules or ration it.
Spend preparation effort on decision value and the user's attention instead.

Locate discoverable facts locally; send the unresolved reasoning problem to
Pro. Do not manufacture complexity or turn an ordinary lookup into a grand
research question. The consultation may challenge the current framing rather
than merely select between the options Codex already prefers.

Treat the user's message as both a request and a source of candidate ideas.
Abstract, tentative, or highly uncertain ideas that point toward a larger
direction should be carried into the consultation as explicitly labeled
hypotheses, mechanisms, or decision branches. Take them seriously without
upgrading them to facts, and adapt the framing to the state of the question:
ask a focused decision question when the decision is mature; ask Pro to help
restate the problem and map the key branches when it is still exploratory.

## Make the prompt independent of the local environment

- State the actual objective, decision to be made, current bottleneck, and
  constraints that could change the answer. Define notation and specialized
  terms. Include the relevant current state even if Pro may remember the user's
  broader interests; do not repeat a biography or assume it knows this run.
- Extract the smallest sufficient evidence from authoritative local sources:
  relevant code or pseudocode, equations, assumptions, data/metric definitions,
  experimental contrasts, results, failures, and counterexamples as needed.
  Preserve quantities such as units, denominators, dose, conditioning and
  evaluation scope when they affect interpretation. Include important negative
  results and limitations, not just evidence favoring the current view.
- Keep two input streams visibly separate: the user's original ideas, intuitions,
  and larger-direction suggestions, and Codex's evidence-backed synthesis.
  Preserve the user's uncertainty and intent while translating abstract ideas
  into candidate hypotheses, mechanisms, or branches that Pro can examine.
- Separate observations, hypotheses, interpretations, and unknowns. Explain
  what has actually been tested versus merely proposed or mechanically checked.
  Include Codex's current judgment and its strongest alternative without asking
  Pro to rubber-stamp either. Label material missing evidence; never invent it.
- A local path, commit, hash or receipt ID is provenance, not accessible
  evidence. Inline the necessary substance. Do not ask Pro to open local files,
  run local commands, inspect an attachment that was not supplied, or recover
  facts from "the previous Codex discussion."
- Shared Notion can carry detailed evidence; publishing there is not mandatory.
  When relying on it, verify the relevant page and include its title, URL,
  section/version and reading order. Keep the objective, core question and key
  findings in the prompt; the prompt plus explicitly identified Notion reading
  must contain the necessary reasoning context. Do not duplicate entire evidence
  tables merely to make the prompt standalone. If page access or freshness cannot
  be verified, inline the needed facts or state the gap. Do not claim local
  material is in Notion or create or update pages unless the user requested it.
- Remove credentials and unrelated private material. Include detailed evidence
  when it is needed for reasoning, not entire logs or a repository dump.

## Ask for a consequential reasoning result

Match the scope of the user's decision: a focused bottleneck needs a sharp
unresolved question; a program-level consultation needs the broader evidence
and competing directions, not only the latest experiment. Use related
subquestions where they help resolve that decision. Take advantage of Pro's
mathematical and complex-reasoning strength through the substance of the
problem, not flattery, role assignments, or instructions such as "you are a
world-class expert."

Use an open-ended, collegial voice: invite Pro to reinterpret the framing,
disagree with both the user and Codex, and say when the evidence cannot decide.
Always consider whether formalization through mathematics, statistics, causal
identification, optimization, or LLM neural-network dynamics could sharpen the
question. Invite a derivation, counterexample, or predictive consequence when
that lens has decision value; do not force formalism or invent equations when it
does not.

Useful question shapes include:

- Are the competing mechanisms identifiable from this evidence? Give a
  counterexample or the smallest intervention that separates them.
- Under these explicit assumptions, is the proposed objective or estimator
  valid? Derive the relevant result or show where it fails, and explain which
  observable consequence would distinguish those cases.
- Which hidden assumption would reverse this decision? Compare the strongest
  alternatives and recommend the next discriminating test, with its expected
  outcomes and the conclusion each would support.
- If the user's direction is still abstract, what is the most useful precise
  formulation of it? What competing interpretations should be kept alive, and
  what is the cheapest evidence that would distinguish them?

These are examples, not a mandatory checklist. Ask for checkable conclusions,
derivations, counterexamples or a prioritized decision as appropriate, rather
than a generic survey or a long unranked list of possible experiments. Make
room for "the evidence cannot decide" and specify what additional information
would then be useful. Do not force formalism where it adds no insight.

## Deliver and return

Deliver one ready-to-copy prompt in the user's language, usually as one clearly
delimited copyable block. Keep any preface outside it brief. A useful prompt
order is objective and decision state, the user's ideas and larger direction,
verified facts and negative results, Codex's current synthesis and strongest
alternative, unknowns, and the open questions for Pro; adapt this order when
the problem calls for it. Use headings or tables if they clarify the evidence;
impose no arbitrary word limit that would remove load-bearing context. Do not
add a persona or redundant background. Before delivery, mentally remove every
local path and unsupplied attachment: the prompt and any required, verified
Notion reading must still provide enough context to address the question.
Distinguish required reading from optional background, and ask Pro to flag
inaccessible evidence rather than infer its contents from filenames or memory.

Stop at the consultation handoff: do not call a model API, automate the web
submission, impersonate Pro's answer, or silently start a new experiment.
When the user returns Pro's response, assess it against the supplied evidence,
check decision-bearing claims locally where possible, and identify what changes
the next step. External advice is not automatic acceptance or authorization
to execute, change research meaning, or publish conclusions.
