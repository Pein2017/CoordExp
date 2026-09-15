---
doc_id: docs.architecture.coordexp-swift-decisions
layer: docs
doc_type: architecture-proposal
status: proposal
domain: architecture
summary: Initial resolved design decisions for the CoordExp-swift rebuild worktree.
updated: 2026-06-29
---

# CoordExp-Swift Decisions

This note records resolved discussion outcomes for the `codex/coordexp-swift`
worktree. It is proposal-scoped: it is not current behavior authority, not a
stable OpenSpec contract, and not an implementation checklist by itself.

## Decision

CoordExp-swift is a from-scratch, production-quality CoordExp training
infrastructure project. It is personal-guided, agent-readable, and intended to
become production ready, not a "lite" or "native" side runner.

The first supported target is Qwen3-VL single-image supervised
vision-language training using Transformers/PyTorch as the reusable substrate.
The infrastructure must own the CoordExp-specific pipeline, packing layout,
token-wise supervision, loss computation, artifacts, and training readability.

Design tradeoffs use this priority order:

1. Accuracy and precision of training semantics, alignment, metrics, and
   artifacts.
2. Training and overall system efficiency.
3. Simplicity, avoiding over-design, duplicated concepts, and redundant
   machinery.
4. Scalability and future extension capability.

When two good-looking designs conflict, prefer the one that protects earlier
priorities. For example, a stricter boundary check wins over a simpler but
ambiguous path; a proven efficient packed path wins over a more general padded
abstraction; and future extension seams should stay as docs until they earn
code without weakening the first three priorities.

## Source Layout

The active source root for this worktree is `src/` directly. There is no extra
`src/coordexp/` package layer by design; by convention, `src/` represents the
new CoordExp implementation in this worktree.

The preferred initial topology is:

```text
src/
  __init__.py
  train.py
  trace_config.py  # support entry; create when config tracing is implemented
  common/
  config/
  data/
  templates/
  qwen/
  packing/
  supervision/
  losses/
  optim/
  training/
  rollouts/
  artifacts/
  metrics/
```

Do not create a top-level `src/cache/` package in the first implementation.
Cache remains a deferred compatibility seam until a concrete payload-specific
card proves that storage, fingerprinting, and replay behavior are needed.
The first implementation must not create an empty `src/cache/` namespace, cache
storage, or mandatory cache receipt.

`src/` should be importable directly as the worktree's package. The preferred
development entry shape for the new training infrastructure is:

```bash
python -m src.train --config ...
```

`eval.py`, `infer.py`, and `visualize.py` remain good future role-named entry
files, but they are not primary V1 promises for this training-infrastructure
rebuild. Current inference, scoring, evaluation, and proxy-bundle workflows
should continue to follow the existing CoordExp infer/eval workflow surfaces
until an explicit migration or replacement is approved.

Do not make `PYTHONPATH=src` the primary invocation style, because that would
turn `config`, `data`, `training`, and other internal packages into accidental
top-level imports.

Do not introduce `cli.py` in V1. The previous `src/__main__.py` dispatcher idea
is no longer assumed. Prefer professional role-named entry files. For V1, the
primary entry is `train.py`; `trace_config.py` is the approved support entry
for config inheritance/debugging when that capability is implemented. Keep
entry files thin: parse arguments, resolve config, and call package-owned
builders or helpers.

The old implementation should be moved out of the active import root before
the rebuild begins, most likely to:

```text
reference/legacy_src/
reference/README.md
```

That move has not happened yet. It should be done deliberately as the first
approved implementation step, before building the new `src/`, preserving the
old source intact as local reference rather than active infrastructure.
`reference/README.md` should state that `reference/legacy_src/` is read-only
historical/reference code, not active implementation. Do not gradually replace
old files in place, do not split legacy files into topic folders, and do not
delete the old source during the rebuild.

Do not create old-source import shims. The new `src/` must not import from
`reference/legacy_src/`, should not keep old entrypoints alive through
compatibility aliases, and should not use path hacks to keep the legacy source
importable. The legacy source is for reading, not execution.

The initial new `src/` skeleton should contain only approved package
directories and minimal `__init__.py` package markers. Do not add broad exports,
TODO classes/functions, or abstract base classes before the corresponding
blueprint card is approved. The top-level `src/__init__.py` should not define a
manual `__version__` in V1; resolved configs, checkpoint metadata, and run
artifacts are the concrete interpretation handles.

Implementation approval should live in one compact blueprint document:

```text
docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md
```

The blueprint owns module cards, vertical-slice cards, and approval status. It
is the working design surface for humans and agents during implementation. This
decision log records resolved principles; the blueprint records the concrete
module/class/functionality approval queue.

## V1 Austerity Rule

A concept may be documented as an invariant without becoming a public class,
config knob, registry, artifact family, or extension point. V1 should preserve
hard correctness boundaries while keeping implementation machinery small.

Do not mirror this document's full design vocabulary into code. Prefer plain
functions, small dataclasses, named JSON receipts, and direct module ownership
over abstract base classes, plugin systems, broad factories, framework-style
registries, report frameworks, or placeholder future modules. Add abstraction
only when it removes real complexity in the first vertical implementation.

This rule is specifically meant to prevent CoordExp-swift from becoming a
smaller MS-Swift-shaped framework. The first implementation should make the
pipeline readable end to end, not maximize generality. Future rollout, cache,
hidden-state loss, inference, and multi-modal extensions may reserve concepts in
docs, but they should not appear as empty code surfaces until an approved
implementation card earns them.

## V1 Scope Freeze And Review Gate

The V1 architecture is now treated as mostly converged. New V1 decisions should
be added only when they unblock implementation, fix an internal contradiction,
answer a cross-review finding, or resolve a user-approved research-meaning fork.
Do not keep expanding the first version proactively.

Future features stay as reserved notes, not placeholder modules. Hidden-state
losses, rollout supervision, visual-row capture, persistent caches, richer
inference/eval entries, and other next-version surfaces should remain documented
as readiness seams until a payload-specific implementation card earns code.

Before implementation starts, run a read-only cross-agent review over
`DECISIONS.md` and `BLUEPRINT.md`. Reviewers should challenge contradictions,
missing boundaries, overdesign, under-specified contracts, and implementation
risks without editing the docs directly. Accepted findings should be triaged and
patched in this proposal directory before the user approves implementation.

Implementation proceeds module-card by module-card with explicit user approval
for important modules, classes, cross-module data types, config schemas, receipt
schemas, entrypoints, and public flow-defining functions. The first milestone
remains the vertical smoke, not broad module completeness.

During this review stage, a blueprint card marked `approved` means the proposal
text is accepted as a design target. It is not standalone coding authorization.
Implementation remains blocked until review findings are triaged and the user
explicitly approves the module/card for implementation. Reviewer timeout or
missing reviewer output is unresolved, not approval.

## Scope

Initial implementation scope:

- Supervised teacher-forced training first.
- Qwen3-VL only.
- Single image only.
- Packing is required as the standard training path.
- Unpacked mode may exist for debugging and parity checks.
- HF inference/runtime behavior may be used as reference where it already
  exists, but training should not be coupled to inference internals.
- No video, no multi-image, no vLLM, no rollout-training implementation in the
  first milestone.

Public package surfaces should be narrow. Each package may expose a small
approved public API and keep implementation helpers private until repeated
reuse earns promotion. Agent-readable code means stable names, local contracts,
and traceable flow, not exporting every helper or letting callers import file
internals casually.
V1 package `__init__.py` files are minimal markers. Do not add broad re-exports
or `__all__` lists until a package's public API has proven itself and the
blueprint approves those names.

Use a small typed contract-error surface in `src/common/errors.py` rather than
a broad exception hierarchy or unstructured `ValueError`s everywhere. Approved
initial errors are:

```text
CoordExpError
ConfigContractError
DataContractError
TemplateContractError
EncodingContractError
PackingContractError
QwenForwardContractError
LossContractError
RuntimeContractError
```

Errors should be precise enough for humans and agents to locate the violated
contract, but they should not become a complex control-flow mechanism. Each
contract error should carry simple serializable fields: `code`, `message`,
`context`, and optional `cause`. Context is a small JSON-like dictionary,
formatted into readable exception text and suitable for bounded debug reports.
Avoid a rich dataclass or Pydantic model hierarchy per error kind in V1.

## Supervised Packed Sequence Semantics

Supervised training should not use a conceptual batch for standard forward
passes. The conceptual unit is a packed single sequence per rank/step.

The default semantics are packed isolated segments:

- multiple encoded examples may be concatenated into one physical sequence
  under `global_max_length`;
- segment boundaries must remain explicit through the packed layout;
- attention, position ids, supervision placement, and loss accounting must
  preserve per-example isolation;
- later examples in the packed sequence must not semantically attend to earlier
  examples unless a separate continuous-stream research recipe is explicitly
  approved.

For Transformers compatibility, tensors may still carry a physical leading
dimension of `1`, for example `[1, L]`. That leading dimension is an adapter
convention, not a research-level batch. If GPU memory is available, the normal
supervised-training response is to increase `global_max_length`, not to pad
multiple independent sequences into one rank.

Use "batch" only for rollout or decoding contexts where multiple independent
generation requests are intentionally active at once. Supervised training
terminology should prefer:

- `EncodedExample`
- `PackPlan`
- `PackedSequence`
- `PackedLayout`
- `TokenSequence`
- `MicroStep`
- `planned_step_id`
- `LossBundle`

Use `Example` as the canonical name for one stable semantic data instance as it
moves through transformations. The standard chain is:

```text
RawExample -> RenderedExample -> EncodedExample -> PackedSequence
```

Use `example_id` and `examples` in code, configs, and artifacts. Reserve
`sample` for stochastic selection, informal inspection wording, or generated
sampling contexts; do not use `RawSample`, `RenderedSample`, or `EncodedSample`
as core class names.

## Position Terminology

Use `position` for sequence locations. Reserve `coordinate` for CoordExp
geometry concepts such as bbox coordinate values, coordinate bins, coordinate
tokens, and coordinate-token objectives.

Canonical position terms:

- `logical_position`: a token location inside one un-packed logical example.
- `physical_position`: a token location inside the packed physical sequence.
- `target_position`: the physical position of the token being supervised.
- `logits_position`: the model-output row used to predict the target token.

For standard causal language-model losses:

```text
logits_position = target_position - 1
```

`target_position == 0` is invalid for standard causal token prediction unless
a special non-causal objective explicitly defines different semantics.

## Module Responsibility Decisions

The first implementation should use these module responsibilities:

- `data/` owns JSONL loading, image path resolution, geometry validation, and
  logical examples. It does not tokenize, pack, or compute loss.
- `templates/` owns pure text/message rendering and local semantic spans. It
  does not call Transformers, process images, build packed sequences, or
  compute loss.
- `qwen/` owns all Qwen3-VL-specific interaction with Transformers and HF
  processor/model behavior: `AutoProcessor`, `Qwen3VLForConditionalGeneration`,
  `pixel_values`, `image_grid_thw`, position ids, MRoPE helpers, and local
  wrappers or monkey patches when necessary.
- `packing/` owns pack planning, packed layouts, packed sequence tensor
  assembly, Qwen position helper calls, sidecar position placement, and
  packing validation.
- `supervision/` owns token-level supervision concepts, including `TokenAtom`,
  `TokenSpan`, `TokenSequence`, target-position concepts, roles, provenance,
  weights, and strict validation.
- `losses/` owns objective math, loss terms, reducers, denominator semantics,
  and loss-runner output. It consumes validated `ModelOutputs` and supervision,
  not datasets, templates, tokenizers, or raw YAML.
- `optim/` owns optimizer/scheduler construction, parameter grouping,
  trainable-surface validation, and optimizer-group receipts.
- `training/` owns the loop, optimizer/scheduler orchestration and stepping,
  backward pass, checkpoint coordination, and calls into the loss runner. It
  does not own objective semantics or parameter-group taxonomy.
- `artifacts/` exists from V1 and owns run traces, debug artifacts, and run
  metadata.
- `metrics/` exists from V1 and owns typed metric/event records plus
  fundamental global metrics such as CE loss and token-level top-1/top-5
  accuracy.
- `rollouts/` is reserved in the architecture for future rollout-derived
  supervision, but should not get implementation code until a `RolloutTrainer`
  approval card exists.
- Avoid `utils/` initially. Add a shared helper module only after repeated
  cross-cutting code earns a deeper interface.

Transformers is the reusable model substrate. CoordExp-swift should freely
import from Transformers and build local adapters, wrappers, subclasses, or
runtime monkey patches when needed. Direct edits to installed upstream
Transformers files remain outside the default plan; if that becomes necessary,
it requires an explicit decision because it changes maintenance and provenance.

## Training And Loss Principles

The model does not own the training loss. CoordExp-swift should call the model
for logits and declared model outputs, then compute all losses in repo-owned
loss modules.

The infrastructure must support flexible token-wise and site-wise research
losses, not merely standard assistant-label cross entropy.

`TokenSequence` is the canonical supervision view. Inside an `EncodedExample`,
it uses local positions. Inside a `PackedSequence`, it uses packed physical
positions. A `TokenAtom` points to one token that should be predicted, not
directly to the logits row. The loss context owns the causal shift from
`target_position` to `logits_position`.

This is the V1 "supervision ledger" abstraction: the ledger role is concrete
`TokenSequence` plus `TokenAtom`/`TokenSpan`, not a second parallel class named
`SupervisionLedger`. Packing may remap that ledger into physical coordinates,
but it must not invent semantic supervision, infer token meaning from position,
or duplicate dense-label semantics. If a future implementation needs a separate
materialized `SupervisionLedger` object for performance or serialization, it
must remain a compiled view of `TokenSequence`, not an independent source of
truth.

Dense `labels` tensors are derived compatibility and debugging artifacts, not
the canonical source of truth for research losses. They may be materialized for
standard CE parity checks, trace dumps, or upstream adapter compatibility, but
objective modules should consume validated `TokenSequence` records. A dense
label tensor must be reproducible from the active `TokenSequence` and
tokenizer/model vocabulary contract; it must not carry independent loss
semantics that are absent from supervision records.

The V1 supervision hierarchy is intentionally small:

```text
TokenAtom
  -> TokenSpan
    -> TokenSequence
```

`TokenAtom` owns one supervised target position and its `TokenTarget`.
`TokenTarget` is a compact token-level record that may represent a hard token
id, multi-positive token ids, or sparse soft token weights. Use this name
instead of broader "distribution" terminology unless a future non-token target
earns a separate abstraction.

`TokenSpan` is contiguous: `[start_position, end_position)` over physical
target positions. It contains related atoms such as assistant text, description
text, bbox fields, coordinate-token runs, object entries, separators, and stop
tokens. Non-contiguous grouping is outside V1.

Roles live at token level and span level: `TokenAtom.role` supports
token-level loss and metric slicing, while `TokenSpan.span_kind` supports human
inspection and span-level summaries. `TokenSequence` contains spans for one
packed sequence and provides validation and lookup helpers.

Do not add a first-class `SupervisionGroup` or broader ontology in V1. If a
future objective needs non-contiguous grouping that cannot be represented as a
span, add that abstraction only when the need is concrete.

Loss composition direction:

- `TokenSequence` should contain one canonical `TokenAtom` per supervised
  `target_position`.
- Multiple `LossTerm`s may consume the same atom.
- `BaseTokenCE` is one protected default stabilizer over all `TokenAtom`s.
- `TokenTypeGateLoss` is also a protected default stabilizer over every
  supervised `TokenAtom` with a resolved target token type.
- Non-default auxiliary loss terms select subsets of the same atoms or spans;
  they should not duplicate supervision records just to apply another objective.

Loss continuity note: the V1 protected baseline is intentionally smaller than
the richest prior CoordExp production objective. Old production-style
coordinate training included IoU/CIoU-aware coordinate soft-CE variants and
object/role/image-balanced reductions. V1 does not claim those objectives are
implemented merely because `BaseTokenCE` and `TokenTypeGateLoss` work. The
important continuity guarantee is architectural: `TokenAtom`, `TokenSpan`,
`LossContext`, and `LossTerm` must make those prior objective families
implementable later as explicit auxiliary terms with their own denominator,
target-support, and metric cards. First-smoke convergence therefore proves the
transparent training path, not production-quality detection parity with the old
objective stack.

The efficient V1 loss abstraction is:

```text
TokenSequence
  -> semantic source of supervised target positions
LossContext
  -> compiled tensor view for one train step
LossRunner
  -> aggregate executor and validator
BaseTokenCE
  -> built-in stabilizer term
TokenTypeGateLoss
  -> built-in stabilizer term for token-type legality
LossTerm
  -> small protocol for auxiliary objectives
LossBundle
  -> total loss, per-term losses, metrics, diagnostics
```

`BaseTokenCE` lives in `losses/base_ce.py` as a built-in term-like
implementation consumed by `LossRunner`. Do not use the Transformers model's
built-in loss for production training. CoordExp-swift disables model-side loss
and computes CE through its own loss path so dense labels never become the
canonical supervision representation.

The default stabilizer surface is not CE-only. `TokenTypeGateLoss` is also a
default protected objective with the same priority as `BaseTokenCE`,
because it encodes basic token legality for the research task rather than an
optional auxiliary experiment. For example, a coordinate-token target position
should strongly favor coordinate-token vocabulary groups and penalize
incompatible token types. This is the CoordExp-swift equivalent of a
schema-aware contrastive stabilizer over token types. The V1 implementation is
per-position allowed-token-type gating: each supervised target position has an
expected token type, and the loss penalizes probability mass assigned outside
that type's allowed vocabulary group. A full bidirectional contrastive matrix
over all positions and all token-type groups is a future extension, not the V1
default.

`TokenTypeGateLoss` requires explicit vocabulary-group resolution from the
tokenizer and template schema. V1 supports four canonical target token types:

- `desc_text`: free-text description tokens;
- `schema`: the four structural wrapper tokens
  `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, and
  `<|box_end|>`;
- `coordinate`: `<|coord_0|>` through `<|coord_999|>`;
- `eos`: `<|im_end|>`.

These target groups, plus blocked non-target groups such as Qwen special,
chat/control, image/video/pad, tool/FIM/repo, and other reserved tokens, must be
represented as typed vocab groups rather than ad hoc integer ranges hidden
inside a loss function. `BaseTokenCE` remains full-vocabulary CE; gate loss uses
the resolved groups to express token-family legality.

Every supervised `TokenAtom` should carry a closed V1 `token_type`. `TokenSpan`
may provide the semantic assignment helper for a contiguous span, but the
compiled truth consumed by `LossContext` is per atom. Missing or unknown token
types should fail fast. V1 token types are closed to `desc_text`, `schema`,
`coordinate`, and `eos` unless a new type is deliberately approved.

The template/rendering layer is the source of intended token-type semantics.
It emits semantic spans over the assistant target with the intended closed
token type. Qwen encoding tokenizes the rendered target, aligns rendered spans
to token spans, expands them into `TokenAtom`s, and validates that each target
token id belongs to its declared token-type group. The loss layer must not
infer token type from token id as its primary source of truth, and packing must
not assign token types during placement.

V1 loss-bearing assistant tokens are the assistant answer content plus the
terminal `<|im_end|>` transition marker. The rendered assistant response still
follows the Qwen/MS-Swift string convention `<|im_end|>\n`, but the trailing
newline is `ignored_text`, not a supervised target. Prompt, system, user,
chat-prefix, and other control/template tokens are not supervised. `desc_text`
covers free-text assistant content outside structural spans; it does not cover
prompt text and does not act as a fallback bucket for control tokens.

The causal supervision boundary is open at the assistant-start boundary and
closed at the terminal `<|im_end|>` transition token. In physical token
coordinates, the loss-bearing target-position interval is:

```text
(assistant_start_boundary_position, im_end_token_position]
```

`<|im_start|>` itself is never a supervised target. The first supervised target
must be exactly the first assistant-target token after the assistant start
boundary. The rendered suffix follows the MS-Swift/Qwen convention
`<|im_end|>\n`, which local Qwen3-VL 2B/4B tokenizers encode as ids
`[151645, 198]`, i.e. `<|im_end|>` plus a trailing newline token. The
`<|im_end|>` token is supervised as `eos`; the trailing newline token is
explicitly ignored. If a concrete Qwen chat template inserts assistant
role/header tokens after `<|im_start|>`, those tokens remain boundary/control
tokens unless the renderer explicitly includes them in
`supervised_response_text`; the V1 default is to supervise assistant answer
content, not role-prefix text. Encoding must validate this boundary against the
actual tokenized model input so off-by-one masking cannot silently shift
supervision onto `<|im_start|>`, role/header tokens, the ignored newline, or
tokens after the terminal suffix.

Span offsets are relative to `supervised_response_text`, not the full rendered
chat text and not the raw JSON fields. `supervised_response_text` is the exact
assistant response string used for alignment: assistant answer content plus one
explicit terminal Qwen suffix `<|im_end|>\n`. The assistant message content sent
to `Qwen3VLProcessor.apply_chat_template` must exclude that terminal suffix when
the processor is responsible for inserting it. In that normal path,
`supervised_response_text` carries a synthetic suffix for span/loss alignment,
and `qwen/encoding` maps it onto the single processor-inserted suffix in the
full tokenized chat text. If a future renderer owns the complete Qwen chat
string directly, it must still prove that exactly one assistant terminal
`<|im_end|>\n` exists. Duplicate assistant terminators are a template/encoding
contract error before training.
The `<|im_end|>` token inside the suffix remains the EOS transition marker; the
following newline is ordinary language/template habit and is represented as
`ignored_text`.
Keep the token type name `eos` for the loss-bearing `<|im_end|>` marker. The
trailing suffix newline is not folded into the `eos` vocab group, not typed as
`desc_text`, and not supervised by CE or `TokenTypeGateLoss`.

Span offsets use Python string half-open intervals `[start, end)` over
`supervised_response_text`, with exact substring validation. Offsets are not
byte offsets. Valid spans satisfy `0 <= start < end <=
len(supervised_response_text)`. Adjacent spans are allowed. Properly nested
parent spans such as `assistant_content` or `object` are allowed for provenance
and grouping; crossing spans are invalid in V1. Leaf supervised spans assign the
token type used by `TokenAtom`; parent spans must not compete with child leaves
for token type assignment. Every loss-bearing character must be covered by
exactly one token-type-bearing leaf span. Special-token leaf spans, including
schema wrappers and `<|im_end|>`, must cover the whole special-token literal
rather than splitting them across spans. The terminal suffix must be represented
without ambiguity: `<|im_end|>` is covered by the rendered `eos_transition` leaf
span and maps to the supervised token type `eos`; the immediately following
newline must be deliberately covered by `ignored_text` rather than left
implicit. Loss-bearing characters that are not covered by a typed supervised
leaf span must either be deliberately assigned to `desc_text` by the renderer or
fail validation; they should not become untyped atoms later.

`RenderedExample` should carry `messages`, `prompt_text`,
`supervised_response_text`, and typed spans over `supervised_response_text`.
Qwen encoding is responsible for mapping the supervised suffix into the full
tokenized model input. Qwen assistant role-prefix or boundary tokens may be
present in the full model input if the chat template requires them, but they
are not part of `supervised_response_text` and are not supervised.

Inside the supervised assistant target, only approved target specials are
allowed: the four schema wrapper tokens, coordinate tokens, and terminal
`<|im_end|>`. The trailing suffix newline is not a supervised target. Any other
Qwen special token in the supervised target, such as
vision/image/video/pad/chat-start/quad controls, should fail unless a new token
type is deliberately approved.

Prompt image placeholders never create `TokenAtom`s. They are input-side visual
alignment tokens and participate in Qwen placeholder/grid validation, but they
are not assistant-target supervision for CE or gate losses.

The `desc_text` allowed-vocabulary group is the complement of reserved
structure and control groups: all tokenizer ids except schema-wrapper tokens,
coordinate tokens, image/control tokens, and eos-like stop tokens. Do not use a
curated natural-language allowlist in V1. The goal is to prevent structure
tokens from leaking into free text while preserving normal language
flexibility.

The reserved side of this complement must be resolved accurately for Qwen3-VL
from tokenizer and processor state, not guessed from a small hand-written list.
It should include tokenizer special tokens, Qwen chat/control tokens,
vision/image/video/pad tokens known to the processor, and any Qwen3-VL
additional special tokens that are not deliberately assigned to a supervised
type. Local Qwen3-VL 2B/4B tokenizers expose additional specials such as
`<|im_start|>`, `<|im_end|>`, the four object/box wrappers,
`<|quad_start|>`, `<|quad_end|>`, `<|vision_start|>`,
`<|vision_end|>`, `<|vision_pad|>`, `<|image_pad|>`, and
`<|video_pad|>`; the resolver must report the effective set for the loaded
model/tokenizer rather than assuming that this list is complete forever.
The resolver must also exclude control-shaped added tokens that may not be
marked as tokenizer specials, including Qwen tool-call/tool-response tags,
FIM/repo/file tags, and think tags such as `<tool_call>`, `</tool_call>`,
`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>`,
`<|repo_name|>`, `<|file_sep|>`, `<tool_response>`, `</tool_response>`,
`<think>`, and `</think>` when present.

The literal sentinel token `<|coord_*|>`, when present in a CoordExp-mutated
tokenizer, is reserved metadata/control. It is not one of the trainable
coordinate target tokens, is not allowed in supervised assistant output in V1,
and must be excluded from `desc_text`, the `coordinate` gate group, and
special-token embedding trainable groups unless a future token type explicitly
approves it.

If a target token id is outside the allowed vocabulary group for its declared
`token_type`, fail during supervision or encoding validation. Do not let CE
train the token while the gate penalizes it, and do not auto-reassign
`token_type` from the target id. A mismatch here is a template or supervision
bug.

The validation order is: resolve vocab groups, render typed spans, tokenize,
align spans to token positions, create `TokenAtom`s, validate target ids against
token types, then pack. This preserves provenance and catches template or
tokenizer mismatches before the loss path.

Before real training, require a tiny MS-Swift/Qwen shifted-loss-consumption
parity fixture around the assistant header, first assistant content token,
`<|im_end|>`, the trailing suffix newline, and image placeholders. The fixture
must compare more than token/input/label triples. For each sentinel supervised
site, it should assert the dense `input_ids[target_position]`, dense
`labels[target_position]`, `TokenAtom.target_position`,
`LossContext.logits_position`, `primary_token_id`, `segment_id`, and the exact
BaseTokenCE logits row consumed by the loss. It should also assert that
full-vocabulary logits are consumed for each supervised row. When compact
supervised rows are requested, the fixture should assert the returned time axis
matches the explicit `logits_to_keep` physical-position map; when compacting is
absent, `outputs.logits.shape[:2] == input_ids.shape` and `logits_to_keep` is
absent or `0`. The fixture exists to
catch template/masking/off-by-one mistakes; it is not a routine training metric
and should not introduce an MS-Swift-style token-loss monitor.

The same parity fixture must include post-expansion visual placeholder evidence:
raw `<|vision_start|>` and `<|vision_end|>` positions when present, the expanded
`<|image_pad|>` count and physical range, `labels == -100` across image
placeholder positions, no `TokenAtom` overlap with image placeholder positions,
the `image_grid_thw` used by Qwen, and the grid-derived expected image-token
count. This proves that prompt-side image placeholders are ignored supervision
regions while still occupying real physical positions in the Qwen forward row.

The permanent smoke fixture should intentionally cover the whole protected
default loss path: at least one `desc_text` span, all four `schema` wrapper
tokens, `coordinate` tokens, and terminal `<|im_end|>\n` suffix.

`TokenTypeGateLoss` is additive with `BaseTokenCE`, not a replacement. CE
selects the exact next token; the gate loss shapes probability mass into the
valid token family. The gate applies to every supervised atom whose closed token
type is known, which should be every V1 `TokenAtom`.

The V1 gate math is:

```text
loss = -log(sum_softmax_prob(allowed_vocab_group))
     = logsumexp(all_logits) - logsumexp(allowed_group_logits)
```

Compute this from fp32 selected logits in log-space for numerical stability.
The reducer still follows the segment-balanced reduction rule used by other
token-wise losses.

This is group-mass CE, not another full-vocabulary exact-token CE. The numerator
is the log-sum probability mass of the target token-type group; the denominator
is the full selected-logit vocabulary mass, including blocked Qwen special and
control groups. Blocked groups are therefore always-negative probability mass
for every supervised atom. They should also be reported separately so a run can
distinguish "wrong supervised family" from "leaking into reserved Qwen/control
tokens."

Loss weights for protected default stabilizers must be explicit in every
runnable config, either directly or through inherited named profiles. In
particular, resolved configs must show `base_ce.weight` and
`token_type_gate.weight`. Do not hide these values as code defaults. The first
smoke profile should use `token_type_gate.weight: 0.1`, treating the gate as a
stabilizer that should not numerically dominate exact-token CE.

The V1 loss config shape should be flat and explicit under `losses`, not a
class list and not grouped by an extra protection-level hierarchy:

```yaml
losses:
  base_ce:
    weight: 1.0
  token_type_gate:
    weight: 0.1
  auxiliary: []
```

Protected-loss disabling syntax should be visually heavy and reason-bearing:

```yaml
losses:
  token_type_gate:
    disable:
      reason: "..."
```

Do not use `enabled: false` for protected default stabilizers.

`TokenTypeGateLoss` should report aggregate gate loss plus per-type diagnostics
for `desc_text`, `schema`, `coordinate`, and `eos`. At minimum, report per-type
loss, selected counts, target-type probability mass, off-type probability mass,
blocked-special probability mass, and top-1 type accuracy. Do not emit full
per-token gate diagnostics every step by default.

Per-type gate metrics should use simple scalar names such as:

```text
loss/token_type_gate/by_type/coordinate
mass_target/token_type_gate/by_type/coordinate
mass_blocked_special/token_type_gate/by_type/coordinate
acc_top1_type/token_type_gate/by_type/coordinate
```

Use analogous names for `desc_text`, `schema`, and `eos`.

Resolved token-type vocabulary groups should be written to
`run_dir/reports/token_type_vocab.json` during run setup and linked from
the run manifest. This receipt is the audit artifact for tokenizer-dependent
vocab ids, including group names, token strings where finite and meaningful,
token ids, blocked Qwen/control groups, processor-reported visual tokens, and
counts. The resolved config artifacts alone are not enough because these ids
depend on the loaded tokenizer.

For protected gate setup, a runnable template/run must provide eligible atoms
for the expected target types `desc_text`, `schema`, `coordinate`, and `eos`.
If an expected protected type has zero eligible atoms, fail unless the type was
explicitly declared optional by a future approved config surface. This is a
run/template contract, not a reason to fail every individual pack that naturally
contains no atom of a rare optional type.

Any protected-loss `disable.reason` override must be recorded in the run
manifest so later readers can distinguish an intentional ablation from an
accidental missing stabilizer.

Loss math should upcast numerically sensitive tensors to `float32`, regardless
of whether the forward pass runs in `bf16`, `fp16`, or `fp32`. This mirrors the
local Transformers causal-LM loss path, which casts logits to float before CE
to avoid precision issues. In CoordExp-swift, the loss path should upcast
logits and other loss inputs at the point of computation, preferably after
selecting the needed logits positions or vocabulary slices to avoid unnecessary
memory pressure. Do not require the Qwen forward wrapper to materialize all
model outputs in fp32.

`BaseTokenCE` must compute cross entropy from fp32 logits. Future auxiliary
loss terms that use logits, probabilities, log-probabilities, hidden-state
similarities, distances, or other numerically sensitive tensors should also
perform their loss math in fp32 unless a term-specific approval card justifies a
different precision policy. Metrics such as top-k accuracy may use logits
without fp32 upcast when the operation is order-only and does not affect the
training objective.

The first implementation wave is logits-loss-only. Hidden-state losses are a
mandatory next-version capability, not a discarded idea, but they require an
approved wrapper-owned capture hook and a real Qwen smoke before becoming
runnable. V1 should preserve future-compatible shapes in the loss interface,
such as `required_inputs` and selected-hidden-state helper names, while
rejecting hidden-state-dependent loss configs until that hook is implemented.

Fp32 loss math is a hard V1 contract, not a config knob. Do not add
`losses.math_dtype` or an equivalent override in V1. Loss terms should not
consume raw bf16/fp16 logits or hidden states for objective math by default.
Any lower-precision loss math requires a term-specific approval card.

The fp32 upcast should be enforced through `LossContext` helper methods, for
example `context.select_logits_fp32(...)` and future selected-hidden-state
helpers. `BaseTokenCE` should select supervised logits positions first and then
upcast the selected `[num_atoms, vocab]` tensor to fp32 for CE. Do not upcast
the full `[batch, seq, vocab]` logits tensor before selection in V1. If the
selected atom count later makes this too large, add CE chunking as a separate
implementation improvement.

`LossContext.select_logits_fp32(...)` should return a compact selected logits
tensor with shape `[num_selected_positions, vocab_size]`, plus aligned target
ids and compact metadata from the context such as segment ids, span ids, roles,
and positions. It should not return logits alone, because that makes target
and provenance alignment too easy to get wrong. It also should not introduce a
rich `SelectedLogits` abstraction in V1 unless implementation pressure proves
that a small object is clearer than the compact tuple-like return.

`LossContext` should not cache selected fp32 logits in V1. Each loss term may
select and upcast what it needs. If profiling later shows repeated identical
selection dominates runtime or memory traffic, add an explicit keyed cache as
a measured optimization.

For differentiable losses, selected logits or hidden states must not be
detached. For metrics, selected logits should be detached and may use the
model dtype by default when the metric is order-only, such as top-k accuracy.
Metrics should request fp32 only if instability is observed or if the metric's
math genuinely depends on numeric precision.

Bad sample or bad step handling should be diagnostic-first and should not become
a second scheduling system. The run clock is the precomputed planned step index;
recoverable bad-sample or bad-step conditions are recorded as warnings and
metrics, and they do not retime eval, checkpoint, logging, or the resolved
planned step count.

NaN/Inf loss or gradient handling is only a corruption guard. Do not silently
replace non-finite losses with zero and do not apply an optimizer update from a
non-finite scalar or gradient. Record diagnostics for the planned step and
continue only when the runtime can do so without corrupting weights. Non-finite
diagnostics should include the offending term name, weighted and raw loss when
available, dtype, selected atom count, pack id, segment ids, planned step,
optimizer-update status, grad norm if available, and min/max logits when cheap
to compute. V1 should not add a public bad-step threshold policy unless a later
approval card asks for one.

Non-finite handling has two distinct gates. First, after `LossRunner.compute`,
the trainer/runtime must check `LossBundle.total_loss` and per-term finite
flags before any backward call. If the scalar is unsafe, skip backward for the
whole planned optimizer-step window, clear any accumulated gradients for that
window, advance schedule state according to the planned-step policy, and record
the same all-rank unsafe decision that would be used at the optimizer boundary.
Second, after successful backward on safe scalars, `TrainRuntime` checks
gradient finite status and backend overflow status before clipping or stepping.
Individual loss terms must not silently skip themselves or replace their own
non-finite values with zero; they report diagnostics into the central policy.

`LossNormalizers` is a planned-step-window contract, not a local pack helper.
It is computed before backward for every `MicroStep` in a planned optimizer
step and carries immutable denominator/count metadata for protected V1 losses:
eligible segment counts, optional eligible atom counts, rank/world context, and
the reduction policy each term is allowed to use. In distributed runs, counts
needed for a protected term's denominator are reduced across ranks before any
rank starts backward for that planned step. `LossRunner` may normalize protected
terms by these planned-step denominators; `SupervisedTrainer` and
`TrainRuntime` must not apply a blind extra `resolved_grad_accum_steps` divisor
to those protected losses.

Backend accumulation semantics must be explicit per runtime backend. Raw
single-process runtime backprops the planned-step-normalized micro contribution
directly. Accelerate and DeepSpeed paths are not allowed to accidentally divide
the same scalar again through backend gradient-accumulation conveniences. The
runtime card must either configure the backend so `LossBundle.total_loss` is
already the scalar actually differentiated, or compensate in a documented,
tested way. DeepSpeed execution is not eligible for support claims until this
loss-scaling and non-finite policy is proven through the real backend path.

In the next version, auxiliary losses that need hidden states should likewise
select the needed hidden states first and upcast those selected tensors to fp32
before loss math, unless their approval card explicitly chooses another
precision policy.

Custom `LossTerm`s should use a minimal interface:

```python
name
required_inputs
select_atoms(context: LossContext)
compute(context: LossContext) -> LossTermResult
```

Do not introduce setup/teardown hooks, registries, or a broad class hierarchy
in V1. Declaring `required_inputs` is enough to prevent hidden assumptions
about logits, hidden states, vocabulary groups, token fields, or other model
outputs while keeping the abstraction small. `select_atoms(...)` should be a
visible selection step so research losses can be audited before math is
applied.

`ModelOutputs` is the stable wrapper around Qwen forward outputs. `logits` is
mandatory. Other outputs are opt-in through declared requirements, starting
with hidden-state capture in the next version after the wrapper-owned hook is
approved and smoke-tested. The resolved loss/eval config gathers
model-output requirements from each term's `required_inputs`; V1 should fail
fast if a runnable config requires `hidden_states` before the hook exists. Do
not expose visual-tower intermediate features in V1 by default. Add them only
when a concrete loss term earns that requirement.

Do not introduce a public `LossRecipe` abstraction in V1. The user-facing and
documentation vocabulary should be loss terms, not recipes. If implementation
pressure later requires a frozen post-config object, it may be added as an
internal resolved execution object, but it should not become a research concept
unless it earns that role.

`LossRunner` is the only aggregate loss executor. Its main interface is:

```python
LossRunner.compute(
    model_outputs: ModelOutputs,
    packed_sequence: PackedSequence,
    loss_config: LossConfig,
    normalizers: LossNormalizers,
) -> LossBundle
```

It installs the protected default stabilizers, initially `BaseTokenCE` plus
`TokenTypeGateLoss`, validates term names and weights, runs non-default
auxiliary `LossTerm`s, applies the approved reduction, and emits `LossBundle`.
The trainer should not manually call individual loss functions. Exact helper
names may change, but the contract must preserve planned-step denominator
semantics rather than silently averaging pack-local means.

`LossBundle` should contain `total_loss`, per-term `LossTermResult`s, scalar
metrics, count diagnostics, and warnings or errors. For V1 protected
token-wise terms, `LossBundle.total_loss` is the differentiable contribution for
the current `MicroStep` after applying the planned optimizer-step denominator;
it is not a pack-local mean that the trainer blindly divides by
`resolved_grad_accum_steps`.

Each `LossTermResult` should expose `name`, `raw_loss`, `weighted_loss`,
`weight`, numerator, denominator, reducer name, selected count, skipped count,
`math_dtype`, and optional diagnostics. Full token-wise loss tensors are
debug-only and should not be carried in every training step by default.

`LossRunner` applies configured weights centrally and records length-invariant
normalized values. Individual loss terms should not own final weighting or
logging policy. Protected default terms such as `BaseTokenCE` and
`TokenTypeGateLoss` fail fast if they have zero eligible atoms. Optional future
auxiliary terms may explicitly allow a zero-eligible no-op, but that must be a
term-level contract rather than silent default behavior.

The default token-wise reducer is `segment_balanced`:

```text
per_segment_loss = mean(loss over eligible atoms in that segment)
term_loss = mean(per_segment_loss over eligible segments)
```

This is the default for `BaseTokenCE`, `TokenTypeGateLoss`, and other token-wise
terms unless a future explicitly approved experiment chooses otherwise. A single
global mean over all eligible packed tokens is only a diagnostic view by
default, because it lets long examples dominate the objective. For a given term,
segments with zero eligible atoms are excluded from that term's denominator;
protected terms still fail if the whole pack has zero eligible atoms. Do not
count zero-eligible segments as zero loss.

Under gradient accumulation, `segment_balanced` means balanced over the full
planned optimizer step/effective batch, not balanced independently per
`MicroStep`. The trainer/loss path must know the eligible segment denominator
for the planned step before backpropagating protected V1 token-wise losses. The
micro-step contribution is:

```text
sum(per_segment_loss for eligible segments in this MicroStep)
/ eligible_segment_count_for_the_planned_step
```

Do not compute a mean per pack and then divide by `resolved_grad_accum_steps`,
because that makes the objective pack-balanced when micro-steps contain
different segment counts. Future loss terms that cannot expose a planned-step
denominator before backward need their own approved reduction contract; they do
not get to inherit pack-balanced behavior accidentally.

Future object-level losses should default to object-balanced reduction:

```text
per_object_loss -> mean over objects -> mean over segments
```

This prevents object-rich examples from dominating object-level objectives by
default. V1 should keep the public denominator config surface minimal:
`segment_balanced` is fixed for the first smoke, and term-level denominator
configuration should be added only when a concrete experiment needs it.

Every `LossTermResult` must expose enough denominator detail to audit the
optimized scalar: selected atom count, eligible segment count, skipped segment
count, denominator mode, numerator, denominator, raw scalar, and weighted scalar.
Metrics may additionally report token-weighted diagnostic means for comparison
with MS-Swift/HF CE behavior, but diagnostic denominator names must be explicit
and must not be confused with the optimized loss.

Stored ordinary loss metric keys contain weighted optimized scalar values only.
Do not emit routine `loss/<term>/raw` or `loss/<term>/weighted` metric pairs.
The short stable names `loss/total`, `loss/base_ce`, and
`loss/token_type_gate` are weighted values; when a term weight is `1.0`, the
weighted value naturally equals the raw term scalar. Raw/unweighted term values,
numerator/denominator details, and reducer internals may remain in
`LossTermResult`, `loss_plan.json`, or targeted diagnostics, but they are not
standard metric events.

`metrics/` records small `MetricEvent`s with `step`, `split`, `name`, `value`,
`scope`, optional `counts`, optional `context`, and a timestamp. Canonical split
names are `train`, `eval.forward`, and future `eval.inference`. When logits and
supervised atoms exist, mandatory metrics are CE loss, `acc_top1`, `acc_top5`,
supervised atom count, contributing segment count, and effective pack cost.
Metric events are keyed by the planned training step id, not by successful
optimizer-update count or wall-clock time. Events may also record
`optimizer_update_applied`, warning/non-finite status, and event kind so static
eval/checkpoint schedules remain interpretable even when a planned step is
diagnostic-only or update-skipped.

Core metric names are a small stable slash-name registry, not arbitrary module
dictionaries. V1 core names include `loss/total`, `loss/base_ce`,
`loss/token_type_gate`, top-level `acc_top1`, top-level `acc_top5`,
`lr/<group>`, `grad_norm/<group>`, `pack/utilization`, and
`pack/supervised_tokens`. Eval events use the same metric `name` values with
`split="eval.forward"`; do not store `eval.forward/acc_top1` as the event name.
Strings such as `eval.forward/acc_top1:max` are selector expressions over
`split/name:mode`, not metric names. Do not name token-level accuracy as
`acc_top1/base_ce` or `acc_top5/base_ce`; top-k accuracy is a global
supervised-token health metric, not a CE-submetric.
Token-weighted diagnostic views use explicit diagnostic suffixes such as
`loss/base_ce/token_weighted_diag` and
`loss/token_type_gate/token_weighted_diag`. Type-sliced scalar names use
`/by_type/<token_type>`, for example `loss/base_ce/by_type/coordinate`,
`acc_top1/by_type/coordinate`, and `loss/token_type_gate/by_type/schema`.
Reducer diagnostics use explicit names such as `loss/base_ce/segment_mean`,
`loss/base_ce/segment_count`, and `loss/base_ce/token_weighted_diag`.
Named counters use the `count/` namespace, including
`count/supervised_atoms`, `count/eligible_segments`,
`count/skipped_segments`, `count/packs`, and `count/examples`. Never put
`example_id`, `object_id`, or arbitrary span text in metric names; those belong
in bounded diagnostic artifacts or structured context.

V1 metric slicing supports aggregate metrics plus optional role and span-kind
slices from `TokenAtom.role` and `TokenSpan.span_kind`. Do not add arbitrary
user-defined groupby expressions in V1.

The canonical metrics storage is `run_dir/metrics/events.jsonl` plus
`run_dir/metrics/summary.json`. TensorBoard or CSV may be secondary exports, but
they are not the source of truth. Do not embed all metrics into
`run_manifest.json`.

Modules return typed metric results or compact metric dictionaries; `MetricSink`
owns rank safety, JSONL append, summary updates, and optional secondary exports.
Metric values may be scalar numbers, booleans, short strings/enums, and bounded
small dictionaries for structured status. They must not carry tensors, arrays,
giant traces, rendered text dumps, or large token-wise payloads.

`LossContext` is the hot-path compiled view of `TokenSequence`, `PackedLayout`,
`ModelOutputs`, tokenizer/vocab metadata, segment/provenance maps, selection
helpers, fp32 selected-logit helpers, and metric helpers. It is built by one
small builder function in
`losses/context.py`:

```python
build_loss_context(packed_sequence, model_outputs) -> LossContext
```

This builder applies the shared `target_position` to `logits_position` rule.
`TokenAtom` stores physical `target_position`; `logits_position` is computed
inside `LossContext` as `target_position - 1` for standard causal objectives.
After packing, the causal shift must remain inside the same `PackedSegment`:
every supervised atom must satisfy
`segment.start_position < target_position < segment.end_position`, and the
derived `logits_position` must belong to the same segment as `target_position`.
No supervised atom may target the first physical token of a packed segment.
This mirrors the MS-Swift safety property where each encoded row's first label
is ignored before rows are concatenated. Pack validation should fail before
`LossContext` construction if this invariant is violated.
`LossContext` should expose tensorized fields such as
`target_positions`, `logits_positions`, `primary_token_ids`, `segment_ids`,
`role_ids`, `span_ids`, and `token_type_ids`, plus optional sparse/ragged
auxiliary target data.
Loss terms should operate on tensor selections from `LossContext`, not iterate
Python `TokenAtom` objects in the hot path.

The approved storage/compute split is: `PackedSequence` stores inspectable
CPU-side supervision records and provenance, while `LossContext` builds the
tensorized hot-path view for the current step. `PackedSequence` should remain
debuggable and artifact-friendly; `LossContext` owns device placement,
index-tensor construction, and any compact tensor/ragged representations needed
by loss terms or metrics. Loss terms must not depend on Python iteration over
`TokenAtom`s during real training. Extending to hidden-state, coordinate,
object-level, or rollout-derived objectives should happen by extending
`LossContext` selections and compiled views, not by making `packing/` aware of
each new loss.

Within a step, `LossContext` may cache compiled index and mask tensors such as
target/logits positions, token-type ids, segment ids, span ids, object ids, and
span/object grouping maps on the logits device. It should not cache selected
fp32 logits in V1; differentiable terms call the relevant fp32 selection helper
when they need selected logits. This keeps repeated alignment work centralized
without adding hidden activation-memory pressure.

Loss terms are pure consumers of `LossContext`. A term may return diagnostics in
`LossTermResult`, but it must not mutate `PackedSequence`, `LossContext`,
`ModelOutputs`, model parameters, or other loss terms. Metrics reuse
`LossContext` selections and detached views; they must not re-run an independent
masking/alignment path.

Token-type vocabulary groups are resolved during loss setup from tokenizer
identity and the approved token families, producing a typed `TokenVocabGroups`
object and `token_type_vocab.json`. Loss terms consume those resolved
groups through `LossContext`; they must not hard-code numeric ranges or resolve
vocabulary groups ad hoc.

`LossContext` is also the precision boundary for loss math. It should provide
fp32 selection helpers for logits and, when needed, hidden states. Loss terms
should use those helpers rather than directly indexing `ModelOutputs.logits`
when computing differentiable objectives.

Every run writes `loss_plan.json`. It records enabled terms, weights,
eligible counts, denominators, token-role counts, vocabulary groups, dtype and
fp32-upcast status, non-finite policy, and metric definitions. The summary is
the audit artifact for the question "what objective did this run actually
optimize?"

Resolved `BaseTokenCE` rules:

- production training should fail closed if the base CE stabilizer is disabled,
  unless an explicit override with a reason is provided;
- `BaseTokenCE` consumes the `primary_token_id` from `TokenTarget`;
- multi-positive token ids and sparse soft token weights are for auxiliary
  terms, not for changing the basic CE target.

Do not add `ce_enabled` to `TokenAtom` in V1. `TokenAtom` existence implies
`BaseTokenCE` applies. Tokens that should not receive token-level supervision,
such as prompts, image placeholders, padding, or ignored regions, should simply
not have `TokenAtom`s. Auxiliary terms reuse the same atoms by selecting on
role, span kind, and optional auxiliary target fields.

Packing performs supervision placement once: from local example positions to
physical packed positions. The operation is not token search. It uses the known
`PackedSegment` location to assign local supervision into the packed sequence:

```python
place_token_sequence_in_pack(
    token_sequence: TokenSequence,
    segment: PackedSegment,
) -> TokenSequence
```

Conceptually:

```text
physical_position = segment.start_position + local_position
```

`packing/` owns this placement while building `PackedSequence`; `supervision/`
defines the structures and validation helpers but does not know packing policy.
After placement, objective modules must not perform their own packing-offset
arithmetic.

Supervision validity is checked during encoding and again after packing
placement. Encoding catches local mistakes such as missing supervised atoms,
bad local positions, or span-alignment failure. Packing catches physical-layout
mistakes such as out-of-range positions, cross-segment spans, or
placeholder/supervision inconsistency after placement. `LossRunner` may assert
preconditions, but it should not be the first validation boundary.

The golden reduction rule is length-invariant: longer segments must not gain
extra objective influence merely because they contain more supervised tokens.
The default reduction is segment-balanced:

```text
segment_loss = mean(selected atom losses inside one packed segment)
term_loss = mean(segment_loss over contributing packed segments)
```

Segments with no selected atoms for a term do not contribute to that term's
mean. Loss and metric diagnostics should still expose raw counts such as
selected atom count, selected span count, contributing segment count, and a
clear reducer name. Optimization defaults should not use padded length, raw
sequence length, conceptual batch size, or MS-Swift-style token weighting as
the denominator.

Explicit weights may reshape loss pressure, but hidden length bias is not
allowed.

MS-Swift scalar CE is token-weighted by default because its dense labels use
`ignore_index=-100` and reduce over supervised token sites. CoordExp-swift
deliberately keeps segment-balanced CE as the default optimized objective.
Boundary parity should be checked through token triples, while training quality
should be read from weighted CoordExp loss metrics, token-level top-1/top-5
accuracy, and the task's evaluation results. If token-weighted CE is reported
for comparison with MS-Swift/HF behavior, it must use an explicit diagnostic
name such as `loss/base_ce/token_weighted_diag` and must not be confused with
the optimized loss.

Loss and metric denominators should be based on supervised target sites,
spans, or contributing packed segments, with the reducer declared by the loss
term.

The core architecture is:

```text
Dataset/template/tokenizer
  -> logical example and local supervision
Packed sequence builder
  -> physical tensors and packed layout
Supervision placement
  -> validated token sequence with physical target positions
Model forward
  -> ModelOutputs
Loss runner
  -> loss context, base CE, auxiliary terms, reductions, metrics
Trainer
  -> backward/optimizer/checkpoint orchestration
```

The recurring core concepts are:

- `TokenSequence`
- `TokenSpan`
- `TokenAtom`
- token target records
- `PackedLayout`
- `ModelOutputs`
- `LossContext`
- `LossRunner`
- `LossTerm`
- `ReductionPolicy`
- provenance axes for dataset, teacher-forced, rollout, pseudo, and diagnostic
  sources
- strict validation profiles

Names are provisional until module approval cards are accepted, but the
conceptual responsibilities are approved.

## Packing Principles

Packing is not a collator trick. It is a first-class physical layout contract.
Do not over-design it. The V1 abstraction should contain only the concepts
needed to make packing inspectable, debuggable, and loss-safe:

- `PackPlan`: example grouping and order before tensor assembly.
- `PackedSegment`: one example's contiguous region inside a packed sequence.
- `PackedLayout`: physical positions, packed segments, and segment boundaries.
- `PackedSequence`: assembled train/eval unit with tensors plus layout.

CoordExp-swift should not clone MS-Swift's mutable `Template` abstraction as
the owner of packing. The design should split packing into explicit concepts:

- pack planning
- packed segment mapping
- Qwen3-VL position and varlen metadata construction
- multimodal feature collation
- supervision placement
- loss denominator semantics

Objective modules should receive already-validated physical positions and
should not reinterpret raw packing offsets.

`PackedSegment` should minimally record `example_id`, encoded fingerprint,
physical start/end positions, target-position range, image placeholder spans,
feature-row cost, and source/render trace ids. It should not copy the full
`EncodedExample`.

The packer consumes an already ordered stream of `EncodedExample`s. Data or
stream-ordering policy decides the order before packing; the packer does not
sort, shuffle, or sample examples internally.

`PackedSequence` is the canonical model-agnostic physical sequence produced by
`packing/`. It contains concatenated `input_ids`, a remapped physical-position
`TokenSequence`, a `PackedSegment` table, local-to-packed mapping records,
visual payload references or concatenation metadata, pack-level provenance, and
pack-cost diagnostics. It is not a Qwen forward kwargs dict and does not store
dense labels as canonical supervision.

`PackedSequence` is therefore the boundary object between sequence layout and
loss semantics: it owns the physical sequence and carries the remapped
supervision ledger, but the meaning of each supervised position still comes
from `TokenSequence`. Packing changes physical positions, not semantic
ownership. Every loss term reads packed positions plus supervision metadata via
`LossContext`; losses must not infer supervision meaning from packed offsets,
dense labels, or example order.

Packing uses greedy streaming admission in V1. The packer attempts to append
the next `EncodedExample` to the current pack; if that would exceed the hard
budget, it commits the current pack and starts a new one with that example. It
does not perform best-fit bin packing, length sorting, random packing, or any
other implicit example reordering. Upstream sampling or data streaming owns
example order.

`global_max_length` counts the full encoded `input_ids` length after Qwen
chat-template and image-placeholder expansion: prompt tokens, image placeholder
tokens, assistant tokens, and the `<|im_end|>\n` suffix all count. It is not
assistant-only length, supervised-token count, or raw pre-expansion text length.
Pack diagnostics still record image-token counts and vision feature row counts
so memory pressure remains visible, but V1 does not hide an extra pack
admission rule behind those diagnostics.

Segment isolation is represented by the `PackedSegment` table. Each segment
records at least physical `start`, `end`, `example_id`, local encoded length,
epoch index when training-stream context is available, realized data-order
metadata when available, image placeholder ranges, and provenance.
Qwen/FlashAttention `cu_seq_lens_q/k` are derived from this table, not treated as
the only human-readable source of segment truth.

Supervision remapping is mechanical and validated. Packing adds the segment
start offset to every local `TokenAtom.target_position` and every `TokenSpan`
boundary from the source `EncodedExample`. It then validates the inverse mapping
back to the original encoded example so loss attribution can be traced from
packed positions to local positions and source spans. Loss code receives
already-remapped physical positions and does not perform its own local-offset
interpretation.

Qwen visual payloads stay ordered by packed segment. `packing/` preserves
per-example image payload references and segment metadata; the Qwen bridge stacks
or concatenates the image processor payloads in segment order when constructing
`QwenForwardInputs`. Image placeholder ranges and `image_grid_thw` remain
traceable per segment. Do not let the trainer concatenate arbitrary HF keys, and
do not drop placeholder-range metadata.

`pack_plan.json` records enough evidence to debug order, utilization,
supervision density, and cache drift. Production runs use a summary-first
receipt: counts, utilization summaries, order policy, seed/fingerprint inputs,
tail-fill status, and validation status. Smoke/debug runs may additionally write
per-pack example ids, segment starts/ends, token lengths, supervised-token counts
by role, image-token counts, utilization, and fingerprint inputs. It should not
dump full token ids for every pack by default.

## Qwen Packing And FlashAttention Boundary

`PackedLayout` is model-agnostic. It owns physical structure: packed segment
boundaries, segment ids, example ids, physical positions, and segment
ownership for length-invariant loss reduction. It must not own Qwen-specific
state such as MRoPE rows, `image_grid_thw`, pixel slices, or FlashAttention
varlen tensors.

Qwen-specific execution state belongs in `qwen/`. The Qwen adapter should reuse
or mimic the official Transformers Qwen3-VL processor/model path wherever
possible, while keeping CoordExp validation local. It should call the HF
processor/image processor through an explicit no-resize call path
(`do_resize=False` or the installed processor's equivalent), preserve Qwen
image/grid ordering, verify image placeholder counts against `image_grid_thw`,
and return typed Qwen feature/forward-input objects.

Visual placeholder expansion happens in `qwen/encoding.py` per
`EncodedExample`, before packing. Qwen encoding expands the chat-template image
placeholder into the model-required placeholder-token count using the
processor/grid contract. `packing/` then concatenates already-expanded token
sequences and validates counts after physical placement. Do not defer visual
placeholder expansion to packing or training.

`global_max_length` is the hard physical sequence-length admission budget for
one packed sequence. It counts physical `input_ids` positions after
chat-template and image-placeholder expansion, not raw text length. Pack
diagnostics should also record the visual burden derived from `image_grid_thw`:

- `effective_pack_cost`
- `physical_input_length`
- `text_token_count`
- `image_placeholder_token_count`
- `vision_feature_row_count`

In V1, `physical_input_length` is the hard `global_max_length` gate, while
`effective_pack_cost` keeps the broader memory-cost evidence visible for
profiling and later approval. Do not silently change admission behavior from
sequence length to a custom cost formula without a separate decision.

`EncodedExample` is atomic and must not exceed `global_max_length` by physical
expanded `input_ids` length. Overlength examples fail fast. The default smart
packing policy is greedy append admission:

```text
try to add the next EncodedExample to the current PackedSequence
if its physical expanded input length fits under global_max_length:
  keep adding
else:
  commit the current PackedSequence
  start a new PackedSequence with that EncodedExample
```

This policy may be improved later, but the core invariant remains: examples
are not split across packed sequences.

No cross-example semantic attention is allowed by default. Segment isolation
must be represented through Qwen/FlashAttention-compatible position and
attention metadata. Continuous-stream attention across packed examples would be
a separate research recipe, not the default packed supervised path.

`packing.policy: one_example` is allowed as an explicit parity/debug mode that
puts one `EncodedExample` in each `PackedSequence`. It is not an automatic
fallback and does not change the default requirement that packed training is
the standard path.

Every run should write `pack_plan.json`; verbose runs may additionally
write a JSONL trace for individual packs. The summary should include
`packs_per_epoch`, requested pack presentations, actual pack presentations,
`tail_fill_pack_count`, `resolved_max_steps`, `effective_batch_size`, world size,
`resolved_grad_accum_steps`, utilization summaries, data-order policy, object
ordering policy, seed/fingerprint inputs, fail-fast counts, segment counts,
min/max/mean effective cost, and representative pack ids.

The packed builder should output two typed objects:

- `PackedSequence`: generic physical sequence, `PackedLayout`, and
  `TokenSequence`.
- `QwenForwardInputs`: exact Qwen3-VL forward inputs such as `input_ids`,
  `pixel_values`, `image_grid_thw`, `position_ids`, FlashAttention varlen
  tensors, and Qwen-only kwargs.

The loss runner consumes `ModelOutputs` plus `PackedSequence`; it should not
depend on raw Qwen forward-input dictionaries.

FlashAttention v2 is required for packed Qwen3-VL supervised training unless an
explicit debug/parity profile disables it. CoordExp-swift should prefer the
Transformers-supported `attn_implementation="flash_attention_2"` path, but it
must own the packed varlen boundary that MS-Swift previously handled
implicitly.

Packed FA2 isolation must be proven at the branch level, not only by tensor
shapes. The training forward path passes no dense 2D `attention_mask` for
segment isolation; it passes explicit `cu_seq_lens_q`, `cu_seq_lens_k`,
`max_length_q`, and `max_length_k` derived from `PackedSegment` boundaries.
Forward-contract probes should capture or wrap the relevant Transformers/Flash
Attention call and assert that the explicit varlen path is used, the max lengths
are Python ints, and position-id resets agree with the same segment table. A
stray attention mask that routes through a padded/unpad branch is a contract
failure for packed training.

For FA2 packed training, `qwen/` should derive execution tensors from
`PackedLayout` and validate them against it:

```text
cu_seq_lens_q
cu_seq_lens_k
max_length_q
max_length_k
position_ids
```

Use the HF forward keyword names `max_length_q` and `max_length_k` at the Qwen
model boundary. The installed Transformers wrapper translates these into the
FlashAttention package's internal `max_seqlen_q/k` convention. Do not expose
`max_seqlen_q/k` as the CoordExp-swift Qwen-forward API names.

`cu_seq_lens_*` must preserve segment isolation and be compatible with the
installed FlashAttention v2 package expectations: flattened token-major
attention, `torch.int32` cumulative lengths, shape
`[num_physical_segments + 1]`, starting at `0`, strictly increasing, and ending
at the physical packed row length after visual placeholder expansion.
`max_length_q/k` must be Python `int`s equal to the maximum segment length from
the corresponding cumulative lengths. For Qwen3-VL text self-attention,
`cu_seq_lens_q == cu_seq_lens_k` and `max_length_q == max_length_k` in the
standard supervised path. Do not rely on a plain 2D `attention_mask` or infer
semantic truth back from `position_ids`.

For padding-free packed FA2, segment isolation must come from explicit
`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k`.
`attention_mask` may exist as compatibility or diagnostic metadata, but it is
not the segment-isolation mechanism. The packed FA2 model call should omit a
2D `attention_mask` or pass an all-valid shape that HF reduces to `None`; a 2D
mask with zeros can route Transformers into the padded unpad branch and bypass
the explicit varlen kwargs. If `cu_seq_lens_q/k` are present and an ordinary
2D `attention_mask` contains zeros, training should fail fast. Non-FA debug
paths may use `packing.policy: one_example` or a true segment-isolating
custom/4D mask; they must not pretend that a 2D padding mask can encode packed
example boundaries.

Image placeholder or feature mismatches fail fast before the training forward.
The error should report the example id, image path, placeholder count,
`image_grid_thw`, expected feature/placeholder relationship, and actual visual
feature rows.

## Data, Template, Encoding, And Cache Boundary

The initial supervised Qwen3-VL single-image flow is:

```text
raw *.jsonl
  -> RawExample
  -> RenderedExample
  -> EncodedExample
  -> PackedSequence + QwenForwardInputs
  -> Qwen3-VL forward
  -> visual tower features replace image-token positions
  -> language-model tower
  -> ModelOutputs
  -> LossRunner
```

`data/` owns JSONL loading and `RawExample`. V1 data parsing supports JSONL
only. Do not add JSON, dataset registries, or multi-format loading before the
first vertical path is solid.

A `RawExample` is a typed, minimally transformed record containing:

- `example_id`;
- `image: {path, width, height}` with `width` and `height` optional;
- `objects`;
- optional `metadata` dict;
- source path, row number/offset when available, and lightweight raw-record
  provenance such as a content hash.

The data-owned raw types live in `src/data/types.py`: `RawExample`,
`RawObject`, and `ImageRef`. They are data-owned even though templates and Qwen
encoding consume them.

Unknown top-level fields fail validation. Extra provenance belongs under the
single `metadata` dict, not as arbitrary top-level schema drift. `RawExample`
does not contain rendered prompt text, assistant text, token ids, Qwen processor
outputs, packing fields, or loss fields. Pre-rendered assistant text from an
input JSONL is treated as raw source content until a template renderer accepts
it explicitly.

The canonical internal `RawExample` field names are intentionally clear:
`image`, `object_id`, `description`, and `bbox`. However, V1 must also read the
current main-branch training JSONL paths without requiring regeneration. The
reference path family includes:

```text
public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl
public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

Those files are currently referenced from main-branch mechanism/training and
inference configs such as
`configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3.yaml`,
`configs/analysis/prefix_state_transition_tomography/fullobj_policy_objective_5ckpt_ckpt3668_phase_a3.yaml`,
and
`configs/infer/recursive_detection_ce/fullobj_sorted_purece_ckpt3668_a3_2_rollout1024_greedy.yaml`.
They are the narrow compatibility target for the new loader. Call this the
current coord-jsonl source format, not "legacy" data. V1 supports the known
current `len12000` coord-jsonl source fields only, not a broad alias surface.
The loader should normalize that current format into the canonical
`RawExample` without rewriting or regenerating the JSONL:

- `images` must contain exactly one item and maps to `image.path`;
- top-level `width` and `height` map to `image.width` and `image.height`;
- explicit `example_id` is preferred, otherwise stable current-source identifiers such
  as `image_id` or `file_name` may be normalized into `example_id`;
- object `desc` maps to `description`;
- object `bbox_2d` may contain four coordinate-token strings such as
  `<|coord_568|>` and maps to integer `bbox`;
- explicit object `object_id` is preferred, otherwise stable current-source ids
  such as `coco_ann_id` may be normalized into `object_id`;
- `category_id`, `category_name`, `image_id`, `file_name`, and source fields are
  preserved under metadata/provenance.

If neither the canonical field nor an approved stable current coord-jsonl source
field exists, the row fails validation. Unknown top-level fields still fail. Do
not generate example ids or object ids from row/object index in V1.

The bbox flow is deliberately explicit:

```text
current coord-jsonl source format:
  bbox_2d: ["<|coord_602|>", "<|coord_140|>", "<|coord_939|>", "<|coord_840|>"]
canonical RawExample:
  bbox: [602, 140, 939, 840]
rendered assistant text:
  <|coord_602|><|coord_140|><|coord_939|><|coord_840|>
```

`data/` parses and validates coordinate-token strings into integer coordinate
bins. `templates/` owns rendering integer bins back into coordinate-token
strings. `qwen/encoding` then verifies those rendered coordinate-token strings
are tokenizer-valid single tokens. Do not let raw source token strings bypass
the canonical geometry validation step.

The coordinate-token parser for the current coord-jsonl source format is strict.
It accepts only exact canonical strings matching `<|coord_N|>` where `N` is a
decimal integer in `[0, 999]` and the whole string matches the token literal.
Reject whitespace, raw integer strings, floats, signs, malformed wrappers,
non-canonical zero padding, split token fragments, and out-of-range values.
Canonical `RawExample` JSON may serialize bbox as integer arrays, but
current-source `bbox_2d` token strings must not accept loose alternatives.

Canonical data records should be lightweight immutable runtime values, not
Pydantic config models. Use frozen dataclasses or dataclass-like immutable
records for `RawExample`, `RawObject`, and `ImageRef`; keep Pydantic primarily
for config validation. Raw JSONL rows must not instantiate these records
directly. They flow through narrow loader/factory functions that own strict
validation, id normalization, bbox parsing, and bounded provenance capture.
Exact function names can be approved during implementation, but the public
construction boundary should remain narrow.

Canonical `example_id` and `object_id` are strings. Numeric source identifiers
such as `image_id` and `coco_ann_id` are stringified during normalization.
Every loaded example carries bounded provenance including source JSONL path,
row index, source format name, and stable source ids when available. Byte offset
is optional debug metadata, not a required contract. `metadata` must be
JSON-serializable, bounded, and non-authoritative: it may explain where data
came from, but training semantics come from canonical fields.

The V1 data loader supports only canonical `RawExample` JSONL plus the approved
current coord-jsonl source format. Unknown canonical top-level fields fail, and
current-source compatibility is restricted to explicitly approved fields that
normalize into canonical records or bounded metadata/provenance. Templates
receive canonical `RawExample` objects only, never raw JSON rows. Basic
coordinate-bin parsing and validation belong under `src/data/geometry.py` in
the new worktree; V1 should keep that helper small and avoid image resizing,
pixel conversion, or broad COCO geometry utilities.

The canonical runtime shape is deliberately small:

```text
RawExample:
  example_id: str
  image: ImageRef
  objects: tuple[RawObject, ...]
  metadata: dict | None

ImageRef:
  path: str
  width: int | None
  height: int | None

RawObject:
  object_id: str
  description: str
  bbox: tuple[int, int, int, int]
  metadata: dict | None
```

Use `description`, not source-dialect `desc`, in the canonical layer.
`RawExample.objects` is a tuple so realized object order cannot be mutated
accidentally after loading. Source `category_id` and `category_name` are
preserved under `RawObject.metadata.source` when present, but they do not affect
template rendering or loss by default. `ImageRef.width` and `ImageRef.height`
are optional in the general type but required by the current Qwen3-VL
supervised training loader path and smoke fixture.

`templates/` accepts only canonical `RawExample` values and returns a typed
`RenderedExample` containing rendered messages/text plus character spans.
`templates/` owns human-readable assistant text and character spans.
`qwen/encoding` owns tokenization, token ids, token spans, image-token
expansion, and model-ready tensors. V1 rendered span kinds must cover at least
`assistant_content`, `description`, `schema_token`, `coordinate_token`,
`object`, `eos_transition`, and `ignored_text`.

Each V1 object requires:

- stable `object_id`;
- description text;
- bbox in integer coordinate-bin space.

Object ids must be unique within one `RawExample`; global object identity is the
pair `(example_id, object_id)`. `example_id` must be unique within each loaded
JSONL file or dataset source. Do not auto-generate object ids by index in V1.
Stable ids make ordering, span provenance, debug reports, cache keys, and loss
attribution auditable.

V1 is exactly single-image. Zero-image and multi-image examples fail validation
rather than silently becoming text-only examples or first-image-only examples.

Bbox geometry validation happens in `data/`, before template rendering. The
canonical field name is `bbox`, not `bbox_2d`. For the initial supervised
Qwen3-VL scope, bbox format is constant `[x1, y1, x2, y2]` in integer
coordinate-bin space `[0, 999]`. In Python, `RawObject.bbox` should be stored as
`tuple[int, int, int, int]` so the geometry is not accidentally mutated. YAML
and JSON fixtures may still serialize it as an array.
`data/` validates bbox schema plus numeric and range sanity: four integers,
each in `[0, 999]`, with `x1 < x2` and `y1 < y2` for teacher-forced supervised
training. Zero-area boxes are schema-invalid for this training path. Do not
clamp invalid values, warn-and-keep-going, or defer conversion from pixels in
V1. Templates, Qwen/tokenization, packing, and loss code should never receive
invalid geometry.

Preserve original current-source `bbox_2d` coordinate-token strings under
metadata or bounded debug provenance when useful, especially in fixtures and
loader diagnostics, but never make those strings the canonical geometry field.

Later rollout training and offline inference must also validate decoded model
geometry instead of accepting arbitrary model output. Generated predictions may
fail parsing or geometry validation and should be reported as invalid decoded
outputs through the relevant inference/eval artifact contract; they must not be
silently converted into trusted supervision or accepted as valid boxes just
because the model decoded them.

Data loading resolves image references relative to the declaring JSONL/config
root, records the resolved path, and checks path existence. It does not open
images or pass live image objects forward. `qwen/` opens the image later for HF
processor conversion. If declared width/height disagree with the decoded image
size later, the Qwen encoding/forward boundary fails fast with declared and
actual size in the diagnostic. `templates/` never resolves or opens images.

Descriptions must be non-empty strings after stripping. Deeper control-token,
chat-template, and Qwen special-token safety checks belong in `templates/`, not
in the raw data parser.

`metadata` must be JSON-serializable: nested primitive values, lists, and dicts
are allowed; arbitrary Python objects are not. Raw provenance should include
source path, row number, optional byte offset when available, and a hash of the
raw line or canonical raw record. Do not store the full raw dict inside every
`RawExample` by default.

Image identity fingerprinting uses resolved path plus file stat by default.
Stricter/debug surfaces such as the permanent smoke fixture may add an image
checksum. Full image checksums are optional, not the default for every run.

`sample_limit` applies in source order after validation of rows encountered for
the sample. In a sample-limited smoke, the loader reads and validates rows in
order until the requested number of valid examples has been accepted; any
invalid row encountered before that point fails the run. It should not stop
after N raw rows before validation, and the trainer/packer should not own
sample limiting.

Invalid records fail the run in V1. Do not add count-and-skip or configurable
skip policy before the strict vertical path is established. This keeps the first
infrastructure honest about data quality instead of hiding schema or geometry
drift behind counters.

Use precise naming for object serialization choices:

- `object_field_order` is the order of fields inside one object serialization,
  for example `desc_first` or `geometry_first`.
- `object_ordering` is the order of objects in the assistant response, for
  example `source_order`, `geometry_sorted`, random, or a sampled permutation.

V1 prompt language is fixed to English. Prompt wording may change with
`object_field_order`, but the logical bbox coordinate order remains
`x1,y1,x2,y2` unless a separate approval card changes it. This is coordinate
order, not necessarily rendered punctuation.

The V1 template defaults are `object_field_order: desc_first`,
`object_ordering: source_order`, and `assistant_format: object_box_closed`.
There is no `template.language` field in V1. Prompt text is a short fixed
English renderer-owned string in `src/templates/`, not an external prompt file
in V1.

Source-order-vs-random is an `object_ordering` choice. It must be explicit in
configs, traces, cache fingerprints, and inspection output. The default
`object_ordering` is `source_order`, meaning preserve the validated object
order from the source JSONL. Do not call this `sorted` in new config surfaces,
because that name hides whether ordering is source order, geometric order, or
lexical order. If a future renderer sorts geometrically, name that value
`geometry_sorted` and define the key precisely, for example top-to-bottom then
left-to-right.

This is a compatibility-sensitive terminology shift. Existing CoordExp configs,
notes, and artifact names historically used `sorted` and `random`, with no
separate "original source order" knob. New CoordExp-swift configs should use
`source_order`; `sorted` is rejected by V1 config validation rather than accepted
as a live alias. If old configs are migrated, migration tooling must rewrite
`sorted` to an explicit canonical value before validation. Do not silently
introduce a new geometric sort under the old `sorted` spelling.

`object_ordering: random` is allowed only with a deterministic seed stream and a
recorded realized order in `RenderedExample`. Random ordering should happen per
example per epoch or pack/render cycle from that deterministic seed stream;
otherwise pack plans and cache entries are not reproducible.

Example order across the training dataset is a separate concept from
`object_ordering` inside one rendered assistant response. The V1 config uses
`data.train_order`, with default `shuffle` for production and optional
`source_order` for smoke/debug reproducibility. `data.train_order: shuffle`
shuffles examples between epoch streams from the global seed-derived data-order
seed. It must not change object order inside an example; that remains owned by
`template.object_ordering`.

The first canonical `assistant_format` is `object_box_closed`. It is
structural, not free prose and not JSON-like text. Each object is rendered as a
compact schema segment with Qwen object/box wrapper tokens around the
description and bbox. The canonical shape is:

```text
<|object_ref_start|>description<|box_start|>x1 y1 x2 y2<|box_end|><|object_ref_end|>
```

The actual V1 wrapper token strings are the tokenizer/Qwen wrapper tokens for
the `schema` token type: `<|object_ref_start|>`,
`<|object_ref_end|>`, `<|box_start|>`, and `<|box_end|>`. Do not introduce new
literal tokens such as `<|obj_start|>` unless a tokenizer change is separately
approved. Coordinate targets inside the rendered assistant response are
tokenizer-visible coordinate token strings such as `<|coord_123|>`, not raw
integers. V1 renders the four bbox coordinate-token strings adjacent to each
other with no comma, space, or other separator, because the wrapper tokens and
fixed coordinate count already define the structure and total sequence length
matters. The supervised text should reveal exactly which token family is being
trained.

Multiple object schema segments are also concatenated without an inserted
separator. The next `<|object_ref_start|>` begins immediately after the previous
`<|object_ref_end|>`. This is a deliberate length-minimization choice, so any
debug view that displays line breaks must make clear that they are presentation
only and not part of `supervised_response_text`.

The image placeholder belongs in the user message before task prompt text,
following Qwen chat-template convention. It is prompt-side input context and
must never create supervised `TokenAtom`s.

Template variants live in the `template:` config surface. At minimum the
surface should include `object_field_order`, `object_ordering`, and
`assistant_format`, with renderer-owned variant names. Python renderer classes
may implement variants, but config names the experiment. A dormant multilingual
knob is intentionally excluded; adding non-English rendering later requires a
separate approval because it changes prompt semantics.
The template fingerprint includes exact prompt text/version plus renderer config,
because prompt wording changes the supervised context.

`templates/` owns `RenderedExample`. A `RenderedExample` contains the rendered
messages, prompt text, `supervised_response_text`, character spans, provenance,
realized object order, and image reference. Rendered messages are the Qwen
chat-template input; `supervised_response_text` is the loss-bearing assistant
text with spans. It is pure and inspectable: no HF processor calls, no image
processing, no token ids, no `TokenAtom`s, no Qwen forward-input fields, and no
final full Qwen chat-template string. `qwen/encoding` owns the exact processor
text and full chat-template materialization.

`RenderedExample` must preserve both canonical object data and rendered string
spans. At minimum it records source object ids, realized object order, object
spans, description spans, schema wrapper spans, bbox spans, coordinate-token
spans, `eos_transition` spans, and `ignored_text` spans for trivial rendered
suffix text such as the post-`<|im_end|>` newline. This is the provenance bridge
from raw data to token-level supervision, so it must be explicit enough to debug
off-by-one errors and loss attribution.

A `RenderedSpan` minimally records `kind`, `char_start`, `char_end`, `text`,
optional `object_id`, optional `field`, and source provenance. Character offsets
are Python string offsets, zero-based and half-open `[char_start, char_end)`;
`RenderedSpan.text` duplicates the substring and must equal
`supervised_response_text[char_start:char_end]`. Nested spans are allowed, but
crossing spans are forbidden. It must not store token positions; token alignment
belongs to `qwen/`.

Template rendering strips leading/trailing whitespace from descriptions and
normalizes internal newlines or tabs to single spaces, while preserving ordinary
spaces. It then validates render/control-token safety. Descriptions may not
contain Qwen special/control tokens or CoordExp wrapper or coordinate-token
syntax unless a future escaping policy is explicitly approved. Do not strip such
content silently; fail the render with a local diagnostic.

`qwen/` owns tokenization, Qwen processor calls, and span-to-token alignment.
It consumes `RenderedExample`, calls the Transformers Qwen processor/tokenizer
with `do_resize=false`, performs image-placeholder expansion/replacement as
the official Qwen path expects, aligns rendered spans to token positions, and
emits `EncodedExample`.

`EncodedExample` is the first complete local training unit. Its `input_ids`
field is a plain CPU `list[int]` before packing; tensors are created later by
packing/forward builders. It contains a typed `QwenProcessorPayload`,
image/grid metadata, span-to-token alignment, a local `TokenSequence`,
source/render provenance, validation metadata, and enough processor trace data
to debug upstream Transformers changes. It is not a loose HF kwargs dict, and
it should not contain a packed or batch dimension.

`QwenProcessorPayload` stores the processor-owned visual inputs such as
`pixel_values` and `image_grid_thw` as CPU tensors or arrays, plus explicit
shape and dtype metadata. The payload is the narrow bridge to Qwen forward
construction; callers should not pass around an untyped copy of HF processor
kwargs.

Qwen encoding validates tokenizer identity once during Qwen setup or encoder
construction, before iterating examples. Required wrapper/control tokens and
all coordinate tokens used by the active template must map to exactly one
tokenizer id and must encode as exactly that one id. This includes
`<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>`,
`<|im_start|>`, `<|im_end|>`, image/vision tokens required by the processor,
and V1 coordinate tokens `<|coord_0|>` through `<|coord_999|>`. If any token is
missing, unknown, split into ordinary pieces, or resolved ambiguously, training
fails before data encoding with a compact token-identity receipt.

Char-span to token-span alignment prefers tokenizer `offset_mapping`, but the
offsets must be interpreted in the same string space the tokenizer actually
receives. `RenderedExample` spans are authored over `supervised_response_text`.
The Qwen encoder must first place those spans into the full rendered chat text,
then construct the exact processor text after Qwen image-placeholder expansion,
and then carry an offset map from original rendered chat text to processor text.
Only after that conversion may tokenizer offsets be used to build token spans.

The span conversion must be invertible enough for debugging. For every
non-empty rendered span, the implementation should verify:

- rendered span text matches the original rendered-chat slice;
- original rendered-chat slice maps to one contiguous processor-text slice;
- processor-text slice maps to whole tokenizer offsets with no partial-token
  boundary overlap;
- decoded token ids for the resulting token span round-trip to the processor
  text slice when `skip_special_tokens=false`.

If usable tokenizer offsets are unavailable, V1 may use a deterministic
fallback aligner over the processor text only. The fallback must use exact
tokenizer calls, single-token special-token validation, and round-trip checks;
approximate post-tokenization string matching is not allowed.

`EncodedExample` preserves the full supervision trace:
`RenderedSpan -> TokenSpan -> TokenAtom`. This trace is the debugger-facing
source of truth for why a token receives CE, schema-gating, coordinate-gating,
EOS, or future hidden-state supervision.

Local positions in `EncodedExample` are zero-based over the full Qwen input
sequence, including prompt, vision placeholder tokens, assistant tokens, and
the `<|im_end|>\n` suffix. A `TokenAtom.target_position` is the target-token
position in that local full-sequence coordinate system. The loss context owns
the causal shift from `target_position` to the logits row consumed by loss.

`EncodedExample` must fit within `global_max_length` before packing. V1 raises
`EncodingContractError` rather than truncating, skipping, or splitting a single
encoded example across multiple packs. Dense `labels` tensors are not stored by
default; they may be materialized only as derived parity/debug artifacts from
the local `TokenSequence`.

Pre-research on 2026-06-28 used local Transformers `4.57.1` and
`/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
That model path loads `Qwen3VLProcessor` with `Qwen2TokenizerFast`; wrapper
tokens, `<|im_start|>`, `<|im_end|>`, `<|image_pad|>`, and
`<|coord_0|>` through `<|coord_999|>` resolve as single token ids. The
processor returns `offset_mapping` when requested. However, for larger images
the Qwen processor expands `<|image_pad|>` into repeated image-pad tokens before
tokenization, and returned offsets are relative to that expanded processor
text, not the original rendered chat text. The encoder therefore must own the
processor-text reconstruction and offset conversion instead of aligning
assistant spans directly against the unexpanded rendered text.

Follow-up local probing on 2026-06-29 confirmed that this model reports
`tie_word_embeddings=true`, the image processor default `do_resize=true`,
`patch_size=16`, and `merge_size=2`. V1 must therefore force `do_resize=False`
at the actual processor/image-processor call and record both the default policy
and call policy. The installed wrapper tokens are
`<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, and
`<|box_end|>`; literals such as `<|object_start|>` and `<|object_end|>` split
into ordinary text tokens and are invalid V1 schema-wrapper aliases. The same
probe observed no-resize 64x64 and 64x96 RGB images pass, while 64x80 and
112x112 fail with upstream reshape errors, supporting a V1 preflight based on
effective divisibility by `patch_size * merge_size` unless a later approved
processor probe proves a broader safe rule.

The defining V1 smoke and default examples should use
`model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
Raw Qwen3-VL checkpoints without the CoordExp coordinate-token vocabulary are
not valid V1 bases for this pipeline. The loader may still accept a user-authored
model path, but the tokenizer preflight decides whether it satisfies the V1
contract; failure happens before encoding or training.

The encoder should split the Qwen processor path transparently rather than
treating `Qwen3VLProcessor.__call__` as an opaque alignment oracle. It should
call the HF image processor for visual payloads, reconstruct the exact Qwen
processor text locally using the same image-token expansion rule, call the
tokenizer directly for `input_ids` and offsets, and keep a parity test against
`Qwen3VLProcessor.__call__`. This preserves HF image/tokenizer behavior while
making the text space used for supervision explicit.

The local split path is parity-tested against `Qwen3VLProcessor.__call__` on the
defining smoke fixture and a small targeted set. The comparison covers
`input_ids`, `image_grid_thw`, image-placeholder expansion, and the relevant
masks/fields used by Qwen forward construction.

The owned text-expansion record is `ProcessorTextPlan`. It contains the original
rendered chat text, the final processor text, image-token expansion records, and
char-offset conversion helpers. The rendered-chat to processor-text conversion
is stored as piecewise range mappings plus explicit inserted/repeated
image-token expansion records, not as a single shift delta. This keeps the V1
single-image path simple while leaving the representation honest enough to
debug future prompt or placeholder changes.

V1 requires exactly one image reference and exactly one prompt-side image
placeholder. Zero-image, multi-image, and multi-placeholder examples fail at
encoding. This is stricter than the underlying Qwen processor, but matches the
approved V1 single-image scope and protects the first alignment implementation.

The smoke fixture's future `expected_tokenization.json` should include token
ids, token strings, selected offsets, processor-text hash, placeholder counts,
`image_grid_thw`, visual payload shapes/dtypes, optional image or payload
hashes, aligned spans, and local `TokenSequence` atoms. It must not dump full
`pixel_values`, and it should not become a giant dump of expanded processor text
unless the tiny fixture remains readable enough for that to be useful.

The token-identity section of `reports/qwen_setup.json` should be compact by default: coordinate-token
range checks, tokenizer/vocab identity, stable hash over the full coordinate-id
mapping, and sampled ids such as 0, 1, 123, and 999. A full 1000-token dump is a
debug option, not the default receipt content.

Supervised-span round-trip checks are exact. If decoded token-span text differs
from the processor-text slice under `skip_special_tokens=false` and tokenizer
cleanup disabled, encoding fails. Do not tolerate "benign" whitespace cleanup
differences in loss-bearing spans.

Qwen component loading lives in `src/qwen/loading.py` and returns a typed
`QwenComponents` object. Loading is Qwen-specific because it must own model,
processor, tokenizer, processor identity, config validation, fingerprinting,
processor default resize-policy recording, no-resize call-policy checks, and
narrow future patch points. Commands and training factories should call this
loader rather than constructing Qwen objects inline.

The public Qwen setup entrypoint is:

```python
load_qwen_components(config) -> QwenComponents
```

It is one transparent setup entry, not a pile of public calls wired manually by
`train.py` or `SupervisedTrainer`. Internally it runs named setup phases:
load the base model/processor/tokenizer, validate tokenizer and processor
identity, record processor default resize policy, validate call-time
`do_resize=False` support and single-image support, load an existing adapter or
initialize the configured adapter, install special-token embedding trainables,
prepare setup receipts, and then hand the model to optimizer construction.
Optimizer construction must happen after all trainable Qwen surfaces exist.

`QwenComponents` is frozen after successful setup. It contains the model,
processor, tokenizer, model config, tokenizer/vocab identity, processor policy,
loaded base identity, optional adapter identity, special-token embedding handle,
token vocab group resolver or resolved handle, and setup receipt handles. Later
modules receive it as read-only setup state rather than rediscovering or
mutating Qwen identity.

Model loading has two first-class modes from V1:

- base model only;
- base model plus optional adapter checkpoint.

Use `model.base_model` for the required base checkpoint and top-level
`adapter:` for adapter tuning or adapter-checkpoint composition. The base model
is required and should normally be authored directly as a
`model_cache/...` path. Do not add a configurable `model_cache_root` in V1.
The expected operating convention is that training is launched from
`/data/CoordExp`; if the path is wrong, ordinary Python/filesystem errors are
acceptable and should not be hidden behind over-designed path indirection.

`adapter.path`, when present, means "load an existing adapter
checkpoint." Adapter initialization is driven by the adapter-tuning config: if
LoRA or DoRA tuning is enabled and no adapter path is provided, initialize a
fresh adapter automatically. Do not require a separate explicit `init` flag in
V1. Path presence means load; path absence plus enabled tuning means initialize.

The public adapter schema lives only under top-level `adapter:`. It should use
`adapter.type`, `adapter.path`, `adapter.target_towers`,
`adapter.target_modules`, `adapter.rank`, `adapter.alpha`, `adapter.dropout`,
and `adapter.bias`. Keep `model:` for base-model loading, tokenizer/processor
policy, and Qwen identity. Do not accept `model.adapter.*` in V1 configs.

Adapters are optional and represent tuned checkpoint or adapter weights layered
on top of the base model. Base-only and base-plus-adapter modes must be
supported by the same Qwen loader rather than by separate training/inference
branches. Run artifacts must record the resolved base model identity, adapter
type, adapter target policy, and, when present, the resolved adapter checkpoint
identity.

V1 adapter support should include PEFT-style LoRA adapter directories and the
project's DoRA path. The adapter schema should be explicit and object-shaped,
with fields such as `type`, optional `path`, `target_towers`, and
`target_modules`. Standard LoRA must support targeting all linear modules
across the Qwen3-VL vision, aligner, and language towers. Target selection must
be explicit; do not silently default to all modules. Loading should fail fast
when an adapter is configured but missing, incompatible with the base model, or
ambiguous.

Any adapter target that the schema accepts is a support promise, not a hopeful
best-effort path. If V1 supports `vision`, `aligner`, or `language` as
`target_towers`, it must guarantee target discovery, adapter injection,
forward/backward participation when trainable, optimizer grouping, and artifact
recording for that tower. If a tower is not verified, reject it in schema rather
than accepting it silently.

The intended V1 adapter support matrix is full support for both `lora` and
`dora` over `target_towers: [language]`, `[aligner]`, `[vision]`, and explicit
combinations of those towers. A tower is considered supported only after
verification for each adapter type covers:

- module target receipt;
- trainable parameter summary;
- one forward/backward smoke;
- optimizer group coverage;
- checkpoint metadata check.

Combination support should be verified through each single tower independently,
plus one combined smoke for `[vision, aligner, language]`. Single-tower checks
localize target-matching failures; the all-tower smoke proves composition.

Adapter target discovery writes `adapter_targets.json` when LoRA or DoRA is
enabled. Keep this separate from `optimizer_groups.json`, because target discovery is a
model-surgery trust gate before optimizer grouping.
The receipt should prove adapter type, base model identity, target towers,
target module policy, matched module names, injected parameter names, trainable
counts, frozen counts, unsupported or missing targets, and verification status
per tower.

Adapter support tests live in `tests/qwen/test_adapters.py`, with executable
smoke configs or fixtures under:

```text
tests/fixtures/smoke/qwen3_vl_single_image_pack/
```

Adapter construction and injection live in `src/qwen/adapters.py`, because it
is Qwen-specific model surgery that depends on tower names and module
structure. `optim/` consumes the resulting trainable parameters and LR group
metadata; it does not perform adapter injection. `training/` decides when to
train, but it does not own model surgery.

DoRA is the intended first-class adapter type behind explicit `type: dora`.
The source study found no separate local, MS-Swift, or PEFT mechanism named
`dlora`; V1 therefore maps the approved `dora` adapter to upstream
DoRA/weight-decomposed LoRA via PEFT `use_dora`. Do not accept `dlora` as an
alias, do not treat it as an unnamed LoRA variant, do not invent a new
CoordExp-owned decomposed or dynamic LoRA mechanism during implementation, and
do not silently fall back from DoRA to standard LoRA. The repo's default tuning
recipe should use DoRA after this definition gate passes, not standard LoRA,
but this should be explicit in runnable configs rather than hidden as a code
default.

Before implementing DoRA, inspect MS-Swift, Transformers, and PEFT as initial
guidance. Local source roots observed during design:

- MS-Swift: `/data/ms-swift/swift`
- Transformers: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`
- PEFT: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/peft`

The final DoRA implementation should still be repo-owned and approval-card
driven. The inspection step is for behavioral guidance, module targeting,
checkpoint format, and avoiding avoidable incompatibilities, not for inheriting
MS-Swift's broader framework shape.

This inspection should produce a dedicated durable note before the DoRA
approval card, for example:

```text
docs/architecture/proposals/2026-06-27-coordexp-swift/source-studies/dora.md
```

Parallel inspection lanes should cover MS-Swift adapter/tuner behavior,
Transformers/PEFT adapter mechanics, and Qwen3-VL module naming plus target
matching. Do not create a placeholder study file; create it when the inspection
has real findings.

`adapter.type: dora` is not implementation-ready until that source study is
accepted and the adapter card records the definition, exact target discovery,
parameter names, optimizer grouping, checkpoint payload layout, metadata, and
load validation. Before that gate, runnable configs that request `dora` should
fail validation rather than silently becoming standard LoRA or a partial
placeholder. Runnable configs that request `dlora` should fail validation as an
unsupported V1 spelling and point to `adapter.type: dora`. The first
adapter-enabled smoke is a DoRA smoke, so the DoRA definition/source study and
minimal DoRA round-trip probe are prerequisites for running it. A pre-DoRA
base-only or standard-LoRA smoke is a separate user decision, not the default
implied by this proposal.

The first DoRA smoke trains only DoRA adapter parameters. It does not train
full base-model parameters from the vision, aligner, or language towers.

Full base-model parameter training is not implemented in this repo's V1 scope.
It is intentionally marked not implemented because it is not affordable for the
intended use. The trainable model surface is LoRA/DoRA adapter parameters plus
explicit special-token embedding parameters. Any future repo-added trainable
modules would need separate approval and explicit LR groups.

The normal recipe freezes the vision tower, may freeze the aligner explicitly,
and mainly injects DoRA over `all_linear` modules in the language tower. The
first smoke should use `target_towers: [language]` and
`target_modules: all_linear`. This exercises adapter injection, optimizer
grouping, backward, metrics, and checkpoint metadata without full base-model
memory pressure.

Production training is expected to use language-tower DoRA first, but this is
only the default recipe. The implementation must still make the full supported
adapter-target matrix reliable. Vision and MLP aligner targets should not be
treated as secondary or speculative if they are exposed in config.

Adapter support may land incrementally by adapter type and tower, but the config
schema must expose only verified combinations. A pair such as
`adapter.type: dora` plus `target_towers: [vision]` is accepted only after its
target discovery, forward/backward, optimizer grouping, receipt, and load
round-trip checks pass. Unverified combinations fail fast as unsupported rather
than running a best-effort match.

If `adapter.type` is `lora` or `dora`, `path` is absent, and the run is
in inference/eval-only mode, fail fast. Path absence plus adapter type means
fresh adapter initialization for tuning, not evaluation of a random adapter.

Do not support user-authored adapter target exclusions in V1.
`target_modules: all_linear` means all matched internal linear modules inside
the explicit `target_towers`, excluding output heads such as `lm_head` by
policy. Add user-authored exclusion support only after a real need appears.

Use repo-clean adapter hyperparameter names in config: `rank`, `alpha`,
`dropout`, and `bias`. Translate internally to PEFT/MS-Swift names such as
`r`, `lora_alpha`, or `lora_dropout` only inside adapter construction code.

Do not add a `model.base_trainable_towers` field in V1. Full base-parameter
training is not a supported path, so the schema should reject base-trainability
fields rather than accepting a field that can only be empty.

Special-token embedding tuning is a first-class trainable surface, separate
from LoRA/DoRA. Use "embedding" in public names rather than "row" when naming
this surface. The config surface is `model.special_token_embeddings`. The
public optimizer group remains `token_embeddings`. The config surface must be
explicitly present in every training config; standard profile YAMLs should
include it. Missing `model.special_token_embeddings` is a fail-fast training
config error rather than an implicit code default. Do not add an `enabled`
field in V1; presence means enabled, and absence is invalid for training.
Do not move this surface under `adapter:` and do not introduce a generic
`trainables:` config section in V1. Special-token embedding tuning is model
surgery over Qwen embedding/head parameters, not an adapter family.
Read-only non-training utilities, such as tokenizer setup inspection or a
base-only forward-contract probe, may omit `model.special_token_embeddings`
because they are not constructing a trainable surface.

The special-token embedding mechanism trains compact selected-token additive
deltas over the loaded base model's corresponding embedding/head rows. The base
model/tokenizer preprocessing already provides the coordinate and wrapper token
rows; runtime Qwen setup installs zero-initialized trainable deltas over those
frozen rows. This setup-time model surgery is not a `TrainRuntime`
responsibility. After setup, these deltas are ordinary model-side parameters
assigned to the `token_embeddings` optimizer group. This is selected-token full
embedding tuning, not LoRA over the embedding/head. Do not directly unfreeze
full embedding/head matrices in V1 and do not rely on masked gradients over full
matrices as the default training mechanism. Trainable selected-token tensors
default to the base embedding dtype; do not add a dtype knob unless later
evidence requires it. Checkpoints should save compact additive deltas relative
to the validated base rows, so runtime loading remains base model plus compact
trainable state rather than a full merged embedding/head export.

Before implementing the special-token embedding surface, perform a short source
study and record the decision in the module card. The study must compare a
custom input/output wrapper pair owned under `src/qwen/`, PEFT
`TrainableTokensConfig` / `TrainableTokensModel`, and LoRA
`trainable_token_indices` when LoRA is already present. The chosen mechanism
must pass the same contract either way: selected token rows in the base model
are validated, trainable deltas are zero-initialized over those frozen rows,
base embedding/head matrices remain frozen, tied-head models update both input
lookup and selected output-logit columns, untied models save separate
input/output deltas, checkpoint payloads remain compact, and loading validates
token strings and ids. If PEFT trainable tokens are used, the implementation
must explicitly avoid PEFT's automatic full embedding save behavior when
base-model vocab metadata differs from the live CoordExp coordinate-token
vocabulary; full embedding/head matrix export is not a valid V1 checkpoint
format.

Special-token embedding tying is internal and automatic, not a user-facing
config knob. Detect the actual loaded model state by parameter identity and
shape:

- tied embedding/head models use one shared selected-token embedding trainable
  and save it as `shared_embed_delta`;
- untied embedding/head models use `input_embed_delta` and
  `output_embed_delta`.

This keeps 2B/4B tied-head behavior simple while preserving a correct path for
8B, 30B, and other untied models. The detected policy, tensor names, and tensor
shapes must be recorded in `special_token_embeddings.json` and in
checkpoint metadata.

The V1 special-token embedding set is fixed and enabled for all training
configs. It does not depend on the active template variant, even when a
template happens not to emit every wrapper token. This avoids template-specific
trainable-surface drift. V1 does not support user-authored token list overrides,
extensions, or wrapper-token disabling.

The first special-token embedding config has one surface with two named groups:

- `coordinate_tokens`: `<|coord_0|>` through `<|coord_999|>`;
- `wrapper_tokens`:
  - `<|object_ref_start|>`;
  - `<|object_ref_end|>`;
  - `<|box_start|>`;
  - `<|box_end|>`.

This grouping keeps coordinate geometry and structural wrapper tokens visible
for diagnostics while preserving one trainable embedding surface.

Authored YAML should use named fixed groups rather than repeating 1000
coordinate token strings in every config, for example:

```yaml
model:
  special_token_embeddings:
    groups:
      coordinate_tokens: default_coord_0_999
      wrapper_tokens: default_object_box_wrappers
```

The resolved config must expand these symbolic defaults into concrete token
strings and tokenizer ids, so the run is self-contained for humans and agents.

All listed special tokens must already exist in the tokenizer before training
starts. Missing tokens are fail-fast errors; do not mutate the tokenizer or
resize embeddings automatically in V1.

After the DoRA source-study gate is satisfied, the first DoRA smoke should
train DoRA plus special-token embeddings, with separate explicit optimizer
groups. `optimizer.groups.token_embeddings` must provide one LR and weight
decay for the whole special-token embedding surface. Do not reuse the adapter
LR. Do not split coordinate-token and wrapper-token LRs in V1.

The smoke fixture/template must include at least one coordinate token and all
four wrapper tokens so both `coordinate_tokens` and `wrapper_tokens` receive
nonzero gradient coverage. Merely marking the parameters trainable is not
enough. Nonzero gradient coverage should be enforced as smoke-only assertions
and may emit group-level debug metrics such as
`grad_norm/token_embeddings/coordinate_tokens` and
`grad_norm/token_embeddings/wrapper_tokens`. Do not log per-token gradient
norms by default.

Implementation lives in `src/qwen/special_token_embeddings.py`, because this
surface attaches to Qwen embedding/head modules and depends on tokenizer/model
identity. It is not owned by `training/` or `runtime/`. `TrainRuntime` must not
handle token embeddings specially; it sees only model parameters that were
already installed by Qwen setup.

The special-token embedding surface writes
`special_token_embeddings.json` during model/tokenizer setup before the
first forward. It is a preflight trust gate, not only a checkpoint-time record.
At minimum it records token groups, token strings, token ids, detected
tied/untied status, base parameter names and shapes, trainable tensor names and
shapes, trainable counts, optimizer group assignment, checkpoint tensor keys,
and tokenizer-identity validation. Checkpoint loading must strictly validate
token identity: every saved token string and token id must match the current
tokenizer or loading fails.

Keep `token_type_vocab.json` and
`special_token_embeddings.json` separate. The token-type receipt proves
loss vocabulary legality; the special-token embedding receipt proves trainable
embedding setup and checkpoint state. They may cross-reference token strings
and ids, but they answer different trust-gate questions.

Wrapper tokens remain supervised by the base CE objective everywhere they
appear. They are structural tokens and should provide a direct learning signal
for their special-token embeddings when emitted by the active template.

The main encoding surface is `QwenExampleEncoder.encode(rendered:
RenderedExample) -> EncodedExample` in `src/qwen/encoding.py`. A small encoder
object is preferred over a free function because it owns processor/tokenizer
identity, image policy, alignment diagnostics, and Qwen config without turning
packing into a Qwen-specific module.

The Qwen encoder opens/loads the image from the `RenderedExample` image
reference, validates image identity and size metadata when available, and
passes the image to the HF processor. `data/` validates paths and metadata, but
does not pass live image objects forward. `templates/` remains pure text.

`do_resize=False` at the actual image-processor call is the hard V1 contract;
the loaded model artifact's processor default may still be resize-enabled. Qwen
setup records both `processor_default_do_resize` and `call_do_resize`. Encoding
must verify an equivalent no-resize call path and reject images that cannot pass
the installed Qwen processor without resizing. For the current fast processor
family this means preflighting `patch_size`, `merge_size`, declared image size,
decoded image size, and no-resize grid admissibility before calling into a path
that would otherwise raise an opaque upstream reshape error. Unless a later
approved probe proves a broader rule, width and height must be compatible with
the no-resize patch/merge grid, effectively divisible by
`patch_size * merge_size`. The receipt records `image_grid_thw` and whether
resize was actually bypassed.

V1 assumes images have already been prepared for the no-resize Qwen path. In
practice the first data and smoke surfaces should use offline-rescaled images
whose decoded width and height are compatible with the current
`patch_size * merge_size` rule, such as the existing `rescale_32` data family.
Raw arbitrary-size images are out of scope unless a later approved data policy
adds explicit pad-or-reject behavior. Because the no-resize path also bypasses
Qwen's smart-resize pixel ceiling, encoding preflight should enforce an
approved pixel or visual-token budget before processor execution rather than
letting a valid-but-huge divisible image exhaust memory.

Pack admission and visual-token counts must come from the actual no-resize
processor payload, especially `image_grid_thw`, or from a local computation
proven equal to that payload under the same no-resize policy. Do not use
`Qwen3VLProcessor._get_num_multimodal_tokens(...)`,
`get_number_of_image_patches(...)`, or other helper paths that may apply
smart-resize assumptions as the V1 source of truth for no-resize pack cost
unless an installed-version probe explicitly approves that helper under
`do_resize=False`.

Qwen processor outputs should be converted from HF `BatchFeature` or raw
processor dictionaries into typed CoordExp objects, while preserving raw key
diagnostics for debugging. `EncodedExample` should not expose arbitrary HF
kwargs as its public contract, but it should carry enough trace metadata to
debug upstream Qwen/Transformers changes.

V1 is single-image only. If processor output or raw data introduces
`pixel_values_videos`, `video_grid_thw`, or any other video payload, the Qwen
encoding/forward boundary should fail with an explicit "video not implemented"
error. Do not pass video fields through untested, and do not silently drop them.

For packed supervised training, CoordExp-swift should supply explicit
Qwen-compatible `position_ids`. Packing owns segment isolation, so relying on
HF Qwen's global `get_rope_index` inference over the packed sequence is too
easy to make subtly wrong. The Qwen package should provide validated helpers
for constructing the position ids needed by the Transformers Qwen3-VL forward
path from the packed layout and Qwen grid metadata.

At the HF Qwen3-VL model boundary, packed `position_ids` use Qwen's 4-row shape
`[4, batch, seq]`: row 0 is text position ids, and rows 1-3 are temporal,
height, and width MRoPE ids. The internal CoordExp-swift representation may
store separate `text_position_ids` with shape `[batch, seq]` plus 3-row
`mrope_position_ids` with shape `[3, batch, seq]`, but the Qwen bridge must
concatenate them into the 4-row HF form before forward. The text-position reset
points must match `cu_seq_lens_q[:-1]` for a flattened batch-size-1 packed row.
Per-example position ids may be concatenated only after each segment's ids have
been computed from that segment's actual expanded `input_ids`, `image_grid_thw`,
optional video-grid inputs when later supported, and masks.
Use MS-Swift's Qwen-VL packing/position-id flow as a reference during
implementation: compute MRoPE positions per original sample/segment, concatenate
the resulting channels only after per-segment computation, and verify the final
HF Qwen shape and reset points. This is a guidance/reference relationship, not
a dependency or permission to import MS-Swift training abstractions.

Qwen forward adaptation lives in `src/qwen/forward.py`. It should expose one
small wrapper that calls the HF model with `QwenForwardInputs` and returns typed
`ModelOutputs`. The wrapper owns labels-disabled checks, required output flags,
dtype/device sanity, and narrow future Qwen hooks. It rejects non-`None`
`labels` and asserts that model-side loss is absent or ignored, because
CoordExp-swift loss is computed only through `LossRunner`.

V1 training forward uses batch size 1 with one long packed row:
`input_ids.shape == [1, seq]`. Multiple examples are represented as isolated
segments inside that row, not as a padded batch dimension. Any later true
batching surface belongs to rollout/decoding or a separately approved training
profile.

Default training forward kwargs are fixed and conservative:

- `labels=None`;
- `use_cache=False`;
- full-vocabulary logits requested for either the full sequence or explicitly
  selected supervised causal rows;
- output object/dataclass shape is asserted by the wrapper; do not require a
  rote `return_dict=True` kwarg when the installed forward signature does not
  declare it;
- no hidden states unless an approved hidden-state loss hook requires them.

Do not mirror HF generation defaults, and do not expose arbitrary HF forward
kwargs as stable config in V1.

Enabled loss terms declare their model-output requirements before forward. A
loss setup/planning step aggregates those declarations into a single
`ForwardRequirements`-style contract consumed by `src/qwen/forward.py`. V1
protected defaults, `BaseTokenCE` and `TokenTypeGateLoss`, request logits only.
Hidden states, attentions, vision states, or other optional outputs are opt-in:
they are requested only when an enabled approved loss declares them. Do not
expose manual config flags such as `output_hidden_states` as the primary way to
satisfy loss requirements, because that can silently drift from the active loss
plan.

Activation requirement names must distinguish text-token activations from
future visual activations. Use `text_hidden_states` for language/text tower
activations aligned to packed token positions. Reserve `vision_activations` for
a future explicitly designed visual hook; it is not implemented in V1. If a
config enables a loss requiring `vision_activations` before that hook exists,
loss setup fails before training starts. Do not treat image placeholder token
positions as visual-row positions: placeholders live in the language sequence,
whereas vision activations live in Qwen visual grid/feature spaces.

When visual activation support is later designed, the capture point must be
named explicitly, for example `vision_pre_projector`, `vision_post_projector`,
or `post_scatter_language_embedding`. Avoid broad labels such as "vision hidden
states" because they are too ambiguous for research claims. The first vertical
smoke proves unsupported visual activation requirements fail cleanly; it does
not capture visual activations. `loss_plan.json` records activation
requirements by source, including logits, text hidden-state selectors, and
whether vision activations were unsupported/requested.

Naming discipline matters for these activation contracts. Use `position` for
sequence-token indices, `span` for contiguous token groups, and reserve `row`
for a true JSONL row or a real vision feature row. Future visual features should
use names such as `feature_index` or `grid_index` rather than casual "row"
language unless a row is literally the coordinate system being modeled.

Text hidden-state selectors use explicit semantic capture names plus explicit
layer indices when needed; do not use regex for activation selectors. The first
approved text capture is `lm_head_input`: the post-final-norm text hidden state
consumed by `lm_head`. In the local Transformers `4.57.1` Qwen3-VL source,
`Qwen3VLTextModel.forward` applies `self.norm(hidden_states)` before returning
`last_hidden_state`, and `Qwen3VLForConditionalGeneration.forward` feeds that
output to `lm_head`. This makes `lm_head_input` precise and makes casual names
such as `final` or `last` too ambiguous for stable selector syntax.

If future losses need pre-final-norm decoder activations, introduce
`decoder_layer_post` with an explicit `layer: -1` or `layer: N` selector and a
wrapper-owned, smoke-tested capture hook. Do not assume
`output_hidden_states=True` returns the required layer tuple for installed
Qwen3-VL. Regex-style matching remains appropriate for module/parameter
selection surfaces such as LoRA targets, LR groups, freeze/trainable parameter
sets, and extra-state checkpoint keys, following the spirit of MS-Swift/PEFT;
it is not the hidden-state activation selector language. `loss_plan.json`
records both the semantic selector and implementation capture path, and invalid
selector names, unsupported captures, out-of-range layer indices, or empty
selected positions fail before training.

The Qwen forward wrapper must reject requirement/output mismatches immediately.
If an enabled loss declares a required output and `ModelOutputs` does not carry
it, fail with the loss name and missing requirement rather than returning a
zero/skipped loss. This keeps loss wiring failures separate from valid
zero-eligible auxiliary terms.

The core Transformers Qwen3-VL forward path to wrap is:

```text
Qwen3VLProcessor.__call__
  -> input_ids, attention_mask, pixel_values, image_grid_thw,
     optional pixel_values_videos/video_grid_thw
Qwen3VLForConditionalGeneration.forward
  -> Qwen3VLModel.forward
  -> lm_head
  -> optional HF loss only when labels are passed
Qwen3VLModel.forward
  -> text input embeddings
  -> visual features from Qwen3VLVisionModel
  -> placeholder-count validation
  -> masked_scatter visual embeddings into language embeddings
  -> MRoPE position_ids / rope_deltas
  -> Qwen3VLTextModel.forward
Qwen3VLTextModel.forward
  -> causal mask
  -> decoder layers / optional hidden states
  -> final norm
```

Wrapper invariants:

- load and wrap `Qwen3VLForConditionalGeneration`; do not edit installed
  `modeling_qwen3_vl.py`;
- pass exactly one of `input_ids` or `inputs_embeds`;
- V1 supervised training uses the `input_ids` plus visual payload path, not a
  precomputed `inputs_embeds` shortcut. The installed Qwen3-VL path injects
  DeepStack visual features through model internals; an `inputs_embeds` shortcut
  can bypass that behavior unless the wrapper also reproduces the corresponding
  visual masks and deepstack payload. Treat such shortcuts as unsupported until
  a dedicated hook/card proves them.
- preserve processor-derived `image_grid_thw` and ordered placeholder
  occurrence; do not reconstruct visual grids from image size after the fact;
- validate visual placeholder token count against
  `sum(prod(image_grid_thw) // spatial_merge_size**2)` before forward and let
  the HF placeholder mask validation remain a second guard;
- preserve Qwen3-VL MRoPE semantics for `position_ids`, `rope_deltas`, and
  generation cache behavior;
- for supervised training, pass explicit packed `position_ids` and set
  `use_cache=False`; cache and `rope_deltas` reuse are generation/inference
  concerns, not the standard training path;
- when explicit packed `position_ids` are passed, treat returned
  `rope_deltas` as non-authoritative for training. The wrapper should not
  require `rope_deltas` in training smoke checks, should not expose stale
  `model.model.rope_deltas` as current-batch truth, and should clear or reject
  stale cache-derived values when needed;
- if padding-free/packed FlashAttention is used for the text tower, pass
  `cu_seq_lens_q/k` and `max_length_q/k` through the HF attention kwargs
  convention; the installed Qwen3-VL implementation computes varlen metadata
  internally only for the vision tower;
- never pass `labels` in production training; HF loss is a vanilla CE fallback,
  while CoordExp-swift consumes logits and explicitly approved optional outputs
  through `LossRunner`;
- request full-vocabulary logits for V1 supervised training. Do not use
  suffix-style `logits_to_keep=1`, because supervised atoms may appear across
  the whole packed sequence. A compact selected-row path is allowed only when
  it passes explicit physical positions and `LossContext` validates that every
  supervised atom is covered;
- treat hidden-state capture as deferred in V1 unless a wrapper-owned hook is
  explicitly approved and shape-tested. The installed public Qwen3-VL forward
  path should not be assumed to return usable `hidden_states` merely because
  `output_hidden_states=True` is requested;
- do not rely on an `image_hidden_states` output field in V1. If visual feature
  diagnostics or losses are later approved, hook `get_image_features`,
  `get_video_features`, `Qwen3VLVisionModel.forward`, or the post-replacement
  embedding seam explicitly.

`QwenForwardInputs` should carry the full model-forward contract needed by
packed Qwen3-VL training, including packed `input_ids`, visual processor
payloads, `image_grid_thw`, explicit `position_ids`, FlashAttention varlen
metadata, Qwen-only kwargs, and trace metadata. These tensors are part of the
Qwen forward contract, not trainer-private state. `QwenForwardInputs` is typed;
it is not a generic `dict[str, Tensor]` copied from HF examples.

Qwen `position_ids` are built in `src/qwen/forward.py` or a close helper under
`src/qwen/`, derived from `PackedSequence`, `PackedSegment` boundaries, expanded
`input_ids`, and `image_grid_thw`. `packing/` owns physical boundaries, but
Qwen owns MRoPE semantics. Do not let HF infer packed training position ids.

For V1 packed training, FlashAttention segment isolation is passed through
`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k` derived
from `PackedSegment` boundaries. Validate dtype, shape, start/end values, and
Python-int max lengths before calling the model. Do not encode segment
boundaries with an ordinary dense 2D attention mask, and do not rely on position
resets alone.

CoordExp-swift should not replace visual embeddings itself in V1. Transformers
Qwen3-VL owns visual feature extraction and the `masked_scatter` replacement of
visual embeddings into language embeddings. CoordExp-swift validates
placeholders, grids, ordering, dtype/device expectations, and may add narrow
diagnostic hooks, but it should not reimplement the replacement path.

The first Qwen forward smoke should assert the real wrapper contract on a
single-image example: image placeholder count matches grid-derived visual rows,
`labels=None` at the model boundary, non-`None` labels are rejected, model-side
`loss is None`, full-vocabulary logits exist for either the full input sequence
or explicit supervised physical positions, `use_cache=False`,
`outputs.past_key_values is None`, and no model-side CE path is used. The smoke
must not require `rope_deltas` for explicit-position training, and must treat
hidden-state capture as unsupported/deferred unless an approved wrapper hook is
present. For packed FA2, the smoke should also assert that explicit
`cu_seq_lens_q/k` and Python-int `max_length_q/k` reach the HF attention path
and, for the installed Transformers version, reach the padding-free/varlen text
decoder branch rather than the padded-unpad branch. It also asserts that no
ordinary 2D padding mask with zeros is used to encode segment boundaries. A
smoke that merely checks "forward does not crash" is insufficient.

The first implementation wave should build the shifted-loss-consumption parity
fixture before building the full `SupervisedTrainer`. Trainer work may proceed
only after the fixture proves the Qwen template, dense labels, `TokenAtom`
positions, `LossContext.logits_position`, full-vocabulary logit-row selection, and image
placeholder masking agree at the model/loss boundary.

Qwen smoke, failure, and explicit debug runs should write
`run_dir/debug/qwen_forward_contract.json`. The receipt should include
input shape, logits shape, `labels` status, `use_cache` status,
`past_key_values` status, `logits_to_keep`/physical-position map, image
placeholder counts and ranges, `image_grid_thw`, expected visual token count,
visual payload shapes,
HF-boundary `position_ids` shape, segment boundaries, `text_position_ids` reset
points, `cu_seq_lens_q/k`, `max_length_q/k`, attention-mask handling, and any
disabled or deferred features such as hidden-state capture.

Char-span to token-span alignment should prefer tokenizer offset mapping when
available, but must align in processor-text space after Qwen image-placeholder
expansion. When the Qwen/chat-template path does not expose usable offsets, V1
may use a deterministic prefix-tokenization fallback over processor text with
strict validation. Approximate post-tokenization string matching is not a V1
strategy.

Placeholder validation happens twice. `qwen/` validates image placeholder counts
against Qwen grid/features per encoded example. `packing/` validates the same
relationship after packing and placement. The two validation layers catch
different classes of bugs and should produce local diagnostics.

`EncodedExample` is the atomic pre-pack unit. It contains CPU-list token ids,
typed Qwen visual feature inputs and grid metadata, local `TokenSequence` in
logical positions, effective pack-cost fields, and trace metadata. It is not a
loose HF kwargs dict. One `EncodedExample` must fit within `global_max_length`
by itself. If it exceeds the budget, the default behavior is fail-fast with
`EncodingContractError` and a clear diagnostic; V1 must not silently truncate or
auto-split one example across multiple packs.

Concretely, `EncodedExample` contains per-example `input_ids: list[int]`, local
token supervision, `QwenProcessorPayload`, `image_grid_thw`, placeholder counts,
feature-row cost, span-to-token alignment diagnostics, rendered-chat to
processor-text offset mapping metadata, source/render provenance, Qwen trace
metadata, and the explicit `RenderedSpan -> TokenSpan -> TokenAtom` trace. It
should not contain a fully assembled batch/pack dimension.

`packing/` assembles model-agnostic physical layout, then calls Qwen helper
functions to produce `QwenForwardInputs`, including packed `input_ids`,
`pixel_values`, `image_grid_thw`, position ids, and FlashAttention-related
metadata. `qwen/` owns model-family-specific tensor conventions; `packing/`
owns physical sequence layout; `training/` does not build model kwargs.

Upstream Qwen model behavior changes should be handled by local wrappers or
subclasses under `src/qwen/`, never by editing installed Transformers files.
Do not vendor the whole Qwen model immediately. Patch points should stay narrow,
documented, and motivated by transparency or research flexibility.

Span-to-token alignment happens during Qwen encoding, immediately after
tokenization. Alignment failures are encoding failures, not loss-runner
problems. The error should identify the example id, rendered span, surrounding
text, tokenizer/model identity, and the attempted token span.

`inspect-example` should expose the whole transformation without launching
training: `RawExample` summary, rendered chat/messages, image path and size,
token ids around important spans, `TokenSpan`s, `TokenAtom`s, image placeholder
counts, `image_grid_thw`, visual feature row counts, and effective pack cost.
In V1, this is a package-owned helper, test/debug capability, or smoke
diagnostic surface rather than a stable entry file. Do not add
`inspect_example.py` or an `inspect-example` command before the data/template
and Qwen encoding path exists and an explicit approval card promotes that
surface.

Encoding is strict. Missing image, invalid geometry, resize drift, unsupported
multi-image/video in V1, tokenizer span-alignment failure,
placeholder/grid mismatch, overlength example, and missing supervised atoms all
fail fast.

V1 should run correctly without persistent caches. Cache support is a future
optimization boundary, not a first-version subsystem. If a persistent cache is
introduced later, it must be typed, fingerprinted, inspectable, disposable, and
approved through a payload-specific design card.

Cache terminology is deliberately split:

- `model_generation_cache` means HF/past-key-value cache. It is forbidden in
  supervised packed training; Qwen forward uses `use_cache=False`.
- `loss_selection_cache` means memoizing selected logits or hidden states inside
  one loss step. It is not V1; selection is recomputed unless profiling proves a
  concrete duplicated cost.
- `activation_cache` means persisted or cross-step visual features, hidden
  states, logits, or model outputs. It is deferred and unsupported in V1.
- `pre_forward_persistent_cache` means deterministic artifacts before the model
  forward. Candidate stages are deferred:

```text
RawIndexCache
  -> JSONL row offsets, source fingerprints, optional image stat/checksum data
RenderedExampleCache
  -> rendered chat/messages and semantic spans
EncodedExampleCache
  -> token ids, logical TokenSequence, Qwen processor outputs, pack-cost fields
PackPlanCache
  -> greedy pack assignments and per-example cost summaries
```

These names are candidate stages, not approved V1 modules. A candidate graduates
only after its owner names one payload, one fingerprint contract, one replay
invariant, and one artifact/receipt contract. Important future fingerprint axes
include:

- raw example content hash and source dataset identity;
- image identity policy, such as path plus stat or checksum;
- template id/version, prompt language/version, and chat-template identity;
- `object_field_order` and `object_ordering`;
- bbox format;
- tokenizer/model id, tokenizer vocabulary/special-token identity, processor
  class/config, `processor_default_do_resize`, `call_do_resize`,
  `patch_size`, `merge_size`, no-resize grid policy, and Qwen settings;
- `global_max_length`, pack-cost model version, and packing policy version for
  pack-plan caches.

Do not add shared cache modes in V1. Future stage-specific caches may introduce
disabled/read-only/read-write/refresh/validate-only modes after their payload
contract is approved. Stale or incompatible entries are disposable; they should
be rejected or regenerated rather than adapted silently.

Do not cache Qwen visual-tower activations as the default V1 behavior. Those
activations depend on model weights, dtype, device, train/freeze policy, and
gradient requirements. Prior coverage-ledger and row-conditioned visual-coverage
work needed same-forward capture, visual-feature intervention, or faithful
row-boundary re-prefill, not a general training-time cache. Future visual-row or
coverage support belongs first under Qwen capture / loss / rollout design, and a
frozen-vision feature cache may be approved only after a parity and gradient
gate proves that reuse is safe.

## Overall Module Design

Keep the broad top-level packages, but make each package own a deep interface.
Do not split immediately into many narrow top-level packages such as
`encoding/`, `layout/`, `forward/`, or `inspection/`. Depth should come from
strong package-owned interfaces rather than from a large top-level folder list.

The golden rule is to avoid over-design. Add a named abstraction only when it
maps to a real execution stage, carries a contract that must be inspected or
validated, removes concrete duplication, or protects research meaning. Do not
introduce protocols, registries, factories, base classes, plugin systems, or
extra package layers just because they might be useful later. Start with the
smallest explicit object that makes the pipeline understandable.

No top-level `cache/` package is approved for V1. Stage modules may expose
future cache identities, but they own their own payload contracts until at least
two real cache stages need common policy. If that happens later, introduce
`src/cache/` as shared policy/fingerprint/receipt infrastructure rather than
as a place to store model tensors.

Approve core modules and classes in dataflow order:

```text
RawExample
RenderedExample
TokenAtom / TokenSpan / TokenSequence
EncodedExample
PackedLayout / PackedSequence
QwenForwardInputs
LossContext / LossRunner
MicroStep
```

`qwen/` should expose a deep `QwenExampleEncoder` interface plus typed records.
Internally it may use small helpers for processor calls, image handling,
tokenization, placeholder validation, and span alignment, but callers should
not orchestrate those Qwen details manually.

`packing/` should expose `PackedSequenceBuilder` as the main interface.
Internally it may split pack planning, layout construction, tensor assembly,
and validation, but training code should not call those pieces separately.

Entry files should stay thin. They parse arguments, resolve config, and call
package-owned builders or inspection helpers. Do not add a `commands/` package
in V1 unless repeated entry parsing or orchestration duplication earns that
layer. Inspection behavior should still be owned by the package whose state is
being inspected, for example `data.inspect`, `qwen.inspect`, `packing.inspect`,
and `losses.inspect`.

## Configuration And Reproducibility

`config/` owns typed config objects, YAML loading, inheritance, resolution,
defaulting, and validation. Package modules should receive already-resolved
typed config slices, not raw YAML dictionaries.

The config system should support current-style YAML inheritance from the start.
The initial design should still stay simple: small explicit YAML files, a
minimal inheritance/include mechanism, strict unknown-key validation, cycle
detection, and a fully flattened resolved-config artifact. Do not introduce a
large Hydra-style composition system in V1.

Use Pydantic v2 strict typed config models for validation. Dataclasses are a
credible fallback only if dependency constraints later force it, but Pydantic is
preferred because it gives nested validation,
strict unknown-key rejection, useful path-aware errors, defaults, and
self-contained serialization without building a custom validation framework.

YAML parsing should use a small repo-owned loader around PyYAML or the existing
YAML parser, then validate into the typed config models. Do not introduce Hydra
or OmegaConf in V1.

Config inheritance uses `extends:` as a single parent YAML path in V1. Each file
may extend at most one parent, but parent files may themselves extend another
parent, so the normal workflow can still be:

```text
configs/base.yaml
  -> configs/directions/<direction>/base.yaml
  -> configs/directions/<direction>/<run>.yaml
```

This supports a concise shared-runtime habit without introducing multiple-parent
precedence ambiguity. A root `base.yaml` can hold shared runtime defaults used
by nearly all training runs, such as precision, dataset workers, FlashAttention,
common debug/receipt settings, and the shared backbone/run skeleton. A
research-direction base can hold defaults for one project line. The concrete
training YAML should then contain only the small differences: hyperparameters,
learning-rate groups, effective pack/step settings, adapter choices, and loss
enabling/weights.

Normal reusable configs live under:

```text
configs/base.yaml
configs/directions/<direction>/base.yaml
configs/directions/<direction>/<run>.yaml
```

Do not put runnable config truth under `research/`, and do not clutter
`configs/` with direction names at the top level. Smoke configs are the one V1
exception: the defining smoke config lives beside its fixture under
`tests/fixtures/smoke/.../config.yaml`.

Parents are resolved first, then child files deep-merge dictionaries over the
parent result. Lists replace rather than append. This avoids quiet surprises in
loss terms, optimizer groups, datasets, schedules, and token lists. Inheritance
must detect cycles and report the config path chain in errors. Ordered lists of
parents and arbitrary include graphs are not V1 features.
`extends` is allowed only at the YAML file top level; nested section-level
extends and package-specific include keys are not V1 features.

Unknown config keys fail fast with path-aware errors. Do not warn-and-ignore
unknown keys and do not preserve unknown keys under an `extras` bucket.
Strict unknown-key handling applies everywhere, including backend-specific
subtrees such as `runtime.accelerate` and `runtime.deepspeed`. Add an explicit
`extra_args` surface only if a future backend integration proves it is needed.
Inherited fragments are partial and are not required to be runnable by
themselves. Only the final runnable root config requires `schema_version: 1` and
all required fields after inheritance is resolved.

Entrypoints resolve and validate config before loading fixture/data files,
opening model paths, or constructing model/processor objects. Config defaults
must not depend on inspecting the dataset.

Final resolved `TrainConfig` objects should be immutable/frozen. Modules receive
typed config slices for reading, not mutation. Runtime-derived values belong in
setup receipts, run manifests, or explicit runtime state, not by mutating the
resolved config object after validation.

Explicit YAML `null` is allowed only for fields typed as optional. It does not
delete inherited keys and is never silently ignored. If a child needs to replace
an inherited value, it must provide the replacement value with the correct type.

Inherited base files may use the string sentinel `REQUIRED` as an explicit
reminder for non-trivial hyperparameters that must be decided by a
direction-level or run-level config. This is especially important for learning
rates, effective batch choices, packing/global-length
choices, DoRA shape/config choices, run-length policy such as `training.epochs`,
optional debug `training.max_steps`, and other research-defining knobs. If any
`REQUIRED` value survives into the final merged runnable config, loading fails
with a `ConfigContractError` before training begins. `REQUIRED` is not a
default, is not converted to `None`, and is not valid in the resolved config
artifacts.

Root `configs/base.yaml` should define the backbone of the run: shared runtime
settings, shared model/loading defaults, common safety/debug/receipt settings,
and visible placeholder shapes for non-trivial knobs that every run must
decide. It should not define loss-weight placeholders by default. Loss modules,
loss weights, and other research-head choices belong in
`configs/directions/<direction>/base.yaml` or the concrete run YAML. In this
sense the root base defines the backbone, while direction/run inheritance
supplies the neck and head.

Relative data, fixture, cache, and reference paths resolve relative to the YAML
file that declares them. `run.artifact_root` remains literal or operator
cwd-relative by choice, because output placement is an operator decision rather
than a property of the config file location. The resolved config artifact should
store resolved paths as absolute paths or repo-root-normalized paths according
to the field's portability needs, but it must not depend on the process current
working directory for data/fixture inputs.

`schema_version` is useful, but should not become per-fragment ceremony. Require
`schema_version: 1` only on runnable root config files. Inherited fragments may
omit it and are validated in the context of the runnable root. Every run writes
`configs/resolved.yaml` and `configs/resolved.json`; these files contain the
fully resolved self-contained config and are the replay/debug source of truth.
The resolved config includes the effective schema version, config fingerprint,
and config schema/model version. It does not copy the authored inheritance
chain or preserve parent/child relative relationships by default.

`src/config/models.py` should contain the initial Pydantic models, using nested
section models inside one file until real size or ownership pressure justifies a
split. Do not start with one file per config section, and do not collapse the
schema into one giant flat `TrainConfig`.

`ResolvedTrainConfig` contains the frozen typed config plus compact resolution
metadata needed during loading and artifact writing: effective
`schema_version`, resolved config fingerprint, config loader/schema version, and
resolved path metadata. Run ids and package version reports belong to
run/artifact setup, not inside the config object. Do not preserve full original
YAML text, copy authored config files, or store the inheritance chain by
default; the final resolved config is the durable reproducibility handle.

The config module public API should stay narrow:

```python
load_train_config(path) -> ResolvedTrainConfig
write_resolved_config_artifacts(
    resolved_config, run_dir
) -> ResolvedConfigArtifacts
```

Internal loader, merge, path-origin, fingerprint, and Pydantic helper functions
should remain private until reuse earns promotion. Callers should not instantiate
Pydantic models directly from YAML dictionaries.

`ResolvedTrainConfig` should be a small named object, not a raw tuple. Writers
and manifests should consume its fingerprint, path-resolution, and schema/loader
metadata directly rather than recomputing those values later.
Generated run ids remain artifact/run setup metadata.

The config package should be split by responsibility, initially:

```text
src/config/models.py
src/config/loader.py
src/config/resolve.py
src/config/paths.py
src/config/fingerprint.py
src/config/writer.py
```

Avoid one large `config.py`, and avoid one file per config section unless the
schema later becomes large enough to justify that split.

Path handling should be two-stage: parse and merge raw YAML, validate enough
shape/type information to know which fields are paths, resolve those paths
according to the declaring YAML file, then validate into the final frozen
`ResolvedTrainConfig`. Do not heuristically rewrite arbitrary strings as paths,
and do not leave input paths ambiguous until runtime.

Config tests should include a tiny three-level inheritance fixture:

```text
configs/base.yaml
  -> configs/directions/<direction>/base.yaml
  -> configs/directions/<direction>/<run>.yaml
```

That fixture should cover `REQUIRED` override, list replacement, inheritance
merge behavior, path resolution, and final frozen config behavior.

Config loading and validation must stay CPU-light and dependency-light. It
should not import torch, Transformers, load models, inspect processors, or touch
CUDA. Qwen/model probing happens later through receipts such as
`reports/qwen_setup.json` and, for smoke/failure/debug mode,
`debug/qwen_forward_contract.json`.

Top-level train config sections are:

```yaml
schema_version: 1
run:
model:
adapter:
data:
template:
packing:
  global_max_length: ...
losses:
optimizer:
training:
runtime:
eval:
checkpoint:
debug:
```

`model` is the user-facing model/loading/processor section, even though Qwen
implementation code lives under `src/qwen/`. `adapter` is separate so LoRA,
DoRA, and adapter loading/initialization are not hidden inside `model`.
Selected special-token embedding tuning remains under
`model.special_token_embeddings`, because it is Qwen embedding/head model
surgery rather than an adapter family. Scheduler settings, when present, belong
under `optimizer`; artifact placement belongs under `run`; metric settings
belong under `training`, `eval`, or `debug` according to their use.

In training configs, `adapter:` is object-shaped when adapter tuning or adapter
checkpoint loading is expected. Do not use `adapter: null`; omit `adapter:` only
for explicitly base-only, non-training utilities such as tokenizer/Qwen setup
inspection or forward-contract probes that do not initialize adapter tuning.
Training configs that tune or load adapters must provide an object under
`adapter:`. V1 config validation should reject `model.adapter.*`.

The V1 trainer config surface uses:

```yaml
training:
  mode: supervised
  epochs: ...
  max_steps: null  # null for production epoch-led runs; integer for smoke/debug override
  effective_batch_size: ...
  max_grad_norm: ...
  precision: bf16  # bf16 | fp16 | fp32
  logging:
    every_fraction: ...
    steps: [...]
runtime:
  backend: single  # single | accelerate | deepspeed
  accelerate: ...
  deepspeed: ...
eval:
  forward:
    every_fraction: ...
    steps: [...]
  inference: ...
checkpoint:
  every_fraction: ...
  steps: [...]
  save_final: true
optimizer:
  ...
  scheduler:
    ...
```

`training.mode: supervised` names the training family in behavior terms and
leaves room for a future `training.mode: rollout`. V1 accepts only
`training.mode: supervised`; the field remains because it improves resolved
config and artifact readability and avoids later churn when rollout training
arrives. `training` owns orchestration knobs such as authored epoch budget,
optional debug step override, global effective batch size, clipping threshold,
precision, and logging cadence.
`runtime` owns backend selection, device/distributed placement, rank guards,
Accelerate settings, and DeepSpeed settings.
Keep the V1 runtime schema small. `runtime.backend: single` is a friendly
single-process config value, but implementation may still use the same minimal
Accelerate-backed `TrainRuntime` seam internally so single-GPU and distributed
paths do not drift. Basic Accelerate-backed single-node launches
should be configured through CoordExp-owned typed fields under
`runtime.accelerate`; they should not require a separate Accelerate config file
for the common path. DeepSpeed should be selected through
`runtime.backend: deepspeed` and a small `runtime.deepspeed` surface, including
`config_path` for the official DeepSpeed JSON/YAML config plus only
conflict-sensitive CoordExp fields. CoordExp validates batch-size and gradient
accumulation conflicts but does not re-model the entire DeepSpeed schema in
Pydantic V1.
`training.epochs` is the production run-length knob. `training.max_steps` is a
nullable smoke/debug override; production launch configs should set
`max_steps: null`. If `max_steps` is a positive integer, it has priority over
`epochs` and directly defines `resolved_max_steps`. If `max_steps` is null, the
concrete `resolved_max_steps` is computed from `training.epochs` and the
resolved packed train dataloader/cardinality before schedule materialization. If
that cardinality cannot be known for an epoch-led run, V1 should fail or run an
explicit preflight pack-count pass rather than starting a scheduled training run
with unknown length.
`training.epochs` is a positive integer in V1. Use `training.max_steps` for
fractional/debug run length. Do not introduce fractional epoch semantics until
there is a concrete research need and a new approval.
`training.effective_batch_size` is the user-facing global number of packed
sequences per optimizer update. Do not expose or accept authored
`training.grad_accum_steps`. Because supervised V1 uses one packed sequence per
rank per micro-step, runtime setup computes `resolved_grad_accum_steps` from
`effective_batch_size` and the actual training world size:

```text
resolved_grad_accum_steps = effective_batch_size / world_size
```

If the value is not an integer, or if `effective_batch_size` is smaller than the
active training world size, the run fails before training. Do not silently round,
pad, drop ranks, or change effective batch size. Record the resolved world size,
effective batch size, and `resolved_grad_accum_steps` in run metadata and
optimizer/training receipts.
`configs/resolved.yaml` and `configs/resolved.json` store resolved
operator-authored input truth such as
`training.effective_batch_size`, not runtime-derived accumulation. Runtime
receipts and `run_manifest.json` record launch-context truth such as world size,
effective batch size, and `resolved_grad_accum_steps`. For Accelerate or
DeepSpeed-backed launches, CoordExp-swift computes `resolved_grad_accum_steps`
and passes/injects it into the backend configuration. If a user-provided
DeepSpeed or Accelerate config also specifies a conflicting accumulation value,
fail fast rather than letting the backend become a second source of truth.

Epoch-led runs use deterministic tail-fill for incomplete final effective-batch
windows. Do not silently discard final packs, and do not perform a smaller
partial final optimizer update in V1. Compute the requested epoch-led pack
presentations from the resolved packed train stream and `training.epochs`; if the
final optimizer step would be incomplete, continue deterministically into the
next epoch/order stream just enough to complete that optimizer step. This makes
`epochs` a minimum full-pass target plus a bounded tail completion, not an exact
presentation count. The tail-fill count is always less than
`effective_batch_size`.

This intentionally differs from current HF `Trainer`, which allows a shorter
final accumulation window and rescales by the actual number of micro-batches,
and from the inspected MS-Swift surfaces, which mix automatic accumulation with
floor-style max-step calculation in one helper. CoordExp-swift chooses
deterministic tail-fill because it preserves constant optimizer-step math,
avoids source-order tail drop bias, and keeps the extra exposure bounded and
auditable. Record requested pack presentations, actual pack presentations,
`tail_fill_pack_count`, tail-fill pack/example ids when cheap, world size,
effective batch size, and `resolved_grad_accum_steps` in the run receipts.

`training.max_steps` debug/smoke mode also consumes a deterministic continuous
pack stream until `max_steps * effective_batch_size` global pack presentations
have been consumed. It should not create partial optimizer steps.

The train stream is defined first in global pack-presentation space, then
partitioned by rank. For each planned optimizer step, the stream owns exactly
`effective_batch_size` global pack presentations with monotonically assigned
`global_pack_presentation_id`s. Runtime world size partitions those
presentations into rank-local `MicroStep`s by a deterministic policy equivalent
to:

```text
rank = local_index_within_planned_step % world_size
rank_local_micro_step = local_index_within_planned_step // world_size
```

Each rank therefore receives exactly `resolved_grad_accum_steps` micro-steps per
planned step. Global pack ids may repeat only for intentional epoch replay or
tail-fill, and those repeats must carry replay/tail-fill metadata. `MicroStep`
records include `planned_step_id`, `rank`, `world_size`,
`rank_local_micro_step_id`, `global_pack_presentation_id`, the pack fingerprint
or compact pack id, and replay/tail-fill status. Distributed receipts must be
able to prove that every planned step contains exactly `effective_batch_size`
pack presentations across ranks and no unmarked duplicate presentation ids.

Epoch-led production runs must perform a deterministic preflight pack-count pass
before training starts. The preflight pass uses the same strict
data/render/encode/pack validation as training, including bad-sample failure
semantics. It does not have to materialize all tensors in memory, but it must
compute `packs_per_epoch`, requested pack presentations,
`actual_pack_presentations`, `tail_fill_pack_count`, `resolved_max_steps`, and
the resolved schedule inputs before the training loop begins. A bad sample found
during preflight fails the run before any optimizer state changes; it is not
skipped, estimated around, or deferred for training to rediscover.
This preflight is allowed to stream and discard heavy payloads, but it is not a
cheap estimator. It must exercise the real validation boundaries that affect
length and supervision correctness, including image open/processor behavior,
tokenizer identity, processor-text expansion, span alignment, Qwen placeholder
counts, and greedy packing when those stages exist for the training path.

Introduce one internal train-stream owner under `src/training/stream.py`. It
composes the encoded-example stream, `data.train_order`, epoch replay,
`src/packing/` greedy packing, deterministic tail-fill, preflight pack count,
and planned optimizer-step grouping. The packer remains lower-level and never
sorts, shuffles, samples, or owns epoch/tail-fill semantics. The trainer consumes
the resulting `MicroStep` stream; it does not own data ordering, packing, or
tail-fill details.

`MicroStep` should be defined at this training-stream boundary, preferably in
`src/training/stream.py` unless the file becomes crowded enough to justify a tiny
`src/training/types.py`. Do not define `MicroStep` in `src/packing/`, because
packing should not know optimizer-step grouping or accumulation. Do not make
`SupervisedTrainer` assign `planned_step_id` or `micro_step_id`; those ids are a
stream responsibility.

A `MicroStep` may carry the prepared `PackedSequence` and `QwenForwardInputs`
needed for one packed forward/backward unit, plus schedule and trace metadata. It
must not carry `ModelOutputs`, `LossBundle`, optimizer state, raw config, raw
examples, or historical execution dumps. It is input to execution, not execution
history. `SupervisedTrainer` may log counts and ids exposed by `MicroStep` or
`PackedSequence`, but it must not inspect raw examples, decide example order, or
reinterpret segment semantics.

Do not serialize every `MicroStep` as a normal run artifact. V1 writes compact
pack/step summaries to metrics and receipts; full micro-step dumps are debug-only
sidecars when an explicit debug path needs them.

Tail-fill consumes from the next deterministic epoch stream using the same
`data.train_order` policy and the next seed/order state. Packing may cross epoch
boundaries, including tail-fill, as long as every packed segment records epoch
index, source example id, realized data-order metadata, and pack-local span
metadata. Do not force pack commits at epoch boundaries in V1.

`reports/pack_plan.json` is a summary-first receipt. It records
`packs_per_epoch`, requested pack presentations, actual pack presentations,
`tail_fill_pack_count`, `resolved_max_steps`, `effective_batch_size`, world size,
`resolved_grad_accum_steps`, pack-utilization summary, `data.train_order`,
`template.object_ordering`, seed/fingerprint inputs, and validation status.
Detailed per-pack listings are required for smoke/debug traces, but not for every
production run by default.

`eval` is a top-level group parallel to `training` because measurement has its
own modes and artifacts. `eval.forward` names packed teacher-forced forward
evaluation over `PackedSequence` and `LossRunner`. `eval.inference` names
offline generation/inference evaluation. `data.train_path` is required, and
`data.eval_path` is optional. If `data.eval_path` is absent, scheduled
`eval.forward` is disabled unless an explicit smoke fixture provides eval data;
do not implicitly split or reuse train JSONL for evaluation. `checkpoint` owns
save cadence, retention/alias behavior, and checkpoint metadata policy.
`optimizer` owns both optimizer and scheduler choices, because scheduler
stepping is part of the optimization policy.

V1 optimizer and scheduler defaults:

```yaml
optimizer:
  name: adamw_torch
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-8
  kwargs: {}
  groups:
    adapters:
      language:
        lr: ...
        weight_decay: ...
      aligner:
        lr: ...
        weight_decay: ...
      vision:
        lr: ...
        weight_decay: ...
    token_embeddings:
      lr: ...
      weight_decay: ...
  scheduler:
    name: cosine_with_warmup
    warmup_ratio: ...
    warmup_steps: null
    kwargs: {}
```

The default optimizer is `adamw_torch`. Use built-in default parameter grouping
that splits decay and no-decay parameters, excluding bias and normalization
weights from weight decay. The default scheduler is `cosine_with_warmup`.
Support both `warmup_ratio` and `warmup_steps`, but exactly one may be set.
Expose the AdamW betas, epsilon, scheduler kwargs, and optimizer-specific kwargs
because these knobs materially affect training behavior in MS-Swift/Transformers
practice. Do not copy upstream config names blindly, but do not omit important
performance knobs merely for minimalism. Before implementation, source-study the
optimizer and scheduler surfaces in MS-Swift, Transformers, and adapter tooling;
the initial handles inspected are `/data/ms-swift/swift/trainers/arguments.py`,
`/data/ms-swift/swift/trainers/utils.py`, and Transformers
`training_args.py`. Advanced optimizer families such as GaLore remain out of V1
unless explicitly approved, but their existence is a reminder not to collapse
the schema to only `lr` and `weight_decay`.

The public run clock is the precomputed planned step index over
`1..resolved_max_steps`, not a separate `global_step` concept and not a
successful-update counter. Scheduler behavior must be recorded against planned
steps. If a corruption guard prevents an optimizer update on a planned step, the
event is logged as update-skipped diagnostic metadata rather than changing
eval/checkpoint/logging cadence. Do not implicitly scale learning rates by world
size. Authored learning rates are literal; profiles may choose scaled values
explicitly.

Do not use a global/default learning rate fallback for trainable parameters.
Every trainable parameter must match exactly one registered optimizer group, and
every matched group must have an explicit learning rate and explicit weight
decay in the final runnable config. Missing group LR, missing group weight
decay, a surviving `REQUIRED` placeholder, declared groups that match no
parameters, unmatched trainable parameters, and parameters matched by multiple
groups are fail-fast configuration errors. This strictness is intentional: LR
and weight-decay assignment are part of the experiment contract, not optimizer
convenience.

V1 optimizer/LR groups, freeze/trainable parameter sets, and extra checkpoint
state selection use prefix and exact-name selectors by default. Regex is not a
general optimizer selector language in V1. Regex-style matching is allowed only
for adapter target discovery when the adapter source study or PEFT/MS-Swift
parity makes it necessary, and those regex match rules must be recorded in
adapter receipts. Do not apply regex to activation selectors, loss semantics, or
hidden-state capture points.

Base configs may still enumerate expected LR groups with `lr: REQUIRED` as a
reminder. This is encouraged for non-trivial trainable surfaces, because it
forces each real run to consciously set the LR while keeping the reusable YAML
shape visible:

```yaml
optimizer:
  groups:
    adapters:
      language:
        lr: REQUIRED
        weight_decay: REQUIRED
      aligner:
        lr: REQUIRED
        weight_decay: REQUIRED
      vision:
        lr: REQUIRED
        weight_decay: REQUIRED
    token_embeddings:
      lr: REQUIRED
      weight_decay: REQUIRED
```

A run may remove or disable groups that correspond to non-trainable towers or
modules, but every trainable group that remains must resolve to a concrete
numeric learning rate and weight decay.

Learning-rate grouping must be first-class and auditable for Qwen3-VL. At
minimum the optimizer builder must be able to separate the visual tower, the
multimodal aligner/projector, and the language model. In MS-Swift vocabulary
these correspond to `vit`, `aligner`, and `llm`; in CoordExp-swift configs we
use `vision`, `aligner`, and `language`. The Qwen3-VL mapping to verify against
installed Transformers/MS-Swift is:

```text
vision   -> model.visual
aligner  -> model.visual.merger, model.visual.deepstack_merger_list
language -> model.language_model, lm_head
```

For adapter target discovery, `target_modules: all_linear` excludes `lm_head`
and output embedding/head modules by default, following PEFT/MS-Swift practice.
The semantic `language` map still records `lm_head` for identity, checkpoint,
and selected-token output-column handling, but adapter injection targets decoder
and language-model internal linear modules. Training selected output columns is
owned by `src/qwen/special_token_embeddings.py`; adapting `lm_head` with LoRA or
DoRA would require a separate approval card.

The public semantic trainable-surface vocabulary is stable:

```text
vision
aligner
language
adapter
token_embeddings
auxiliary_modules
```

Qwen-internal names such as `vit`, `mlp`, and `llm` may appear in mapping
metadata or source-study notes, but they are not the public config vocabulary.
V1 does not support full base-model parameter fine-tuning. Trainable base
weights outside approved adapter modules and approved special-token embedding
deltas are configuration errors unless a future design card explicitly adds
that training mode.

Optimizer grouping must support explicit groups for LoRA/DoRA adapter
parameters and special-token embedding parameters. Adapter LR groups are
tower-scoped under `optimizer.groups.adapters`: `language`, `aligner`, and
`vision`. Only groups for trainable configured adapter towers are required;
missing LR or weight decay for a configured trainable adapter tower fails fast.
Do not require unused adapter tower groups, and do not inherit a default adapter
LR or weight decay.

The `token_embeddings` group is the single public optimizer group name for
special-token embedding trainables and lives at
`optimizer.groups.token_embeddings`.
It may contain `shared_embed_delta` for tied models or `input_embed_delta` and
`output_embed_delta` for untied models. Auxiliary research modules should use
an `auxiliary` group namespace with named subgroups such as `coverage_ledger`
rather than being folded into `language`.

Model setup must validate tied input-embedding/lm-head state by runtime tensor
identity or pointer equality, not by config assumption. The result is recorded
in the model/setup receipt and determines whether special-token embedding
deltas are saved as `shared_embed_delta` or separate input/output deltas.
Special-token embedding trainables cover coordinate tokens plus the four schema
wrapper tokens, are fully trainable selected additive deltas, and are saved as
embedding-delta safetensors. Do not save full embedding/head matrices as the V1
checkpoint payload for this surface.

The currently preferred V1 special-token embedding hook is a small wrapper pair
installed during Qwen setup, unless the source study proves a PEFT
trainable-token path satisfies the full contract with less model surgery. In
the custom path, the input embedding wrapper calls the frozen base embedding and
adds a trainable delta only at selected token ids. The output head wrapper calls
the frozen base `lm_head`, then scatter-adds the selected-column correction
`hidden @ delta.T` for those same token ids. Tied models share one
`shared_embed_delta` between the input wrapper and output-column correction;
untied models use `input_embed_delta` and `output_embed_delta`. This makes the
selected tokens fully trainable without unfreezing or saving full
`embed_tokens.weight` or `lm_head.weight`. A setup test must prove that
perturbing one selected delta changes the matching input embedding lookup and
matching output-logit column, while non-selected tokens and frozen base matrices
remain unchanged.

The grouping builder should assign each trainable parameter to exactly one
semantic group, deduplicate tied/shared parameters by object identity, split
decay/no-decay within each semantic group, and emit `reports/optimizer_groups.json` in run
artifacts. A trainable parameter matched by multiple semantic groups is a
fail-fast error before optimizer construction.

Every run writes `optimizer_groups.json`. The name is intentionally
specific: it records parameter grouping, not every optimizer runtime detail. It
must prove every trainable parameter name, shape, dtype, semantic group, LR,
weight decay, decay/no-decay bucket, match rule, parameter count, trainable
status, trainable source, and whether the parameter came from LoRA, DoRA,
special-token embeddings, or another approved trainable module. It also records
frozen summaries and the validation result for unmatched, duplicate-matched,
and no-parameter groups.
The adapter target receipt and optimizer group receipt should cross-reference
each other when adapters are enabled: `adapter_targets.json` names injected
modules and trainable parameters, while `optimizer_groups.json` assigns those
trainable parameters to LR groups. Keep the receipts separate but easy to
compare.

The default adapter training profile is language-tower DoRA, but this is only
the default profile. If the config schema exposes `vision`, `aligner`, or
`language` adapter targets for LoRA or DoRA, optimizer construction must cover
those choices with explicit trainable-surface recording and group validation.
Do not accept a trainable adapter tower whose parameters cannot be discovered,
validated, assigned to an optimizer group, stepped, and reported.

## Optimizer Package And Group Builder

Optimizer construction is first-class infrastructure and should live in
`src/optim/`, not under `training/` or `qwen/`. `training/` asks for an optimizer
and scheduler; it should not own parameter taxonomy. `qwen/` may provide Qwen
parameter-group knowledge, but it should not own generic optimizer construction.

The public optimizer entrypoint is a function:

```python
build_optimizer(model, trainable_records, optimizer_config) -> OptimizerBundle
```

Use a function before introducing an `OptimizerBuilder` class.
`OptimizerBundle` carries the optimizer, scheduler, resolved semantic groups,
decay/no-decay splits, optimizer group receipt, trainable parameter summary, and
validation summary. The bundle is not runtime-prepared; `TrainRuntime.prepare`
owns device/distributed wrapping after semantic optimizer construction.
Do not introduce a public `TrainableRegistry` class in V1. Qwen setup should
produce a small typed list of trainable parameter records, or an equivalently
compact setup object, for `src/optim/` to consume. A registry-like abstraction
may be promoted only after multiple real trainable surfaces prove that plain
records are insufficient.

Group matching should combine built-in semantic matchers with explicit config
overrides. Built-ins provide the expected Qwen3-VL and common module taxonomy;
config overrides provide experiment-specific modules. The matching primitive is
prefix-based first:

```yaml
include_prefixes:
  - model.visual
exclude_prefixes: []
include_names: []
exclude_names: []
```

Regex and callback selectors are not V1 defaults. They can be added later if a
real module taxonomy requires them, but prefix/exact-name selectors are easier
to audit, easier for agents to reason about, and less likely to silently catch
the wrong parameter.

Qwen-specific built-in group maps live in `src/qwen/optim_groups.py` and are
consumed by `src/optim/`. This keeps model-family knowledge near Qwen while
keeping optimizer validation and construction model-agnostic.

The V1 public trainable-surface taxonomy is `vision`, `aligner`, `language`,
`adapter`, `token_embeddings`, and reserved `auxiliary_modules`. Qwen-specific
helpers may map these names to installed model paths and MS-Swift vocabulary
such as `vit`, `aligner`, and `llm`, but runnable config and optimizer receipts
should use the CoordExp-swift terms.

Frozen parameters are excluded from LR assignment, but the optimizer receipt must
still summarize frozen counts by semantic group. This makes adapter-only or
partial-finetuning runs auditable without forcing frozen parameters through the
same fail-fast LR contract as trainable parameters.

Do not hardcode Qwen3-VL input/output embedding tying. The official
`Qwen/Qwen3-VL-2B-Instruct` and `Qwen/Qwen3-VL-4B-Instruct` configs currently
report `tie_word_embeddings=true`, while `Qwen/Qwen3-VL-8B-Instruct`, the
`Qwen/Qwen3-VL-30B-A3B-Instruct` MoE config, and the bare installed
`Qwen3VLConfig()` default report `tie_word_embeddings=false`. The model class
declares `lm_head.weight` as a tied-weight key, but actual tying depends on the
loaded config and resulting parameter identity. CoordExp-swift should detect and
validate the loaded model/config. Any `token_embeddings` optimizer logic must
branch on the actual tied/untied parameter identity rather than assuming one
mode.

Local preflight on
`model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
reports `tie_word_embeddings=true`. For tied models, the special-token
embedding delta must act as a tied surface: the shared delta participates in
input embedding lookup and in the corresponding output-logit columns. For
untied models, separate input and output deltas are used. This keeps
special-token embedding tuning as compact full-vector tuning for selected
tokens, not LoRA over the embedding/head and not full-matrix unfreezing.

Defaults live in typed config schemas plus named profile YAMLs. Code defaults
must be visible through an inspection command and the resolved config artifact;
they should not be hidden inside constructors in a way that makes runs hard to
reproduce.

New training-infrastructure entrypoints should receive config through
`--config PATH`. V1 rejects CLI config overrides such as `--set key=value`;
explicit smoke/debug YAMLs should carry temporary differences. The V1 design
should stay config-first rather than accumulating many stable CLI flags.
Environment variables should not be the primary configuration surface.
Existing CoordExp infer/eval workflows remain YAML-first under their current
workflow contract.

Use `run.artifact_root` as the authored base directory for training artifacts.
Avoid `output_dir` in the new training infrastructure because it is too generic
and overloaded by HF-style training and existing infer/eval workflows. The
resolved concrete directory for one execution is `run_dir`; it is derived from
`run.artifact_root` plus the run identity or naming policy and is written into
run metadata. In short:

```text
artifact_root = authored parent/base for training artifacts
run.name      = human research label
run_id        = generated execution id, default UTC timestamp
run_dir       = artifact_root / run.name / run_id
output_dir    = avoid in CoordExp-swift training config
```

Runnable training configs require `run.name`. The name should be a human
authored research label, not inferred from the config filename. Each execution
generates a `run_id`; the default format is filesystem-safe UTC timestamp
`YYYYMMDDTHHMMSSZ`. Run-directory creation must be atomic. If the generated
directory already exists, append a short random or hash suffix immediately, for
example `YYYYMMDDTHHMMSSZ-<shorthex>`, and retry rather than overwriting or
failing nondeterministically on same-second launches.

Default `run_dir` construction is:

```text
run_dir = run.artifact_root / run.name / run_id
```

This preserves failed attempts and reruns without changing YAML and without
mutating previous artifacts. V1 should not support an overwrite knob. A run
fails only if atomic timestamp-plus-suffix creation cannot produce a unique
`run_dir`.

The V1 run directory should use a small stable layout:

```text
run_dir/
  run_manifest.json
  configs/
    resolved.yaml
    resolved.json
  checkpoints/
  metrics/
  eval/
    forward/
  reports/
  debug/
```

`run_manifest.json` lives directly under `run_dir` because it is the run's
entrypoint for humans and agents. `configs/resolved.yaml` and
`configs/resolved.json` are the canonical resolved config artifacts, and the
manifest points to them.

The manifest is an index, not a duplicated dump. It should link run id, run
directory, resolved config artifacts, resolved config fingerprint, receipt
paths, metric files, checkpoint pointers, eval summaries, and enabled cache
identities when present. It should not embed full configs, full receipts,
package inventories, git changed-file inventories, or environment ledgers in
V1.

Minimum V1 `run_manifest.json` top-level sections are:

```text
run_id
run_name
run_dir
status
created_at
updated_at
configs
resolution
runtime_identity
schedule
receipts
metrics
checkpoints
eval
runtime
warnings
```

Manifest status values are `initializing`, `dry_run`, `running`, `failed`, and
`completed`. Paths inside the manifest are run-dir-relative when they point
inside the run directory, and absolute when they point outside it. Each receipt,
checkpoint, eval summary, or skipped planned artifact records a short status and
reason; the manifest links the detailed file rather than embedding it.

Any entrypoint that creates a run or inference artifact root must write the full
self-contained resolved config before execution begins. The resolved config
artifacts contain fully resolved values, defaults, validated values, resolved
paths, effective schema version, and config fingerprint. They should be
self-contained enough to run or inspect without reopening the original authored
YAML chain. Do not copy or link authored YAML files into the run by default, and
do not preserve original YAML comments or relative parent/child relationships as
run truth. YAML is the human-readable resolved view; JSON is the
machine-readable resolved view.

Resolved config artifacts do include compact resolution provenance. This is
audit metadata, not replay truth: `resolution.sources.entry_config`, ordered
`resolution.sources.parents`, each source path's content fingerprint, and
field-level path-origin metadata for path fields. Input paths such as dataset,
fixture, model/cache, image, and reference paths serialize as resolved absolute
paths, with repo-relative companions when they sit under the repo. Authored
`run.artifact_root` remains an operator placement value; `run_manifest.json`
records the concrete absolute `run_dir`.

The resolved config fingerprint covers behaviorally meaningful resolved config
content only. It excludes `run_id`, timestamps, package versions, git state,
runtime world size, and other launch-context metadata. Those values can be
recorded as compact run interpretation metadata, but they are not part of the
resolved config fingerprint.

V1 runtime/checkpoint metadata should include compact `runtime_identity` or
`code_identity`, distinct from `resolved_config_fingerprint`: repo path or
package label, git commit when available, dirty/unknown status, Python version,
and selected dependency versions for PyTorch, Transformers, Accelerate,
DeepSpeed, PEFT, safetensors, and flash-attn when installed. Do not embed full
package inventories, git diffs, changed-file inventories, environment dumps, or
conda/pip ledgers in `run_manifest.json` or checkpoint metadata by default.

V1 runtime artifacts are not a report framework. They are compact runtime
receipts for questions that cannot be answered from code or config alone after
the run starts. Write a receipt only when it proves one of these things:

- what exact config/run was executed;
- whether a fragile boundary resolved as intended;
- how to interpret a checkpoint later.

Required V1 receipts are therefore deliberately narrow:

```text
configs/resolved.yaml
configs/resolved.json
run_manifest.json
resolved_step_schedule.json
reports/qwen_setup.json
reports/pack_plan.json
reports/loss_plan.json
reports/token_type_vocab.json
reports/special_token_embeddings.json
reports/optimizer_groups.json
debug/qwen_forward_contract.json  # smoke/failure/explicit-debug only
```

The exact names of Qwen setup sub-receipts may be refined during module approval,
but the intent is small boundary evidence, not a broad per-subsystem paperwork
surface. Do not add a generic report base class, formal JSON Schema files for
every receipt, mandatory shared report headers, or a `src/reports/` framework in
V1. Use typed Python serialization in the owning module and smoke tests for the
required receipts.

If a future cache stage is enabled, it must register a cache receipt or manifest
entry. Uncached V1 smoke runs should not be forced to create
`reports/cache.json`; the run manifest may simply omit cache identities
or record an empty `cache.enabled_stages: []` summary.

Do not create stable `inspect_*.py` entry files in V1. Inspection capabilities
should live as package-owned helpers, tests, or temporary development utilities
until they earn a stable entry surface. If a future inspection entry is added,
it should be read-only by default and may accept an explicit `--output PATH` to
write a JSON or debug dump for agents, tests, or deeper manual inspection.

`trace_config.py` should show the config inheritance chain, resolved values,
unknown-key failures, fingerprints, and artifact paths without building model
tensors or launching GPU-heavy code.

Use one authored global seed in V1. Derived component seeds may be computed for
data ordering, object ordering, packing, dataloader workers, torch, and
distributed ranks, but those derived values should be deterministic from the
global seed and recorded in run artifacts rather than authored as many separate
seed knobs.

If a future cache stage is enabled, `object_ordering: random` is allowed only
when the global seed and epoch policy make rendered output deterministic and
cache-fingerprintable. If random object ordering would change per-index length
or pack fit in a way that is not represented in the cache fingerprint and epoch
policy, the run should fail fast instead of reusing or writing unsafe cache
entries.

Do not accept MS-Swift compatibility aliases in V1. If a setting matters, name
it in CoordExp-swift terms. MS-Swift configs may be referenced for
understanding or migration, but they are not accepted config syntax.

## Training Loop, Checkpointing, And Trainer Substrate

`training/` owns orchestration only: invoking Qwen setup, invoking `src/optim/`
to construct the optimizer/scheduler bundle, runtime-derived gradient
accumulation, backward pass, clipping, distributed coordination, optimizer and
scheduler stepping, checkpointing, evaluation hooks, and calls into
`LossRunner`. It does not own data processing, packing semantics, Qwen
encoding, optimizer parameter taxonomy, or loss semantics.

Use `MicroStep` as the internal one-packed-sequence forward/backward unit. A
`MicroStep` carries one `PackedSequence` per rank plus accumulation and trace
metadata. The plain word "step" in public training docs, metrics, checkpoint
cadence, and schedules means the planned optimizer-update step, not one packed
forward. Do not use `Batch`, `MicroBatch`, or `TrainStep` as the conceptual
supervised training unit in V1.

Gradient accumulation is over `MicroStep`s, but its count is not an authored
config knob. Runtime setup derives `resolved_grad_accum_steps` from
`training.effective_batch_size` and the actual training world size. Observed
counters such as packed segments, supervised atoms, supervised spans, effective
pack cost, and wall-time rate should be logged, but they do not replace
`MicroStep` as the accumulation unit in V1.

Do not use `Stage1` in public class names, command names, or package names. It
requires background familiarity and hides the actual behavior. Prefer
descriptive training-mode language such as `supervised`, `offline`, or
`rollout` once the final names are approved. New implementation names should
avoid background-dependent stage labels.

The trainer substrate is a local explicit supervised training loop, not HF
`Trainer` as the production base class. The loop should be manual at the
CoordExp semantic layer and upstream-backed at the systems layer:

- CoordExp owns `MicroStep`, `PackedSequence`, `PackedLayout`,
  `QwenForwardInputs`, `LossContext`, `LossRunner`, metrics, artifacts, and
  checkpoint metadata.
- Transformers owns the Qwen3-VL model architecture, processor behavior, model
  loading, and official Qwen/FlashAttention integration surfaces where they are
  correct and inspectable.
- Accelerate and, when explicitly configured, DeepSpeed own or strongly guide
  device placement, distributed wrapping, mixed precision, gradient scaling,
  backward, gradient clipping, optimizer-step synchronization, rank-safe saves,
  and other numerically delicate execution mechanics.

This is not a bare PyTorch reinvention. The local loop must mimic and reference
official HF/Accelerate/DeepSpeed behavior for efficiency, precision, autocast,
gradient scaling, clipping, scheduler order, bad-step diagnostics, distributed
wrapping, save/load conventions, and numerical correctness.

HF `Trainer` remains valuable as reference material and as a narrow parity
oracle, but it is not the V1 packed-training substrate. It may be used for
small CE-only or unpacked parity tests, implementation comparison, or source
reading. It should not own the production data path, `labels` semantics,
collation, loss denominator, save/eval schedule, or sidecar routing. A thin HF
`Trainer` subclass is a rejected-but-credible fallback only if future evidence
shows the local loop is more fragile than the adapter.

The deciding trade-off is semantic ownership. HF `Trainer` can probably be bent
to carry a single packed row plus sidecars through `compute_loss`, and it
already implements useful scheduling and checkpoint machinery. However, the
core CoordExp-swift contracts are deliberately non-standard: one packed
sequence per rank/step, segment-isolated attention and position behavior,
canonical `TokenSequence` supervision, physical-position `LossContext`, flexible
token-wise auxiliary losses, segment-balanced reduction, and explicit Qwen
FlashAttention varlen inputs. Making those contracts first-class is more
important than inheriting a conventional batch/labels/collator trainer surface.

Distributed substrate is therefore local-loop-first with upstream primitives.
V1 should not bind itself to MS-Swift orchestration or HF `Trainer`
orchestration, but it should freely reuse valuable external infrastructure such
as Transformers, Accelerate, DeepSpeed, PyTorch, and PEFT where those tools do
not own CoordExp data, packing, supervision, or loss semantics.

Before implementation, the first `training/` approval card should state which
HF/Accelerate/DeepSpeed behaviors are delegated, mimicked, or intentionally out
of scope. Minimum V1 verification should include:

- a deterministic one-step segment-balanced loss test with unequal segment
  lengths;
- a tiny local-loop versus HF reference parity check for plain hard-CE behavior;
- a packed FlashAttention varlen validation test proving `PackedLayout`
  produces the expected `cu_seq_lens_*`, max sequence lengths, and fail-fast
  behavior when multi-segment packed training lacks explicit varlen tensors;
- a resolved max-steps-relative save/eval schedule artifact emitted before
  training starts.

Two temporary trainer-substrate studies were digested into this decision and
removed. Durable takeaways:

- HF `Trainer` can probably be adapted, but its `labels`,
  `num_items_in_batch`, `model_accepts_loss_kwargs`, callback, and
  batch-sampling contracts become recurring hidden context for every custom loss
  and packed-training change.
- The useful systems machinery is mostly Accelerate/DeepSpeed/PyTorch, not HF
  `Trainer` itself; call those primitives directly through a named
  `training/` runtime seam.
- Packed Qwen MRoPE positions and FlashAttention varlen tensors are our
  responsibility under any substrate, so HF `Trainer` does not remove the main
  Qwen/packing correctness burden.
- HF supports fractional `save_steps`/`eval_steps`, but CoordExp-swift should
  materialize concrete absolute save/eval steps into run artifacts before
  training starts.
- Reopen this substrate decision only if implementation evidence shows the
  local Accelerate-backed loop is less reliable than a thin HF `Trainer`
  adapter.

Use semantic class names rather than CoordExp-prefixed class names. `CoordExp`
belongs in docs, artifact schemas, run metadata, and logs; implementation class
names should describe behavior. Adopt `SupervisedTrainer` for supervised
teacher-forced/offline packed training, reserve `RolloutTrainer` for future
rollout-derived supervision, and avoid `CoordExpSupervisedTrainer`,
`CoordExpRolloutTrainer`, bare `Trainer`, and `Stage1Trainer`.

CoordExp should still have a central object-oriented trainer object. The
rejected part is inheriting from or centering HF `Trainer`, not having a
trainer-like object at all. The first concrete central object is
`SupervisedTrainer`. It should compose package-owned builders and runners rather
than own their semantics: config resolution, `PackedSequence` iteration, Qwen
forward inputs, `LossRunner`, `TrainRuntime`, metrics, schedules, checkpoints,
and evaluation hooks.

`SupervisedTrainer` owns the supervised execution loop: step iteration,
runtime-derived gradient accumulation, backward, optimizer and scheduler
stepping, eval hooks, checkpoint calls, metric emission, non-finite runtime
policy, and run lifecycle. It does not own objective semantics. It calls
`LossRunner` and treats `LossBundle.total_loss` as the backward-ready
planned-step-normalized contribution for the current `MicroStep`. It must not
special-case CE, token-type gate, or auxiliary loss math.

Do not introduce a public `TrainingCoordinator` class name in V1. The term is
less familiar and does not buy enough clarity. Also do not introduce a public
`TrainerProtocol` or a `BaseTrainer` before there is more than one concrete
trainer; shared abstractions should earn their place. If `RolloutTrainer`,
command dispatch, tests, or typing later need a common type, add a tiny
protocol then with lifecycle methods such as `train()` and `evaluate()`.
Do not put shared implementation in that protocol or in an early base class;
keep shared behavior in components such as `TrainRuntime`, `MilestoneSchedule`,
`CheckpointWriter`, `MetricSink`, and `LossRunner`.

Wire `SupervisedTrainer` through a plain factory function:

```python
build_supervised_trainer(config) -> SupervisedTrainer
```

The factory may resolve config and assemble package-owned components, but it
must not own rendering, encoding, packing, loss, runtime, metric, checkpoint, or
evaluation semantics. Avoid a generic `TrainerFactory` in V1; that name invites
MS-Swift-style trainer routing and hides the concrete training mode too early.
`src/train.py` should call `build_supervised_trainer(config)` rather than
manually wiring the full object graph inline.

`SupervisedTrainer` exposes only `train()` and `evaluate_forward()` as public
V1 lifecycle methods, plus small internal helpers as needed. `train()` owns the
visible supervised optimization loop. `evaluate_forward()` runs the same
packed forward/loss path under `torch.no_grad()` without backward, so packed
forward eval is callable and testable instead of being hidden inside training.

Use `MicroStep` vocabulary only for the physical packed forward/backward unit.
The public planned step is one intended optimizer update after accumulation
across ranks. `planned_step_id` refers to the planned optimizer-update id, not to
each micro-step. One `MicroStep` contains `planned_step_id`, `micro_step_id`,
`PackedSequence`, `QwenForwardInputs`, schedule context, and trace metadata. It
does not contain `ModelOutputs`, `LossBundle`, raw examples, or raw config. A
`MicroStep` is the input to one forward/backward micro-step, not its result and
not a historical archive.

The canonical optimizer-step order is:

```text
MicroStep(PackedSequence)
  -> move tensors / QwenForwardInputs
  -> qwen forward with labels=None
  -> ModelOutputs
  -> build LossContext
  -> LossRunner.compute weighted planned-step-normalized contribution and top-level metrics
  -> pre-backward finite check on LossBundle.total_loss and per-term status
  -> runtime.backward
  -> accumulate until optimizer-step boundary
  -> unscale/check gradients when applicable
  -> clip gradients
  -> optimizer.step
  -> scheduler.step
  -> zero_grad
  -> write metric event
  -> scheduled eval/checkpoint if due
```

The trainer receives an iterator or stream of `MicroStep`s produced by
package-owned builders. It must not build raw examples through packing itself,
and it must not consume standard padded PyTorch batches.

Training should not require materializing all packs before the run starts. The
default path is a streaming iterator with optional stage-owned caches for data,
rendering, encoding, or packing when those caches are explicitly approved and
fingerprinted. Do not make "precompute every pack into memory" or "always cache
packs to disk first" the V1 training contract.

`TrainRuntime` is the required runtime wrapper from day one, even on a single
GPU. It owns or delegates device/distributed `prepare`, `backward`, gradient
clipping helpers, optimizer-step helpers, metric gathering, rank guards,
bad-step diagnostics, and optional Accelerate/DeepSpeed save helpers.
Single-GPU execution should use the same runtime seam so debug, local smoke,
and distributed runs do not drift. It does not own loss computation or
checkpoint policy.

`TrainRuntime` is Accelerate-first and must support both Accelerate and
DeepSpeed-backed execution. Raw single-GPU execution, distributed data parallel
execution, and DeepSpeed execution should all pass through the same runtime
interface. DeepSpeed is configured through the runtime/Accelerate configuration
surface rather than by creating a separate trainer substrate. This keeps
systems mechanics reusable while preserving CoordExp-owned data, packing,
forward, and loss semantics.

Support claims are evidence-scoped. The first vertical smoke proves the
single-process/single-GPU semantic training path and DeepSpeed config-conflict
validation only. DeepSpeed execution is a V1 runtime goal, but it is not
production-supported until a separate DeepSpeed systems smoke proves prepare,
backward, accumulation, non-finite consensus, scheduler, checkpoint save, and
rank-safe artifact behavior through the same `TrainRuntime` seam.
Use explicit DeepSpeed status vocabulary in docs and artifacts:
`schema_accepted`, `conflict_validation_implemented`,
`systems_smoke_verified`, and `production_supported`. V1 baseline may claim
only the statuses it has actually proven.

`TrainRuntime.prepare(...)` owns device and distributed wrapping for the model,
optimizer, scheduler, and any runtime-owned iterable or tensor handles. It does
not own raw data loading, rendering, Qwen encoding, packing policy, or loss
construction. Those semantic builders remain package-owned and are passed into
the trainer/runtime boundary after they have been configured and validated.

Gradient clipping happens after accumulation is ready and before the optimizer
update for the planned step, using runtime/Accelerate-safe clipping helpers.
It is never per micro-step. The approved optimizer-step order is:
accumulate `MicroStep`s, unscale/check gradients when applicable, gather
all-rank finite/overflow status, clip gradients when safe, `optimizer.step()`
when safe, advance `scheduler.step()` on the planned-step schedule, `zero_grad()`,
then emit optimizer-step metrics and lifecycle artifacts. If unsafe non-finite
state skips the optimizer update, do not call `optimizer.step()`, but still
advance the scheduler on the planned-step clock and record
`optimizer_update_applied: false`, `scheduler_step_applied: true`, plus the
global/per-rank non-finite status. If a backend cannot advance the scheduler
independently from an optimizer update, that backend configuration is not
eligible for this non-finite policy until a runtime card resolves it.
The planned step index remains the source of truth for logging,
eval/checkpoint milestones, and run interpretation. Bad-sample or bad-step
warnings do not create a second counter and do not retime milestones. If an
unsafe non-finite scalar or gradient prevents the optimizer update, record that
status on the planned step and keep the static schedule semantics explicit.

Distributed non-finite handling uses one all-rank decision. `TrainRuntime`
should expose an `OptimizerStepStatus`-style result that records per-rank loss
finite flags, per-rank gradient finite/norm status when available, backend
overflow status, the reduced global decision, whether the optimizer update ran,
whether the scheduler advanced, and whether gradients were cleared. If any rank
is unsafe, all ranks skip the optimizer update for that planned step, clear
accumulated gradients, advance the scheduler according to the planned-step
policy above, and emit one rank-safe status event. No rank may step while
another skips.

Distributed artifact writes are rank-safe. `TrainRuntime` exposes rank guards
and scalar-gathering helpers; rank 0 writes metric files, receipts, manifests,
and checkpoints unless a backend-specific save helper requires coordinated
participation. Components should not independently invent rank-writing policy.

`CheckpointWriter` lives under `src/artifacts/checkpoints.py` and owns
checkpoint schema, adapter payloads, special-token embedding deltas, metadata,
and aliases. `TrainRuntime` provides rank-safe save helpers plus unwrap/prepare
utilities, but it does not own checkpoint semantics or artifact schema.
`SupervisedTrainer` calls the checkpoint writer; it does not write checkpoint
files inline.

V1 checkpoints save adapter weights when enabled, special-token embedding
deltas, tokenizer/processor/model identity, resolved config fingerprint,
loss/pack/cache metadata, counters, and run-interpretation metadata. They do
not promise optimizer, scheduler, scaler, dataloader, iterator, or RNG resume.
Checkpoint and eval cadence are based on the
precomputed planned step schedule, not wall-clock time, not raw examples, and
not a separate successful-update counter.

Mixed precision is a runtime policy, not a loss-math policy. The model forward
runs under the configured precision/autocast profile, defaulting to `bf16` when
configured and supported. Loss math still selects and upcasts differentiable
logits through `LossContext` to fp32 before objective computation. Backward runs
from the resulting scalar loss. V1 does not add a custom scaler for bf16; fp16
support, if enabled, should follow upstream runtime/Accelerate conventions
rather than custom ad hoc scaling.

Gradient accumulation boundaries live visibly in the central training loop, but
loss denominator semantics live in `LossRunner` plus the normalizer metadata
provided for the planned step. The loop should make the convention explicit, for
example:

```python
loss_bundle = loss_runner.compute(..., normalizers=planned_step_normalizers)
status = runtime.check_loss_finite(loss_bundle)
if status.safe_to_backward:
    runtime.backward(loss_bundle.total_loss)
```

`resolved_grad_accum_steps` still controls how many rank-local `MicroStep`s form
one planned optimizer step. It is not a universal scalar divisor for
`segment_balanced` protected losses, because those losses are normalized by the
eligible segment denominator over the whole planned step. A future loss term may
use a different approved denominator, but it must expose that choice explicitly.

Bad sample or bad step handling is centralized in
`SupervisedTrainer`/`TrainRuntime`. Recoverable issues are recorded or warned on
the current planned step; they do not rewrite `resolved_max_steps`, retime
scheduled eval or checkpoint calls, or introduce a new
`global_step`/successful-step clock. If a loss or gradient norm is non-finite,
the runtime must not apply a corrupted optimizer update or replace the loss with
zero. Individual loss terms should not skip themselves independently; they
should return diagnostics for the central runtime/trainer policy to record.

Step metrics should record enough to debug optimization without dumping tensors:
planned step, micro-step id, pack id, segment count, physical length, supervised
atom count, per-term losses, token-level `acc_top1` and `acc_top5`, gradient
norm, optimizer LR groups, optimizer-update status, warning/non-finite status,
accumulation context, and timing. Metrics should record the unscaled
`LossBundle.total_loss`; gradient accumulation scaling is reported as runtime
context.

The first implementation slice is a single-GPU vertical smoke, not a
config-only skeleton and not a production stub-trainer layer. The first proof
point should exercise the real path end to end with the real configured
`global_max_length`, a permanent tiny fixture dataset, a dataset
`sample_limit`, five planned training steps, and the default
resolved-run-relative `eval.forward` trigger schedule:

```text
bounded fixture JSONL dataset slice
  -> RawExample
  -> RenderedExample
  -> EncodedExample
  -> PackedSequence + QwenForwardInputs under real global_max_length
  -> Qwen3-VL forward using base model or base+adapter loading
  -> LossContext
  -> BaseTokenCE + TokenTypeGateLoss
  -> backward
  -> 5 planned steps with optimizer-update status
  -> scheduled eval.forward calls
  -> metric events
  -> checkpoint metadata and status records
```

The V1 vertical smoke should not run DeepSpeed. It should verify DeepSpeed
config conflict handling at config/runtime setup level, especially gradient
accumulation and batch-size conflicts, while real DeepSpeed execution becomes a
later systems smoke after the single-GPU supervised path is proven. This keeps
the first proof point focused on semantic pipeline correctness rather than
cluster/runtime availability.

For smoke, `sample_limit` limits loaded examples. The fixture and
`training.effective_batch_size: 2` plus `training.max_steps: 5` should make the
single-rank smoke resolve to `resolved_grad_accum_steps=2` and
`resolved_max_steps=5`. This deliberately exercises accumulation and
planned-step denominator behavior in the smallest vertical proof. This is the
intended use of the max-step override: bounded debugging and smoke
verification. Production configs should keep `max_steps: null` and let epochs
plus the packed dataloader determine the run length.

Smoke should use explicit resolved-run-relative eval cadence so the first
acceptance path remains small and deterministic. With `resolved_max_steps=5`,
the smoke uses `eval.forward.steps: [2, 4]`: roughly 40% and 80%. The final
planned step still saves the required final checkpoint, but it does not imply a
third `eval.forward` unless that step is explicitly scheduled. These are
planned-step schedule events, not raw examples, not packed eval units, and not a
successful-update counter. Each scheduled `eval.forward` call uses the
configured bounded packed eval stream under `torch.no_grad()`. It is not
generation and does not run backward.

This first slice should still create the real config and artifact skeleton as
part of the smoke: `run_dir`, `run_manifest.json`,
`configs/resolved.yaml`, `configs/resolved.json`, metric events, and checkpoint
metadata. Those files are outputs of the real path, not a separate
`train_stub.json` contract.

Smoke fixtures and smoke configs should be permanent repo artifacts because
they are regression anchors and agent-readable examples. The first smoke
fixture should be self-contained and real enough to exercise the Qwen
image/template path: use two short existing CoordExp-style local examples and
their copied images, then store stable metadata beside the fixture JSONL.
Renderer-produced expected rendered-text snapshots are added later after the
real renderer exists. Do not depend on an external dataset path at smoke time,
and do not use a synthetic-only fixture as the defining vertical smoke.

The selected source image must be copied into
`tests/fixtures/smoke/qwen3_vl_single_image_pack/images/`. The fixture should
record the original source path, source file stat, copied fixture path, checksum,
and selection rationale in `checksums.json`. The fixture README may summarize
or point to that provenance, but `checksums.json` is the structured source of
truth. Use checksum for this permanent fixture even though production image
identity defaults to path plus file stat.
Keep `checksums.json` fixture-specific rather than turning it into a generic
provenance manifest. It should be small and explicit: checksum algorithm,
`source_examples`, an `images` list, original source paths, copied fixture
paths, source file stats, checksums, and selection rationale.
Use SHA-256 for the copied fixture image checksum and record
`"algorithm": "sha256"` once in `checksums.json`. V1 does not need configurable
checksum algorithms for the smoke fixture.

Put the deterministic fixture and its smoke config under the fixture directory:

```text
tests/fixtures/smoke/qwen3_vl_single_image_pack/
```

The fixture directory should use this compact layout:

```text
tests/fixtures/smoke/qwen3_vl_single_image_pack/
  README.md
  config.yaml
  examples.jsonl
  expected_rendered.json
  images/
  checksums.json
```

The fixture `config.yaml` should be self-contained and should not use
`extends`. It still uses the same strict `TrainConfig` schema as normal
training configs; it simply does not depend on production profile inheritance.
Config inheritance remains covered by separate config tests. This keeps the
smoke fixture runnable and stable even if reusable production profiles churn.

The fixture config should write generated smoke runs under an ignored artifact
root such as:

```text
outputs/smoke/qwen3_vl_single_image_pack/
```

Run directories under that artifact root still use normal timestamped run-id
semantics. Do not write smoke run outputs under
`tests/fixtures/smoke/qwen3_vl_single_image_pack/`; fixture source files are
permanent regression anchors, while smoke outputs are generated artifacts.

The first fixture should prefer a real current `len12000` training row with
exactly two valid objects, short rendered length, ordinary image dimensions,
and descriptions without special/control token hazards. If no suitable
exactly-two-object row exists, the implementer may select a real valid row with
more than two objects and create the canonical fixture example from exactly two
copied objects from that row. That reduction must be recorded in
`checksums.json` selection rationale. Choose the first two source-order objects
that are valid, short, and description-safe; do not choose visually distinct or
hard objects by subjective judgment for the first fixture.

Prioritize boring validity over representational diversity. Two objects is the
smallest useful case that exercises object ordering, repeated schema wrappers,
coordinate spans, and object boundaries without turning the smoke into a broad
dataset test. The defining smoke should lock object ordering to deterministic
`object_ordering: source_order`; random object ordering can be tested
separately after the deterministic path is stable.
The fixture descriptions should be ordinary safe strings copied from source
after confirming they do not contain Qwen/control/CoordExp token syntax. Do not
intentionally include newline, tab, control-token, or escaping edge cases in the
first vertical smoke.

The fixture `examples.jsonl` should use the new canonical `RawExample` shape,
not current coord-jsonl source-format field names such as `images`, `bbox_2d`,
or `desc`. Current `len12000` coord-jsonl source-format normalization remains
covered by a separate loader test so compatibility failures are localized away
from the vertical smoke.
For derived fixture provenance, record original source `bbox_2d` coordinate
tokens under `metadata.source` when useful, but canonical fixture objects use
integer `bbox` bins.
Inside the fixture JSONL, the image path should be fixture-local and relative,
for example `images/<filename>`. The loader resolves it relative to the
declaring JSONL/config root. Do not store an absolute image path in the
permanent fixture JSONL.
The fixture `example_id` should preserve the original stable source identifier
with a fixture suffix such as `<source_example_id>__smoke2obj`; record the
original id in metadata. Preserve original object ids for selected objects when
available, and record any normalization in metadata. Do not rename selected
objects to generic ids such as `obj_0` and `obj_1` unless source object ids are
absent or invalid and the normalization is explicitly recorded.
Source provenance belongs under `metadata.source`, including original dataset
path, original example id, original object ids, reduction note when applicable,
and source row number when available. Keep top-level fixture fields canonical
and clean rather than adding provenance fields beside `example_id`, `image`, or
`objects`.
The fixture JSONL must include image `width` and `height`; Qwen encoding must
validate decoded image size against those declared dimensions.

`expected_rendered.json` should stay pre-tokenization and pure text. It records
messages, prompt text, supervised response text, typed char spans, realized
object order, and image metadata. The render snapshot check should be exact:
messages, prompt text, supervised response text, spans, object order, and image
metadata match byte-for-byte except for generated provenance fields explicitly
excluded by the test. It should not include token ids, Qwen tensors, or
packed-position metadata.

`expected_rendered.json` should be produced once by the renderer through the
real renderer path, reviewed, and committed as the frozen expected artifact. Do
not hand-write it from scratch, and do not let the test regenerate and compare
against the same transient output without a committed frozen file. Snapshot
updates are manual and review-gated. A test may print or write a diff for the
developer, but it must not auto-update `expected_rendered.json` or
`expected_tokenization.json` unless a future explicit developer command is
approved.
When a snapshot mismatch occurs, write any bounded diff artifact under the
ignored smoke/debug output root, not beside the fixture files. Stdout-only
diffs are allowed, but they are not sufficient when an artifact path is already
available for the failed smoke or snapshot check.

After the Qwen encoder exists, generate `expected_tokenization.json` through the
real `QwenExampleEncoder` path and freeze the generated snapshot after review.
Do not hand-author token ids before the tokenizer/encoder path is implemented.

`README.md` should be contract-oriented: purpose, files, invariants, how to run,
and what failures mean. It is a regression anchor for humans and agents, not a
general tutorial.

The smoke config should live with the fixture rather than under the main
`configs/` tree. The approved invocation still uses the normal config-first
entry shape, for example `python -m src.train --config
tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml`.
The fixture-local smoke config uses the same strict `TrainConfig` schema as
normal training configs; V1 should not create a special smoke mini-schema or
hard-code smoke defaults in tests.

The exact source image/example should be chosen when the smoke fixture card
becomes the active implementation card, not during this design-grilling phase.

Smoke run outputs are not permanent source artifacts. They should be generated
under the configured `run.artifact_root`, normally an ignored path such as
`outputs/smoke/qwen3_vl_single_image_pack/`, and interpreted through
`run_manifest.json`, `configs/resolved.yaml`, `configs/resolved.json`, metrics,
and checkpoint metadata. Generated smoke outputs are ignored and disposable. Do
not keep
selected smoke run outputs as golden source artifacts; the permanent source
truth is the fixture, expected snapshots, checksums/provenance, and docs.

The first vertical implementation slice is not accepted on unit tests alone.
Acceptance requires the five-step smoke to produce the resolved config,
Qwen setup receipt, pack plan, loss plan, metric events and summary, scheduled
`eval.forward` summaries, checkpoint metadata, and
`checkpoints/checkpoint-final.json`.
Module-level tests are useful progress checks, but the architectural proof is
the full tiny path running end to end.

Do not add an optional real-local-dataset smoke path in V1. The permanent tiny
fixture is the first smoke anchor. Larger local smoke runs can be added later
when they earn a separate config.

Cheap contract tests are still useful, but they are guardrails rather than the
definition of success. The valuable minimal tests are config inheritance and
unknown-key failure, run directory and run id materialization, artifact path
creation, and possibly an entrypoint/factory test. They should not delay or
replace the full smoke run.

Do not introduce a production `FakeSupervisedTrainer`, `StubTrainer`, or
`DryRunTrainer` in V1. A tiny fake object may appear inside an entrypoint test
only if it keeps that test focused on argv/config/factory behavior. The real
pipeline is verified by the vertical smoke.

The first implementation order is:

```text
archive old src -> create minimal new src skeleton
  -> data/template
  -> Qwen encoding
  -> packing
  -> Qwen forward contract smoke
  -> shifted-loss-consumption parity
  -> LossContext + BaseTokenCE + TokenTypeGateLoss
  -> trainer smoke
```

The new `src/` skeleton should contain only approved package directories and
package markers. Do not add TODO classes/functions, broad abstract bases, or
placeholder implementations before the relevant module/functionality card is
approved in the blueprint.

`eval.forward` should exist in V1 as a minimal packed evaluation loop using the
same render, encode, pack, Qwen forward wrapper, `ModelOutputs`, `LossContext`,
and `LossRunner` path as training under `torch.no_grad()`, producing the same
core supervised metrics as train wherever the same logits/supervision context
exists. It does not run backward or optimizer updates, and it should not emit
train-only runtime metrics such as LR, grad norm, accumulation state, or
backward timing. `eval.forward` uses the same packing implementation and
`global_max_length`; eval config may set its own sample limit, but it must not
use a separate padded batching path or a one-example-only fallback by default.

Canonical `eval.forward` metric events use split `eval.forward` and the same
core metric names as train, including `loss/total`, `loss/base_ce`,
`loss/token_type_gate`, top-level `acc_top1`, top-level `acc_top5`,
`pack/supervised_tokens`, supervised atom count, contributing segment count,
physical length, and effective pack cost when available. `acc_top1` remains the
preferred scalar selection signal when a single accuracy metric is needed, but
the eval surface should not narrow itself to one metric.

Each `eval.forward` run writes metric events plus a compact summary at
`eval/forward/step-<planned_step_id>.json`, for example
`eval/forward/step-4.json`. Planned-step ids in eval summary paths are unpadded,
matching checkpoint directory names. The summary contains planned step id,
split, eval dataset identity, sample count, pack count, aggregate metrics,
metric event references, linked checkpoint pointer or directory when present,
warning/update status context, and bounded timing. It does not include
per-pack, per-example, full-vocabulary logit, or per-token prediction dumps by default;
debug mode may add bounded example ids or failed-pack diagnostics.

`run_manifest.json` links eval summary paths and may store a tiny latest/best
scalar snapshot, but it must not embed full eval summaries. When eval is
triggered near a checkpoint, the manifest and pointer metadata should provide
checkpoint-to-eval links.

`eval.inference` is a separate offline generation/inference evaluation mode. V1
should not run `eval.inference` inside the training loop. Instead, training saves
checkpoints, and offline inference is a separate workflow over saved checkpoints,
following the existing CoordExp infer/eval workflow until an explicit migration
is approved.

Each scheduled `eval.forward` event records the triggering `planned_step_id` and,
when available, the linked checkpoint path plus optimizer-update and
warning/non-finite status for that planned step. Eval metrics should be
interpretable without guessing which planned schedule event triggered them.

Resolved schedules de-duplicate cadence/final collisions, so a final planned
step that is also a fractional eval milestone runs `eval.forward` once and may
record multiple trigger reasons. Static schedule still fires `eval.forward` on
warning-only or update-skipped planned steps; the eval summary records that
status instead of silently retiming or skipping the event.

Inference-style evaluation must be explicit and format-stable, not mixed
implicitly into the teacher-forced loss path.

Checkpoint V1 scope is adapter weights when enabled, special-token embedding
deltas, processor identity, resolved config fingerprints, data, template, Qwen,
cache, packing, and loss metadata, counters, and run interpretation metadata.
One shared Qwen checkpoint-loading path should support base-only,
base-plus-adapter, and base-plus-adapter-plus-special-token-embedding delta
composition for training weight initialization, offline evaluation, and
inference. This is weights-only loading, not exact training resume: optimizer
resume, scheduler restore, scaler restore, dataloader/iterator restore, and RNG
restore are not V1 contracts.

`training/` does not own dataloading, rendering, encoding, or packing semantics.
It consumes an iterator or stream of `PackedSequence + QwenForwardInputs` and
coordinates execution. Wiring factories are acceptable only if they call into
package-owned builders and do not become a second semantic owner.

Checkpointing is for weights, evaluation, provenance, and run interpretation
first. V1 saves numbered milestone checkpoint directories keyed by planned step
ids using unpadded HF-like names, for example `checkpoints/checkpoint-2`,
`checkpoints/checkpoint-4`, and `checkpoints/checkpoint-5`. Do not zero-pad
checkpoint directory step ids. `checkpoints/checkpoint-final.json` and
`checkpoints/best_acc_top1.json` are JSON pointer files, not duplicate
checkpoint directories or symlinks.
Static schedule wins: if a checkpoint milestone
lands on a planned step with warnings, the checkpoint is still written and the
manifest/checkpoint metadata records the warning and optimizer-update status.
If an unsafe non-finite guard prevented the optimizer update at that planned
step, scheduled eval/checkpoint events still fire, but the artifact must clearly
record `optimizer_update_applied: false` or an equivalent status.

`checkpoints/checkpoint-final.json` must always exist and means "final state of
this run", not "best" or "cleanest" state. It points to the final planned-step
checkpoint directory and is written even if the final planned step had warnings
or an update-skipped guard status. It must contain at least the target
checkpoint directory, planned step id, selection reason, optimizer-update
status, warning/non-finite status, and relevant schedule identity.

`checkpoints/best_acc_top1.json` may be written when best checkpointing is
enabled, selected by `eval.forward/acc_top1:max`. Warning-only checkpoints may
be selected by this alias. Unsafe non-finite or update-skipped checkpoints are
excluded from best-checkpoint selection unless a future explicit config surface
opts into considering them. The pointer file must contain the target checkpoint
directory, planned step id, metric name/value, selection mode, checkpoint status,
and reason.

The internal V1 checkpoint payload layout is explicit and compact:

```text
checkpoints/checkpoint-5/
  checkpoint_metadata.json
  adapter/                         # only when adapter exists
  special_token_embeddings.safetensors
  special_token_embeddings.json
```

Do not copy or export the base model into each checkpoint. Adapter payloads live
under `adapter/`; standard PEFT-compatible adapter layouts should be preserved
when the adapter type supports them, while DoRA may use its own explicit
metadata in that folder. Special-token embedding tensors are saved in
`special_token_embeddings.safetensors`; `special_token_embeddings.json` records
token ids, token strings, tied/untied mode, tensor keys, checkpoint tensor
shapes/dtypes, and base identity checks.

Checkpoints should include adapters, special-token embedding deltas, and enough
metadata to interpret the run:
planned step id, `resolved_max_steps`, resolved schedule fingerprint, optimizer-update
status, warning/non-finite status, linked eval outputs and metric summaries,
resolved config identity, run/artifact fingerprints, compact runtime/code identity,
training counters, loss-term summary, pack/cache identity when relevant, and
Qwen/processor identity. Optimizer, scheduler, scaler, dataloader/iterator, and
RNG state should not be part of the V1 promise, even if a runtime backend can
technically save them.

Do not merge or export special-token embedding deltas into full
`embed_tokens.weight` or `lm_head.weight` matrices in V1 checkpoints. Runtime
loading remains explicit composition: base model plus optional adapter plus
optional special-token embedding deltas. Inference and later training should
load the same composed mode rather than requiring a full merged model export.

Checkpoint loading must fail fast if base model identity, tokenizer/vocab
identity, special-token strings or ids, coordinate-token mapping, tied/untied
mode, adapter type, adapter target policy, or tensor shape/dtype expectations
do not match. Do not warn-and-try or rely only on tensor shapes for these
identity checks.

V1 does not need optimizer/RNG resume. Training runs are assumed short enough
that exact continuation is not worth the design complexity. Loading weights may
initialize a new training run or run offline inference/evaluation, with clear
metadata checks and explicit "weights-only" semantics.

Do not implement hidden-state or visual-activation caches in V1, and do not
bundle cache contents into checkpoints. If future pre-forward cache identities
or manifests exist, they may be referenced in run metadata, but cache contents
must live outside checkpoints and remain disposable.

Checkpoint and evaluation cadence should be relative to `resolved_max_steps` by
default. Absolute training steps remain the execution scale, but save/eval/logging
milestones can be expressed as fractions of the planned run, for example every
40% of `resolved_max_steps`. The default milestone cadence is
`every_fraction: 0.4`, and the final `resolved_max_steps` checkpoint must always
be saved. `resolved_max_steps` is the source of truth after run-length
resolution: it comes from `training.max_steps` when set, otherwise from
`training.epochs` plus dataloader/pack cardinality. Before training starts, the
run materializes planned step ids `1..resolved_max_steps` and all schedule events
are attached to those ids. The canonical authored cadence fields are
`checkpoint.every_fraction`, `checkpoint.steps`, `checkpoint.save_final`,
`eval.forward.every_fraction`, `eval.forward.steps`,
`training.logging.every_fraction`, and `training.logging.steps`. Do not add
parallel aliases such as `save_steps`, `eval_steps`, or
`logging_steps` in V1. Do not introduce a separate `global_step` concept for
V1, and do not rebase schedules onto successful-update counts. Per-micro-step
diagnostics may still be emitted separately when useful.

Save/checkpoint schedules and eval schedules are independently materialized,
even when their defaults happen to produce the same milestone steps. Do not
couple checkpoint writes to eval calls in code. The smoke should use the
default checkpoint cadence independently from the default eval cadence.

Every run must materialize a concrete `resolved_step_schedule.json` before
training starts. It resolves fractional and absolute cadence settings into
deterministic planned-step milestones, de-duplicates collisions, and includes
the final `resolved_max_steps` milestone. Fractional milestones use
`ceil(fraction * resolved_max_steps)`, clamped to `[1, resolved_max_steps]`. The schedule config
should support both relative cadence such as `every_fraction: 0.4` and explicit
absolute `steps: [...]`; both are materialized into the resolved schedule
artifact. This is one schedule file with separate event lists for
`eval.forward`, checkpoint, logging, and final events, not one file per
subsystem and not config-only implicit behavior.

Each resolved schedule event records at minimum `planned_step_id`, `event`
(`checkpoint`, `eval.forward`, `training.logging`, or `final`),
`trigger_reasons`, `source_config_path`, `deduped_from`, and `required`.
`trigger_reasons` distinguishes fractional cadence, explicit authored step,
default final inclusion, and deduplicated collisions. The final checkpoint
event is required even when it collides with a cadence checkpoint; collisions
are represented in `deduped_from` rather than by writing duplicate event
records.

Every run must also write a top-level `run_manifest.json` enumerating resolved
configs, step schedule, optimizer group receipt, checkpoints, eval outputs,
metric files, enabled cache identities when present, and other compact run
interpretation metadata. The manifest records planned-step checkpoint/eval
events, optimizer-update status, warning/non-finite status, aliases such as
`checkpoints/checkpoint-final.json` and `checkpoints/best_acc_top1.json`, and
links from checkpoints to adjacent eval outputs when available. Metrics should
use a canonical JSONL event stream plus a compact summary JSON. TensorBoard or
CSV can be secondary exports, not the source of truth.

`src/artifacts/` owns run-directory creation, manifest writing, artifact path
registration, and rank-safe artifact writes. Keep this surface small: a
`RunArtifactManager`-style object is acceptable, but do not introduce an event
bus, plugin registry, database, or broad artifact framework in V1. `src/train.py`
and `SupervisedTrainer` use this surface instead of manually inventing artifact
paths or manifest mutation policy.

`run_manifest.json` is a structured index of the current run, not a metric event
log. It should be rewritten atomically from structured manifest state after
major lifecycle events. Metric history remains `metrics/events.jsonl`, and
`metrics/summary.json` provides compact scalar summaries.

Boundary owners write their own compact receipts when the boundary is approved
as fragile or high-value. `run_manifest.json` registers the receipt path and a
short status or summary when useful. Do not require every subsystem to write a
receipt, and do not require every receipt to have a stable fingerprint in V1.
The artifact manager owns discoverability and atomic writes; boundary owners own
the receipt contents. Do not make the artifact manager collect raw subsystem
objects and serialize every receipt itself.

Artifact and metric writes are rank-safe through `ArtifactManager` and
`MetricSink`, using `TrainRuntime.is_main_process` or runtime-provided rank-safe
helpers. Individual subsystems should not each implement independent rank-write
policy. Named setup receipts may be written atomically and replaced by the same
subsystem during setup; metric event history and checkpoint directories are
append-only in ordinary operation. There is no global overwrite knob.

Receipts should be bounded and summary-first. Large details belong in explicit
debug sidecars written only for smoke, failure, or debug mode. Default receipts
must not dump whole datasets, full rendered examples, full processor tensors,
or giant token traces. If a receipt needs source context for debugging, include
bounded snippets such as example id, span id, local token window, short rendered
text excerpt, config path, or artifact pointer.

Setup/provenance receipts live under `run_dir/reports/` by default, while
execution-contract and failure-local diagnostics such as
`debug/qwen_forward_contract.json` live under `run_dir/debug/`.
`run_manifest.json` links both categories. Do not put all JSON files at run root
and do not collapse setup provenance into generic debug output.

Cache receipts are conditional. If no cache stage is enabled, no
`reports/cache.json` is required. If a future cache stage is enabled, its
subsystem must register a concise cache receipt or manifest link from
`run_manifest.json`, recording stage mode, fingerprint, hits/misses, stale or
rejected entries, and effective cache identity without forcing readers to inspect
cache roots.

## Rollout Training Readiness

Rollout training is not implemented in the first milestone, but the core
infrastructure should be ready for it.

Readiness is achieved through shared abstractions, not empty rollout-training
code:

- provenance-aware token atoms and spans
- generated-target-compatible token targets
- source-agnostic `PackedLayout`
- loss terms that consume supervision rather than datasets
- recipe-specific artifact and metric metadata
- a reserved `rollouts/` package for future rollout-derived supervision
- rollout example/source provenance reserved in supervision metadata
- inference-style `eval.inference` remains a separate future workflow over
  saved checkpoints
- hidden-state losses are reserved as a next-version required capability through
  an approved Qwen wrapper hook
- future rollout supervision hooks should attach to supervision/loss inputs,
  not to a separate trainer-only side channel

Supervised training should not know rollout training exists. Future rollout
training should enter as another training recipe or rollout-derived supervision
producer that reuses
`supervision/`, `packing/`, `losses/`, `training/`, `artifacts/`, and
`metrics/`.

Do not implement rollout-training modules now. Avoid placeholder code beyond
reserved architecture notes or package markers when they are truly useful.

## Entry Surface Direction

The V1 entry surface is training-first. It should support learning and
inspection through package-owned helpers and config tracing, but the only
primary stable entry promised for the new infrastructure is `src/train.py`.

The previous `python -m src <command>` dispatcher shape is no longer assumed.
Do not create a `commands/` package in V1 unless repeated entry parsing or
orchestration duplication earns it. The approved invocation style is module
execution, for example:

```bash
python -m src.train --config ...
```

`eval.py`, `infer.py`, and `visualize.py` remain good future role-named entry
files, but current inference/evaluation work should refer to the existing
CoordExp infer/eval workflow rather than being redesigned inside this training
infrastructure pass.

`src/train.py` exposes a professional, testable entry shape:

```python
def main(argv: Sequence[str] | None = None) -> int:
    ...

if __name__ == "__main__":
    raise SystemExit(main())
```

The stable V1 train arguments are deliberately tiny: `--config PATH` plus
optional `--dry-run`. Do not add stable flags for run root, model, learning
rate, step count, packing, loss, or runtime settings; those belong in YAML. Do
not add `--set key=value` or other CLI config override syntax in V1. Use explicit
smoke/debug YAMLs instead. Do not add a separate `--validate-only`; `--dry-run`
is the single launch-level preflight mode.

`src/train.py` should execute in this order:

```text
parse args
  -> load and resolve config
  -> generate run_id
  -> derive/create run_dir from run.artifact_root / run.name / run_id
  -> dump full self-contained configs/resolved.yaml and configs/resolved.json
  -> initialize run_manifest.json linking resolved config artifacts
  -> if --dry-run: validate non-heavy setup and write available plan receipts,
     then exit before model/optimizer mutation
  -> build_supervised_trainer(config)
  -> trainer.train()
```

The train entry should let domain errors fail fast with clear exception types
and stack traces. Do not add a broad catch-all that hides debugging context in
V1, and do not retry failed phases automatically.

`--dry-run` is narrow. It resolves and validates config, creates the run
directory, writes `run_manifest.json`, writes `configs/resolved.yaml` and
`configs/resolved.json`, validates cheap path/schema/setup contracts, and writes
available plan receipts that do not require CUDA or model forward when possible.
It stops before optimizer/model mutation and is not a fake training run. It does
not use a production `DryRunTrainer`, `FakeSupervisedTrainer`, or stubbed long
pipeline. `trace_config.py` may still exist as a support entry for config
inheritance/debugging, but it is not the replacement for `train.py --dry-run`.

A dry-run manifest uses `status: dry_run`. Required dry-run outputs are the run
directory, `run_manifest.json`, `configs/resolved.yaml`, `configs/resolved.json`,
compact config-resolution metadata, and any cheap path/schema receipts whose
owners can run without CUDA, model mutation, optimizer construction, or Qwen
forward. Receipts that require heavy/model-mutating work are represented as
`skipped` entries with short reasons, not silently omitted. Dry-run must not
write checkpoints, training metric events, eval summaries, adapter weights, or
special-token embedding deltas, and must not instantiate a fake trainer to make
those artifacts appear.

An optional minimal `src/train.py` test may verify argv/config resolution and
factory delegation with a tiny in-test fake object. This is a cheap entrypoint
guardrail only. The real training path is tested by the vertical smoke, not by
overloading the entrypoint test.

## Documentation And OpenSpec

Current behavior docs should remain untouched during the initial design phase.
This proposal directory is the correct durable surface for the worktree's
architecture decisions.

The old `openspec/` tree remains a legacy reference and reminder for now. It
should not be deleted or rewritten before CoordExp-swift replacement contracts
exist. Once the new infrastructure stabilizes, the OpenSpec surface should be
rebuilt deliberately around the new stable contracts.

The current phase is architecture grilling and decision refinement. After this
grilling pass and the user's manual module approval, the user plans to launch
additional review, debate, and refinement rounds over the decisions and
blueprint. Only after those rounds converge should implementation start. The
implementation stage should switch deliberately to the new OpenSpec workflow and
superpower implementation workflow, rather than continuing as informal
grilling/chat-driven coding.

## Approval Discipline

Every important module, class, or functionality requires explicit approval
before implementation. Approval cards live in the compact blueprint at
`docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md`. Each
approval card should state:

- name
- location
- purpose
- interface
- inputs and outputs
- owned state
- invariants
- non-goals
- failure modes
- tests or parity checks
- open questions
- recommended decision

This approval discipline is intentional. It is part of the learning and code
quality process, not a bureaucratic afterthought.

Explicit user approval is required before coding public modules, important
classes, cross-module data types, config schemas, receipt schemas, entrypoints,
and public functions that define the training flow. Small private helpers inside
an already approved module card may be implemented without separate approval as
long as they do not change public contracts, research meaning, artifact
semantics, or cross-module ownership.

Blueprint cards are the source of truth for module/class approval. Cross-cutting
architecture decisions live in this `DECISIONS.md` file. Do not create per-module
ADR files or leave approval only in chat. If implementation discovers a contract
mismatch that affects public behavior or research semantics, stop, update the
relevant decision/blueprint text, and request approval before continuing.

Blocking design questions must be resolved before coding the affected module.
`Open questions` may remain in a blueprint card only when they are non-blocking
implementation details. Do not introduce placeholder classes, TODO modules,
fake implementations, or broad base abstractions before the owning module card
is approved. Package markers, empty directories, and explicitly approved
reserved package names are acceptable when they support navigation without
pretending functionality exists.

Subagent review should be used at module boundaries rather than for every tiny
helper: pre-implementation contract audits for large modules, post-implementation
reviews before smoke, and cross-agent debate/refinement rounds before the
implementation phase begins.

The first approval card should cover the full vertical smoke slice as one
coherent implementation slice:

```text
archive old src -> new src skeleton
  -> data/template
  -> Qwen encoding
  -> packing
  -> Qwen forward contract smoke
  -> LossContext + CE + token-type gate
  -> trainer smoke
```

Internal module cards can still be reviewed inside that slice, but the
acceptance criterion is that the tiny real path can run the approved five-step
smoke with scheduled `eval.forward`, metrics, and
`checkpoints/checkpoint-final.json`.

## Rationale

The project aims to reduce dependence-driven learning laziness and make the
complete training path understandable locally. MS-Swift remains valuable as a
reference, but CoordExp-swift should not inherit its broad generic framework
shape.

The main simplification is to organize the new code around validated
supervision, physical packed layout, and declarative loss terms instead of
around a monolithic trainer override or a god-template abstraction.

## Consequence

Next design work should focus on the architecture blueprint and the first
module approval cards. Implementation should not start until the source layout,
top-level module responsibilities, and first command path are approved.

Evidence scope so far is discussion plus read-only subagent synthesis over the
current repo, linked worktrees, selected research/progress notes, and current
MS-Swift/packing behavior. No training or parity tests have been run for this
worktree yet.

## Inference V1 Scope

Resolved on 2026-07-02 during CoordExp-swift inference/decode planning. Patched
after five-agent review convergence on 2026-07-02.

V1 inference is an offline benchmark and evaluation path, not an interactive
demo and not the Stage-2 rollout backend. The primary flow is JSONL input plus
base model and optional adapter input, producing evaluator-facing artifacts.
The first backend is Hugging Face Transformers `model.generate`, wired through
the same Qwen loading, adapter, special-token embedding, no-resize processor,
and template semantics established for supervised training. vLLM is a required
future backend, but it is not part of the first implementation slice unless a
later OpenSpec change explicitly promotes it.

The public entrypoint should be the thin file `src/infer.py`, invoked as
`python -m src.infer --config ...`, to match `python -m src.train`. The
implementation package should use the non-colliding namespace `src/inference/`;
do not attempt to create both `src/infer.py` and `src/infer/` as siblings.
Inference should define its own strict `InferConfig` surface while reusing
shared model/template/data pieces when that keeps contracts aligned without
importing the full training schema.

The first `InferConfig` contract should be minimal and explicit:
`schema_version`, `run`/`artifact_root`, shared `model`, optional explicit
`adapter`, optional explicit special-token embedding delta, inference/eval
`data`, `template`, `backend`, `generation`, `scoring`, `artifacts`, and
`debug`. It should reject training-only keys such as `optimizer`, `training`,
and `checkpoint` unless a later approved schema intentionally shares them.
`backend.type: hf` is the only active V1 backend. vLLM values may be reserved in
the schema but must validate as not implemented.

V1 consumes the same offline JSONL example family used by training and
forward-eval, including image path, dimensions, GT objects, and prompt-relevant
fields. V1 parses generated text immediately into canonical prediction objects
while always preserving raw decode text and parse diagnostics. Parse failures
or invalid object spans should be recorded and counted rather than silently
erasing useful partial predictions or aborting the entire run on the first bad
row.

Inference configs live under `configs/coordexp_swift/infer/`. Legacy
`configs/infer/*` and old `/data/CoordExp/src/infer/*` are reference-only
materials, not authoritative schema or implementation surfaces. The first config
family should include a shallow `base.yaml` plus production leaves matching the
benchmark setup. Inheritance should remain shallow: shared model, runtime, and
template defaults may be inherited, while dataset, checkpoint/adapter,
generation, and artifact fields are explicit in leaf configs.

Unlike supervised training, inference should use batched generation for
throughput. V1 should expose an explicit required decode batch size in
`InferConfig`, default production configs should set it above one, and
`batch_size: 1` should be treated as an intentional debug or smoke override
rather than the normal path. Production configs with `batch_size: 1` should
fail validation before runtime setup.

The V1 decoding default is deterministic greedy generation with the Qwen chat
stop transition `<|im_end|>`. Do not add `<|endoftext|>` as a default stop
token. `max_new_tokens` is a public, explicit inference config value because it
directly changes invalid-output and under-generation rates. Raw trace decoding
must preserve special tokens with `skip_special_tokens=False` or equivalent
tokenizer-piece logic; parser-facing text may strip terminal `<|im_end|>` only
under a recorded policy.

Inference prompt construction must stay highly aligned with training behavior.
The inference module may expose inference-specific prompt helpers under
`src/inference/prompt.py`, but those helpers should delegate to or share the
same template semantics used by supervised training rather than copying a
divergent prompt implementation. Tiny and smoke paths must check local rendered
prompt token ids against backend prompt token ids and record template id/order,
processor identity, image metadata, and `do_resize=False` evidence.

Inference model loading follows the same base-model plus optional adapter
contract as training. The base model is resolved from `model_cache/...`.
Adapter checkpoints and special-token embedding deltas are optional but must be
explicit when used. Inference should support adapter paths that point at
training `checkpoint-final` metadata or explicit concrete checkpoint paths,
resolving them to the actual adapter and embedding-delta payloads while
preserving the resolved identity in artifacts. `best_acc_top1` style aliases are
deferred unless a later decision promotes them.

Adapter loading must do more than prove that a path exists. The V1 source study
and OpenSpec must require PEFT/base identity checks, fatal handling for missing
or unexpected adapter keys from captured PEFT `load_result` or equivalent
evidence, `set_adapter`, `get_model_status()` enabled/active adapter checks,
non-irregular active adapter status, no unexpected merged state, and expected
base/tokenizer/token-string/token-id identity checks for special-token
embedding deltas. Warning-only PEFT load paths are not sufficient. Inference
must not require a fully merged model directory and must not silently infer
arbitrary checkpoint behavior from a loose HF path.

Evidence scope: none-yet. This is a proposal-scoped planning decision. It must
be promoted into a new OpenSpec change before implementation begins.

## Inference Trace, Scoring, And Eval Contract

Resolved on 2026-07-02 during CoordExp-swift inference/decode planning. Patched
after five-agent review convergence on 2026-07-02.

V1 inference must define a backend-neutral decode result and token-trace
contract even though the first implemented backend is Hugging Face Transformers.
This prepares for required future vLLM support without implementing vLLM in the
first slice.

When score-bearing output is requested, the HF backend must request generation
scores with `return_dict_in_generate=True` and `output_scores=True`. Missing
token ids, token text, logprobs, prompt token ids, stop reason, backend name,
generation config fingerprint, tokenizer identity, or model identity is a
contract failure for scored inference. The system should fail fast rather than
falling back to constant scores when mAP-style evaluation is requested.

The OpenSpec must pin the HF score-alignment algorithm. It should define prompt
padded width, generated token indexing, whether `<|im_end|>` is retained in the
trace, whether padding after stop is trace-excluded, and whether scores are
obtained through `compute_transition_scores(..., normalize_logits=True)` or an
equivalent `log_softmax(scores[t])` gather. Every generated trace item should
record `step_index`, `token_id`, `token_text`, `logprob`, `is_stop`, `is_pad`,
and backend source. Structural trace shape mismatch is a contract failure.

`pred[*].score` is fixed for V1 as
`exp(sum(selected_token_logprobs) / n_selected)`, yielding a length-invariant
probability-like confidence value. The score-policy fingerprint must include
the selected token families, scalar transform, logprob normalization rule,
stop-token policy, and invalid-alignment policy.

The default selected token set is exactly four schema wrapper tokens plus four
coordinate tokens for the parsed object span:
`<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`,
`<|box_end|>`, and the four coordinate tokens. A valid V1 compact object has
`n_selected == 8`; free-text description/category text is excluded from object
scoring. The OpenSpec must define object boundary source, contiguity/gap
policy, duplicate/ambiguity policy, selected-token count expectations, and
persisted replay evidence: row id, object span id, generated-step indices,
token ids/text, selected logprobs, selected count, and score-policy
fingerprint. Missing or ambiguous required trace alignment makes the affected
object invalid for scored output and must be recorded diagnostically rather
than assigned a fallback score.

The first parser target is the compact object-box-closed format aligned with
training. Do not broaden V1 to parse both compact and JSON assistant responses
unless a later change promotes that compatibility. Inference uses the same
template controls as training, preserves model prediction order, and does not
geo-sort predictions after decoding.

The raw artifact and scored artifact serve different roles. `gt_vs_pred.jsonl`
may preserve partially valid predictions for debugging and salvage analysis,
but it must still preserve exactly one row per input row; extra diagnostics
belong in sidecars. `gt_vs_pred_scored.jsonl` is stricter: it must preserve
exactly one row per raw row with identical image identity, image dimensions, GT
payload, row order, and record index, but its `pred` list must contain only
predictions with finite valid comparable scores. Rows with no scoreable
predictions remain present with `pred: []` and diagnostics.

Every parsed row should carry inline parser and metric-eligibility fields such
as `parser_id`, `parser_policy`, `metric_bearing`, `parse_status`,
`valid_pred_object_count`, `dropped_pred_object_count`, and
`dropped_pred_objects`. `parse_diagnostics.jsonl` is an additive detailed
sidecar keyed by stable row id or line index; it is not a replacement for
inline row diagnostics needed by standalone evaluator/debug artifacts.

Score-bearing rows must use evaluator-readable provenance, not only a broad
`run_manifest.json`. The scored artifact contract should include row-local
non-empty `pred_score_source`, integer `pred_score_version`, and finite scores
in `[0.0, 1.0]`, plus a portable scored provenance carrier such as
`gt_vs_pred_scored.jsonl.provenance.json` that records artifact schema version,
source raw artifact SHA256, scored artifact identity when available,
detection-template id, `prompt_policy_fingerprint`,
`decode_policy_fingerprint` or exact generation config fingerprint,
`model_identity_fingerprint`, processor/template identity, `parser_policy`,
`score_policy_fingerprint`, and row-count or row-identity binding evidence.
Paths may be recorded for convenience but must not be the portable identity.

Inference must use no-resize image processing like training. The call path must
set `do_resize=False` and record processor identity plus per-row image-plan
evidence so eval does not accidentally measure image-processing drift. Runtime
must also verify processor/model vision parity for patch size, merge size, and
temporal patch size before benchmark-eligible inference.
`image_plan.jsonl` is mandatory for V1, not optional. It should record declared
dimensions, decoded dimensions, processor patch/merge sizes, expected and
observed `image_grid_thw`, raw patch rows, merged visual-token count, and
batch-order index.

The minimum final production acceptance target is full val or benchmark
inference with batched decoding, scored artifacts, and mAP evaluation. Smaller
tiny/smoke runs are implementation gates only; they are not the final evidence
for inference correctness. The first inference OpenSpec must name the evaluator
owner for this acceptance path. Given the self-contained CoordExp-swift goal,
the default should be a minimal rebuilt `src.eval` detection consumer for
`gt_vs_pred_scored.jsonl`; if a legacy/current evaluator bridge is used instead,
the user must explicitly approve that deviation and the OpenSpec must name the
command, row schema, score fields, metric outputs, and compatibility boundary.

`src.infer` produces inference artifacts. It should not directly own metric
reduction in V1. The named `src.eval` consumer or explicit bridge consumes the
scored artifact. This keeps decode and metric reduction separately testable
while keeping the production benchmark acceptance concrete.

Evidence scope: none-yet. This is a proposal-scoped planning decision. It must
be promoted into a new OpenSpec change before implementation begins.

## Inference Source Topology And Review Gates

Resolved on 2026-07-02 during CoordExp-swift inference/decode planning. Patched
after five-agent review convergence on 2026-07-02.

The inference OpenSpec change should be named
`build-coordexp-swift-inference-infra`. The name is intentionally broader than
just decoding because the first useful surface includes config, checkpoint
resolution, backend generation, token traces, parsing, scoring, artifact
contracts, evaluator-consumer compatibility, and production benchmark
acceptance.

The first inference source topology is:

```text
src/inference/
  __init__.py
  runtime.py
  backend.py
  prompt.py
  parsing.py
  scoring.py
  artifacts.py
  pipeline.py
src/infer.py
```

`src/inference/backend.py` owns backend-neutral decode contracts
(`DecodeRequest`, `DecodeResult`, and `TokenTrace`) plus the first HF
`generate_batch` implementation. Keep backend-neutral names independent from
HF-only mechanics. Future vLLM support should reserve `backend`, `backend_mode`,
and `response_family` fields because OpenAI-compatible and ms-swift response
families expose trace fields differently; vLLM implementation remains out of
scope for V1.

`src/inference/runtime.py` is an inference assembler, not the owner of every
lower-level mechanic. Qwen loading and processor identity should stay under
`src/qwen`, adapter setup/reload/status policy under `src/adapters`,
checkpoint alias and delta reload identity under `src/artifacts` and Qwen
helpers, strict config loading under `src/config`, and role-specific inference
setup under `src/inference/runtime.py`.
Inference-facing modules must not import `TrainConfig`, `ResolvedTrainConfig`,
`ResolvedStepSchedule`, `load_train_config()`, or unallowlisted
`src.training.*`; the V1 training-import allowlist is empty unless a source
study and OpenSpec patch name a concrete exception. Shared owner APIs must be
config-neutral enough for inference to call without training-owned wrapper
types.

`src/inference/pipeline.py` owns inference orchestration: dataset iteration,
batched prompt construction, backend calls, parsing, scoring, artifact writing,
and summary counters. It does not run the evaluator in V1.

`src/inference/parsing.py` owns generated-text parsing, object-span salvage, and
invalid-span diagnostics. Coordinate-token recognition, bbox validation, and
norm1000-to-pixel conversion should reuse or deepen shared `src/templates`,
`src/qwen`, and `src/data/geometry` semantics rather than becoming private
inference knowledge. V1 does not add a generic JSON assistant parser.

`src/inference/scoring.py` owns token-trace alignment to parsed object spans,
selected-token score extraction, scored prediction construction, and
score-policy fingerprinting. Its selected-token policy should be tested against
the same compact object schema used by training render/encode code.

`src/inference/artifacts.py` owns inference artifact names and row writers, but
write-once path safety, resolved-config writing, and manifest core behavior
should reuse or deepen `src/artifacts` rather than copying a training-shaped
manager. Inference should follow the training resolved-config convention:
`configs/resolved.json` and `configs/resolved.yaml` under the run directory,
with manifest links, rather than introducing top-level `resolved_config.json`.

The first artifact set is:

- `configs/resolved.json`
- `configs/resolved.yaml`
- `run_manifest.json`
- `summary.json`
- `gt_vs_pred.jsonl`
- `gt_vs_pred_scored.jsonl`
- `gt_vs_pred_scored.jsonl.provenance.json`
- `pred_token_trace.jsonl`
- `parse_diagnostics.jsonl`
- `image_plan.jsonl`

`run_manifest.json` should record artifact paths, config fingerprint,
model/adapter identity, backend, backend mode, response family, dataset
identity, generation config, score policy fingerprint, trace/scoring status,
prompt/template identity, processor identity, and evaluator-consumer status.

The inference OpenSpec should use these delta specs:

- `coordexp-swift-infer-config-runtime`
- `coordexp-swift-infer-backend-trace`
- `coordexp-swift-infer-prompt-parsing`
- `coordexp-swift-infer-scoring-artifacts`
- `coordexp-swift-infer-pipeline`
- `coordexp-swift-infer-benchmark-smoke`

Required source studies before coding are: old `/data/CoordExp/src/infer/*` as
reference-only material, current CoordExp-swift Qwen training
template/processor code, the Transformers `generate` score contract, PEFT
adapter loading/status behavior, special-token embedding delta validation, and
the current eval mAP input contract.

Verification should not rely on mocked backend tests as meaningful evidence.
Mocks may still be used narrowly for pure unit tests when unavoidable, but they
do not count as acceptance gates in this repo. The first useful smoke should be
a real tiny single-image or sample-limited HF/Qwen path, followed by a small
real Qwen sample-limit smoke with adapter loading, then a full benchmark. The
accepted production staging is therefore tiny real smoke -> full benchmark,
with optional intermediate sample-limited real runs only when they reduce debug
cost.

Test-first implementation slices should cover strict infer config validation,
`checkpoint-final` and explicit checkpoint resolution, adapter/delta identity
checks, HF score trace shape and prompt-width alignment, no-resize image-plan
parity, compact parser salvage, selected-token alignment, score formula and
fingerprint, artifact/provenance writing, scored-row cardinality, batched
pipeline behavior, evaluator-consumer compatibility, and production benchmark
acceptance.

OpenSpec must forbid untraced scored outputs. If
`gt_vs_pred_scored.jsonl` is written, trace provenance and a score-policy
fingerprint are required. The OpenSpec should reserve a backend enum value and
backend-neutral decode result contract for future vLLM support, while marking
vLLM implementation out of scope for V1. It should also require a
production-like inference config leaf matching the benchmark setup: base model,
adapter checkpoint, batched decode, full trace, scored output, and named mAP
consumer.

The OpenSpec tasks should use approval gates by contract surface: config/runtime,
backend trace, prompt/image parity, parser/geometry, scoring/artifacts,
eval-consumer compatibility, and production benchmark acceptance. Implementation
should not start from one broad approval that hides changes to public schemas or
artifact semantics.

Evidence scope: none-yet. This is a proposal-scoped planning decision. It must
be promoted into a new OpenSpec change before implementation begins.
