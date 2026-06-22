# Training Infrastructure Greenfield Recursive Detection CE Refactor Design

Status: approved superpowers design scaffold; production implementation has not started.

Date: 2026-05-02

Owner: CoordExp training infrastructure

Target worktree: `codex/compact-detection-sequence`

Primary target method: Random-Permutation `ET-RMP-CE` with support/balance reweighting

Primary target sequence: compact detection sequence using coordinate special tokens, `<|object_ref_start|>`, and `<|box_start|>`

## Purpose

This refactor makes the detection training data path explicit, typed, modular, and template-aware without binding sequence rendering to a particular training objective. The same normalized detection sample should be reusable for standard sorted SFT, random-order SFT, Random-Permutation `ET-RMP-CE`, and future detection-object templates. The target is a greenfield latest-schema training stack, not a backward-compatibility layer around old set-continuation code.

The current code already supports important Stage-1 surfaces, but the responsibilities are entangled. Template rendering, prompt selection, token-type masking, suffix-row construction, legacy set-continuation branch construction, and packing eligibility are spread across dataset builders, prompt helpers, trainers, and config `custom` fields. The refactor should not preserve that shape. It should replace it with a smaller full-sequence teacher-forced pipeline where `object_ordering: random_permutation` exposes prefix states and the trie target builder supplies the only non-hard-CE target distribution.

The design below turns the pipeline into explicit contracts:

```text
raw input sample
-> normalized internal data container
-> template registry / converter
-> rendered prompt + assistant response sequence
-> objective-specific token target preparation
-> standard SFT / random-order SFT / Random-Permutation ET-RMP-CE training
```

## Source Of Truth

The raw input data contract for this refactor is:

```text
public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
```

Repository inspection of that file showed:

```text
rows: 4,951
objects: 36,273
empty-object rows: 0
min objects per row: 1
max objects per row: 55
top-level keys: file_name, height, image_id, images, metadata, objects, width
object keys: bbox_2d, category_id, category_name, coco_ann_id, desc
metadata keys: source, split
```

Every object has exactly the fields needed for detection-object training:

```text
desc
bbox_2d
category_id
category_name
coco_ann_id
```

The `bbox_2d` values are coordinate-token strings in the inspected source file. This matters because compact-template v1 is intentionally coordinate-token-first rather than raw numeric norm1000 text. Numeric norm1000 compact rendering can be added later as a separate capability after explicit tests are in place.

## Design Commitments

- Mathematical correctness is mandatory. Alignment, masks, object-entry trie nodes, single-sequence teacher-forced targets, and support/balance reweighted losses must fail fast on inconsistent spans.
- Raw data parsing, normalization, template rendering, tokenization, objective preparation, packing, and evaluation remain separate concerns.
- The registry is small and explicit. It is not a general template DSL.
- The future codebase is latest-schema only. Do not add dual-read config logic, old compatibility adapters, or reproduction-only objective modules to the new architecture.
- The canonical Stage-1 JSON surface remains a first-class template for ablation and language-prior control. It is named `stage1_json_pretty`; it is not a legacy-compatibility template.
- The compact sequence is a second first-class template, not a mutation of the canonical JSON contract.
- Stage-1 is the first supported implementation surface. Stage-2 rollout-aware training must fail fast for unsupported templates until it is separately ported.
- V1 supports one selected detection-object template per run. Mixed-template per-sample training is a future extension.
- Prompt selection remains explicit, but artifacts should record a derived prompt-template profile to make mismatches visible.
- Evaluation and inference default to strict expected-template parsing. Salvage and auto-detect parsing stay diagnostic opt-ins, not default success paths.
- Candidate-balanced set-continuation, energy/logZ candidate objectives, chunk-level MP, candidate branch CE, suffix-row candidate scoring, explicit prefix sampling, and PEM/margin losses tied to candidate energy ranking are retired. They should not be modeled as first-class modules, training modes, config surfaces, or compatibility paths in the refactor.
- Do not use "set continuation" to mean the old energy-based candidate/chunk objective. If the phrase remains, restrict it to the broad idea of training from partial object states.
- The future objective family should be named around token-level recursive detection CE, for example `recursive_detection_ce`, `random_permutation_et_rmp_ce`, or `et_rmp_ce`.
- Do not describe ET-RMP-CE as "MP on the first token of each object." The correct rule is trie-based at every token position inside the current object-entry span.
- Standardize terminology on `trie`: use `object-entry trie`, `trie node`, `multi-child trie node`, `trie child token`, and `valid child set`. Do not introduce "effect splitting node", "split node", or generic production "branch node" terminology.

## Current State

The current implementation has these important entrypoints:

```text
src/datasets/contracts.py
src/datasets/builders/jsonlines.py
src/common/detection_sequence.py
src/config/prompts.py
src/data_collators/token_types.py
src/sft.py
src/trainers/stage1_set_continuation/
src/config/schema.py
```

Observed current behavior:

- `validate_conversation_record` validates raw JSONL rows but returns a broad cast-like record rather than a rich normalized detection container.
- `JSONLinesBuilder` converts raw rows into messages and assistant payloads, then chooses rendering through `detection_sequence_format`.
- `render_compact_detection_sequence` already exists and supports compact variants, but its capability constraints and parser/mask contracts are not centralized.
- Prompt helpers have compact-specific branches that are separate from the template renderer, including hard-coded bbox wording.
- Legacy candidate-branch set-continuation and current suffix-row `ET-RMP-CE` code are tightly coupled to CoordJSON text construction, separator strings, and structural close spans.
- Token-type computation reserializes JSON/CoordJSON to infer masks rather than consuming authoritative spans from rendering.
- `CustomConfig` is overloaded with data, prompt, runtime, loss, packing, token adaptation, and set-continuation knobs.
- Packing and encoded-cache eligibility are handled in `sft.py` with fingerprints that include some custom fields but do not represent a first-class template contract.

## Pain Points

- Template identity is split across `prompt_variant`, `detection_sequence_format`, inference parsing, eval parsing, token-type masks, and trainer internals.
- Training-mode support is centralized through large switchboards and dict-like metadata instead of typed preparation contracts.
- Template capability constraints are implicit. Compact with non-xyxy, polygon geometry, or numeric coordinate text can drift into runtime errors instead of config-time failures.
- Loss masks can become template-dependent indirectly through string reserialization, which is risky for compact sequences.
- Current `ET-RMP-CE` suffix-row and object-entry trie construction assume CoordJSON grammar. The new design should avoid suffix-row generation entirely and derive trie targets from the rendered full assistant sequence.
- The `custom` config section has become a compatibility catch-all, which makes intent harder to validate and artifact provenance harder to interpret.
- Packing decisions are too intertwined with SFT preprocessing and not explicit enough for future MP or padding-free packed runtime variants.

## Target Architecture

The target code layout is intentionally compact and greenfield-oriented. It keeps
the conceptual seams, but avoids one module per historical mechanism. Prefix
state, suffix tail, remaining-object multiset, trie construction, support/balance
targets, and objective metrics all live inside one recursive detection CE engine.

```text
src/detection/
  data.py
  template.py
  tokenization.py
  objective.py
  packing.py
  evaluation.py

src/config/
  schema.py
  resolve.py
```

Module responsibilities:

- `src/detection/data.py` owns raw-row parsing, typed containers, source-schema validation, normalization, and object provenance.
- `src/detection/template.py` owns the small detection-output template registry, `stage1_json_pretty`, `compact_full`, strict parsing, assistant-local rendered spans, and template capability checks.
- `src/detection/tokenization.py` owns Qwen chat-template application, post-chat-template assistant spans, char-span to token-span alignment, role masks, and label masks.
- `src/detection/objective.py` owns sorted SFT, random-order SFT, Random-Permutation ET-RMP-CE, internal remaining-object multiset state, object-entry trie construction, token-level targets, support/balance math, objective-state weighting, normalization, and metrics metadata.
- `src/detection/packing.py` owns packing eligibility, fingerprints, and packed metadata rules.
- `src/detection/evaluation.py` owns strict expected-template parsing helpers and debug render/parse diagnostics.
- `src/config/resolve.py` owns strict latest-schema resolution and obsolete-key rejection if canonicalization is more than direct schema parsing.

Runtime ownership stays deliberately narrow. `src/sft.py` remains the Stage-1
entrypoint, but latest Stage-1 detection runs should not select behavior through
`trainer_variant`. Instead, resolved `objective.id` decides which
objective-preparation adapter is used. Sorted SFT, random-order SFT, and
recursive detection CE all produce teacher-forced token-level targets for the
same causal-LM forward process. If a small loss adapter is needed for sparse
multi-positive targets, it owns only token-level CE aggregation; it must not
recreate candidate branch scoring, suffix rows, or a separate set-continuation
trainer. Stage-2 specialized trainer routing can remain separate until it is
explicitly ported to these contracts.

The old set-continuation and candidate-scoring entrypoints should be removed from the target architecture. If an archived run ever needs reproduction, that should happen on an archived branch or pinned commit, not through compatibility shims in this new code path.

## Normalized Data Model

The internal detection representation is independent of rendering and objective.

```python
@dataclass(frozen=True)
class RawDetectionRow:
    image_id: str | int
    file_name: str
    width: int
    height: int
    images: list[dict[str, Any]]
    objects: list[RawDetectionObject]
    metadata: DetectionMetadata

@dataclass(frozen=True)
class RawDetectionObject:
    desc: str
    bbox_2d: tuple[str, str, str, str]
    category_id: int
    category_name: str
    coco_ann_id: int

@dataclass(frozen=True)
class DetectionMetadata:
    source: str
    split: str

@dataclass(frozen=True)
class CoordinateTokenBox:
    x1: str
    y1: str
    x2: str
    y2: str

@dataclass(frozen=True)
class NormalizedDetectionObject:
    object_index: int
    desc: str
    geometry: CoordinateTokenBox
    category_id: int
    category_name: str
    coco_ann_id: int

@dataclass(frozen=True)
class NormalizedDetectionSample:
    image_id: str | int
    file_name: str
    width: int
    height: int
    images: tuple[dict[str, Any], ...]
    objects: tuple[NormalizedDetectionObject, ...]
    metadata: DetectionMetadata
```

Validation rules:

- Reject rows without `images`, `objects`, `width`, `height`, `image_id`, `file_name`, and `metadata`.
- Reject object entries missing `desc` or `bbox_2d`.
- Reject compact v1 if bbox geometry is not exactly four coordinate-token strings.
- Reject rows with object order mutations after normalization unless an explicit ordering strategy is applied and recorded.
- Preserve source object provenance and normalized object index separately.

Future geometry types can be added by extending `DetectionGeometry`, but compact v1 must keep its narrow coordinate-token contract.

## Rendered Sequence Model

Rendered output must carry authoritative spans. Training code should not infer object spans by reserializing payloads after the fact.

```python
TemplateId = Literal[
    "stage1_json_pretty",
    "compact_full",
]

@dataclass(frozen=True)
class TemplateCapabilities:
    supports_sft: bool
    supports_recursive_detection_ce: bool
    supports_et_rmp_ce: bool
    supports_static_packing: bool
    coordinate_surface: Literal["coord_token", "norm1000_text"]
    geometry_kinds: frozenset[str]
    object_separator: str
    terminal_close: str

@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int

@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

@dataclass(frozen=True)
class RenderedObjectEntry:
    object_instance_id: str
    object_index: int
    source_object_index: int
    entry_span: CharSpan
    object_ref_start_span: CharSpan | None
    desc_span: CharSpan | None
    bbox_start_span: CharSpan | None
    bbox_span: CharSpan | None
    coord_spans: tuple[CharSpan, CharSpan, CharSpan, CharSpan] | None
    separator_span: CharSpan | None
    structural_spans: tuple[CharSpan, ...]
    control_spans: tuple[CharSpan, ...]
    trie_eligible_span: CharSpan

@dataclass(frozen=True)
class RenderedAssistantSequence:
    template_id: str
    template_version: str
    text: str
    object_entries: tuple[RenderedObjectEntry, ...]
    structural_spans: tuple[CharSpan, ...]
    terminal_span: CharSpan | None
    eos_or_stop_span: CharSpan | None

@dataclass(frozen=True)
class RenderedConversation:
    prompt_text: str
    assistant: RenderedAssistantSequence

@dataclass(frozen=True)
class TokenizedObjectEntry:
    object_instance_id: str
    object_index: int
    source_object_index: int
    entry_span: TokenSpan
    desc_span: TokenSpan | None
    bbox_span: TokenSpan | None
    coord_spans: tuple[TokenSpan, TokenSpan, TokenSpan, TokenSpan] | None
    separator_span: TokenSpan | None
    trie_eligible_span: TokenSpan

@dataclass(frozen=True)
class TokenizedRenderedConversation:
    rendered: RenderedConversation
    full_text: str
    input_ids: tuple[int, ...]
    assistant_span_after_chat_template: CharSpan
    object_entries: tuple[TokenizedObjectEntry, ...]
    structural_spans: tuple[TokenSpan, ...]
    terminal_span: TokenSpan | None
    eos_or_stop_span: TokenSpan | None
```

The renderer is responsible for:

- Producing the assistant response text.
- Returning assistant-local spans for object entries, object-start markers, desc fields, bbox-start markers, bbox fields, coordinate tokens, separators, terminal structure, and template-defined stop markers when they are part of the assistant text.
- Reporting capability constraints before tokenization.
- Providing strict parsing for model outputs generated under the same template.
- Preserving object provenance so exact duplicate rendered entries remain separate instances in the remaining-object multiset.

The tokenizer alignment layer is responsible for:

- Mapping character spans onto token spans.
- Adding the full assistant span after Qwen chat-template rendering.
- Failing if a span boundary cannot be represented in token space under the configured tokenizer.
- Building token-role masks from template-provided spans.
- Treating control spans as role metadata, not as a static hard-CE policy; hard-vs-MP is decided by the object-entry trie target builder at each token position.

## Template Registry

The registry maps a small stable template ID to a concrete spec:

```python
class DetectionSequenceTemplate(Protocol):
    template_id: str
    capabilities: TemplateCapabilities

    def validate_sample(self, sample: NormalizedDetectionSample) -> None: ...
    def render_assistant(self, sample: NormalizedDetectionSample) -> RenderedAssistantSequence: ...
    def parse_assistant(self, text: str) -> ParsedAssistantSequence: ...
    def render_entry(self, obj: NormalizedDetectionObject) -> RenderedObjectEntry: ...
    def render_separator(self, before_index: int, after_index: int) -> str: ...
    def render_terminal_close(self) -> str: ...
```

V1 templates:

| Template ID | Purpose | SFT | Random-Permutation ET-RMP-CE | Notes |
| --- | --- | --- | --- | --- |
| `stage1_json_pretty` | Canonical Stage-1 JSON object template with strict closure | yes | yes, after parity tests | Current JSON baseline; preserves model's pretrained language prior more than compact |
| `compact_full` | Compact sequence with desc and bbox tokens | yes | yes, after span/trie/support-balance tests | Coord-token xyxy only in v1 |

Standard pretty JSON grammar:

```text
response =
{
  "objects": [
    {
      "desc": <desc>,
      "bbox_2d": [<x1>, <y1>, <x2>, <y2>]
    },
    {
      "desc": <next_desc>,
      "bbox_2d": [<next_x1>, <next_y1>, <next_x2>, <next_y2>]
    }
  ]
}
```

The order is JSON object list -> object `desc` -> object `bbox_2d` -> next
object `desc` -> next object `bbox_2d` -> strict ending closure. This is not a
new JSON variant: `stage1_json_pretty` names the current canonical Stage-1 SFT
JSON behavior. The registry must preserve this structure so compact-vs-canonical
JSON ablations can test whether compact hurts performance by moving away from
the model's pretrained JSON/language prior.

`stage1_json_pretty` is a pinned v1 profile, not a family name. Its parity
fixture must record:

- `object_field_order: desc_first`.
- `coordinate_surface: coord_token`.
- `bbox_format: xyxy`.
- The exact canonical spacing, newline, quote, comma, and strict closure shape
  emitted by the current Stage-1 SFT path.

Other existing JSON-like surfaces, including `geometry_first` and numeric
norm1000 JSON, are out of scope for `stage1_json_pretty` parity unless they are
added later as separately named template profiles with their own fixtures.

Compact v1 row grammar:

```text
entry(object) = <|object_ref_start|>{desc}<|box_start|>{x1}{y1}{x2}{y2}
separator = "\n"
terminal_close = ""
final supervision = template-defined terminal stop span(s) after last compact row
```

This gives `ET-RMP-CE` a clean single-sequence grammar:

- The full assistant response is rendered once in the selected object order.
- Teacher forcing walks the rendered response left to right.
- Separator, terminal close, EOS, and chat-template stop positions come from template spans.
- There is no JSON `]}` close for compact; any code assuming a global close string must use the template grammar instead.

## Prompt Compatibility

Prompt text and assistant sequence template are separate but coupled by validation.

Proposed prompt config:

```yaml
prompt:
  system_variant: stage1_detection
  user_variant: dense_detection
  include_template_summary: true
  template_summary_style: compact_full
```

Proposed detection-template config:

```yaml
detection_template:
  id: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy
  object_fields: [desc, bbox_2d]
  object_separator: newline
  strict_parse: true
```

Validation rules:

- Keep the existing framework `template:` section reserved for Qwen/ms-swift
  chat-template runtime settings. Detection-output grammar lives under
  `detection_template:`.
- Reject `detection_template.id=compact_full` with a JSON-schema prompt summary unless explicitly allowed for a diagnostic run.
- Reject compact prompts that describe numeric `x1 y1 x2 y2` when the template emits coordinate special tokens.
- Record the resolved prompt-template profile in run artifacts.
- Do not infer template ID from prompt prose at runtime.

## Objective Preparation

Objective adapters consume normalized samples plus tokenized rendered assistant sequences. They do not own raw data parsing or template string construction.

```python
@dataclass(frozen=True)
class TokenizedTrainingExample:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    rendered: TokenizedRenderedConversation
    masks: LossMaskBundle
    metadata: TrainingExampleMetadata

@dataclass(frozen=True)
class LossMaskBundle:
    assistant_label_mask: tuple[bool, ...]
    coord_label_mask: tuple[bool, ...]
    structural_label_mask: tuple[bool, ...]
    object_entry_label_mask: tuple[bool, ...]
    desc_label_mask: tuple[bool, ...]

@dataclass(frozen=True)
class TrainingModeSpec:
    mode_id: str
    required_template_capabilities: frozenset[str]
    prepare: Callable[..., TrainingModeBatch]
```

SFT preparation:

- Render conversation through the selected template.
- Tokenize once.
- Build labels and loss masks from aligned spans.
- Apply static packing only if the template and loss masks declare static-packing compatibility.

Random-order SFT preparation:

- Select a seeded random object permutation before rendering.
- Render one ordinary assistant sequence in that order.
- Train with hard CE everywhere.
- Record the permutation seed, source object indices, and rendered object-entry order in metadata.

Random-Permutation `ET-RMP-CE` preparation:

- Keep full-vocabulary support/balance reweight semantics unchanged.
- Use dynamic random object permutations to render ordinary teacher-forced target sequences.
- Build object-entry trie targets from the same rendered object-entry token sequences.
- Do not build separate suffix rows, candidate chunks, candidate-energy scores, or explicit prefix samples.
- Outside `trie_eligible_span`, always use hard CE for schema, separator, terminal, assistant-end, EOS, and chat-stop positions.
- Inside `trie_eligible_span`, decide hard CE versus support/balance CE only from the object-entry trie valid child set.
- Fail fast if support/balance masks include tokens outside intended object-entry trie support, balance, or structural spans.

The important unification is that `sft`, `random_order_sft`, and
`random_permutation_et_rmp_ce` all prepare the same kind of tokenized
full-sequence example. They differ only in object ordering and in whether the
target builder replaces one-hot CE with trie support/balance CE at true
multi-child object-entry trie nodes. This keeps the autoregressive forward
process identical across modes.

## Random-Permutation ET-RMP-CE

The future architecture should not be energy-based set continuation with
candidate chunk scoring. It should be token-level recursive/permutation
detection CE: a standard teacher-forced causal-LM sequence loss with sparse
multi-positive target distributions at multi-child object-entry trie nodes.

For each training sample:

1. Choose a seeded object permutation. For `random_permutation_et_rmp_ce`, this
   is dynamic random order; sorted/source/fixed order must be separate explicit
   modes or ablations.
2. Render the target assistant sequence through the selected detection template.
3. Train left-to-right with ordinary teacher forcing.
4. At ordinary singleton next-token positions, use hard CE.
5. Inside the current object-entry span, build a trie over all currently
   remaining un-emitted objects rendered by the same template.
6. At each object-entry token position, follow the teacher-forced prefix of the
   currently selected object entry to a trie node, then restrict active trie
   objects to those still compatible at that node.
7. If the valid next-token set has one token, use ordinary hard CE.
8. If the valid next-token set has multiple tokens, use multi-positive CE /
   support-balance CE.
9. Continue along the teacher-forced selected object.
10. After the selected object is fully emitted, remove exactly that object
    instance from the remaining multiset and rebuild the trie for the next
    object-entry state. Exact duplicate rendered entries remain separate object
    instances through stable provenance even when their token path is identical.

This rule is trie-based, not slot-based. It must not be described as MP on the
first object token. In compact format, every object may share
`<|object_ref_start|>` and many same-class objects may share desc/category
tokens. The first true branch may occur at `<|box_start|>` or at coordinate
tokens such as `x1`, `y1`, `x2`, or `y2`. Entry-internal control tokens such as
`<|object_ref_start|>` and `<|box_start|>` may be hard CE or support/balance CE
depending on the trie. Separators, final JSON closure, compact terminal stop
spans, assistant-end markers, EOS, and chat-template stop tokens are outside the
object-entry trie and remain hard CE by construction.

The implementation rule is `len(valid_next_tokens)`. Token role is diagnostic
metadata, not the source of truth for the target distribution. For example:

- The compact root token is usually hard CE even with many remaining objects,
  because all valid entries share `<|object_ref_start|>`.
- Same-class objects may keep desc tokens and `<|box_start|>` hard CE until
  the first divergent coordinate token.
- A desc-prefix collision can make `<|box_start|>` one valid trie child while
  another object entry continues the desc text, so `<|box_start|>` must not be
  hard-coded as always-hard.
- Identical rendered entries may produce no multi-child trie node at all; the selected
  teacher instance is still removed exactly once after its full entry span is
  emitted.

## Unified Autoregressive Objective

The cleanest formulation is one teacher-forced autoregressive forward process:

```text
objects O_i = {o_1, ..., o_n}
permutation pi_i ~ object_ordering
target y_i = render_template(prompt_i, o_{pi_i(1)}, ..., o_{pi_i(n)})

L_i = sum_t alpha(i,t) * CE_or_trie_CE(theta, y_{i,t} | y_{i,<t}, image_i)
```

The target builder carries a small internal state while scanning `y_i`:

```text
remaining multiset R_t
current teacher object o*
teacher-forced entry-prefix tokens u_<j
object-entry trie T(R_t)
valid child set V_t at the current trie node
```

At non-object-entry positions, `CE_or_trie_CE` is ordinary hard CE. At
object-entry positions, `V_t` is computed from `T(R_t)` and the teacher-forced
entry prefix. If `|V_t| = 1`, the loss is ordinary hard CE. If `|V_t| > 1`, the
loss is support/balance multi-positive CE over `V_t`. After the current teacher
object entry is fully emitted, exactly that object instance is removed from
`R_t`.

This is equivalent to the useful token-level part of the prior ET-RMP-CE idea
when suffix rows were only a way to expose the sequence of remaining-set states
that already appear inside one randomly permuted teacher-forced sequence:

```text
empty prefix -> after 1 emitted object -> ... -> after n-1 emitted objects
```

Exact equivalence still depends on both state exposure and reduction. If the
prior objective sampled an explicit prefix length `K` and averaged only the
supervised suffix row, the exact compiled full-sequence loss is:

```text
L_old(x) = E_{pi,K}[
  (1 / |U_{pi,K}|) * sum_{t in U_{pi,K}} l_t
]

L_compiled(x) = E_pi[sum_t alpha_t(x, pi) * l_t]

alpha_t(x, pi) = E_K[
  1{t in U_{pi,K}} / |U_{pi,K}|
]
```

where `U_{pi,K}` is the supervised token set that the old sampled row would have
trained. This preserves the old recursive-state exposure and row-mean reduction
without keeping a runtime prefix sampler or suffix-row generator. This exact
profile should be named something like
`legacy_row_mean_prefix_mixture_equivalence`.

`alpha_t` parity is necessary but not sufficient for objective equivalence. The
new full-sequence builder must also preserve local target parity for every
legacy sampled state `(pi, K)` and every `t in U_{pi,K}` under the canonical
`stage1_json_pretty` fixture: the same local loss term `l_t`, the same hard-CE
versus object-entry trie CE decision, the same valid child set `V_t`, and the
same multiplicity-weighted `q(v)` after chat-template tokenization. A test that
matches only compiled position weights can still be mathematically wrong if it
moves MP to the wrong token or changes trie children under context-sensitive
tokenization.

If the implementation intentionally changes reduction away from old row-mean
behavior, a simpler exposure profile is possible:

```text
entry_state_exposure(n, j) = Pr(K < j)
separator_state_exposure(n, j) = Pr(K < j)
terminal_state_exposure(n) = 1
```

This is easier to reason about, but it is not strict equality to the current
implementation's row-reduced objective. Setting every state weight to `1.0` is
the cleaner plain random-permutation objective, but it is also an objective
change and must be treated as an ablation rather than claimed as exact
equivalence.

For the current documented prefix mixture
`empty/random_subset/leave_one_out/full_prefix = 0.30/0.45/0.20/0.05`, the
simplified exposure profile would be explicit. For `n > 1`:

```text
entry_state_exposure(n, j)
  = 0.30
    + 0.45 * (j - 1) / (n - 1)
    + 0.20 * 1[j = n]

separator_state_exposure(n, j) = entry_state_exposure(n, j)
terminal_state_exposure(n) = 1.0
```

This means the first emitted object and opening schema/control tokens that only
appear under an empty prefix receive weight `0.30`, the final object receives
weight `0.95`, and terminal close/EOS receives weight `1.0`. Singleton samples
need an explicit degenerate rule because `leave_one_out` collapses to an
empty-prefix state; do not infer it from the `n > 1` formula. The parity test
must cover this edge case and record the exact singleton weight used by the
current implementation. Exact parity tests must additionally include the
`1 / |U_{pi,K}|` row-reduction factor from `alpha_t`; exposure-only tests are
useful diagnostics, not full objective identity.

Template parity is also scoped. The weighted full-sequence rewrite can target
parity with the current canonical JSON surface. `compact_full` changes terminal
and closure semantics because compact has newline separators and EOS/chat stop
but no JSON `]}` closure. Compact should therefore be treated as a paired
template/objective surface, not as proof of byte- or stop-equivalence with
`stage1_json_pretty`.

The simplified formulation is not equivalent to candidate-energy/logZ scoring,
candidate chunk ranking, explicit runtime subset-prefix sampling, or separately
weighted suffix-row objectives. Those behaviors are intentionally retired
because they either conflict with the causal-LM next-token interface or add
duplicate runtime machinery. The only "remaining-object" concept kept in the new
codebase is the private multiset used by the trie target builder for the
currently rendered sequence.

## Support/Balance Loss

For a multi-child trie node with valid next-token set `V`:

```text
P_valid = sum_{v in V} p_theta(v | context)

L_valid_support = -log(P_valid)

L_valid_balance = - sum_{v in V} q(v) * log(p_theta(v | context) / P_valid)

L_branch = trie_support_weight * L_valid_support
         + trie_balance_weight * L_valid_balance
```

where `q(v)` is object-multiplicity-uniform under the remaining-object trie:

```text
q(v) = number of active remaining objects under child token v
       / number of active remaining objects at this trie node
```

Important semantics:

- `trie_support_weight=1.0` and `trie_balance_weight=1.0` reproduce sparse object-uniform soft CE at multi-child trie nodes.
- `trie_support_weight=2.0` and `trie_balance_weight=1.0` is the support-reweighted variant intended to increase valid-continuation mass without explicit stop-token suppression or decode hacks.
- The main objective remains full-vocabulary token-level CE, including coordinate tokens.
- Coord-vocabulary-normalized chunk scores are not part of the main `ET-RMP-CE` objective.
- Hard structural supervision remains explicit for positions outside `trie_eligible_span`, including schema, separator, terminal, assistant-end, EOS, and chat-stop tokens.
- Loss masks remain aligned to tokenizer positions rather than approximate string slices.

## Length-Insensitive Loss Normalization

Length-insensitive normalization is important for the future compact-vs-JSON
training stack, but it is not a behavior-preserving refactor by itself. It
changes the aggregation layer even when the token-level trie targets are
unchanged. Therefore the implementation should support an equivalence profile
that preserves the current denominator for parity tests, and a separate
`semantic_image_bucket_balanced` profile for the intended production ablation.

Raw token averaging is not an acceptable default for the refactor. If the loss
is

```text
L_token_mean = sum_i sum_t loss(i,t) / sum_i T_i
```

then the effective weight of image `i` is proportional to its rendered target
length `T_i`, and the effective weight of a template component is proportional
to the number of tokens used to spell that component. That is exactly the wrong
invariance for comparing canonical Stage-1 JSON against compact rows: JSON can
receive extra weight for braces, quotes, keys, indentation, and closure tokens,
while compact can receive a different coordinate/control ratio simply because it
uses fewer natural-language/JSON tokens. Token averaging also makes loss curves
difficult to interpret across object-count buckets and can hide early-stop bias
behind denominator changes.

The target compact-candidate profile is semantic image-balanced normalization:

1. The objective builder emits typed `LossAtom` records. Every supervised token
   belongs to exactly one semantic role, for example `desc_identity`,
   `bbox_coord`, `object_control`, `entry_trie_branch`, `separator_continue`,
   `terminal_stop`, `schema_control`, or `chat_stop`.
2. Token losses are first averaged inside a semantic atom or span, so a longer
   JSON spelling does not automatically outweigh a compact spelling:

```text
L_atom(a) = sum_{t in tokens(a)} token_weight(t) * loss(t)
            / sum_{t in tokens(a)} token_weight(t)
```

3. Object losses are then averaged across semantic object parts:

```text
L_object(i,k) =
  sum_{a in object_atoms(i,k)} role_weight(role(a)) * L_atom(a)
  / sum_{a in object_atoms(i,k)} role_weight(role(a))
```

4. Image object loss is object-balanced, not token-length-balanced:

```text
L_objects(i) = (1 / max(1, num_objects_i)) * sum_k L_object(i,k)
```

5. Boundary and terminal losses are normalized as semantic decisions, not raw
   token counts:

```text
L_continue(i) = mean separator_continue losses when num_objects_i > 1
L_stop(i)     = terminal_stop / EOS / chat-stop loss
L_boundary(i) = mean of present boundary-decision components
```

6. The final image loss is a fixed semantic mixture:

```text
L_image(i) =
  lambda_objects  * L_objects(i)
  + lambda_boundary * L_boundary(i)
  + lambda_schema   * L_schema_control(i)
```

where missing components are omitted and the present component weights are
renormalized. The initial recommended constants should live in code as the
definition of the objective, not as a pile of YAML knobs. Config should expose
only the selected normalization strategy unless an ablation explicitly requires
a new strategy ID.

7. Batch loss is image-balanced and may optionally be GT-count-bucket balanced
for high-count continuation experiments:

```text
L_batch = sum_i bucket_weight(bucket(num_objects_i)) * L_image(i)
          / sum_i bucket_weight(bucket(num_objects_i))
```

The recommended first production strategy is
`semantic_image_bucket_balanced`, with fixed GT-count buckets matching the eval
diagnostics, for example `0-3`, `4-6`, `7-10`, and `11+`. This directly targets
short-sequence bias: many low-object-count images should not dominate the
training signal that teaches the model to continue emitting valid remaining
objects in high-count scenes.

Recommended v1 constants:

```text
Object semantic-role weights:
  desc_identity       = 0.35
  bbox_coord          = 0.45
  entry_trie_decision = 0.15
  object_control      = 0.05

Boundary semantic-role weights:
  separator_continue  = 0.50
  terminal_stop       = 0.50

Image mixture weights:
  lambda_objects      = 1.00
  lambda_boundary     = 0.30
  lambda_schema       = 0.10

GT-count buckets:
  0-3, 4-6, 7-10, 11+

Bucket weighting:
  bucket_weight(b) = clipped_inverse_dataset_frequency(b)
  max_bucket_weight_ratio = 4.0
```

Missing semantic components are omitted and the present weights are
renormalized within their local mixture. For example, pure SFT rows without
multi-child trie nodes omit `entry_trie_decision`; a one-object image omits
`separator_continue` and keeps `terminal_stop` as the boundary decision. The
bucket weights are computed once from the training set, recorded in the run
manifest, clipped to avoid unstable rare-bucket explosions, and then used for
all training batches in that run.

These constants are intentionally fixed v1 objective constants, not a broad YAML
surface. Changing them should create a new `loss_normalization.strategy` ID so
metrics and artifacts remain comparable.

Normalization invariants:

- Do not divide the main training loss by total rendered token count.
- Do not let canonical JSON structural length change object, coordinate, or trie
  supervision weight relative to compact.
- Do not let each image's single terminal stop token dominate low-count scenes
  or disappear inside long JSON strings.
- Report denominator diagnostics for every run: atom counts, object counts,
  semantic-role token counts, multi-child trie-token fraction, boundary/stop
  fraction, effective weighted trie contribution, and loss by GT-count bucket.
- Keep terminal/control CE hard and explicit; length-insensitive normalization
  changes weighting, not the target distribution.

Template-specific differences:

| Surface | Canonical Stage-1 JSON | Compact full |
| --- | --- | --- |
| Entry prefix | Pretty-printed JSON object text | `<|object_ref_start|>` |
| Coordinate opener | `"bbox_2d": [` | `<|box_start|>` |
| Entry separator | comma plus pretty JSON whitespace | newline |
| Final close | JSON strict closure | EOS |
| Entry-trie unit | serialized object entry | compact object row |

Compact `ET-RMP-CE` is appropriate only when:

- The selected tokenizer treats coordinate special tokens, `<|object_ref_start|>`, and `<|box_start|>` as stable atomic or validated token spans.
- Object-entry trie alternatives are rendered by the same template that rendered the full assistant sequence.
- Causal training contexts are generated from normalized object order plus template grammar, not by slicing arbitrary previous output.
- Close/EOS supervision is validated for each object-count boundary.
- Support/balance masks match the valid remaining-object trie children exactly.

## Why This Is The Forward Architecture

The simplified design is valuable because:

- It aligns with the causal-LM training interface: next-token prediction.
- It removes unstable candidate-energy, logZ, and whole-chunk comparison machinery.
- It keeps ordinary SFT infrastructure mostly intact.
- It makes the objective a loss-mask / target-distribution modification rather than a runtime branch-forward scorer.
- It makes compact-sequence support easier because the objective depends on template-provided spans and trie metadata, not hardcoded JSON fragments.
- It naturally handles permutation invariance without branch-forward candidate scoring.
- It trains every permutation-prefix state along a shuffled sequence in a single teacher-forced forward pass.
- It provides a clean support-reweight knob for valid continuation mass.
- It keeps terminal/control tokens supervised by hard CE, reducing hidden decode-time mismatch.
- It is more compatible with efficient static, dynamic, and future padding-free packing work than candidate-branch energy scoring.

## No Explicit Runtime Prefix Sampling

Dynamic random object ordering already exposes the model to the useful prefix
states inside one normal teacher-forced pass:

```text
empty prefix
first object
first two objects
...
all but final object
```

Across repeated random permutations, the prefix of length `k` approximates a
random subset of size `k`. Therefore, one standard left-to-right pass can train
many continuation states without runtime candidate-branch sampling.

The latest architecture removes explicit random subset-prefix sampling as a
runtime mode. Do not add `prefix_sampling`, `prefix_conditioning`, or
explicit-prefix ET-RMP-CE config knobs to the latest config surface. If exact
parity with an earlier prefix-mixture objective is required for a test, compile
that mixture into deterministic prefix-position weights inside
`recursive_detection_ce`; do not resurrect a separate sampler, suffix-row path,
or candidate-branch runtime.

## Span Contract For Random-Permutation ET-RMP-CE

The rendering and tokenization layers together must expose enough metadata for
the token-level target builder:

- Rendered object-entry spans.
- Assistant-local character spans before chat-template expansion.
- Token positions eligible for object-entry trie target building.
- Control token spans for role labeling; these are not a static hard-CE-only list.
- Explicit object-start marker spans, including `<|object_ref_start|>` for compact templates.
- Explicit bbox-start marker spans, including `<|box_start|>` for compact templates.
- Object boundary and separator spans.
- Coordinate slot spans.
- Terminal, EOS, and chat-template stop spans.
- Object identity and desc spans.
- Bbox spans.
- Object provenance linking rendered entries back to normalized object instances.
- Template identity, template version, coordinate surface, bbox format, object separator, and terminal policy.

The tokenization/alignment layer then adds the full assistant span after Qwen
chat-template rendering and maps assistant-local character spans to token spans.
The template layer must not claim ownership of absolute post-chat-template
offsets before chat-template expansion has happened.

For positions outside `trie_eligible_span`, always use hard CE. For positions
inside `trie_eligible_span`, the decision is dynamic per token position:
`len(valid_next_tokens) == 1` means hard CE, and `len(valid_next_tokens) > 1`
means support/balance CE. In compact format, shared `<|object_ref_start|>` and
shared `<|box_start|>` tokens are usually hard CE, while coordinate tokens may
become multi-child trie nodes when same-class or otherwise identical-prefix
remaining objects first diverge at bbox coordinates. Conversely,
`<|box_start|>` can itself be a trie child in a desc-prefix collision; no token
string should be globally hard-coded. Separators, final JSON closure, compact
terminal stop spans, assistant-end markers, EOS, and chat-template stop tokens
remain hard CE because they are outside the object-entry trie.

The compact renderer must expose final EOS/stop supervision as explicit
terminal metadata because `compact_full` has no CoordJSON `]}` closure. Compact
ET-RMP-CE is blocked until rendered continuation fixtures prove that compact
variants do not contain CoordJSON keys or JSON closure fragments.

## Packing Design

Packing remains a separate layer from template rendering and objective construction.

Static SFT packing:

- Eligible for templates whose tokenized examples have independent labels, masks, and metadata boundaries.
- Fingerprint includes template ID, template version, prompt profile, tokenizer ID, coordinate surface, object ordering, loss mask version, and preprocessing version.

Recursive detection CE packing:

- V1 keeps recursive detection CE unpacked unless packing can preserve example boundaries, token ranges, label masks, and trie target metadata exactly.
- Padding-free packed runtime is treated as an explicit runtime optimization with its own validation.
- Packing must not reintroduce candidate scoring, suffix-row generation, or prefix sampling as an efficiency workaround.
- Any future ET-RMP-CE packing must preserve image boundary, rendered sequence boundary, valid remaining-object multiset state, label masks, and object-entry trie target metadata.

Required packed metadata:

```python
@dataclass(frozen=True)
class PackedSequenceMetadata:
    packed_example_id: str
    source_example_ids: tuple[str, ...]
    token_ranges: tuple[TokenSpan, ...]
    label_ranges: tuple[TokenSpan, ...]
    image_ranges: tuple[TokenSpan, ...]
    trie_metadata_ranges: tuple[TokenSpan, ...]
    template_id: str
    training_mode: str
```

## Config Hierarchy

The target config surface should expose only the latest typed knobs. Obsolete
knobs should be removed rather than carried forward as hidden aliases. The new
codebase does not support archived-config migration or legacy dual-read paths.
Production training should fail fast on old `custom`, `trainer_variant`,
`stage1_set_continuation`, `prefix_conditioning`, `legacy_candidate_branch`,
`candidate_balanced`, `branch_support_weight`, `branch_balance_weight`, or
explicit prefix-sampling knobs.

Proposed top-level sections keep framework runtime config separate from
detection-task config. The existing framework `template:` namespace remains for
Qwen/ms-swift chat-template/runtime settings. Detection-output rendering uses
`detection_template:` so the config loader never has to guess which template
layer a key belongs to.

```yaml
model:
  # Existing model/runtime model settings.

template:
  # Existing Qwen/ms-swift chat-template settings. Not the detection output grammar.

training:
  # Existing TrainArguments-style framework knobs: output, optimizer, batch,
  # schedule, checkpoint, save/eval cadence, and distributed runtime settings.

deepspeed:
  # Existing framework runtime section when used.

data:
  train_jsonl: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
  val_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
  image_root: public_data/coco
  max_objects: 60
  object_ordering: random_permutation

prompt:
  system_variant: stage1_detection
  user_variant: dense_detection
  include_template_summary: true

detection_template:
  id: stage1_json_pretty  # or compact_full for the paired compact ablation
  coordinate_surface: coord_token
  bbox_format: xyxy
  object_field_order: desc_first
  strict_parse: true

objective:
  id: recursive_detection_ce
  variant: random_permutation_et_rmp_ce
  forward_process: teacher_forced_full_sequence
  hard_ce_weight: 1.0
  entry_trie_rmp_ce_weight: 1.0
  trie_support_weight: 2.0
  trie_balance_weight: 1.0
  structural_ce_weight: 1.0
  state_weighting: legacy_row_mean_prefix_mixture_equivalence
  normalization: legacy_row_mean_equivalence

packing:
  static_packing: false
  padding_free_packed: false

evaluation:
  expected_template: stage1_json_pretty
  parser_mode: strict_expected

validation:
  validate_span_alignment: true
  validate_template_capabilities: true

debug:
  dump_rendered_examples: 8
  dump_alignment_examples: 8
```

Recommended profile split:

```yaml
profiles:
  parity_stage1_json_pretty:
    detection_template:
      id: stage1_json_pretty
      coordinate_surface: coord_token
      bbox_format: xyxy
      object_field_order: desc_first
    objective:
      state_weighting: legacy_row_mean_prefix_mixture_equivalence
      normalization: legacy_row_mean_equivalence

  compact_candidate:
    detection_template:
      id: compact_full
      coordinate_surface: coord_token
      bbox_format: xyxy
    objective:
      state_weighting: uniform_permutation
      normalization: semantic_image_bucket_balanced
```

Config validation rules:

- New training configs use only explicit typed sections.
- Obsolete or legacy keys fail fast rather than being silently dual-read.
- There is no archived config migration path in the new codebase.
- `detection_template:` is the only section allowed to select
  `stage1_json_pretty` or `compact_full`; framework `template:` keys must not
  affect detection rendering.
- Run manifests record both raw input config and resolved canonical config.
- Cache fingerprints use canonical resolved config fields.

## Validation Strategy

Validation must start narrow and become progressively more integrated.

Gate-0 render/objective contract:

- For every requested detection sequence format, rendered text must match the selected template grammar before any objective is enabled.
- `compact_full` must render compact rows with `<|object_ref_start|>`, `<|box_start|>`, coordinate special tokens, newline separators, and EOS/chat stop supervision, not JSON keys or `]}` closure.
- `stage1_json_pretty` must preserve strict JSON closure and the canonical object list -> `desc` -> `bbox_2d` -> next object structure.
- The rendered-text fixture must include object-entry spans, token spans, separator spans, terminal/EOS spans, trie-eligible spans, and object provenance.
- Random-Permutation `ET-RMP-CE` remains disabled for a template until the rendered sequence and span fixtures prove that trie targets are derived from that template's own rendered entries.

Unit-level validation:

- Raw schema smoke on `val.coord.jsonl`.
- Normalized sample round-trip from raw row to typed container.
- Template rendering for `stage1_json_pretty` and `compact_full`.
- Strict parser round-trip for each template.
- Char-span and token-span alignment for coordinate tokens and control tokens.
- Loss-mask construction from spans.

Objective-level validation:

- SFT example construction parity for `stage1_json_pretty`.
- Pretty JSON grammar fixture preserving object list -> `desc` -> `bbox_2d` -> next object -> strict ending closure.
- Compact SFT rendered text and mask snapshot.
- Random-order SFT matches ordinary hard CE when trie targets are disabled.
- Random-Permutation `ET-RMP-CE` position-weight parity tests reproduce the old prefix-exposure mixture when the equivalence profile is selected.
- Position-weight parity tests cover opener/control tokens, separators, close/EOS, `n > 1` object positions, and the singleton-object degeneracy.
- Local target parity fixtures for `stage1_json_pretty`, packing disabled, prove the new full-sequence builder preserves hard-CE versus trie-CE decisions, valid child sets `V_t`, and multiplicity weights `q(v)` for every legacy sampled state represented by the frozen fixture.
- Plain random-permutation `ET-RMP-CE` with all state weights `1.0` is tested separately and labeled as an objective ablation.
- Support/balance mask sums and valid-child counts on controlled toy samples.
- Same-class objects that share desc tokens branch at bbox or coordinate tokens.
- Shared object-start and bbox-start markers remain hard CE when all active objects share them.
- Desc-prefix collision tests prove `<|box_start|>` can be a trie child when one object desc ends and another compatible desc continues.
- Exact duplicate rendered entries remain distinct object instances and are removed one teacher instance at a time.
- Equivalence-normalization tests prove the new full-sequence path can preserve the old denominator when requested.
- Semantic-normalization tests prove `semantic_image_bucket_balanced` is reported as a distinct objective profile.

Packing validation:

- Static SFT packing cache fingerprint parity for canonical sorted SFT.
- Static SFT packing rejection for unsupported template/mask combinations.
- Recursive detection CE packing rejection until trie metadata preservation is implemented.
- Padding-free packed runtime validation for explicit runtime variant only.

Smoke validation:

```bash
conda run -n ms python -m pytest \
  tests/test_detection_raw_schema_contract.py \
  tests/test_detection_template_registry.py \
  tests/test_detection_template_span_alignment.py

conda run -n ms python -m pytest \
  tests/test_latest_training_config_contract.py \
  tests/test_sft_preparation_contract.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_random_permutation_et_rmp_ce_contract.py
```

## Greenfield Implementation Plan

Phase 1 creates `src/detection/data.py` with typed raw/normalized containers,
source-schema validation, normalization, and object provenance.

Phase 2 creates `src/detection/template.py` with the small template registry,
canonical `stage1_json_pretty`, `compact_full`, strict parsers, and authoritative
rendered spans.

Phase 3 creates `src/detection/tokenization.py` with chat-template application,
char-span to token-span alignment, role masks, and label masks.

Phase 4 introduces latest-schema config sections and removes obsolete config
knobs from the new schema. There is no legacy dual-read or archived-config
migration path. This phase also preserves the framework `training:` surface and
keeps framework `template:` separate from detection-output
`detection_template:`.

Phase 5 ports sorted SFT and random-order SFT preparation onto normalized
samples plus template-rendered full sequences.

Phase 6 adds the unified recursive detection CE target builder:
random object ordering, internal remaining-object multiset, object-entry trie
construction, hard-vs-support/balance target decisions, exact duplicate
handling, local target parity fixtures, and state-position weighting.

Phase 7 adds Random-Permutation `ET-RMP-CE` support/balance metrics,
equivalence-normalization mode, semantic image-balanced normalization mode, and
denominator diagnostics.

Phase 8 extracts static packing contracts and fingerprints from `sft.py` into
`src/detection/packing.py`. Recursive detection CE packing stays disabled until
trie metadata preservation is implemented.

Phase 9 aligns inference/evaluation with strict expected-template parsing and
diagnostic-only salvage.

This is a metric-surface change relative to broad salvage/auto-detect parsing.
Historical numbers produced under salvage or auto-detect parser modes are not
directly comparable to strict-template metrics unless the comparison table
reports `expected_template`, `parser_mode`, coordinate surface, and benchmark
scope explicitly.

Phase 10 enables compact SFT, random-order compact SFT, and compact
Random-Permutation `ET-RMP-CE` only after the render, parser, span, trie, and
normalization gates pass.

## Adoption Conditions

Compact Random-Permutation `ET-RMP-CE` with support/balance reweight should not
become a default Stage-1 path until these conditions hold:

- Canonical `stage1_json_pretty` SFT and `ET-RMP-CE` tests pass under the new full-sequence path.
- Compact rendered examples match the intended sequence grammar exactly.
- Compact objective construction contains no JSON keys or `]}` closure fragments.
- Coordinate control tokens align cleanly in tokenized form.
- Loss masks supervise only intended assistant, coordinate, structure, object-entry trie, separator, terminal, and EOS positions.
- Support/balance object-entry trie alternatives are identical to valid remaining objects.
- Close/EOS behavior is template-correct and tested at object-count boundaries.
- Random-order SFT proves the ordering change itself is not already harmful.
- Trie-disabled recursive detection CE reproduces ordinary hard CE on the rendered random sequence.
- The equivalence profile and semantic-normalization profile are reported separately in artifacts.
- Run manifests show template ID, prompt profile, objective variant, state-weighting policy, normalization policy, parser mode, and packing settings.

## Ablations Before Default Adoption

Recommended full ablation matrix:

- Eval-only reference checkpoint: anchors decoding controls and metric scope.
- `stage1_json_pretty` sorted SFT: canonical JSON SFT with deterministic spatial/order policy and strict JSON closure.
- `compact_full` sorted SFT: isolates compact-format shift under deterministic order.
- `stage1_json_pretty` random-order SFT: isolates random ordering while preserving the JSON/language prior.
- `compact_full` random-order SFT: isolates compact plus random-order interaction.
- `stage1_json_pretty` trie-disabled recursive detection CE: verifies the new recursive scaffold reproduces hard CE under canonical JSON.
- `compact_full` trie-disabled recursive detection CE: verifies the new recursive scaffold reproduces hard CE under compact rows.
- `stage1_json_pretty` Random-Permutation ET-RMP-CE `1.0/1.0`: pure multi-child trie CE under canonical JSON.
- `compact_full` Random-Permutation ET-RMP-CE `1.0/1.0`: pure multi-child trie CE under compact rows.
- `stage1_json_pretty` Random-Permutation ET-RMP-CE `2.0/1.0`: support-reweighted canonical JSON control.
- `compact_full` Random-Permutation ET-RMP-CE `2.0/1.0`: support-reweighted compact production candidate.
- `legacy_row_mean_prefix_mixture_equivalence` versus all-ones state weights: isolates the effect of removing explicit prefix sampling as a runtime surface.
- `legacy_row_mean_equivalence` versus `semantic_image_bucket_balanced`: isolates the loss-normalization change.
- Packing disabled versus static SFT packing for pure SFT.
- Padding-free packed runtime disabled versus explicit experimental runtime once metadata validation is ready.

Minimal ablation when compute is limited:

- Eval-only reference checkpoint.
- `stage1_json_pretty` sorted SFT.
- `compact_full` sorted SFT.
- `stage1_json_pretty` random-order SFT.
- `compact_full` random-order SFT.
- `stage1_json_pretty` Random-Permutation ET-RMP-CE `2.0/1.0` with `legacy_row_mean_prefix_mixture_equivalence` and `legacy_row_mean_equivalence`.
- `compact_full` Random-Permutation ET-RMP-CE `1.0/1.0` with semantic normalization.
- `compact_full` Random-Permutation ET-RMP-CE `2.0/1.0` with semantic normalization.

If budget is extremely tight, run paired canonical JSON and compact sorted SFT
first to measure format-prior damage, then paired canonical JSON and compact
`2.0/1.0` ET-RMP-CE. Do not claim the mechanism is trie MP or support reweight
unless the compact `1.0/1.0` control also runs.

Key metrics and diagnostics:

- Parse success under strict expected-template parser.
- Predicted object count.
- Count error by GT-count bucket.
- FN rate, especially on high-object-count images.
- Duplicate / repeated-object rate.
- Early-stop rate.
- Over-continuation rate.
- Object coverage.
- Coordinate-token loss and structural-token loss.
- `valid_child_mass_mean`, `p10`, `p50`, and `p90`, split by desc/coordinate/structural/other where possible.
- Multi-child trie-node count by token type.
- Multi-child trie-node entropy.
- Teacher trie-child top-1 accuracy.
- Valid child top-1 accuracy.
- Coordinate trie-child accuracy.
- Bbox quality for matched objects.
- Per-object-count performance buckets.
- Multi-child trie-token fraction, unique-token fraction, boundary/close/EOS fraction, effective weighted trie contribution, and per-image loss by GT-count bucket.
- State-position weight summaries by object count and emitted-object index.
- Normalization denominator diagnostics for equivalence and semantic profiles.
- Standard detection metrics, with scope reported explicitly as `val200`, full-val, proxy, raw-text, coord-token, or other exact surface.
- Training throughput, memory, and cache reuse.

## Risk Analysis

- Random object order may remove useful spatial ordering priors; random-order SFT is required as a control.
- Removing runtime prefix sampling without compiling old prefix exposure into state-position weights changes the objective.
- `semantic_image_bucket_balanced` changes the aggregation layer; it should be evaluated as an objective ablation, not hidden inside an equivalence refactor.
- Support reweight may reduce FN but increase over-continuation or FP if terminal stop calibration is weak.
- Compact templates must provide precise token spans; otherwise MP can be applied to the wrong tokens.
- Same-class objects often branch first at bbox or coordinate tokens, not desc tokens.
- Desc-prefix collisions can make `<|box_start|>` a true trie child; do not hard-code it as always-hard.
- Exact duplicate rendered entries must not be deduplicated; multiplicity affects `q(v)` and teacher-instance removal.
- Images with many objects produce more multi-child trie nodes; state weighting and loss normalization must be reported by object-count bucket.
- Terminal stop/control CE must remain strong enough so the model learns when to stop.
- Static packing and future padding-free packing should be considered separately from the objective; do not resurrect candidate scoring for efficiency.

## Subagent Review Summary

Objective-equivalence review:

- Random permutation can remove explicit prefix sampling as a runtime surface, but exact equivalence requires compiling the old prefix exposure into state-position weights.
- Exact equivalence also requires local target parity: hard-CE versus trie-CE decisions, valid child sets, and multiplicity weights must match frozen canonical JSON fixtures, not just global `alpha_t` weights.
- The active remaining-object multiset is mathematically required for trie targets and exact duplicate handling, but it should stay private to the target builder.
- Length-insensitive semantic normalization is valuable but is a separate objective layer from architecture simplification.

Module/config audit:

- The target should be greenfield latest-schema only, with no `coordjson_legacy`, no old config migration, and no `legacy_candidate_branch` module.
- `sft`, `random_order_sft`, and `random_permutation_et_rmp_ce` should share one full-sequence preparation path and differ only in object order and target policy.
- Config should collapse training-mode/loss/normalization knobs into one `objective` section, with state weighting and normalization explicitly recorded.
- Framework `template:` must remain separate from detection-output `detection_template:`, and framework `training:` must remain the home for runnable optimizer/batch/checkpoint settings.
- Latest Stage-1 routing should be owned by `objective.id`, not `trainer_variant`.

Data-flow/span audit:

- The minimal runtime flow is raw row -> normalized sample -> seeded object order -> rendered full assistant sequence with spans -> token alignment -> token-level target builder -> loss.
- `stage1_json_pretty` must be pinned as a v1 canonical JSON profile with exact field order, coordinate surface, bbox format, and byte-identical fixture.
- Template rendering owns assistant-local spans; tokenization owns post-chat-template assistant spans and token spans.
- Both templates need object-entry spans, trie-eligible spans, separator/terminal spans, coordinate spans, token spans, and object provenance.
- Strict expected-template parsing should be the metric-bearing path; auto-detect and salvage remain diagnostic only and are not directly comparable metric surfaces unless parser mode is reported.

Documentation hygiene audit:

- Retired candidate-scoring and set-continuation docs should remain as archived history, not active execution sources.
- Active docs should avoid live compatibility architecture for retired paths.
- The current superpowers spec/plan should be the execution source for the greenfield refactor.

## Remaining Empirical Gates

- Random-order SFT must answer whether random object order removes useful spatial priors before attributing gains or regressions to ET-RMP-CE.
- Prefix-state weighting and row-mean reduction must be tested before claiming exact equivalence to the prior ET-RMP-CE objective.
- Exact-equivalence claims are scoped to the same template, same denominator, packing disabled, and same random-order family. `compact_full`, semantic normalization, unit state weights, and packed runtime are separate ablations.
- Support/balance CE can be enabled from the first Stage-1 step only after the ablation gates pass; this is an empirical adoption gate, not an architecture blocker.
- Compact outputs use strict expected-template parsing for metric-bearing evaluation. Diagnostic salvage, if ever used, is non-metric artifact analysis only.

## Definition Of Ready For Implementation

Implementation can begin when:

- This spec and the paired implementation plan are accepted as the execution source.
- The approved first implementation slice is raw normalization -> template registry -> canonical `stage1_json_pretty` parity -> `compact_full` template -> strict spans.
- The first ET-RMP-CE implementation exposes both `legacy_row_mean_prefix_mixture_equivalence` for canonical JSON parity tests and `uniform_permutation` for compact/new-objective ablations; neither is silently treated as equivalent to the other.
- The owner confirms that no production training run should consume compact Random-Permutation `ET-RMP-CE` until the stated validation gates pass.
