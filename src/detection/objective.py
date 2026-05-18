"""Preparation helpers for detection SFT training examples."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import Any, Literal, Mapping, Sequence

from src.detection.data import (
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.rollin import ObjectInstanceId, RollinState, make_prefix_rollin_state
from src.detection.template import (
    CompactFullTemplate,
    DetectionSequenceTemplate,
    RenderedAssistantSequence,
)
from src.detection.token_types import (
    allowed_type_token_ids_for_target,
    build_compact_token_type_groups,
)
from src.detection.tokenizer_contract import (
    CompactTrainingStopContract,
    resolve_compact_training_stop_contract,
)
from src.detection.tokenization import (
    TokenRole,
    TokenizedDetectionExample,
    TokenizedObjectEntry,
    TokenizerWithOffsets,
    tokenize_rendered_detection_conversation,
)
from src.tokens.coord.codec import token_to_int

DetectionTrainingMode = Literal[
    "sorted_sft",
    "random_order_sft",
    "random_permutation_et_rmp_ce",
    "prefix_rollin_et_rmp_ce",
]
TrieTargetKind = Literal["hard_ce", "trie_multi_positive"]
CoordSlotName = Literal["x1", "y1", "x2", "y2"]
StateWeightingStrategy = Literal[
    "legacy_row_mean_prefix_mixture_equivalence",
    "uniform_permutation",
]
LossNormalizationStrategy = Literal[
    "legacy_row_mean_equivalence",
    "semantic_image_bucket_balanced",
]

_EMPTY_PREFIX_PROBABILITY = 0.30
_RANDOM_SUBSET_PROBABILITY = 0.45
_LEAVE_ONE_OUT_PROBABILITY = 0.20
_FULL_PREFIX_PROBABILITY = 0.05
_OBJECT_ROLE_WEIGHTS = {
    "desc_identity": 0.35,
    "bbox_coord": 0.45,
    "entry_trie_decision": 0.15,
    "object_control": 0.05,
}
_IMAGE_MIXTURE_WEIGHTS = {
    "objects": 1.00,
    "schema": 0.10,
}
_STATE_WEIGHTING_STRATEGIES = {
    "legacy_row_mean_prefix_mixture_equivalence",
    "uniform_permutation",
}
_LOSS_NORMALIZATION_STRATEGIES = {
    "legacy_row_mean_equivalence",
    "semantic_image_bucket_balanced",
}


class SemanticRole(str, Enum):
    DESC_IDENTITY = "desc_identity"
    BBOX_COORD = "bbox_coord"
    ENTRY_TRIE_DECISION = "entry_trie_decision"
    OBJECT_CONTROL = "object_control"
    SEPARATOR_CONTINUE = "separator_continue"
    TERMINAL_STOP = "terminal_stop"
    SCHEMA_CONTROL = "schema_control"
    CHAT_STOP = "chat_stop"


@dataclass(frozen=True)
class LossAtom:
    atom_id: str
    semantic_role: SemanticRole
    token_positions: tuple[int, ...]
    object_instance_id: str | None = None
    object_index: int | None = None


@dataclass(frozen=True)
class StateWeightingDiagnostics:
    profile_id: StateWeightingStrategy
    prefix_length_probabilities: tuple[float, ...]
    supervised_token_counts_by_prefix_length: tuple[int, ...]
    entry_exposures: tuple[float, ...]
    separator_exposures: tuple[float, ...]
    terminal_exposure: float


@dataclass(frozen=True)
class LossNormalizationDiagnostics:
    profile_id: LossNormalizationStrategy
    state_weight_sum: float
    atom_count: int
    object_count: int
    gt_count_bucket: str
    semantic_role_token_counts: dict[SemanticRole, int]
    semantic_role_atom_counts: dict[SemanticRole, int]
    multi_child_trie_token_fraction: float
    boundary_fraction: float
    effective_weighted_trie_contribution: float
    component_losses: dict[str, float]
    bucket_weights: dict[str, float]


@dataclass(frozen=True)
class LossNormalizationResult:
    profile_id: LossNormalizationStrategy
    normalized_loss: float
    diagnostics: LossNormalizationDiagnostics


@dataclass(frozen=True)
class TrieBranchTarget:
    token_id: int
    multiplicity: int
    probability: float


@dataclass(frozen=True)
class CoordSoftTargetSpec:
    object_instance_id: str
    slot_name: CoordSlotName
    bbox_xyxy: tuple[int, int, int, int]
    probability: float


@dataclass(frozen=True)
class TokenTarget:
    position: int
    teacher_token_id: int
    kind: TrieTargetKind
    trie_branch_targets: tuple[TrieBranchTarget, ...]
    object_instance_id: str | None
    token_role: TokenRole
    state_weight: float = 1.0
    state_exposure: float = 1.0
    semantic_role: SemanticRole = SemanticRole.OBJECT_CONTROL
    loss_atom_id: str | None = None
    loss_weight: float = 1.0
    type_gate_token_ids: tuple[int, ...] = ()
    type_gate_weight: float = 0.0
    coord_soft_targets: tuple[CoordSoftTargetSpec, ...] = ()

    @property
    def valid_token_ids(self) -> tuple[int, ...]:
        return tuple(target.token_id for target in self.trie_branch_targets)

    @property
    def child_multiplicities(self) -> tuple[int, ...]:
        return tuple(target.multiplicity for target in self.trie_branch_targets)

    @property
    def child_probabilities(self) -> tuple[float, ...]:
        return tuple(target.probability for target in self.trie_branch_targets)


@dataclass(frozen=True)
class RecursiveDetectionTargets:
    token_targets: tuple[TokenTarget, ...]
    state_weighting: StateWeightingStrategy
    normalization: LossNormalizationStrategy
    loss_atoms: tuple[LossAtom, ...]
    state_weighting_diagnostics: StateWeightingDiagnostics
    token_position_origin: str = "PreparedDetectionExample.tokenized"


@dataclass(frozen=True)
class PreparedDetectionExample:
    mode: DetectionTrainingMode
    normalized_sample: NormalizedDetectionSample
    object_ordering: ObjectOrderingPlan
    rendered_assistant: RenderedAssistantSequence
    tokenized: TokenizedDetectionExample
    template_id: str
    template_version: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_mask: tuple[bool, ...]
    recursive_detection_targets: RecursiveDetectionTargets | None = None

    @property
    def realized_source_object_indices(self) -> tuple[int, ...]:
        return self.object_ordering.realized_source_object_indices


@dataclass(frozen=True)
class PrefixRollinDebugSpan:
    token_positions: tuple[int, ...]


@dataclass(frozen=True)
class PreparedPrefixRollinExample:
    mode: Literal["prefix_rollin_et_rmp_ce"]
    normalized_sample: NormalizedDetectionSample
    rollin_state: RollinState
    rendered_assistant: RenderedAssistantSequence
    tokenized: TokenizedDetectionExample
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_mask: tuple[bool, ...]
    recursive_detection_targets: RecursiveDetectionTargets
    debug_spans: Mapping[str, PrefixRollinDebugSpan]
    stop_contract: CompactTrainingStopContract

    @property
    def object_ordering(self) -> ObjectOrderingPlan:
        return self.normalized_sample.object_ordering

    @property
    def template_id(self) -> str:
        return self.rendered_assistant.template_id

    @property
    def template_version(self) -> int:
        return self.rendered_assistant.template_version

    @property
    def realized_source_object_indices(self) -> tuple[int, ...]:
        return self.object_ordering.realized_source_object_indices

    @property
    def chat_text(self) -> str:
        return self.tokenized.chat_text

    @property
    def assistant_char_span(self):
        return self.tokenized.assistant_char_span

    @property
    def assistant_stop_token_span(self):
        span = self.tokenized.assistant_stop_token_span
        if span is None:
            raise ValueError(
                "prefix_rollin_et_rmp_ce requires assistant <|im_end|> span"
            )
        return span

    @property
    def assistant_stop_char_span(self):
        char_span = self.assistant_stop_token_span.char_span
        if char_span is None:
            raise ValueError(
                "prefix_rollin_et_rmp_ce requires assistant stop char span"
            )
        return char_span

    @property
    def assistant_stop_token_text(self) -> str:
        return self.assistant_stop_char_span.text(self.chat_text)


def prepare_detection_training_example(
    sample: NormalizedDetectionSample,
    *,
    template: DetectionSequenceTemplate,
    tokenizer: TokenizerWithOffsets,
    mode: DetectionTrainingMode,
    state_weighting: StateWeightingStrategy = "uniform_permutation",
    normalization: LossNormalizationStrategy = "semantic_image_bucket_balanced",
    system_prompt: str | None = None,
    user_content: str = "<image>",
    messages: Sequence[Mapping[str, Any]] | None = None,
) -> PreparedDetectionExample:
    """Build a single prepared SFT example from a normalized detection sample."""

    # validating the requested training surface
    _validate_state_weighting_strategy(state_weighting)
    _validate_loss_normalization_strategy(normalization)
    prepared_sample = _prepare_sample_for_mode(sample, mode=mode)
    _validate_template_capabilities(template, mode=mode)

    # rendering and tokenizing the teacher-forced full sequence
    rendered_assistant = template.render_assistant(prepared_sample)
    tokenized = tokenize_rendered_detection_conversation(
        rendered_assistant,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_content=user_content,
        messages=messages,
    )

    # attaching recursive trie metadata only for recursive detection CE
    recursive_detection_targets = None
    if mode == "random_permutation_et_rmp_ce":
        recursive_detection_targets = build_recursive_detection_targets(
            prepared_sample,
            tokenized=tokenized,
            state_weighting=state_weighting,
            normalization=normalization,
        )

    return PreparedDetectionExample(
        mode=mode,
        normalized_sample=prepared_sample,
        object_ordering=prepared_sample.object_ordering,
        rendered_assistant=rendered_assistant,
        tokenized=tokenized,
        template_id=rendered_assistant.template_id,
        template_version=rendered_assistant.template_version,
        input_ids=tokenized.input_ids,
        labels=tokenized.labels,
        assistant_mask=tokenized.assistant_mask,
        recursive_detection_targets=recursive_detection_targets,
    )


def build_compact_prefix_rollin_example(
    *,
    objects: Sequence[NormalizedDetectionObject],
    rollin_order: Sequence[NormalizedDetectionObject],
    k: int,
    tokenizer: TokenizerWithOffsets,
    normalized_sample: NormalizedDetectionSample | None = None,
    type_gate_config: Any | None = None,
    system_prompt: str | None = None,
    user_content: str = "<image>",
    messages: Sequence[Mapping[str, Any]] | None = None,
) -> PreparedPrefixRollinExample:
    """Build one compact_full prefix-rollin example for tests and materialization."""

    source_objects = tuple(
        normalized_sample.objects if normalized_sample is not None else objects
    )
    if normalized_sample is not None and tuple(objects) != source_objects:
        raise ValueError("objects must match normalized_sample.objects when provided")
    source_by_id = {obj.object_instance_id: obj for obj in source_objects}
    if len(source_by_id) != len(source_objects):
        raise ValueError("objects must have unique object_instance_id values")
    rollin_ids = tuple(obj.object_instance_id for obj in rollin_order)
    if len(rollin_ids) != len(source_objects) or set(rollin_ids) != set(source_by_id):
        raise ValueError("rollin_order must contain each object exactly once")
    if normalized_sample is not None:
        normalized_order_ids = tuple(obj.object_instance_id for obj in source_objects)
        if rollin_ids != normalized_order_ids:
            raise ValueError(
                "rollin_order must match normalized_sample.objects order until "
                "explicit emitted/suffix builder support is implemented"
            )
    ordered_objects = tuple(source_by_id[object_id] for object_id in rollin_ids)

    rollin_state = make_prefix_rollin_state(
        tuple(ObjectInstanceId(obj.object_instance_id) for obj in source_objects),
        permutation=tuple(
            ObjectInstanceId(obj.object_instance_id) for obj in ordered_objects
        ),
        k=k,
    )
    stop_contract = resolve_compact_training_stop_contract(tokenizer)
    sample = (
        normalized_sample
        if normalized_sample is not None
        else _build_prefix_rollin_sample(ordered_objects)
    )

    rendered_assistant = CompactFullTemplate().render_assistant(sample)
    tokenized = tokenize_rendered_detection_conversation(
        rendered_assistant,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_content=user_content,
        messages=messages,
        assistant_stop_markers=(stop_contract.training_eos_token_text,),
    )
    _validate_compact_prefix_rollin_stop(tokenized, stop_contract=stop_contract)

    (
        prefix_positions,
        suffix_positions,
        suffix_entry_positions,
    ) = _prefix_rollin_payload_positions(
        tokenized,
        k=int(k),
    )
    eos_positions = tuple(tokenized.assistant_stop_token_span.token_indices())
    active_positions = set(suffix_positions) | set(eos_positions)
    labels = tuple(
        token_id if position in active_positions else -100
        for position, token_id in enumerate(tokenized.input_ids)
    )
    masked_tokenized = replace(tokenized, labels=labels)

    full_targets = build_recursive_detection_targets(
        sample,
        tokenized=tokenized,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
    )
    filtered_targets = tuple(
        target
        for target in full_targets.token_targets
        if target.position in active_positions
    )
    filtered_targets = _apply_prefix_rollin_type_gate(
        filtered_targets,
        tokenizer=tokenizer,
        type_gate_config=type_gate_config,
    )
    loss_atoms = _build_loss_atoms(
        tokenized=masked_tokenized,
        token_targets=filtered_targets,
    )
    recursive_detection_targets = RecursiveDetectionTargets(
        token_targets=_assign_loss_atoms(
            token_targets=filtered_targets,
            loss_atoms=loss_atoms,
        ),
        state_weighting=full_targets.state_weighting,
        normalization=full_targets.normalization,
        loss_atoms=loss_atoms,
        state_weighting_diagnostics=_prefix_rollin_state_weighting_diagnostics(
            tokenized=masked_tokenized,
            k=int(k),
            active_target_count=len(filtered_targets),
        ),
    )

    debug_spans = {
        "rollin_prefix": PrefixRollinDebugSpan(prefix_positions),
        "supervised_suffix": PrefixRollinDebugSpan(suffix_positions),
        "supervised_suffix_entries": PrefixRollinDebugSpan(suffix_entry_positions),
        "semantic_eos": PrefixRollinDebugSpan(eos_positions),
    }
    return PreparedPrefixRollinExample(
        mode="prefix_rollin_et_rmp_ce",
        normalized_sample=sample,
        rollin_state=rollin_state,
        rendered_assistant=rendered_assistant,
        tokenized=masked_tokenized,
        input_ids=masked_tokenized.input_ids,
        labels=labels,
        assistant_mask=masked_tokenized.assistant_mask,
        recursive_detection_targets=recursive_detection_targets,
        debug_spans=debug_spans,
        stop_contract=stop_contract,
    )


def _build_prefix_rollin_sample(
    ordered_objects: Sequence[NormalizedDetectionObject],
) -> NormalizedDetectionSample:
    ordered = tuple(ordered_objects)
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=ordered,
        width=1,
        height=1,
        image_id=0,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="prefix_rollin"),
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=0,
            seed_source="prefix_rollin_unit",
        ).with_realized(tuple(obj.source_object_index for obj in ordered)),
    )


def _validate_compact_prefix_rollin_stop(
    tokenized: TokenizedDetectionExample,
    *,
    stop_contract: CompactTrainingStopContract,
) -> None:
    span = tokenized.assistant_stop_token_span
    if span is None:
        raise ValueError("prefix_rollin_et_rmp_ce requires assistant <|im_end|> span")
    stop_ids = tuple(tokenized.input_ids[span.start : span.end])
    if stop_ids != (stop_contract.im_end_token_id,):
        raise ValueError(
            "prefix_rollin_et_rmp_ce requires semantic_eos to be the single "
            "<|im_end|> token"
        )


def _prefix_rollin_payload_positions(
    tokenized: TokenizedDetectionExample,
    *,
    k: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if k < 0 or k > len(tokenized.object_entries):
        raise ValueError(f"k must satisfy 0 <= k <= {len(tokenized.object_entries)}")

    prefix_positions: list[int] = []
    suffix_positions: list[int] = []
    suffix_entry_positions: list[int] = []
    object_count = len(tokenized.object_entries)
    for entry_index, entry in enumerate(tokenized.object_entries):
        if entry_index < k:
            prefix_positions.extend(entry.entry_span.token_indices())
        else:
            entry_positions = tuple(entry.entry_span.token_indices())
            suffix_positions.extend(entry_positions)
            suffix_entry_positions.extend(entry_positions)
        if entry.separator_span is not None:
            separator_destination = prefix_positions
            if entry_index >= k - 1 and entry_index < object_count - 1:
                separator_destination = suffix_positions
            separator_destination.extend(entry.separator_span.token_indices())
    return (
        tuple(prefix_positions),
        tuple(suffix_positions),
        tuple(suffix_entry_positions),
    )


def _prefix_rollin_state_weighting_diagnostics(
    *,
    tokenized: TokenizedDetectionExample,
    k: int,
    active_target_count: int,
) -> StateWeightingDiagnostics:
    object_count = len(tokenized.object_entries)
    probabilities = tuple(
        1.0 if prefix_length == k else 0.0 for prefix_length in range(object_count + 1)
    )
    counts = tuple(
        active_target_count if prefix_length == k else 0
        for prefix_length in range(object_count + 1)
    )
    entry_exposures = tuple(
        0.0 if entry_index < k else 1.0 for entry_index in range(object_count)
    )
    separator_exposures = tuple(
        1.0 if separator_index >= k - 1 else 0.0
        for separator_index in range(max(object_count - 1, 0))
    )
    return StateWeightingDiagnostics(
        profile_id="uniform_permutation",
        prefix_length_probabilities=probabilities,
        supervised_token_counts_by_prefix_length=counts,
        entry_exposures=entry_exposures,
        separator_exposures=separator_exposures,
        terminal_exposure=1.0,
    )


def _cfg_value(cfg: Any, field_name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(field_name, default)
    return getattr(cfg, field_name, default)


def _apply_prefix_rollin_type_gate(
    token_targets: Sequence[TokenTarget],
    *,
    tokenizer: TokenizerWithOffsets,
    type_gate_config: Any | None,
) -> tuple[TokenTarget, ...]:
    if not bool(_cfg_value(type_gate_config, "enabled", False)):
        return tuple(token_targets)
    groups = build_compact_token_type_groups(tokenizer)
    weights_cfg = _cfg_value(type_gate_config, "weights")

    def _weight(name: str) -> float:
        value = _cfg_value(weights_cfg, name, 0.0)
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"type_gate.weights.{name} must be finite and >= 0")
        return weight

    weights_by_group = {
        "struct": _weight("struct"),
        "coord": _weight("coord"),
        "desc": _weight("desc"),
        "eos": _weight("eos"),
    }

    out: list[TokenTarget] = []
    for target in token_targets:
        allowed_ids = allowed_type_token_ids_for_target(target, groups)
        group_weights: list[float] = []
        if any(token_id in groups.struct for token_id in allowed_ids):
            group_weights.append(weights_by_group["struct"])
        if any(token_id in groups.coord for token_id in allowed_ids):
            group_weights.append(weights_by_group["coord"])
        if any(token_id in groups.desc for token_id in allowed_ids):
            group_weights.append(weights_by_group["desc"])
        if any(token_id in groups.eos for token_id in allowed_ids):
            group_weights.append(weights_by_group["eos"])
        out.append(
            replace(
                target,
                type_gate_token_ids=tuple(sorted(allowed_ids)),
                type_gate_weight=max(group_weights) if group_weights else 0.0,
            )
        )
    return tuple(out)


def _prepare_sample_for_mode(
    sample: NormalizedDetectionSample,
    *,
    mode: DetectionTrainingMode,
) -> NormalizedDetectionSample:
    _validate_sample_ordering(sample)
    if mode == "sorted_sft":
        if sample.object_ordering.strategy != "sorted":
            raise ValueError(
                "sorted_sft requires sample.object_ordering.strategy='sorted'"
            )
        return sample
    if mode == "random_order_sft":
        _validate_random_permutation_sample(
            sample,
            mode_name="random_order_sft",
        )
        return sample
    if mode == "random_permutation_et_rmp_ce":
        _validate_random_permutation_sample(
            sample,
            mode_name="random_permutation_et_rmp_ce",
        )
        return sample
    raise ValueError(
        "mode must be one of {'sorted_sft', 'random_order_sft', "
        "'random_permutation_et_rmp_ce'}; "
        f"got {mode!r}"
    )


def _validate_sample_ordering(sample: NormalizedDetectionSample) -> None:
    object_source_indices = tuple(obj.source_object_index for obj in sample.objects)
    realized = sample.object_ordering.realized_source_object_indices
    if realized != object_source_indices:
        raise ValueError(
            "sample.object_ordering.realized_source_object_indices must match the "
            "current normalized object order"
        )


def _validate_random_permutation_sample(
    sample: NormalizedDetectionSample,
    *,
    mode_name: str,
) -> None:
    if sample.object_ordering.strategy != "random_permutation":
        raise ValueError(
            f"{mode_name} requires sample.object_ordering.strategy='random_permutation'"
        )
    if sample.object_ordering.seed is None:
        raise ValueError(f"{mode_name} requires sample.object_ordering.seed")
    if not sample.object_ordering.seed_source:
        raise ValueError(f"{mode_name} requires sample.object_ordering.seed_source")


def _validate_template_capabilities(
    template: DetectionSequenceTemplate,
    *,
    mode: DetectionTrainingMode,
) -> None:
    if mode in {"sorted_sft", "random_order_sft"}:
        if not template.capabilities.supports_sft:
            raise ValueError(f"template {template.template_id!r} does not support SFT")
        return
    if not template.capabilities.supports_recursive_detection_ce:
        raise ValueError(
            f"template {template.template_id!r} does not support recursive detection CE"
        )
    if not template.capabilities.supports_random_permutation_et_rmp_ce:
        raise ValueError(
            f"template {template.template_id!r} does not support "
            "random_permutation_et_rmp_ce"
        )


def _validate_state_weighting_strategy(value: str) -> None:
    if value not in _STATE_WEIGHTING_STRATEGIES:
        raise ValueError(
            "state_weighting must be one of "
            f"{sorted(_STATE_WEIGHTING_STRATEGIES)}, got {value!r}"
        )


def _validate_loss_normalization_strategy(value: str) -> None:
    if value not in _LOSS_NORMALIZATION_STRATEGIES:
        raise ValueError(
            "normalization must be one of "
            f"{sorted(_LOSS_NORMALIZATION_STRATEGIES)}, got {value!r}"
        )


@dataclass(frozen=True)
class _TrieObjectInstance:
    object_instance_id: str
    object_index: int
    source_object_index: int
    token_ids: tuple[int, ...]
    bbox_xyxy: tuple[int, int, int, int]
    hard_bbox_supervision: bool


@dataclass
class _EntryTrieNode:
    terminal_count: int = 0
    children: dict[int, "_EntryTrieNode"] = field(default_factory=dict)
    descendant_instances: list[_TrieObjectInstance] = field(default_factory=list)
    _cached_descendant_count: int | None = None

    def descendant_count(self) -> int:
        if self._cached_descendant_count is None:
            self._cached_descendant_count = int(
                self.terminal_count
                + sum(child.descendant_count() for child in self.children.values())
            )
        return self._cached_descendant_count


def build_recursive_detection_targets(
    sample: NormalizedDetectionSample,
    *,
    tokenized: TokenizedDetectionExample,
    state_weighting: StateWeightingStrategy = "uniform_permutation",
    normalization: LossNormalizationStrategy = "semantic_image_bucket_balanced",
) -> RecursiveDetectionTargets:
    """Build token-level trie target metadata for recursive detection CE."""

    # validating the recursive input contract
    _validate_state_weighting_strategy(state_weighting)
    _validate_loss_normalization_strategy(normalization)
    _validate_random_permutation_sample(
        sample,
        mode_name="random_permutation_et_rmp_ce",
    )
    remaining_instances = list(
        _build_trie_object_instances(sample, tokenized=tokenized)
    )

    # walking the assistant sequence around teacher object entries
    token_targets: list[TokenTarget] = []
    cursor = tokenized.assistant_token_span.start
    for entry in tokenized.object_entries:
        hard_bbox_supervision = _allows_hard_bbox_supervision(entry)
        entry_coord_positions = _entry_coord_positions(entry)
        entry_coord_soft_targets: Mapping[int, tuple[CoordSoftTargetSpec, ...]] = {}
        if hard_bbox_supervision:
            entry_coord_soft_targets = _single_object_coord_soft_targets_by_position(
                entry,
                object_instance=_find_object_instance(
                    remaining_instances,
                    object_instance_id=entry.object_instance_id,
                ),
            )
        _append_hard_ce_targets(
            token_targets,
            tokenized=tokenized,
            start=cursor,
            end=entry.entry_span.start,
            object_instance_id=None,
        )
        _append_hard_ce_targets(
            token_targets,
            tokenized=tokenized,
            start=entry.entry_span.start,
            end=entry.trie_eligible_span.start,
            object_instance_id=entry.object_instance_id,
            coord_soft_targets_by_position=entry_coord_soft_targets,
            excluded_positions=entry_coord_positions
            if not hard_bbox_supervision
            else (),
        )
        _append_recursive_entry_targets(
            token_targets,
            tokenized=tokenized,
            entry=entry,
            remaining_instances=remaining_instances,
            excluded_positions=entry_coord_positions
            if not hard_bbox_supervision
            else (),
        )
        _append_hard_ce_targets(
            token_targets,
            tokenized=tokenized,
            start=entry.trie_eligible_span.end,
            end=entry.entry_span.end,
            object_instance_id=entry.object_instance_id,
            coord_soft_targets_by_position=entry_coord_soft_targets,
            excluded_positions=entry_coord_positions
            if not hard_bbox_supervision
            else (),
        )
        _remove_object_instance(
            remaining_instances,
            object_instance_id=entry.object_instance_id,
        )
        cursor = entry.entry_span.end

    _append_hard_ce_targets(
        token_targets,
        tokenized=tokenized,
        start=cursor,
        end=len(tokenized.labels),
        object_instance_id=None,
    )
    weighted_targets, state_weighting_diagnostics = _apply_state_weighting(
        token_targets=tuple(token_targets),
        tokenized=tokenized,
        state_weighting=state_weighting,
    )
    loss_atoms = _build_loss_atoms(
        tokenized=tokenized,
        token_targets=weighted_targets,
    )
    annotated_targets = _assign_loss_atoms(
        token_targets=weighted_targets,
        loss_atoms=loss_atoms,
    )
    return RecursiveDetectionTargets(
        token_targets=annotated_targets,
        state_weighting=state_weighting,
        normalization=normalization,
        loss_atoms=loss_atoms,
        state_weighting_diagnostics=state_weighting_diagnostics,
    )


def _apply_state_weighting(
    *,
    token_targets: tuple[TokenTarget, ...],
    tokenized: TokenizedDetectionExample,
    state_weighting: StateWeightingStrategy,
) -> tuple[tuple[TokenTarget, ...], StateWeightingDiagnostics]:
    _validate_state_weighting_strategy(state_weighting)
    object_count = len(tokenized.object_entries)
    prefix_length_probabilities = _prefix_length_probabilities(object_count)
    target_prefix_lengths = {
        target.position: _prefix_lengths_for_position(
            target.position,
            tokenized=tokenized,
        )
        for target in token_targets
    }
    supervised_token_counts_by_prefix_length = [0 for _ in prefix_length_probabilities]
    for prefix_lengths in target_prefix_lengths.values():
        for prefix_length in prefix_lengths:
            supervised_token_counts_by_prefix_length[prefix_length] += 1

    weighted_targets: list[TokenTarget] = []
    for target in token_targets:
        prefix_lengths = target_prefix_lengths[target.position]
        if state_weighting == "uniform_permutation":
            state_exposure = 1.0
            state_weight = 1.0
        else:
            state_exposure = sum(
                prefix_length_probabilities[prefix_length]
                for prefix_length in prefix_lengths
            )
            state_weight = sum(
                prefix_length_probabilities[prefix_length]
                / supervised_token_counts_by_prefix_length[prefix_length]
                for prefix_length in prefix_lengths
            )
        weighted_targets.append(
            replace(
                target,
                state_weight=float(state_weight),
                state_exposure=float(state_exposure),
            )
        )

    if state_weighting == "uniform_permutation":
        entry_exposures = tuple(1.0 for _ in range(object_count))
    else:
        entry_exposures = tuple(
            float(
                sum(
                    prefix_length_probabilities[prefix_length]
                    for prefix_length in range(entry_index)
                )
            )
            for entry_index in range(1, object_count + 1)
        )
    separator_exposures = entry_exposures[:-1]
    terminal_exposure = (
        1.0
        if state_weighting == "uniform_permutation"
        else float(sum(prefix_length_probabilities))
    )
    diagnostics = StateWeightingDiagnostics(
        profile_id=state_weighting,
        prefix_length_probabilities=prefix_length_probabilities,
        supervised_token_counts_by_prefix_length=tuple(
            int(count) for count in supervised_token_counts_by_prefix_length
        ),
        entry_exposures=entry_exposures,
        separator_exposures=separator_exposures,
        terminal_exposure=terminal_exposure,
    )
    return tuple(weighted_targets), diagnostics


def _prefix_length_probabilities(object_count: int) -> tuple[float, ...]:
    if object_count <= 0:
        return (1.0,)
    if object_count == 1:
        total = (
            _EMPTY_PREFIX_PROBABILITY
            + _LEAVE_ONE_OUT_PROBABILITY
            + _FULL_PREFIX_PROBABILITY
        )
        return (
            float((_EMPTY_PREFIX_PROBABILITY + _LEAVE_ONE_OUT_PROBABILITY) / total),
            float(_FULL_PREFIX_PROBABILITY / total),
        )

    probabilities = [0.0 for _ in range(object_count + 1)]
    probabilities[0] += _EMPTY_PREFIX_PROBABILITY
    for prefix_length in range(1, object_count):
        probabilities[prefix_length] += _RANDOM_SUBSET_PROBABILITY / (object_count - 1)
    probabilities[object_count - 1] += _LEAVE_ONE_OUT_PROBABILITY
    probabilities[object_count] += _FULL_PREFIX_PROBABILITY
    return tuple(float(probability) for probability in probabilities)


def _prefix_lengths_for_position(
    position: int,
    *,
    tokenized: TokenizedDetectionExample,
) -> tuple[int, ...]:
    object_entries = tokenized.object_entries
    if not object_entries:
        return (0,)

    first_entry = object_entries[0]
    if position < first_entry.entry_span.start:
        return (0,)

    for entry_index, entry in enumerate(object_entries, start=1):
        if entry.entry_span.start <= position < entry.entry_span.end:
            return tuple(range(entry_index))
        if entry.separator_span is not None and (
            entry.separator_span.start <= position < entry.separator_span.end
        ):
            return tuple(range(entry_index))

    return tuple(range(len(object_entries) + 1))


def _build_loss_atoms(
    *,
    tokenized: TokenizedDetectionExample,
    token_targets: tuple[TokenTarget, ...],
) -> tuple[LossAtom, ...]:
    target_by_position = {target.position: target for target in token_targets}
    atoms: list[LossAtom] = []
    used_positions: set[int] = set()

    def add_atom(
        *,
        atom_id: str,
        semantic_role: SemanticRole,
        positions: Sequence[int],
        object_instance_id: str | None = None,
        object_index: int | None = None,
    ) -> None:
        filtered_positions = tuple(
            position
            for position in positions
            if position in target_by_position and position not in used_positions
        )
        if not filtered_positions:
            return
        atoms.append(
            LossAtom(
                atom_id=atom_id,
                semantic_role=semantic_role,
                token_positions=filtered_positions,
                object_instance_id=object_instance_id,
                object_index=object_index,
            )
        )
        used_positions.update(filtered_positions)

    first_entry_start = (
        tokenized.object_entries[0].entry_span.start
        if tokenized.object_entries
        else len(tokenized.labels)
    )
    add_atom(
        atom_id="schema_control:assistant_prefix",
        semantic_role=SemanticRole.SCHEMA_CONTROL,
        positions=range(tokenized.assistant_token_span.start, first_entry_start),
    )

    for entry in tokenized.object_entries:
        entry_positions = {
            position
            for position in entry.entry_span.token_indices()
            if position in target_by_position
        }
        trie_positions = {
            position
            for position in entry_positions
            if target_by_position[position].kind == "trie_multi_positive"
        }
        desc_positions = {
            position
            for position in entry.desc_span.token_indices()
            if position in target_by_position and position not in trie_positions
        }
        coord_positions = {
            position
            for coord_span in entry.coord_spans
            for position in coord_span.token_indices()
            if _allows_hard_bbox_supervision(entry)
            and position in target_by_position
            and position not in trie_positions
        }
        object_control_positions = (
            entry_positions - trie_positions - desc_positions - coord_positions
        )

        add_atom(
            atom_id=f"object:{entry.object_index}:entry_trie_decision",
            semantic_role=SemanticRole.ENTRY_TRIE_DECISION,
            positions=sorted(trie_positions),
            object_instance_id=entry.object_instance_id,
            object_index=entry.object_index,
        )
        add_atom(
            atom_id=f"object:{entry.object_index}:desc_identity",
            semantic_role=SemanticRole.DESC_IDENTITY,
            positions=sorted(desc_positions),
            object_instance_id=entry.object_instance_id,
            object_index=entry.object_index,
        )
        add_atom(
            atom_id=f"object:{entry.object_index}:bbox_coord",
            semantic_role=SemanticRole.BBOX_COORD,
            positions=sorted(coord_positions),
            object_instance_id=entry.object_instance_id,
            object_index=entry.object_index,
        )
        add_atom(
            atom_id=f"object:{entry.object_index}:object_control",
            semantic_role=SemanticRole.OBJECT_CONTROL,
            positions=sorted(object_control_positions),
            object_instance_id=entry.object_instance_id,
            object_index=entry.object_index,
        )
        if entry.separator_span is not None:
            add_atom(
                atom_id=f"object:{entry.object_index}:separator_continue",
                semantic_role=SemanticRole.SEPARATOR_CONTINUE,
                positions=range(entry.separator_span.start, entry.separator_span.end),
            )

    chat_stop_positions: set[int] = set()
    if tokenized.assistant_stop_token_span is not None:
        chat_stop_positions.update(tokenized.assistant_stop_token_span.token_indices())

    terminal_positions: set[int] = set()
    if tokenized.terminal_span is not None:
        terminal_positions.update(tokenized.terminal_span.token_indices())
    for stop_marker_span in tokenized.stop_marker_spans:
        terminal_positions.update(stop_marker_span.token_indices())
    terminal_positions.difference_update(chat_stop_positions)

    add_atom(
        atom_id="terminal_stop:assistant_terminal",
        semantic_role=SemanticRole.TERMINAL_STOP,
        positions=sorted(terminal_positions),
    )
    add_atom(
        atom_id="chat_stop:assistant_stop",
        semantic_role=SemanticRole.CHAT_STOP,
        positions=sorted(chat_stop_positions),
    )

    unassigned_positions = sorted(set(target_by_position) - used_positions)
    if unassigned_positions:
        raise ValueError(
            "every recursive TokenTarget must map to exactly one semantic LossAtom; "
            f"unassigned positions={unassigned_positions}"
        )
    return tuple(atoms)


def _assign_loss_atoms(
    *,
    token_targets: tuple[TokenTarget, ...],
    loss_atoms: tuple[LossAtom, ...],
) -> tuple[TokenTarget, ...]:
    position_to_atom: dict[int, LossAtom] = {}
    for atom in loss_atoms:
        for position in atom.token_positions:
            if position in position_to_atom:
                raise ValueError(
                    f"duplicate LossAtom assignment for token position {position}"
                )
            position_to_atom[position] = atom

    annotated_targets: list[TokenTarget] = []
    for target in token_targets:
        atom = position_to_atom.get(target.position)
        if atom is None:
            raise ValueError(
                f"missing LossAtom assignment for token position {target.position}"
            )
        annotated_targets.append(
            replace(
                target,
                semantic_role=atom.semantic_role,
                loss_atom_id=atom.atom_id,
            )
        )
    return tuple(annotated_targets)


def _build_trie_object_instances(
    sample: NormalizedDetectionSample,
    *,
    tokenized: TokenizedDetectionExample,
) -> tuple[_TrieObjectInstance, ...]:
    if len(sample.objects) != len(tokenized.object_entries):
        raise ValueError(
            "tokenized object entries must match normalized sample objects"
        )

    seen_ids: set[str] = set()
    instances: list[_TrieObjectInstance] = []
    for expected_index, (obj, entry) in enumerate(
        zip(sample.objects, tokenized.object_entries, strict=True)
    ):
        _validate_trie_eligible_span(entry)
        if entry.object_index != expected_index:
            raise ValueError(
                "tokenized object entries must preserve teacher object order"
            )
        if entry.object_instance_id != obj.object_instance_id:
            raise ValueError(
                "tokenized object entries must preserve object_instance_id order"
            )
        if entry.source_object_index != obj.source_object_index:
            raise ValueError(
                "tokenized object entries must preserve source_object_index order"
            )
        if entry.object_instance_id in seen_ids:
            raise ValueError(
                f"duplicate object_instance_id in active state: {entry.object_instance_id}"
            )

        seen_ids.add(entry.object_instance_id)
        token_ids = tuple(
            tokenized.input_ids[
                entry.trie_eligible_span.start : entry.trie_eligible_span.end
            ]
        )
        if not token_ids:
            raise ValueError(
                f"object entry {entry.object_instance_id} has an empty trie token span"
            )
        instances.append(
            _TrieObjectInstance(
                object_instance_id=entry.object_instance_id,
                object_index=entry.object_index,
                source_object_index=entry.source_object_index,
                token_ids=token_ids,
                bbox_xyxy=_bbox_xyxy_from_object(obj),
                hard_bbox_supervision=_allows_hard_bbox_supervision(entry),
            )
        )
    return tuple(instances)


def _validate_trie_eligible_span(entry: TokenizedObjectEntry) -> None:
    if entry.entry_span.start >= entry.entry_span.end:
        raise ValueError(
            f"object entry {entry.object_instance_id} must occupy at least one token"
        )
    if entry.trie_eligible_span.start >= entry.trie_eligible_span.end:
        raise ValueError(
            f"object entry {entry.object_instance_id} has an invalid trie token span"
        )
    if entry.trie_eligible_span.start < entry.entry_span.start:
        raise ValueError(
            f"object entry {entry.object_instance_id} trie span starts before entry span"
        )
    if entry.trie_eligible_span.end > entry.entry_span.end:
        raise ValueError(
            f"object entry {entry.object_instance_id} trie span ends after entry span"
        )


def _append_recursive_entry_targets(
    token_targets: list[TokenTarget],
    *,
    tokenized: TokenizedDetectionExample,
    entry: TokenizedObjectEntry,
    remaining_instances: list[_TrieObjectInstance],
    excluded_positions: set[int] | frozenset[int] | tuple[int, ...] = (),
) -> None:
    trie_root = _build_entry_trie(remaining_instances)
    teacher_instance = _find_object_instance(
        remaining_instances,
        object_instance_id=entry.object_instance_id,
    )
    teacher_token_ids = tuple(
        tokenized.input_ids[
            entry.trie_eligible_span.start : entry.trie_eligible_span.end
        ]
    )
    if teacher_token_ids != teacher_instance.token_ids:
        raise ValueError(
            f"object entry {entry.object_instance_id} trie tokens do not match "
            "the teacher-forced sequence"
        )

    node = trie_root
    for offset, teacher_token_id in enumerate(teacher_token_ids):
        position = entry.trie_eligible_span.start + offset
        coord_slot_name = _coord_slot_name_for_position(entry, position)
        if teacher_token_id not in node.children:
            raise ValueError(
                f"teacher token {teacher_token_id} at position {position} is not a "
                "valid remaining-object trie child"
            )

        if coord_slot_name is None:
            active_count = node.descendant_count()
            child_multiplicities = tuple(
                (child_token_id, child.descendant_count())
                for child_token_id, child in sorted(node.children.items())
            )
        else:
            child_multiplicities = tuple(
                (child_token_id, hard_descendant_count)
                for child_token_id, child in sorted(node.children.items())
                if (
                    hard_descendant_count := sum(
                        1
                        for instance in child.descendant_instances
                        if instance.hard_bbox_supervision
                    )
                )
                > 0
            )
            active_count = sum(
                child_multiplicity
                for _, child_multiplicity in child_multiplicities
            )
        trie_branch_targets = tuple(
            TrieBranchTarget(
                token_id=child_token_id,
                multiplicity=child_multiplicity,
                probability=float(child_multiplicity / max(active_count, 1)),
            )
            for child_token_id, child_multiplicity in child_multiplicities
        )
        if trie_branch_targets:
            probability_mass = sum(target.probability for target in trie_branch_targets)
            if abs(probability_mass - 1.0) > 1e-9:
                raise ValueError(
                    f"trie child probabilities must sum to 1.0 at position {position}; "
                    f"got {probability_mass}"
                )
        kind: TrieTargetKind = (
            "trie_multi_positive" if len(trie_branch_targets) > 1 else "hard_ce"
        )
        if position not in excluded_positions:
            token_targets.append(
                TokenTarget(
                    position=position,
                    teacher_token_id=teacher_token_id,
                    kind=kind,
                    trie_branch_targets=trie_branch_targets,
                    object_instance_id=entry.object_instance_id,
                    token_role=tokenized.token_roles[position],
                    coord_soft_targets=(
                        _coord_soft_targets_for_instances(
                            tuple(
                                instance
                                for instance in node.descendant_instances
                                if instance.hard_bbox_supervision
                            ),
                            slot_name=coord_slot_name,
                        )
                        if coord_slot_name is not None
                        else ()
                    ),
                )
            )
        node = node.children[teacher_token_id]

    if node.terminal_count <= 0:
        raise ValueError(
            f"teacher path for object entry {entry.object_instance_id} did not reach "
            "a remaining serialized object entry"
        )


def _append_hard_ce_targets(
    token_targets: list[TokenTarget],
    *,
    tokenized: TokenizedDetectionExample,
    start: int,
    end: int,
    object_instance_id: str | None,
    coord_soft_targets_by_position: Mapping[int, tuple[CoordSoftTargetSpec, ...]]
    | None = None,
    excluded_positions: set[int] | frozenset[int] | tuple[int, ...] = (),
) -> None:
    coord_soft_targets_by_position = coord_soft_targets_by_position or {}
    for position in range(start, end):
        if tokenized.labels[position] == -100:
            continue
        if position in excluded_positions:
            continue
        teacher_token_id = tokenized.input_ids[position]
        token_targets.append(
            TokenTarget(
                position=position,
                teacher_token_id=teacher_token_id,
                kind="hard_ce",
                trie_branch_targets=(
                    TrieBranchTarget(
                        token_id=teacher_token_id,
                        multiplicity=1,
                        probability=1.0,
                    ),
                ),
                object_instance_id=object_instance_id,
                token_role=tokenized.token_roles[position],
                coord_soft_targets=coord_soft_targets_by_position.get(position, ()),
            )
        )


def normalize_recursive_detection_token_losses(
    recursive_targets: RecursiveDetectionTargets,
    per_token_losses: Mapping[int, float] | Sequence[float],
) -> LossNormalizationResult:
    _validate_loss_normalization_strategy(recursive_targets.normalization)
    position_losses = {
        target.position: _lookup_loss(per_token_losses, target.position)
        for target in recursive_targets.token_targets
    }
    diagnostics = _normalization_diagnostics_base(
        recursive_targets=recursive_targets,
    )

    if recursive_targets.normalization == "legacy_row_mean_equivalence":
        state_weight_sum = sum(
            target.state_weight for target in recursive_targets.token_targets
        )
        normalized_loss = sum(
            target.state_weight
            * _target_loss_weight(target)
            * position_losses[target.position]
            for target in recursive_targets.token_targets
        ) / max(state_weight_sum, 1e-12)
        diagnostics = replace(
            diagnostics,
            profile_id="legacy_row_mean_equivalence",
            state_weight_sum=float(state_weight_sum),
        )
        return LossNormalizationResult(
            profile_id="legacy_row_mean_equivalence",
            normalized_loss=float(normalized_loss),
            diagnostics=diagnostics,
        )

    target_by_position = {
        target.position: target for target in recursive_targets.token_targets
    }
    atom_losses = {
        atom.atom_id: sum(
            position_losses[position]
            * _target_loss_weight(target_by_position[position])
            for position in atom.token_positions
        )
        / max(len(atom.token_positions), 1)
        for atom in recursive_targets.loss_atoms
    }
    object_losses = _semantic_object_losses(
        recursive_targets=recursive_targets,
        atom_losses=atom_losses,
    )
    component_losses: dict[str, float] = {}
    bucket_weights: dict[str, float] = {}
    loss_terms: list[float] = []
    loss_weights: list[float] = []

    if object_losses:
        object_component = float(sum(object_losses.values()) / len(object_losses))
        component_losses["objects"] = object_component
        bucket_weights["objects"] = _IMAGE_MIXTURE_WEIGHTS["objects"]
        loss_terms.append(object_component)
        loss_weights.append(_IMAGE_MIXTURE_WEIGHTS["objects"])

    ordinary_boundary_losses = _ordinary_boundary_losses(
        recursive_targets=recursive_targets,
        atom_losses=atom_losses,
    )
    if ordinary_boundary_losses:
        component_losses["boundary"] = float(
            sum(ordinary_boundary_losses) / len(ordinary_boundary_losses)
        )
        bucket_weights["boundary_tokens"] = 1.0
        loss_terms.extend(ordinary_boundary_losses)
        loss_weights.extend(1.0 for _ in ordinary_boundary_losses)

    schema_losses = [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role is SemanticRole.SCHEMA_CONTROL
    ]
    if schema_losses:
        schema_component = float(sum(schema_losses) / len(schema_losses))
        component_losses["schema"] = schema_component
        bucket_weights["schema"] = _IMAGE_MIXTURE_WEIGHTS["schema"]
        loss_terms.append(schema_component)
        loss_weights.append(_IMAGE_MIXTURE_WEIGHTS["schema"])

    normalized_loss = _weighted_float_mean(loss_terms, loss_weights)
    diagnostics = replace(
        diagnostics,
        profile_id="semantic_image_bucket_balanced",
        state_weight_sum=float(
            sum(target.state_weight for target in recursive_targets.token_targets)
        ),
        component_losses=component_losses,
        bucket_weights=bucket_weights,
        effective_weighted_trie_contribution=float(
            _effective_weighted_trie_contribution(
                recursive_targets=recursive_targets,
                object_losses=object_losses,
            )
        ),
    )
    return LossNormalizationResult(
        profile_id="semantic_image_bucket_balanced",
        normalized_loss=float(normalized_loss),
        diagnostics=diagnostics,
    )


def _lookup_loss(
    per_token_losses: Mapping[int, float] | Sequence[float],
    position: int,
) -> float:
    if isinstance(per_token_losses, Mapping):
        if position not in per_token_losses:
            raise KeyError(f"missing per-token loss for position {position}")
        return float(per_token_losses[position])
    if position >= len(per_token_losses):
        raise IndexError(f"missing per-token loss for position {position}")
    return float(per_token_losses[position])


def _target_loss_weight(target: TokenTarget) -> float:
    weight = float(target.loss_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("TokenTarget.loss_weight must be a non-negative finite float")
    return weight


def _normalization_diagnostics_base(
    *,
    recursive_targets: RecursiveDetectionTargets,
) -> LossNormalizationDiagnostics:
    semantic_role_token_counts: dict[SemanticRole, int] = {}
    for target in recursive_targets.token_targets:
        semantic_role_token_counts[target.semantic_role] = (
            semantic_role_token_counts.get(target.semantic_role, 0) + 1
        )

    semantic_role_atom_counts: dict[SemanticRole, int] = {}
    for atom in recursive_targets.loss_atoms:
        semantic_role_atom_counts[atom.semantic_role] = (
            semantic_role_atom_counts.get(atom.semantic_role, 0) + 1
        )

    total_targets = max(len(recursive_targets.token_targets), 1)
    multi_child_targets = sum(
        1
        for target in recursive_targets.token_targets
        if target.kind == "trie_multi_positive"
    )
    boundary_targets = sum(
        1
        for target in recursive_targets.token_targets
        if target.semantic_role
        in {
            SemanticRole.SEPARATOR_CONTINUE,
            SemanticRole.TERMINAL_STOP,
            SemanticRole.CHAT_STOP,
        }
    )
    return LossNormalizationDiagnostics(
        profile_id=recursive_targets.normalization,
        state_weight_sum=0.0,
        atom_count=len(recursive_targets.loss_atoms),
        object_count=len(
            {
                atom.object_instance_id
                for atom in recursive_targets.loss_atoms
                if atom.object_instance_id is not None
            }
        ),
        gt_count_bucket=_gt_count_bucket(
            len(
                {
                    atom.object_instance_id
                    for atom in recursive_targets.loss_atoms
                    if atom.object_instance_id is not None
                }
            )
        ),
        semantic_role_token_counts=semantic_role_token_counts,
        semantic_role_atom_counts=semantic_role_atom_counts,
        multi_child_trie_token_fraction=float(multi_child_targets / total_targets),
        boundary_fraction=float(boundary_targets / total_targets),
        effective_weighted_trie_contribution=0.0,
        component_losses={},
        bucket_weights={},
    )


def _semantic_object_losses(
    *,
    recursive_targets: RecursiveDetectionTargets,
    atom_losses: Mapping[str, float],
) -> dict[str, float]:
    object_atoms: dict[str, list[LossAtom]] = {}
    for atom in recursive_targets.loss_atoms:
        if atom.object_instance_id is None:
            continue
        object_atoms.setdefault(atom.object_instance_id, []).append(atom)

    object_losses: dict[str, float] = {}
    for object_instance_id, atoms in object_atoms.items():
        atom_values = [atom_losses[atom.atom_id] for atom in atoms]
        roles = [atom.semantic_role.value for atom in atoms]
        object_losses[object_instance_id] = float(
            _weighted_mean(atom_values, roles, _OBJECT_ROLE_WEIGHTS)
        )
    return object_losses


def _ordinary_boundary_losses(
    *,
    recursive_targets: RecursiveDetectionTargets,
    atom_losses: Mapping[str, float],
) -> list[float]:
    return [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role
        in {
            SemanticRole.SEPARATOR_CONTINUE,
            SemanticRole.TERMINAL_STOP,
            SemanticRole.CHAT_STOP,
        }
    ]


def _weighted_mean(
    values: Sequence[float],
    roles: Sequence[str],
    weights: Mapping[str, float],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, role in zip(values, roles, strict=True):
        weight = float(weights[role])
        numerator += weight * float(value)
        denominator += weight
    return numerator / max(denominator, 1e-12)


def _weighted_float_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in zip(values, weights, strict=True):
        weight = float(weight)
        numerator += weight * float(value)
        denominator += weight
    return numerator / max(denominator, 1e-12)


def _effective_weighted_trie_contribution(
    *,
    recursive_targets: RecursiveDetectionTargets,
    object_losses: Mapping[str, float],
) -> float:
    trie_atoms = [
        atom
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role is SemanticRole.ENTRY_TRIE_DECISION
        and atom.object_instance_id in object_losses
    ]
    if not trie_atoms or not object_losses:
        return 0.0
    return float(
        _IMAGE_MIXTURE_WEIGHTS["objects"]
        * _OBJECT_ROLE_WEIGHTS["entry_trie_decision"]
        / sum(_OBJECT_ROLE_WEIGHTS.values())
    )


def _gt_count_bucket(object_count: int) -> str:
    if object_count <= 3:
        return "0-3"
    if object_count <= 6:
        return "4-6"
    if object_count <= 10:
        return "7-10"
    return "11+"


def _single_object_coord_soft_targets_by_position(
    entry: TokenizedObjectEntry,
    *,
    object_instance: _TrieObjectInstance,
) -> dict[int, tuple[CoordSoftTargetSpec, ...]]:
    by_position: dict[int, tuple[CoordSoftTargetSpec, ...]] = {}
    for slot_name, coord_span in zip(
        ("x1", "y1", "x2", "y2"),
        entry.coord_spans,
        strict=True,
    ):
        spec = CoordSoftTargetSpec(
            object_instance_id=object_instance.object_instance_id,
            slot_name=slot_name,
            bbox_xyxy=object_instance.bbox_xyxy,
            probability=1.0,
        )
        for position in coord_span.token_indices():
            by_position[position] = (spec,)
    return by_position


def _allows_hard_bbox_supervision(entry: TokenizedObjectEntry) -> bool:
    if entry.hard_bbox_supervision is False:
        return False
    if entry.source_role == "proxy_candidate":
        return False
    if entry.coordinate_weight == 0.0 or entry.regression_weight == 0.0:
        return False
    return True


def _entry_coord_positions(entry: TokenizedObjectEntry) -> set[int]:
    return {
        position
        for coord_span in entry.coord_spans
        for position in coord_span.token_indices()
    }


def _coord_soft_targets_for_instances(
    instances: Sequence[_TrieObjectInstance],
    *,
    slot_name: CoordSlotName,
) -> tuple[CoordSoftTargetSpec, ...]:
    if not instances:
        return ()
    probability = 1.0 / float(len(instances))
    return tuple(
        CoordSoftTargetSpec(
            object_instance_id=instance.object_instance_id,
            slot_name=slot_name,
            bbox_xyxy=instance.bbox_xyxy,
            probability=probability,
        )
        for instance in instances
    )


def _coord_slot_name_for_position(
    entry: TokenizedObjectEntry,
    position: int,
) -> CoordSlotName | None:
    for slot_name, coord_span in zip(
        ("x1", "y1", "x2", "y2"),
        entry.coord_spans,
        strict=True,
    ):
        if coord_span.start <= position < coord_span.end:
            return slot_name
    return None


def _bbox_xyxy_from_object(
    obj: NormalizedDetectionObject,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (token_to_int(token) for token in obj.bbox_2d.tokens)
    return (x1, y1, x2, y2)


def _build_entry_trie(
    remaining_instances: list[_TrieObjectInstance],
) -> _EntryTrieNode:
    root = _EntryTrieNode()
    for instance in remaining_instances:
        node = root
        node.descendant_instances.append(instance)
        for token_id in instance.token_ids:
            node = node.children.setdefault(token_id, _EntryTrieNode())
            node.descendant_instances.append(instance)
        node.terminal_count += 1
    return root


def _find_object_instance(
    remaining_instances: list[_TrieObjectInstance],
    *,
    object_instance_id: str,
) -> _TrieObjectInstance:
    for instance in remaining_instances:
        if instance.object_instance_id == object_instance_id:
            return instance
    raise ValueError(f"teacher object instance {object_instance_id} is not active")


def _remove_object_instance(
    remaining_instances: list[_TrieObjectInstance],
    *,
    object_instance_id: str,
) -> None:
    for index, instance in enumerate(remaining_instances):
        if instance.object_instance_id == object_instance_id:
            remaining_instances.pop(index)
            return
    raise ValueError(f"teacher object instance {object_instance_id} is not active")
