"""Preparation helpers for detection SFT training examples."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from src.detection.data import NormalizedDetectionSample, ObjectOrderingPlan
from src.detection.template import DetectionSequenceTemplate, RenderedAssistantSequence
from src.detection.tokenization import (
    TokenRole,
    TokenizedDetectionExample,
    TokenizedObjectEntry,
    TokenizerWithOffsets,
    tokenize_rendered_detection_conversation,
)

DetectionTrainingMode = Literal[
    "sorted_sft",
    "random_order_sft",
    "random_permutation_et_rmp_ce",
]
TrieTargetKind = Literal["hard_ce", "trie_multi_positive"]
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
_BOUNDARY_ROLE_WEIGHTS = {
    "separator_continue": 0.50,
    "terminal_stop": 0.50,
}
_IMAGE_MIXTURE_WEIGHTS = {
    "objects": 1.00,
    "boundary": 0.30,
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
    component_weights: dict[str, float]


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
            f"{mode_name} requires "
            "sample.object_ordering.strategy='random_permutation'"
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


@dataclass
class _EntryTrieNode:
    terminal_count: int = 0
    children: dict[int, "_EntryTrieNode"] = field(default_factory=dict)
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
        )
        _append_recursive_entry_targets(
            token_targets,
            tokenized=tokenized,
            entry=entry,
            remaining_instances=remaining_instances,
        )
        _append_hard_ce_targets(
            token_targets,
            tokenized=tokenized,
            start=entry.trie_eligible_span.end,
            end=entry.entry_span.end,
            object_instance_id=entry.object_instance_id,
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
            float(
                (_EMPTY_PREFIX_PROBABILITY + _LEAVE_ONE_OUT_PROBABILITY) / total
            ),
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
            if position in target_by_position and position not in trie_positions
        }
        object_control_positions = entry_positions - trie_positions - desc_positions - coord_positions

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
                raise ValueError(f"duplicate LossAtom assignment for token position {position}")
            position_to_atom[position] = atom

    annotated_targets: list[TokenTarget] = []
    for target in token_targets:
        atom = position_to_atom.get(target.position)
        if atom is None:
            raise ValueError(f"missing LossAtom assignment for token position {target.position}")
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
        raise ValueError("tokenized object entries must match normalized sample objects")

    seen_ids: set[str] = set()
    instances: list[_TrieObjectInstance] = []
    for expected_index, (obj, entry) in enumerate(
        zip(sample.objects, tokenized.object_entries, strict=True)
    ):
        _validate_trie_eligible_span(entry)
        if entry.object_index != expected_index:
            raise ValueError("tokenized object entries must preserve teacher object order")
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
) -> None:
    trie_root = _build_entry_trie(remaining_instances)
    teacher_instance = _find_object_instance(
        remaining_instances,
        object_instance_id=entry.object_instance_id,
    )
    teacher_token_ids = tuple(
        tokenized.input_ids[entry.trie_eligible_span.start : entry.trie_eligible_span.end]
    )
    if teacher_token_ids != teacher_instance.token_ids:
        raise ValueError(
            f"object entry {entry.object_instance_id} trie tokens do not match "
            "the teacher-forced sequence"
        )

    node = trie_root
    for offset, teacher_token_id in enumerate(teacher_token_ids):
        position = entry.trie_eligible_span.start + offset
        if teacher_token_id not in node.children:
            raise ValueError(
                f"teacher token {teacher_token_id} at position {position} is not a "
                "valid remaining-object trie child"
            )

        active_count = node.descendant_count()
        trie_branch_targets = tuple(
            TrieBranchTarget(
                token_id=child_token_id,
                multiplicity=child.descendant_count(),
                probability=float(child.descendant_count() / max(active_count, 1)),
            )
            for child_token_id, child in sorted(node.children.items())
        )
        if trie_branch_targets:
            probability_mass = sum(target.probability for target in trie_branch_targets)
            if abs(probability_mass - 1.0) > 1e-9:
                raise ValueError(
                    f"trie child probabilities must sum to 1.0 at position {position}; "
                    f"got {probability_mass}"
                )
        kind: TrieTargetKind = (
            "trie_multi_positive"
            if len(trie_branch_targets) > 1
            else "hard_ce"
        )
        token_targets.append(
            TokenTarget(
                position=position,
                teacher_token_id=teacher_token_id,
                kind=kind,
                trie_branch_targets=trie_branch_targets,
                object_instance_id=entry.object_instance_id,
                token_role=tokenized.token_roles[position],
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
) -> None:
    for position in range(start, end):
        if tokenized.labels[position] == -100:
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
            target.state_weight * position_losses[target.position]
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

    atom_losses = {
        atom.atom_id: sum(position_losses[position] for position in atom.token_positions)
        / max(len(atom.token_positions), 1)
        for atom in recursive_targets.loss_atoms
    }
    object_losses = _semantic_object_losses(
        recursive_targets=recursive_targets,
        atom_losses=atom_losses,
    )
    component_losses: dict[str, float] = {}
    component_weights: dict[str, float] = {}

    if object_losses:
        component_losses["objects"] = float(sum(object_losses.values()) / len(object_losses))
        component_weights["objects"] = _IMAGE_MIXTURE_WEIGHTS["objects"]

    boundary_losses = _semantic_boundary_losses(
        recursive_targets=recursive_targets,
        atom_losses=atom_losses,
    )
    if boundary_losses:
        component_losses["boundary"] = float(
            _weighted_mean(
                list(boundary_losses.values()),
                list(boundary_losses.keys()),
                _BOUNDARY_ROLE_WEIGHTS,
            )
        )
        component_weights["boundary"] = _IMAGE_MIXTURE_WEIGHTS["boundary"]

    schema_losses = [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role is SemanticRole.SCHEMA_CONTROL
    ]
    if schema_losses:
        component_losses["schema"] = float(sum(schema_losses) / len(schema_losses))
        component_weights["schema"] = _IMAGE_MIXTURE_WEIGHTS["schema"]

    normalized_loss = _renormalized_component_mean(
        component_losses=component_losses,
        component_weights=component_weights,
    )
    diagnostics = replace(
        diagnostics,
        profile_id="semantic_image_bucket_balanced",
        state_weight_sum=float(
            sum(target.state_weight for target in recursive_targets.token_targets)
        ),
        component_losses=component_losses,
        component_weights=component_weights,
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
        component_weights={},
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


def _semantic_boundary_losses(
    *,
    recursive_targets: RecursiveDetectionTargets,
    atom_losses: Mapping[str, float],
) -> dict[str, float]:
    boundary_losses: dict[str, float] = {}

    separator_losses = [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role is SemanticRole.SEPARATOR_CONTINUE
    ]
    if separator_losses:
        boundary_losses["separator_continue"] = float(
            sum(separator_losses) / len(separator_losses)
        )

    stop_losses = [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role in {SemanticRole.TERMINAL_STOP, SemanticRole.CHAT_STOP}
    ]
    if stop_losses:
        boundary_losses["terminal_stop"] = float(sum(stop_losses) / len(stop_losses))

    return boundary_losses


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


def _renormalized_component_mean(
    *,
    component_losses: Mapping[str, float],
    component_weights: Mapping[str, float],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for component_name, component_loss in component_losses.items():
        weight = float(component_weights[component_name])
        numerator += weight * float(component_loss)
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


def _build_entry_trie(
    remaining_instances: list[_TrieObjectInstance],
) -> _EntryTrieNode:
    root = _EntryTrieNode()
    for instance in remaining_instances:
        node = root
        for token_id in instance.token_ids:
            node = node.children.setdefault(token_id, _EntryTrieNode())
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
