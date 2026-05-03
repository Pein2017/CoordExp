from __future__ import annotations

from dataclasses import replace
import re

import pytest

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.objective import build_recursive_detection_targets, prepare_detection_training_example
from src.detection.template import Stage1JsonPrettyTemplate

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class SpecialTokenAwareTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False

        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            match = _SPECIAL_TOKEN_RE.match(text, cursor)
            if match is not None:
                token_text = match.group(0)
                token_end = match.end()
            else:
                token_text = text[cursor]
                token_end = cursor + 1
            token_id = self._token_to_id.setdefault(token_text, len(self._token_to_id) + 1)
            self._id_to_token.setdefault(token_id, token_text)
            input_ids.append(token_id)
            offsets.append((cursor, token_end))
            cursor = token_end

        return {
            "input_ids": input_ids,
            "offset_mapping": offsets,
        }


def _object(
    *,
    normalized_index: int,
    source_index: int,
    instance_id: str,
    desc: str,
    coords: tuple[str, str, str, str],
) -> NormalizedDetectionObject:
    return NormalizedDetectionObject(
        normalized_object_index=normalized_index,
        source_object_index=source_index,
        object_instance_id=instance_id,
        desc=desc,
        bbox_2d=CoordinateTokenBox(*coords),
        category_id=normalized_index + 1,
        category_name=desc,
        coco_ann_id=7000 + source_index,
    )


def _sample(
    *objects: NormalizedDetectionObject,
) -> NormalizedDetectionSample:
    realized = tuple(obj.source_object_index for obj in objects)
    indexed_objects = tuple(
        replace(obj, normalized_object_index=index)
        for index, obj in enumerate(objects)
    )
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=indexed_objects,
        width=640,
        height=480,
        image_id=17,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=23,
            seed_source="unit-test",
        ).with_realized(realized),
    )


def _prepare(
    sample: NormalizedDetectionSample,
    *,
    state_weighting: str,
) -> object:
    return prepare_detection_training_example(
        sample,
        template=Stage1JsonPrettyTemplate(),
        tokenizer=SpecialTokenAwareTokenizer(),
        mode="random_permutation_et_rmp_ce",
        state_weighting=state_weighting,
        normalization="legacy_row_mean_equivalence",
    )


def _target_map(prepared: object) -> dict[int, object]:
    assert prepared.recursive_detection_targets is not None
    return {
        target.position: target
        for target in prepared.recursive_detection_targets.token_targets
    }


def _first_supervised_target_between(
    targets: dict[int, object],
    *,
    start: int,
    end: int,
) -> object:
    for position in range(start, end):
        target = targets.get(position)
        if target is not None:
            return target
    raise AssertionError(f"no supervised token target in range [{start}, {end})")


def test_n_gt_one_entry_exposures_follow_prefix_mixture_formula() -> None:
    prepared = _prepare(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-17:ann-701:src-7",
                desc="cat",
                coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
            ),
            _object(
                normalized_index=1,
                source_index=3,
                instance_id="img-17:ann-702:src-3",
                desc="dog",
                coords=("<|coord_110|>", "<|coord_120|>", "<|coord_130|>", "<|coord_140|>"),
            ),
            _object(
                normalized_index=2,
                source_index=9,
                instance_id="img-17:ann-703:src-9",
                desc="owl",
                coords=("<|coord_210|>", "<|coord_220|>", "<|coord_230|>", "<|coord_240|>"),
            ),
        ),
        state_weighting="legacy_row_mean_prefix_mixture_equivalence",
    )

    targets = _target_map(prepared)
    first_entry, second_entry, third_entry = prepared.tokenized.object_entries

    first_object = _first_supervised_target_between(
        targets,
        start=first_entry.entry_span.start,
        end=first_entry.entry_span.end,
    )
    second_object = _first_supervised_target_between(
        targets,
        start=second_entry.entry_span.start,
        end=second_entry.entry_span.end,
    )
    third_object = _first_supervised_target_between(
        targets,
        start=third_entry.entry_span.start,
        end=third_entry.entry_span.end,
    )
    first_separator = _first_supervised_target_between(
        targets,
        start=first_entry.separator_span.start,
        end=first_entry.separator_span.end,
    )
    second_separator = _first_supervised_target_between(
        targets,
        start=second_entry.separator_span.start,
        end=second_entry.separator_span.end,
    )

    assert first_object.state_exposure == pytest.approx(0.30)
    assert second_object.state_exposure == pytest.approx(0.30 + 0.45 * 0.5)
    assert third_object.state_exposure == pytest.approx(0.30 + 0.45 + 0.20)
    assert first_separator.state_exposure == pytest.approx(0.30)
    assert second_separator.state_exposure == pytest.approx(0.30 + 0.45 * 0.5)


def test_opening_schema_tokens_only_use_empty_prefix_and_row_mean_alpha() -> None:
    prepared = _prepare(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-17:ann-801:src-7",
                desc="cat",
                coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
            ),
            _object(
                normalized_index=1,
                source_index=3,
                instance_id="img-17:ann-802:src-3",
                desc="dog",
                coords=("<|coord_110|>", "<|coord_120|>", "<|coord_130|>", "<|coord_140|>"),
            ),
        ),
        state_weighting="legacy_row_mean_prefix_mixture_equivalence",
    )

    assert prepared.recursive_detection_targets is not None
    diagnostics = prepared.recursive_detection_targets.state_weighting_diagnostics
    targets = _target_map(prepared)
    first_entry = prepared.tokenized.object_entries[0]
    opening_target = _first_supervised_target_between(
        targets,
        start=prepared.tokenized.assistant_token_span.start,
        end=first_entry.entry_span.start,
    )

    expected = (
        diagnostics.prefix_length_probabilities[0]
        / diagnostics.supervised_token_counts_by_prefix_length[0]
    )
    assert opening_target.state_exposure == pytest.approx(0.30)
    assert opening_target.state_weight == pytest.approx(expected)


def test_separator_tokens_include_matching_exposure_and_row_mean_denominators() -> None:
    prepared = _prepare(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-17:ann-901:src-7",
                desc="cat",
                coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
            ),
            _object(
                normalized_index=1,
                source_index=3,
                instance_id="img-17:ann-902:src-3",
                desc="dog",
                coords=("<|coord_110|>", "<|coord_120|>", "<|coord_130|>", "<|coord_140|>"),
            ),
            _object(
                normalized_index=2,
                source_index=9,
                instance_id="img-17:ann-903:src-9",
                desc="owl",
                coords=("<|coord_210|>", "<|coord_220|>", "<|coord_230|>", "<|coord_240|>"),
            ),
        ),
        state_weighting="legacy_row_mean_prefix_mixture_equivalence",
    )

    assert prepared.recursive_detection_targets is not None
    diagnostics = prepared.recursive_detection_targets.state_weighting_diagnostics
    targets = _target_map(prepared)
    second_entry = prepared.tokenized.object_entries[1]
    second_separator = _first_supervised_target_between(
        targets,
        start=second_entry.separator_span.start,
        end=second_entry.separator_span.end,
    )

    expected = sum(
        diagnostics.prefix_length_probabilities[prefix_length]
        / diagnostics.supervised_token_counts_by_prefix_length[prefix_length]
        for prefix_length in (0, 1)
    )
    assert second_separator.state_exposure == pytest.approx(0.30 + 0.45 * 0.5)
    assert second_separator.state_weight == pytest.approx(expected)


def test_terminal_close_and_chat_stop_have_unit_exposure_and_row_mean_alpha() -> None:
    prepared = _prepare(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-17:ann-1001:src-7",
                desc="cat",
                coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
            ),
            _object(
                normalized_index=1,
                source_index=3,
                instance_id="img-17:ann-1002:src-3",
                desc="dog",
                coords=("<|coord_110|>", "<|coord_120|>", "<|coord_130|>", "<|coord_140|>"),
            ),
        ),
        state_weighting="legacy_row_mean_prefix_mixture_equivalence",
    )

    assert prepared.recursive_detection_targets is not None
    diagnostics = prepared.recursive_detection_targets.state_weighting_diagnostics
    targets = _target_map(prepared)
    terminal_target = _first_supervised_target_between(
        targets,
        start=prepared.tokenized.terminal_span.start,
        end=prepared.tokenized.terminal_span.end,
    )
    chat_stop_span = prepared.tokenized.stop_marker_spans[-1]
    chat_stop_target = _first_supervised_target_between(
        targets,
        start=chat_stop_span.start,
        end=chat_stop_span.end,
    )
    expected = sum(
        probability / count
        for probability, count in zip(
            diagnostics.prefix_length_probabilities,
            diagnostics.supervised_token_counts_by_prefix_length,
            strict=True,
        )
    )

    assert terminal_target.state_exposure == pytest.approx(1.0)
    assert terminal_target.state_weight == pytest.approx(expected)
    assert chat_stop_target.state_exposure == pytest.approx(1.0)
    assert chat_stop_target.state_weight == pytest.approx(expected)


def test_singleton_object_uses_explicit_degenerate_rule() -> None:
    prepared = _prepare(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-17:ann-1101:src-7",
                desc="cat",
                coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
            ),
        ),
        state_weighting="legacy_row_mean_prefix_mixture_equivalence",
    )

    assert prepared.recursive_detection_targets is not None
    diagnostics = prepared.recursive_detection_targets.state_weighting_diagnostics
    targets = _target_map(prepared)
    entry = prepared.tokenized.object_entries[0]
    object_target = _first_supervised_target_between(
        targets,
        start=entry.entry_span.start,
        end=entry.entry_span.end,
    )
    terminal_target = _first_supervised_target_between(
        targets,
        start=prepared.tokenized.terminal_span.start,
        end=prepared.tokenized.terminal_span.end,
    )

    assert diagnostics.prefix_length_probabilities == pytest.approx((10.0 / 11.0, 1.0 / 11.0))
    assert object_target.state_exposure == pytest.approx(10.0 / 11.0)
    assert terminal_target.state_exposure == pytest.approx(1.0)


def test_uniform_permutation_assigns_unit_weight_to_every_supervised_target() -> None:
    prepared = _prepare(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-17:ann-1201:src-7",
                desc="cat",
                coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
            ),
            _object(
                normalized_index=1,
                source_index=3,
                instance_id="img-17:ann-1202:src-3",
                desc="dog",
                coords=("<|coord_110|>", "<|coord_120|>", "<|coord_130|>", "<|coord_140|>"),
            ),
        ),
        state_weighting="uniform_permutation",
    )

    assert prepared.recursive_detection_targets is not None
    diagnostics = prepared.recursive_detection_targets.state_weighting_diagnostics
    assert all(
        target.state_weight == pytest.approx(1.0)
        for target in prepared.recursive_detection_targets.token_targets
    )
    assert all(
        target.state_exposure == pytest.approx(1.0)
        for target in prepared.recursive_detection_targets.token_targets
    )
    assert diagnostics.entry_exposures == pytest.approx((1.0, 1.0))
    assert diagnostics.separator_exposures == pytest.approx((1.0,))
    assert diagnostics.terminal_exposure == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_weighting": "typo_strategy"},
        {"normalization": "typo_norm"},
    ],
)
def test_recursive_profiles_fail_fast_on_invalid_ids(
    kwargs: dict[str, str],
) -> None:
    sample = _sample(
        _object(
            normalized_index=0,
            source_index=7,
            instance_id="img-17:ann-1301:src-7",
            desc="cat",
            coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ),
    )

    with pytest.raises(ValueError, match="state_weighting|normalization"):
        prepare_detection_training_example(
            sample,
            template=Stage1JsonPrettyTemplate(),
            tokenizer=SpecialTokenAwareTokenizer(),
            mode="random_permutation_et_rmp_ce",
            **kwargs,
        )


def test_build_recursive_targets_fails_fast_on_invalid_profile_ids() -> None:
    prepared = _prepare(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-17:ann-1401:src-7",
                desc="cat",
                coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
            ),
        ),
        state_weighting="uniform_permutation",
    )

    with pytest.raises(ValueError, match="state_weighting"):
        build_recursive_detection_targets(
            prepared.normalized_sample,
            tokenized=prepared.tokenized,
            state_weighting="not_a_strategy",
        )
