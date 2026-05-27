from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest


def _one_image_sample(tmp_path: Path) -> dict:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDAT"
        b"x\x9cc\xf8\x0f\x00\x01\x01\x01\x00\x1b\xb6\xeeV"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return {
        "image": str(image_path),
        "images": [str(image_path)],
        "width": 1,
        "height": 1,
        "objects": [{"bbox_2d": [10, 20, 110, 220], "desc": "cat"}],
    }


def test_prompt_codec_rejects_multiple_images(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    sample["images"] = [sample["images"][0], sample["images"][0]]

    with pytest.raises(ValueError, match="exactly one image"):
        build_prompt_bundle(sample, DetectionPromptPolicy.default_for_detection())


def test_prompt_policy_rejects_unsupported_image_count(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    policy = replace(DetectionPromptPolicy.default_for_detection(), image_count=2)

    with pytest.raises(ValueError, match="exactly one image|unsupported"):
        build_prompt_bundle(sample, policy)


def test_prompt_policy_rejects_do_resize(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    policy = replace(DetectionPromptPolicy.default_for_detection(), do_resize=True)

    with pytest.raises(ValueError, match="do_resize|unsupported"):
        build_prompt_bundle(sample, policy)


def test_prompt_policy_fingerprint_is_stable(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    policy = DetectionPromptPolicy.default_for_detection()
    first = build_prompt_bundle(sample, policy)
    second = build_prompt_bundle(sample, policy)

    assert first.prompt_policy_fingerprint == second.prompt_policy_fingerprint
    assert first.visual_metadata["image_count"] == 1
    assert first.visual_metadata["do_resize"] is False
    assert first.visual_metadata["original_width"] == 1
    assert first.visual_metadata["original_height"] == 1
    assert first.visual_metadata["post_preprocessing_width"] == 1
    assert first.visual_metadata["post_preprocessing_height"] == 1
    assert first.visual_metadata["image_placement"] == "messages[1].content[0]"


def test_prompt_codec_rejects_missing_image_file(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    missing_path = tmp_path / "missing.png"
    sample = {
        "image": str(missing_path),
        "images": [str(missing_path)],
        "width": 1,
        "height": 1,
        "objects": [],
    }

    with pytest.raises(ValueError, match="does not exist"):
        build_prompt_bundle(sample, DetectionPromptPolicy.default_for_detection())


def test_prompt_codec_resolves_relative_image_from_root_image_dir(
    tmp_path: Path,
) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    image_path = Path(sample["image"])
    sample["image"] = image_path.name
    sample["images"] = [image_path.name]

    bundle = build_prompt_bundle(
        sample,
        DetectionPromptPolicy.default_for_detection(),
        root_image_dir=tmp_path,
    )

    assert bundle.visual_metadata["image_path"] == str(image_path.resolve())


def test_prompt_codec_rejects_conflicting_image_fields(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    other_path = tmp_path / "other.png"
    other_path.write_bytes(Path(sample["image"]).read_bytes())
    sample["image"] = str(other_path)

    with pytest.raises(ValueError, match=r"image and images\[0\]"):
        build_prompt_bundle(sample, DetectionPromptPolicy.default_for_detection())


@pytest.mark.parametrize("bad_images", [123, {"path": "sample.png"}, "sample.png"])
def test_prompt_codec_rejects_malformed_images_field(
    tmp_path: Path,
    bad_images: object,
) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    sample["images"] = bad_images

    with pytest.raises(ValueError, match="sample.images"):
        build_prompt_bundle(sample, DetectionPromptPolicy.default_for_detection())


def test_prompt_codec_rejects_non_path_images_list_element(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    sample["images"] = [123]

    with pytest.raises(ValueError, match=r"sample.images\[0\]"):
        build_prompt_bundle(sample, DetectionPromptPolicy.default_for_detection())


def test_prompt_codec_rejects_stale_sample_dimensions(
    tmp_path: Path,
) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    sample["width"] = 640
    sample["height"] = 480

    with pytest.raises(ValueError, match="width does not match actual image width"):
        build_prompt_bundle(sample, DetectionPromptPolicy.default_for_detection())


def test_prompt_codec_accepts_matching_declared_dimensions(
    tmp_path: Path,
) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    sample["width"] = 1
    sample["height"] = 1

    bundle = build_prompt_bundle(sample, DetectionPromptPolicy.default_for_detection())

    assert bundle.visual_metadata["width"] == 1
    assert bundle.visual_metadata["height"] == 1
    assert bundle.visual_metadata["original_width"] == 1
    assert bundle.visual_metadata["original_height"] == 1
    assert bundle.visual_metadata["post_preprocessing_width"] == 1
    assert bundle.visual_metadata["post_preprocessing_height"] == 1
    assert bundle.visual_metadata["declared_width"] == 1
    assert bundle.visual_metadata["declared_height"] == 1


def test_prompt_messages_use_list_content_and_image_then_text(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    bundle = build_prompt_bundle(sample, DetectionPromptPolicy.default_for_detection())

    assert all(isinstance(message["content"], list) for message in bundle.messages)
    assert bundle.messages[0] == {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": DetectionPromptPolicy.default_for_detection().system_prompt,
            },
        ],
    }
    user_content = bundle.messages[1]["content"]
    assert [part["type"] for part in user_content] == ["image", "text"]


def test_prompt_bundle_does_not_tokenize_simplified_text_for_parity(
    tmp_path: Path,
) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    class SimplifiedTokenizer:
        def __call__(self, text: str) -> dict:
            return {"input_ids": [len(text)]}

    sample = _one_image_sample(tmp_path)
    bundle = build_prompt_bundle(
        sample,
        DetectionPromptPolicy.default_for_detection(),
        tokenizer=SimplifiedTokenizer(),
    )

    assert bundle.prompt_token_ids is None
    assert bundle.prompt_token_ids_source == "unavailable"
    assert bundle.prompt_token_parity == "unverifiable"


def test_prompt_bundle_records_template_token_ids_only_from_chat_template(
    tmp_path: Path,
) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    class TemplateTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str | list[int]:
            assert add_generation_prompt is True
            if tokenize:
                return [101, 202, 303]
            return f"template-rendered:{len(messages)}"

    sample = _one_image_sample(tmp_path)
    bundle = build_prompt_bundle(
        sample,
        DetectionPromptPolicy.default_for_detection(),
        tokenizer=TemplateTokenizer(),
    )

    assert bundle.prompt_text == "template-rendered:2"
    assert bundle.prompt_token_ids == [101, 202, 303]
    assert bundle.prompt_token_ids_source == "chat_template"
    assert bundle.prompt_token_parity == "unverified"


def test_prompt_parity_is_verified_only_after_bundle_comparison(
    tmp_path: Path,
) -> None:
    from src.infer.prompt import (
        DetectionPromptPolicy,
        build_prompt_bundle,
        compare_prompt_bundle_parity,
    )

    class TemplateTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str | list[int]:
            assert add_generation_prompt is True
            return [101, 202, 303] if tokenize else "template-rendered"

    sample = _one_image_sample(tmp_path)
    policy = DetectionPromptPolicy.default_for_detection()
    reference = build_prompt_bundle(sample, policy, tokenizer=TemplateTokenizer())
    candidate = build_prompt_bundle(sample, policy, tokenizer=TemplateTokenizer())

    parity = compare_prompt_bundle_parity(reference, candidate)

    assert reference.prompt_token_parity == "unverified"
    assert candidate.prompt_token_parity == "unverified"
    assert parity.prompt_token_parity == "verified"
    assert parity.verified is True


def test_prompt_parity_comparison_rejects_visual_metadata_mismatch(
    tmp_path: Path,
) -> None:
    from src.infer.prompt import (
        DetectionPromptPolicy,
        build_prompt_bundle,
        compare_prompt_bundle_parity,
    )

    class TemplateTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str | list[int]:
            assert add_generation_prompt is True
            return [101, 202, 303] if tokenize else "template-rendered"

    first = _one_image_sample(tmp_path)
    second = _one_image_sample(tmp_path)
    second_image_path = tmp_path / "second.png"
    second_image_path.write_bytes(Path(second["image"]).read_bytes())
    second["image"] = str(second_image_path)
    second["images"] = [str(second_image_path)]

    policy = DetectionPromptPolicy.default_for_detection()
    reference = build_prompt_bundle(first, policy, tokenizer=TemplateTokenizer())
    candidate = build_prompt_bundle(second, policy, tokenizer=TemplateTokenizer())

    parity = compare_prompt_bundle_parity(reference, candidate)

    assert parity.prompt_token_parity == "unverified"
    assert parity.verified is False
    assert "visual metadata" in parity.reason
