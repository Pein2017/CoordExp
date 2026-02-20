from src.trainers.rollout_matching_sft import RolloutMatchingSFTTrainer


def test_clip_text_returns_full_string_when_disabled() -> None:
    clip = RolloutMatchingSFTTrainer._clip_text

    text = "x" * 12345
    out = clip(text, max_chars=0)

    # Contract: <=0 disables clipping.
    assert out == text


def test_clip_text_truncates_when_positive() -> None:
    clip = RolloutMatchingSFTTrainer._clip_text

    text = "y" * 100
    out = clip(text, max_chars=10)

    assert out.startswith("y" * 10)
    assert out.endswith("...<truncated>")
    assert len(out) == 10 + len("...<truncated>")
