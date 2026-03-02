from pathlib import Path


def test_stage2_launcher_passes_template_flags_to_swift_rollout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "train_stage2.sh"
    src = script.read_text(encoding="utf-8")

    # Regression guard: the server process must use the same template settings as
    # learner-side template.encode to keep prompt_token_ids aligned.
    assert "SERVER_TEMPLATE" in src
    assert "SERVER_MAX_LENGTH" in src
    assert "SERVER_TRUNCATION_STRATEGY" in src
    assert "--template" in src
    assert "--max_length" in src
    assert "--truncation_strategy" in src
    assert "--max_pixels" in src
