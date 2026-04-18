from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

import src.analysis.unmatched_proposal_verifier as verifier_module
from src.infer.checkpoints import (
    AdapterCheckpointInfo,
    ResolvedInferenceCheckpoint,
)


def test_teacher_forced_scorer_uses_base_processor_for_adapter_shorthand(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_dir = tmp_path / "base-model"
    adapter_dir = tmp_path / "adapter-checkpoint"
    base_dir.mkdir()
    adapter_dir.mkdir()
    monkeypatch.setattr(
        verifier_module,
        "resolve_inference_checkpoint",
        lambda model_checkpoint: ResolvedInferenceCheckpoint(
            checkpoint_mode="adapter_shorthand",
            requested_model_checkpoint=str(adapter_dir),
            requested_adapter_checkpoint=None,
            resolved_base_model_checkpoint=str(base_dir),
            resolved_adapter_checkpoint=str(adapter_dir),
            adapter_info=AdapterCheckpointInfo(
                path=str(adapter_dir),
                base_model_name_or_path=str(base_dir),
                modules_to_save=(),
                coord_offset_spec=None,
            ),
        ),
    )

    class _Tokenizer:
        pad_token_id = None
        eos_token_id = 7
        padding_side = "right"

    class _Processor:
        tokenizer = _Tokenizer()

    captured: dict[str, object] = {}

    def _fake_from_pretrained(
        path: str,
        *,
        trust_remote_code: bool,
        local_files_only: bool,
    ) -> _Processor:
        captured["path"] = path
        captured["trust_remote_code"] = trust_remote_code
        captured["local_files_only"] = local_files_only
        return _Processor()

    monkeypatch.setattr(
        verifier_module.TeacherForcedScorer,
        "_load_model",
        lambda self: object(),
    )
    monkeypatch.setattr(
        verifier_module.AutoProcessor,
        "from_pretrained",
        _fake_from_pretrained,
    )

    scorer = verifier_module.TeacherForcedScorer(
        checkpoint_path=adapter_dir,
        device="cuda:0",
    )

    assert scorer.resolved_checkpoint.checkpoint_mode == "adapter_shorthand"
    assert captured["path"] == str(base_dir)
    assert captured["trust_remote_code"] is True
    assert captured["local_files_only"] is True
    assert scorer.tokenizer.padding_side == "left"
    assert scorer.tokenizer.pad_token_id == 7


def test_teacher_forced_scorer_build_messages_uses_requested_coord_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_dir = tmp_path / "base-model"
    checkpoint_dir = tmp_path / "checkpoint"
    base_dir.mkdir()
    checkpoint_dir.mkdir()
    monkeypatch.setattr(
        verifier_module,
        "resolve_inference_checkpoint",
        lambda model_checkpoint: ResolvedInferenceCheckpoint(
            checkpoint_mode="full_model",
            requested_model_checkpoint=str(checkpoint_dir),
            requested_adapter_checkpoint=None,
            resolved_base_model_checkpoint=str(base_dir),
            resolved_adapter_checkpoint=None,
            adapter_info=AdapterCheckpointInfo(
                path=str(checkpoint_dir),
                base_model_name_or_path=str(base_dir),
                modules_to_save=(),
                coord_offset_spec=None,
            ),
        ),
    )

    class _Tokenizer:
        pad_token_id = None
        eos_token_id = 7
        padding_side = "right"

    class _Processor:
        tokenizer = _Tokenizer()

    captured: dict[str, object] = {}

    def _fake_get_template_prompts(
        *,
        ordering: str,
        coord_mode: str,
        prompt_variant: str,
        object_field_order: str,
    ) -> tuple[str, str]:
        captured["ordering"] = ordering
        captured["coord_mode"] = coord_mode
        captured["prompt_variant"] = prompt_variant
        captured["object_field_order"] = object_field_order
        return ("system", "user")

    monkeypatch.setattr(
        verifier_module.TeacherForcedScorer,
        "_load_model",
        lambda self: object(),
    )
    monkeypatch.setattr(
        verifier_module.AutoProcessor,
        "from_pretrained",
        lambda path, **kwargs: _Processor(),
    )
    monkeypatch.setattr(
        verifier_module,
        "get_template_prompts",
        _fake_get_template_prompts,
    )

    scorer = verifier_module.TeacherForcedScorer(
        checkpoint_path=checkpoint_dir,
        device="cuda:0",
        coord_mode="norm1000_text",
    )

    prompt_messages, full_messages = scorer.build_messages(
        image=Image.new("RGB", (8, 8), color=(0, 0, 0)),
        assistant_text='{"objects": []}',
        prompt_variant="qwen3_vl",
        object_field_order="desc_first",
    )

    assert captured == {
        "ordering": "sorted",
        "coord_mode": "norm1000_text",
        "prompt_variant": "qwen3_vl",
        "object_field_order": "desc_first",
    }
    assert len(prompt_messages) == 2
    assert len(full_messages) == 3


def test_teacher_forced_scorer_score_prepared_batch_spans_handles_left_padding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_dir = tmp_path / "base-model"
    checkpoint_dir = tmp_path / "checkpoint"
    base_dir.mkdir()
    checkpoint_dir.mkdir()
    monkeypatch.setattr(
        verifier_module,
        "resolve_inference_checkpoint",
        lambda model_checkpoint: ResolvedInferenceCheckpoint(
            checkpoint_mode="full_model",
            requested_model_checkpoint=str(checkpoint_dir),
            requested_adapter_checkpoint=None,
            resolved_base_model_checkpoint=str(base_dir),
            resolved_adapter_checkpoint=None,
            adapter_info=AdapterCheckpointInfo(
                path=str(checkpoint_dir),
                base_model_name_or_path=str(base_dir),
                modules_to_save=(),
                coord_offset_spec=None,
            ),
        ),
    )

    class _Tokenizer:
        pad_token_id = None
        eos_token_id = 0
        padding_side = "right"

    encoded = {
        "short": [10, 11, 12],
        "longer": [20, 21, 22, 23, 24],
    }

    class _Processor:
        tokenizer = _Tokenizer()

        def __call__(
            self,
            *,
            text: list[str],
            images: list[object],
            return_tensors: str,
            padding: bool,
        ) -> dict[str, torch.Tensor]:
            del images, return_tensors, padding
            rows = [encoded[item] for item in text]
            padded_len = max(len(row) for row in rows)
            padded_rows = [
                [0] * (padded_len - len(row)) + list(row)
                for row in rows
            ]
            return {"input_ids": torch.tensor(padded_rows, dtype=torch.long)}

    class _Model:
        def __call__(self, **kwargs):
            input_ids = kwargs["input_ids"]
            logits = torch.full(
                (
                    int(input_ids.shape[0]),
                    int(input_ids.shape[1]),
                    32,
                ),
                -20.0,
            )
            logits[0, 2, 11] = 5.0
            logits[0, 3, 12] = 5.0
            logits[1, 2, 23] = 5.0
            logits[1, 3, 24] = 5.0
            return type("Outputs", (), {"logits": logits})()

    monkeypatch.setattr(
        verifier_module.TeacherForcedScorer,
        "_load_model",
        lambda self: _Model(),
    )
    monkeypatch.setattr(
        verifier_module.AutoProcessor,
        "from_pretrained",
        lambda path, **kwargs: _Processor(),
    )

    scorer = verifier_module.TeacherForcedScorer(
        checkpoint_path=checkpoint_dir,
        device="cpu",
        coord_mode="norm1000_text",
    )
    examples = [
        verifier_module.PreparedExample(
            full_text="short",
            assistant_text="short",
            desc_positions=[],
            full_input_ids=encoded["short"],
        ),
        verifier_module.PreparedExample(
            full_text="longer",
            assistant_text="longer",
            desc_positions=[],
            full_input_ids=encoded["longer"],
        ),
    ]

    scored = scorer.score_prepared_batch_spans(
        examples=examples,
        images=[
            Image.new("RGB", (2, 2), color=(0, 0, 0)),
            Image.new("RGB", (2, 2), color=(0, 0, 0)),
        ],
        spans_list=[[1, 2], [3, 4]],
    )

    assert [row["count"] for row in scored] == [2, 2]
    assert all(float(row["mean_logprob"]) > -0.1 for row in scored)
