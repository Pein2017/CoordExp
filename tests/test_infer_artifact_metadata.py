from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.infer.artifacts import (
    build_infer_resolved_meta,
    build_infer_resolved_meta_from_facts,
    build_infer_summary_payload,
    build_infer_summary_payload_from_facts,
    load_comparable_artifact,
    resolve_infer_artifact_facts_from_owner,
)


class _Counters:
    def to_summary(self) -> dict[str, int]:
        return {"count": 0}


def _owner(
    *,
    distributed: bool = False,
    requested_mode: str = "text",
    backend_cfg: dict[str, object] | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        model_checkpoint="model",
        adapter_checkpoint=None,
        checkpoint_mode="full_model",
        requested_model_checkpoint="model",
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint="model",
        resolved_adapter_checkpoint=None,
        gt_jsonl="gt.jsonl",
        pred_coord_mode="auto",
        device="cpu",
        limit=200,
        distributed_enabled=distributed,
        rank=2 if distributed else 0,
        local_rank=1 if distributed else 0,
        world_size=4 if distributed else 1,
        backend=backend_cfg,
    )
    gen_cfg = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        batch_size=1,
        seed=123,
        stop_pressure_mode=None,
        stop_pressure_min_new_tokens=0,
        stop_pressure_trigger_rule=None,
        stop_pressure_logit_bias=0.0,
        stop_pressure_active=False,
    )
    return SimpleNamespace(
        cfg=cfg,
        gen_cfg=gen_cfg,
        resolved_mode="text",
        requested_mode=requested_mode,
        mode_reason="requested",
        prompt_variant="coco_80",
        bbox_format="xyxy",
        detection_sequence_format="compact_full",
        object_field_order="compact_full_row",
        object_ordering="original",
        prompt_template_hash="0" * 64,
        attn_implementation_requested=None,
        attn_implementation_selected=None,
        qwen_generation_token_ids=SimpleNamespace(
            eos_token_id=151645,
            pad_token_id=151643,
        ),
    )


def test_infer_artifacts_do_not_emit_grammar_decode_provenance() -> None:
    owner = _owner()

    resolved = build_infer_resolved_meta(
        owner=owner,
        backend="hf",
        batch_size=1,
        out_path=Path("gt_vs_pred.jsonl"),
        summary_path=Path("summary.json"),
        trace_path=Path("pred_token_trace.jsonl"),
    )
    summary = build_infer_summary_payload(
        owner=owner,
        counters=_Counters(),
        backend="hf",
        determinism="deterministic",
        batch_size=1,
    )

    assert resolved["detection_sequence_format"] == "compact_full"
    assert "compact_grammar" not in resolved["generation"]
    assert resolved["generation"]["qwen_chat_generation"] == {
        "eos_token": "<|im_end|>",
        "eos_token_id": 151645,
        "pad_token": "<|endoftext|>",
        "pad_token_id": 151643,
        "stop_tokens": ["<|im_end|>"],
        "processor_do_resize": False,
    }
    assert summary["infer"]["detection_sequence_format"] == "compact_full"
    assert "compact_grammar" not in summary["generation"]
    assert summary["generation"]["qwen_chat_generation"] == {
        "eos_token": "<|im_end|>",
        "eos_token_id": 151645,
        "pad_token": "<|endoftext|>",
        "pad_token_id": 151643,
        "stop_tokens": ["<|im_end|>"],
        "processor_do_resize": False,
    }


def test_infer_artifacts_core_builders_consume_resolved_facts() -> None:
    cases = [
        ("hf", _owner()),
        (
            "vllm",
            _owner(
                backend_cfg={
                    "mode": "server",
                    "base_url": "http://127.0.0.1:8000",
                    "model": "demo",
                    "timeout_s": 30,
                    "client_concurrency": 2,
                    "private": "hidden",
                },
            ),
        ),
        ("hf", _owner(distributed=True)),
        ("hf", _owner(requested_mode="auto")),
    ]

    for backend, owner in cases:
        facts = resolve_infer_artifact_facts_from_owner(
            owner=owner,
            backend=backend,
            batch_size=1,
        )
        resolved = build_infer_resolved_meta_from_facts(
            facts=facts,
            out_path=Path("gt_vs_pred.jsonl"),
            summary_path=Path("summary.json"),
            trace_path=Path("pred_token_trace.jsonl"),
        )
        summary = build_infer_summary_payload_from_facts(
            facts=facts,
            counters=_Counters(),
            determinism="deterministic",
        )

        assert resolved == build_infer_resolved_meta(
            owner=owner,
            backend=backend,
            batch_size=1,
            out_path=Path("gt_vs_pred.jsonl"),
            summary_path=Path("summary.json"),
            trace_path=Path("pred_token_trace.jsonl"),
        )
        assert summary == build_infer_summary_payload(
            owner=owner,
            counters=_Counters(),
            backend=backend,
            determinism="deterministic",
            batch_size=1,
        )
        if backend == "vllm":
            assert resolved["backend_cfg"] == {
                "mode": "server",
                "base_url": "http://127.0.0.1:8000",
                "model": "demo",
                "timeout_s": 30,
                "client_concurrency": 2,
            }
        if owner.cfg.distributed_enabled:
            assert resolved["distributed"]["world_size"] == 4
            assert summary["distributed"]["rank"] == 2
        if owner.requested_mode == "auto":
            assert summary["mode_resolution_reason"] == owner.mode_reason


def test_comparable_artifact_loader_accepts_run_relative_artifact_bindings(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("run")
    run_dir.mkdir()
    raw_path = run_dir / "gt_vs_pred.jsonl"
    scored_path = run_dir / "gt_vs_pred_scored.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    scored_path.write_text("{}\n", encoding="utf-8")

    generation_provenance = {
        "prompt_policy_fingerprint": "prompt_policy:v1:" + "a" * 64,
        "decode_policy_fingerprint": "decode:" + "b" * 64,
        "model_identity_fingerprint": "model:" + "c" * 64,
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "inference_provenance": {
                    "comparable": True,
                    "score_policy": "none",
                    **generation_provenance,
                },
                "artifacts": {
                    "gt_vs_pred_jsonl": str(raw_path),
                    "gt_vs_pred_scored_jsonl": str(scored_path),
                },
            }
        ),
        encoding="utf-8",
    )
    scored_path.with_suffix(scored_path.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                **generation_provenance,
                "score_policy_fingerprint": "score_policy:v1:" + "d" * 64,
                "artifact_path": str(scored_path),
                "metric_bearing": True,
            }
        ),
        encoding="utf-8",
    )

    raw_loaded = load_comparable_artifact(raw_path)
    scored_loaded = load_comparable_artifact(scored_path, require_score=True)

    assert raw_loaded["provenance_path"] == str(run_dir / "resolved_config.json")
    assert scored_loaded["provenance_path"] == str(
        scored_path.with_suffix(scored_path.suffix + ".provenance.json")
    )
