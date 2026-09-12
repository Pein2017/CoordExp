from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import pytest
import torch

from src.config.inference import load_research_infer_config
from probes.dora_owner_learning import prepare
from probes.dora_owner_learning import train


def test_source256_round_modules_exist() -> None:
    assert prepare.K == 4
    assert train.WORLD_SIZE == 8


def test_action_identity_distinguishes_natural_stop_from_length_cap_and_accepts_immediate_eos() -> None:
    assert prepare._action_token_ids([7, 8], "im_end") == [7, 8, prepare.EOS_TOKEN_ID]
    assert prepare._action_token_ids([7, 8], "length") == [7, 8]
    assert prepare._action_token_ids([], "im_end") == [prepare.EOS_TOKEN_ID]
    with pytest.raises(ValueError, match="length-stopped action cannot be empty"):
        prepare._action_token_ids([], "length")
    with pytest.raises(ValueError, match="unexpectedly contains"):
        prepare._action_token_ids([prepare.EOS_TOKEN_ID], "im_end")


def _first_raw() -> object:
    from src.data import load_raw_examples

    return load_raw_examples(prepare.TRAIN256)[0]


def _matching_prediction(raw: object) -> dict[str, object]:
    from src.data.geometry import coord_bins_to_pixel_xyxy

    obj = raw.objects[0]
    box = coord_bins_to_pixel_xyxy(
        obj.bbox,
        image_width=raw.image.width,
        image_height=raw.image.height,
        field="test",
    )
    return {"description": obj.description, "bbox": list(box)}


def _rollout_artifact(raw: object, *, missing_last: bool = False) -> dict[str, object]:
    prompt_ids = [101, 102]
    meta = {
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": prepare.json_sha256(prompt_ids),
        "chat_text_sha256": "chat",
        "image_path": str(raw.image.path),
        "image_sha256": prepare.file_sha256(raw.image.path),
        "width": raw.image.width,
        "height": raw.image.height,
    }
    rows = []
    for offset, seed in enumerate(prepare._expected_seeds(1)):
        body = [] if offset == 0 else [200 + offset]
        rows.append({
            "image_id": prepare._physical_image_id(raw),
            "example_id": raw.example_id,
            "seed": seed,
            "decode_mode": "sampled",
            "generated_token_ids": body,
            "generated_token_ids_sha256": prepare.json_sha256(body),
            "generated_text": "",
            "stop_reason": "im_end" if offset == 0 else "length",
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": prepare.json_sha256(prompt_ids),
            "observed_image_grid_thw": [1, 2, 3],
            "executed_media_sha256": "a" * 64,
            "predictions": {
                "valid_prediction_count": 1,
                "dropped_prediction_count": 1,
                "predictions": [_matching_prediction(raw)],
                "dropped_predictions": [{"reason": "incomplete"}],
            },
        })
    if missing_last:
        rows.pop()
    return {
        "schema_version": prepare.ROLLOUT_SCHEMA,
        "config": {
            "seeds": list(prepare._expected_seeds(1)),
            "decode_mode": "sampled",
            "raw_softmax": True,
            "top_k": 0,
            "generation_config_source": "fresh_transformers_generation_config",
            "use_model_defaults": False,
        },
        "prompt_metadata": {raw.example_id: meta},
        "rollouts": rows,
    }


def test_imperfect_valid_rewards_and_flat_k4_actions_are_all_retained() -> None:
    raw = _first_raw()
    artifact = _rollout_artifact(raw)
    groups = prepare._rloo_groups([raw], [(Path("bank.json"), artifact)], round_index=1)
    assert len(groups) == 1
    group = groups[0]
    assert group["semantic_flat"] is True
    assert len(group["actions"]) == 4
    assert group["actions"][0]["action_token_ids"] == [prepare.EOS_TOKEN_ID]
    assert all(action["matching"]["matched_owner_count"] == 1 for action in group["actions"])
    assert all(action["matching"]["invalid_prediction_count"] == 0 for action in group["actions"])
    assert all(action["matching"]["parser_dropped_prediction_count"] == 1 for action in group["actions"])
    assert all(action["advantage"] == 0.0 for action in group["actions"])

    plan_groups = []
    for index in range(8):
        copied = json.loads(json.dumps(group))
        copied.update(image_id=str(index), example_id=f"row-{index}", source_order=index)
        plan_groups.append(copied)
    assignments = prepare._assign(plan_groups)
    plan = {
        "schema_version": prepare.PLAN_SCHEMA,
        "arm": "rloo",
        "round": 1,
        "sources": {},
        "model": {},
        "lineage": None,
        "population": {
            "mode": "qualification",
            "image_count": 8,
            "world_size": 8,
            "images_per_rank": 1,
            "k": 4,
            "actions_per_image": 4,
            "groups": plan_groups,
            "assignments": assignments,
        },
        "objective": {},
        "sampling": {
            "max_new_tokens": prepare.PRODUCTION_MAX_NEW_TOKENS,
            "qualification_max_new_tokens": None,
        },
        "unsupported": [],
    }
    plan["content_sha256"] = prepare.json_sha256(plan)
    observed = prepare.validate_plan(plan, verify_sources=False)
    assert sum(len(item["actions"]) for item in observed["population"]["groups"]) == 32
    assert all(item["semantic_flat"] for item in observed["population"]["groups"])

    shortcap = json.loads(json.dumps(plan))
    shortcap["sampling"] = {
        "max_new_tokens": 32,
        "qualification_max_new_tokens": 32,
    }
    shortcap["content_sha256"] = prepare.json_sha256(prepare._plan_content(shortcap))
    assert prepare.validate_plan(shortcap, verify_sources=False)["sampling"]["max_new_tokens"] == 32

    default_mismatch = json.loads(json.dumps(shortcap))
    default_mismatch["sampling"]["max_new_tokens"] = prepare.PRODUCTION_MAX_NEW_TOKENS
    default_mismatch["content_sha256"] = prepare.json_sha256(prepare._plan_content(default_mismatch))
    with pytest.raises(ValueError, match="bound to the plan contract"):
        prepare.validate_plan(default_mismatch, verify_sources=False)


def test_shortcap_is_bound_to_explicit_eight_image_qualification() -> None:
    assert prepare._sampling_max_new_tokens(
        arm="rloo", image_count=8, qualification_max_new_tokens=32
    ) == 32
    with pytest.raises(ValueError, match="exactly 8 images"):
        prepare._sampling_max_new_tokens(
            arm="rloo", image_count=256, qualification_max_new_tokens=32
        )
    with pytest.raises(ValueError, match="only valid for RLOO"):
        prepare._sampling_max_new_tokens(
            arm="ce", image_count=8, qualification_max_new_tokens=32
        )
    with pytest.raises(ValueError, match="1..3083"):
        prepare._sampling_max_new_tokens(
            arm="rloo", image_count=8, qualification_max_new_tokens=prepare.PRODUCTION_MAX_NEW_TOKENS
        )


def test_shortcap_rollout_metadata_and_length_stop_are_checked(tmp_path: Path) -> None:
    config = tmp_path / "infer.yaml"
    config.write_text("x", encoding="utf-8")
    artifact = _rollout_artifact(_first_raw())
    artifact["config"].update({
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 32,
        "seeds": list(prepare._expected_seeds(1)),
        "model_dtype": "fp32",
        "resolved_fingerprint": "config-id",
        "infer_config_path": str(config),
    })
    artifact["model_identity"] = {
        "effective_settings": {
            "observed_attn_implementation": "sdpa",
            "observed_model_dtype": {"parameter_dtype_names": ["torch.float32"]},
        },
        "model_identity": {
            "adapter": {
                "adapter_path": str(tmp_path / "current-adapter"),
                "merged_adapters": [],
                "adapter_state_evidence": {"state_checked": True},
            }
        },
    }
    for row in artifact["rollouts"]:
        if row["stop_reason"] == "length":
            row["generated_token_ids"] = list(range(32))
            row["generated_token_ids_sha256"] = prepare.json_sha256(list(range(32)))
    prepare._validate_rollout_policy(
        artifact,
        round_index=1,
        config_path=config.resolve(),
        config_fingerprint="config-id",
        current_adapter={"root": str(tmp_path / "current-adapter")},
        max_new_tokens=32,
    )
    artifact["rollouts"][1]["generated_token_ids"] = list(range(31))
    artifact["rollouts"][1]["generated_token_ids_sha256"] = prepare.json_sha256(list(range(31)))
    with pytest.raises(ValueError, match="did not reach"):
        prepare._validate_rollout_policy(
            artifact,
            round_index=1,
            config_path=config.resolve(),
            config_fingerprint="config-id",
            current_adapter={"root": str(tmp_path / "current-adapter")},
            max_new_tokens=32,
        )


def test_missing_or_corrupt_rollout_cells_fail_closed() -> None:
    raw = _first_raw()
    with pytest.raises(ValueError, match="missing or extra"):
        prepare._rloo_groups(
            [raw],
            [(Path("bank.json"), _rollout_artifact(raw, missing_last=True))],
            round_index=1,
        )
    artifact = _rollout_artifact(raw)
    artifact["rollouts"][0]["generated_token_ids_sha256"] = "wrong"
    with pytest.raises(ValueError, match="body hash"):
        prepare._rloo_groups([raw], [(Path("bank.json"), artifact)], round_index=1)


def test_rollout_policy_requires_raw_softmax_and_rejects_stale_adapter(tmp_path: Path) -> None:
    config = tmp_path / "infer.yaml"
    config.write_text("x", encoding="utf-8")
    artifact = {
        "schema_version": prepare.ROLLOUT_SCHEMA,
        "config": {
            "decode_mode": "sampled",
            "temperature": 1.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 3084,
            "seeds": list(prepare._expected_seeds(2)),
            "model_dtype": "fp32",
            "resolved_fingerprint": "config-id",
            "infer_config_path": str(config),
            "raw_softmax": True,
            "top_k": 0,
            "generation_config_source": "fresh_transformers_generation_config",
            "use_model_defaults": False,
        },
        "model_identity": {
            "effective_settings": {
                "observed_attn_implementation": "sdpa",
                "observed_model_dtype": {"parameter_dtype_names": ["torch.float32"]},
            },
            "model_identity": {
                "adapter": {
                    "adapter_path": str(tmp_path / "old-adapter"),
                    "merged_adapters": [],
                    "adapter_state_evidence": {"state_checked": True},
                }
            },
        },
        "rollouts": [{
            "image_id": "1",
            "seed": prepare._expected_seeds(2)[0],
            "generated_token_ids": [1],
            "generated_text": "x",
            "stop_reason": "im_end",
            "predictions": {"predictions": []},
        }],
    }
    with pytest.raises(ValueError, match="stale"):
        prepare._validate_rollout_policy(
            artifact,
            round_index=2,
            config_path=config.resolve(),
            config_fingerprint="config-id",
            current_adapter={"root": str(tmp_path / "current-adapter")},
        )

    artifact["model_identity"]["model_identity"]["adapter"]["adapter_path"] = str(
        tmp_path / "current-adapter"
    )
    prepare._validate_rollout_policy(
        artifact,
        round_index=2,
        config_path=config.resolve(),
        config_fingerprint="config-id",
        current_adapter={"root": str(tmp_path / "current-adapter")},
    )

    artifact["config"].pop("raw_softmax")
    with pytest.raises(ValueError, match="explicit raw-softmax"):
        prepare._validate_rollout_policy(
            artifact,
            round_index=2,
            config_path=config.resolve(),
            config_fingerprint="config-id",
            current_adapter={"root": str(tmp_path / "current-adapter")},
        )


def test_native_ce_target_is_exact_prompt_suffix_through_eos() -> None:
    from src.config.fingerprint import sha256_json
    from src.inference.runtime import assemble_frontend

    resolved = load_research_infer_config(prepare.SOURCE_INFER_CONFIG)
    frontend = assemble_frontend(
        resolved.config,
        generation_config_fingerprint=sha256_json(
            resolved.config.generation.model_dump(mode="json")
        ),
    )
    group = prepare._native_ce_group(
        _first_raw(), index=0, config=resolved.config, frontend=frontend
    )
    action = group["actions"][0]
    assert action["action_token_ids"][-1] == prepare.EOS_TOKEN_ID
    assert action["native_ignored_post_eos_token_count"] == 1
    assert prepare.json_sha256(group["prompt_token_ids"]) == group["prompt_token_ids_sha256"]


def test_reused_native_ce_plan_rejects_identity_eos_and_span_corruption() -> None:
    from src.config.fingerprint import sha256_json
    from src.inference.runtime import assemble_frontend
    from src.inference.inputs import plan_examples

    config = load_research_infer_config(prepare.SOURCE_INFER_CONFIG).config
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    raw = _first_raw()
    planned, = plan_examples([raw], config=config, components=frontend.qwen, target_max_length=12000)
    target = planned.target
    assert target is not None
    group = prepare._native_ce_group_from_plan(raw, index=0, planned=planned)
    assert group == prepare._native_ce_group(raw, index=0, config=config, frontend=frontend)
    bad_eos = list(target.input_ids)
    bad_eos[target.supervised_token_spans[-1].physical_token_end - 1] = 7
    wrong_prefix = list(target.input_ids)
    wrong_prefix[0] += 1
    bad_spans = (replace(target.supervised_token_spans[0], physical_token_end=target.supervised_token_spans[0].physical_token_end - 1),
                 *target.supervised_token_spans[1:])
    for changed, message in (
        (replace(planned, target=None), "missing its annotated target"),
        (replace(planned, prompt=replace(planned.prompt, row_index=1)), "planned row identity"),
        (replace(planned, target=replace(target, input_ids=tuple(bad_eos))), "does not end"),
        (replace(planned, target=replace(target, input_ids=tuple(wrong_prefix))), "prefix differs"),
        (replace(planned, target=replace(target, supervised_token_spans=bad_spans)), "not contiguous"),
        (replace(planned, target=replace(target, ignored_token_spans=(replace(target.ignored_token_spans[0], physical_token_start=0),))), "post-EOS"),
    ):
        with pytest.raises(ValueError, match=message):
            prepare._native_ce_group_from_plan(raw, index=0, planned=changed)


def test_strict_preflight_reuses_plans_and_keeps_source_gate(monkeypatch):
    from probes.dora_owner_learning import preflight
    from src.inference import inputs

    counts = {"render": 0, "image_plan": 0, "materialize": 0}

    def count(key, operation):
        def call(*args, **kwargs):
            counts[key] += 1
            return operation(*args, **kwargs)
        return call

    monkeypatch.setattr(inputs, "render_example", count("render", inputs.render_example))
    monkeypatch.setattr(inputs, "plan_qwen_image", count("image_plan", inputs.plan_qwen_image))
    monkeypatch.setattr(preflight, "prepare_native_inputs", count("materialize", preflight.prepare_native_inputs))
    result = preflight.preflight(rows=2)
    assert counts == {"render": 2, "image_plan": 2, "materialize": 2}
    assert result["model_weights_loaded"] is False and result["population_count"] == 256
    assert all(row["ends_at_eos"] for row in result["rows"])
    monkeypatch.setattr(preflight, "TRAIN256_SHA256", "wrong source")
    with pytest.raises(ValueError, match="input identity"):
        preflight.preflight(rows=2)
    assert counts == {"render": 2, "image_plan": 2, "materialize": 2}


def test_ddp_objective_scaling_matches_global_ce_and_rloo_formulas() -> None:
    assert train._objective_scale("ce", image_count=256) == pytest.approx(8 / 256)
    token_logprobs = torch.tensor([2.0, 4.0], requires_grad=True)
    ce = train._action_loss(token_logprobs, arm="ce", advantage=None, image_count=256)
    ce.backward()
    assert token_logprobs.grad.tolist() == pytest.approx([-8 / 256 / 2] * 2)

    token_logprobs = torch.tensor([2.0, 4.0], requires_grad=True)
    rloo = train._action_loss(token_logprobs, arm="rloo", advantage=3.0, image_count=256)
    rloo.backward()
    assert token_logprobs.grad.tolist() == pytest.approx([-3 * 8 / 256 / 4] * 2)


def _minimal_ce_plan() -> dict[str, object]:
    groups = []
    for index in range(8):
        prompt = [100 + index]
        action = [200 + index, prepare.EOS_TOKEN_ID]
        groups.append({
            "image_id": str(index),
            "example_id": f"row-{index}",
            "source_order": index,
            "rank": None,
            "prompt_token_ids": prompt,
            "prompt_token_ids_sha256": prepare.json_sha256(prompt),
            "actions": [{
                "action_token_ids": action,
                "action_token_ids_sha256": prepare.json_sha256(action),
                "action_token_count": len(action),
                "terminal_eos_included": True,
            }],
        })
    assignments = prepare._assign(groups)
    plan = {
        "schema_version": prepare.PLAN_SCHEMA,
        "arm": "ce",
        "round": 1,
        "sources": {},
        "model": {},
        "lineage": None,
        "population": {
            "mode": "qualification",
            "image_count": 8,
            "world_size": 8,
            "images_per_rank": 1,
            "k": 4,
            "actions_per_image": 1,
            "groups": groups,
            "assignments": assignments,
        },
        "objective": {},
        "sampling": None,
        "unsupported": [],
    }
    plan["content_sha256"] = prepare.json_sha256(plan)
    return plan


def test_plan_serialization_readback_preserves_all_qualification_actions(tmp_path: Path) -> None:
    plan = _minimal_ce_plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    observed = prepare.validate_plan(path, verify_sources=False)
    assert observed == plan
    assert observed["population"]["image_count"] == 8
    assert sum(len(group["actions"]) for group in observed["population"]["groups"]) == 8


def test_round2_missing_or_wrong_optimizer_state_fails_before_load(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    named = (("adapter.weight", parameter),)
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=train.LEARNING_RATE,
        betas=train.ADAMW_BETAS,
        eps=train.ADAMW_EPSILON,
        weight_decay=0.0,
        foreach=False,
    )
    plan = {
        "round": 2,
        "arm": "ce",
        "lineage": {"optimizer_state": {"path": str(tmp_path / "missing.pt"), "sha256": "none"}},
        "model": {
            "current_adapter": {"fingerprint": "adapter"},
            "source_embedding": {"fingerprint": "embedding"},
        },
    }
    with pytest.raises(FileNotFoundError):
        train._load_previous_optimizer(plan, named=named, optimizer=optimizer)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    layout = train._parameter_layout(named)
    document = {
        "schema_version": prepare.OPTIMIZER_SCHEMA,
        "arm": "ce",
        "round": 1,
        "optimizer_step": 1,
        "parameter_layout": layout,
        "parameter_layout_sha256": prepare.json_sha256(layout),
        "saved_adapter_fingerprint": "wrong-adapter",
        "source_embedding_fingerprint": "embedding",
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with pytest.raises(ValueError, match="binding changed"):
        train._validate_optimizer_state_document(
            document,
            arm="ce",
            previous_round=1,
            layout=layout,
            current_adapter_fingerprint="adapter",
            source_embedding_fingerprint="embedding",
        )
    document["saved_adapter_fingerprint"] = "adapter"
    path = tmp_path / "optimizer.pt"
    torch.save(document, path)
    roundtrip = torch.load(path, map_location="cpu", weights_only=True)
    assert train._validate_optimizer_state_document(
        roundtrip,
        arm="ce",
        previous_round=1,
        layout=layout,
        current_adapter_fingerprint="adapter",
        source_embedding_fingerprint="embedding",
    )["optimizer_step"] == 1
