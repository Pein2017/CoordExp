from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import src.analysis.autoreg_fn_rescue_continuation as fn_rescue_module
from src.analysis.autoreg_fn_rescue_continuation import (
    BOX_START_TOKEN,
    FN_RESCUE_MERGE_JSONL_FILES,
    FN_RESCUE_STAGES,
    FnRescueConfig,
    FnRescueExecutionConfig,
    FnRescuePaths,
    FnRescueSelectionConfig,
    build_rescue_region_membership,
    build_fn_rescue_dry_run_plan,
    build_rescue_assistant_prefix,
    canonical_target_gt_idx,
    check_chat_template_continuation_feasibility,
    choose_wrong_control_source,
    coord_token,
    decode_rescue_tail,
    fn_rescue_stratum_key,
    fn_rescue_shard_label,
    load_fn_rescue_config,
    materialize_fn_rescue_attention_replay_shard,
    materialize_fn_rescue_decode_shard,
    materialize_fn_rescue_feasibility_shard,
    materialize_fn_rescue_select_cases_shard,
    merge_fn_rescue_shards,
    normalize_fn_rescue_shard,
    parse_generated_bbox,
    prefix_text_from_raw_compact_predictions,
    read_jsonl,
    replay_rescue_attention_rows,
    render_rescue_hint_row_prefix,
    summarize_fn_rescue_artifacts,
    score_rescue_box,
    strip_generation_terminal,
    write_fn_rescue_gallery,
    write_fn_rescue_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(
    REPO_ROOT
    / "configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked_smoke.yaml"
)
RUNNER_PATH = REPO_ROOT / "scripts/analysis/run_autoreg_fn_rescue_continuation.py"
LAUNCHER_PATH = REPO_ROOT / "scripts/analysis/launch_autoreg_fn_rescue_continuation_tmux.sh"
RAW_BUS_BOX_TOKENS = (
    "<|coord_457|>",
    "<|coord_509|>",
    "<|coord_546|>",
    "<|coord_747|>",
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self._token_to_id = {"<pad>": 0}
        self._id_to_token = {0: "<pad>"}
        self._next_id = 1

    def _ensure(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1
        return self._token_to_id[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens: list[int] = []
        index = 0
        while index < len(text):
            if text[index : index + 2] == "<|":
                end = text.find("|>", index)
                if end >= 0:
                    piece = text[index : end + 2]
                    tokens.append(self._ensure(piece))
                    index = end + 2
                    continue
            tokens.append(self._ensure(text[index]))
            index += 1
        return tokens

    def decode(
        self, token_ids: int | list[int] | tuple[int, ...], skip_special_tokens: bool = False
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        pieces: list[str] = []
        for token_id in token_ids:
            token = self._id_to_token[int(token_id)]
            if skip_special_tokens and (token == "<pad>" or token.startswith("<|")):
                continue
            pieces.append(token)
        return "".join(pieces)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ensure(token)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._id_to_token[int(token_id)]


class FakeProcessor:
    def __init__(self, tokenizer: FakeTokenizer, *, append_terminal: bool = False) -> None:
        self.tokenizer = tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.image_processor = SimpleNamespace(merge_size=1)
        self.append_terminal = append_terminal

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        continue_final_message: bool = False,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        parts: list[str] = []
        for index, message in enumerate(messages):
            role = str(message["role"])
            content = message["content"]
            if isinstance(content, str):
                text = content
            else:
                pieces: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type in {"image", "image_url"}:
                        pieces.append("<|image_pad|>")
                    elif item_type == "text":
                        pieces.append(str(item.get("text", "")))
                text = "".join(pieces)
            parts.append(f"<|im_start|>{role}\n{text}")
            is_last = index == len(messages) - 1
            if (not is_last) or (not continue_final_message) or self.append_terminal:
                parts.append("<|im_end|>")
        return "".join(parts)


class FakeGenerateModel:
    def __init__(self, generated_sequences: list[list[int]]) -> None:
        self.generated_sequences = generated_sequences
        self.last_generate_kwargs: dict[str, object] | None = None

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.last_generate_kwargs = kwargs
        return self.generated_sequences


class FakeReplayModel:
    def __init__(
        self,
        *,
        attentions: object = None,
        attn_implementation: str | None = "eager",
    ) -> None:
        config = (
            SimpleNamespace(_attn_implementation=attn_implementation)
            if attn_implementation is not None
            else SimpleNamespace()
        )
        self.config = config
        self.attentions = attentions
        self.last_call_kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.last_call_kwargs = kwargs
        return SimpleNamespace(attentions=self.attentions)


class FakeSequentialGenerateModel:
    def __init__(self, generated_sequences: list[list[int]]) -> None:
        self.generated_sequences = [list(sequence) for sequence in generated_sequences]
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_calls.append(dict(kwargs))
        if not self.generated_sequences:
            raise AssertionError("no generated sequence queued")
        return [self.generated_sequences.pop(0)]


class FakeDynamicReplayModel:
    def __init__(self, *, attn_implementation: str = "eager") -> None:
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)
        self.last_call_kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        torch = pytest.importorskip("torch")
        self.last_call_kwargs = dict(kwargs)
        input_ids = kwargs.get("input_ids")
        if isinstance(input_ids, list):
            seq_len = len(input_ids[0])
        else:
            seq_len = len(input_ids[0])  # type: ignore[index]
        attentions = (torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32),)
        return SimpleNamespace(attentions=attentions)


def _scored_row_with_objects(*objects: dict[str, object]) -> dict[str, object]:
    return {"raw_output_json": {"objects": list(objects)}}


def _rollout_row(
    *,
    raw_pred_idx: int | None,
    pred_desc: str,
    suppressed_by_guard: bool = False,
) -> dict[str, object]:
    return {
        "raw_pred_idx": raw_pred_idx,
        "pred_desc": pred_desc,
        "suppressed_by_guard": suppressed_by_guard,
        "pred_points": [10, 20, 30, 40],
    }


def _write_config_without_key(tmp_path: Path, section: str, key: str) -> Path:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    del payload[section][key]
    path = tmp_path / f"missing_{section}_{key}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_config_with_value(
    tmp_path: Path, section: str, key: str, value: object
) -> Path:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload[section][key] = value
    path = tmp_path / f"bad_{section}_{key}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tiny_config_payload(artifact_root: Path, paths: dict[str, Path]) -> dict[str, object]:
    return {
        "paths": {
            "artifact_root": str(artifact_root),
            "checkpoint": str(paths["checkpoint"]),
            "dataset_jsonl": str(paths["dataset_jsonl"]),
            "attention_atlas_root": str(paths["attention_atlas_root"]),
            "source_selected_cases": str(paths["source_selected_cases"]),
            "source_candidate_regions": str(paths["source_candidate_regions"]),
            "rollout_anatomy_per_row": str(paths["rollout_anatomy_per_row"]),
            "gt_vs_pred_scored": str(paths["gt_vs_pred_scored"]),
            "pred_token_trace": str(paths["pred_token_trace"]),
            "infer_resolved_config": str(paths["infer_resolved_config"]),
            "lane_c_study_config": None,
        },
        "selection": {
            "evidence_scope": "val200_attention_atlas_linked_fn_stratified",
            "sample_limit": 8,
            "per_stratum_cap": 8,
            "duplicate_iou_threshold": 0.95,
            "wrong_control_overlap_threshold": 0.05,
            "context_expansion_norm1000": 64,
        },
        "execution": {
            "attn_implementation": "eager",
            "torch_dtype": "float32",
            "decoding": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens_desc_only": 4,
            "max_new_tokens_desc_x1": 3,
            "gallery_per_bucket": 2,
        },
    }


def _write_tiny_fn_rescue_fixture(
    tmp_path: Path, *, candidate_target_desc: str = "vase", second_case: bool = False
) -> FnRescueConfig:
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"

    checkpoint = source_root / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    _write_text(checkpoint / "weights.bin", "stub")
    attention_atlas_root = source_root / "attention_atlas"
    attention_atlas_root.mkdir(parents=True, exist_ok=True)
    _write_text(attention_atlas_root / "atlas.txt", "stub")
    dataset_jsonl = _write_jsonl(
        source_root / "dataset.jsonl",
        [{"image": "images/val2017/000000000001.jpg"}],
    )
    infer_resolved_config = _write_text(
        source_root / "resolved_config.json",
        json.dumps({"processor_do_resize": False}, sort_keys=True),
    )

    selected_rows = [
        {
            "attention_case_family": "missed_gt_evidence_routing",
            "case_id": "case-0",
            "intended_target_gt_idx": 19,
            "prefix_depth": 1,
            "prefix_mode": "self_prefix",
            "prefix_quality": "clean_prefix",
            "selection_policy": "tiny",
            "selection_reason": "unit_test",
            "source_line_idx": 0,
            "target_desc": "vase",
            "x1_target_rank": 2,
            "x1_top_peak_attribution": "same_desc_competitor_gt_object",
        }
    ]
    if second_case:
        selected_rows.append(
            {
                "attention_case_family": "missed_gt_evidence_routing",
                "case_id": "case-1",
                "intended_target_gt_idx": 21,
                "prefix_depth": 0,
                "prefix_mode": "self_prefix",
                "prefix_quality": "empty_prefix",
                "selection_policy": "tiny",
                "selection_reason": "unit_test",
                "source_line_idx": 1,
                "target_desc": "lamp",
                "x1_target_rank": 99,
                "x1_top_peak_attribution": "no_local_object_diffuse",
            }
        )
    source_selected_cases = _write_jsonl(
        source_root / "selected_cases.jsonl", selected_rows
    )

    candidate_rows = [
        {
            "bbox_xyxy": [100, 100, 200, 200],
            "case_id": "case-0",
            "desc": candidate_target_desc,
            "gt_idx": 19,
            "prefix_depth": 1,
            "prefix_mode": "self_prefix",
            "prefix_quality": "clean_prefix",
            "region_kind": "target_gt",
            "source_line_idx": 0,
            "target_desc": "vase",
            "target_gt_idx": 19,
        },
        {
            "bbox_xyxy": [80, 80, 220, 220],
            "case_id": "case-0",
            "desc": candidate_target_desc,
            "exclude_bbox_xyxy": [100, 100, 200, 200],
            "gt_idx": 19,
            "prefix_depth": 1,
            "prefix_mode": "self_prefix",
            "prefix_quality": "clean_prefix",
            "region_kind": "context_ring",
            "source_line_idx": 0,
            "target_desc": "vase",
            "target_gt_idx": 19,
        },
        {
            "bbox_xyxy": [0, 0, 999, 999],
            "case_id": "case-0",
            "desc": None,
            "exclude_bbox_xyxy": [100, 100, 200, 200],
            "gt_idx": None,
            "prefix_depth": 1,
            "prefix_mode": "self_prefix",
            "prefix_quality": "clean_prefix",
            "region_kind": "far_background",
            "source_line_idx": 0,
            "target_desc": "vase",
            "target_gt_idx": 19,
        },
    ]
    if second_case:
        candidate_rows.extend(
            [
                {
                    "bbox_xyxy": [300, 300, 360, 380],
                    "case_id": "case-1",
                    "desc": "lamp",
                    "gt_idx": 21,
                    "prefix_depth": 0,
                    "prefix_mode": "self_prefix",
                    "prefix_quality": "empty_prefix",
                    "region_kind": "target_gt",
                    "source_line_idx": 1,
                    "target_desc": "lamp",
                    "target_gt_idx": 21,
                },
                {
                    "bbox_xyxy": [270, 270, 390, 410],
                    "case_id": "case-1",
                    "desc": "lamp",
                    "exclude_bbox_xyxy": [300, 300, 360, 380],
                    "gt_idx": 21,
                    "prefix_depth": 0,
                    "prefix_mode": "self_prefix",
                    "prefix_quality": "empty_prefix",
                    "region_kind": "context_ring",
                    "source_line_idx": 1,
                    "target_desc": "lamp",
                    "target_gt_idx": 21,
                },
                {
                    "bbox_xyxy": [0, 0, 999, 999],
                    "case_id": "case-1",
                    "desc": None,
                    "exclude_bbox_xyxy": [300, 300, 360, 380],
                    "gt_idx": None,
                    "prefix_depth": 0,
                    "prefix_mode": "self_prefix",
                    "prefix_quality": "empty_prefix",
                    "region_kind": "far_background",
                    "source_line_idx": 1,
                    "target_desc": "lamp",
                    "target_gt_idx": 21,
                },
            ]
        )
    source_candidate_regions = _write_jsonl(
        source_root / "candidate_regions.jsonl", candidate_rows
    )

    rollout_rows = [
        {
            "source_line_idx": 0,
            "dataset_gt_count": 7,
            "raw_pred_idx": 0,
            "guarded_pred_idx": 0,
            "pred_desc": "bench",
            "pred_points": [10, 10, 40, 40],
            "matched_gt_idx": 3,
            "guarded_matched_gt_idx": 3,
            "suppressed_by_guard": False,
        },
        {
            "source_line_idx": 0,
            "dataset_gt_count": 7,
            "raw_pred_idx": 1,
            "guarded_pred_idx": 1,
            "pred_desc": "vase",
            "pred_points": [100, 100, 200, 200],
            "matched_gt_idx": None,
            "guarded_matched_gt_idx": None,
            "suppressed_by_guard": False,
        },
    ]
    if second_case:
        rollout_rows.append(
            {
                "source_line_idx": 1,
                "dataset_gt_count": 4,
                "raw_pred_idx": 0,
                "guarded_pred_idx": 0,
                "pred_desc": "lamp",
                "pred_points": [300, 300, 360, 380],
                "matched_gt_idx": None,
                "guarded_matched_gt_idx": None,
                "suppressed_by_guard": False,
            }
        )
    rollout_anatomy_per_row = _write_jsonl(
        source_root / "rollout_anatomy.jsonl", rollout_rows
    )

    scored_rows = [
        {
            "image": "images/val2017/000000000001.jpg",
            "width": 1000,
            "height": 1000,
            "gt": [
                {"desc": "vase", "points": [100, 100, 200, 200]},
                {"desc": "block1", "points": [0, 0, 100, 100]},
                {"desc": "block2", "points": [899, 0, 999, 100]},
                {"desc": "block3", "points": [0, 899, 100, 999]},
                {"desc": "block4", "points": [899, 899, 999, 999]},
                {"desc": "block5", "points": [0, 449, 100, 549]},
                {"desc": "block6", "points": [899, 449, 999, 549]},
            ],
            "pred": [
                {"type": "bbox_2d", "points": [10, 10, 40, 40], "desc": "bench"},
                {"type": "bbox_2d", "points": [100, 100, 200, 200], "desc": "vase"},
            ],
            "raw_output_json": {
                "objects": [
                    {
                        "desc": "bench",
                        "bbox_2d": [
                            "<|coord_10|>",
                            "<|coord_10|>",
                            "<|coord_40|>",
                            "<|coord_40|>",
                        ],
                    },
                    {
                        "desc": "vase",
                        "bbox_2d": [
                            "<|coord_100|>",
                            "<|coord_100|>",
                            "<|coord_200|>",
                            "<|coord_200|>",
                        ],
                    },
                ]
            },
        }
    ]
    if second_case:
        scored_rows.append(
            {
                "image": "images/val2017/000000000002.jpg",
                "width": 1000,
                "height": 1000,
                "gt": [
                    {"desc": "lamp", "points": [300, 300, 360, 380]},
                ],
                "pred": [
                    {"type": "bbox_2d", "points": [300, 300, 360, 380], "desc": "lamp"},
                ],
                "raw_output_json": {
                    "objects": [
                        {
                            "desc": "lamp",
                            "bbox_2d": [
                                "<|coord_300|>",
                                "<|coord_300|>",
                                "<|coord_360|>",
                                "<|coord_380|>",
                            ],
                        }
                    ]
                },
            }
        )
    gt_vs_pred_scored = _write_jsonl(source_root / "gt_vs_pred_scored.jsonl", scored_rows)

    token_trace_rows = [
        {
            "line_idx": 0,
            "generated_token_text": [
                "<|object_ref_start|>bench<|box_start|>",
                "<|coord_10|>",
                "<|coord_10|>",
                "<|coord_40|>",
                "<|coord_40|>",
                "<|im_end|>",
            ],
        }
    ]
    if second_case:
        token_trace_rows.append(
            {
                "line_idx": 1,
                "generated_token_text": [
                    "<|object_ref_start|>lamp<|box_start|>",
                    "<|coord_300|>",
                    "<|coord_300|>",
                    "<|coord_360|>",
                    "<|coord_380|>",
                    "<|im_end|>",
                ],
            }
        )
    pred_token_trace = _write_jsonl(source_root / "pred_token_trace.jsonl", token_trace_rows)

    payload = _tiny_config_payload(
        artifact_root,
        {
            "checkpoint": checkpoint,
            "dataset_jsonl": dataset_jsonl,
            "attention_atlas_root": attention_atlas_root,
            "source_selected_cases": source_selected_cases,
            "source_candidate_regions": source_candidate_regions,
            "rollout_anatomy_per_row": rollout_anatomy_per_row,
            "gt_vs_pred_scored": gt_vs_pred_scored,
            "pred_token_trace": pred_token_trace,
            "infer_resolved_config": infer_resolved_config,
        },
    )
    config_path = tmp_path / "tiny_fn_rescue.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return load_fn_rescue_config(config_path)


def _load_script_module(script_path: Path, module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_launcher_config(
    tmp_path: Path,
    *,
    artifact_root: Path | None = None,
    attention_atlas_root: Path | None = None,
) -> Path:
    fixture_root = tmp_path / "fixture"
    config = _write_tiny_fn_rescue_fixture(fixture_root)
    base_config_path = fixture_root / "tiny_fn_rescue.yaml"
    payload = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if artifact_root is not None:
        payload["paths"]["artifact_root"] = str(artifact_root)
    if attention_atlas_root is not None:
        payload["paths"]["attention_atlas_root"] = str(attention_atlas_root)
    config_path = tmp_path / "launcher.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return config_path


def _fake_processor_inputs_for_prefix(
    tokenizer: FakeTokenizer,
    processor: FakeProcessor,
    assistant_prefix_text: str,
    *,
    messages: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if messages is None:
        prompt_text = assistant_prefix_text
    else:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    prefix_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return {
        "input_ids": [prefix_ids],
        "image_grid_thw": [[1, 1, 1]],
        "image_token_id": processor.image_token_id,
    }


def test_feasibility_probe_allows_multimodal_image_token_expansion() -> None:
    tokenizer = FakeTokenizer()
    processor = FakeProcessor(tokenizer)
    assistant_prefix = render_rescue_hint_row_prefix("vase", tier="desc_only", x1=None)
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "detect"}]},
        {"role": "assistant", "content": assistant_prefix},
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    prompt_input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    image_token_id = processor.image_token_id
    image_index = prompt_input_ids.index(image_token_id)
    expanded_prompt_ids = [
        *prompt_input_ids[:image_index],
        image_token_id,
        image_token_id,
        *prompt_input_ids[image_index + 1 :],
    ]
    handle = SimpleNamespace(
        tokenizer=tokenizer,
        processor=processor,
        model=FakeDynamicReplayModel(),
    )

    probe, replay_result = fn_rescue_module._run_fn_rescue_attention_probe(
        model_handle=handle,
        processor_inputs={
            "input_ids": [expanded_prompt_ids],
            "image_grid_thw": [[1, 1, 2]],
            "image_token_id": image_token_id,
        },
        tier="desc_only",
        stored_assistant_prefix_token_ids=tokenizer.encode(
            assistant_prefix, add_special_tokens=False
        ),
        region_rows=[
            {
                "region_kind": "target_gt",
                "region_instance_id": "gt:0",
                "bbox_xyxy": [0, 0, 999, 999],
            }
        ],
        prompt_input_ids=prompt_input_ids,
    )

    assert probe["prompt_input_ids_aligned"] is True
    assert probe["replay_prefix_token_match"] is True
    assert int(probe["visual_span_end"]) - int(probe["visual_span_start"]) == 2
    assert replay_result.query_role == "pre_x1"


def test_import_autoreg_fn_rescue_continuation_is_clean() -> None:
    env = {**os.environ, "PYTHONPATH": "/data/CoordExp"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.analysis.autoreg_fn_rescue_continuation; print('imported')",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stdout == "imported\n"
    assert result.stderr == ""


def test_coord_token_renders_compact_full_coord_marker() -> None:
    assert coord_token(0) == "<|coord_0|>"
    assert coord_token(999) == "<|coord_999|>"


def test_coord_token_rejects_out_of_range_values() -> None:
    for bad in [-1, 1000, 1200]:
        with pytest.raises(ValueError):
            coord_token(bad)


def test_render_desc_only_hint_row_prefix_uses_compact_markers() -> None:
    assert (
        render_rescue_hint_row_prefix("vase", tier="desc_only", x1=None)
        == "<|object_ref_start|>vase<|box_start|>"
    )


def test_render_desc_x1_hint_row_prefix_appends_x1_coord_token() -> None:
    assert (
        render_rescue_hint_row_prefix("person", tier="desc_x1", x1=7)
        == "<|object_ref_start|>person<|box_start|><|coord_7|>"
    )


@pytest.mark.parametrize(
    ("tier", "x1", "expected_token"),
    [
        ("desc_x1", 1003, "<|coord_999|>"),
        ("desc_x1_wrong_control", 1127, "<|coord_999|>"),
        ("desc_x1_wrong_control", -8, "<|coord_0|>"),
    ],
)
def test_render_x1_hint_row_prefix_clamps_to_norm1000_token_range(
    tier: str, x1: int, expected_token: str
) -> None:
    assert render_rescue_hint_row_prefix("cup", tier=tier, x1=x1).endswith(
        expected_token
    )


def test_hint_x1_record_preserves_raw_and_used_clamped_values() -> None:
    assert fn_rescue_module._hint_x1_record(None) == {
        "hint_x1": None,
        "hint_x1_raw": None,
        "hint_x1_used": None,
        "hint_x1_clamped": False,
    }
    assert fn_rescue_module._hint_x1_record(1003) == {
        "hint_x1": 999,
        "hint_x1_raw": 1003,
        "hint_x1_used": 999,
        "hint_x1_clamped": True,
    }
    assert fn_rescue_module._hint_x1_record(-8) == {
        "hint_x1": 0,
        "hint_x1_raw": -8,
        "hint_x1_used": 0,
        "hint_x1_clamped": True,
    }


def test_render_hint_prefix_has_no_newline_or_im_end_suffix() -> None:
    for tier, x1 in [("desc_only", None), ("desc_x1", 7)]:
        hint = render_rescue_hint_row_prefix("person", tier=tier, x1=x1)
        assert "\n" not in hint
        assert "\r" not in hint
        assert not hint.endswith("<|im_end|>")


def test_render_hint_rejects_compact_forbidden_desc_tokens() -> None:
    bad_descs = [
        "bad\nrow",
        "bad\rrow",
        "bad\trow",
        "<|im_end|>",
        "<|coord_7|>",
        "<|im_start|>",
    ]
    for bad in bad_descs:
        with pytest.raises(ValueError):
            render_rescue_hint_row_prefix(bad, tier="desc_only", x1=None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("abc<|im_end|>", "abc"),
        ("abc<|endoftext|> \n\t", "abc"),
        ("abc<pad><|im_end|>\n", "abc"),
        ("keep<|im_end|>middle", "keep<|im_end|>middle"),
    ],
)
def test_strip_generation_terminal_removes_only_terminal_suffixes(
    text: str, expected: str
) -> None:
    assert strip_generation_terminal(text) == expected


def test_prefix_text_from_raw_compact_predictions_returns_empty_for_depth_zero() -> None:
    prefix = prefix_text_from_raw_compact_predictions(
        rollout_rows=[_rollout_row(raw_pred_idx=0, pred_desc="bus")],
        scored_row=_scored_row_with_objects(
            {
                "desc": "bus",
                "bbox_2d": list(RAW_BUS_BOX_TOKENS),
            }
        ),
        prefix_depth=0,
    )

    assert prefix == ""


def test_prefix_text_from_raw_compact_predictions_uses_raw_coord_tokens_not_pixel_points() -> None:
    prefix = prefix_text_from_raw_compact_predictions(
        rollout_rows=[_rollout_row(raw_pred_idx=0, pred_desc="bus")],
        scored_row=_scored_row_with_objects(
            {
                "desc": "bus",
                "bbox_2d": list(RAW_BUS_BOX_TOKENS),
                "pred_points": [1, 2, 3, 4],
            }
        ),
        prefix_depth=1,
    )

    assert prefix == "<|object_ref_start|>bus<|box_start|>" + "".join(RAW_BUS_BOX_TOKENS)
    assert "<|coord_10|>" not in prefix


def test_prefix_text_from_raw_compact_predictions_direct_concats_multiple_rows_in_raw_index_order() -> None:
    prefix = prefix_text_from_raw_compact_predictions(
        rollout_rows=[
            _rollout_row(raw_pred_idx=1, pred_desc="bench"),
            _rollout_row(raw_pred_idx=0, pred_desc="bus"),
        ],
        scored_row=_scored_row_with_objects(
            {
                "desc": "bus",
                "bbox_2d": list(RAW_BUS_BOX_TOKENS),
            },
            {
                "desc": "bench",
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
            },
        ),
        prefix_depth=2,
    )

    assert prefix == (
        "<|object_ref_start|>bus<|box_start|>" + "".join(RAW_BUS_BOX_TOKENS)
        + "<|object_ref_start|>bench<|box_start|><|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    assert "\n" not in prefix
    assert not prefix.endswith(" ")
    assert not prefix.endswith("<|im_end|>")


def test_prefix_text_from_raw_compact_predictions_excludes_suppressed_rows() -> None:
    prefix = prefix_text_from_raw_compact_predictions(
        rollout_rows=[
            _rollout_row(raw_pred_idx=0, pred_desc="bus"),
            _rollout_row(raw_pred_idx=1, pred_desc="bench", suppressed_by_guard=True),
        ],
        scored_row=_scored_row_with_objects(
            {
                "desc": "bus",
                "bbox_2d": list(RAW_BUS_BOX_TOKENS),
            },
            {
                "desc": "bench",
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
            },
        ),
        prefix_depth=2,
    )

    assert prefix == "<|object_ref_start|>bus<|box_start|>" + "".join(RAW_BUS_BOX_TOKENS)


def test_prefix_text_from_raw_compact_predictions_rejects_desc_mismatch() -> None:
    with pytest.raises(ValueError, match="desc mismatch"):
        prefix_text_from_raw_compact_predictions(
            rollout_rows=[_rollout_row(raw_pred_idx=0, pred_desc="truck")],
            scored_row=_scored_row_with_objects(
                {
                    "desc": "bus",
                    "bbox_2d": list(RAW_BUS_BOX_TOKENS),
                }
            ),
            prefix_depth=1,
        )


def test_prefix_text_from_raw_compact_predictions_rejects_missing_raw_output_instead_of_falling_back_to_pixels() -> None:
    with pytest.raises(ValueError, match="raw_output_json"):
        prefix_text_from_raw_compact_predictions(
            rollout_rows=[_rollout_row(raw_pred_idx=0, pred_desc="bus")],
            scored_row={"pred": [{"desc": "bus", "pred_points": [457, 509, 546, 747]}]},
            prefix_depth=1,
        )


def test_prefix_text_from_raw_compact_predictions_cross_checks_trace_success() -> None:
    prefix = prefix_text_from_raw_compact_predictions(
        rollout_rows=[
            _rollout_row(raw_pred_idx=0, pred_desc="bus"),
            _rollout_row(raw_pred_idx=1, pred_desc="bench"),
        ],
        scored_row=_scored_row_with_objects(
            {
                "desc": "bus",
                "bbox_2d": list(RAW_BUS_BOX_TOKENS),
            },
            {
                "desc": "bench",
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
            },
        ),
        token_trace_row={
            "generated_token_text": [
                "<|object_ref_start|>bus<|box_start|>",
                *RAW_BUS_BOX_TOKENS,
                "<|object_ref_start|>bench<|box_start|>",
                "<|coord_10|>",
                "<|coord_20|>",
                "<|coord_30|>",
                "<|coord_40|>",
                "<|im_end|>",
            ]
        },
        prefix_depth=2,
    )

    assert prefix.startswith("<|object_ref_start|>bus")


def test_prefix_text_from_raw_compact_predictions_rejects_trace_that_only_matches_after_newline_normalization() -> None:
    with pytest.raises(ValueError, match="generated_token_text"):
        prefix_text_from_raw_compact_predictions(
            rollout_rows=[
                _rollout_row(raw_pred_idx=0, pred_desc="bus"),
                _rollout_row(raw_pred_idx=1, pred_desc="bench"),
            ],
            scored_row=_scored_row_with_objects(
                {
                    "desc": "bus",
                    "bbox_2d": list(RAW_BUS_BOX_TOKENS),
                },
                {
                    "desc": "bench",
                    "bbox_2d": [
                        "<|coord_10|>",
                        "<|coord_20|>",
                        "<|coord_30|>",
                        "<|coord_40|>",
                    ],
                },
            ),
            token_trace_row={
                "generated_token_text": [
                    "<|object_ref_start|>bus<|box_start|>",
                    *RAW_BUS_BOX_TOKENS,
                    "\n",
                    "<|object_ref_start|>bench<|box_start|>",
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                    "<|im_end|>",
                ]
            },
            prefix_depth=2,
        )


def test_build_rescue_assistant_prefix_direct_concats_prefix_and_hint() -> None:
    assistant_prefix = build_rescue_assistant_prefix(
        prefix_text="<|object_ref_start|>bus<|box_start|>" + "".join(RAW_BUS_BOX_TOKENS) + "<|im_end|>",
        target_desc="bench",
        tier="desc_x1",
        hint_x1=10,
    )

    assert assistant_prefix == (
        "<|object_ref_start|>bus<|box_start|>" + "".join(RAW_BUS_BOX_TOKENS)
        + "<|object_ref_start|>bench<|box_start|><|coord_10|>"
    )
    assert "\n" not in assistant_prefix


def test_check_chat_template_continuation_feasibility_desc_only_success() -> None:
    tokenizer = FakeTokenizer()
    model_handle = SimpleNamespace(
        tokenizer=tokenizer,
        processor=FakeProcessor(tokenizer),
    )
    assistant_prefix = render_rescue_hint_row_prefix("bench", tier="desc_only", x1=None)
    messages = [
        {"role": "user", "content": "find bench"},
        {"role": "assistant", "content": assistant_prefix},
    ]

    result = check_chat_template_continuation_feasibility(
        model_handle,
        messages,
        assistant_prefix,
        tier="desc_only",
    )

    assert result.feasible is True
    assert result.status == "ok"
    assert result.last_token_text == BOX_START_TOKEN
    assert result.rendered_prompt.endswith(assistant_prefix)


def test_check_chat_template_continuation_feasibility_desc_x1_success() -> None:
    tokenizer = FakeTokenizer()
    model_handle = SimpleNamespace(
        tokenizer=tokenizer,
        processor=FakeProcessor(tokenizer),
    )
    assistant_prefix = render_rescue_hint_row_prefix("bench", tier="desc_x1", x1=10)
    messages = [
        {"role": "user", "content": "find bench"},
        {"role": "assistant", "content": assistant_prefix},
    ]

    result = check_chat_template_continuation_feasibility(
        model_handle,
        messages,
        assistant_prefix,
        tier="desc_x1",
        hint_x1=10,
    )

    assert result.feasible is True
    assert result.status == "ok"
    assert result.last_token_text == "<|coord_10|>"
    assert result.rendered_prompt.endswith(assistant_prefix)


def test_check_chat_template_continuation_feasibility_reports_terminalized_prompt() -> None:
    tokenizer = FakeTokenizer()
    model_handle = SimpleNamespace(
        tokenizer=tokenizer,
        processor=FakeProcessor(tokenizer, append_terminal=True),
    )
    assistant_prefix = render_rescue_hint_row_prefix("bench", tier="desc_only", x1=None)
    messages = [{"role": "assistant", "content": assistant_prefix}]

    result = check_chat_template_continuation_feasibility(
        model_handle,
        messages,
        assistant_prefix,
        tier="desc_only",
    )

    assert result.feasible is False
    assert result.status == "prompt_suffix_mismatch"
    assert any("assistant_prefix" in error for error in result.errors)


@pytest.mark.parametrize(
    ("tier", "assistant_prefix", "tail_text", "expected_max_new_tokens", "expected_box"),
    [
        (
            "desc_only",
            None,
            "<|coord_10|><|coord_20|><|coord_30|><|coord_40|><|im_end|>",
            4,
            (10, 20, 30, 40),
        ),
        (
            "desc_x1",
            "<|object_ref_start|>bench<|box_start|><|coord_10|>",
            "<|coord_20|><|coord_30|><|coord_40|><|im_end|>",
            3,
            (10, 20, 30, 40),
        ),
    ],
)
def test_decode_rescue_tail_uses_exact_generation_kwargs_and_parses_tail(
    tier: str,
    assistant_prefix: str | None,
    tail_text: str,
    expected_max_new_tokens: int,
    expected_box: tuple[int, int, int, int],
) -> None:
    tokenizer = FakeTokenizer()
    prompt_ids = tokenizer.encode("<|im_start|>assistant\nprefix")
    tail_ids = tokenizer.encode(tail_text)
    model_handle = SimpleNamespace(
        tokenizer=tokenizer,
        processor=FakeProcessor(tokenizer),
        model=FakeGenerateModel([prompt_ids + tail_ids]),
    )

    result = decode_rescue_tail(
        model_handle,
        {"input_ids": [prompt_ids], "attention_mask": [[1] * len(prompt_ids)]},
        tier=tier,
        assistant_prefix_token_ids=None
        if assistant_prefix is None
        else tokenizer.encode(assistant_prefix),
    )

    assert result.generated_tail_ids == tuple(tail_ids)
    assert result.generated_tail_text == tail_text
    assert result.parsed_generation.generated_box_xyxy == expected_box
    assert model_handle.model.last_generate_kwargs is not None
    assert model_handle.model.last_generate_kwargs["do_sample"] is False
    assert model_handle.model.last_generate_kwargs["num_beams"] == 1
    assert model_handle.model.last_generate_kwargs["max_new_tokens"] == expected_max_new_tokens
    assert model_handle.model.last_generate_kwargs["pad_token_id"] == tokenizer.pad_token_id
    assert (
        model_handle.model.last_generate_kwargs["eos_token_id"]
        == tokenizer.convert_tokens_to_ids("<|im_end|>")
    )
    assert "output_attentions" not in model_handle.model.last_generate_kwargs


def test_decode_rescue_tail_scrubs_contaminated_generate_return_kwargs() -> None:
    tokenizer = FakeTokenizer()
    prompt_ids = tokenizer.encode("<|im_start|>assistant\nprefix")
    tail_ids = tokenizer.encode("<|coord_10|><|coord_20|><|coord_30|><|coord_40|><|im_end|>")
    model_handle = SimpleNamespace(
        tokenizer=tokenizer,
        processor=FakeProcessor(tokenizer),
        model=FakeGenerateModel([prompt_ids + tail_ids]),
    )

    result = decode_rescue_tail(
        model_handle,
        {
            "input_ids": [prompt_ids],
            "attention_mask": [[1] * len(prompt_ids)],
            "output_attentions": True,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
            "output_scores": True,
        },
        tier="desc_only",
    )

    assert result.parsed_generation.generated_box_xyxy == (10, 20, 30, 40)
    assert model_handle.model.last_generate_kwargs is not None
    assert "output_attentions" not in model_handle.model.last_generate_kwargs
    assert "return_dict_in_generate" not in model_handle.model.last_generate_kwargs
    assert "output_hidden_states" not in model_handle.model.last_generate_kwargs
    assert "output_scores" not in model_handle.model.last_generate_kwargs


def test_generation_kwargs_artifact_summary_is_json_safe() -> None:
    torch = pytest.importorskip("torch")

    summary = fn_rescue_module._generation_kwargs_artifact_summary(
        {
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "image_grid_thw": torch.zeros((1, 3), dtype=torch.long),
            "pixel_values": torch.zeros((1, 2, 4), dtype=torch.float32),
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 4,
            "pad_token_id": 0,
            "eos_token_id": 9,
            "output_attentions": True,
            "return_dict_in_generate": True,
        }
    )

    json.dumps(summary, sort_keys=True)
    assert summary == {
        "do_sample": False,
        "eos_token_id": 9,
        "image_grid_thw_shape": [1, 3],
        "input_ids_shape": [1, 3],
        "max_new_tokens": 4,
        "num_beams": 1,
        "pad_token_id": 0,
        "pixel_values_shape": [1, 2, 4],
    }
    assert "input_ids" not in summary
    assert "image_grid_thw" not in summary
    assert "pixel_values" not in summary
    assert "output_attentions" not in summary
    assert "return_dict_in_generate" not in summary


def test_replay_rescue_attention_rows_fails_fast_on_prefix_token_mismatch() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    processor = FakeProcessor(tokenizer)
    model_handle = SimpleNamespace(tokenizer=tokenizer, processor=processor)
    input_ids = [[7, processor.image_token_id, 8, 9]]
    attentions = (torch.zeros((1, 1, 4, 4)),)

    with pytest.raises(ValueError, match="assistant prefix token ids"):
        replay_rescue_attention_rows(
            model_handle=model_handle,
            processor_inputs={
                "input_ids": input_ids,
                "image_grid_thw": [[1, 1, 1]],
            },
            tier="desc_only",
            stored_assistant_prefix_token_ids=[11, 12],
            region_rows=[],
            attention_tensors=attentions,
        )


def test_replay_rescue_attention_rows_returns_union_and_instance_rows() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    processor = FakeProcessor(tokenizer)
    model_handle = SimpleNamespace(tokenizer=tokenizer, processor=processor)
    last_token_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    input_ids = [[5, processor.image_token_id, processor.image_token_id, processor.image_token_id, last_token_id]]
    attentions = (
        torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.2, 0.3, 0.4, 0.1],
                    ]
                ]
            ],
            dtype=torch.float32,
        ),
    )

    result = replay_rescue_attention_rows(
        model_handle=model_handle,
        processor_inputs={
            "input_ids": input_ids,
            "image_grid_thw": [[1, 1, 3]],
        },
        tier="desc_only",
        stored_assistant_prefix_token_ids=[processor.image_token_id, last_token_id],
        region_rows=[
            {
                "region_kind": "left",
                "region_instance_id": "a",
                "bbox_xyxy": [0, 0, 400, 999],
            },
            {
                "region_kind": "right",
                "region_instance_id": "b",
                "bbox_xyxy": [300, 0, 999, 999],
            },
        ],
        attention_tensors=attentions,
    )

    assert result.status == "ok"
    assert result.query_role == "pre_x1"
    assert result.query_index == 4
    assert all(row["rescue_tier"] == "desc_only" for row in result.rows)
    masses = {
        (row["region_kind"], row["head"]): row["attention_mass"] for row in result.rows
    }
    assert masses[("instance:left:a", 0)] == pytest.approx(0.2)
    assert masses[("union:right", 0)] == pytest.approx(0.7)


def test_replay_rescue_attention_rows_desc_x1_uses_pre_y1_role() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    processor = FakeProcessor(tokenizer)
    model_handle = SimpleNamespace(tokenizer=tokenizer, processor=processor)
    last_token_id = tokenizer.convert_tokens_to_ids("<|coord_10|>")
    input_ids = [[5, processor.image_token_id, processor.image_token_id, processor.image_token_id, last_token_id]]
    attentions = (torch.zeros((1, 1, 5, 5), dtype=torch.float32),)

    result = replay_rescue_attention_rows(
        model_handle=model_handle,
        processor_inputs={
            "input_ids": input_ids,
            "image_grid_thw": [[1, 1, 3]],
        },
        tier="desc_x1",
        stored_assistant_prefix_token_ids=[processor.image_token_id, last_token_id],
        region_rows=[],
        attention_tensors=attentions,
    )

    assert result.status == "ok"
    assert result.query_role == "pre_y1"
    assert all(row["role"] == "pre_y1" for row in result.rows)


def test_replay_rescue_attention_rows_uses_last_non_pad_input_index() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    processor = FakeProcessor(tokenizer)
    model_handle = SimpleNamespace(tokenizer=tokenizer, processor=processor)
    last_token_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    input_ids = [[5, processor.image_token_id, processor.image_token_id, processor.image_token_id, last_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id]]
    attentions = (torch.zeros((1, 1, 5, 5), dtype=torch.float32),)

    result = replay_rescue_attention_rows(
        model_handle=model_handle,
        processor_inputs={
            "input_ids": input_ids,
            "image_grid_thw": [[1, 1, 3]],
        },
        tier="desc_only",
        stored_assistant_prefix_token_ids=[processor.image_token_id, last_token_id],
        region_rows=[],
        attention_tensors=attentions,
    )

    assert result.status == "ok"
    assert result.query_index == 4


def test_replay_rescue_attention_rows_fails_with_two_visual_spans() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    processor = FakeProcessor(tokenizer)
    model_handle = SimpleNamespace(tokenizer=tokenizer, processor=processor)
    last_token_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    input_ids = [[5, processor.image_token_id, processor.image_token_id, 7, processor.image_token_id, last_token_id]]
    attentions = (torch.zeros((1, 1, 6, 6), dtype=torch.float32),)

    with pytest.raises(ValueError, match="exactly 1 visual span.*found 2"):
        replay_rescue_attention_rows(
            model_handle=model_handle,
            processor_inputs={
                "input_ids": input_ids,
                "image_grid_thw": [[1, 1, 2]],
            },
            tier="desc_only",
            stored_assistant_prefix_token_ids=[processor.image_token_id, last_token_id],
            region_rows=[],
            attention_tensors=attentions,
        )


def test_replay_rescue_attention_rows_requires_eager_when_recomputing_attentions() -> None:
    tokenizer = FakeTokenizer()
    processor = FakeProcessor(tokenizer)
    model_handle = SimpleNamespace(
        tokenizer=tokenizer,
        processor=processor,
        model=FakeReplayModel(attentions=(), attn_implementation="sdpa"),
    )
    last_token_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    input_ids = [[5, processor.image_token_id, processor.image_token_id, processor.image_token_id, last_token_id]]

    with pytest.raises(ValueError, match="eager"):
        replay_rescue_attention_rows(
            model_handle=model_handle,
            processor_inputs={
                "input_ids": input_ids,
                "image_grid_thw": [[1, 1, 3]],
            },
            tier="desc_only",
            stored_assistant_prefix_token_ids=[processor.image_token_id, last_token_id],
            region_rows=[],
            attention_tensors=None,
        )


def test_replay_rescue_attention_rows_rejects_missing_attentions_output() -> None:
    tokenizer = FakeTokenizer()
    processor = FakeProcessor(tokenizer)
    replay_model = FakeReplayModel(attentions=None, attn_implementation="eager")
    model_handle = SimpleNamespace(
        tokenizer=tokenizer,
        processor=processor,
        model=replay_model,
    )
    last_token_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    input_ids = [[5, processor.image_token_id, processor.image_token_id, processor.image_token_id, last_token_id]]

    with pytest.raises(ValueError, match="attentions"):
        replay_rescue_attention_rows(
            model_handle=model_handle,
            processor_inputs={
                "input_ids": input_ids,
                "image_grid_thw": [[1, 1, 3]],
            },
            tier="desc_only",
            stored_assistant_prefix_token_ids=[processor.image_token_id, last_token_id],
            region_rows=[],
            attention_tensors=None,
        )

    assert replay_model.last_call_kwargs is not None
    assert replay_model.last_call_kwargs["output_attentions"] is True


def test_parse_desc_only_requires_four_leading_coord_tokens_and_ignores_trailing_text() -> None:
    parsed = parse_generated_bbox(
        tier="desc_only",
        generated_tail_text="<|coord_10|><|coord_20|><|coord_30|><|coord_40|>extra text",
        hint_x1=None,
    )

    assert parsed.valid_parse is True
    assert parsed.generated_box_xyxy == (10, 20, 30, 40)
    assert parsed.generated_coord_tokens == (
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    )


def test_parse_desc_only_rejects_missing_or_non_leading_coord_tokens() -> None:
    too_short = parse_generated_bbox(
        tier="desc_only",
        generated_tail_text="<|coord_10|><|coord_20|><|coord_30|>",
        hint_x1=None,
    )
    prefixed = parse_generated_bbox(
        tier="desc_only",
        generated_tail_text="oops <|coord_10|><|coord_20|><|coord_30|><|coord_40|>",
        hint_x1=None,
    )

    assert too_short.valid_parse is False
    assert too_short.generated_box_xyxy is None
    assert prefixed.valid_parse is False
    assert prefixed.generated_box_xyxy is None


def test_parse_desc_x1_prepends_hint_x1_and_requires_three_generated_coords() -> None:
    parsed = parse_generated_bbox(
        tier="desc_x1",
        generated_tail_text="<|coord_20|><|coord_30|><|coord_40|>",
        hint_x1=10,
    )

    assert parsed.valid_parse is True
    assert parsed.generated_box_xyxy == (10, 20, 30, 40)
    assert parsed.generated_coord_tokens == (
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    )


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("paths", "source_candidate_regions"),
        ("selection", "duplicate_iou_threshold"),
        ("execution", "torch_dtype"),
        ("execution", "do_sample"),
    ],
)
def test_load_fn_rescue_config_requires_required_keys(
    tmp_path: Path, section: str, key: str
) -> None:
    path = _write_config_without_key(tmp_path, section, key)

    with pytest.raises(ValueError, match=key):
        load_fn_rescue_config(path)


def test_load_fn_rescue_config_requires_literal_boolean_false_do_sample(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["execution"]["do_sample"] = "false"
    path = tmp_path / "string_do_sample.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do_sample"):
        load_fn_rescue_config(path)


def test_load_fn_rescue_config_validates_torch_dtype(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["execution"]["torch_dtype"] = "float128"
    path = tmp_path / "bad_dtype.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="torch_dtype"):
        load_fn_rescue_config(path)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("selection", "sample_limit", True),
        ("execution", "num_beams", True),
        ("selection", "duplicate_iou_threshold", "0.95"),
        ("execution", "max_new_tokens_desc_only", "4"),
    ],
)
def test_load_fn_rescue_config_requires_real_numeric_yaml_scalars(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    path = _write_config_with_value(tmp_path, section, key, value)

    with pytest.raises(ValueError, match=key):
        load_fn_rescue_config(path)


def test_load_fn_rescue_config_distinguishes_file_hash_and_directory_fingerprint() -> None:
    cfg = load_fn_rescue_config(CONFIG_PATH)

    selected_cases = cfg.source_artifacts["source_selected_cases"]
    assert selected_cases["hash_kind"] == "file_sha256"
    assert selected_cases["sha256"]
    assert "structural_fingerprint" not in selected_cases

    checkpoint = cfg.source_artifacts["checkpoint"]
    assert checkpoint["hash_kind"] == "directory_structural_fingerprint"
    assert checkpoint["structural_fingerprint"]
    assert "sha256" not in checkpoint


def test_parse_desc_x1_rejects_missing_generated_coord() -> None:
    parsed = parse_generated_bbox(
        tier="desc_x1",
        generated_tail_text="<|coord_20|><|coord_30|>",
        hint_x1=10,
    )

    assert parsed.valid_parse is False
    assert parsed.generated_box_xyxy is None


def test_parse_rejects_non_positive_area_box() -> None:
    parsed = parse_generated_bbox(
        tier="desc_only",
        generated_tail_text="<|coord_10|><|coord_20|><|coord_10|><|coord_40|>",
        hint_x1=None,
    )

    assert parsed.valid_parse is False
    assert parsed.generated_box_xyxy is None


def test_score_rescue_success_and_duplicate_flag_are_separate() -> None:
    score = score_rescue_box(
        generated_box=[100, 100, 200, 200],
        target_box=[100, 100, 200, 200],
        same_desc_rollout_boxes=[[100, 100, 200, 200]],
        duplicate_iou_threshold=0.95,
    )

    assert score.target_iou == pytest.approx(1.0)
    assert score.success_iou30 is True
    assert score.success_iou50 is True
    assert score.success_iou75 is True
    assert score.same_desc_duplicate_iou95 is True
    assert score.max_same_desc_existing_iou == pytest.approx(1.0)
    assert score.duplicate_source_bbox_xyxy == (100, 100, 200, 200)
    assert score.primary_rescue_success is False


def test_duplicate_threshold_is_strictly_greater_than_policy() -> None:
    exact = score_rescue_box(
        generated_box=[0, 0, 100, 100],
        target_box=[0, 0, 100, 100],
        same_desc_rollout_boxes=[[0, 0, 95, 100]],
        duplicate_iou_threshold=0.95,
    )
    over = score_rescue_box(
        generated_box=[0, 0, 100, 100],
        target_box=[0, 0, 100, 100],
        same_desc_rollout_boxes=[[0, 0, 96, 100]],
        duplicate_iou_threshold=0.95,
    )

    assert exact.max_same_desc_existing_iou == pytest.approx(0.95)
    assert exact.same_desc_duplicate_iou95 is False
    assert exact.primary_rescue_success is True
    assert over.max_same_desc_existing_iou == pytest.approx(0.96)
    assert over.same_desc_duplicate_iou95 is True
    assert over.primary_rescue_success is False


def test_wrong_control_prefers_same_desc_gt_over_rollout_prediction() -> None:
    source = choose_wrong_control_source(
        target_box=[100, 100, 200, 200],
        same_desc_gt_boxes=[[300, 100, 400, 200]],
        same_desc_rollout_boxes=[[500, 100, 600, 200]],
        all_gt_boxes=[[100, 100, 200, 200], [300, 100, 400, 200]],
        context_expansion_norm1000=64,
    )

    assert source is not None
    assert source.kind == "same_desc_competitor_gt_object"
    assert source.bbox_xyxy == (300, 100, 400, 200)
    assert source.x1 == 300


def test_wrong_control_uses_same_desc_rollout_prediction_after_gt_sources() -> None:
    source = choose_wrong_control_source(
        target_box=[100, 100, 200, 200],
        same_desc_gt_boxes=[],
        same_desc_rollout_boxes=[[500, 100, 600, 200]],
        all_gt_boxes=[[100, 100, 200, 200]],
        context_expansion_norm1000=64,
    )

    assert source is not None
    assert source.kind == "same_desc_rollout_prediction"
    assert source.bbox_xyxy == (500, 100, 600, 200)
    assert source.x1 == 500


def test_wrong_control_accepts_same_desc_candidate_below_duplicate_threshold() -> None:
    source = choose_wrong_control_source(
        target_box=[100, 100, 200, 200],
        same_desc_gt_boxes=[[100, 100, 200, 110]],
        same_desc_rollout_boxes=[],
        all_gt_boxes=[[100, 100, 200, 200]],
        context_expansion_norm1000=64,
        duplicate_iou_threshold=0.95,
        wrong_control_overlap_threshold=0.05,
    )

    assert source is not None
    assert source.kind == "same_desc_competitor_gt_object"
    assert source.bbox_xyxy == (100, 100, 200, 110)


def test_wrong_control_falls_back_when_same_desc_candidates_are_rejected() -> None:
    source = choose_wrong_control_source(
        target_box=[100, 100, 200, 200],
        same_desc_gt_boxes=[[100, 100, 200, 200]],
        same_desc_rollout_boxes=[],
        all_gt_boxes=[[100, 100, 200, 200]],
        context_expansion_norm1000=64,
        duplicate_iou_threshold=0.95,
    )

    assert source is not None
    assert source.kind == "far_background_concrete"
    assert source.bbox_xyxy != (0, 0, 999, 999)


def test_wrong_control_far_background_fallback_is_concrete_not_broad_placeholder() -> None:
    source = choose_wrong_control_source(
        target_box=[100, 100, 200, 200],
        same_desc_gt_boxes=[],
        same_desc_rollout_boxes=[],
        all_gt_boxes=[[100, 100, 200, 200]],
        context_expansion_norm1000=64,
    )

    assert source is not None
    assert source.kind == "far_background_concrete"
    assert source.bbox_xyxy != (0, 0, 999, 999)


def test_wrong_control_unavailable_preserves_rejected_far_background_candidates() -> None:
    source = choose_wrong_control_source(
        target_box=[100, 100, 200, 200],
        same_desc_gt_boxes=[],
        same_desc_rollout_boxes=[],
        all_gt_boxes=[
            [100, 100, 200, 200],
            [0, 0, 100, 100],
            [899, 0, 999, 100],
            [0, 899, 100, 999],
            [899, 899, 999, 999],
            [0, 449, 100, 549],
            [899, 449, 999, 549],
        ],
        context_expansion_norm1000=64,
        wrong_control_overlap_threshold=0.05,
    )

    assert source is not None
    assert source.kind == "wrong_control_unavailable"
    assert source.skip_reason == "wrong_control_unavailable"
    assert source.bbox_xyxy is None
    assert source.x1 is None
    assert source.source_index is None
    assert source.rejected_candidates
    assert all("label" in item for item in source.rejected_candidates)
    assert all(item.get("reasons") for item in source.rejected_candidates)
    with pytest.raises(TypeError):
        source.rejected_candidates[0]["label"] = "mutated"


def test_canonical_target_gt_idx_accepts_intended_key_from_atlas_ledger() -> None:
    row = {"intended_target_gt_idx": 19}

    assert canonical_target_gt_idx(row) == 19


def test_object_count_bucket_is_part_of_stratum_key() -> None:
    base = {
        "prefix_quality": "clean_prefix",
        "x1_top_peak_attribution": "no_local_object_diffuse",
        "x1_target_rank": 100,
        "prefix_depth": 2,
    }

    small = fn_rescue_stratum_key({**base, "dataset_gt_count": 3})
    crowded = fn_rescue_stratum_key({**base, "dataset_gt_count": 20})

    assert small != crowded


def test_same_kind_union_deduplicates_overlapping_region_tokens() -> None:
    rows = [
        {
            "region_kind": "same_desc_competitor_gt_object",
            "region_instance_id": "a",
            "token_indices": [1, 2, 3],
        },
        {
            "region_kind": "same_desc_competitor_gt_object",
            "region_instance_id": "b",
            "token_indices": [3, 4],
        },
    ]

    membership = build_rescue_region_membership(rows)

    assert membership["instance:same_desc_competitor_gt_object:a"] == [1, 2, 3]
    assert membership["instance:same_desc_competitor_gt_object:b"] == [3, 4]
    assert membership["union:same_desc_competitor_gt_object"] == [1, 2, 3, 4]


def test_split_rescue_attention_region_key_preserves_instance_ids_with_colons() -> None:
    assert fn_rescue_module._split_rescue_attention_region_key(
        "instance:target_gt:gt:11"
    ) == {
        "aggregation_scope": "instance",
        "region_kind": "target_gt",
        "region_instance_id": "gt:11",
    }
    assert fn_rescue_module._split_rescue_attention_region_key(
        "union:same_desc_competitor_gt_object"
    ) == {
        "aggregation_scope": "union",
        "region_kind": "same_desc_competitor_gt_object",
        "region_instance_id": "union:same_desc_competitor_gt_object",
    }
    with pytest.raises(ValueError, match="unsupported FN-rescue attention region key"):
        fn_rescue_module._split_rescue_attention_region_key("bad")


def test_shard_label_normalization_and_dry_run_plan_include_expected_labels(
    tmp_path: Path,
) -> None:
    config = _write_tiny_fn_rescue_fixture(tmp_path)

    assert fn_rescue_shard_label(1, 12) == "shard_001-of-012"
    assert normalize_fn_rescue_shard("shard_001-of-012") == {
        "shard_index": 1,
        "num_shards": 12,
        "shard_label": "shard_001-of-012",
    }
    assert normalize_fn_rescue_shard(0, 3)["shard_label"] == "shard_000-of-003"

    plan = build_fn_rescue_dry_run_plan(
        config,
        stages=["select_cases", "merge"],
        shard_index=1,
        num_shards=3,
    )

    assert plan["stages"] == ["select_cases", "merge"]
    assert plan["expected_shard_labels"] == [
        "shard_000-of-003",
        "shard_001-of-003",
        "shard_002-of-003",
    ]
    assert plan["shard"]["shard_label"] == "shard_001-of-003"
    assert plan["source_artifacts"]["source_selected_cases"]["hash_kind"] == "file_sha256"
    assert plan["source_artifacts"]["checkpoint"]["hash_kind"] == "directory_structural_fingerprint"
    assert plan["decode_settings"]["decoding"] == "greedy"
    assert plan["available_stages"] == list(FN_RESCUE_STAGES)


def test_materialize_select_cases_shard_emits_denominator_and_candidate_rows(
    tmp_path: Path,
) -> None:
    config = _write_tiny_fn_rescue_fixture(tmp_path)

    result = materialize_fn_rescue_select_cases_shard(config, shard_index=0, num_shards=1)
    shard_root = Path(result["shard_root"])

    rescue_rows = [
        row
        for row in (
            json.loads(line)
            for line in (shard_root / "rescue_rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    assert {(row["case_id"], row["planned_rescue_tier"]) for row in rescue_rows} == {
        ("case-0", "desc_only"),
        ("case-0", "desc_x1"),
        ("case-0", "desc_x1_wrong_control"),
    }
    wrong_control_row = next(
        row for row in rescue_rows if row["planned_rescue_tier"] == "desc_x1_wrong_control"
    )
    assert wrong_control_row["attempt_status"] == "skipped"
    assert wrong_control_row["skip_reason"] == "wrong_control_unavailable"

    candidate_rows = [
        json.loads(line)
        for line in (shard_root / "rescue_candidate_region_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert candidate_rows
    assert all(row["aggregation_scope"] == "instance" for row in candidate_rows)
    assert all(row["region_instance_id"] for row in candidate_rows)
    assert {row["region_kind"] for row in candidate_rows} >= {
        "target_gt",
        "context_ring",
        "far_background",
        "previous_generated_object",
        "same_desc_rollout_prediction",
    }

    wrong_control_rows = (
        shard_root / "wrong_control_rows.jsonl"
    ).read_text(encoding="utf-8")
    assert wrong_control_rows == ""
    assert json.loads((shard_root / "summary.json").read_text(encoding="utf-8"))[
        "row_counts"
    ]["rescue_rows"] == 3


def test_materialize_select_cases_shard_fails_fast_on_candidate_join_mismatch(
    tmp_path: Path,
) -> None:
    config = _write_tiny_fn_rescue_fixture(
        tmp_path, candidate_target_desc="not-the-target-desc"
    )

    with pytest.raises(ValueError, match="target_desc"):
        materialize_fn_rescue_select_cases_shard(config, shard_index=0, num_shards=1)


def _write_merge_shard(
    root: Path,
    shard_label: str,
    *,
    selected_rows: list[dict[str, object]],
    rescue_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]] | None = None,
    wrong_control_rows: list[dict[str, object]] | None = None,
    generation_rows: list[dict[str, object]] | None = None,
    replay_rows: list[dict[str, object]] | None = None,
    attention_rows: list[dict[str, object]] | None = None,
    decision_rows: list[dict[str, object]] | None = None,
) -> Path:
    shard_root = root / "shards" / shard_label
    shard_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = "cfg-hash"
    source_artifacts = {
        "source_selected_cases": {
            "path": "/tmp/source_selected_cases.jsonl",
            "exists": True,
            "byte_size": 10,
            "row_count": 1,
            "hash_kind": "file_sha256",
            "sha256": "selected-hash",
        },
        "checkpoint": {
            "path": "/tmp/checkpoint",
            "exists": True,
            "byte_size": 0,
            "row_count": None,
            "hash_kind": "directory_structural_fingerprint",
            "structural_fingerprint": "checkpoint-fingerprint",
        },
    }
    base_counts = {
        "selected_rescue_cases": len(selected_rows),
        "rescue_rows": len(rescue_rows),
        "rescue_candidate_region_rows": len(candidate_rows or []),
        "wrong_control_rows": len(wrong_control_rows or []),
        "rescue_generation_rows": len(generation_rows or []),
        "rescue_replay_prefix_rows": len(replay_rows or []),
        "rescue_attention_region_rows": len(attention_rows or []),
        "rescue_decision_context_rows": len(decision_rows or []),
    }
    summary = {
        "artifact_root": str(root),
        "shard_label": shard_label,
        "config_sha256": config_sha256,
        "evidence_scope": "val200_attention_atlas_linked_fn_stratified",
        "checkpoint": "/tmp/checkpoint",
        "source_artifacts": source_artifacts,
        "row_counts": base_counts,
    }
    _write_jsonl(shard_root / "selected_rescue_cases.jsonl", selected_rows)
    _write_jsonl(shard_root / "rescue_rows.jsonl", rescue_rows)
    _write_jsonl(
        shard_root / "rescue_candidate_region_rows.jsonl", candidate_rows or []
    )
    _write_jsonl(shard_root / "wrong_control_rows.jsonl", wrong_control_rows or [])
    _write_jsonl(
        shard_root / "rescue_generation_rows.jsonl", generation_rows or []
    )
    _write_jsonl(
        shard_root / "rescue_replay_prefix_rows.jsonl", replay_rows or []
    )
    _write_jsonl(
        shard_root / "rescue_attention_region_rows.jsonl", attention_rows or []
    )
    _write_jsonl(
        shard_root / "rescue_decision_context_rows.jsonl", decision_rows or []
    )
    _write_text(shard_root / "summary.json", json.dumps(summary, sort_keys=True))
    return shard_root


def _base_rescue_row(
    case_id: str, shard_label: str, tier: str, *, attempt_status: str = "emitted"
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "source_line_idx": 0 if case_id == "case-a" else 1,
        "target_gt_idx": 11 if case_id == "case-a" else 22,
        "target_desc": "vase" if case_id == "case-a" else "lamp",
        "prefix_depth": 1 if case_id == "case-a" else 0,
        "prefix_quality": "clean_prefix" if case_id == "case-a" else "empty_prefix",
        "binding_bucket": "same_desc_competitor",
        "depth_bucket": "d1_3" if case_id == "case-a" else "d0",
        "object_count_bucket": "gt_6_15",
        "planned_rescue_tier": tier,
        "attempt_status": attempt_status,
        "emitted": attempt_status == "emitted",
        "skip_reason": None if attempt_status == "emitted" else "wrong_control_unavailable",
        "prefix_reconstruction_status": "ok",
        "wrong_control_status": None,
        "source_artifacts": {"source_selected_cases": {"hash_kind": "file_sha256"}},
        "config_sha256": "cfg-hash",
        "shard_label": shard_label,
    }


def test_merge_fn_rescue_shards_rejects_missing_shard(tmp_path: Path) -> None:
    root = tmp_path / "merge_missing"
    _write_merge_shard(
        root,
        "shard_000-of-002",
        selected_rows=[{"case_id": "case-a"}],
        rescue_rows=[_base_rescue_row("case-a", "shard_000-of-002", "desc_only")],
    )

    with pytest.raises(FileNotFoundError, match="shard_001-of-002"):
        merge_fn_rescue_shards(root, ["shard_000-of-002", "shard_001-of-002"])


def test_merge_fn_rescue_shards_rejects_duplicate_rescue_keys(tmp_path: Path) -> None:
    root = tmp_path / "merge_duplicate"
    duplicate_row = _base_rescue_row("case-a", "shard_000-of-002", "desc_only")
    _write_merge_shard(
        root,
        "shard_000-of-002",
        selected_rows=[{"case_id": "case-a"}],
        rescue_rows=[duplicate_row],
    )
    _write_merge_shard(
        root,
        "shard_001-of-002",
        selected_rows=[{"case_id": "case-b"}],
        rescue_rows=[{**duplicate_row, "shard_label": "shard_001-of-002"}],
    )

    with pytest.raises(ValueError, match="duplicate key"):
        merge_fn_rescue_shards(root, ["shard_000-of-002", "shard_001-of-002"])


def test_merge_fn_rescue_shards_rejects_shard_summary_count_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "merge_bad_summary_counts"
    shard_root = _write_merge_shard(
        root,
        "shard_000-of-001",
        selected_rows=[{"case_id": "case-a"}],
        rescue_rows=[_base_rescue_row("case-a", "shard_000-of-001", "desc_only")],
        generation_rows=[
            {
                "case_id": "case-a",
                "rescue_tier": "desc_only",
                "valid_parse": True,
                "target_desc_preserved": True,
                "geometry_valid": True,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": False,
                "primary_rescue_success": True,
                "same_desc_duplicate_iou95": False,
                "duplicate_source_kind": None,
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            }
        ],
        replay_rows=[{"case_id": "case-a", "rescue_tier": "desc_only"}],
        decision_rows=[{"case_id": "case-a", "rescue_tier": "desc_only"}],
    )
    summary_path = shard_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["row_counts"]["rescue_generation_rows"] = 2
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="row_counts"):
        merge_fn_rescue_shards(root, ["shard_000-of-001"])


def test_merge_fn_rescue_shards_rejects_emitted_rows_without_decode_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "merge_missing_decode_outputs"
    _write_merge_shard(
        root,
        "shard_000-of-001",
        selected_rows=[{"case_id": "case-a"}],
        rescue_rows=[
            _base_rescue_row("case-a", "shard_000-of-001", "desc_only"),
            _base_rescue_row("case-a", "shard_000-of-001", "desc_x1"),
            _base_rescue_row(
                "case-a",
                "shard_000-of-001",
                "desc_x1_wrong_control",
                attempt_status="skipped",
            ),
        ],
        generation_rows=[],
        replay_rows=[],
        decision_rows=[],
    )

    with pytest.raises(ValueError, match="emitted decode attempts"):
        merge_fn_rescue_shards(root, ["shard_000-of-001"])


def test_merge_fn_rescue_shards_rejects_missing_planned_tier_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "merge_missing_planned_tier"
    _write_merge_shard(
        root,
        "shard_000-of-001",
        selected_rows=[
            {
                "case_id": "case-a",
                "source_line_idx": 0,
                "target_gt_idx": 11,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            }
        ],
        rescue_rows=[
            _base_rescue_row("case-a", "shard_000-of-001", "desc_only"),
            _base_rescue_row("case-a", "shard_000-of-001", "desc_x1"),
        ],
        generation_rows=[
            {
                "case_id": "case-a",
                "rescue_tier": "desc_only",
                "valid_parse": True,
                "target_desc_preserved": True,
                "geometry_valid": True,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": False,
                "primary_rescue_success": True,
                "same_desc_duplicate_iou95": False,
                "duplicate_source_kind": None,
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            },
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "valid_parse": True,
                "target_desc_preserved": True,
                "geometry_valid": True,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": False,
                "primary_rescue_success": True,
                "same_desc_duplicate_iou95": False,
                "duplicate_source_kind": None,
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            },
        ],
        replay_rows=[
            {"case_id": "case-a", "rescue_tier": "desc_only"},
            {"case_id": "case-a", "rescue_tier": "desc_x1"},
        ],
        decision_rows=[
            {"case_id": "case-a", "rescue_tier": "desc_only"},
            {"case_id": "case-a", "rescue_tier": "desc_x1"},
        ],
    )

    with pytest.raises(ValueError, match="planned tier coverage"):
        merge_fn_rescue_shards(root, ["shard_000-of-001"])


def test_merge_summary_report_and_gallery_succeed_on_tiny_shards(tmp_path: Path) -> None:
    root = tmp_path / "merge_ok"
    shard0 = "shard_000-of-002"
    shard1 = "shard_001-of-002"
    _write_merge_shard(
        root,
        shard0,
        selected_rows=[
            {
                "case_id": "case-a",
                "source_line_idx": 0,
                "target_gt_idx": 11,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            }
        ],
        rescue_rows=[
            _base_rescue_row("case-a", shard0, "desc_only"),
            _base_rescue_row("case-a", shard0, "desc_x1"),
            _base_rescue_row(
                "case-a", shard0, "desc_x1_wrong_control", attempt_status="skipped"
            ),
        ],
        candidate_rows=[
            {
                "case_id": "case-a",
                "planned_rescue_tier": "desc_only",
                "region_kind": "target_gt",
                "region_instance_id": "gt:11",
                "source_index": 11,
                "aggregation_scope": "instance",
            }
        ],
        generation_rows=[
            {
                "case_id": "case-a",
                "source_line_idx": 0,
                "target_gt_idx": 11,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
                "rescue_tier": "desc_only",
                "valid_parse": True,
                "target_desc_preserved": True,
                "geometry_valid": True,
                "primary_rescue_success": True,
                "same_desc_duplicate_iou95": False,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": False,
                "duplicate_source_kind": None,
            },
            {
                "case_id": "case-a",
                "source_line_idx": 0,
                "target_gt_idx": 11,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
                "rescue_tier": "desc_x1",
                "valid_parse": True,
                "target_desc_preserved": False,
                "geometry_valid": True,
                "primary_rescue_success": False,
                "same_desc_duplicate_iou95": True,
                "success_iou30": True,
                "success_iou50": False,
                "success_iou75": False,
                "duplicate_source_kind": "same_desc_rollout_prediction",
            },
        ],
        replay_rows=[
            {"case_id": "case-a", "rescue_tier": "desc_only", "status": "ok"},
            {"case_id": "case-a", "rescue_tier": "desc_x1", "status": "ok"},
        ],
        decision_rows=[
            {"case_id": "case-a", "rescue_tier": "desc_only"},
            {"case_id": "case-a", "rescue_tier": "desc_x1"},
        ],
        attention_rows=[
            {
                "case_id": "case-a",
                "rescue_tier": "desc_only",
                "role": "pre_x1",
                "layer": 0,
                "head": 0,
                "aggregation_scope": "union",
                "region_kind": "target_gt",
                "region_instance_id": "union:target_gt",
                "attention_mass": 0.7,
            },
            {
                "case_id": "case-a",
                "rescue_tier": "desc_only",
                "role": "pre_x1",
                "layer": 0,
                "head": 0,
                "aggregation_scope": "instance",
                "region_kind": "target_gt",
                "region_instance_id": "gt:11",
                "attention_mass": 0.7,
            },
        ],
    )
    _write_merge_shard(
        root,
        shard1,
        selected_rows=[
            {
                "case_id": "case-b",
                "source_line_idx": 1,
                "target_gt_idx": 22,
                "target_desc": "lamp",
                "prefix_quality": "empty_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d0",
                "object_count_bucket": "gt_6_15",
            }
        ],
        rescue_rows=[
            _base_rescue_row("case-b", shard1, "desc_only"),
            _base_rescue_row("case-b", shard1, "desc_x1"),
            _base_rescue_row("case-b", shard1, "desc_x1_wrong_control"),
        ],
        candidate_rows=[
            {
                "case_id": "case-b",
                "planned_rescue_tier": "desc_only",
                "region_kind": "target_gt",
                "region_instance_id": "gt:22",
                "source_index": 22,
                "aggregation_scope": "instance",
            }
        ],
        wrong_control_rows=[
            {
                "case_id": "case-b",
                "rescue_tier": "desc_x1_wrong_control",
                "wrong_control_source_kind": "same_desc_competitor_gt_object",
                "wrong_control_source_bbox_xyxy": [400, 400, 450, 450],
                "wrong_control_source_x1": 400,
            }
        ],
        generation_rows=[
            {
                "case_id": "case-b",
                "source_line_idx": 1,
                "target_gt_idx": 22,
                "target_desc": "lamp",
                "prefix_quality": "empty_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d0",
                "object_count_bucket": "gt_6_15",
                "rescue_tier": "desc_only",
                "valid_parse": False,
                "target_desc_preserved": True,
                "geometry_valid": False,
                "primary_rescue_success": False,
                "same_desc_duplicate_iou95": False,
                "success_iou30": False,
                "success_iou50": False,
                "success_iou75": False,
                "duplicate_source_kind": None,
            },
            {
                "case_id": "case-b",
                "source_line_idx": 1,
                "target_gt_idx": 22,
                "target_desc": "lamp",
                "prefix_quality": "empty_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d0",
                "object_count_bucket": "gt_6_15",
                "rescue_tier": "desc_x1",
                "valid_parse": True,
                "target_desc_preserved": True,
                "geometry_valid": True,
                "primary_rescue_success": True,
                "same_desc_duplicate_iou95": False,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": True,
                "duplicate_source_kind": None,
            },
            {
                "case_id": "case-b",
                "source_line_idx": 1,
                "target_gt_idx": 22,
                "target_desc": "lamp",
                "prefix_quality": "empty_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d0",
                "object_count_bucket": "gt_6_15",
                "rescue_tier": "desc_x1_wrong_control",
                "valid_parse": True,
                "target_desc_preserved": True,
                "geometry_valid": True,
                "primary_rescue_success": False,
                "same_desc_duplicate_iou95": True,
                "success_iou30": True,
                "success_iou50": False,
                "success_iou75": False,
                "duplicate_source_kind": "same_desc_competitor_gt_object",
            },
        ],
        replay_rows=[
            {"case_id": "case-b", "rescue_tier": "desc_only", "status": "ok"},
            {"case_id": "case-b", "rescue_tier": "desc_x1", "status": "ok"},
            {"case_id": "case-b", "rescue_tier": "desc_x1_wrong_control", "status": "ok"},
        ],
        decision_rows=[
            {"case_id": "case-b", "rescue_tier": "desc_only"},
            {"case_id": "case-b", "rescue_tier": "desc_x1"},
            {"case_id": "case-b", "rescue_tier": "desc_x1_wrong_control"},
        ],
        attention_rows=[
            {
                "case_id": "case-b",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 0,
                "aggregation_scope": "union",
                "region_kind": "same_desc_competitor_gt_object",
                "region_instance_id": "union:same_desc_competitor_gt_object",
                "attention_mass": 0.55,
            },
            {
                "case_id": "case-b",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 0,
                "aggregation_scope": "instance",
                "region_kind": "same_desc_competitor_gt_object",
                "region_instance_id": "gt:22",
                "attention_mass": 0.55,
            }
        ],
    )

    merge_result = merge_fn_rescue_shards(root, [shard0, shard1])
    assert merge_result["validation_status"] == "ok"
    assert (root / "summary.json").exists()
    assert (root / "merge_summary.json").exists()
    assert set(FN_RESCUE_MERGE_JSONL_FILES.values()) <= {
        path.name for path in root.iterdir() if path.is_file()
    }
    merge_summary = json.loads((root / "merge_summary.json").read_text(encoding="utf-8"))
    for key, metadata in merge_summary["output_files"].items():
        output_path = Path(metadata["path"])
        assert output_path.exists()
        assert output_path.parent == root
        if output_path.suffix == ".jsonl":
            final_rows = read_jsonl(output_path)
            assert metadata["row_count"] == len(final_rows)
        assert metadata["sha256"] == _sha256_file(output_path)

    summary = summarize_fn_rescue_artifacts(root)
    assert summary["row_counts"]["rescue_rows"] == 6
    assert summary["source_selected_cases_path"] == "/tmp/source_selected_cases.jsonl"
    assert summary["selected_case_counts_by_bucket"]["prefix_quality"] == {
        "clean_prefix": 1,
        "empty_prefix": 1,
    }
    assert summary["denominator"]["attempted"] == 5
    assert summary["denominator"]["skipped"] == 1
    assert summary["denominator"]["invalid_parse"] == 1
    assert summary["denominator"]["target_desc_not_preserved"] == 1
    assert summary["denominator"]["geometry_invalid"] == 1
    assert summary["denominator"]["wrong_control_unavailable"] == 1
    assert summary["generation_stats"]["by_tier"]["desc_only"]["success_iou50"] == pytest.approx(
        0.5
    )
    assert summary["generation_stats"]["by_tier"]["desc_only"]["parse_valid_rate"] == pytest.approx(
        0.5
    )
    assert summary["generation_stats"]["by_tier"]["desc_x1"]["raw_rescue_success_rate"] == pytest.approx(
        0.5
    )
    assert summary["generation_stats"]["by_tier"]["desc_x1"]["primary_rescue_success_rate"] == pytest.approx(
        0.5
    )
    assert summary["generation_stats"]["by_tier"]["desc_x1"]["duplicate_rejected_count"] == 1
    assert summary["generation_stats"]["by_stratum"]
    assert summary["wrong_control_source_distribution"] == {
        "same_desc_competitor_gt_object": 1
    }
    assert summary["attention_summary"]["union_by_tier_role_layer_group"]
    assert summary["attention_summary"]["instance_diagnostics"]
    assert summary["attention_summary"]["union_by_tier_role_layer_group"][0]["layer_group"]

    report_path = write_fn_rescue_report(root)
    report_text = report_path.read_text(encoding="utf-8")
    assert "Interpretation Bounds" in report_text
    assert "Gallery rows are qualitative only" in report_text
    assert "Denominator Ledger" in report_text
    assert "Source Selected Cases Path" in report_text
    assert "Parse-Valid Rates By Tier" in report_text
    assert "IoU Rates By Tier" in report_text
    assert "IoU Rates By Stratum" in report_text
    assert "Wrong-Control Source Distribution" in report_text
    assert "Attention Union Summary" in report_text
    assert "Per-Instance Attention Diagnostics" in report_text
    assert "Source Artifacts" in report_text
    assert "Merged Output Row Counts" in report_text
    assert "/tmp/source_selected_cases.jsonl" in report_text
    assert "file_sha256" in report_text
    assert "selected-hash" in report_text
    assert "directory_structural_fingerprint" in report_text
    assert "checkpoint-fingerprint" in report_text
    assert "rescue_rows" in report_text
    assert "rescue_generation_rows" in report_text

    with pytest.raises(ValueError, match="gallery runtime unavailable"):
        write_fn_rescue_gallery(root, per_bucket=1)


def test_write_fn_rescue_gallery_renders_pngs_with_canonical_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    config = _write_tiny_fn_rescue_fixture(tmp_path)
    root = config.paths.artifact_root
    materialize_fn_rescue_select_cases_shard(config, shard_index=0, num_shards=1)
    _write_merge_shard(
        root,
        "shard_000-of-001",
        selected_rows=[
            {
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            }
        ],
        rescue_rows=[
            {
                **_base_rescue_row("case-a", "shard_000-of-001", "desc_only"),
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
            },
            {
                **_base_rescue_row("case-a", "shard_000-of-001", "desc_x1"),
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
            },
            {
                **_base_rescue_row(
                    "case-a",
                    "shard_000-of-001",
                    "desc_x1_wrong_control",
                    attempt_status="emitted",
                ),
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
            },
        ],
        generation_rows=[
            {
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "rescue_tier": "desc_only",
                "generated_box_xyxy": [110, 110, 190, 190],
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
                "valid_parse": True,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": True,
                "primary_rescue_success": True,
                "same_desc_duplicate_iou95": False,
            },
            {
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "rescue_tier": "desc_x1",
                "generated_box_xyxy": [100, 120, 210, 220],
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
                "valid_parse": True,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": True,
                "primary_rescue_success": False,
                "same_desc_duplicate_iou95": False,
            },
            {
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "rescue_tier": "desc_x1_wrong_control",
                "generated_box_xyxy": [80, 120, 140, 220],
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
                "valid_parse": True,
                "success_iou30": False,
                "success_iou50": False,
                "success_iou75": False,
                "primary_rescue_success": False,
                "same_desc_duplicate_iou95": True,
            },
        ],
        wrong_control_rows=[
            {
                "case_id": "case-0",
                "rescue_tier": "desc_x1_wrong_control",
                "wrong_control_source_kind": "same_desc_competitor_gt_object",
                "wrong_control_source_bbox_xyxy": [70, 130, 145, 235],
                "wrong_control_source_x1": 70,
            }
        ],
        replay_rows=[
            {"case_id": "case-0", "rescue_tier": "desc_only", "status": "ok"},
            {"case_id": "case-0", "rescue_tier": "desc_x1", "status": "ok"},
            {"case_id": "case-0", "rescue_tier": "desc_x1_wrong_control", "status": "ok"},
        ],
        decision_rows=[
            {"case_id": "case-0", "rescue_tier": "desc_only"},
            {"case_id": "case-0", "rescue_tier": "desc_x1"},
            {"case_id": "case-0", "rescue_tier": "desc_x1_wrong_control"},
        ],
    )
    merge_fn_rescue_shards(root, 1)

    captured_source_rows: list[dict[str, object]] = []

    def _fake_vis_helpers() -> tuple[object, object]:
        def _ensure(
            source_jsonl: Path,
            *,
            output_path: Path | None = None,
            source_kind: str = "offline_single_run",
            materialize_matching: bool = True,
        ) -> Path:
            del source_kind, materialize_matching
            assert output_path is not None
            captured_source_rows.extend(read_jsonl(source_jsonl))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(source_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
            return output_path

        def _render(
            jsonl_path: Path,
            *,
            out_dir: Path,
            limit: int = 20,
            root_image_dir: Path | None = None,
            root_source: str = "none",
            record_order: str = "input",
        ) -> None:
            del jsonl_path, limit, root_image_dir, root_source, record_order
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(out_dir / "vis_0000.png")

        return _ensure, _render

    monkeypatch.setattr(fn_rescue_module, "_canonical_gt_vs_pred_helpers", _fake_vis_helpers)
    gallery_result = write_fn_rescue_gallery(config, per_bucket=1)
    index_rows = read_jsonl(Path(gallery_result["gallery_index"]))
    rendered_rows = [row for row in index_rows if row["rendering_status"] == "rendered"]
    assert rendered_rows
    rendered_image = Path(rendered_rows[0]["rendered_image"])
    assert rendered_image.exists()
    assert rendered_image.suffix == ".png"
    assert "gallery/images/" in str(rendered_image)
    assert captured_source_rows
    pred_descs = [str(obj["desc"]) for obj in captured_source_rows[0]["pred"]]
    assert "FN rescue desc_only" in pred_descs
    assert "FN rescue desc_x1" in pred_descs
    assert "FN rescue desc_x1_wrong_control" in pred_descs
    assert "wrong-control source" in pred_descs
    debug_payload = captured_source_rows[0]["debug"]
    assert debug_payload["visual_roles"]["rescue_desc_only"]["desc"] == "FN rescue desc_only"
    assert debug_payload["visual_roles"]["wrong_control_source"]["desc"] == "wrong-control source"
    provenance = captured_source_rows[0]["provenance"]
    assert provenance["gallery_case_id"] == "case-0"
    assert provenance["gallery_bucket_name"]


def test_summarize_fn_rescue_artifacts_returns_explicit_zero_tables_when_generation_is_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "summary_zero_tables"
    _write_merge_shard(
        root,
        "shard_000-of-001",
        selected_rows=[
            {
                "case_id": "case-a",
                "source_line_idx": 0,
                "target_gt_idx": 11,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            }
        ],
        rescue_rows=[
            _base_rescue_row("case-a", "shard_000-of-001", "desc_only", attempt_status="skipped"),
            _base_rescue_row("case-a", "shard_000-of-001", "desc_x1", attempt_status="skipped"),
            _base_rescue_row(
                "case-a",
                "shard_000-of-001",
                "desc_x1_wrong_control",
                attempt_status="skipped",
            ),
        ],
    )
    merge_summary = {
        "artifact_root": str(root),
        "checkpoint": "/tmp/checkpoint",
        "evidence_scope": "val200_attention_atlas_linked_fn_stratified",
        "config_sha256": "cfg-hash",
        "expected_shard_labels": ["shard_000-of-001"],
        "source_artifacts": {
            "source_selected_cases": {
                "path": "/tmp/source_selected_cases.jsonl",
                "exists": True,
                "byte_size": 10,
                "row_count": 1,
                "hash_kind": "file_sha256",
                "sha256": "selected-hash",
            }
        },
        "output_files": {},
        "validation_status": "test_fixture",
    }
    (root / "merge_summary.json").write_text(
        json.dumps(merge_summary, sort_keys=True), encoding="utf-8"
    )

    summary = summarize_fn_rescue_artifacts(root)
    assert summary["generation_stats"]["total_rows"] == 0
    assert summary["generation_stats"]["by_tier"]["desc_only"]["count"] == 0
    assert summary["generation_stats"]["by_tier"]["desc_only"]["parse_valid_rate"] == 0.0
    assert summary["attention_summary"]["union_by_tier_role_layer_group"] == []
    assert summary["attention_summary"]["instance_diagnostics"] == []


def test_write_fn_rescue_gallery_root_only_uses_manifest_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    config = _write_tiny_fn_rescue_fixture(tmp_path)
    root = config.paths.artifact_root
    materialize_fn_rescue_select_cases_shard(config, shard_index=0, num_shards=1)
    manifest_path = root / "shards_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_path"] = str(root / "missing_config.yaml")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    _write_merge_shard(
        root,
        "shard_000-of-001",
        selected_rows=[
            {
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            }
        ],
        rescue_rows=[
            {
                **_base_rescue_row("case-a", "shard_000-of-001", "desc_only"),
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
            },
            {
                **_base_rescue_row("case-a", "shard_000-of-001", "desc_x1"),
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
            },
            {
                **_base_rescue_row(
                    "case-a",
                    "shard_000-of-001",
                    "desc_x1_wrong_control",
                    attempt_status="emitted",
                ),
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
            },
        ],
        generation_rows=[
            {
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "rescue_tier": "desc_only",
                "generated_box_xyxy": [110, 110, 190, 190],
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
                "valid_parse": True,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": True,
                "primary_rescue_success": True,
                "same_desc_duplicate_iou95": False,
            },
            {
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "rescue_tier": "desc_x1",
                "generated_box_xyxy": [100, 120, 210, 220],
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
                "valid_parse": True,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": True,
                "primary_rescue_success": False,
                "same_desc_duplicate_iou95": False,
            },
            {
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "rescue_tier": "desc_x1_wrong_control",
                "generated_box_xyxy": [80, 120, 140, 220],
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
                "valid_parse": True,
                "success_iou30": False,
                "success_iou50": False,
                "success_iou75": False,
                "primary_rescue_success": False,
                "same_desc_duplicate_iou95": True,
            },
        ],
        wrong_control_rows=[
            {
                "case_id": "case-0",
                "rescue_tier": "desc_x1_wrong_control",
                "wrong_control_source_kind": "same_desc_competitor_gt_object",
                "wrong_control_source_bbox_xyxy": [70, 130, 145, 235],
                "wrong_control_source_x1": 70,
            }
        ],
        replay_rows=[
            {"case_id": "case-0", "rescue_tier": "desc_only", "status": "ok"},
            {"case_id": "case-0", "rescue_tier": "desc_x1", "status": "ok"},
            {"case_id": "case-0", "rescue_tier": "desc_x1_wrong_control", "status": "ok"},
        ],
        decision_rows=[
            {"case_id": "case-0", "rescue_tier": "desc_only"},
            {"case_id": "case-0", "rescue_tier": "desc_x1"},
            {"case_id": "case-0", "rescue_tier": "desc_x1_wrong_control"},
        ],
    )
    merge_fn_rescue_shards(root, 1)

    def _fake_vis_helpers() -> tuple[object, object]:
        def _ensure(
            source_jsonl: Path,
            *,
            output_path: Path | None = None,
            source_kind: str = "offline_single_run",
            materialize_matching: bool = True,
        ) -> Path:
            del source_kind, materialize_matching
            assert output_path is not None
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(source_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
            return output_path

        def _render(
            jsonl_path: Path,
            *,
            out_dir: Path,
            limit: int = 20,
            root_image_dir: Path | None = None,
            root_source: str = "none",
            record_order: str = "input",
        ) -> None:
            del jsonl_path, limit, root_source, record_order
            assert root_image_dir is not None
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(out_dir / "vis_0000.png")

        return _ensure, _render

    monkeypatch.setattr(fn_rescue_module, "_canonical_gt_vs_pred_helpers", _fake_vis_helpers)
    gallery_result = write_fn_rescue_gallery(root, per_bucket=1)
    index_rows = read_jsonl(Path(gallery_result["gallery_index"]))
    assert any(row["rendering_status"] == "rendered" for row in index_rows)


def test_write_fn_rescue_gallery_root_only_requires_manifest_runtime(
    tmp_path: Path,
) -> None:
    config = _write_tiny_fn_rescue_fixture(tmp_path)
    root = config.paths.artifact_root
    materialize_fn_rescue_select_cases_shard(config, shard_index=0, num_shards=1)
    _write_merge_shard(
        root,
        "shard_000-of-001",
        selected_rows=[
            {
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            }
        ],
        rescue_rows=[
            {
                **_base_rescue_row("case-a", "shard_000-of-001", "desc_only"),
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
            },
            {
                **_base_rescue_row("case-a", "shard_000-of-001", "desc_x1"),
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
            },
            {
                **_base_rescue_row(
                    "case-a",
                    "shard_000-of-001",
                    "desc_x1_wrong_control",
                    attempt_status="skipped",
                ),
                "case_id": "case-0",
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
            },
        ],
        generation_rows=[
            {
                "case_id": "case-0",
                "rescue_tier": "desc_only",
                "generated_box_xyxy": [110, 110, 190, 190],
                "valid_parse": True,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": True,
                "primary_rescue_success": True,
                "same_desc_duplicate_iou95": False,
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            },
            {
                "case_id": "case-0",
                "rescue_tier": "desc_x1",
                "generated_box_xyxy": [100, 120, 210, 220],
                "valid_parse": True,
                "success_iou30": True,
                "success_iou50": True,
                "success_iou75": True,
                "primary_rescue_success": False,
                "same_desc_duplicate_iou95": False,
                "source_line_idx": 0,
                "target_gt_idx": 19,
                "target_desc": "vase",
                "prefix_quality": "clean_prefix",
                "binding_bucket": "same_desc_competitor",
                "depth_bucket": "d1_3",
                "object_count_bucket": "gt_6_15",
            },
        ],
        replay_rows=[
            {"case_id": "case-0", "rescue_tier": "desc_only", "status": "ok"},
            {"case_id": "case-0", "rescue_tier": "desc_x1", "status": "ok"},
        ],
        decision_rows=[
            {"case_id": "case-0", "rescue_tier": "desc_only"},
            {"case_id": "case-0", "rescue_tier": "desc_x1"},
        ],
    )
    merge_fn_rescue_shards(root, 1)
    (root / "shards_manifest.json").unlink()

    with pytest.raises(ValueError, match="gallery runtime"):
        write_fn_rescue_gallery(root, per_bucket=1)


def test_runner_dry_run_prints_compact_json(tmp_path: Path) -> None:
    _write_tiny_fn_rescue_fixture(tmp_path)
    config_path = tmp_path / "tiny_fn_rescue.yaml"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--config",
            str(config_path),
            "--stages",
            "select_cases",
            "--shard-index",
            "0",
            "--num-shards",
            "2",
            "--dry-run",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["stages"] == ["select_cases"]
    assert payload["available_stages"] == list(FN_RESCUE_STAGES)
    assert payload["shard"] == {
        "num_shards": 2,
        "shard_index": 0,
        "shard_label": "shard_000-of-002",
    }
    assert payload["expected_shard_labels"] == ["shard_000-of-002", "shard_001-of-002"]


def test_runner_rejects_unknown_stage(tmp_path: Path) -> None:
    _write_tiny_fn_rescue_fixture(tmp_path)
    config_path = tmp_path / "tiny_fn_rescue.yaml"
    env = {**os.environ, "PYTHONPATH": "/data/CoordExp"}

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--config",
            str(config_path),
            "--stages",
            "select_cases,not_a_stage",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "unknown FN-rescue stage(s): not_a_stage" in result.stderr


def test_runner_rejects_non_merge_stage_when_merge_shards_enabled(tmp_path: Path) -> None:
    _write_tiny_fn_rescue_fixture(tmp_path)
    config_path = tmp_path / "tiny_fn_rescue.yaml"
    env = {**os.environ, "PYTHONPATH": "/data/CoordExp"}

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--config",
            str(config_path),
            "--stages",
            "select_cases",
            "--merge-shards",
            "--num-shards",
            "2",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "--merge-shards requires stages drawn from merge,report,gallery" in result.stderr


def test_runner_executes_ordered_multi_stage_shard_dispatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script_module(RUNNER_PATH, "autoreg_fn_rescue_runner_multi_stage_test")
    config = load_fn_rescue_config(CONFIG_PATH)
    calls: list[str] = []

    monkeypatch.setattr(module, "load_fn_rescue_config", lambda _: config)
    monkeypatch.setattr(
        module,
        "materialize_fn_rescue_select_cases_shard",
        lambda *_args, **_kwargs: calls.append("select_cases") or {"stage": "select_cases"},
    )
    monkeypatch.setattr(
        module,
        "materialize_fn_rescue_feasibility_shard",
        lambda *_args, **_kwargs: calls.append("feasibility") or {"stage": "feasibility"},
    )
    monkeypatch.setattr(
        module,
        "materialize_fn_rescue_decode_shard",
        lambda *_args, **_kwargs: calls.append("rescue_decode") or {"stage": "rescue_decode"},
    )
    monkeypatch.setattr(
        module,
        "materialize_fn_rescue_attention_replay_shard",
        lambda *_args, **_kwargs: calls.append("attention_replay")
        or {"stage": "attention_replay"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--config",
            str(CONFIG_PATH),
            "--stages",
            "select_cases,feasibility,rescue_decode,attention_replay",
            "--shard-index",
            "0",
            "--num-shards",
            "3",
        ],
    )

    assert module.main() == 0
    assert calls == ["select_cases", "feasibility", "rescue_decode", "attention_replay"]
    payload = json.loads(capsys.readouterr().out)
    assert list(payload.keys()) == [
        "select_cases",
        "feasibility",
        "rescue_decode",
        "attention_replay",
    ]
    assert payload["rescue_decode"]["stage"] == "rescue_decode"


def test_stage_pipeline_materializes_manifest_and_mocked_stage_outputs(
    tmp_path: Path,
) -> None:
    config = _write_tiny_fn_rescue_fixture(tmp_path)
    select_payload = materialize_fn_rescue_select_cases_shard(
        config, shard_index=0, num_shards=1
    )
    shard_root = Path(select_payload["shard_root"])
    manifest_path = config.paths.artifact_root / "shards_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["analysis_name"] == "autoreg_fn_rescue_continuation"
    assert manifest["expected_shard_labels"] == ["shard_000-of-001"]
    assert manifest["pre_y1_definition"]

    feasibility_tokenizer = FakeTokenizer()
    feasibility_processor = FakeProcessor(feasibility_tokenizer)
    feasibility_handle = SimpleNamespace(
        tokenizer=feasibility_tokenizer,
        processor=feasibility_processor,
        model=FakeDynamicReplayModel(),
    )
    def _feasibility_inputs_builder(
        *,
        model_handle: object,
        assistant_prefix_text: str,
        messages: list[dict[str, object]],
        **_: object,
    ) -> dict[str, object]:
        del model_handle
        return _fake_processor_inputs_for_prefix(
            feasibility_tokenizer,
            feasibility_processor,
            assistant_prefix_text,
            messages=messages,
        )
    feasibility_payload = materialize_fn_rescue_feasibility_shard(
        config,
        shard_index=0,
        num_shards=1,
        model_handle_loader=lambda _config: feasibility_handle,
        processor_input_builder=_feasibility_inputs_builder,
    )
    feasibility_rows = read_jsonl(shard_root / "feasibility_rows.jsonl")
    assert feasibility_payload["row_counts"]["feasibility_rows"] == 2
    assert len(feasibility_rows) == 2
    assert all(row["feasible"] is True for row in feasibility_rows)
    assert all(row["replay_prefix_token_match"] is True for row in feasibility_rows)
    assert all(
        (int(row["visual_span_end"]) - int(row["visual_span_start"])) == 1
        for row in feasibility_rows
    )
    assert all(row["visual_grid_h"] == 1 for row in feasibility_rows)
    assert all(row["visual_grid_w"] == 1 for row in feasibility_rows)
    assert all(row["attention_layer_count"] == 1 for row in feasibility_rows)
    assert all(row["attention_shape_0"] == 1 for row in feasibility_rows)
    assert all(row["processor_do_resize_or_policy"] is False for row in feasibility_rows)
    assert {row["query_role"] for row in feasibility_rows} == {"pre_x1", "pre_y1"}

    decode_tokenizer = FakeTokenizer()
    decode_processor = FakeProcessor(decode_tokenizer)
    selected_rows = read_jsonl(shard_root / "selected_rescue_cases.jsonl")
    dataset_rows = read_jsonl(config.paths.dataset_jsonl)
    runtime_spec = fn_rescue_module._fn_rescue_runtime_spec(config)
    emitted_rows = [
        row
        for row in read_jsonl(shard_root / "rescue_rows.jsonl")
        if bool(row["emitted"])
    ]
    selected_by_case = {str(row["case_id"]): row for row in selected_rows}
    generated_sequences: list[list[int]] = []
    for rescue_row in emitted_rows:
        selected_row = selected_by_case[str(rescue_row["case_id"])]
        assistant_prefix = build_rescue_assistant_prefix(
            prefix_text=str(selected_row["prefix_text"] or ""),
            target_desc=str(rescue_row["target_desc"]),
            tier=str(rescue_row["planned_rescue_tier"]),
            hint_x1=(
                None
                if rescue_row.get("hint_x1") is None
                else int(rescue_row["hint_x1"])
            ),
        )
        prompt_ids = _fake_processor_inputs_for_prefix(
            decode_tokenizer,
            decode_processor,
            assistant_prefix,
            messages=fn_rescue_module._build_fn_rescue_messages(
                runtime_spec=runtime_spec,
                image_path=fn_rescue_module._resolve_fn_rescue_image_path(
                    dataset_rows[int(rescue_row["source_line_idx"])],
                    image_root=Path(runtime_spec["root_image_dir"]),
                ),
                assistant_prefix_text=assistant_prefix,
            ),
        )["input_ids"][0]
        tail_text = (
            "<|coord_100|><|coord_100|><|coord_200|><|coord_200|>"
            if rescue_row["planned_rescue_tier"] == "desc_only"
            else "<|coord_100|><|coord_200|><|coord_200|>"
        )
        tail_ids = decode_tokenizer.encode(tail_text, add_special_tokens=False)
        generated_sequences.append([*prompt_ids, *tail_ids])
    decode_model = FakeSequentialGenerateModel(generated_sequences)
    decode_handle = SimpleNamespace(
        tokenizer=decode_tokenizer,
        processor=decode_processor,
        model=decode_model,
    )

    def _decode_inputs_builder(
        *,
        model_handle: object,
        assistant_prefix_text: str,
        messages: list[dict[str, object]],
        **_: object,
    ) -> dict[str, object]:
        del model_handle
        return _fake_processor_inputs_for_prefix(
            decode_tokenizer,
            decode_processor,
            assistant_prefix_text,
            messages=messages,
        )

    decode_payload = materialize_fn_rescue_decode_shard(
        config,
        shard_index=0,
        num_shards=1,
        model_handle_loader=lambda _config: decode_handle,
        processor_input_builder=_decode_inputs_builder,
    )
    generation_rows = read_jsonl(shard_root / "rescue_generation_rows.jsonl")
    assert decode_payload["row_counts"]["rescue_generation_rows"] == 2
    assert len(generation_rows) == 2
    assert all("assistant_prefix_text" in row for row in generation_rows)
    assert all("assistant_prefix_token_ids" in row for row in generation_rows)
    assert all("generated_token_ids" in row for row in generation_rows)
    assert all("generation_kwargs" in row for row in generation_rows)
    assert all("input_ids" not in row["generation_kwargs"] for row in generation_rows)
    assert all("attention_mask" not in row["generation_kwargs"] for row in generation_rows)
    assert all("image_grid_thw" not in row["generation_kwargs"] for row in generation_rows)
    assert all(row["generation_kwargs"]["do_sample"] is False for row in generation_rows)
    assert all(row["generation_kwargs"]["num_beams"] == 1 for row in generation_rows)
    assert all(row["valid_parse"] is True for row in generation_rows)
    assert all(row["target_desc_preserved"] is True for row in generation_rows)
    assert all(row["geometry_valid"] is True for row in generation_rows)
    allowed_outcomes = {
        "primary_rescue_success",
        "duplicate_rejected",
        "target_iou_below_0p50",
    }
    assert all(row["outcome_bucket"] in allowed_outcomes for row in generation_rows)
    for row in generation_rows:
        if row["primary_rescue_success"]:
            assert row["outcome_bucket"] == "primary_rescue_success"
        elif row["same_desc_duplicate_iou95"]:
            assert row["outcome_bucket"] == "duplicate_rejected"

    replay_tokenizer = FakeTokenizer()
    replay_processor = FakeProcessor(replay_tokenizer)
    replay_handle = SimpleNamespace(
        tokenizer=replay_tokenizer,
        processor=replay_processor,
        model=FakeDynamicReplayModel(),
    )

    def _replay_inputs_builder(
        *,
        model_handle: object,
        assistant_prefix_text: str,
        messages: list[dict[str, object]],
        **_: object,
    ) -> dict[str, object]:
        del model_handle
        return _fake_processor_inputs_for_prefix(
            replay_tokenizer,
            replay_processor,
            assistant_prefix_text,
            messages=messages,
        )

    replay_payload = materialize_fn_rescue_attention_replay_shard(
        config,
        shard_index=0,
        num_shards=1,
        model_handle_loader=lambda _config: replay_handle,
        processor_input_builder=_replay_inputs_builder,
    )
    replay_prefix_rows = read_jsonl(shard_root / "rescue_replay_prefix_rows.jsonl")
    decision_rows = read_jsonl(shard_root / "rescue_decision_context_rows.jsonl")
    attention_rows = read_jsonl(shard_root / "rescue_attention_region_rows.jsonl")
    assert replay_payload["row_counts"]["rescue_replay_prefix_rows"] == 2
    assert replay_payload["row_counts"]["rescue_decision_context_rows"] == 2
    assert len(replay_prefix_rows) == 2
    assert len(decision_rows) == 2
    assert attention_rows
    assert {row["aggregation_scope"] for row in attention_rows} >= {"instance", "union"}
    assert all("region_instance_id" in row for row in attention_rows)
    assert all(
        not str(row["region_kind"]).startswith(("instance:", "union:"))
        for row in attention_rows
    )
    assert all(row["query_role"] in {"pre_x1", "pre_y1"} for row in decision_rows)
    assert all(row["replay_prefix_token_match"] is True for row in replay_prefix_rows)


def test_attention_replay_allows_decode_completed_with_zero_generation_rows(
    tmp_path: Path,
) -> None:
    config = _write_tiny_fn_rescue_fixture(tmp_path)
    select_payload = materialize_fn_rescue_select_cases_shard(
        config, shard_index=0, num_shards=1
    )
    shard_root = Path(select_payload["shard_root"])
    _write_jsonl(shard_root / "rescue_generation_rows.jsonl", [])
    summary_path = shard_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["stages_completed"] = ["select_cases", "rescue_decode"]
    summary["row_counts"]["rescue_generation_rows"] = 0
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    def _unexpected_loader(_config: FnRescueConfig) -> object:
        raise AssertionError("zero-generation attention replay should not load model")

    replay_payload = materialize_fn_rescue_attention_replay_shard(
        config,
        shard_index=0,
        num_shards=1,
        model_handle_loader=_unexpected_loader,
    )

    refreshed = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "attention_replay" in refreshed["stages_completed"]
    assert replay_payload["row_counts"]["rescue_generation_rows"] == 0
    assert replay_payload["row_counts"]["rescue_replay_prefix_rows"] == 0
    assert replay_payload["row_counts"]["rescue_decision_context_rows"] == 0
    assert replay_payload["row_counts"]["rescue_attention_region_rows"] == 0
    assert read_jsonl(shard_root / "rescue_replay_prefix_rows.jsonl") == []
    assert read_jsonl(shard_root / "rescue_decision_context_rows.jsonl") == []
    assert read_jsonl(shard_root / "rescue_attention_region_rows.jsonl") == []


def test_default_fn_rescue_processor_input_builder_loads_real_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    torch = pytest.importorskip("torch")
    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (4, 4), color=(12, 34, 56)).save(image_path)
    tokenizer = FakeTokenizer()
    processor = FakeProcessor(tokenizer)
    captured: dict[str, object] = {}

    import src.common.qwen_generation as qwen_generation_module

    def _fake_call_processor_with_qwen_geometry(
        processor_arg: object,
        *,
        text: list[str],
        images: list[object],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, object]:
        assert processor_arg is processor
        assert return_tensors == "pt"
        assert padding is False
        assert len(images) == 1
        captured["text"] = text[0]
        captured["image_mode"] = getattr(images[0], "mode", None)
        captured["image_size"] = getattr(images[0], "size", None)
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "image_grid_thw": [[1, 1, 1]],
        }

    monkeypatch.setattr(
        qwen_generation_module,
        "call_processor_with_qwen_geometry",
        _fake_call_processor_with_qwen_geometry,
    )
    model_handle = SimpleNamespace(processor=processor, model=None)
    inputs = fn_rescue_module._default_fn_rescue_processor_input_builder(
        model_handle=model_handle,
        assistant_prefix_text="<|object_ref_start|>vase<|box_start|>",
        image_path=image_path,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "find vase"},
                ],
            },
            {"role": "assistant", "content": "<|object_ref_start|>vase<|box_start|>"},
        ],
    )
    assert captured["image_mode"] == "RGB"
    assert captured["image_size"] == (4, 4)
    assert "<|image_pad|>" in str(captured["text"])
    assert inputs["input_ids"].shape == (1, 3)


def test_runner_dispatches_merge_report_and_gallery(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module(RUNNER_PATH, "autoreg_fn_rescue_runner_test")
    config = load_fn_rescue_config(CONFIG_PATH)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(module, "load_fn_rescue_config", lambda _: config)

    def _merge(root: Path, *, expected_shards: int) -> dict[str, object]:
        calls.append(("merge", (root, expected_shards)))
        return {"artifact_root": str(root), "validation_status": "ok"}

    def _report(root: Path) -> Path:
        calls.append(("report", root))
        return Path(root) / "report.md"

    def _gallery(root: Path) -> dict[str, object]:
        calls.append(("gallery", root))
        return {"gallery_root": str(Path(root) / "gallery")}

    monkeypatch.setattr(module, "merge_fn_rescue_shards", _merge)
    monkeypatch.setattr(module, "write_fn_rescue_report", _report)
    monkeypatch.setattr(module, "write_fn_rescue_gallery", _gallery)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--config",
            str(CONFIG_PATH),
            "--stages",
            "merge,report,gallery",
            "--merge-shards",
            "--num-shards",
            "3",
        ],
    )

    assert module.main() == 0
    assert calls == [
        ("merge", (config.paths.artifact_root, 3)),
        ("report", config.paths.artifact_root),
        ("gallery", config.paths.artifact_root),
    ]


@pytest.mark.parametrize("stage_name", ["report", "gallery"])
def test_runner_dispatches_singleton_report_or_gallery(
    monkeypatch: pytest.MonkeyPatch, stage_name: str
) -> None:
    module = _load_script_module(RUNNER_PATH, f"autoreg_fn_rescue_runner_{stage_name}_test")
    config = load_fn_rescue_config(CONFIG_PATH)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(module, "load_fn_rescue_config", lambda _: config)
    monkeypatch.setattr(
        module,
        "write_fn_rescue_report",
        lambda root: calls.append(("report", root)) or Path(root) / "report.md",
    )
    monkeypatch.setattr(
        module,
        "write_fn_rescue_gallery",
        lambda root: calls.append(("gallery", root))
        or {"gallery_root": str(Path(root) / "gallery")},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--config",
            str(CONFIG_PATH),
            "--stages",
            stage_name,
        ],
    )

    assert module.main() == 0
    assert calls == [(stage_name, config.paths.artifact_root)]


def test_launcher_dry_run_writes_expected_command_file(tmp_path: Path) -> None:
    config_path = _write_launcher_config(tmp_path)
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "CONFIG": str(config_path),
        "SESSION": "autoreg_fn_rescue_test",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "DRY_RUN=1; not starting tmux session." in result.stdout
    assert "Started tmux session" not in result.stdout

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(payload["paths"]["artifact_root"])
    command_file = root / "logs" / "autoreg_fn_rescue_test_commands.sh"
    assert command_file.exists()
    command_text = command_file.read_text(encoding="utf-8")
    assert "--stages select_cases" in command_text
    assert "--stages feasibility" in command_text
    assert "--stages rescue_decode" in command_text
    assert "--stages attention_replay" in command_text
    assert "--stages merge,report,gallery" in command_text
    assert f"cd {str(REPO_ROOT)!r}" in command_text or f"cd {REPO_ROOT}" in command_text
    assert str(RUNNER_PATH) in command_text
    assert "PYTHONPATH=" + str(REPO_ROOT) in command_text
    assert "CUDA_VISIBLE_DEVICES=" in command_text
    assert "shard_000-of-008_select_cases.log" in command_text
    assert "shard_000-of-008_feasibility.log" in command_text
    assert "shard_000-of-008_rescue_decode.log" in command_text
    assert "shard_000-of-008_attention_replay.log" in command_text
    assert "merge.log" in command_text


@pytest.mark.parametrize(
    "artifact_root_factory",
    [
        lambda base: base / "source" / "attention_atlas",
        lambda base: base / "source" / "attention_atlas" / "nested" / "bad_root",
    ],
)
def test_launcher_rejects_unsafe_artifact_root(
    tmp_path: Path, artifact_root_factory: object
) -> None:
    artifact_root = artifact_root_factory(tmp_path / "fixture")
    config_path = _write_launcher_config(tmp_path, artifact_root=artifact_root)
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "CONFIG": str(config_path),
        "SESSION": "autoreg_fn_rescue_unsafe",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "artifact_root must not" in result.stderr
