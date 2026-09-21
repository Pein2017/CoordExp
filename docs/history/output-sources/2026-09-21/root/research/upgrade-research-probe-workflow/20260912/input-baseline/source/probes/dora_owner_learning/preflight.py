"""CPU-only Source256 saved-input encoding and optional immutable-plan checks."""

import argparse
import json
from pathlib import Path

from src.config.inference import load_research_infer_config
from src.config.fingerprint import sha256_json
from src.data import load_raw_examples
from src.inference.runtime import assemble_frontend
from src.qwen.native import prepare_native_inputs

from .prepare import EOS_TOKEN_ID, TRAIN256_SHA256, _native_ce_group, file_sha256, validate_plan
from .runtime import DEFAULT_CONFIG, build_request


def preflight(config_path=DEFAULT_CONFIG, *, rows=1, plan=None):
    if not 1 <= rows <= 8:
        raise ValueError("CPU preflight covers one to eight rows")
    resolved = load_research_infer_config(config_path)
    config = resolved.config
    input_path = Path(config.data.input_jsonl)
    if file_sha256(input_path) != TRAIN256_SHA256:
        raise ValueError("Source256 input identity changed")
    examples = list(load_raw_examples(input_path))
    if len(examples) != 256:
        raise ValueError("Source256 requires the saved 256-image input")
    frontend = assemble_frontend(
        config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json"))
    )
    if frontend.qwen.model is not None:
        raise ValueError("CPU encoding preflight unexpectedly loaded model weights")
    encoded = [_native_ce_group(raw, index=index, config=config, frontend=frontend)
               for index, raw in enumerate(examples[:rows])]
    media = []
    for index, (raw, group) in enumerate(zip(examples[:rows], encoded, strict=True)):
        request, _, _ = build_request(raw, config=config, qwen=frontend.qwen, row_index=index)
        batch = prepare_native_inputs(frontend.qwen.processor, (request,), device="cpu", record_media_identity=True)
        if (list(batch.prompt_token_ids[0]) != group["prompt_token_ids"]
                or list(batch.image_grids[0]) != group["expected_image_grid_thw"]):
            raise ValueError("native CPU inputs differ from planned CE encoding")
        media.append(batch.media_sha256[0])
    if plan is not None:
        validate_plan(Path(plan))
    return {
        "status": "cpu_input_preflight_passed", "model_weights_loaded": False,
        "population_count": len(examples), "encoded_count": len(encoded),
        "config_fingerprint": resolved.fingerprint, "input_sha256": file_sha256(input_path),
        "plan_checked": None if plan is None else str(Path(plan).resolve()),
        "rows": [{"example_id": group["example_id"], "executed_media_sha256": media_hash,
                  "prompt_sha256": group["prompt_token_ids_sha256"],
                  "action_sha256": group["actions"][0]["action_token_ids_sha256"],
                  "action_token_count": group["actions"][0]["action_token_count"],
                  "ends_at_eos": group["actions"][0]["action_token_ids"][-1] == EOS_TOKEN_ID}
                 for group, media_hash in zip(encoded, media, strict=True)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    print(json.dumps(preflight(args.config, rows=args.rows, plan=args.plan), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
