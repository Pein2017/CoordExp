from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator")
CODETR = ROOT / "codetr_probe.py"
SEMANTIC = ROOT / "semantic_probe.py"
RULE = ROOT / "profile_rule.py"
SPEC = ROOT / "selected-candidate-v1.json"
CODETR_PYTHONPATH = "/data/CoordExp/external/Co-DETR"
FROZEN_HASHES = {
    CODETR: "d8bef8c851fdb774d8bd9cc570758a071a24a80db266d87f54c955510ef787a9",
    SEMANTIC: "71a618e8118e38912d2b782b1db718b4143e2a0294bd626dd937408ee33baffd",
    RULE: "f0029189880518b29093fcc90a64485d48120b85914c9690d07ef7f4272c2216",
    SPEC: "f2344113da286c6c271325fd3f2970eeea01a423d98f62d592c99ad2ffceaace",
    ROOT / "reground-dev-v1/run.py": "a1920b89c4c225736cbfcd9b855ecf0b9f0e00ced97589611d88849705563d90",
    ROOT / "reground-dev-v1/requests.jsonl": "af6625ad50166b1418167606c676d2f11acde262278c09d59636ee8d059d7a0d",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/config.json"): "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/model.safetensors.index.json"): "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/model-00001-of-00004.safetensors"): "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/model-00002-of-00004.safetensors"): "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/model-00003-of-00004.safetensors"): "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/model-00004-of-00004.safetensors"): "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
    Path("/data/CoordExp/external/Co-DETR/models/co_dino_5scale_vit_large_coco.pth"): "733d2ccde180a55151a68a6cab7c9f42b117d24d38d6197b37caf3189243256c",
    Path("/data/CoordExp/external/Co-DETR/projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py"): "f53d4e7f009d39c94ef8456e3f44514408c4225a3027530965d1692e3ffa8077",
    Path("/data/CoordExp/external/Co-DETR/tools/codetr_infer_human_refined12.py"): "6a89c4cf51fd742e0bb7bc04a784b35f624618e74be866cbe6fb77a383bfba08",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/tokenizer.json"): "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/merges.txt"): "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/tokenizer_config.json"): "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/generation_config.json"): "8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/preprocessor_config.json"): "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/video_preprocessor_config.json"): "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/vocab.json"): "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/configuration.json"): "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9",
    Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct/chat_template.json"): "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
}
REQUIRED = {"case_id", "image_id", "image_path", "bbox", "width", "height", "category"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty candidates file")
    if len(rows) > 100:
        raise ValueError("bounded profile accepts at most100 candidates per invocation")
    seen, filenames = set(), set()
    for row in rows:
        if not isinstance(row, dict) or not REQUIRED.issubset(row):
            raise ValueError(f"candidate missing fields: {REQUIRED - set(row) if isinstance(row, dict) else REQUIRED}")
        case = row["case_id"]
        if not isinstance(case, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", case) or case in seen:
            raise ValueError(f"invalid or duplicate case_id: {case!r}")
        seen.add(case)
        filename = case.replace(":", "_")
        if filename in filenames:
            raise ValueError(f"case_id filename collision: {case}")
        filenames.add(filename)
        if not all(isinstance(row[k], str) and row[k] for k in ("image_id", "image_path", "category")):
            raise ValueError(f"candidate string fields invalid: {case}")
        if type(row["width"]) is not int or type(row["height"]) is not int or row["width"] <= 0 or row["height"] <= 0:
            raise ValueError(f"candidate dimensions invalid: {case}")
        box = row["bbox"]
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
            raise ValueError(f"candidate bbox invalid: {case}")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
            raise ValueError(f"candidate bbox out of bounds: {case}")
        image = Path(row["image_path"])
        if not image.is_file():
            raise ValueError(f"missing image: {case}")
        if "image_sha256" in row and sha(image) != row["image_sha256"]:
            raise ValueError(f"declared image hash mismatch: {case}")
        with Image.open(image) as im:
            if im.size != (row["width"], row["height"]):
                raise ValueError(f"image dimensions mismatch: {case}")
    return rows


def _check_sources() -> dict[str, str]:
    hashes = {}
    for path, expected in FROZEN_HASHES.items():
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"frozen source hash mismatch: {path}")
        hashes[path.name] = actual
    return hashes


def _run_stage(name: str, command: list[str], env: dict[str, str], out: Path, receipts: list[dict]) -> None:
    command = ["timeout", "--kill-after=30s", "1800s", *command]
    stdout, stderr = out / f"{name}.stdout.log", out / f"{name}.stderr.log"
    started = time.perf_counter()
    try:
        with stdout.open("x") as stdout_handle, stderr.open("x") as stderr_handle:
            result = subprocess.run(command, env=env, text=True, stdout=stdout_handle,
                                    stderr=stderr_handle, check=False)
        exit_code = result.returncode
    except BaseException as exc:
        exit_code = None
        (out / f"{name}.exception.txt").write_text(repr(exc) + "\n")
    receipt = {"name": name, "command": command,
               "environment": {key: env[key] for key in
                               ("CUDA_VISIBLE_DEVICES", "CODETR_ADMIT_IOU", "PYTHONPATH", "VLLM_WORKER_MULTIPROC_METHOD",
                                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_OFFLINE") if key in env},
               "exit_code": exit_code,
               "wall_seconds": time.perf_counter() - started,
               "stdout_path": str(stdout), "stderr_path": str(stderr)}
    receipts.append(receipt)
    if exit_code != 0:
        raise RuntimeError(f"{name} failed with exit {exit_code}")


def _load_rule():
    spec = importlib.util.spec_from_file_location("frozen_profile_rule", RULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run(candidates_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    candidates_path, output_dir = Path(candidates_path), Path(output_dir)
    started = time.perf_counter()
    receipts: list[dict] = []
    created = False
    try:
        source_hashes = _check_sources()
        candidate_hash = sha(candidates_path)
        rows = _validate(candidates_path)
        output_dir.mkdir(parents=True, exist_ok=False)
        created = True
        detector_dir, semantic_dir = output_dir / "detector", output_dir / "semantic"
        semantic_input = output_dir / "semantic-eligible.jsonl"
        detector_env = os.environ.copy()
        detector_env.update({"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": CODETR_PYTHONPATH,
                             "CODETR_ADMIT_IOU": "0.75"})
        _run_stage("detector", ["conda", "run", "-n", "mmdet", "python", str(CODETR),
                                 str(candidates_path), str(detector_dir), "context"], detector_env,
                   output_dir, receipts)
        detector_config = json.loads((detector_dir / "config.json").read_text())
        if sha(candidates_path) != candidate_hash or detector_config.get("source_sha256") != candidate_hash or detector_config.get("code_sha256") != source_hashes[CODETR.name]:
            raise RuntimeError("detector source/code identity mismatch")
        decisions = json.loads((detector_dir / "decisions.json").read_text())
        if [d.get("case_id") for d in decisions] != [r["case_id"] for r in rows]:
            raise RuntimeError("detector case identity mismatch")
        eligible = [row for row, decision in zip(rows, decisions) if decision.get("decision") == "accept"]
        semantic_input.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eligible))
        semantic_env = os.environ.copy()
        semantic_env.update({"CUDA_VISIBLE_DEVICES": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_OFFLINE": "1"})
        _run_stage("semantic", ["conda", "run", "-n", "ms", "python", str(SEMANTIC),
                                 str(semantic_input), str(semantic_dir)], semantic_env, output_dir, receipts)
        semantic_config = json.loads((semantic_dir / "config.json").read_text())
        if semantic_config.get("source_sha256") != sha(semantic_input) or semantic_config.get("code_sha256") != source_hashes[SEMANTIC.name]:
            raise RuntimeError("semantic source/code identity mismatch")
        observations: dict[str, dict[str, dict]] = {r["case_id"]: {} for r in eligible}
        for line in (semantic_dir / "responses.jsonl").read_text().splitlines():
            item = json.loads(line)
            case, form = item.get("case_id"), item.get("form")
            if case not in observations or form not in {"full", "fixed_context"} or form in observations[case]:
                raise RuntimeError("missing, duplicate, or extraneous semantic form")
            error, parsed, finish = item.get("error"), item.get("parsed"), item.get("finish_reason")
            valid = error is None and finish == "stop" and isinstance(parsed, dict) and isinstance(parsed.get("category"), str)
            abstention = parsed is None and (
                (error in {"unresolved", "invalid_schema"} and finish == "stop")
                or (error == "not_stopped" and finish != "stop"))
            if not (valid or abstention):
                raise RuntimeError(f"invalid semantic receipt: {case}/{form}")
            observations[case][form] = item
        if any(set(forms) != {"full", "fixed_context"} for forms in observations.values()):
            raise RuntimeError("missing semantic form")
        rule = _load_rule()
        final = []
        for row, detector in zip(rows, decisions):
            judged = rule.judge(row, detector, list(observations.get(row["case_id"], {}).values()))
            judged["detector_decision"] = detector["decision"]
            final.append(judged)
        decisions_path = output_dir / "decisions.jsonl"
        decisions_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in final))
        summary = {"status": "complete", "cases": len(rows), "detector_eligible": len(eligible),
                   "decisions_path": str(decisions_path), "source_path": str(candidates_path.resolve()),
                   "source_sha256": sha(candidates_path), "frozen_source_hashes": source_hashes,
                   "stage_receipts": receipts, "elapsed_seconds": time.perf_counter() - started,
                   "scope": "screening-only candidate profile; no GT or reference labels"}
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        (output_dir / "run-receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
        return {"summary": str(summary_path), "decisions": str(decisions_path)}
    except BaseException as exc:
        if created:
            receipt = {"status": "failed", "error": repr(exc), "stage_receipts": receipts,
                       "elapsed_seconds": time.perf_counter() - started}
            (output_dir / "run-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise
