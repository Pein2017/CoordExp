"""Benchmark rollout-stage inference: HF (PtEngine) vs vLLM (VllmEngine).

This script focuses on the rollout stage used by `src/trainers/rollout_matching_sft.py`:
  - generate a rollout response
  - strict token-aligned parsing
  - optional matching against GT assistant_payload (for sanity + qualitative diffs)

Design goals (repo conventions):
  - config-first (YAML in `configs/`)
  - deterministic sampling (fixed seeds)
  - clear logging via `src.utils.logger.get_logger`
  - reproducible artifacts (JSON + JSONL outputs)

Example (single GPU):
  CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/ms/bin/python \
    scripts/analysis/rollout_backend_bench/benchmark_rollout_backends.py \
    --config configs/bench/rollout_backend_bench.yaml --backend both

Example (use all GPUs in parallel; recommended):
  /root/miniconda3/envs/ms/bin/python \
    scripts/analysis/rollout_backend_bench/benchmark_rollout_backends.py \
    --config configs/bench/rollout_backend_bench.yaml --multi_gpu
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.config.loader import ConfigLoader
from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.trainers.rollout_matching_sft import (
    GTObject,
    hungarian_match_maskiou,
    parse_rollout_for_matching,
    _extract_gt_objects,
    _points_from_coord_tokens,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

Backend = Literal["hf", "vllm"]


@dataclass(frozen=True)
class BenchConfig:
    model_dir: str
    train_jsonl: str
    system_prompt: Optional[str]
    user_prompt: str
    coord_tokens_enabled: bool
    coord_skip_bbox_norm: bool

    template_max_length: Optional[int]
    template_max_pixels: Optional[int]
    template_truncation_strategy: str

    num_samples: int
    max_records_to_scan: int
    max_new_tokens: int
    temperature: float
    warmup_steps: int
    repeats: int
    batch_size: int
    output_dir: str

    hf_kwargs: Mapping[str, Any]
    vllm_kwargs: Mapping[str, Any]


@dataclass
class PerSampleResult:
    line_idx: int
    image: str | None
    width: int
    height: int
    gt: List[Dict[str, Any]]

    # Raw outputs
    text: str
    token_ids: List[int]

    # Rollout parsing/matching metrics
    parse_dropped_invalid: int
    parse_dropped_ambiguous: int
    parse_truncated: bool
    valid_pred_objects: int
    matched_for_supervision: int
    gating_rejections: int

    # Decoded predicted objects in pixel space (for visualization)
    pred: List[Dict[str, Any]]


@dataclass
class RepeatMetrics:
    repeat_idx: int
    infer_time_s: float
    end_to_end_time_s: float
    samples: int
    rollouts_per_s: float
    gen_tokens: int
    gen_tokens_per_s: float
    p50_latency_s: float
    p90_latency_s: float
    peak_gpu_mem_mb: int


@dataclass
class BackendRunResult:
    backend: Backend
    physical_gpu_id: int
    seed: int
    engine_init_s: float
    repeats: List[RepeatMetrics]
    # Per-sample outputs from the *first* repeat (used for diff/vis)
    samples: List[PerSampleResult]


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark HF vs vLLM rollouts")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--base_config", default=None, help="Optional base YAML to merge")
    p.add_argument(
        "--backend",
        default="both",
        choices=["hf", "vllm", "both"],
        help="Which backend to run on this process",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override benchmark seed (otherwise config default; multi_gpu adds gpu offset).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory (otherwise config default).",
    )
    p.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Spawn one worker per GPU and aggregate results. Uses all visible GPUs.",
    )
    p.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated physical GPU ids for --multi_gpu (default: all).",
    )
    p.add_argument(
        "--per_gpu_seed_offset",
        type=int,
        default=1,
        help="Seed offset per GPU for --multi_gpu (seed + gpu_id * offset).",
    )
    p.add_argument(
        "--no_vis_jsonl",
        action="store_true",
        help="Skip writing per-sample pred JSONLs (faster, fewer artifacts).",
    )
    return p.parse_args()


def _get_physical_gpu_id() -> int:
    # We expect the driver to set CUDA_VISIBLE_DEVICES to a single physical id per worker.
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return 0
    # Common cases:
    #   "2" -> physical 2
    #   "2,3" -> treat first as monitor target (we don't support multi-device per worker here)
    head = raw.split(",")[0].strip()
    try:
        return int(head)
    except Exception:
        return 0


def _nvidia_smi_mem_used_mb(physical_gpu_id: int) -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
                "-i",
                str(int(physical_gpu_id)),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Output is a number, possibly with multiple lines if query expands (shouldn't).
        line = out.splitlines()[0].strip()
        return int(float(line))
    except Exception:
        return -1


class _GPUMemoryMonitor:
    def __init__(self, physical_gpu_id: int, *, interval_s: float = 0.05) -> None:
        import threading

        self._gpu = int(physical_gpu_id)
        self._interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_mb: int = -1
        self._last_mb: int = -1

    def start(self) -> None:
        self.peak_mb = -1
        self._last_mb = -1
        self._stop.clear()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass

    def reset_peak(self) -> None:
        self.peak_mb = self._last_mb

    def _run(self) -> None:
        while not self._stop.is_set():
            mb = _nvidia_smi_mem_used_mb(self._gpu)
            self._last_mb = mb
            if mb >= 0 and (self.peak_mb < 0 or mb > self.peak_mb):
                self.peak_mb = mb
            time.sleep(self._interval_s)


def _reservoir_sample_jsonl(
    *,
    jsonl_path: str,
    num_samples: int,
    seed: int,
    max_records_to_scan: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Reservoir-sample records from a large JSONL without loading it into RAM."""
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    k = max(1, int(num_samples))
    rng = random.Random(int(seed))

    reservoir: List[Tuple[int, Dict[str, Any]]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_records_to_scan and line_idx >= int(max_records_to_scan):
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append((line_idx, rec))
            else:
                j = rng.randint(0, seen - 1)
                if j < k:
                    reservoir[j] = (line_idx, rec)

    if not reservoir:
        raise ValueError("No valid JSONL records were sampled; check the file format")

    # Sort by line_idx for stable ordering across processes/backends.
    reservoir.sort(key=lambda t: t[0])
    return reservoir


def _to_pixel(points_norm1000: Sequence[int], width: int, height: int) -> List[int]:
    out: List[int] = []
    w = max(1, int(width))
    h = max(1, int(height))

    # Mirror core conversion: frac=v/999, then scale by (W-1)/(H-1).
    denom_x = max(1, w - 1)
    denom_y = max(1, h - 1)

    for i, v in enumerate(points_norm1000):
        vv = int(v)
        vv = 0 if vv < 0 else 999 if vv > 999 else vv
        frac = float(vv) / 999.0
        if i % 2 == 0:
            out.append(int(round(frac * denom_x)))
        else:
            out.append(int(round(frac * denom_y)))
    return out


def _gt_to_vis(
    gt_objs: Sequence[GTObject], width: int, height: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for obj in gt_objs:
        pts = _to_pixel(obj.points_norm1000, width, height)
        out.append({"type": obj.geom_type, "points": pts, "desc": obj.desc})
    return out


def _pred_from_parse_for_vis(
    *,
    tokenizer: Any,
    sample: Mapping[str, Any],
    response_token_ids: Sequence[int],
    parse,
) -> List[Dict[str, Any]]:
    # Parse per-object snippet to recover `desc` and geometry tokens; then map to pixels.
    width = int(sample.get("width") or 0)
    height = int(sample.get("height") or 0)

    # Map coord token id -> bin (0..999)
    from src.coord_tokens.codec import get_coord_token_ids

    coord_ids = get_coord_token_ids(tokenizer)
    coord_id_to_bin = {int(tok_id): int(i) for i, tok_id in enumerate(coord_ids)}

    preds: List[Dict[str, Any]] = []
    for pobj in parse.valid_objects:
        start, end = pobj.value_span
        if end <= start or end > len(parse.prefix_text) + len(parse.response_text):
            # Defensive: value_span should be within response_text; ignore if inconsistent.
            end = max(end, start)
        snippet = parse.response_text[start:end]
        try:
            parsed = json.loads(snippet)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        desc = parsed.get("desc")
        if not isinstance(desc, str):
            desc = pobj.key

        pts_norm = _points_from_coord_tokens(
            response_token_ids=response_token_ids,
            coord_token_indices=pobj.coord_token_indices,
            coord_id_to_bin=coord_id_to_bin,
        )
        if pts_norm is None:
            continue
        pts_pix = _to_pixel(pts_norm, width, height)
        preds.append({"type": pobj.geom_type, "points": pts_pix, "desc": str(desc)})
    return preds


def _load_bench_config(args: argparse.Namespace) -> BenchConfig:
    raw = ConfigLoader.load_yaml_with_extends(args.config)
    if args.base_config:
        base = ConfigLoader.load_yaml_with_extends(args.base_config)
        raw = ConfigLoader.merge_configs(base, raw)

    prompts = ConfigLoader.resolve_prompts(raw)

    model_dir = str(((raw.get("model") or {}) or {}).get("model") or "")
    if not model_dir:
        raise ValueError("config.model.model must be set to a local model directory")

    custom = raw.get("custom") or {}
    train_jsonl = str(custom.get("train_jsonl") or custom.get("jsonl") or "")
    if not train_jsonl:
        raise ValueError("custom.train_jsonl must be set in the YAML")

    template_cfg = raw.get("template") or {}
    system_prompt = template_cfg.get("system", prompts.system)
    user_prompt = str(custom.get("user_prompt") or prompts.user or "")

    truncation_strategy = str(template_cfg.get("truncation_strategy", "right"))
    max_pixels_raw = template_cfg.get("max_pixels", None)
    try:
        template_max_pixels = (
            int(max_pixels_raw) if max_pixels_raw is not None else None
        )
    except Exception:
        template_max_pixels = None

    # Prefer an explicit template.max_length, then global_max_length, then vllm.max_model_len.
    max_length_raw = template_cfg.get("max_length", None)
    if max_length_raw is None:
        max_length_raw = raw.get("global_max_length", None)

    coord_cfg = custom.get("coord_tokens") or {}
    coord_tokens_enabled = bool(coord_cfg.get("enabled", False))
    coord_skip_bbox_norm = bool(coord_cfg.get("skip_bbox_norm", True))

    extra = (custom.get("extra") or {}).get("rollout_backend_bench") or {}
    if not isinstance(extra, dict):
        raise TypeError("custom.extra.rollout_backend_bench must be a mapping")

    def _get_int(key: str, default: int) -> int:
        v = extra.get(key, default)
        try:
            return int(v)
        except Exception:
            return int(default)

    def _get_float(key: str, default: float) -> float:
        v = extra.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    num_samples = _get_int("num_samples", 32)
    max_records_to_scan = _get_int("max_records_to_scan", 0)
    max_new_tokens = _get_int("max_new_tokens", 256)
    temperature = _get_float("temperature", 0.0)
    warmup_steps = _get_int("warmup_steps", 2)
    repeats = _get_int("repeats", 3)
    batch_size = _get_int("batch_size", 1)
    output_dir = str(
        args.output_dir
        or extra.get("output_dir")
        or "output/bench/rollout_backend_bench"
    )

    hf_kwargs = extra.get("hf") or {}
    vllm_kwargs = extra.get("vllm") or {}
    if not isinstance(hf_kwargs, dict):
        hf_kwargs = {}
    if not isinstance(vllm_kwargs, dict):
        vllm_kwargs = {}

    # If still unset, fall back to vllm max_model_len (common for benchmark configs).
    if max_length_raw is None and isinstance(vllm_kwargs, Mapping):
        max_length_raw = vllm_kwargs.get("max_model_len", None)
    try:
        template_max_length = (
            int(max_length_raw) if max_length_raw is not None else None
        )
    except Exception:
        template_max_length = None

    return BenchConfig(
        model_dir=model_dir,
        train_jsonl=train_jsonl,
        system_prompt=str(system_prompt) if system_prompt is not None else None,
        user_prompt=user_prompt,
        coord_tokens_enabled=coord_tokens_enabled,
        coord_skip_bbox_norm=coord_skip_bbox_norm,
        template_max_length=template_max_length,
        template_max_pixels=template_max_pixels,
        template_truncation_strategy=truncation_strategy,
        num_samples=num_samples,
        max_records_to_scan=max_records_to_scan,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        warmup_steps=warmup_steps,
        repeats=repeats,
        batch_size=batch_size,
        output_dir=output_dir,
        hf_kwargs=hf_kwargs,
        vllm_kwargs=vllm_kwargs,
    )


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _build_infer_requests_and_gt(
    *,
    bench: BenchConfig,
    seed: int,
) -> Tuple[List[int], List[Any], List[Dict[str, Any]]]:
    """Build InferRequest objects and per-sample metadata for parsing/matching/vis."""
    # Align with sft.py: default ROOT_IMAGE_DIR to the JSONL directory if unset.
    if os.environ.get("ROOT_IMAGE_DIR") in (None, ""):
        os.environ["ROOT_IMAGE_DIR"] = str(Path(bench.train_jsonl).resolve().parent)

    sampled = _reservoir_sample_jsonl(
        jsonl_path=bench.train_jsonl,
        num_samples=bench.num_samples,
        seed=seed,
        max_records_to_scan=bench.max_records_to_scan,
    )

    builder = JSONLinesBuilder(
        user_prompt=bench.user_prompt,
        emit_norm="none",
        mode="dense",
        json_format="standard",
        coord_tokens_enabled=bench.coord_tokens_enabled,
    )

    from swift.llm import InferRequest

    line_indices: List[int] = []
    infer_requests: List[InferRequest] = []
    sample_meta: List[Dict[str, Any]] = []

    for line_idx, rec in sampled:
        merged = builder.build_many([rec])
        messages = list(merged.get("messages") or [])
        if bench.system_prompt is not None:
            messages = [
                {"role": "system", "content": str(bench.system_prompt)},
                *messages,
            ]
        # Minimal fields required by rollout parsing/matching utilities.
        sample = {
            "messages": messages,
            "assistant_payload": merged.get("assistant_payload"),
            "objects": merged.get("objects") or {},
            "width": int(rec.get("width") or 0),
            "height": int(rec.get("height") or 0),
            "image": (rec.get("images") or [None])[0],
        }
        req = InferRequest(messages=messages, objects=sample["objects"])
        line_indices.append(int(line_idx))
        infer_requests.append(req)
        sample_meta.append(sample)

    return line_indices, infer_requests, sample_meta


def _init_hf_engine(bench: BenchConfig):
    import torch
    from swift.llm import PtEngine, get_model_tokenizer
    from swift.llm.template import get_template

    model_dir = bench.model_dir
    dtype = torch.bfloat16

    # Create a template instance with our desired knobs; engine will re-init it with its own processor.
    _, proc = get_model_tokenizer(
        model_dir,
        torch_dtype=dtype,
        load_model=False,
        download_model=False,
    )
    template = get_template(
        template_type="qwen3_vl",
        processor=proc,
        default_system=bench.system_prompt,
        max_length=bench.template_max_length,
        truncation_strategy=bench.template_truncation_strategy,
        max_pixels=bench.template_max_pixels,
    )

    from src.coord_tokens.template_adapter import apply_coord_template_adapter
    from src.config.schema import CoordTokensConfig

    coord_cfg = CoordTokensConfig(
        enabled=bench.coord_tokens_enabled, skip_bbox_norm=bench.coord_skip_bbox_norm
    )
    apply_coord_template_adapter(template, coord_cfg)

    attn_impl = str((bench.hf_kwargs or {}).get("attn_impl", "flash_attention_2"))
    engine = PtEngine(
        model_dir,
        torch_dtype=dtype,
        attn_impl=attn_impl,
        load_model=True,
        download_model=False,
        template=template,
    )
    return engine


def _init_vllm_engine(bench: BenchConfig):
    import torch
    from swift.llm import VllmEngine, get_model_tokenizer
    from swift.llm.template import get_template

    model_dir = bench.model_dir
    dtype = torch.bfloat16

    vcfg = dict(bench.vllm_kwargs or {})

    # Create a template instance with our desired knobs; engine will re-init it with its own processor.
    _, proc = get_model_tokenizer(
        model_dir,
        torch_dtype=dtype,
        load_model=False,
        download_model=False,
    )
    template = get_template(
        template_type="qwen3_vl",
        processor=proc,
        default_system=bench.system_prompt,
        max_length=bench.template_max_length,
        truncation_strategy=bench.template_truncation_strategy,
        max_pixels=bench.template_max_pixels,
    )

    from src.coord_tokens.template_adapter import apply_coord_template_adapter
    from src.config.schema import CoordTokensConfig

    coord_cfg = CoordTokensConfig(
        enabled=bench.coord_tokens_enabled, skip_bbox_norm=bench.coord_skip_bbox_norm
    )
    apply_coord_template_adapter(template, coord_cfg)

    engine = VllmEngine(
        model_dir,
        torch_dtype=dtype,
        template=template,
        gpu_memory_utilization=float(vcfg.get("gpu_memory_utilization", 0.85)),
        tensor_parallel_size=int(vcfg.get("tensor_parallel_size", 1)),
        max_model_len=int(vcfg.get("max_model_len", bench.template_max_length or 4096))
        if vcfg.get("max_model_len") is not None
        or bench.template_max_length is not None
        else None,
        max_num_seqs=int(vcfg.get("max_num_seqs", 256)),
        enforce_eager=bool(vcfg.get("enforce_eager", False)),
        disable_custom_all_reduce=bool(vcfg.get("disable_custom_all_reduce", True)),
    )
    return engine


def _infer_one_pass(
    *,
    backend: Backend,
    engine: Any,
    infer_requests: Sequence[Any],
    request_config: Any,
    batch_size: int,
    sample_meta: Sequence[Mapping[str, Any]],
    collect_details: bool,
) -> Tuple[List[float], int, List[PerSampleResult]]:
    """Run inference once over all samples.

    Returns:
      (per_sample_infer_latencies_s, total_generated_tokens, per_sample_outputs)

    Note:
      `per_sample_outputs` is expensive to compute (parsing + matching); callers can
      request it only on the first repeat by calling this function and then discarding
      the list on later repeats.
    """
    latencies: List[float] = []
    results: List[PerSampleResult] = []
    total_gen_tokens = 0

    # Tokenizer is needed for strict parsing.
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(
            getattr(engine, "default_template", None), "tokenizer", None
        )
    if tokenizer is None:
        raise RuntimeError("Unable to locate tokenizer on engine")

    # Matching knobs (keep aligned with rollout trainer defaults).
    top_k = 10
    gate_thr = 0.3
    mask_res = 256
    fp_cost = 1.0
    fn_cost = 1.0

    idx = 0
    n = len(infer_requests)
    while idx < n:
        chunk = list(infer_requests[idx : idx + batch_size])
        t0 = time.perf_counter()
        outputs = engine.infer(
            list(chunk), request_config=request_config, use_tqdm=False
        )
        dt = time.perf_counter() - t0
        # Approx per-sample latency for the chunk (end-to-end on this process).
        per = dt / max(1, len(chunk))
        latencies.extend([per] * len(chunk))

        # Collect detailed per-sample outputs for the first pass only.
        for j, out in enumerate(outputs):
            # If ms-swift returns an Exception object, surface it as empty prediction.
            text = ""
            token_ids: List[int] = []
            if isinstance(out, Exception):
                text = f"<exception: {type(out).__name__}>"
                token_ids = []
            else:
                try:
                    choice = out.choices[0]
                    text = choice.message.content or ""
                    token_ids = list(choice.token_ids or [])
                except Exception:
                    text = ""
                    token_ids = []
            total_gen_tokens += int(len(token_ids))

            meta = sample_meta[idx + j]
            if collect_details:
                parse = parse_rollout_for_matching(
                    tokenizer=tokenizer, response_token_ids=token_ids
                )
                preds_for_match: List[GTObject] = []
                pred_meta = list(parse.valid_objects)

                # Map coord token id -> bin (0..999)
                from src.coord_tokens.codec import get_coord_token_ids

                coord_ids = get_coord_token_ids(tokenizer)
                coord_id_to_bin = {
                    int(tok_id): int(i) for i, tok_id in enumerate(coord_ids)
                }

                for pobj in pred_meta:
                    pts = _points_from_coord_tokens(
                        response_token_ids=parse.response_token_ids,
                        coord_token_indices=pobj.coord_token_indices,
                        coord_id_to_bin=coord_id_to_bin,
                    )
                    if pts is None:
                        continue
                    preds_for_match.append(
                        GTObject(
                            index=int(pobj.index),
                            geom_type=pobj.geom_type,
                            points_norm1000=pts,
                            desc="",
                        )
                    )

                # Extract GT and match for sanity.
                try:
                    gts = _extract_gt_objects(meta)
                except Exception:
                    gts = []
                match = hungarian_match_maskiou(
                    preds=preds_for_match,
                    gts=gts,
                    top_k=top_k,
                    gate_threshold=gate_thr,
                    mask_resolution=mask_res,
                    fp_cost=fp_cost,
                    fn_cost=fn_cost,
                )

                gt_vis = _gt_to_vis(
                    gts, int(meta.get("width") or 0), int(meta.get("height") or 0)
                )
                pred_vis = _pred_from_parse_for_vis(
                    tokenizer=tokenizer,
                    sample=meta,
                    response_token_ids=parse.response_token_ids,
                    parse=parse,
                )

                results.append(
                    PerSampleResult(
                        line_idx=-1,  # filled by caller
                        image=meta.get("image"),
                        width=int(meta.get("width") or 0),
                        height=int(meta.get("height") or 0),
                        gt=gt_vis,
                        text=str(text),
                        token_ids=[int(t) for t in token_ids],
                        parse_dropped_invalid=int(parse.dropped_invalid),
                        parse_dropped_ambiguous=int(parse.dropped_ambiguous),
                        parse_truncated=bool(parse.truncated),
                        valid_pred_objects=int(len(parse.valid_objects)),
                        matched_for_supervision=int(len(match.matched_pairs)),
                        gating_rejections=int(match.gating_rejections),
                        pred=pred_vis,
                    )
                )

        idx += batch_size

    return latencies, int(total_gen_tokens), results


def _summarize_repeat(
    *,
    repeat_idx: int,
    latencies_s: Sequence[float],
    gen_tokens: int,
    infer_time_s: float,
    end_to_end_time_s: float,
    samples: int,
    peak_mem_mb: int,
) -> RepeatMetrics:
    lats = np.asarray(list(latencies_s), dtype=np.float64)
    p50 = float(np.percentile(lats, 50)) if lats.size else 0.0
    p90 = float(np.percentile(lats, 90)) if lats.size else 0.0
    infer_s = float(infer_time_s)
    rps = float(samples / infer_s) if infer_s > 0 else 0.0
    tps = float(gen_tokens / infer_s) if infer_s > 0 else 0.0
    return RepeatMetrics(
        repeat_idx=int(repeat_idx),
        infer_time_s=float(infer_s),
        end_to_end_time_s=float(end_to_end_time_s),
        samples=int(samples),
        rollouts_per_s=float(rps),
        gen_tokens=int(gen_tokens),
        gen_tokens_per_s=float(tps),
        p50_latency_s=float(p50),
        p90_latency_s=float(p90),
        peak_gpu_mem_mb=int(peak_mem_mb),
    )


def _run_backend(
    *,
    backend: Backend,
    bench: BenchConfig,
    seed: int,
    line_indices: Sequence[int],
    infer_requests: Sequence[Any],
    sample_meta: Sequence[Mapping[str, Any]],
    write_vis_jsonl: bool,
    out_dir: Path,
) -> BackendRunResult:
    _set_all_seeds(seed)
    physical_gpu_id = _get_physical_gpu_id()
    mem_mon = _GPUMemoryMonitor(physical_gpu_id, interval_s=0.05)
    mem_mon.start()

    engine_t0 = time.perf_counter()
    if backend == "hf":
        engine = _init_hf_engine(bench)
    else:
        engine = _init_vllm_engine(bench)
    engine_init_s = time.perf_counter() - engine_t0

    from swift.llm.infer.protocol import RequestConfig

    req_cfg = RequestConfig(
        max_tokens=int(bench.max_new_tokens),
        temperature=float(bench.temperature),
        num_beams=1,
        seed=int(seed),
        stream=False,
        return_details=True,
    )

    # Warmup (excluded from timing).
    warmup = max(0, int(bench.warmup_steps))
    if warmup > 0:
        logger.info(
            "[%s] warmup: %s steps (batch_size=%s)", backend, warmup, bench.batch_size
        )
    for _ in range(warmup):
        _ = engine.infer([infer_requests[0]], request_config=req_cfg, use_tqdm=False)

    repeats: List[RepeatMetrics] = []
    first_pass_samples: List[PerSampleResult] = []

    for rep in range(int(max(1, bench.repeats))):
        mem_mon.reset_peak()
        t0 = time.perf_counter()
        collect_details = rep == 0
        latencies, gen_tokens, sample_results = _infer_one_pass(
            backend=backend,
            engine=engine,
            infer_requests=infer_requests,
            request_config=req_cfg,
            batch_size=max(1, int(bench.batch_size)),
            sample_meta=sample_meta,
            collect_details=collect_details,
        )
        end_to_end = time.perf_counter() - t0
        infer_time_s = (
            float(np.sum(np.asarray(latencies, dtype=np.float64))) if latencies else 0.0
        )

        # Fill line_idx for stable backend merges.
        if sample_results:
            for i, sr in enumerate(sample_results):
                sr.line_idx = int(line_indices[i])

        # Even if we skip per-sample parsing/matching on later repeats, we still benchmark the
        # same number of rollout generations. `sample_results` is only populated when
        # collect_details=True, so never use it to derive the sample count.
        sample_count = len(infer_requests)

        rep_metrics = _summarize_repeat(
            repeat_idx=rep,
            latencies_s=latencies,
            gen_tokens=gen_tokens,
            infer_time_s=infer_time_s,
            end_to_end_time_s=end_to_end,
            samples=sample_count,
            peak_mem_mb=mem_mon.peak_mb,
        )
        repeats.append(rep_metrics)
        if rep == 0:
            first_pass_samples = sample_results
        logger.info(
            "[%s] repeat=%s: rollouts/s=%.3f p50=%.3fs p90=%.3fs peak_mem=%sMB",
            backend,
            rep,
            rep_metrics.rollouts_per_s,
            rep_metrics.p50_latency_s,
            rep_metrics.p90_latency_s,
            rep_metrics.peak_gpu_mem_mb,
        )

    mem_mon.stop()

    # Write per-sample JSONL for visualization/diff (first pass only).
    if write_vis_jsonl:
        pred_jsonl = out_dir / f"pred_{backend}.jsonl"
        with pred_jsonl.open("w", encoding="utf-8") as f:
            for sr in first_pass_samples:
                f.write(
                    json.dumps(
                        {
                            "line_idx": sr.line_idx,
                            "image": sr.image,
                            "width": sr.width,
                            "height": sr.height,
                            "gt": sr.gt,
                            "pred": sr.pred,
                            "errors": [],
                            "backend": backend,
                            "text": sr.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info("[%s] wrote %s", backend, pred_jsonl)

    # Best-effort cleanup (process exit is the strongest cleanup, but be polite).
    try:
        del engine
    except Exception:
        pass
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    return BackendRunResult(
        backend=backend,
        physical_gpu_id=int(physical_gpu_id),
        seed=int(seed),
        engine_init_s=float(engine_init_s),
        repeats=repeats,
        samples=first_pass_samples,
    )


def _merge_backend_jsonl(hf_jsonl: Path, vllm_jsonl: Path, out_path: Path) -> None:
    def _load(path: Path) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[int(rec["line_idx"])] = rec
        return out

    hf = _load(hf_jsonl)
    vv = _load(vllm_jsonl)
    keys = sorted(set(hf.keys()) & set(vv.keys()))
    with out_path.open("w", encoding="utf-8") as f:
        for k in keys:
            rec = {
                "line_idx": k,
                "image": hf[k].get("image") or vv[k].get("image"),
                "width": hf[k].get("width") or vv[k].get("width"),
                "height": hf[k].get("height") or vv[k].get("height"),
                "gt": hf[k].get("gt") or [],
                "pred_hf": hf[k].get("pred") or [],
                "pred_vllm": vv[k].get("pred") or [],
                "hf_text": hf[k].get("text") or "",
                "vllm_text": vv[k].get("text") or "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _aggregate_and_report(run_dir: Path) -> None:
    results = []
    for p in sorted(run_dir.glob("result_*.json")):
        try:
            results.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not results:
        logger.warning("No result_*.json found under %s", run_dir)
        return

    # Aggregate repeats across all workers.
    by_backend: Dict[str, List[Dict[str, Any]]] = {"hf": [], "vllm": []}
    for r in results:
        backend = r.get("backend")
        if backend in by_backend:
            by_backend[backend].append(r)

    def _flatten_metrics(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flat: List[Dict[str, Any]] = []
        for item in items:
            for rep in item.get("repeats", []) or []:
                flat.append(rep)
        return flat

    def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
        v = np.asarray(list(values), dtype=np.float64)
        if v.size == 0:
            return 0.0, 0.0
        return float(v.mean()), float(v.std(ddof=1) if v.size > 1 else 0.0)

    report: Dict[str, Any] = {"run_dir": str(run_dir), "backends": {}}
    for backend, items in by_backend.items():
        reps = _flatten_metrics(items)
        rps = [float(x.get("rollouts_per_s") or 0.0) for x in reps]
        p50 = [float(x.get("p50_latency_s") or 0.0) for x in reps]
        p90 = [float(x.get("p90_latency_s") or 0.0) for x in reps]
        mem = [float(x.get("peak_gpu_mem_mb") or 0.0) for x in reps]
        mean_rps, std_rps = _mean_std(rps)
        mean_p50, std_p50 = _mean_std(p50)
        mean_p90, std_p90 = _mean_std(p90)
        mean_mem, std_mem = _mean_std(mem)
        report["backends"][backend] = {
            "n_workers": len(items),
            "n_repeats_total": len(reps),
            "rollouts_per_s_mean": mean_rps,
            "rollouts_per_s_std": std_rps,
            "p50_latency_s_mean": mean_p50,
            "p50_latency_s_std": std_p50,
            "p90_latency_s_mean": mean_p90,
            "p90_latency_s_std": std_p90,
            "peak_gpu_mem_mb_mean": mean_mem,
            "peak_gpu_mem_mb_std": std_mem,
        }

    # Compute improvement vs HF.
    hf_rps = report["backends"].get("hf", {}).get("rollouts_per_s_mean", 0.0)
    vv_rps = report["backends"].get("vllm", {}).get("rollouts_per_s_mean", 0.0)
    speedup = (vv_rps / hf_rps) if hf_rps > 0 else 0.0
    report["vllm_vs_hf_rollouts_per_s_speedup"] = speedup

    _json_dump(run_dir / "aggregate_report.json", report)
    logger.info("Aggregate report written: %s", run_dir / "aggregate_report.json")
    logger.info("vLLM vs HF speedup (rollouts/s mean): %.3fx", speedup)


def _run_single_process(args: argparse.Namespace) -> int:
    bench = _load_bench_config(args)

    seed = int(args.seed) if args.seed is not None else int(17)

    # The config seed lives under extra; use that as the default.
    # (args.seed overrides)
    raw = ConfigLoader.load_yaml_with_extends(args.config)
    extra = ((raw.get("custom") or {}).get("extra") or {}).get(
        "rollout_backend_bench"
    ) or {}
    if isinstance(extra, dict) and args.seed is None:
        try:
            seed = int(extra.get("seed", seed))
        except Exception:
            pass

    _set_all_seeds(seed)
    line_indices, infer_requests, sample_meta = _build_infer_requests_and_gt(
        bench=bench, seed=seed
    )
    # Attach line_idx to meta for matching helpers.
    for i, meta in enumerate(sample_meta):
        meta["line_idx"] = int(line_indices[i])

    # If output_dir is explicitly provided (driver mode), treat it as the run directory.
    explicit_out_dir = args.output_dir is not None
    run_dir = (
        Path(bench.output_dir)
        if explicit_out_dir
        else (Path(bench.output_dir) / _now_tag())
    )
    _safe_mkdir(run_dir)

    write_vis_jsonl = not bool(args.no_vis_jsonl)

    # Materialize a run id to keep multi-process outputs distinct.
    physical_gpu_id = _get_physical_gpu_id()
    run_id = f"gpu{physical_gpu_id}_seed{seed}"

    backends: List[Backend]
    if args.backend == "both":
        backends = ["hf", "vllm"]
    else:
        backends = [args.backend]  # type: ignore[list-item]

    results: List[BackendRunResult] = []
    for b in backends:
        out_dir = run_dir / f"{run_id}_{b}"
        _safe_mkdir(out_dir)
        logger.info(
            "Running backend=%s on physical_gpu=%s (seed=%s)", b, physical_gpu_id, seed
        )
        res = _run_backend(
            backend=b,
            bench=bench,
            seed=seed,
            line_indices=line_indices,
            infer_requests=infer_requests,
            sample_meta=sample_meta,
            write_vis_jsonl=write_vis_jsonl,
            out_dir=out_dir,
        )
        results.append(res)

        # Write result JSON.
        payload = dataclasses.asdict(res)
        _json_dump(run_dir / f"result_{run_id}_{b}.json", payload)

    # If both ran, merge the per-sample pred JSONLs into a compare file.
    if write_vis_jsonl and args.backend == "both":
        hf_jsonl = run_dir / f"{run_id}_hf" / "pred_hf.jsonl"
        vv_jsonl = run_dir / f"{run_id}_vllm" / "pred_vllm.jsonl"
        if hf_jsonl.is_file() and vv_jsonl.is_file():
            merged = run_dir / f"compare_{run_id}.jsonl"
            _merge_backend_jsonl(hf_jsonl, vv_jsonl, merged)
            logger.info("Wrote compare JSONL: %s", merged)

    logger.info("Run dir: %s", run_dir)
    return 0


def _run_multi_gpu(args: argparse.Namespace) -> int:
    bench = _load_bench_config(args)
    run_dir = Path(bench.output_dir) / _now_tag()
    _safe_mkdir(run_dir)

    # Determine physical GPUs to use.
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    else:
        # Fall back to nvidia-smi list.
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
            gpu_ids = [
                i
                for i, _ in enumerate(out.splitlines())
                if _.strip().startswith("GPU ")
            ]
        except Exception:
            gpu_ids = [0]
    gpu_ids = [int(i) for i in gpu_ids]
    logger.info("multi_gpu: using physical GPUs: %s", gpu_ids)

    # Config seed.
    raw = ConfigLoader.load_yaml_with_extends(args.config)
    extra = ((raw.get("custom") or {}).get("extra") or {}).get(
        "rollout_backend_bench"
    ) or {}
    base_seed = (
        int(args.seed)
        if args.seed is not None
        else int(extra.get("seed", 17) if isinstance(extra, dict) else 17)
    )
    per_gpu_offset = int(args.per_gpu_seed_offset)

    # Spawn one subprocess per GPU per backend (two waves) to keep cleanup simple.
    script = Path(__file__).resolve()
    python = sys.executable

    procs: List[subprocess.Popen] = []
    task_specs: List[Tuple[int, Backend, int]] = []  # (gpu, backend, seed)
    for gpu in gpu_ids:
        seed = base_seed + int(gpu) * per_gpu_offset
        task_specs.append((gpu, "hf", seed))
    for gpu in gpu_ids:
        seed = base_seed + int(gpu) * per_gpu_offset
        task_specs.append((gpu, "vllm", seed))

    def _run_wave(wave: List[Tuple[int, Backend, int]]) -> None:
        nonlocal procs
        procs = []
        for gpu, backend, seed in wave:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
            cmd = [
                python,
                str(script),
                "--config",
                args.config,
                "--backend",
                backend,
                "--seed",
                str(int(seed)),
                "--output_dir",
                str(run_dir),
            ]
            if args.base_config:
                cmd.extend(["--base_config", args.base_config])
            if args.no_vis_jsonl:
                cmd.append("--no_vis_jsonl")
            logger.info("spawn: gpu=%s backend=%s seed=%s", gpu, backend, seed)
            procs.append(subprocess.Popen(cmd, env=env, cwd=str(Path.cwd())))
        # Wait for all in wave.
        for p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Subprocess failed with code {rc}")

    # Run at most len(gpu_ids) processes at once (one per GPU).
    wave1 = task_specs[: len(gpu_ids)]
    wave2 = task_specs[len(gpu_ids) : 2 * len(gpu_ids)]
    _run_wave(wave1)
    _run_wave(wave2)

    # Merge per-GPU compare JSONLs when predictions are enabled.
    if not args.no_vis_jsonl:
        for gpu in gpu_ids:
            seed = base_seed + int(gpu) * per_gpu_offset
            run_id = f"gpu{int(gpu)}_seed{int(seed)}"
            hf_jsonl = run_dir / f"{run_id}_hf" / "pred_hf.jsonl"
            vv_jsonl = run_dir / f"{run_id}_vllm" / "pred_vllm.jsonl"
            if hf_jsonl.is_file() and vv_jsonl.is_file():
                merged = run_dir / f"compare_{run_id}.jsonl"
                _merge_backend_jsonl(hf_jsonl, vv_jsonl, merged)
                logger.info("merged compare JSONL: %s", merged)

    _aggregate_and_report(run_dir)
    logger.info("multi_gpu run dir: %s", run_dir)
    return 0


def main() -> int:
    args = _parse_args()
    if args.multi_gpu:
        return _run_multi_gpu(args)
    return _run_single_process(args)


if __name__ == "__main__":
    raise SystemExit(main())
