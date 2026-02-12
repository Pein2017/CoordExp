"""Centralized inference engine for CoordExp.

Features
--------
- Single entrypoint that works for coord-token (normalized) and pure-text
  checkpoints via an explicit ``mode`` switch.
- Standardized ``gt_vs_pred.jsonl`` with pixel-space geometries for both GT and
  predictions; polygons are preserved.
- Per-sample error reporting plus run-level counters and summary JSON.
- Deterministic generation when ``--seed`` is provided (torch + CUDA seeding).

Schema (per line of ``gt_vs_pred.jsonl``)
-----------------------------------
```
{
  "image": str,              # relative or absolute path
  "width": int,
  "height": int,
  "mode": "coord" | "text",
  "coord_mode": "norm1000" | "pixel" | null,  # optional trace/debug
  "gt": [ {"type","points","desc","score"} ],
  "pred": [ {"type","points","desc","score"} ],
  "raw_output_json": dict | null,  # parsed prediction dict (best-effort)
  "raw_special_tokens": [str],     # e.g. <|im_end|>, <|coord_123|>, ...
  "raw_ends_with_im_end": bool,
  "errors": ["..."]         # empty when none
}
```

Summary (``summary.json``)
-------------------------------
```
{
  "mode": "coord" | "text",
  "total_read": int,
  "total_emitted": int,
  "counters": {"invalid_json": 0, ...},
  "error_codes": ["invalid_coord", ...]
}
```

Notes
-----
- ``gt`` and ``pred`` points are always pixel-space when emitted.
- ``points_text`` mirrors the pixel coords as a text string to unify downstream
  consumers that expect pure-text geometry.
- Polygons are kept verbatim (single ring) for COCO segmentation.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.common.coord_standardizer import CoordinateStandardizer
from src.common.geometry import flatten_points, has_coord_tokens
from src.config.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.common.prediction_parsing import extract_special_tokens, load_prediction_dict
from src.common.paths import resolve_image_path_best_effort
from src.utils import get_logger

# Map fine-grained error tags to canonical counter buckets.
ERROR_CANONICAL = {
    "geometry_keys": "invalid_geometry",
    "geometry_points": "invalid_geometry",
    "geometry_kind": "invalid_geometry",
    "bbox_points": "invalid_geometry",
    "poly_points": "invalid_geometry",
    "degenerate": "invalid_geometry",
    "coord_parse": "invalid_coord",
    "coord_range": "invalid_coord",
    "mode_gt_mismatch": "mode_gt_mismatch",
    "size_mismatch": "size_mismatch",
    "empty_pred": "empty_pred",
    "generation_failed": "generation_failed",
    "image_load_failed": "image_load_failed",
    "multi_image_not_supported": "multi_image_not_supported",
}


@dataclass
class GenerationConfig:
    temperature: float = 0.01
    top_p: float = 0.95
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.05
    # Number of samples to decode per forward pass (HF) / per client micro-batch (vLLM).
    # Keep at 1 by default to preserve memory headroom.
    batch_size: int = 1
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    text: str = ""
    error: Optional[Exception] = None


@dataclass
class InferenceConfig:
    gt_jsonl: str
    model_checkpoint: str
    mode: Literal["coord", "text", "auto"]
    pred_coord_mode: Literal["auto", "norm1000", "pixel"] = "auto"

    # Canonical unified artifact names (can be overridden by pipeline runner).
    out_path: str = "gt_vs_pred.jsonl"
    summary_path: Optional[str] = None

    # Optional pipeline-resolved root image dir used by infer/eval/vis for a
    # single deterministic image-resolution decision.
    root_image_dir: Optional[str] = None

    device: str = "cuda:0"
    limit: int = 0

    backend_type: Literal["hf", "vllm"] = "hf"
    backend: Dict[str, Any] = field(default_factory=dict)

    # When mode=auto, how many GT records to scan (see OpenSpec for rules).
    detect_samples: int = 128


def detect_mode_from_gt(
    gt_jsonl: str,
    *,
    sample_size: int = 128,
) -> Tuple[Literal["coord", "text"], str]:
    """Deterministically resolve coord vs text from GT JSONL (OpenSpec).

    Scan the first N *valid* records:
    - ignore invalid JSON
    - ignore records with no objects
    - ignore records without valid int width/height

    Resolution:
    - coord if any coord tokens are found in any geometry
    - coord if any numeric coordinate exceeds max(width, height)
    - else text
    """

    checked = 0
    path = Path(gt_jsonl)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if checked >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue

            width = rec.get("width")
            height = rec.get("height")
            try:
                width_i = int(width)
                height_i = int(height)
            except Exception:
                continue
            if width_i <= 0 or height_i <= 0:
                continue

            objs = rec.get("objects") or rec.get("gt") or []
            if not isinstance(objs, list) or len(objs) == 0:
                continue

            max_dim = max(width_i, height_i)
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                pts_raw = flatten_points(
                    obj.get("bbox_2d") or obj.get("poly") or obj.get("points") or []
                )
                if not pts_raw:
                    continue

                if has_coord_tokens(pts_raw):
                    return "coord", "coord_tokens_found"

                # Only consider numeric coordinates for bounds check.
                numeric = [v for v in pts_raw if isinstance(v, (int, float))]
                if numeric and max(numeric) > max_dim:
                    return "coord", "points_exceed_image"

            checked += 1

    if checked == 0:
        return "text", "no_valid_records"

    return "text", "within_image_bounds"


class RunCounters:
    """Aggregated counters for run-level summary."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self.error_codes: set[str] = set()
        self.total_read: int = 0
        self.total_emitted: int = 0

    def add(self, code: str) -> None:
        self.counts[code] = self.counts.get(code, 0) + 1
        self.error_codes.add(code)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "counters": self.counts,
            "error_codes": sorted(self.error_codes),
            "total_read": self.total_read,
            "total_emitted": self.total_emitted,
        }


class InferenceEngine:
    def __init__(
        self,
        cfg: InferenceConfig,
        gen_cfg: GenerationConfig,
        *,
        logger=None,
    ) -> None:
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.logger = logger or get_logger(__name__)

        self.requested_mode = cfg.mode
        self.resolved_mode = cfg.mode
        self.mode_reason: Optional[str] = None
        if cfg.mode == "auto":
            self.resolved_mode, self.mode_reason = detect_mode_from_gt(
                cfg.gt_jsonl, sample_size=int(cfg.detect_samples or 128)
            )

        # Shared parser/standardizer: always emit pixel-space points.
        self.coord = CoordinateStandardizer(
            self.resolved_mode, pred_coord_mode=cfg.pred_coord_mode
        )

        self.processor: AutoProcessor | None = None
        self.model: Qwen3VLForConditionalGeneration | None = None
        self.vllm_llm: Any | None = None
        self.attn_implementation_requested: Optional[str] = None
        self.attn_implementation_selected: Optional[str] = None

    def _vllm_mode(self) -> str:
        mode_raw = (self.cfg.backend or {}).get("mode", "server")
        mode = str(mode_raw or "server").strip().lower()
        return "local" if mode == "local" else "server"

    def _load_vllm_local(self) -> None:
        if self.vllm_llm is not None:
            return

        # vLLM's CLI defaults to "spawn" for safety, but the library API does not.
        # Without this, using vLLM inside a process that already touched CUDA can fail with:
        #   "Cannot re-initialize CUDA in forked subprocess"
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        try:
            from vllm import LLM
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "vLLM local backend requires the 'vllm' package. Install it in the ms env, or set infer.backend.type=hf."
            ) from exc

        model = str(self.cfg.backend.get("model") or self.cfg.model_checkpoint).strip()
        if not model:
            raise RuntimeError(
                "infer.backend.model (or infer.model_checkpoint) is required for vLLM local mode"
            )

        # Reuse server_options-style knobs when present for reproducibility.
        server_opts = self.cfg.backend.get("server_options") or {}
        allowed_local_media_path = str(
            self.cfg.root_image_dir
            or os.environ.get("ROOT_IMAGE_DIR")
            or Path(self.cfg.gt_jsonl).parent.resolve()
        )

        kwargs: Dict[str, Any] = {}
        try:
            tp = server_opts.get("vllm_tensor_parallel_size", None)
            if tp is not None:
                kwargs["tensor_parallel_size"] = int(tp)
        except Exception:
            pass
        try:
            util = server_opts.get("vllm_gpu_memory_utilization", None)
            if util is not None:
                kwargs["gpu_memory_utilization"] = float(util)
        except Exception:
            pass
        # `max_model_len` is a vLLM kwarg (mirrors --max-model-len on the server).
        try:
            mml = server_opts.get("vllm_max_model_len", None)
            if mml is not None:
                kwargs["max_model_len"] = int(mml)
        except Exception:
            pass

        # Note: local vLLM does not guarantee single OS process (it may manage workers
        # internally), but avoids running a separate HTTP server.
        self.vllm_llm = LLM(
            model=model,
            trust_remote_code=True,
            allowed_local_media_path=str(allowed_local_media_path or ""),
            seed=int(self.gen_cfg.seed) if self.gen_cfg.seed is not None else None,
            **kwargs,
        )

    def _vllm_sampling_params(self):
        try:
            from vllm import SamplingParams
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "vLLM backend requires the 'vllm' package. Install it in the ms env, or set infer.backend.type=hf."
            ) from exc

        return SamplingParams(
            temperature=float(self.gen_cfg.temperature),
            top_p=float(self.gen_cfg.top_p),
            max_tokens=int(self.gen_cfg.max_new_tokens),
            repetition_penalty=float(self.gen_cfg.repetition_penalty or 1.0),
            seed=int(self.gen_cfg.seed) if self.gen_cfg.seed is not None else None,
        )

    def _seed(self) -> None:
        if self.gen_cfg.seed is None:
            return

        seed = int(self.gen_cfg.seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Best-effort determinism for HF generation.
        # (vLLM backend does not guarantee byte-identical outputs.)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

    def _validate_vllm_backend(self) -> None:
        """Fail fast on global vLLM backend misconfiguration/unavailability."""
        if self._vllm_mode() == "local":
            # Local mode bypasses the HTTP connectivity checks.
            return

        base_url = str(
            self.cfg.backend.get("base_url") or os.environ.get("VLLM_BASE_URL") or ""
        ).strip()
        if not base_url:
            raise RuntimeError(
                "infer.backend.type=vllm requires infer.backend.base_url (or env VLLM_BASE_URL) when backend.mode=server. "
                "To run without a server, set infer.backend.mode=local. To disable vLLM, set infer.backend.type=hf."
            )

        try:
            import requests
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "vLLM backend requires the 'requests' package. Install it in the ms env, or set infer.backend.type=hf."
            ) from exc

        # Best-effort preflight connectivity check to avoid per-sample generation_failed.
        # We keep it lightweight and do not require any specific response schema.
        timeout_s = float(self.cfg.backend.get("timeout_s", 3.0))
        root = base_url.rstrip("/")
        models_url = (
            (root + "/models") if root.endswith("/v1") else (root + "/v1/models")
        )
        try:
            resp = requests.get(models_url, timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to reach vLLM server for infer.backend.type=vllm. "
                f"Tried GET {models_url}. To disable vLLM, set infer.backend.type=hf."
            ) from exc
        if int(getattr(resp, "status_code", 0) or 0) >= 400:
            raise RuntimeError(
                "vLLM server preflight check failed for infer.backend.type=vllm. "
                f"GET {models_url} returned status={resp.status_code}. To disable vLLM, set infer.backend.type=hf."
            )

    def load_model(self) -> None:
        backend = str(self.cfg.backend_type).lower().strip()

        # HF backend loads model+processor. For vLLM we support two modes:
        # - server: OpenAI-compatible HTTP server (default)
        # - local: in-process vLLM Python API (no HTTP server)
        if backend == "vllm":
            self._seed()
            if self._vllm_mode() == "local":
                self._load_vllm_local()
                return
            self._validate_vllm_backend()
            return

        if self.model is not None:
            return

        self._seed()

        attn_requested_raw = (self.cfg.backend or {}).get("attn_implementation")
        attn_requested = str(attn_requested_raw or "").strip()
        if not attn_requested or attn_requested.lower() == "auto":
            device = str(self.cfg.device or "").lower()
            if "cuda" in device and torch.cuda.is_available():
                attn_requested = "flash_attention_2"
            else:
                attn_requested = "sdpa"

        attn_requested = attn_requested.lower()
        self.attn_implementation_requested = attn_requested

        candidates: List[str] = []
        for cand in [attn_requested, "flash_attention_2", "sdpa", "eager"]:
            c = str(cand).strip().lower()
            if c and c not in candidates:
                candidates.append(c)

        last_exc: Exception | None = None
        errors: List[str] = []
        for idx, cand in enumerate(candidates):
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.cfg.model_checkpoint,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=cand,
                )
                self.model = model.to(self.cfg.device)
                self.model.eval()
                self.attn_implementation_selected = cand
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                errors.append(f"{cand}: {type(exc).__name__}: {exc}")
                if idx == 0 and len(candidates) > 1:
                    self.logger.warning(
                        "HF attention backend '%s' unavailable; falling back. Error: %s",
                        cand,
                        exc,
                    )

                # Best-effort cleanup between attempts.
                try:
                    import gc

                    gc.collect()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        if self.model is None:
            raise RuntimeError(
                "Failed to load HF model with any attention backend. "
                f"candidates={candidates} errors={errors[:3]}"
            ) from last_exc

        if self.attn_implementation_selected != self.attn_implementation_requested:
            self.logger.warning(
                "HF attention backend fallback: requested=%s selected=%s",
                self.attn_implementation_requested,
                self.attn_implementation_selected,
            )

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_checkpoint, trust_remote_code=True
        )

        # Decoder-only models require left padding for correct generation.
        try:
            if getattr(self.processor, "tokenizer", None) is not None:
                self.processor.tokenizer.padding_side = "left"
                if self.processor.tokenizer.pad_token_id is None:
                    self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        except Exception:
            pass

    def _resolve_image_path(self, jsonl_path: Path, image_rel: str) -> Path:
        root_image_dir: Path | None = None
        root_raw = str(self.cfg.root_image_dir or "").strip()
        if root_raw:
            root_image_dir = Path(root_raw).resolve()
        return resolve_image_path_best_effort(
            image_rel,
            jsonl_dir=jsonl_path.parent,
            root_image_dir=root_image_dir,
        )

    def _build_messages(self, image: Image.Image) -> List[Dict[str, Any]]:
        user_prompt = USER_PROMPT
        system_prompt = SYSTEM_PROMPT
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]

    def _generate(self, image: Image.Image) -> str:
        backend = str(self.cfg.backend_type).strip().lower()
        if backend == "hf":
            return self._generate_hf(image)
        if backend == "vllm":
            return self._generate_vllm(image)
        raise ValueError(f"infer.backend.type must be hf|vllm, got {backend!r}")

    def _generate_batch(self, images: List[Image.Image]) -> List[GenerationResult]:
        """Generate texts for a micro-batch.

        For HF we do a single batched `model.generate()` when possible.
        For vLLM (OpenAI-compatible server) we issue concurrent requests.
        """

        if not images:
            return []

        backend = str(self.cfg.backend_type).strip().lower()
        if backend == "hf":
            return self._generate_hf_batch(images)
        if backend == "vllm":
            return self._generate_vllm_batch(images)
        raise ValueError(f"infer.backend.type must be hf|vllm, got {backend!r}")

    def _generate_hf_batch(self, images: List[Image.Image]) -> List[GenerationResult]:
        assert self.model is not None and self.processor is not None
        if not images:
            return []

        # Best-effort batched generate; fall back to per-sample generation on error.
        try:
            messages = [self._build_messages(img) for img in images]
            prompt_texts = [
                self.processor.apply_chat_template(
                    m, add_generation_prompt=True, tokenize=False
                )
                for m in messages
            ]

            model_inputs = self.processor(
                text=prompt_texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {k: v.to(self.cfg.device) for k, v in model_inputs.items()}

            if "attention_mask" in model_inputs:
                prompt_lens = model_inputs["attention_mask"].sum(dim=1).tolist()
            else:
                # Fallback: assume no padding.
                prompt_lens = [int(model_inputs["input_ids"].shape[1])] * int(
                    len(images)
                )

            gen_kwargs = dict(
                max_new_tokens=self.gen_cfg.max_new_tokens,
                do_sample=self.gen_cfg.temperature > 0,
                temperature=max(1e-4, self.gen_cfg.temperature),
                top_p=self.gen_cfg.top_p,
                use_cache=True,
            )
            if self.gen_cfg.repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = self.gen_cfg.repetition_penalty

            with torch.inference_mode():
                gen_ids = self.model.generate(**model_inputs, **gen_kwargs)

            out: List[GenerationResult] = []
            for i in range(len(images)):
                prompt_len = int(prompt_lens[i])
                gen_only = gen_ids[i, prompt_len:]
                raw_text = self.processor.tokenizer.decode(
                    gen_only,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                out.append(GenerationResult(text=raw_text, error=None))
            return out
        except Exception:
            out: List[GenerationResult] = []
            for img in images:
                try:
                    out.append(
                        GenerationResult(text=self._generate_hf(img), error=None)
                    )
                except Exception as exc:  # noqa: BLE001
                    out.append(GenerationResult(text="", error=exc))
            return out

    def _generate_vllm_batch(self, images: List[Image.Image]) -> List[GenerationResult]:
        if not images:
            return []

        if self._vllm_mode() == "local":
            # Local mode supports true batched chat() in-process.
            return self._generate_vllm_local_batch(images)

        # Server mode: limit concurrency to avoid overwhelming the server.
        max_workers_raw = self.cfg.backend.get("client_concurrency")
        try:
            max_workers = (
                int(max_workers_raw)
                if max_workers_raw is not None
                else int(len(images))
            )
        except Exception:
            max_workers = int(len(images))
        max_workers = max(1, min(int(max_workers), int(len(images))))

        out = [GenerationResult() for _ in images]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_idx = {
                ex.submit(self._generate_vllm_server, img): i
                for i, img in enumerate(images)
            }
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                try:
                    out[i].text = fut.result()
                    out[i].error = None
                except Exception as exc:  # noqa: BLE001
                    out[i].text = ""
                    out[i].error = exc
        return out

    def _generate_hf(self, image: Image.Image) -> str:
        assert self.model is not None and self.processor is not None

        messages = self._build_messages(image)
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        model_inputs = self.processor(
            text=prompt_text, images=[image], return_tensors="pt"
        )
        model_inputs = {k: v.to(self.cfg.device) for k, v in model_inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=self.gen_cfg.temperature > 0,
            temperature=max(1e-4, self.gen_cfg.temperature),
            top_p=self.gen_cfg.top_p,
            use_cache=True,
        )
        if self.gen_cfg.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = self.gen_cfg.repetition_penalty

        # NOTE: Do not pass `generator=` into `model.generate()`.
        #
        # Some upstream / remote-code model implementations (incl. some Qwen3-VL
        # checkpoints) treat unknown kwargs as `model_kwargs` and raise:
        #   "The following `model_kwargs` are not used by the model: ['generator']"
        #
        # We seed torch/CUDA globally in `_seed()` for deterministic sampling.
        with torch.inference_mode():
            gen_ids = self.model.generate(**model_inputs, **gen_kwargs)

        prompt_len = model_inputs["input_ids"].shape[1]
        gen_only = gen_ids[:, prompt_len:]
        raw_text = self.processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return raw_text

    def _generate_vllm(self, image: Image.Image) -> str:
        """Generate via vLLM.

        Modes:
        - server (default): OpenAI-compatible HTTP server
        - local: in-process vLLM Python API (no HTTP server)
        """

        if self._vllm_mode() == "local":
            # Use the same OpenAI-style message structure as the server, but route
            # through the in-process vLLM API.
            out = self._generate_vllm_local_batch([image])
            if not out:
                return ""
            if out[0].error is not None:
                raise out[0].error
            return out[0].text

        # Server mode (OpenAI-compatible HTTP requests).
        return self._generate_vllm_server(image)

    def _generate_vllm_server(self, image: Image.Image) -> str:
        """Generate via an OpenAI-compatible vLLM server (best-effort).

        Config:
        - infer.backend.base_url: e.g. http://127.0.0.1:8000 or http://127.0.0.1:8000/v1
        - infer.backend.model: optional; defaults to infer.model_checkpoint
        - infer.backend.timeout_s: optional
        """

        try:
            import base64
            import io

            import requests
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "vLLM backend requires the 'requests' package. Install it in the ms env, or set infer.backend.type=hf."
            ) from exc

        base_url = str(
            self.cfg.backend.get("base_url") or os.environ.get("VLLM_BASE_URL") or ""
        ).strip()
        if not base_url:
            raise RuntimeError(
                "infer.backend.type=vllm requires infer.backend.base_url (or env VLLM_BASE_URL). "
                "To run without a server, set infer.backend.mode=local. To disable vLLM, set infer.backend.type=hf."
            )

        model = str(self.cfg.backend.get("model") or self.cfg.model_checkpoint).strip()
        timeout_s = float(self.cfg.backend.get("timeout_s", 180.0))

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ]

        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            url = base_url + "/chat/completions"
        else:
            url = base_url + "/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(self.gen_cfg.temperature),
            "top_p": float(self.gen_cfg.top_p),
            "max_tokens": int(self.gen_cfg.max_new_tokens),
            "stream": False,
        }
        if self.gen_cfg.repetition_penalty is not None:
            payload["repetition_penalty"] = float(self.gen_cfg.repetition_penalty)
        if self.gen_cfg.seed is not None:
            payload["seed"] = int(self.gen_cfg.seed)

        headers = {"Content-Type": "application/json"}

        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(
                f"vLLM server error status={resp.status_code}: {resp.text[:2000]}"
            )

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM server returned no choices: {data}")

        c0 = choices[0] if isinstance(choices, list) else choices
        if isinstance(c0, dict) and isinstance(c0.get("message"), dict):
            content = c0["message"].get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(str(p.get("text", "")))
                return "".join(parts)

        if isinstance(c0, dict) and "text" in c0:
            return str(c0.get("text") or "")

        raise RuntimeError(f"Unrecognized vLLM response schema: {data}")

    def _generate_vllm_local_batch(
        self, images: List[Image.Image]
    ) -> List[GenerationResult]:
        """Generate via the in-process vLLM Python API (no HTTP server)."""

        if not images:
            return []

        self._load_vllm_local()
        assert self.vllm_llm is not None

        try:
            import base64
            import io
        except Exception as exc:  # noqa: BLE001
            return [GenerationResult(text="", error=exc) for _ in images]

        # Build OpenAI-style messages; vLLM supports a batch of message lists.
        msg_batch = []
        for image in images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            msg_batch.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    },
                ]
            )

        sp = self._vllm_sampling_params()
        try:
            outs = self.vllm_llm.chat(msg_batch, sampling_params=sp, use_tqdm=False)
        except Exception as exc:  # noqa: BLE001
            return [GenerationResult(text="", error=exc) for _ in images]

        results: List[GenerationResult] = []
        for o in outs:
            try:
                seqs = getattr(o, "outputs", None) or []
                text = seqs[0].text if seqs else ""
                results.append(GenerationResult(text=text, error=None))
            except Exception as exc:  # noqa: BLE001
                results.append(GenerationResult(text="", error=exc))

        # vLLM should return one output per request; enforce alignment.
        if len(results) != len(images):
            missing = len(images) - len(results)
            if missing > 0:
                results.extend(
                    [
                        GenerationResult(
                            text="",
                            error=RuntimeError(
                                "vLLM returned fewer outputs than requests"
                            ),
                        )
                        for _ in range(missing)
                    ]
                )
            else:
                results = results[: len(images)]

        return results

    def _process_gt(
        self,
        record: Dict[str, Any],
        *,
        width: int,
        height: int,
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        return self.coord.process_record_gt(
            record, width=width, height=height, errors=errors
        )

    def _process_pred(
        self,
        raw_text: str,
        *,
        width: int,
        height: int,
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        return self.coord.process_prediction_text(
            raw_text, width=width, height=height, errors=errors
        )

    @staticmethod
    def _compact_objects(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip internal fields to the unified gt_vs_pred.jsonl schema."""
        compact: List[Dict[str, Any]] = []
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            kind = obj.get("type")
            points = obj.get("points")
            if kind not in {"bbox_2d", "poly"}:
                continue
            if not isinstance(points, list):
                continue
            desc = str(obj.get("desc", "") or "").strip()
            compact.append(
                {
                    "type": kind,
                    "points": points,
                    "desc": desc,
                    "score": 1.0,
                }
            )
        return compact

    def _prepare_image(
        self, jsonl_path: Path, record: Dict[str, Any]
    ) -> Tuple[Optional[Path], Optional[Image.Image]]:
        images = record.get("images") or []
        if len(images) != 1:
            return None, None
        img_path = self._resolve_image_path(jsonl_path, images[0])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return img_path, None
        return img_path, image

    def infer(self) -> Tuple[Path, Path]:
        jsonl_path = Path(self.cfg.gt_jsonl)
        out_path = Path(self.cfg.out_path)
        summary_path = Path(self.cfg.summary_path or (out_path.parent / "summary.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        backend = str(self.cfg.backend_type).strip().lower()
        determinism = "strict" if backend == "hf" else "best_effort"

        try:
            batch_size = int(getattr(self.gen_cfg, "batch_size", 1) or 1)
        except Exception:
            batch_size = 1
        batch_size = max(1, int(batch_size))

        counters = RunCounters()
        self.load_model()

        resolved_meta = {
            "mode": self.resolved_mode,
            "mode_resolution_reason": self.mode_reason,
            "backend": backend,
            "model_checkpoint": self.cfg.model_checkpoint,
            "gt_jsonl": self.cfg.gt_jsonl,
            "pred_coord_mode": self.cfg.pred_coord_mode,
            "device": self.cfg.device,
            "limit": self.cfg.limit,
            "generation": {
                "temperature": self.gen_cfg.temperature,
                "top_p": self.gen_cfg.top_p,
                "max_new_tokens": self.gen_cfg.max_new_tokens,
                "repetition_penalty": self.gen_cfg.repetition_penalty,
                "batch_size": batch_size,
                "seed": self.gen_cfg.seed,
            },
            "artifacts": {
                "gt_vs_pred_jsonl": str(out_path),
                "summary_json": str(summary_path),
            },
        }
        if backend == "vllm":
            # Persist only non-sensitive backend fields (allowlist).
            allowed = {"mode", "base_url", "model", "timeout_s", "client_concurrency"}
            resolved_meta["backend_cfg"] = {
                k: v for k, v in (self.cfg.backend or {}).items() if str(k) in allowed
            }

        self.logger.info("Inference resolved config: %s", json.dumps(resolved_meta))

        def _emit(output: Dict[str, Any], errors: List[str]) -> None:
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            for code in errors:
                counters.add(ERROR_CANONICAL.get(code, code))
            counters.total_emitted += 1

        def _flush_pending(pending: List[Dict[str, Any]]) -> None:
            if not pending:
                return

            # Safety: avoid exceeding limit by flushing only the remainder.
            if self.cfg.limit and self.cfg.limit > 0:
                remaining = int(self.cfg.limit) - int(counters.total_emitted)
                if remaining <= 0:
                    return
                if len(pending) > remaining:
                    pending = pending[:remaining]

            images = [p["image_obj"] for p in pending]
            results = self._generate_batch(images)

            for p, res in zip(pending, results):
                errors = list(p["errors"])
                raw_output_json: Dict[str, Any] | None = None
                raw_special_tokens: List[str] = []
                raw_ends_with_im_end = False
                pred: List[Dict[str, Any]] = []

                if res.error is not None:
                    errors.append("generation_failed")
                else:
                    raw_text = res.text
                    raw_special_tokens = extract_special_tokens(raw_text)
                    raw_ends_with_im_end = raw_text.endswith("<|im_end|>")
                    raw_output_json = load_prediction_dict(raw_text)
                    try:
                        pred_errors: List[str] = []
                        pred = self._process_pred(
                            raw_text,
                            width=int(p["width"]),
                            height=int(p["height"]),
                            errors=pred_errors,
                        )
                        pred = self._compact_objects(pred)
                        errors.extend(pred_errors)
                    except Exception:
                        errors.append("generation_failed")
                        pred = []

                output = {
                    "image": p["image"],
                    "width": p["width"],
                    "height": p["height"],
                    "mode": self.resolved_mode,
                    # Points are emitted in pixel space; hint downstream to skip re-denorm.
                    "coord_mode": "pixel",
                    "gt": p["gt"],
                    "pred": pred,
                    "raw_output_json": raw_output_json,
                    "raw_special_tokens": raw_special_tokens,
                    "raw_ends_with_im_end": raw_ends_with_im_end,
                    "errors": errors,
                }
                _emit(output, errors)

        pbar_total = self.cfg.limit if self.cfg.limit > 0 else None
        pending: List[Dict[str, Any]] = []

        with (
            jsonl_path.open("r", encoding="utf-8") as fin,
            out_path.open("w", encoding="utf-8") as fout,
            tqdm(
                total=pbar_total,
                desc="Infer",
                unit="samples",
                dynamic_ncols=True,
                smoothing=0.1,
                mininterval=1.0,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar,
        ):
            for line in fin:
                if self.cfg.limit and counters.total_emitted >= self.cfg.limit:
                    break

                line = line.strip()
                if not line:
                    continue

                # We update on non-empty lines for a smoother display.
                pbar.update(1)

                counters.total_read += 1
                try:
                    record = json.loads(line)
                except Exception:
                    counters.add("invalid_json")
                    continue

                errors: List[str] = []
                width = record.get("width")
                height = record.get("height")
                try:
                    width = int(width)
                    height = int(height)
                except Exception:
                    width = None
                    height = None

                image_key = (record.get("images") or [None])[0]

                if not width or not height:
                    errors.append("size_mismatch")
                    output = {
                        "image": image_key,
                        "width": width,
                        "height": height,
                        "mode": self.resolved_mode,
                        "coord_mode": None,
                        "gt": [],
                        "pred": [],
                        "raw_output_json": None,
                        "raw_special_tokens": [],
                        "raw_ends_with_im_end": False,
                        "errors": errors,
                    }
                    _emit(output, errors)
                    continue

                # Process GT first to catch mode mismatches early.
                gt_errors: List[str] = []
                gt = self._process_gt(
                    record, width=width, height=height, errors=gt_errors
                )
                gt = self._compact_objects(gt)
                errors.extend(gt_errors)

                # Mode/GT mismatch detected by processor -> skip generation but still emit record.
                run_generation = "mode_gt_mismatch" not in errors

                images = record.get("images") or []
                image_obj: Optional[Image.Image] = None
                if len(images) != 1:
                    errors.append("multi_image_not_supported")
                    run_generation = False
                else:
                    _, image_obj = self._prepare_image(jsonl_path, record)
                    if image_obj is None:
                        errors.append("image_load_failed")
                        run_generation = False

                if not run_generation:
                    output = {
                        "image": image_key,
                        "width": width,
                        "height": height,
                        "mode": self.resolved_mode,
                        "coord_mode": "pixel",
                        "gt": gt,
                        "pred": [],
                        "raw_output_json": None,
                        "raw_special_tokens": [],
                        "raw_ends_with_im_end": False,
                        "errors": errors,
                    }
                    _emit(output, errors)
                    continue

                assert image_obj is not None
                pending.append(
                    {
                        "image": image_key,
                        "width": width,
                        "height": height,
                        "gt": gt,
                        "errors": errors,
                        "image_obj": image_obj,
                    }
                )

                # Flush when we have a full micro-batch, but never overshoot the limit.
                target = batch_size
                if self.cfg.limit and self.cfg.limit > 0:
                    remaining = int(self.cfg.limit) - int(counters.total_emitted)
                    target = max(1, min(int(target), int(remaining)))

                if len(pending) >= target:
                    _flush_pending(pending)
                    pending = []

            # Flush any final partial batch.
            _flush_pending(pending)

        summary_payload: Dict[str, Any] = {
            "mode": self.resolved_mode,
            "determinism": determinism,
            **counters.to_summary(),
            "backend": {
                "type": backend,
                "model_checkpoint": self.cfg.model_checkpoint,
            },
            "generation": {
                "temperature": self.gen_cfg.temperature,
                "top_p": self.gen_cfg.top_p,
                "max_new_tokens": self.gen_cfg.max_new_tokens,
                "repetition_penalty": self.gen_cfg.repetition_penalty,
                "batch_size": batch_size,
                "seed": self.gen_cfg.seed,
            },
            "infer": {
                "gt_jsonl": self.cfg.gt_jsonl,
                "pred_coord_mode": self.cfg.pred_coord_mode,
                "device": self.cfg.device,
                "limit": self.cfg.limit,
            },
        }
        if self.requested_mode == "auto":
            summary_payload["mode_resolution_reason"] = self.mode_reason

        if backend == "hf":
            summary_payload["backend"]["attn_implementation_requested"] = (
                self.attn_implementation_requested
            )
            summary_payload["backend"]["attn_implementation_selected"] = (
                self.attn_implementation_selected
            )

        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.info(
            "Inference finished: %s samples emitted, summary=%s",
            counters.total_emitted,
            summary_path,
        )
        return out_path, summary_path


__all__ = ["GenerationConfig", "InferenceConfig", "InferenceEngine"]
