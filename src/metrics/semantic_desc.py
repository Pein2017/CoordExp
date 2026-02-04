from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import re

import numpy as np


def normalize_desc(desc: str) -> str:
    """Normalize LVIS-like category strings for semantic matching.

    Keep this consistent with the offline evaluator.
    """

    s = str(desc or "").strip().lower()
    if not s:
        return ""
    s = s.replace("_", " ").replace("/", " ")
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _resolve_device(device: str) -> str:
    d = (device or "auto").strip().lower()
    if d == "auto":
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return d


@dataclass
class SemanticDescEncoder:
    """Lightweight sentence-embedding encoder with caching.

    Runs under inference mode (no gradients). This is used for both monitoring and
    Stage-2 AB semantic gating: it can affect training by masking/weighting CE on
    matched description tokens, but it does not introduce gradients through the
    encoder itself.
    """

    model_name: str
    revision: str | None = None
    device: str = "auto"
    batch_size: int = 64
    max_length: int = 64

    _tokenizer: object | None = None
    _model: object | None = None
    _cache: Dict[str, np.ndarray] | None = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        resolved_device = _resolve_device(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision
        )
        model = AutoModel.from_pretrained(self.model_name, revision=self.revision)
        model.to(resolved_device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._cache = {}

    def encode_norm_texts(self, norm_texts: Sequence[str]) -> Dict[str, Optional[np.ndarray]]:
        """Encode and return embeddings for normalized texts.

        Returned embeddings are L2-normalized float32 numpy arrays on CPU.
        """

        self._ensure_loaded()
        assert self._cache is not None

        # Filter to texts that are non-empty and not cached.
        missing: List[str] = []
        for t in norm_texts:
            tt = str(t or "")
            if not tt:
                continue
            if tt not in self._cache:
                missing.append(tt)

        if missing:
            import torch
            import torch.nn.functional as F

            resolved_device = _resolve_device(self.device)
            tokenizer = self._tokenizer
            model = self._model
            assert tokenizer is not None and model is not None

            bs = max(1, int(self.batch_size))
            max_len = max(8, int(self.max_length))

            with torch.inference_mode():
                for i in range(0, len(missing), bs):
                    batch = missing[i : i + bs]
                    inputs = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(resolved_device) for k, v in inputs.items()}
                    out = model(**inputs)
                    last = out.last_hidden_state  # [B, T, D]
                    mask = inputs.get("attention_mask")
                    if mask is None:
                        pooled = last.mean(dim=1)
                    else:
                        mask_f = mask.unsqueeze(-1).to(dtype=last.dtype)
                        denom = mask_f.sum(dim=1).clamp(min=1e-6)
                        pooled = (last * mask_f).sum(dim=1) / denom
                    pooled = F.normalize(pooled, p=2, dim=-1)
                    vecs = pooled.detach().cpu().to(dtype=torch.float32).numpy()
                    for t, v in zip(batch, vecs):
                        self._cache[str(t)] = v

        return {t: self._cache.get(str(t), None) for t in norm_texts}


def semantic_ok_and_sim(
    *,
    pred_desc: str,
    gt_desc: str,
    encoder: Optional[SemanticDescEncoder],
    threshold: float,
) -> tuple[bool, float | None]:
    """Return (ok, sim) for a single matched pair.

    ok := exact match OR (cos_sim >= threshold) when encoder is provided.
    sim := cosine similarity when encoder is provided and both embeddings exist.
    """

    p = normalize_desc(pred_desc)
    g = normalize_desc(gt_desc)
    exact_ok = bool(p) and (p == g)
    if encoder is None or not p or not g:
        return exact_ok, None

    emb = encoder.encode_norm_texts([p, g])
    pv = emb.get(p)
    gv = emb.get(g)
    if pv is None or gv is None:
        return exact_ok, None
    sim = float(np.dot(pv, gv))
    return bool(exact_ok or sim >= float(threshold)), sim
