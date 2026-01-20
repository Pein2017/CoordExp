from __future__ import annotations

from typing import Any

import numpy as np


def preprocess_logits_for_token_accuracy(logits: Any, labels: Any) -> Any:
    """Reduce eval memory by converting logits -> argmax token ids.

    Transformers will gather the returned object across processes and pass it to
    compute_metrics. Returning argmax token ids is far cheaper than returning the
    full float logits.
    """

    # Some models return a tuple of logits.
    if isinstance(logits, (tuple, list)) and logits:
        logits = logits[0]
    try:
        return logits.argmax(dim=-1)
    except Exception:
        # If logits isn't a torch.Tensor, fall back to returning it unchanged.
        return logits


def compute_token_accuracy_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute next-token accuracy over supervised (labels != -100) tokens.

    Returns keys without the `eval_` prefix; Trainer will add it automatically.
    """

    preds = getattr(eval_pred, "predictions", None)
    labels = getattr(eval_pred, "label_ids", None)
    if preds is None or labels is None:
        return {}

    preds_np = np.asarray(preds)
    labels_np = np.asarray(labels)

    # If logits slipped through (N, L, V), convert to argmax ids.
    if preds_np.ndim == 3:
        preds_np = preds_np.argmax(axis=-1)

    # Expect (N, L) for both.
    if preds_np.ndim != 2 or labels_np.ndim != 2:
        return {}

    if preds_np.shape != labels_np.shape:
        # Some trainers pad predictions/labels differently; best-effort only.
        min_len = min(preds_np.shape[-1], labels_np.shape[-1])
        preds_np = preds_np[:, :min_len]
        labels_np = labels_np[:, :min_len]

    if preds_np.shape[-1] < 2:
        return {}

    preds_next = preds_np[:, :-1]
    labels_next = labels_np[:, 1:]

    mask = labels_next != -100
    total = int(mask.sum())
    if total <= 0:
        return {"token_acc": 0.0, "supervised_tokens": 0.0}

    correct = int(((preds_next == labels_next) & mask).sum())
    return {
        "token_acc": float(correct / total),
        "supervised_tokens": float(total),
    }
