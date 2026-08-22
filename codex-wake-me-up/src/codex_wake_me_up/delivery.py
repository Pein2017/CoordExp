"""Thread-delivery value vocabulary.

Behavior is introduced test-first; the initial declarations keep focused tests
collectable while their assertions remain red.
"""

from __future__ import annotations

import hashlib
import uuid
from enum import StrEnum
from typing import Any, Mapping


DELIVERY_POINTER_MAX_CHARS = 512


class DeliveryKind(StrEnum):
    GOAL = "goal"
    THREAD = "thread"


class ThreadDeliveryState(StrEnum):
    UNATTEMPTED = "unattempted"
    ADMISSION_IN_PROGRESS = "admission_in_progress"
    QUEUE_ACCEPTED = "queue_accepted"
    RECORDED = "recorded"
    DELIVERY_REJECTED = "delivery_rejected"
    DELIVERY_UNCERTAIN = "delivery_uncertain"
    DELIVERY_MODIFIED = "delivery_modified"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLATION_IN_PROGRESS = "cancellation_in_progress"
    CANCELLED = "cancelled"
    CANCELLATION_TOO_LATE = "cancellation_too_late"


def build_thread_delivery(
    *, monitor_id: str, thread_id: str, capability: Mapping[str, Any]
) -> dict[str, Any]:
    if not monitor_id or not thread_id:
        raise ValueError("monitor_id and thread_id must be non-empty")
    delivery_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"codex-wake-me-up:{monitor_id}:thread-delivery:v1",
        )
    )
    pointer = (
        f"[codex-wake-me-up monitor={monitor_id} delivery={delivery_id}] "
        "Inspect wake_me_up_status once. This pointer is not a success claim."
    )
    if len(pointer) > DELIVERY_POINTER_MAX_CHARS:
        raise ValueError("thread-delivery pointer exceeds its bound")
    return {
        "schema_epoch": 1,
        "kind": DeliveryKind.THREAD.value,
        "thread_id": thread_id,
        "delivery_id": delivery_id,
        "pointer": pointer,
        "pointer_digest": hashlib.sha256(pointer.encode("utf-8")).hexdigest(),
        "capability": dict(capability),
    }
