"""Store generated model-card text as lossless checkpoint metadata, not source."""

import hashlib
import json
import os
from pathlib import Path


def package_model_card(adapter_dir: Path) -> Path | None:
    """Package a newly generated PEFT card before atomic checkpoint publication.

    Existing model weights/configuration are never changed. An occupied,
    conflicting metadata file fails rather than discarding either version.
    """
    card = Path(adapter_dir) / "README.md"
    if not card.exists():
        return None
    if card.is_symlink():
        raise ValueError("generated model card must not be a symlink")
    raw = card.read_bytes()
    payload = {
        "schema": "coordexp.generated_model_card.v1",
        "original_name": "README.md",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content_utf8": raw.decode("utf-8"),
    }
    target = card.with_name("model_card.json")
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    restored = json.loads(target.read_text())["content_utf8"].encode("utf-8")
    if restored != raw or card.read_bytes() != raw:
        raise ValueError("model card changed during metadata packaging")
    card.unlink()
    return target
