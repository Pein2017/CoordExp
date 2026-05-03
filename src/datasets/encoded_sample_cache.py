from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

_ENCODED_SAMPLE_CACHE_VERSION = 1
_DEFAULT_ENCODED_SAMPLE_SHARD_SIZE = 512
_DEFAULT_MAX_RESIDENT_SHARDS = 4

EncodedSampleCacheManifestStatus = Literal["building", "complete", "error"]


@dataclass(frozen=True)
class EncodedSampleCacheRequest:
    enabled: bool
    root_dir: Path
    ineligible_policy: str
    wait_timeout_s: float
    max_resident_shards: int
    dataset_split: str
    dataset_jsonl: Any
    fingerprint: dict[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EncodedSampleCacheRequest":
        timeout_s = float(payload.get("wait_timeout_s", 7200.0) or 0.0)
        if not math.isfinite(timeout_s):
            raise ValueError(
                f"encoded_sample_cache.wait_timeout_s must be finite, got {timeout_s!r}"
            )
        return cls(
            enabled=bool(payload.get("enabled", False)),
            root_dir=Path(str(payload.get("root_dir") or ".")).resolve(),
            ineligible_policy=str(payload.get("ineligible_policy") or "error"),
            wait_timeout_s=timeout_s,
            max_resident_shards=max(
                int(
                    payload.get(
                        "max_resident_shards", _DEFAULT_MAX_RESIDENT_SHARDS
                    )
                    or 1
                ),
                1,
            ),
            dataset_split=str(payload.get("dataset_split") or "train"),
            dataset_jsonl=payload.get("dataset_jsonl"),
            fingerprint=_canonicalize_fingerprint(
                dict(payload.get("fingerprint") or {})
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "root_dir": str(self.root_dir),
            "ineligible_policy": self.ineligible_policy,
            "wait_timeout_s": self.wait_timeout_s,
            "max_resident_shards": self.max_resident_shards,
            "dataset_split": self.dataset_split,
            "dataset_jsonl": self.dataset_jsonl,
            "fingerprint": dict(self.fingerprint),
        }


@dataclass(frozen=True)
class EncodedSampleShard:
    shard_index: int
    file: str
    start: int
    end: int
    count: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EncodedSampleShard":
        return cls(
            shard_index=int(payload.get("shard_index") or 0),
            file=str(payload.get("file") or ""),
            start=int(payload.get("start") or 0),
            end=int(payload.get("end") or 0),
            count=int(payload.get("count") or 0),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "file": self.file,
            "start": self.start,
            "end": self.end,
            "count": self.count,
        }


@dataclass(frozen=True)
class EncodedSampleCacheManifest:
    version: int
    status: EncodedSampleCacheManifestStatus
    fingerprint: dict[str, Any]
    fingerprint_sha256: str
    dataset_split: str
    dataset_jsonl: Any
    num_samples: int | None = None
    shard_size: int | None = None
    payload_keys: tuple[str, ...] = ()
    shards: tuple[EncodedSampleShard, ...] = ()
    build_started_at: str | None = None
    build_completed_at: str | None = None
    build_failed_at: str | None = None
    error: str | None = None
    path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        path: Path | None = None,
    ) -> "EncodedSampleCacheManifest":
        status = str(payload.get("status") or "")
        location = f": {path}" if path is not None else ""
        if status not in {"building", "complete", "error"}:
            raise ValueError(
                "Invalid encoded sample cache manifest status"
                f"{location}: {status!r}"
            )
        fingerprint = payload.get("fingerprint")
        if not isinstance(fingerprint, Mapping):
            raise TypeError(
                f"Encoded sample cache fingerprint must be a mapping{location}"
            )
        canonical_fingerprint = _canonicalize_fingerprint(fingerprint)
        is_complete = status == "complete"
        if is_complete:
            for key in (
                "fingerprint_sha256",
                "num_samples",
                "shard_size",
                "payload_keys",
                "shards",
            ):
                if key not in payload:
                    raise ValueError(
                        "Encoded sample cache complete manifest missing "
                        f"{key}{location}"
                    )
            observed_digest = str(payload.get("fingerprint_sha256") or "")
            if not observed_digest:
                raise ValueError(
                    "Encoded sample cache complete manifest fingerprint_sha256 "
                    f"is required{location}"
                )
            expected_digest = _fingerprint_digest(canonical_fingerprint)
            if observed_digest != expected_digest:
                raise ValueError(
                    "Encoded sample cache complete manifest fingerprint_sha256 "
                    f"mismatch{location}: expected={expected_digest} "
                    f"observed={observed_digest}"
                )
        raw_payload_keys = payload.get("payload_keys")
        if raw_payload_keys is None or (not is_complete and not raw_payload_keys):
            raw_payload_keys = []
        if not isinstance(raw_payload_keys, list):
            raise TypeError(
                f"Encoded sample cache payload_keys must be a list{location}"
            )
        raw_shards = payload.get("shards")
        if raw_shards is None or (not is_complete and not raw_shards):
            raw_shards = []
        if not isinstance(raw_shards, list):
            raise TypeError(
                f"Encoded sample cache shards payload must be a list{location}"
            )
        shards: list[EncodedSampleShard] = []
        for shard in raw_shards:
            if not isinstance(shard, Mapping):
                raise TypeError(
                    f"Encoded sample cache shard metadata must be a mapping{location}"
                )
            typed_shard = EncodedSampleShard.from_mapping(shard)
            if is_complete:
                if not typed_shard.file:
                    raise ValueError(
                        "Encoded sample cache complete manifest shard file "
                        f"is required{location}"
                    )
                if typed_shard.start < 0 or typed_shard.end < typed_shard.start:
                    raise ValueError(
                        "Encoded sample cache complete manifest shard range "
                        f"is invalid{location}"
                    )
                if typed_shard.count != typed_shard.end - typed_shard.start:
                    raise ValueError(
                        "Encoded sample cache complete manifest shard count "
                        f"mismatch{location}"
                    )
            shards.append(typed_shard)
        num_samples = (
            int(payload["num_samples"])
            if payload.get("num_samples") is not None
            else None
        )
        shard_size = (
            int(payload["shard_size"])
            if payload.get("shard_size") is not None
            else None
        )
        if is_complete:
            if num_samples is None or num_samples < 0:
                raise ValueError(
                    "Encoded sample cache complete manifest num_samples "
                    f"must be non-negative{location}"
                )
            if shard_size is None or shard_size <= 0:
                raise ValueError(
                    "Encoded sample cache complete manifest shard_size "
                    f"must be positive{location}"
                )
            if num_samples > 0 and not shards:
                raise ValueError(
                    "Encoded sample cache complete manifest shards are required "
                    f"when num_samples is positive{location}"
                )

        return cls(
            version=int(payload.get("version") or -1),
            status=status,  # type: ignore[arg-type]
            fingerprint=canonical_fingerprint,
            fingerprint_sha256=str(payload.get("fingerprint_sha256") or ""),
            dataset_split=str(payload.get("dataset_split") or "train"),
            dataset_jsonl=payload.get("dataset_jsonl"),
            num_samples=num_samples,
            shard_size=shard_size,
            payload_keys=tuple(str(k) for k in raw_payload_keys),
            shards=tuple(shards),
            build_started_at=(
                str(payload["build_started_at"])
                if payload.get("build_started_at") is not None
                else None
            ),
            build_completed_at=(
                str(payload["build_completed_at"])
                if payload.get("build_completed_at") is not None
                else None
            ),
            build_failed_at=(
                str(payload["build_failed_at"])
                if payload.get("build_failed_at") is not None
                else None
            ),
            error=str(payload["error"]) if payload.get("error") is not None else None,
            path=path,
        )

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "status": self.status,
            "fingerprint": dict(self.fingerprint),
            "fingerprint_sha256": self.fingerprint_sha256,
            "dataset_split": self.dataset_split,
            "dataset_jsonl": self.dataset_jsonl,
        }
        if self.status == "complete":
            payload.update(
                {
                    "num_samples": int(self.num_samples or 0),
                    "shard_size": int(self.shard_size or 0),
                    "payload_keys": list(self.payload_keys),
                    "shards": [shard.to_mapping() for shard in self.shards],
                    "build_started_at": self.build_started_at,
                    "build_completed_at": self.build_completed_at,
                }
            )
        elif self.status == "building":
            payload["build_started_at"] = self.build_started_at
        elif self.status == "error":
            payload.update(
                {
                    "build_started_at": self.build_started_at,
                    "build_failed_at": self.build_failed_at,
                    "error": self.error,
                }
            )
        return payload


def _json_canonical_dumps(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise ValueError(
            "Encoded sample cache fingerprint contains non-JSON-serializable values."
        ) from exc


def _canonicalize_fingerprint(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_json_canonical_dumps(dict(fingerprint)))


def _fingerprint_digest(fingerprint: Mapping[str, Any]) -> str:
    text = _json_canonical_dumps(fingerprint)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _write_torch_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Encoded sample cache file not found: {path}"
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse encoded sample cache JSON file: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Encoded sample cache JSON must be an object: {path}")
    return payload


def _resolve_rank_world() -> tuple[int, int]:
    rank = 0
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            rank = int(torch.distributed.get_rank())
            world_size = max(int(torch.distributed.get_world_size()), 1)
            return rank, world_size
        except (RuntimeError, TypeError, ValueError):
            pass

    rank_env = os.environ.get("RANK")
    if isinstance(rank_env, str) and rank_env.strip().lstrip("-").isdigit():
        rank = int(rank_env)

    world_env = os.environ.get("WORLD_SIZE")
    if isinstance(world_env, str) and world_env.strip().isdigit():
        world_size = max(int(world_env), 1)

    return rank, world_size


def _wait_for_complete_manifest(path: Path, *, timeout_s: float) -> dict[str, Any]:
    timeout = float(timeout_s)
    if not math.isfinite(timeout):
        raise ValueError(
            f"encoded_sample_cache.wait_timeout_s must be finite, got {timeout_s!r}"
        )

    deadline = None
    if timeout > 0:
        deadline = time.monotonic() + timeout

    while True:
        if path.exists():
            manifest = _read_json(path)
            status = str(manifest.get("status") or "")
            if status == "complete":
                return manifest
            if status == "error":
                reason = str(manifest.get("error") or "unknown cache build failure")
                raise RuntimeError(
                    "Encoded sample cache build failed: "
                    f"{reason}. Cache manifest={path}"
                )

        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                "Timed out waiting for encoded sample cache manifest: "
                f"{path}. Increase training.encoded_sample_cache.wait_timeout_s "
                "for large datasets or use a fresh cache root."
            )
        time.sleep(0.2)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_bypass_info(
    *,
    request: Mapping[str, Any],
    resolved_root: Path,
    fingerprint: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    fingerprint_sha = _fingerprint_digest(fingerprint)
    cache_dir = resolved_root / fingerprint_sha
    return {
        "enabled": True,
        "status": "bypassed",
        "reason": str(reason),
        "policy": str(request.get("ineligible_policy") or "error"),
        "wait_timeout_s": float(request.get("wait_timeout_s", 7200.0) or 0.0),
        "dataset_split": str(request.get("dataset_split") or "train"),
        "dataset_jsonl": request.get("dataset_jsonl"),
        "fingerprint": dict(fingerprint),
        "fingerprint_sha256": fingerprint_sha,
        "root_dir": str(resolved_root),
        "cache_dir": str(cache_dir),
        "manifest_path": str(cache_dir / "manifest.json"),
    }


def _cache_ineligible_reason(dataset: Any) -> str | None:
    if getattr(dataset, "preprocessor", None) is not None:
        return (
            "Encoded sample cache requires a deterministic dataset without "
            "fetch-time preprocessors or augmentation."
        )
    curriculum_state = getattr(dataset, "_curriculum_state", None)
    if curriculum_state:
        return (
            "Encoded sample cache does not support non-empty curriculum_state "
            "because it can mutate fetch-time dataset behavior."
        )
    if str(getattr(dataset, "object_ordering", "") or "").strip().lower() != "sorted":
        return (
            "Encoded sample cache requires custom.object_ordering='sorted'; "
            "random object ordering is unsupported."
        )
    hard_sample_plan = getattr(dataset, "_hard_sample_plan", None)
    if hard_sample_plan:
        return (
            "Encoded sample cache does not support hard-sample or epoch-varying "
            "sample plans."
        )
    return None


class EncodedSampleCacheStore:
    def __init__(
        self,
        dataset: Any,
        *,
        request: Mapping[str, Any],
    ) -> None:
        cache_request = EncodedSampleCacheRequest.from_mapping(request)
        self._dataset = dataset
        self._request = cache_request.to_mapping()
        self._fingerprint = dict(cache_request.fingerprint)
        self._fingerprint_sha = _fingerprint_digest(self._fingerprint)
        self._root_dir = cache_request.root_dir
        self._cache_dir = self._root_dir / self._fingerprint_sha
        self._manifest_path = self._cache_dir / "manifest.json"
        self._lock_path = self._cache_dir / "build.lock"
        self._wait_timeout_s = cache_request.wait_timeout_s
        self._max_resident_shards = cache_request.max_resident_shards
        self._policy = cache_request.ineligible_policy
        self._split = cache_request.dataset_split
        self._dataset_jsonl = cache_request.dataset_jsonl
        self._status = "disabled"
        self._manifest: dict[str, Any] | None = None
        self._shards_by_index: dict[int, dict[str, Any]] = {}
        self._loaded_shards: OrderedDict[int, Any] = OrderedDict()
        self._loaded_shards_lock = threading.Lock()
        self._ensure_ready()

    @property
    def status(self) -> str:
        return self._status

    def info(self) -> dict[str, Any]:
        manifest = self._manifest or {}
        return {
            "enabled": True,
            "status": self._status,
            "policy": self._policy,
            "wait_timeout_s": self._wait_timeout_s,
            "max_resident_shards": self._max_resident_shards,
            "dataset_split": self._split,
            "dataset_jsonl": self._dataset_jsonl,
            "fingerprint": dict(self._fingerprint),
            "fingerprint_sha256": self._fingerprint_sha,
            "root_dir": str(self._root_dir),
            "cache_dir": str(self._cache_dir),
            "manifest_path": str(self._manifest_path),
            "num_samples": manifest.get("num_samples"),
            "shard_count": len(manifest.get("shards") or []),
            "payload_keys": list(manifest.get("payload_keys") or []),
            "artifact_version": manifest.get("version"),
        }

    @property
    def thread_safe(self) -> bool:
        return True

    def load_sample(self, base_idx: int) -> dict[str, Any]:
        manifest = self._manifest or {}
        shards = manifest.get("shards") or []
        if not isinstance(shards, list):
            raise TypeError(
                f"Encoded sample cache manifest has invalid shards payload: {self._manifest_path}"
            )
        shard_idx = int(base_idx) // int(manifest.get("shard_size") or 1)
        shard_meta = self._shards_by_index.get(shard_idx)
        if shard_meta is None:
            raise KeyError(
                f"Encoded sample cache has no shard for base_idx={base_idx} "
                f"(shard_idx={shard_idx})"
            )
        with self._loaded_shards_lock:
            shard_payload = self._loaded_shards.get(shard_idx)
            if shard_payload is not None:
                self._loaded_shards.move_to_end(shard_idx)

        if shard_payload is None:
            loaded_payload = _load_torch(self._cache_dir / str(shard_meta["file"]))
            with self._loaded_shards_lock:
                shard_payload = self._loaded_shards.get(shard_idx)
                if shard_payload is None:
                    shard_payload = loaded_payload
                    self._loaded_shards[shard_idx] = shard_payload
                self._loaded_shards.move_to_end(shard_idx)
                while len(self._loaded_shards) > int(self._max_resident_shards):
                    self._loaded_shards.popitem(last=False)

        samples = shard_payload.get("samples")
        if not isinstance(samples, list):
            raise TypeError(
                f"Encoded sample cache shard has invalid sample list: {shard_meta['file']}"
            )
        start = int(shard_payload.get("start") or 0)
        offset = int(base_idx) - start
        if offset < 0 or offset >= len(samples):
            raise IndexError(
                "Encoded sample cache shard index out of range for "
                f"base_idx={base_idx}, shard={shard_meta['file']}"
            )
        sample = samples[offset]
        if not isinstance(sample, dict):
            raise TypeError(
                f"Encoded sample cache sample payload must be a dict: {shard_meta['file']}"
            )
        return sample

    def _ensure_ready(self) -> None:
        existing = self._try_load_complete_manifest()
        if existing is not None:
            self._manifest = existing
            self._index_shards()
            self._status = "reused"
            logger.info(
                "Encoded sample cache reused: split=%s fingerprint_sha256=%s root=%s cache_dir=%s",
                self._split,
                self._fingerprint_sha,
                self._root_dir,
                self._cache_dir,
            )
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        is_writer = self._try_acquire_lock()
        if is_writer:
            try:
                self._manifest = self._build_cache()
                self._index_shards()
                self._status = "built"
                logger.info(
                    "Encoded sample cache built: split=%s fingerprint_sha256=%s root=%s cache_dir=%s",
                    self._split,
                    self._fingerprint_sha,
                    self._root_dir,
                    self._cache_dir,
                )
            finally:
                try:
                    self._lock_path.unlink()
                except FileNotFoundError:
                    pass
        else:
            manifest = _wait_for_complete_manifest(
                self._manifest_path, timeout_s=self._wait_timeout_s
            )
            self._validate_manifest(manifest)
            self._manifest = manifest
            self._index_shards()
            self._status = "reused"
            logger.info(
                "Encoded sample cache reused after wait: split=%s fingerprint_sha256=%s root=%s cache_dir=%s",
                self._split,
                self._fingerprint_sha,
                self._root_dir,
                self._cache_dir,
            )

    def _try_load_complete_manifest(self) -> dict[str, Any] | None:
        if not self._manifest_path.exists():
            return None
        manifest = _read_json(self._manifest_path)
        status = str(manifest.get("status") or "")
        if status != "complete":
            return None
        self._validate_manifest(manifest)
        return manifest

    def _try_acquire_lock(self) -> bool:
        try:
            fd = os.open(
                self._lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            return False

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "fingerprint_sha256": self._fingerprint_sha,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            handle.write("\n")
        return True

    def _build_cache(self) -> dict[str, Any]:
        num_samples = int(len(getattr(self._dataset, "base_records", [])))
        shard_size = _DEFAULT_ENCODED_SAMPLE_SHARD_SIZE
        started_at = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(
            self._manifest_path,
            {
                "version": _ENCODED_SAMPLE_CACHE_VERSION,
                "status": "building",
                "fingerprint": dict(self._fingerprint),
                "fingerprint_sha256": self._fingerprint_sha,
                "build_started_at": started_at,
                "dataset_split": self._split,
                "dataset_jsonl": self._dataset_jsonl,
            },
        )

        payload_keys: set[str] = set()
        shards: list[dict[str, Any]] = []
        try:
            for shard_idx, shard_start in enumerate(range(0, num_samples, shard_size)):
                shard_end = min(shard_start + shard_size, num_samples)
                samples: list[dict[str, Any]] = []
                for base_idx in range(shard_start, shard_end):
                    encoded = self._dataset._encode_base_record_for_cache(base_idx)  # type: ignore[attr-defined]
                    if not isinstance(encoded, dict):
                        encoded = dict(encoded)
                    encoded.pop("sample_id", None)
                    encoded.pop("dataset", None)
                    encoded.pop("base_idx", None)
                    payload_keys.update(str(k) for k in encoded.keys())
                    samples.append(encoded)
                file_name = f"shard-{int(shard_idx):05d}.pt"
                _write_torch_atomic(
                    self._cache_dir / file_name,
                    {
                        "version": _ENCODED_SAMPLE_CACHE_VERSION,
                        "fingerprint_sha256": self._fingerprint_sha,
                        "shard_index": int(shard_idx),
                        "start": int(shard_start),
                        "end": int(shard_end),
                        "samples": samples,
                    },
                )
                shards.append(
                    EncodedSampleShard(
                        shard_index=int(shard_idx),
                        file=file_name,
                        start=int(shard_start),
                        end=int(shard_end),
                        count=int(shard_end - shard_start),
                    ).to_mapping()
                )
        except Exception as exc:
            _write_json_atomic(
                self._manifest_path,
                {
                    "version": _ENCODED_SAMPLE_CACHE_VERSION,
                    "status": "error",
                    "fingerprint": dict(self._fingerprint),
                    "fingerprint_sha256": self._fingerprint_sha,
                    "dataset_split": self._split,
                    "dataset_jsonl": self._dataset_jsonl,
                    "build_started_at": started_at,
                    "build_failed_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                },
            )
            raise

        manifest = {
            "version": _ENCODED_SAMPLE_CACHE_VERSION,
            "status": "complete",
            "fingerprint": dict(self._fingerprint),
            "fingerprint_sha256": self._fingerprint_sha,
            "dataset_split": self._split,
            "dataset_jsonl": self._dataset_jsonl,
            "num_samples": num_samples,
            "shard_size": int(shard_size),
            "payload_keys": sorted(payload_keys),
            "shards": shards,
            "build_started_at": started_at,
            "build_completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(self._manifest_path, manifest)
        return manifest

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        typed_manifest = EncodedSampleCacheManifest.from_mapping(
            manifest,
            path=self._manifest_path,
        )
        if typed_manifest.version != _ENCODED_SAMPLE_CACHE_VERSION:
            raise ValueError(
                "Unsupported encoded sample cache manifest version: "
                f"{typed_manifest.version!r}"
            )
        observed = dict(typed_manifest.fingerprint)
        if observed != self._fingerprint:
            raise ValueError(
                "Encoded sample cache fingerprint mismatch at "
                f"{self._manifest_path}. expected_sha256={self._fingerprint_sha} "
                f"observed_sha256={_fingerprint_digest(observed)}"
            )

    def _index_shards(self) -> None:
        manifest = self._manifest or {}
        typed_manifest = EncodedSampleCacheManifest.from_mapping(
            manifest,
            path=self._manifest_path,
        )
        indexed: dict[int, dict[str, Any]] = {}
        for shard in typed_manifest.shards:
            indexed[shard.shard_index] = shard.to_mapping()
        self._shards_by_index = indexed


def setup_encoded_sample_cache_for_dataset(
    dataset: Any,
    request: Mapping[str, Any] | None,
) -> tuple[EncodedSampleCacheStore | None, dict[str, Any] | None]:
    if not request or not bool(request.get("enabled", False)):
        return None, None

    root_dir_raw = request.get("root_dir")
    if root_dir_raw is None:
        raise ValueError(
            "Encoded sample cache request must include a resolved root_dir when enabled."
        )

    cache_request = EncodedSampleCacheRequest.from_mapping(request)
    fingerprint = dict(cache_request.fingerprint)
    resolved_root = Path(str(root_dir_raw)).resolve()
    reason = _cache_ineligible_reason(dataset)
    policy = cache_request.ineligible_policy
    if reason is not None:
        if policy == "bypass":
            info = _build_bypass_info(
                request=request,
                resolved_root=resolved_root,
                fingerprint=fingerprint,
                reason=reason,
            )
            logger.warning(
                "Encoded sample cache bypassed: split=%s reason=%s fingerprint_sha256=%s root=%s cache_dir=%s",
                info["dataset_split"],
                info["reason"],
                info["fingerprint_sha256"],
                info["root_dir"],
                info["cache_dir"],
            )
            return None, info
        raise ValueError(reason)

    store = EncodedSampleCacheStore(dataset, request={**dict(request), "root_dir": str(resolved_root)})
    return store, store.info()
