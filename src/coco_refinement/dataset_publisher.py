"""Publish one standalone committed generation into the mutable training view.

The refinement runtime owns ``working.norm.jsonl``.  This module derives an
exact coord-token sibling, validates both candidates, and replaces only the
selected split under the processed COCO training root.  Images remain shared;
the published rows contain relative locators back to the existing image store.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from src.common.errors import DataContractError, RuntimeContractError
from src.data import iter_raw_examples
from src.label_studio_coco_refinement.models import Split, WorkingRow


PUBLISHER_VERSION = "coco-refinement-dataset-publisher-v1"
RECEIPT_NAME = "training.publish.receipt.json"
TRANSACTION_NAME = ".training.publish.transaction.json"
TARGET_RELATIVE_ROOT = Path("public_data/coco/rescale_32_1024_bbox_len12000")
IMAGE_RELATIVE_ROOT = Path("public_data/coco/rescale_32_1024_bbox/images")
DEFAULT_TRAINING_CONFIG = Path(
    "configs/coordexp_swift/prod/"
    "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_"
    "accelerate8_ebs128_4epoch.yaml"
)
_SHA256_LENGTH = 64


class DatasetPublishError(RuntimeContractError):
    """A committed generation cannot be safely published."""


@dataclass(frozen=True)
class TokenBudgetValidation:
    row_count: int
    max_total_tokens: int
    max_total_tokens_seen: int
    training_config_path: Path
    training_config_sha256: str
    training_config_fingerprint: str
    tokenizer_sha256: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "max_total_tokens": self.max_total_tokens,
            "max_total_tokens_seen": self.max_total_tokens_seen,
            "training_config_path": str(self.training_config_path),
            "training_config_sha256": self.training_config_sha256,
            "training_config_fingerprint": self.training_config_fingerprint,
            "tokenizer_sha256": self.tokenizer_sha256,
            "status": "passed",
        }


class TokenBudgetValidator(Protocol):
    def validate(
        self,
        coord_jsonl: Path,
        *,
        expected_row_count: int,
    ) -> TokenBudgetValidation: ...


class CoordExpSwiftTokenBudgetValidator:
    """Validate with the current loader, renderer, and Qwen encoding seam."""

    def __init__(
        self,
        training_config_path: str | Path,
        *,
        max_total_tokens: int = 12000,
    ) -> None:
        self.training_config_path = Path(training_config_path).expanduser().resolve()
        if (
            isinstance(max_total_tokens, bool)
            or not isinstance(max_total_tokens, int)
            or max_total_tokens <= 0
        ):
            raise DatasetPublishError(
                "max_total_tokens must be a positive integer",
                code="coco_refinement.publish_token_budget",
                context={"max_total_tokens": max_total_tokens},
            )
        self.max_total_tokens = max_total_tokens

    def validate(
        self,
        coord_jsonl: Path,
        *,
        expected_row_count: int,
    ) -> TokenBudgetValidation:
        from src.config.loader import load_train_config
        from src.qwen import encode_rendered_example, load_qwen_components
        from src.templates import render_example

        resolved = load_train_config(self.training_config_path)
        config = resolved.config
        if config.packing.global_max_length != self.max_total_tokens:
            raise DatasetPublishError(
                "training config token budget does not match the publication budget",
                code="coco_refinement.publish_token_budget_config",
                context={
                    "training_config": str(self.training_config_path),
                    "config_global_max_length": config.packing.global_max_length,
                    "publication_max_total_tokens": self.max_total_tokens,
                },
            )
        components = load_qwen_components(config, load_model=False)
        row_count = 0
        max_seen = 0
        for raw_example in iter_raw_examples(coord_jsonl):
            rendered = render_example(raw_example, config.template)
            try:
                encoded = encode_rendered_example(
                    raw_example,
                    rendered,
                    components=components,
                    processor_config=config.model.processor,
                    global_max_length=self.max_total_tokens,
                    materialize_image_pixels=False,
                )
            except DataContractError:
                raise
            except Exception as exc:
                raise DatasetPublishError(
                    "candidate row failed the current CoordExp-Swift encoding path",
                    code="coco_refinement.publish_token_validation",
                    context={
                        "example_id": raw_example.example_id,
                        "row_number": raw_example.source.row_number,
                        "max_total_tokens": self.max_total_tokens,
                    },
                    cause=exc,
                ) from exc
            row_count += 1
            max_seen = max(max_seen, encoded.input_length)
        if row_count != expected_row_count:
            raise DatasetPublishError(
                "token validator row count differs from the candidate inventory",
                code="coco_refinement.publish_token_row_count",
                context={"expected": expected_row_count, "actual": row_count},
            )
        return TokenBudgetValidation(
            row_count=row_count,
            max_total_tokens=self.max_total_tokens,
            max_total_tokens_seen=max_seen,
            training_config_path=self.training_config_path,
            training_config_sha256=_sha256_file(self.training_config_path),
            training_config_fingerprint=resolved.fingerprint,
            tokenizer_sha256=components.tokenizer_sha256,
        )


@dataclass(frozen=True)
class DatasetPublicationReceipt:
    split: Split
    generation: int
    runtime_root: Path
    working_path: Path
    working_sha256: str
    target_norm_path: Path
    target_norm_sha256: str
    target_coord_path: Path
    target_coord_sha256: str
    row_count: int
    object_count: int
    negative_object_count: int
    journal_path: Path
    journal_sha256: str
    token_budget: TokenBudgetValidation
    transaction_id: str
    published_at_utc: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "code": "coco_refinement.committed_generation_published",
            "publisher_version": PUBLISHER_VERSION,
            "split": self.split,
            "generation": self.generation,
            "runtime_root": str(self.runtime_root),
            "working": {
                "path": str(self.working_path),
                "sha256": self.working_sha256,
            },
            "outputs": {
                "norm": {
                    "path": str(self.target_norm_path),
                    "sha256": self.target_norm_sha256,
                    "size_bytes": self.target_norm_path.stat().st_size,
                },
                "coord": {
                    "path": str(self.target_coord_path),
                    "sha256": self.target_coord_sha256,
                    "size_bytes": self.target_coord_path.stat().st_size,
                },
            },
            "row_count": self.row_count,
            "object_count": self.object_count,
            "identity_authority": {
                "journal_path": str(self.journal_path),
                "journal_sha256": self.journal_sha256,
                "negative_object_count": self.negative_object_count,
                "status": "passed",
            },
            "loader_attestation": {
                "seam": "src.data.iter_raw_examples",
                "coord_row_count": self.row_count,
                "status": "passed",
            },
            "norm_schema_attestation": {
                "seam": "WorkingRow.from_mapping",
                "row_count": self.row_count,
                "status": "passed",
            },
            "token_budget": self.token_budget.to_artifact_dict(),
            "images": {
                "mode": "shared_relative_locator",
                "copied": False,
                "root": str(self.target_norm_path.parents[1] / "rescale_32_1024_bbox/images"),
            },
            "transaction_id": self.transaction_id,
            "published_at_utc": self.published_at_utc,
        }


@dataclass(frozen=True)
class _CandidatePair:
    norm_path: Path
    coord_path: Path
    working_sha256: str
    norm_sha256: str
    coord_sha256: str
    row_count: int
    object_count: int
    negative_object_count: int


class CommittedGenerationPublisher:
    """Validate and publish one selected split without copying shared images."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        runtime_root: str | Path,
        split: Split,
        token_budget_validator: TokenBudgetValidator,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).expanduser().resolve(strict=True)
        self.runtime_root = Path(runtime_root).expanduser().resolve(strict=True)
        if split not in ("train", "val"):
            raise DatasetPublishError(
                "split must be train or val",
                code="coco_refinement.publish_split",
                context={"split": split},
            )
        if not callable(getattr(token_budget_validator, "validate", None)):
            raise DatasetPublishError(
                "publisher requires a token budget validator",
                code="coco_refinement.publish_token_validator",
            )
        self.split = split
        self.token_budget_validator = token_budget_validator
        self.fault_injector = fault_injector
        self.split_root = self.runtime_root / split
        self.working_path = self.split_root / "working.norm.jsonl"
        self.manifest_path = self.split_root / "project.json"
        self.journal_path = self.split_root / "journal.jsonl"
        self.lock_path = self.split_root / ".commit.lock"
        self.process_lock_path = self.split_root / ".batch-process.lock"
        self.target_root = self.repository_root / TARGET_RELATIVE_ROOT
        self.image_root = self.repository_root / IMAGE_RELATIVE_ROOT
        self.target_norm_path = self.target_root / f"{split}.norm.jsonl"
        self.target_coord_path = self.target_root / f"{split}.coord.jsonl"
        self.receipt_path = self.split_root / RECEIPT_NAME
        self.transaction_path = self.split_root / TRANSACTION_NAME
        self._validate_paths()

    def publish(self) -> DatasetPublicationReceipt:
        # Keep the store generation fixed without holding the committed-file
        # lock through the expensive candidate/token pass.  The store worker
        # uses this same process barrier, while task navigation only needs the
        # short committed-file lock and therefore remains responsive.
        with self._batch_process_barrier():
            with self._committed_generation_lock():
                self._recover_incomplete_transaction()
                authority = self._read_generation_authority()
            candidates = self._build_candidates(authority)
            try:
                coord_loader_rows = sum(1 for _ in iter_raw_examples(candidates.coord_path))
                if coord_loader_rows != candidates.row_count:
                    raise DatasetPublishError(
                        "current loader row count differs from the candidate inventory",
                        code="coco_refinement.publish_loader_row_count",
                        context={
                            "expected": candidates.row_count,
                            "coord": coord_loader_rows,
                        },
                    )
                token_budget = self.token_budget_validator.validate(
                    candidates.coord_path,
                    expected_row_count=candidates.row_count,
                )
                with self._committed_generation_lock():
                    self._recheck_generation_authority(
                        authority, candidates.working_sha256
                    )
                    return self._publish_candidates(
                        candidates, authority, token_budget
                    )
            finally:
                candidates.norm_path.unlink(missing_ok=True)
                candidates.coord_path.unlink(missing_ok=True)

    def _validate_paths(self) -> None:
        if not self.repository_root.is_dir() or not self.runtime_root.is_dir():
            raise DatasetPublishError(
                "repository_root and runtime_root must be existing directories",
                code="coco_refinement.publish_root",
            )
        for path, kind in (
            (self.split_root, "runtime split"),
            (self.target_root, "processed target"),
            (self.image_root, "shared image"),
        ):
            if not path.is_dir():
                raise DatasetPublishError(
                    f"{kind} root does not exist",
                    code="coco_refinement.publish_root",
                    context={"path": str(path)},
                )
        for path in (
            self.working_path,
            self.manifest_path,
            self.journal_path,
            self.lock_path,
        ):
            if not path.is_file():
                raise DatasetPublishError(
                    "runtime generation authority is incomplete",
                    code="coco_refinement.publish_runtime_authority",
                    context={"path": str(path)},
                )

    @contextmanager
    def _committed_generation_lock(self) -> Iterator[None]:
        with self.lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DatasetPublishError(
                    "split is busy publishing or reconciling a Commit",
                    code="coco_refinement.publish_split_busy",
                    context={"split": self.split, "runtime_root": str(self.runtime_root)},
                    cause=exc,
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _batch_process_barrier(self) -> Iterator[None]:
        """Serialize the publisher with store recovery/batch processing."""

        with self.process_lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_generation_authority(self) -> dict[str, Any]:
        manifest = _read_json_object(self.manifest_path)
        if manifest.get("split") != self.split:
            raise DatasetPublishError(
                "runtime manifest belongs to another split",
                code="coco_refinement.publish_manifest_split",
            )
        generation = manifest.get("generation")
        task_count = manifest.get("task_count")
        working_sha256 = manifest.get("working_sha256")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
            or isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count <= 0
            or not _is_sha256(working_sha256)
        ):
            raise DatasetPublishError(
                "runtime manifest lacks a valid committed-generation identity",
                code="coco_refinement.publish_manifest_generation",
            )
        actual_hash = _sha256_file(self.working_path)
        if actual_hash != working_sha256:
            raise DatasetPublishError(
                "working JSONL does not match the committed manifest",
                code="coco_refinement.publish_working_hash",
                context={"expected": working_sha256, "actual": actual_hash},
            )
        return {
            "generation": generation,
            "task_count": task_count,
            "working_sha256": working_sha256,
            "manifest_sha256": _sha256_file(self.manifest_path),
            "journal_sha256": _sha256_file(self.journal_path),
        }

    def _recheck_generation_authority(
        self,
        authority: Mapping[str, Any],
        working_sha256: str,
    ) -> None:
        current = self._read_generation_authority()
        if current != dict(authority) or current["working_sha256"] != working_sha256:
            raise DatasetPublishError(
                "committed generation changed during candidate validation",
                code="coco_refinement.publish_generation_changed",
            )

    def _build_candidates(self, authority: Mapping[str, Any]) -> _CandidatePair:
        norm_fd, norm_name = tempfile.mkstemp(
            prefix=f".{self.split}.norm.candidate-",
            suffix=".jsonl",
            dir=self.target_root,
        )
        coord_fd, coord_name = tempfile.mkstemp(
            prefix=f".{self.split}.coord.candidate-",
            suffix=".jsonl",
            dir=self.target_root,
        )
        norm_path = Path(norm_name)
        coord_path = Path(coord_name)
        working_digest = hashlib.sha256()
        norm_digest = hashlib.sha256()
        coord_digest = hashlib.sha256()
        row_count = 0
        object_count = 0
        negative_object_count = 0
        keep = False
        try:
            with (
                self.working_path.open("rb") as source,
                os.fdopen(norm_fd, "wb") as norm_out,
                os.fdopen(coord_fd, "wb") as coord_out,
            ):
                for row_number, raw_line in enumerate(source, start=1):
                    working_digest.update(raw_line)
                    if not raw_line.strip():
                        raise DatasetPublishError(
                            "working JSONL contains a blank row",
                            code="coco_refinement.publish_blank_row",
                            context={"row_number": row_number},
                        )
                    try:
                        payload = json.loads(raw_line, parse_constant=_reject_json_constant)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise DatasetPublishError(
                            "working JSONL row is not valid strict JSON",
                            code="coco_refinement.publish_json_decode",
                            context={"row_number": row_number},
                            cause=exc,
                        ) from exc
                    row = WorkingRow.from_mapping(
                        payload,
                        field=f"working[{row_number}]",
                    ).validate_for_split(self.split)
                    public_locator = os.path.relpath(
                        self.image_root / f"{self.split}2017" / f"{row.image_id:012d}.jpg",
                        start=self.target_root,
                    )
                    norm_payload = row.to_json_dict(coord_tokens=False)
                    norm_payload["images"] = [public_locator]
                    coord_payload = row.to_json_dict(coord_tokens=True)
                    coord_payload["images"] = [public_locator]
                    norm_encoded = _jsonl_bytes(norm_payload)
                    coord_encoded = _jsonl_bytes(coord_payload)
                    norm_out.write(norm_encoded)
                    coord_out.write(coord_encoded)
                    norm_digest.update(norm_encoded)
                    coord_digest.update(coord_encoded)
                    row_count += 1
                    object_count += len(row.objects)
                    negative_object_count += sum(
                        1 for obj in row.objects if obj.coco_ann_id < 0
                    )
                norm_out.flush()
                coord_out.flush()
                os.fsync(norm_out.fileno())
                os.fsync(coord_out.fileno())
            if row_count != authority["task_count"]:
                raise DatasetPublishError(
                    "working row count differs from the committed task inventory",
                    code="coco_refinement.publish_task_count",
                    context={"expected": authority["task_count"], "actual": row_count},
                )
            observed_working_hash = working_digest.hexdigest()
            if observed_working_hash != authority["working_sha256"]:
                raise DatasetPublishError(
                    "working JSONL changed while candidates were generated",
                    code="coco_refinement.publish_working_changed",
                )
            keep = True
            return _CandidatePair(
                norm_path=norm_path,
                coord_path=coord_path,
                working_sha256=observed_working_hash,
                norm_sha256=norm_digest.hexdigest(),
                coord_sha256=coord_digest.hexdigest(),
                row_count=row_count,
                object_count=object_count,
                negative_object_count=negative_object_count,
            )
        finally:
            if not keep:
                try:
                    os.close(norm_fd)
                except OSError:
                    pass
                try:
                    os.close(coord_fd)
                except OSError:
                    pass
                norm_path.unlink(missing_ok=True)
                coord_path.unlink(missing_ok=True)

    def _publish_candidates(
        self,
        candidates: _CandidatePair,
        authority: Mapping[str, Any],
        token_budget: TokenBudgetValidation,
    ) -> DatasetPublicationReceipt:
        transaction_id = uuid.uuid4().hex
        backup_norm = self.target_root / f".{self.split}.norm.rollback-{transaction_id}"
        backup_coord = self.target_root / f".{self.split}.coord.rollback-{transaction_id}"
        backup_receipt = self.split_root / f".{RECEIPT_NAME}.rollback-{transaction_id}"
        state = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "status": "preparing",
            "split": self.split,
            "targets": [
                _transaction_target(self.target_norm_path, backup_norm),
                _transaction_target(self.target_coord_path, backup_coord),
                _transaction_target(self.receipt_path, backup_receipt),
            ],
            "candidates": [str(candidates.norm_path), str(candidates.coord_path)],
        }
        _atomic_write_json(self.transaction_path, state)
        try:
            for item in state["targets"]:
                if item["existed"]:
                    os.link(item["path"], item["backup"])
            state["status"] = "prepared"
            _atomic_write_json(self.transaction_path, state)
            self._inject("after_prepare")
            os.replace(candidates.norm_path, self.target_norm_path)
            self._inject("after_norm_replace")
            os.replace(candidates.coord_path, self.target_coord_path)
            _fsync_directory(self.target_root)
            self._inject("after_coord_replace")
            receipt = DatasetPublicationReceipt(
                split=self.split,
                generation=int(authority["generation"]),
                runtime_root=self.runtime_root,
                working_path=self.working_path,
                working_sha256=candidates.working_sha256,
                target_norm_path=self.target_norm_path,
                target_norm_sha256=candidates.norm_sha256,
                target_coord_path=self.target_coord_path,
                target_coord_sha256=candidates.coord_sha256,
                row_count=candidates.row_count,
                object_count=candidates.object_count,
                negative_object_count=candidates.negative_object_count,
                journal_path=self.journal_path,
                journal_sha256=str(authority["journal_sha256"]),
                token_budget=token_budget,
                transaction_id=transaction_id,
                published_at_utc=datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
            )
            _atomic_write_json(self.receipt_path, receipt.to_artifact_dict())
            self._inject("after_receipt_replace")
            state["status"] = "committed"
            _atomic_write_json(self.transaction_path, state)
            self._cleanup_transaction(state)
            return receipt
        except BaseException:
            if state["status"] == "preparing":
                self._cleanup_transaction(state)
            else:
                self._rollback_transaction(state)
            raise

    def _recover_incomplete_transaction(self) -> None:
        if not self.transaction_path.exists():
            return
        state = _read_json_object(self.transaction_path)
        self._validate_transaction_state(state)
        if state["status"] == "committed":
            self._cleanup_transaction(state)
        elif state["status"] == "preparing":
            self._cleanup_transaction(state)
        else:
            self._rollback_transaction(state)

    def _validate_transaction_state(self, state: Mapping[str, Any]) -> None:
        transaction_id = state.get("transaction_id")
        if (
            state.get("schema_version") != 1
            or state.get("split") != self.split
            or state.get("status") not in {"preparing", "prepared", "committed"}
            or not isinstance(state.get("targets"), list)
            or not isinstance(transaction_id, str)
            or not transaction_id
            or any(char not in "0123456789abcdefghijklmnopqrstuvwxyz-" for char in transaction_id)
        ):
            raise DatasetPublishError(
                "publication recovery state is invalid",
                code="coco_refinement.publish_transaction_state",
            )
        expected_targets = {
            self.target_norm_path.resolve(),
            self.target_coord_path.resolve(),
            self.receipt_path.resolve(),
        }
        observed_targets = {
            Path(item.get("path", "")).resolve()
            for item in state["targets"]
            if isinstance(item, Mapping)
        }
        if observed_targets != expected_targets:
            raise DatasetPublishError(
                "publication recovery state names unexpected targets",
                code="coco_refinement.publish_transaction_targets",
            )
        expected_backups = {
            self.target_norm_path.resolve(): (
                self.target_root / f".{self.split}.norm.rollback-{transaction_id}"
            ).resolve(),
            self.target_coord_path.resolve(): (
                self.target_root / f".{self.split}.coord.rollback-{transaction_id}"
            ).resolve(),
            self.receipt_path.resolve(): (
                self.split_root / f".{RECEIPT_NAME}.rollback-{transaction_id}"
            ).resolve(),
        }
        for item in state["targets"]:
            if not isinstance(item, Mapping):
                raise DatasetPublishError(
                    "publication recovery target entry is invalid",
                    code="coco_refinement.publish_transaction_targets",
                )
            path = Path(str(item.get("path", ""))).resolve()
            backup = Path(str(item.get("backup", ""))).resolve()
            if backup != expected_backups.get(path) or not isinstance(
                item.get("existed"), bool
            ):
                raise DatasetPublishError(
                    "publication recovery state names an unexpected rollback file",
                    code="coco_refinement.publish_transaction_backups",
                )
        candidates = state.get("candidates")
        if not isinstance(candidates, list):
            raise DatasetPublishError(
                "publication recovery candidates must be a list",
                code="coco_refinement.publish_transaction_candidates",
            )
        for candidate in candidates:
            path = Path(str(candidate)).resolve()
            if path.parent != self.target_root or not (
                path.name.startswith(f".{self.split}.norm.candidate-")
                or path.name.startswith(f".{self.split}.coord.candidate-")
            ):
                raise DatasetPublishError(
                    "publication recovery state names an unexpected candidate",
                    code="coco_refinement.publish_transaction_candidates",
                )

    def _rollback_transaction(self, state: Mapping[str, Any]) -> None:
        self._validate_transaction_state(state)
        for item in state["targets"]:
            path = Path(item["path"])
            backup = Path(item["backup"])
            if item["existed"]:
                if not backup.exists():
                    raise DatasetPublishError(
                        "publication rollback backup is missing",
                        code="coco_refinement.publish_rollback_backup",
                        context={"path": str(path), "backup": str(backup)},
                    )
                os.replace(backup, path)
            else:
                path.unlink(missing_ok=True)
        _fsync_directory(self.target_root)
        _fsync_directory(self.split_root)
        self._cleanup_transaction(state)

    def _cleanup_transaction(self, state: Mapping[str, Any]) -> None:
        for item in state.get("targets", []):
            if isinstance(item, Mapping):
                Path(str(item.get("backup", ""))).unlink(missing_ok=True)
        for candidate in state.get("candidates", []):
            Path(str(candidate)).unlink(missing_ok=True)
        self.transaction_path.unlink(missing_ok=True)
        _fsync_directory(self.target_root)
        _fsync_directory(self.split_root)

    def _inject(self, stage: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(stage)


def _transaction_target(path: Path, backup: Path) -> dict[str, Any]:
    if path.exists() and not path.is_file():
        raise DatasetPublishError(
            "publication target must be a regular file when present",
            code="coco_refinement.publish_target_type",
            context={"path": str(path)},
        )
    return {"path": str(path), "backup": str(backup), "existed": path.exists()}


def _jsonl_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DatasetPublishError(
            "publication authority is not readable strict JSON",
            code="coco_refinement.publish_json_authority",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(value, dict):
        raise DatasetPublishError(
            "publication authority must be a JSON object",
            code="coco_refinement.publish_json_authority",
            context={"path": str(path)},
        )
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        encoded = _jsonl_bytes(payload)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _reject_json_constant(value: str) -> None:
    raise DatasetPublishError(
        "NaN and Infinity are not valid publication JSON",
        code="coco_refinement.publish_json_constant",
        context={"constant": value},
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CommittedGenerationPublisher",
    "CoordExpSwiftTokenBudgetValidator",
    "DatasetPublicationReceipt",
    "DatasetPublishError",
    "TokenBudgetValidation",
    "TokenBudgetValidator",
]
