#!/usr/bin/env python3
"""Pure detached trajectory-credit projection and packed score-function loss.

Task-2 acquisitions intentionally contain policy tokens, not parser objects.  The
small parsed-acquisition records below bind CPU parser output to that immutable
acquisition before matching or credit assignment.  Projection imports neither a
model nor an optimizer; PyTorch is imported lazily only by the differentiable
loss helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import weakref

from scripts.research.analyze_human13_k_union import (
    _manifest_sha256,
    _match_prefix,
)
from scripts.research.build_human13_k_union_manifest import (
    PANEL_PATH,
    Human13KUnionManifest,
    ImageRecord,
)
from scripts.research.collect_human13_rp_crossover import (
    AdmittedPublication,
    AcquisitionExecution,
    PublicationBinding,
)
from scripts.research.compare_clean_rollout_owner_coverage import iou_xyxy
from scripts.research.human13_k_trajectory_contracts import AcquisitionGroup
from src.data.geometry import coord_bins_to_pixel_xyxy, parse_coord_token
from src.eval.detection_categories import normalize_coco_category_name
from src.inference.parsing import (
    PARSER_ID,
    PARSER_POLICY,
    parse_compact_object_box_closed,
)
from src.templates.renderer import OBJECT_REF_END_TOKEN, OBJECT_REF_START_TOKEN


SCHEMA_VERSION = "human13_trajectory_credit_ledger.v4"
ACQUISITION_SCHEMA_VERSION = "human13_trajectory_credit_acquisition.v1"
PANEL_ACQUISITION_SCHEMA_VERSION = "human13_trajectory_credit_panel_acquisition.v1"
PARSER_PROJECTION_SCHEMA_VERSION = "human13_canonical_parser_projection.v3"
FIXED_SCIENTIFIC_K = 16
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_TOKENIZER_FACTORY_MARKER = object()
_SCIENTIFIC_LEDGER_FACTORY_MARKER = object()
_SCIENTIFIC_LEDGER_ADMISSIONS: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}


def _canonical_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(value)).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_mapping(value: object, *, field: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field} schema differs")
    return value


def _valid_box(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    x1, y1, x2, y2 = result
    if x1 >= x2 or y1 >= y2:
        return None
    return result  # type: ignore[return-value]


@dataclass(frozen=True, init=False)
class CanonicalTokenizerDecodeAdapter:
    """Factory-only attestation over one real loaded Transformers tokenizer."""

    _decode_backend: Any
    _tokenizer_json_bytes: bytes
    _factory_marker: object
    _base_model_path: Path
    _runtime_receipt_json: str
    tokenizer_id: str
    tokenizer_sha256: str
    implementation_id: str
    tokenizer_class: str
    tokenizer_module: str
    tokenizer_config_sha256: str
    runtime_receipt_sha256: str
    backend_tokenizer_sha256: str
    tokenizers_package_version: str
    decode_backend_id: str
    decode_policy: str = (
        "tokenizers-snapshot-skip-special-false-serialized-spacing-concat.v3"
    )

    @classmethod
    def from_runtime(
        cls,
        *,
        tokenizer: object,
        base_model_path: str | Path,
        runtime_receipt: Mapping[str, Any],
        manifest: Human13KUnionManifest,
        publication: AdmittedPublication,
    ) -> CanonicalTokenizerDecodeAdapter:
        attestation = _inspect_tokenizer_runtime(
            tokenizer=tokenizer,
            base_model_path=base_model_path,
            runtime_receipt=runtime_receipt,
            manifest=manifest,
            publication=publication,
        )
        instance = object.__new__(cls)
        for field, value in attestation.items():
            object.__setattr__(instance, field, value)
        object.__setattr__(
            instance,
            "_factory_marker",
            _VERIFIED_TOKENIZER_FACTORY_MARKER,
        )
        return instance

    @classmethod
    def from_qwen_components(
        cls,
        components: object,
        *,
        manifest: Human13KUnionManifest,
        publication: AdmittedPublication,
    ) -> CanonicalTokenizerDecodeAdapter:
        from src.qwen.runtime_loading import QwenComponents, _processor_identity
        from src.qwen.tokens import validate_qwen_token_identity

        if type(components) is not QwenComponents:
            raise ValueError("verified tokenizer factory requires exact QwenComponents")
        if getattr(components.processor, "tokenizer", None) is not components.tokenizer:
            raise ValueError(
                "Qwen components processor.tokenizer is not the captured tokenizer object"
            )
        if (
            _processor_identity(components.processor, components.tokenizer)
            != components.processor_identity
        ):
            raise ValueError("Qwen components processor runtime identity differs")
        if (
            validate_qwen_token_identity(components.tokenizer)
            != components.token_identity
        ):
            raise ValueError("Qwen components token runtime identity differs")
        return cls.from_runtime(
            tokenizer=components.tokenizer,
            base_model_path=components.base_model_path,
            runtime_receipt=components.to_artifact_dict(),
            manifest=manifest,
            publication=publication,
        )

    @property
    def base_model_path(self) -> Path:
        return self._base_model_path

    def verify_for(
        self,
        *,
        manifest: Human13KUnionManifest,
        publication: AdmittedPublication,
    ) -> None:
        if (
            getattr(self, "_factory_marker", None)
            is not _VERIFIED_TOKENIZER_FACTORY_MARKER
        ):
            raise ValueError("tokenizer adapter lacks a factory-verified attestation")
        _verify_tokenizer_snapshot(
            self,
            manifest=manifest,
            publication=publication,
        )

    def decode(self, token_ids: Sequence[int]) -> tuple[str, tuple[str, ...]]:
        ids = tuple(_integer(token, field="generated token id") for token in token_ids)
        token_texts = tuple(
            str(self._decode_backend.decode([token], skip_special_tokens=False))
            for token in ids
        )
        sequence_text = str(
            self._decode_backend.decode(list(ids), skip_special_tokens=False)
        )
        if "".join(token_texts) != sequence_text:
            raise ValueError(
                "per-token decoded text differs from canonical sequence decoding"
            )
        return sequence_text, token_texts

    def identity_dict(self) -> dict[str, str]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "implementation_id": self.implementation_id,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_module": self.tokenizer_module,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "runtime_receipt_sha256": self.runtime_receipt_sha256,
            "backend_tokenizer_sha256": self.backend_tokenizer_sha256,
            "tokenizers_package_version": self.tokenizers_package_version,
            "decode_backend_id": self.decode_backend_id,
            "decode_policy": self.decode_policy,
        }


def _revalidate_admitted_publication(value: object) -> AdmittedPublication:
    # Load Task 2's replay artifact type only at the admission boundary; the
    # detached ledger itself depends only on the immutable receipt fields.
    from scripts.research.human13_rp_policy import AcquisitionGroupParityReceipt

    if type(value) is not AdmittedPublication:
        raise ValueError("trajectory credit requires exact Task2 AdmittedPublication")
    publication = value
    execution = AcquisitionExecution(
        plan=publication.execution.plan,
        plan_sha256=publication.execution.plan_sha256,
        plan_request_ids=publication.execution.plan_request_ids,
        native_batch_receipts=publication.execution.native_batch_receipts,
        group=AcquisitionGroup.from_dict(publication.execution.group.to_dict()),
    )
    # ``to_dict`` is an immutable in-process view and intentionally preserves
    # tuples.  The artifact loader is stricter and accepts only JSON arrays, so
    # cross the real JSON boundary while revalidating the admitted publication.
    binding = PublicationBinding.from_dict(
        json.loads(_canonical_json(publication.binding.to_dict()))
    )
    replayed = AcquisitionGroup.from_dict(publication.replayed_group.to_dict())
    parity = AcquisitionGroupParityReceipt.from_dict(
        publication.parity_receipt.to_dict()
    )
    return AdmittedPublication(binding, execution, replayed, parity)


def _inspect_tokenizer_runtime(
    *,
    tokenizer: object,
    base_model_path: str | Path,
    runtime_receipt: Mapping[str, Any],
    manifest: Human13KUnionManifest,
    publication: AdmittedPublication,
) -> dict[str, Any]:
    """Derive tokenizer identity from the installed class, object, and files."""

    import inspect
    from importlib import metadata

    import transformers
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    from src.config.fingerprint import sha256_file
    from src.qwen.tokens import tokenizer_config_class_matches_runtime

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "verified tokenizer factory requires a real Transformers tokenizer"
        )
    tokenizer_class = type(tokenizer)
    transformers_root = Path(transformers.__file__).resolve().parent
    try:
        class_path = Path(inspect.getfile(tokenizer_class)).resolve()
    except (OSError, TypeError) as exc:
        raise ValueError(
            "Transformers tokenizer class has no installed source"
        ) from exc
    if not class_path.is_relative_to(transformers_root):
        raise ValueError(
            "verified tokenizer must use an installed Transformers concrete class"
        )
    try:
        decode_path = Path(inspect.getfile(tokenizer_class.decode)).resolve()
    except (OSError, TypeError) as exc:
        raise ValueError("tokenizer decode implementation cannot be inspected") from exc
    if not decode_path.is_relative_to(transformers_root):
        raise ValueError(
            "tokenizer decode implementation is outside installed Transformers"
        )
    if "decode" in vars(tokenizer):
        raise ValueError(
            "verified tokenizer object carries an instance decode override"
        )

    try:
        resolved_base = Path(base_model_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("sealed base-model directory is unavailable") from exc
    try:
        resolved_name_or_path = (
            Path(str(tokenizer.name_or_path)).expanduser().resolve(strict=True)
        )
    except (AttributeError, FileNotFoundError, OSError) as exc:
        raise ValueError(
            "tokenizer name_or_path is not a resolved local directory"
        ) from exc
    if resolved_name_or_path != resolved_base:
        raise ValueError(
            "tokenizer name_or_path differs from the sealed base-model directory"
        )

    if not isinstance(manifest, Human13KUnionManifest):
        raise ValueError("manifest must be a Human13KUnionManifest")
    admitted = _revalidate_admitted_publication(publication)
    manifest_sha256 = _manifest_sha256(manifest)
    if admitted.replayed_group.identity.manifest_sha256 != manifest_sha256:
        raise ValueError("Task2 publication manifest lineage differs")

    tokenizer_path = resolved_base / "tokenizer.json"
    tokenizer_config_path = resolved_base / "tokenizer_config.json"
    try:
        tokenizer_json_bytes = tokenizer_path.read_bytes()
        tokenizer_sha256 = sha256_file(tokenizer_path)
        tokenizer_config_sha256 = sha256_file(tokenizer_config_path)
        tokenizer_payload = json.loads(tokenizer_json_bytes.decode("utf-8"))
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            "sealed tokenizer runtime files are unavailable or invalid"
        ) from exc
    if tokenizer_sha256 != manifest.binding.surface.tokenizer_sha256:
        raise ValueError("tokenizer.json SHA-256 differs from the manifest")

    try:
        receipt = json.loads(_canonical_json(dict(runtime_receipt)))
    except (TypeError, ValueError) as exc:
        raise ValueError("tokenizer runtime receipt is not canonical JSON") from exc
    receipt = _strict_mapping(
        receipt,
        field="Qwen runtime-loading receipt",
        keys={
            "base_model_path",
            "base_config_sha256",
            "tokenizer_sha256",
            "load_model",
            "attn_implementation",
            "processor",
            "model",
            "tokens",
            "package_versions",
            "runtime_patches",
        },
    )
    if receipt.get("base_model_path") != str(resolved_base):
        raise ValueError("tokenizer runtime receipt base-model path differs")
    try:
        base_config_sha256 = sha256_file(resolved_base / "config.json")
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("sealed Qwen config identity file is unavailable") from exc
    if receipt.get("base_config_sha256") != base_config_sha256:
        raise ValueError("tokenizer runtime receipt base config SHA-256 differs")
    if receipt.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("tokenizer runtime receipt tokenizer SHA-256 differs")
    processor_receipt = receipt.get("processor")
    actual_class_name = tokenizer_class.__name__
    actual_module = tokenizer_class.__module__
    if (
        not isinstance(processor_receipt, Mapping)
        or processor_receipt.get("tokenizer_class") != actual_class_name
    ):
        raise ValueError("tokenizer runtime receipt tokenizer class differs")
    if not tokenizer_config_class_matches_runtime(tokenizer_config.get("tokenizer_class"), tokenizer_class):
        raise ValueError("tokenizer class differs from tokenizer_config identity")
    if manifest.binding.surface.tokenizer_class != actual_class_name:
        raise ValueError("tokenizer class differs from manifest identity")
    package_versions = receipt.get("package_versions")
    if not isinstance(package_versions, Mapping) or package_versions.get(
        "transformers"
    ) != metadata.version("transformers"):
        raise ValueError("tokenizer runtime receipt Transformers version differs")
    tokenizers_version = metadata.version("tokenizers")
    if package_versions.get("tokenizers") != tokenizers_version:
        raise ValueError("tokenizer runtime receipt tokenizers version differs")

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if type(backend) is not Tokenizer:
        raise ValueError(
            "verified tokenizer does not capture the installed tokenizers backend"
        )
    try:
        backend_payload = json.loads(backend.to_str())
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "loaded tokenizer backend cannot be canonically inspected"
        ) from exc
    if backend_payload != tokenizer_payload:
        raise ValueError("loaded tokenizer backend differs from tokenizer.json")
    try:
        decode_backend = Tokenizer.from_str(tokenizer_json_bytes.decode("utf-8"))
        decode_backend_payload = json.loads(decode_backend.to_str())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "sealed tokenizer.json cannot reconstruct the private decode backend"
        ) from exc
    if decode_backend_payload != tokenizer_payload:
        raise ValueError("private decode backend differs from sealed tokenizer.json")

    tokenizer_id = str(resolved_base)
    for group in (admitted.execution.group, admitted.replayed_group):
        if group.identity.tokenizer_id != tokenizer_id:
            raise ValueError(
                "Task2 tokenizer identity differs from the verified runtime path"
            )
    return {
        "_decode_backend": decode_backend,
        "_tokenizer_json_bytes": tokenizer_json_bytes,
        "_base_model_path": resolved_base,
        "_runtime_receipt_json": _canonical_json(receipt),
        "tokenizer_id": tokenizer_id,
        "tokenizer_sha256": tokenizer_sha256,
        "implementation_id": f"tokenizers.Tokenizer@{tokenizers_version}",
        "tokenizer_class": actual_class_name,
        "tokenizer_module": actual_module,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "runtime_receipt_sha256": _sha256(receipt),
        "backend_tokenizer_sha256": _sha256(decode_backend_payload),
        "tokenizers_package_version": tokenizers_version,
        "decode_backend_id": "tokenizers.Tokenizer.from_str(tokenizer.json)",
        "decode_policy": (
            "tokenizers-snapshot-skip-special-false-serialized-spacing-concat.v3"
        ),
    }


def _verify_tokenizer_snapshot(
    adapter: CanonicalTokenizerDecodeAdapter,
    *,
    manifest: Human13KUnionManifest,
    publication: AdmittedPublication,
) -> None:
    """Revalidate the sealed private decoder without consulting the HF object."""

    from importlib import metadata

    from tokenizers import Tokenizer

    admitted = _revalidate_admitted_publication(publication)
    manifest_sha256 = _manifest_sha256(manifest)
    if admitted.replayed_group.identity.manifest_sha256 != manifest_sha256:
        raise ValueError("Task2 publication manifest lineage differs")
    if manifest.binding.surface.tokenizer_sha256 != adapter.tokenizer_sha256:
        raise ValueError("verified tokenizer snapshot differs from manifest SHA-256")
    if manifest.binding.surface.tokenizer_class != adapter.tokenizer_class:
        raise ValueError("verified tokenizer snapshot differs from manifest class")
    if str(adapter._base_model_path) != adapter.tokenizer_id:
        raise ValueError("verified tokenizer snapshot base-model identity differs")
    for group in (admitted.execution.group, admitted.replayed_group):
        if group.identity.tokenizer_id != adapter.tokenizer_id:
            raise ValueError("Task2 tokenizer identity differs from verified snapshot")

    if hashlib.sha256(adapter._tokenizer_json_bytes).hexdigest() != (
        adapter.tokenizer_sha256
    ):
        raise ValueError("private tokenizer.json snapshot SHA-256 differs")
    try:
        file_payload = json.loads(adapter._tokenizer_json_bytes.decode("utf-8"))
        if type(adapter._decode_backend) is not Tokenizer:
            raise ValueError("private decoder is not the exact tokenizers backend")
        backend_payload = json.loads(adapter._decode_backend.to_str())
    except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private tokenizer snapshot cannot be revalidated") from exc
    if backend_payload != file_payload:
        raise ValueError("private decoder differs from tokenizer.json snapshot")
    if _sha256(backend_payload) != adapter.backend_tokenizer_sha256:
        raise ValueError("private decoder serialization SHA-256 differs")

    tokenizers_version = metadata.version("tokenizers")
    if adapter.tokenizers_package_version != tokenizers_version:
        raise ValueError("private decoder tokenizers package version differs")
    if adapter.implementation_id != f"tokenizers.Tokenizer@{tokenizers_version}":
        raise ValueError("private decoder implementation identity differs")
    if adapter.decode_backend_id != "tokenizers.Tokenizer.from_str(tokenizer.json)":
        raise ValueError("private decoder construction identity differs")
    if adapter.decode_policy != (
        "tokenizers-snapshot-skip-special-false-serialized-spacing-concat.v3"
    ):
        raise ValueError("private decoder option policy differs")

    receipt = json.loads(adapter._runtime_receipt_json)
    if _sha256(receipt) != adapter.runtime_receipt_sha256:
        raise ValueError("private tokenizer runtime receipt SHA-256 differs")
    if (
        receipt.get("base_model_path") != adapter.tokenizer_id
        or receipt.get("tokenizer_sha256") != adapter.tokenizer_sha256
    ):
        raise ValueError("private tokenizer runtime receipt identity differs")
    package_versions = receipt.get("package_versions")
    if (
        not isinstance(package_versions, Mapping)
        or package_versions.get("tokenizers") != tokenizers_version
    ):
        raise ValueError("private tokenizer runtime package receipt differs")


@dataclass(frozen=True)
class TrajectoryCreditPanelAcquisition:
    """One exact RP/seed cell of admitted Task-2 K16 publications."""

    publications: tuple[AdmittedPublication, ...]

    def __post_init__(self) -> None:
        publications = tuple(
            _revalidate_admitted_publication(publication)
            for publication in self.publications
        )
        if not publications:
            raise ValueError("trajectory-credit panel requires admitted publications")
        image_ids = tuple(
            publication.execution.plan.image_id for publication in publications
        )
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("trajectory-credit panel image plans must be unique")
        repetition_penalties = {
            publication.execution.plan.repetition_penalty
            for publication in publications
        }
        if len(repetition_penalties) != 1:
            raise ValueError("trajectory-credit panel must use one common training RP")
        seed_groups = {
            publication.execution.plan.seed_group_id for publication in publications
        }
        if len(seed_groups) != 1:
            raise ValueError("trajectory-credit panel must use one common seed group")
        static_surfaces = {
            (
                publication.replayed_group.identity.source_sha256,
                publication.replayed_group.identity.manifest_sha256,
                publication.replayed_group.identity.model_id,
                publication.replayed_group.identity.tokenizer_id,
                publication.replayed_group.identity.processor_id,
                publication.replayed_group.policy_contract.temperature,
                publication.replayed_group.policy_contract.processor_order,
                publication.replayed_group.policy_contract.sampler_backend_id,
                publication.replayed_group.policy_contract.top_p,
                publication.replayed_group.policy_contract.top_k,
            )
            for publication in publications
        }
        if len(static_surfaces) != 1:
            raise ValueError(
                "trajectory-credit panel Source/manifest/policy surface differs"
            )
        if any(
            len(publication.replayed_group.trajectories) != FIXED_SCIENTIFIC_K
            for publication in publications
        ):
            raise ValueError("trajectory-credit scientific groups require exact K16")
        object.__setattr__(self, "publications", publications)

    @property
    def training_repetition_penalty(self) -> float:
        return self.publications[0].execution.plan.repetition_penalty

    @property
    def seed_group_id(self) -> str:
        return self.publications[0].execution.plan.seed_group_id

    def _preimage(self) -> dict[str, Any]:
        return {
            "schema_version": PANEL_ACQUISITION_SCHEMA_VERSION,
            "training_repetition_penalty": self.training_repetition_penalty,
            "seed_group_id": self.seed_group_id,
            "publications": [
                {
                    "image_id": publication.execution.plan.image_id,
                    "plan_sha256": publication.execution.plan_sha256,
                    "native_receipts_sha256": publication.binding.native_receipts_sha256,
                    "sampled_group_sha256": publication.binding.sampled_group_sha256,
                    "replayed_group_sha256": publication.binding.replayed_group_sha256,
                    "parity_receipt_sha256": publication.binding.parity_receipt_sha256,
                    "tolerance_sha256": publication.binding.tolerance_sha256,
                }
                for publication in self.publications
            ],
        }

    @property
    def content_sha256(self) -> str:
        return _sha256(self._preimage())


def _construct_trajectory_credit_ledger(
    *,
    source_sha256: str,
    manifest_sha256: str,
    acquisition_sha256: str,
    logical_image_count: int,
    logical_k: int,
    images: tuple[ImageCreditLedger, ...],
    training_repetition_penalty: float | None = None,
    seed_group_id: str | None = None,
    admit_scientific: bool = False,
) -> TrajectoryCreditLedger:
    ledger = object.__new__(TrajectoryCreditLedger)
    for field, value in (
        ("source_sha256", source_sha256),
        ("manifest_sha256", manifest_sha256),
        ("acquisition_sha256", acquisition_sha256),
        ("logical_image_count", logical_image_count),
        ("logical_k", logical_k),
        ("images", images),
        ("training_repetition_penalty", None),
        ("seed_group_id", None),
        ("admission_sha256", None),
        ("_factory_marker", None),
    ):
        object.__setattr__(ledger, field, value)
    ledger._validate()
    if not admit_scientific:
        return ledger
    if training_repetition_penalty is None or seed_group_id is None:
        raise ValueError("scientific ledger admission lineage is incomplete")
    if logical_k != FIXED_SCIENTIFIC_K:
        raise ValueError("scientific ledger admission requires exact K16")
    if any(
        any(
            value is None
            for value in (
                image.plan_sha256,
                image.native_receipts_sha256,
                image.parity_receipt_sha256,
                image.parser_projection_sha256,
            )
        )
        for image in ledger.images
    ):
        raise ValueError("scientific ledger admission requires complete image lineage")
    object.__setattr__(
        ledger, "training_repetition_penalty", training_repetition_penalty
    )
    object.__setattr__(ledger, "seed_group_id", seed_group_id)
    object.__setattr__(ledger, "_factory_marker", _SCIENTIFIC_LEDGER_FACTORY_MARKER)
    admission_sha256 = _sha256(ledger._admission_preimage())
    object.__setattr__(ledger, "admission_sha256", admission_sha256)
    ledger._validate()
    ledger_id = id(ledger)

    def _discard(reference: weakref.ReferenceType[Any]) -> None:
        current = _SCIENTIFIC_LEDGER_ADMISSIONS.get(ledger_id)
        if current is not None and current[0] is reference:
            _SCIENTIFIC_LEDGER_ADMISSIONS.pop(ledger_id, None)

    reference = weakref.ref(ledger, _discard)
    _SCIENTIFIC_LEDGER_ADMISSIONS[ledger_id] = (reference, admission_sha256)
    return ledger


def _require_scientific_ledger_admission(ledger: object) -> TrajectoryCreditLedger:
    if type(ledger) is not TrajectoryCreditLedger:
        raise ValueError("scientific ledger admission requires the exact ledger type")
    admitted = _SCIENTIFIC_LEDGER_ADMISSIONS.get(id(ledger))
    if (
        ledger._factory_marker is not _SCIENTIFIC_LEDGER_FACTORY_MARKER
        or admitted is None
        or admitted[0]() is not ledger
        or admitted[1] != ledger.admission_sha256
    ):
        raise ValueError("scientific ledger admission is absent or forged")
    ledger._validate()
    if (
        ledger.logical_k != FIXED_SCIENTIFIC_K
        or ledger.training_repetition_penalty is None
        or ledger.seed_group_id is None
        or ledger.admission_sha256 != _sha256(ledger._admission_preimage())
        or any(
            any(
                value is None
                for value in (
                    image.plan_sha256,
                    image.native_receipts_sha256,
                    image.parity_receipt_sha256,
                    image.parser_projection_sha256,
                )
            )
            for image in ledger.images
        )
    ):
        raise ValueError("scientific ledger admission lineage or seal differs")
    return ledger


@dataclass(frozen=True)
class CanonicalProjectionEvent:
    event_order: int
    canonical_generated_order: int | None
    kind: str
    category: str
    bbox: tuple[float, float, float, float] | None
    token_start: int
    token_end: int
    drop_reason: str | None

    def __post_init__(self) -> None:
        _integer(self.event_order, field="event_order")
        if self.canonical_generated_order is not None:
            _integer(
                self.canonical_generated_order,
                field="canonical_generated_order",
            )
        if self.kind not in {"prediction", "invalid", "malformed"}:
            raise ValueError("canonical projection event kind differs")
        if not isinstance(self.category, str):
            raise ValueError("canonical projection category must be a string")
        if self.bbox is not None:
            bbox = tuple(_finite(value, field="projection bbox") for value in self.bbox)
            if len(bbox) != 4:
                raise ValueError("canonical projection bbox must contain four values")
            object.__setattr__(self, "bbox", bbox)
        start = _integer(self.token_start, field="projection token_start")
        end = _integer(self.token_end, field="projection token_end", minimum=1)
        if end <= start:
            raise ValueError("canonical projection token span must be nonempty")
        if self.drop_reason is not None and not isinstance(self.drop_reason, str):
            raise ValueError("canonical projection drop reason differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_order": self.event_order,
            "canonical_generated_order": self.canonical_generated_order,
            "kind": self.kind,
            "category": self.category,
            "bbox": None if self.bbox is None else list(self.bbox),
            "token_start": self.token_start,
            "token_end": self.token_end,
            "drop_reason": self.drop_reason,
        }


@dataclass(frozen=True)
class CanonicalTrajectoryProjection:
    request_id: str
    acquisition_trajectory_sha256: str
    generated_token_ids_sha256: str
    decoded_text_sha256: str
    parse_status: str
    valid_prediction_count: int
    dropped_prediction_count: int
    canonical_predictions_json: str
    canonical_drops_json: str
    events: tuple[CanonicalProjectionEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("canonical trajectory projection requires request_id")
        for field in (
            "acquisition_trajectory_sha256",
            "generated_token_ids_sha256",
            "decoded_text_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        if not isinstance(self.parse_status, str) or not self.parse_status:
            raise ValueError("canonical trajectory projection parse status differs")
        _integer(self.valid_prediction_count, field="valid_prediction_count")
        _integer(self.dropped_prediction_count, field="dropped_prediction_count")
        predictions = json.loads(self.canonical_predictions_json)
        drops = json.loads(self.canonical_drops_json)
        if (
            not isinstance(predictions, list)
            or not isinstance(drops, list)
            or len(predictions) != self.valid_prediction_count
            or len(drops) != self.dropped_prediction_count
        ):
            raise ValueError("canonical parser prediction/drop counts differ")
        events = tuple(self.events)
        if [event.event_order for event in events] != list(range(len(events))):
            raise ValueError("canonical parser events must be chronologically ordered")
        object.__setattr__(self, "events", events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "acquisition_trajectory_sha256": self.acquisition_trajectory_sha256,
            "generated_token_ids_sha256": self.generated_token_ids_sha256,
            "decoded_text_sha256": self.decoded_text_sha256,
            "parse_status": self.parse_status,
            "valid_prediction_count": self.valid_prediction_count,
            "dropped_prediction_count": self.dropped_prediction_count,
            "canonical_predictions": json.loads(self.canonical_predictions_json),
            "canonical_drops": json.loads(self.canonical_drops_json),
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True)
class CanonicalParserProjectionReceipt:
    image_id: int
    source_sha256: str
    manifest_sha256: str
    model_id: str
    processor_id: str
    training_repetition_penalty: float
    seed_group_id: str
    request_ids: tuple[str, ...]
    plan_sha256: str
    native_receipts_sha256: str
    sampled_group_sha256: str
    replayed_group_sha256: str
    parity_receipt_sha256: str
    tolerance_sha256: str
    tokenizer_id: str
    tokenizer_sha256: str
    tokenizer_implementation_id: str
    tokenizer_class: str
    tokenizer_module: str
    tokenizer_base_model_path: str
    tokenizer_config_sha256: str
    tokenizer_runtime_receipt_sha256: str
    tokenizer_backend_sha256: str
    tokenizer_backend_package_version: str
    tokenizer_decode_backend_id: str
    tokenizer_decode_policy: str
    parser_id: str
    parser_policy: str
    trajectories: tuple[CanonicalTrajectoryProjection, ...]

    def __post_init__(self) -> None:
        _integer(self.image_id, field="projection image_id")
        for field in (
            "plan_sha256",
            "source_sha256",
            "manifest_sha256",
            "native_receipts_sha256",
            "sampled_group_sha256",
            "replayed_group_sha256",
            "parity_receipt_sha256",
            "tolerance_sha256",
            "tokenizer_sha256",
            "tokenizer_config_sha256",
            "tokenizer_runtime_receipt_sha256",
            "tokenizer_backend_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        for field in (
            "model_id",
            "processor_id",
            "seed_group_id",
            "tokenizer_id",
            "tokenizer_implementation_id",
            "tokenizer_class",
            "tokenizer_module",
            "tokenizer_base_model_path",
            "tokenizer_backend_package_version",
            "tokenizer_decode_backend_id",
            "tokenizer_decode_policy",
            "parser_id",
            "parser_policy",
        ):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"projection {field} must be nonempty")
        repetition_penalty = _finite(
            self.training_repetition_penalty,
            field="training_repetition_penalty",
        )
        if repetition_penalty not in {1.0, 1.10}:
            raise ValueError("projection training RP differs")
        request_ids = tuple(self.request_ids)
        if (
            len(request_ids) != FIXED_SCIENTIFIC_K
            or len(set(request_ids)) != FIXED_SCIENTIFIC_K
        ):
            raise ValueError("projection request ordering must be exact K16")
        object.__setattr__(self, "request_ids", request_ids)
        if self.parser_id != PARSER_ID or self.parser_policy != PARSER_POLICY:
            raise ValueError("projection parser identity or policy differs")
        trajectories = tuple(self.trajectories)
        if len(trajectories) != FIXED_SCIENTIFIC_K:
            raise ValueError("canonical parser projection requires exact K16")
        if len({item.request_id for item in trajectories}) != FIXED_SCIENTIFIC_K:
            raise ValueError("canonical parser projection requests must be unique")
        if tuple(item.request_id for item in trajectories) != request_ids:
            raise ValueError("canonical parser projection request order differs")
        object.__setattr__(self, "trajectories", trajectories)

    def _preimage(self) -> dict[str, Any]:
        return {
            "schema_version": PARSER_PROJECTION_SCHEMA_VERSION,
            "image_id": self.image_id,
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "model_id": self.model_id,
            "processor_id": self.processor_id,
            "training_repetition_penalty": self.training_repetition_penalty,
            "seed_group_id": self.seed_group_id,
            "request_ids": list(self.request_ids),
            "plan_sha256": self.plan_sha256,
            "native_receipts_sha256": self.native_receipts_sha256,
            "sampled_group_sha256": self.sampled_group_sha256,
            "replayed_group_sha256": self.replayed_group_sha256,
            "parity_receipt_sha256": self.parity_receipt_sha256,
            "tolerance_sha256": self.tolerance_sha256,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokenizer_implementation_id": self.tokenizer_implementation_id,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_module": self.tokenizer_module,
            "tokenizer_base_model_path": self.tokenizer_base_model_path,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "tokenizer_runtime_receipt_sha256": self.tokenizer_runtime_receipt_sha256,
            "tokenizer_backend_sha256": self.tokenizer_backend_sha256,
            "tokenizer_backend_package_version": (
                self.tokenizer_backend_package_version
            ),
            "tokenizer_decode_backend_id": self.tokenizer_decode_backend_id,
            "tokenizer_decode_policy": self.tokenizer_decode_policy,
            "parser_id": self.parser_id,
            "parser_policy": self.parser_policy,
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
        }

    def to_dict(self) -> dict[str, Any]:
        preimage = self._preimage()
        return {**preimage, "content_sha256": _sha256(preimage)}

    @property
    def content_sha256(self) -> str:
        return _sha256(self._preimage())

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        manifest: Human13KUnionManifest,
        publication: AdmittedPublication,
        tokenizer_adapter: CanonicalTokenizerDecodeAdapter,
    ) -> CanonicalParserProjectionReceipt:
        expected = build_canonical_parser_projection_receipt(
            manifest,
            publication,
            tokenizer_adapter=tokenizer_adapter,
        )
        if not isinstance(value, Mapping) or dict(value) != expected.to_dict():
            raise ValueError("stored canonical parser projection differs from rerun")
        return expected


@dataclass(frozen=True)
class _ParsedCreditRow:
    """One parser-emitted row and its generated-token half-open span."""

    generated_order: int
    category: str
    bbox: tuple[float, float, float, float] | None
    token_start: int
    token_end: int
    geometry_valid: bool = True

    def __post_init__(self) -> None:
        _integer(self.generated_order, field="generated_order")
        if not isinstance(self.category, str):
            raise ValueError("category must be a string")
        if self.bbox is not None:
            try:
                bbox = tuple(float(item) for item in self.bbox)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "bbox must contain four numeric coordinates"
                ) from error
            if len(bbox) != 4 or not all(math.isfinite(item) for item in bbox):
                raise ValueError("bbox must contain four finite coordinates")
            object.__setattr__(self, "bbox", bbox)
        start = _integer(self.token_start, field="token_start")
        end = _integer(self.token_end, field="token_end", minimum=1)
        if end <= start:
            raise ValueError("parsed row token span must be nonempty")
        if not isinstance(self.geometry_valid, bool):
            raise ValueError("geometry_valid must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_order": self.generated_order,
            "category": self.category,
            "bbox": None if self.bbox is None else list(self.bbox),
            "token_start": self.token_start,
            "token_end": self.token_end,
            "geometry_valid": self.geometry_valid,
        }

    @classmethod
    def from_dict(cls, value: object) -> _ParsedCreditRow:
        item = _strict_mapping(
            value,
            field="parsed credit row",
            keys={
                "generated_order",
                "category",
                "bbox",
                "token_start",
                "token_end",
                "geometry_valid",
            },
        )
        bbox = item["bbox"]
        return cls(
            generated_order=item["generated_order"],
            category=item["category"],
            bbox=None if bbox is None else tuple(bbox),
            token_start=item["token_start"],
            token_end=item["token_end"],
            geometry_valid=item["geometry_valid"],
        )


@dataclass(frozen=True)
class _MalformedRowSpan:
    """One chronological row-equivalent span rejected by the canonical parser."""

    generated_order: int
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        _integer(self.generated_order, field="generated_order")
        start = _integer(self.token_start, field="token_start")
        end = _integer(self.token_end, field="token_end", minimum=1)
        if end <= start:
            raise ValueError("malformed row-equivalent token span must be nonempty")

    def to_dict(self) -> dict[str, int]:
        return {
            "generated_order": self.generated_order,
            "token_start": self.token_start,
            "token_end": self.token_end,
        }

    @classmethod
    def from_dict(cls, value: object) -> _MalformedRowSpan:
        item = _strict_mapping(
            value,
            field="malformed row span",
            keys={"generated_order", "token_start", "token_end"},
        )
        return cls(item["generated_order"], item["token_start"], item["token_end"])


@dataclass(frozen=True)
class _ParsedTrajectoryCreditInput:
    request_id: str
    rows: tuple[_ParsedCreditRow, ...]
    malformed_spans: tuple[_MalformedRowSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("parsed trajectory requires request_id")
        rows = tuple(self.rows)
        malformed = tuple(self.malformed_spans)
        if any(not isinstance(row, _ParsedCreditRow) for row in rows):
            raise ValueError("rows must contain ParsedCreditRow")
        if any(not isinstance(span, _MalformedRowSpan) for span in malformed):
            raise ValueError("malformed_spans must contain MalformedRowSpan")
        orders = [row.generated_order for row in rows] + [
            span.generated_order for span in malformed
        ]
        if len(set(orders)) != len(orders):
            raise ValueError("parsed trajectory generated orders must be unique")
        spans = sorted(
            [(row.token_start, row.token_end) for row in rows]
            + [(span.token_start, span.token_end) for span in malformed]
        )
        if any(
            right_start < left_end
            for (_, left_end), (right_start, _) in zip(spans, spans[1:])
        ):
            raise ValueError("parsed trajectory token spans must not overlap")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "malformed_spans", malformed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "rows": [row.to_dict() for row in self.rows],
            "malformed_spans": [span.to_dict() for span in self.malformed_spans],
        }

    @classmethod
    def from_dict(cls, value: object) -> _ParsedTrajectoryCreditInput:
        item = _strict_mapping(
            value,
            field="parsed trajectory credit input",
            keys={"request_id", "rows", "malformed_spans"},
        )
        return cls(
            request_id=item["request_id"],
            rows=tuple(_ParsedCreditRow.from_dict(row) for row in item["rows"]),
            malformed_spans=tuple(
                _MalformedRowSpan.from_dict(span) for span in item["malformed_spans"]
            ),
        )


@dataclass(frozen=True)
class _AcquisitionGroupCreditInput:
    image_id: int
    acquisition_group: AcquisitionGroup
    parsed_trajectories: tuple[_ParsedTrajectoryCreditInput, ...]

    def __post_init__(self) -> None:
        _integer(self.image_id, field="image_id")
        if not isinstance(self.acquisition_group, AcquisitionGroup):
            raise ValueError("acquisition_group must be sealed Task-2 evidence")
        parsed = tuple(self.parsed_trajectories)
        if any(not isinstance(item, _ParsedTrajectoryCreditInput) for item in parsed):
            raise ValueError("parsed_trajectories contain an invalid record")
        expected_ids = tuple(
            item.identity.request_id for item in self.acquisition_group.trajectories
        )
        if tuple(item.request_id for item in parsed) != expected_ids:
            raise ValueError(
                "parsed trajectory order differs from acquisition requests"
            )
        for evidence, projection in zip(
            self.acquisition_group.trajectories, parsed, strict=True
        ):
            content_end = len(evidence.generated_tokens) - (
                evidence.terminal_kind == "natural_stop"
            )
            for start, end in [
                (row.token_start, row.token_end) for row in projection.rows
            ] + [
                (span.token_start, span.token_end)
                for span in projection.malformed_spans
            ]:
                if not (0 <= start < end <= content_end):
                    raise ValueError(
                        "parsed row span includes STOP or exceeds generated evidence"
                    )
        object.__setattr__(self, "parsed_trajectories", parsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "acquisition_group": self.acquisition_group.to_dict(),
            "acquisition_group_sha256": self.acquisition_group.content_sha256,
            "parsed_trajectories": [
                item.to_dict() for item in self.parsed_trajectories
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> _AcquisitionGroupCreditInput:
        item = _strict_mapping(
            value,
            field="acquisition group credit input",
            keys={
                "image_id",
                "acquisition_group",
                "acquisition_group_sha256",
                "parsed_trajectories",
            },
        )
        group = AcquisitionGroup.from_dict(item["acquisition_group"])
        if item["acquisition_group_sha256"] != group.content_sha256:
            raise ValueError("acquisition group content SHA-256 differs")
        return cls(
            image_id=item["image_id"],
            acquisition_group=group,
            parsed_trajectories=tuple(
                _ParsedTrajectoryCreditInput.from_dict(parsed)
                for parsed in item["parsed_trajectories"]
            ),
        )


@dataclass(frozen=True)
class _TrajectoryCreditAcquisition:
    groups: tuple[_AcquisitionGroupCreditInput, ...]
    logical_k: int = FIXED_SCIENTIFIC_K

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not groups or any(
            not isinstance(group, _AcquisitionGroupCreditInput) for group in groups
        ):
            raise ValueError("trajectory-credit acquisition requires sealed groups")
        logical_k = _integer(self.logical_k, field="logical_k", minimum=2)
        if len({group.image_id for group in groups}) != len(groups):
            raise ValueError("trajectory-credit acquisition image ids must be unique")
        if any(len(group.parsed_trajectories) != logical_k for group in groups):
            raise ValueError(
                "every group trajectory count must equal explicit logical K"
            )
        source_hashes = {
            group.acquisition_group.identity.source_sha256 for group in groups
        }
        manifest_hashes = {
            group.acquisition_group.identity.manifest_sha256 for group in groups
        }
        if len(source_hashes) != 1 or len(manifest_hashes) != 1:
            raise ValueError(
                "acquisition groups must share Source and manifest lineage"
            )
        object.__setattr__(self, "groups", groups)

    def _preimage(self) -> dict[str, Any]:
        return {
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "logical_k": self.logical_k,
            "groups": [group.to_dict() for group in self.groups],
        }

    def to_dict(self) -> dict[str, Any]:
        preimage = self._preimage()
        return {**preimage, "content_sha256": _sha256(preimage)}

    @classmethod
    def from_dict(cls, value: object) -> _TrajectoryCreditAcquisition:
        item = _strict_mapping(
            value,
            field="trajectory credit acquisition",
            keys={"schema_version", "logical_k", "groups", "content_sha256"},
        )
        preimage = {key: item[key] for key in ("schema_version", "logical_k", "groups")}
        if item["schema_version"] != ACQUISITION_SCHEMA_VERSION or item[
            "content_sha256"
        ] != _sha256(preimage):
            raise ValueError("trajectory credit acquisition content SHA-256 differs")
        return cls(
            groups=tuple(
                _AcquisitionGroupCreditInput.from_dict(group)
                for group in item["groups"]
            ),
            logical_k=item["logical_k"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self._preimage())


@dataclass(frozen=True)
class TokenCredit:
    request_id: str
    token_index: int
    row_position: int
    outcome: str
    advantage: float
    scored: bool

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("token credit requires request_id")
        _integer(self.token_index, field="token_index")
        _integer(self.row_position, field="row_position")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("token credit requires outcome")
        object.__setattr__(
            self, "advantage", _finite(self.advantage, field="token advantage")
        )
        if not isinstance(self.scored, bool):
            raise ValueError("token scored flag must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token_index": self.token_index,
            "row_position": self.row_position,
            "outcome": self.outcome,
            "advantage": self.advantage,
            "scored": self.scored,
        }

    @classmethod
    def from_dict(cls, value: object) -> TokenCredit:
        item = _strict_mapping(
            value,
            field="token credit",
            keys={
                "request_id",
                "token_index",
                "row_position",
                "outcome",
                "advantage",
                "scored",
            },
        )
        return cls(**item)


@dataclass(frozen=True)
class RowCredit:
    request_id: str
    generated_order: int | None
    position_index: int | None
    outcome: str
    matched_owner_id: str | None
    owner_stratum: str | None
    token_indices: tuple[int, ...]
    immediate_credit: float
    return_to_go: float
    unclamped_advantage: float
    advantage: float
    scored: bool
    tokens: tuple[TokenCredit, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("row credit requires request_id")
        if self.generated_order is not None:
            _integer(self.generated_order, field="generated_order")
        if self.position_index is not None:
            _integer(self.position_index, field="position_index")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("row credit requires outcome")
        token_indices = tuple(self.token_indices)
        if any(
            _integer(index, field="token_index") != index for index in token_indices
        ):
            raise ValueError("row token indices differ")
        if len(set(token_indices)) != len(token_indices):
            raise ValueError("row token indices must be unique")
        object.__setattr__(self, "token_indices", token_indices)
        for field in (
            "immediate_credit",
            "return_to_go",
            "unclamped_advantage",
            "advantage",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field=field))
        if not isinstance(self.scored, bool):
            raise ValueError("row scored flag must be boolean")
        tokens = tuple(self.tokens)
        if tokens:
            if tuple(token.token_index for token in tokens) != token_indices:
                raise ValueError("row token credits differ from token span")
            if any(
                token.request_id != self.request_id
                or token.row_position != self.position_index
                or token.outcome != self.outcome
                or token.advantage != self.advantage
                or token.scored != self.scored
                for token in tokens
            ):
                raise ValueError("row token credit metadata differs")
        object.__setattr__(self, "tokens", tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "generated_order": self.generated_order,
            "position_index": self.position_index,
            "outcome": self.outcome,
            "matched_owner_id": self.matched_owner_id,
            "owner_stratum": self.owner_stratum,
            "token_indices": list(self.token_indices),
            "immediate_credit": self.immediate_credit,
            "return_to_go": self.return_to_go,
            "unclamped_advantage": self.unclamped_advantage,
            "advantage": self.advantage,
            "scored": self.scored,
            "tokens": [token.to_dict() for token in self.tokens],
        }

    @classmethod
    def from_dict(cls, value: object) -> RowCredit:
        item = _strict_mapping(
            value,
            field="row credit",
            keys={
                "request_id",
                "generated_order",
                "position_index",
                "outcome",
                "matched_owner_id",
                "owner_stratum",
                "token_indices",
                "immediate_credit",
                "return_to_go",
                "unclamped_advantage",
                "advantage",
                "scored",
                "tokens",
            },
        )
        return cls(
            request_id=item["request_id"],
            generated_order=item["generated_order"],
            position_index=item["position_index"],
            outcome=item["outcome"],
            matched_owner_id=item["matched_owner_id"],
            owner_stratum=item["owner_stratum"],
            token_indices=tuple(item["token_indices"]),
            immediate_credit=item["immediate_credit"],
            return_to_go=item["return_to_go"],
            unclamped_advantage=item["unclamped_advantage"],
            advantage=item["advantage"],
            scored=item["scored"],
            tokens=tuple(TokenCredit.from_dict(token) for token in item["tokens"]),
        )


@dataclass(frozen=True)
class TrajectoryLedger:
    request_id: str
    acquisition_trajectory_sha256: str
    policy_contract_sha256: str
    token_count: int
    terminal_kind: str
    rows: tuple[RowCredit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("trajectory ledger requires request_id")
        object.__setattr__(
            self,
            "acquisition_trajectory_sha256",
            _digest(
                self.acquisition_trajectory_sha256,
                field="acquisition_trajectory_sha256",
            ),
        )
        object.__setattr__(
            self,
            "policy_contract_sha256",
            _digest(self.policy_contract_sha256, field="policy_contract_sha256"),
        )
        _integer(self.token_count, field="token_count", minimum=1)
        if self.terminal_kind not in {"natural_stop", "cap_stop"}:
            raise ValueError("trajectory ledger terminal kind differs")
        rows = tuple(self.rows)
        if any(
            not isinstance(row, RowCredit) or row.request_id != self.request_id
            for row in rows
        ):
            raise ValueError("trajectory ledger rows differ from request")
        positions = [
            row.position_index for row in rows if row.position_index is not None
        ]
        if positions != list(range(len(positions))):
            raise ValueError("trajectory row positions must be contiguous")
        allowed_outcomes = {
            "trusted_first_hit",
            "duplicate",
            "invalid",
            "trusted_owner_repeat",
            "unmatched",
            "legacy_m",
            "malformed",
            "natural_stop",
            "cap_shortfall",
        }
        if any(row.outcome not in allowed_outcomes for row in rows):
            raise ValueError("trajectory ledger contains an unknown row outcome")
        token_indices = [index for row in rows for index in row.token_indices]
        if len(set(token_indices)) != len(token_indices) or any(
            index >= self.token_count for index in token_indices
        ):
            raise ValueError(
                "trajectory ledger token assignments overlap or exceed evidence"
            )
        running = 0.0
        expected_returns: list[float] = []
        for row in reversed(rows):
            running += row.immediate_credit
            expected_returns.append(running)
        if any(
            row.return_to_go != expected
            for row, expected in zip(rows, reversed(expected_returns), strict=True)
        ):
            raise ValueError("trajectory ledger return-to-go does not telescope")
        terminal = rows[-1] if rows else None
        if self.terminal_kind == "natural_stop":
            if (
                terminal is None
                or terminal.outcome != "natural_stop"
                or terminal.token_indices != (self.token_count - 1,)
                or not terminal.scored
            ):
                raise ValueError("natural STOP ledger term differs from evidence")
        elif (
            terminal is None
            or terminal.outcome != "cap_shortfall"
            or terminal.position_index is not None
            or terminal.token_indices
            or terminal.scored
        ):
            raise ValueError("cap shortfall must not create a scored terminal action")
        if any(
            row.scored != (row.outcome not in {"legacy_m", "cap_shortfall"})
            for row in rows
        ):
            raise ValueError("trajectory ledger score mask differs from row outcome")
        object.__setattr__(self, "rows", rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "acquisition_trajectory_sha256": self.acquisition_trajectory_sha256,
            "policy_contract_sha256": self.policy_contract_sha256,
            "token_count": self.token_count,
            "terminal_kind": self.terminal_kind,
            "rows": [row.to_dict() for row in self.rows],
        }

    @classmethod
    def from_dict(cls, value: object) -> TrajectoryLedger:
        item = _strict_mapping(
            value,
            field="trajectory ledger",
            keys={
                "request_id",
                "acquisition_trajectory_sha256",
                "policy_contract_sha256",
                "token_count",
                "terminal_kind",
                "rows",
            },
        )
        return cls(
            request_id=item["request_id"],
            acquisition_trajectory_sha256=item["acquisition_trajectory_sha256"],
            policy_contract_sha256=item["policy_contract_sha256"],
            token_count=item["token_count"],
            terminal_kind=item["terminal_kind"],
            rows=tuple(RowCredit.from_dict(row) for row in item["rows"]),
        )


@dataclass(frozen=True)
class ImageCreditLedger:
    image_id: int
    acquisition_group_sha256: str
    trusted_owner_ids: tuple[str, ...]
    legacy_m_owner_ids: tuple[str, ...]
    owner_weight: float
    trajectories: tuple[TrajectoryLedger, ...]
    position_returns: tuple[tuple[float, ...], ...]
    plan_sha256: str | None = None
    native_receipts_sha256: str | None = None
    parity_receipt_sha256: str | None = None
    parser_projection_sha256: str | None = None

    def __post_init__(self) -> None:
        _integer(self.image_id, field="image_id")
        object.__setattr__(
            self,
            "acquisition_group_sha256",
            _digest(self.acquisition_group_sha256, field="acquisition_group_sha256"),
        )
        trusted = tuple(self.trusted_owner_ids)
        legacy = tuple(self.legacy_m_owner_ids)
        if (
            not trusted
            or len(set(trusted)) != len(trusted)
            or set(trusted) & set(legacy)
        ):
            raise ValueError("image ledger owner partition differs")
        object.__setattr__(self, "trusted_owner_ids", trusted)
        object.__setattr__(self, "legacy_m_owner_ids", legacy)
        weight = _finite(self.owner_weight, field="owner_weight")
        if weight != 1.0 / len(trusted):
            raise ValueError("owner weight must be fixed uniform trusted mass")
        trajectories = tuple(self.trajectories)
        if len({trajectory.request_id for trajectory in trajectories}) != len(
            trajectories
        ):
            raise ValueError("image ledger request ids must be unique")
        object.__setattr__(self, "trajectories", trajectories)
        position_returns = tuple(
            tuple(_finite(value, field="position return") for value in row)
            for row in self.position_returns
        )
        if any(len(row) != len(trajectories) for row in position_returns):
            raise ValueError("position return matrix differs from logical K")
        max_positions = max(
            (
                sum(row.position_index is not None for row in trajectory.rows)
                for trajectory in trajectories
            ),
            default=0,
        )
        if len(position_returns) != max_positions:
            raise ValueError("position return matrix does not cover every row position")
        rows_by_position = [
            {
                row.position_index: row
                for row in trajectory.rows
                if row.position_index is not None
            }
            for trajectory in trajectories
        ]
        for position, returns in enumerate(position_returns):
            for trajectory_index, rows in enumerate(rows_by_position):
                row = rows.get(position)
                if row is None:
                    if returns[trajectory_index] != 0.0:
                        raise ValueError("terminated-path padding return must be zero")
                    continue
                if returns[trajectory_index] != row.return_to_go:
                    raise ValueError(
                        "position return differs from trajectory return-to-go"
                    )
                baseline = (sum(returns) - returns[trajectory_index]) / (
                    len(trajectories) - 1
                )
                raw_advantage = row.return_to_go - baseline
                advantage = (
                    min(raw_advantage, 0.0)
                    if row.outcome == "natural_stop"
                    else raw_advantage
                )
                if (
                    row.unclamped_advantage != raw_advantage
                    or row.advantage != advantage
                ):
                    raise ValueError("row RLOO advantage differs from detached returns")
        object.__setattr__(self, "position_returns", position_returns)
        lineage = (
            self.plan_sha256,
            self.native_receipts_sha256,
            self.parity_receipt_sha256,
            self.parser_projection_sha256,
        )
        if any(value is not None for value in lineage):
            if any(value is None for value in lineage):
                raise ValueError("image ledger Task2/parser lineage must be complete")
            for field in (
                "plan_sha256",
                "native_receipts_sha256",
                "parity_receipt_sha256",
                "parser_projection_sha256",
            ):
                object.__setattr__(
                    self,
                    field,
                    _digest(getattr(self, field), field=field),
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "acquisition_group_sha256": self.acquisition_group_sha256,
            "trusted_owner_ids": list(self.trusted_owner_ids),
            "legacy_m_owner_ids": list(self.legacy_m_owner_ids),
            "owner_weight": self.owner_weight,
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
            "position_returns": [list(row) for row in self.position_returns],
            "plan_sha256": self.plan_sha256,
            "native_receipts_sha256": self.native_receipts_sha256,
            "parity_receipt_sha256": self.parity_receipt_sha256,
            "parser_projection_sha256": self.parser_projection_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ImageCreditLedger:
        item = _strict_mapping(
            value,
            field="image credit ledger",
            keys={
                "image_id",
                "acquisition_group_sha256",
                "trusted_owner_ids",
                "legacy_m_owner_ids",
                "owner_weight",
                "trajectories",
                "position_returns",
                "plan_sha256",
                "native_receipts_sha256",
                "parity_receipt_sha256",
                "parser_projection_sha256",
            },
        )
        return cls(
            image_id=item["image_id"],
            acquisition_group_sha256=item["acquisition_group_sha256"],
            trusted_owner_ids=tuple(item["trusted_owner_ids"]),
            legacy_m_owner_ids=tuple(item["legacy_m_owner_ids"]),
            owner_weight=item["owner_weight"],
            trajectories=tuple(
                TrajectoryLedger.from_dict(trajectory)
                for trajectory in item["trajectories"]
            ),
            position_returns=tuple(tuple(row) for row in item["position_returns"]),
            plan_sha256=item["plan_sha256"],
            native_receipts_sha256=item["native_receipts_sha256"],
            parity_receipt_sha256=item["parity_receipt_sha256"],
            parser_projection_sha256=item["parser_projection_sha256"],
        )


@dataclass(frozen=True, init=False)
class TrajectoryCreditLedger:
    source_sha256: str
    manifest_sha256: str
    acquisition_sha256: str
    logical_image_count: int
    logical_k: int
    images: tuple[ImageCreditLedger, ...]
    training_repetition_penalty: float | None
    seed_group_id: str | None
    admission_sha256: str | None
    _factory_marker: object | None = dataclass_field(repr=False, compare=False)

    def _validate(self) -> None:
        for field in ("source_sha256", "manifest_sha256", "acquisition_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        _integer(self.logical_image_count, field="logical_image_count", minimum=1)
        _integer(self.logical_k, field="logical_k", minimum=2)
        images = tuple(self.images)
        if len(images) != self.logical_image_count or len(
            {image.image_id for image in images}
        ) != len(images):
            raise ValueError("ledger images differ from logical image count")
        if any(len(image.trajectories) != self.logical_k for image in images):
            raise ValueError("ledger trajectories differ from logical K")
        request_ids = [
            trajectory.request_id
            for image in images
            for trajectory in image.trajectories
        ]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("ledger request ids must be globally unique")
        object.__setattr__(self, "images", images)
        scientific_lineage = (
            self.training_repetition_penalty,
            self.seed_group_id,
            self.admission_sha256,
        )
        if any(value is not None for value in scientific_lineage):
            if any(value is None for value in scientific_lineage):
                raise ValueError("scientific ledger lineage must be complete")
            repetition_penalty = _finite(
                self.training_repetition_penalty,
                field="training_repetition_penalty",
            )
            if repetition_penalty <= 0.0:
                raise ValueError("training_repetition_penalty must be positive")
            if not isinstance(self.seed_group_id, str) or not self.seed_group_id:
                raise ValueError("seed_group_id must be a non-empty string")
            object.__setattr__(
                self,
                "admission_sha256",
                _digest(self.admission_sha256, field="admission_sha256"),
            )

    @property
    def logical_denominator(self) -> int:
        return self.logical_image_count * self.logical_k

    @property
    def scored_tokens(self) -> tuple[TokenCredit, ...]:
        return tuple(
            token
            for image in self.images
            for trajectory in image.trajectories
            for row in trajectory.rows
            for token in row.tokens
            if token.scored
        )

    def _preimage(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "acquisition_sha256": self.acquisition_sha256,
            "logical_image_count": self.logical_image_count,
            "logical_k": self.logical_k,
            "training_repetition_penalty": self.training_repetition_penalty,
            "seed_group_id": self.seed_group_id,
            "admission_sha256": self.admission_sha256,
            "images": [image.to_dict() for image in self.images],
        }

    def _admission_preimage(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "acquisition_sha256": self.acquisition_sha256,
            "logical_image_count": self.logical_image_count,
            "logical_k": self.logical_k,
            "training_repetition_penalty": self.training_repetition_penalty,
            "seed_group_id": self.seed_group_id,
            "parser_id": PARSER_ID,
            "parser_policy": PARSER_POLICY,
            "images": [image.to_dict() for image in self.images],
        }

    def to_dict(self) -> dict[str, Any]:
        preimage = self._preimage()
        return {**preimage, "content_sha256": _sha256(preimage)}

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        manifest: Human13KUnionManifest,
        acquisition: TrajectoryCreditPanelAcquisition,
        tokenizer_adapter: CanonicalTokenizerDecodeAdapter,
    ) -> TrajectoryCreditLedger:
        expected = build_trajectory_credit_ledger(
            manifest,
            acquisition,
            tokenizer_adapter=tokenizer_adapter,
        )
        if not isinstance(value, Mapping) or dict(value) != expected.to_dict():
            raise ValueError(
                "stored trajectory credit ledger differs from rerun canonical projection"
            )
        return expected

    @property
    def content_sha256(self) -> str:
        return _sha256(self._preimage())


def _canonical_panel_dimensions(
    manifest: Human13KUnionManifest,
) -> dict[int, tuple[int, int]]:
    panel_path = Path(PANEL_PATH).resolve(strict=True)
    payload = panel_path.read_bytes()
    panel_sha256 = hashlib.sha256(payload).hexdigest()
    if panel_sha256 != manifest.binding.panel.panel_sha256:
        raise ValueError("manifest panel differs from the canonical dimension source")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    dimensions = {
        _integer(row.get("image_id"), field="panel image_id"): (
            _integer(row.get("width"), field="panel width", minimum=1),
            _integer(row.get("height"), field="panel height", minimum=1),
        )
        for row in rows
    }
    expected = tuple(image.image_id for image in manifest.binding.panel.images)
    if tuple(row["image_id"] for row in rows) != expected:
        raise ValueError("canonical panel image order differs from manifest binding")
    return dimensions


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _token_span_for_chars(
    token_texts: tuple[str, ...],
    *,
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    if not (0 <= char_start < char_end):
        raise ValueError("canonical parser emitted an empty or invalid character span")
    boundaries = [0]
    for text in token_texts:
        boundaries.append(boundaries[-1] + len(text))
    try:
        token_start = boundaries.index(char_start)
        token_end = boundaries.index(char_end)
    except ValueError as error:
        raise ValueError(
            "canonical parser character span does not align to generated tokens"
        ) from error
    if token_end <= token_start:
        raise ValueError("canonical parser token span must be nonempty")
    return token_start, token_end


def _drop_category(raw_text: str) -> str:
    start = raw_text.find(OBJECT_REF_START_TOKEN)
    end = raw_text.find(OBJECT_REF_END_TOKEN, start + len(OBJECT_REF_START_TOKEN))
    if start < 0 or end < 0:
        return ""
    return raw_text[start + len(OBJECT_REF_START_TOKEN) : end].strip()


def _drop_bbox(
    drop: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    spans = drop.get("coord_token_spans")
    if not isinstance(spans, list) or len(spans) != 4:
        return None
    try:
        bins = [
            parse_coord_token(
                str(span["text"]),
                field=f"canonical_drop.bbox[{index}]",
            )
            for index, span in enumerate(spans)
        ]
        x1, y1, x2, y2 = coord_bins_to_pixel_xyxy(
            bins,
            image_width=image_width,
            image_height=image_height,
            field="canonical_drop.bbox",
        )
        return float(x1), float(y1), float(x2), float(y2)
    except Exception:
        return None


def _trajectory_projection(
    evidence: Any,
    *,
    image_width: int,
    image_height: int,
    tokenizer_adapter: CanonicalTokenizerDecodeAdapter,
) -> CanonicalTrajectoryProjection:
    generated_token_ids = tuple(evidence.identity.generated_token_ids)
    body_token_ids = (
        generated_token_ids[:-1]
        if evidence.terminal_kind == "natural_stop"
        else generated_token_ids
    )
    decoded_text, token_texts = tokenizer_adapter.decode(body_token_ids)
    parsed = parse_compact_object_box_closed(
        decoded_text,
        row_id=f"{evidence.identity.request_id}:canonical-parser",
        row_index=0,
        image_width=image_width,
        image_height=image_height,
    )
    if parsed.parser_id != PARSER_ID or parsed.parser_policy != PARSER_POLICY:
        raise ValueError("canonical parser returned a different identity or policy")
    chronological: list[tuple[int, int, dict[str, Any]]] = []
    for prediction in parsed.predictions:
        char_start = _integer(
            prediction.get("char_start"), field="prediction.char_start"
        )
        char_end = _integer(
            prediction.get("char_end"), field="prediction.char_end", minimum=1
        )
        token_start, token_end = _token_span_for_chars(
            token_texts,
            char_start=char_start,
            char_end=char_end,
        )
        chronological.append(
            (
                char_start,
                0,
                {
                    "canonical_generated_order": prediction.get("generated_order"),
                    "kind": "prediction",
                    "category": str(prediction.get("description", "")),
                    "bbox": tuple(float(value) for value in prediction["bbox"]),
                    "token_start": token_start,
                    "token_end": token_end,
                    "drop_reason": None,
                },
            )
        )
    for drop_index, drop in enumerate(parsed.dropped_predictions):
        char_start = _integer(drop.get("char_start"), field="drop.char_start")
        char_end = _integer(drop.get("char_end"), field="drop.char_end", minimum=1)
        if char_end <= char_start:
            continue
        token_start, token_end = _token_span_for_chars(
            token_texts,
            char_start=char_start,
            char_end=char_end,
        )
        reason = str(drop.get("reason", ""))
        invalid = reason in {"empty_description", "geometry_invalid"}
        raw_text = str(drop.get("raw_text", ""))
        chronological.append(
            (
                char_start,
                1 + drop_index,
                {
                    "canonical_generated_order": drop.get("generated_order"),
                    "kind": "invalid" if invalid else "malformed",
                    "category": _drop_category(raw_text) if invalid else "",
                    "bbox": _drop_bbox(
                        drop,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    if invalid
                    else None,
                    "token_start": token_start,
                    "token_end": token_end,
                    "drop_reason": reason,
                },
            )
        )
    chronological.sort(key=lambda item: (item[0], item[1]))
    events = tuple(
        CanonicalProjectionEvent(event_order=event_order, **payload)
        for event_order, (_, _, payload) in enumerate(chronological)
    )
    return CanonicalTrajectoryProjection(
        request_id=evidence.identity.request_id,
        acquisition_trajectory_sha256=evidence.content_sha256,
        generated_token_ids_sha256=_sha256(
            {"generated_token_ids": list(generated_token_ids)}
        ),
        decoded_text_sha256=hashlib.sha256(decoded_text.encode("utf-8")).hexdigest(),
        parse_status=parsed.parse_status,
        valid_prediction_count=parsed.valid_prediction_count,
        dropped_prediction_count=parsed.dropped_prediction_count,
        canonical_predictions_json=_canonical_json(parsed.predictions),
        canonical_drops_json=_canonical_json(parsed.dropped_predictions),
        events=events,
    )


def build_canonical_parser_projection_receipt(
    manifest: Human13KUnionManifest,
    publication: AdmittedPublication,
    *,
    tokenizer_adapter: CanonicalTokenizerDecodeAdapter,
) -> CanonicalParserProjectionReceipt:
    """Rerun canonical decode, parse, and token alignment over admitted Task 2."""

    if not isinstance(manifest, Human13KUnionManifest):
        raise ValueError("manifest must be a Human13KUnionManifest")
    if manifest.binding.surface.parser != PARSER_POLICY:
        raise ValueError("manifest parser policy differs from canonical parser policy")
    if type(tokenizer_adapter) is not CanonicalTokenizerDecodeAdapter:
        raise ValueError("a factory-verified canonical tokenizer adapter is required")
    admitted = _revalidate_admitted_publication(publication)
    tokenizer_adapter.verify_for(manifest=manifest, publication=admitted)
    plan = admitted.execution.plan
    image_by_id = {image.image_id: image for image in manifest.images}
    if plan.image_id not in image_by_id:
        raise ValueError("Task2 plan image is absent from supplied manifest")
    manifest_sha256 = _manifest_sha256(manifest)
    for group in (admitted.execution.group, admitted.replayed_group):
        if group.identity.manifest_sha256 != manifest_sha256:
            raise ValueError("Task2 publication manifest lineage differs")
        if (
            group.identity.source_sha256
            != admitted.replayed_group.identity.source_sha256
        ):
            raise ValueError("Task2 publication Source lineage differs")
        if group.identity.model_id != admitted.replayed_group.identity.model_id:
            raise ValueError("Task2 publication model lineage differs")
        if group.identity.tokenizer_id != admitted.replayed_group.identity.tokenizer_id:
            raise ValueError("Task2 publication tokenizer lineage differs")
        if group.identity.processor_id != admitted.replayed_group.identity.processor_id:
            raise ValueError("Task2 publication processor lineage differs")
    if (
        plan.repetition_penalty
        != admitted.replayed_group.policy_contract.repetition_penalty
    ):
        raise ValueError("Task2 plan training RP differs from replayed policy")
    if plan.seed_group_id != admitted.replayed_group.seed_group_id:
        raise ValueError("Task2 plan seed group differs from replayed acquisition")
    request_ids = tuple(item.request_id for item in plan.requests)
    if request_ids != admitted.execution.plan_request_ids or request_ids != tuple(
        trajectory.identity.request_id
        for trajectory in admitted.replayed_group.trajectories
    ):
        raise ValueError("Task2 request ordering differs across admitted evidence")
    if tokenizer_adapter.tokenizer_id != admitted.replayed_group.identity.tokenizer_id:
        raise ValueError("canonical tokenizer identity differs from Task2 evidence")
    if tokenizer_adapter.tokenizer_sha256 != manifest.binding.surface.tokenizer_sha256:
        raise ValueError("canonical tokenizer SHA-256 differs from manifest")
    dimensions = _canonical_panel_dimensions(manifest)
    image_width, image_height = dimensions[plan.image_id]
    trajectories = tuple(
        _trajectory_projection(
            evidence,
            image_width=image_width,
            image_height=image_height,
            tokenizer_adapter=tokenizer_adapter,
        )
        for evidence in admitted.replayed_group.trajectories
    )
    return CanonicalParserProjectionReceipt(
        image_id=plan.image_id,
        source_sha256=admitted.replayed_group.identity.source_sha256,
        manifest_sha256=manifest_sha256,
        model_id=admitted.replayed_group.identity.model_id,
        processor_id=admitted.replayed_group.identity.processor_id,
        training_repetition_penalty=plan.repetition_penalty,
        seed_group_id=plan.seed_group_id,
        request_ids=request_ids,
        plan_sha256=admitted.execution.plan_sha256,
        native_receipts_sha256=admitted.binding.native_receipts_sha256,
        sampled_group_sha256=admitted.binding.sampled_group_sha256,
        replayed_group_sha256=admitted.binding.replayed_group_sha256,
        parity_receipt_sha256=admitted.binding.parity_receipt_sha256,
        tolerance_sha256=admitted.binding.tolerance_sha256,
        tokenizer_id=tokenizer_adapter.tokenizer_id,
        tokenizer_sha256=tokenizer_adapter.tokenizer_sha256,
        tokenizer_implementation_id=tokenizer_adapter.implementation_id,
        tokenizer_class=tokenizer_adapter.tokenizer_class,
        tokenizer_module=tokenizer_adapter.tokenizer_module,
        tokenizer_base_model_path=str(tokenizer_adapter.base_model_path),
        tokenizer_config_sha256=tokenizer_adapter.tokenizer_config_sha256,
        tokenizer_runtime_receipt_sha256=(tokenizer_adapter.runtime_receipt_sha256),
        tokenizer_backend_sha256=tokenizer_adapter.backend_tokenizer_sha256,
        tokenizer_backend_package_version=(
            tokenizer_adapter.tokenizers_package_version
        ),
        tokenizer_decode_backend_id=tokenizer_adapter.decode_backend_id,
        tokenizer_decode_policy=tokenizer_adapter.decode_policy,
        parser_id=PARSER_ID,
        parser_policy=PARSER_POLICY,
        trajectories=trajectories,
    )


def _utility(coverage: float) -> float:
    return math.expm1(coverage) / math.expm1(1.0)


def _owner_repeat(
    row: _ParsedCreditRow,
    hit_owner_ids: set[str],
    owner_by_id: Mapping[str, Any],
    owner_iou_threshold: float,
) -> bool:
    bbox = _valid_box(row.bbox)
    category = normalize_coco_category_name(row.category)
    if bbox is None or not category:
        return False
    return any(
        normalize_coco_category_name(owner_by_id[owner_id].category) == category
        and iou_xyxy(owner_by_id[owner_id].bbox, bbox) >= owner_iou_threshold
        for owner_id in hit_owner_ids
    )


def _project_one_trajectory(
    image: ImageRecord,
    evidence: Any,
    parsed: _ParsedTrajectoryCreditInput,
    *,
    trusted_owner_ids: tuple[str, ...],
    legacy_owner_ids: tuple[str, ...],
    owner_weight: float,
    duplicate_iou_threshold: float,
    owner_iou_threshold: float,
) -> TrajectoryLedger:
    owner_by_id = {owner.owner_id: owner for owner in image.owners}
    rows_by_order = {row.generated_order: row for row in parsed.rows}
    malformed_by_order = {span.generated_order: span for span in parsed.malformed_spans}
    retained: list[_ParsedCreditRow] = []
    duplicate_orders: set[int] = set()
    invalid_orders: set[int] = set()
    valid_rows: list[_ParsedCreditRow] = []
    for row in sorted(
        parsed.rows, key=lambda item: (item.generated_order, item.token_start)
    ):
        bbox = _valid_box(row.bbox)
        duplicate = bbox is not None and any(
            iou_xyxy(earlier.bbox, bbox) > duplicate_iou_threshold
            for earlier in retained
            if earlier.bbox is not None
        )
        if duplicate:
            duplicate_orders.add(row.generated_order)
            continue
        category = normalize_coco_category_name(row.category)
        if bbox is None or not category or not row.geometry_valid:
            invalid_orders.add(row.generated_order)
            continue
        retained.append(row)
        valid_rows.append(row)

    projection = _match_prefix(
        image,
        [
            {
                "generated_order": row.generated_order,
                "category": row.category,
                "bbox": row.bbox,
            }
            for row in valid_rows
        ],
        duplicate_iou_threshold=duplicate_iou_threshold,
        owner_iou_threshold=owner_iou_threshold,
    )
    if projection["duplicate_rows"] or projection["invalid_rows"]:
        raise ValueError("preclassified rows differ from canonical matcher")
    matched_owner_by_order = {
        int(match["generated_order"]): owner_id
        for owner_id, match in projection["owner_matches"].items()
    }
    trusted = set(trusted_owner_ids)
    legacy = set(legacy_owner_ids)
    hit_trusted: set[str] = set()
    raw_rows: list[RowCredit] = []
    position = 0
    event_orders = sorted(set(rows_by_order) | set(malformed_by_order))
    for order in event_orders:
        if order in malformed_by_order:
            span = malformed_by_order[order]
            raw_rows.append(
                RowCredit(
                    request_id=parsed.request_id,
                    generated_order=order,
                    position_index=position,
                    outcome="malformed",
                    matched_owner_id=None,
                    owner_stratum=None,
                    token_indices=tuple(range(span.token_start, span.token_end)),
                    immediate_credit=-owner_weight,
                    return_to_go=0.0,
                    unclamped_advantage=0.0,
                    advantage=0.0,
                    scored=True,
                )
            )
            position += 1
            continue
        row = rows_by_order[order]
        token_indices = tuple(range(row.token_start, row.token_end))
        owner_id = matched_owner_by_order.get(order)
        stratum = owner_by_id[owner_id].stratum if owner_id is not None else None
        if order in duplicate_orders:
            outcome, immediate, owner_id, stratum, scored = (
                "duplicate",
                -owner_weight,
                None,
                None,
                True,
            )
        elif order in invalid_orders:
            outcome, immediate, owner_id, stratum, scored = (
                "invalid",
                -owner_weight,
                None,
                None,
                True,
            )
        elif owner_id in trusted:
            if owner_id in hit_trusted:
                outcome, immediate, owner_id, stratum, scored = (
                    "trusted_owner_repeat",
                    -owner_weight,
                    None,
                    None,
                    True,
                )
            else:
                before = len(hit_trusted) / len(trusted)
                hit_trusted.add(owner_id)
                after = len(hit_trusted) / len(trusted)
                outcome, immediate, scored = (
                    "trusted_first_hit",
                    _utility(after) - _utility(before),
                    True,
                )
        elif owner_id in legacy:
            outcome, immediate, scored = "legacy_m", 0.0, False
        elif _owner_repeat(row, hit_trusted, owner_by_id, owner_iou_threshold):
            outcome, immediate, owner_id, stratum, scored = (
                "trusted_owner_repeat",
                -owner_weight,
                None,
                None,
                True,
            )
        else:
            outcome, immediate, owner_id, stratum, scored = (
                "unmatched",
                -owner_weight,
                None,
                None,
                True,
            )
        raw_rows.append(
            RowCredit(
                request_id=parsed.request_id,
                generated_order=order,
                position_index=position,
                outcome=outcome,
                matched_owner_id=owner_id,
                owner_stratum=stratum,
                token_indices=token_indices,
                immediate_credit=immediate,
                return_to_go=0.0,
                unclamped_advantage=0.0,
                advantage=0.0,
                scored=scored,
            )
        )
        position += 1

    remaining_mass = 1.0 - len(hit_trusted) / len(trusted)
    terminal_credit = -remaining_mass if remaining_mass > 0.0 else 0.0
    if evidence.terminal_kind == "natural_stop":
        raw_rows.append(
            RowCredit(
                request_id=parsed.request_id,
                generated_order=None,
                position_index=position,
                outcome="natural_stop",
                matched_owner_id=None,
                owner_stratum=None,
                token_indices=(len(evidence.generated_tokens) - 1,),
                immediate_credit=terminal_credit,
                return_to_go=0.0,
                unclamped_advantage=0.0,
                advantage=0.0,
                scored=True,
            )
        )
    else:
        raw_rows.append(
            RowCredit(
                request_id=parsed.request_id,
                generated_order=None,
                position_index=None,
                outcome="cap_shortfall",
                matched_owner_id=None,
                owner_stratum=None,
                token_indices=(),
                immediate_credit=terminal_credit,
                return_to_go=terminal_credit,
                unclamped_advantage=0.0,
                advantage=0.0,
                scored=False,
            )
        )

    running = 0.0
    with_returns: list[RowCredit] = []
    for row in reversed(raw_rows):
        running += row.immediate_credit
        with_returns.append(replace(row, return_to_go=running))
    with_returns.reverse()
    return TrajectoryLedger(
        request_id=parsed.request_id,
        acquisition_trajectory_sha256=evidence.content_sha256,
        policy_contract_sha256=evidence.policy_contract.content_sha256,
        token_count=len(evidence.generated_tokens),
        terminal_kind=evidence.terminal_kind,
        rows=tuple(with_returns),
    )


def _attach_rloo(
    trajectories: tuple[TrajectoryLedger, ...],
) -> tuple[tuple[TrajectoryLedger, ...], tuple[tuple[float, ...], ...]]:
    k = len(trajectories)
    max_positions = max(
        sum(row.position_index is not None for row in trajectory.rows)
        for trajectory in trajectories
    )
    position_returns: list[tuple[float, ...]] = []
    returns_by_request: dict[str, dict[int, float]] = {}
    for trajectory in trajectories:
        returns_by_request[trajectory.request_id] = {
            row.position_index: row.return_to_go
            for row in trajectory.rows
            if row.position_index is not None
        }
    for position in range(max_positions):
        position_returns.append(
            tuple(
                returns_by_request[trajectory.request_id].get(position, 0.0)
                for trajectory in trajectories
            )
        )

    updated: list[TrajectoryLedger] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        rows: list[RowCredit] = []
        for row in trajectory.rows:
            if row.position_index is None:
                rows.append(row)
                continue
            returns = position_returns[row.position_index]
            baseline = (sum(returns) - returns[trajectory_index]) / (k - 1)
            raw_advantage = row.return_to_go - baseline
            advantage = (
                min(raw_advantage, 0.0)
                if row.outcome == "natural_stop"
                else raw_advantage
            )
            tokens = tuple(
                TokenCredit(
                    request_id=row.request_id,
                    token_index=token_index,
                    row_position=row.position_index,
                    outcome=row.outcome,
                    advantage=advantage,
                    scored=row.scored,
                )
                for token_index in row.token_indices
            )
            rows.append(
                replace(
                    row,
                    unclamped_advantage=raw_advantage,
                    advantage=advantage,
                    tokens=tokens,
                )
            )
        updated.append(replace(trajectory, rows=tuple(rows)))
    return tuple(updated), tuple(position_returns)


def _project_trajectory_credit_ledger(
    manifest: Human13KUnionManifest,
    acquisition: _TrajectoryCreditAcquisition,
) -> TrajectoryCreditLedger:
    """Internal math projection over canonical parser output."""

    if not isinstance(manifest, Human13KUnionManifest):
        raise ValueError("manifest must be a Human13KUnionManifest")
    if not isinstance(acquisition, _TrajectoryCreditAcquisition):
        raise ValueError("acquisition must be sealed trajectory-credit evidence")
    manifest_sha = _manifest_sha256(manifest)
    if any(
        group.acquisition_group.identity.manifest_sha256 != manifest_sha
        for group in acquisition.groups
    ):
        raise ValueError("acquisition manifest lineage differs from supplied manifest")
    image_by_id = {image.image_id: image for image in manifest.images}
    if any(group.image_id not in image_by_id for group in acquisition.groups):
        raise ValueError("acquisition image is absent from manifest")
    matcher = manifest.binding.matcher
    if (
        matcher.algorithm != "cardinality_first_max_total_iou"
        or not matcher.same_category
        or matcher.duplicate_comparison != "strictly_greater"
    ):
        raise ValueError(
            "manifest matcher differs from canonical trajectory-credit matcher"
        )

    image_ledgers: list[ImageCreditLedger] = []
    for group in acquisition.groups:
        image = image_by_id[group.image_id]
        trusted_owner_ids = tuple(
            owner.owner_id for owner in image.owners if owner.stratum in {"G", "H"}
        )
        legacy_owner_ids = tuple(
            owner.owner_id for owner in image.owners if owner.stratum == "M"
        )
        if set(trusted_owner_ids) != set(image.g_owner_ids) | set(image.h_owner_ids):
            raise ValueError("manifest trusted G/H owner partition differs")
        if set(legacy_owner_ids) != set(image.m_owner_ids):
            raise ValueError("manifest legacy-M owner partition differs")
        if not trusted_owner_ids:
            raise ValueError("trajectory credit requires at least one trusted owner")
        owner_weight = 1.0 / len(trusted_owner_ids)
        trajectories = tuple(
            _project_one_trajectory(
                image,
                evidence,
                parsed,
                trusted_owner_ids=trusted_owner_ids,
                legacy_owner_ids=legacy_owner_ids,
                owner_weight=owner_weight,
                duplicate_iou_threshold=matcher.duplicate_iou_threshold,
                owner_iou_threshold=matcher.owner_iou_threshold,
            )
            for evidence, parsed in zip(
                group.acquisition_group.trajectories,
                group.parsed_trajectories,
                strict=True,
            )
        )
        trajectories, position_returns = _attach_rloo(trajectories)
        image_ledgers.append(
            ImageCreditLedger(
                image_id=group.image_id,
                acquisition_group_sha256=group.acquisition_group.content_sha256,
                trusted_owner_ids=trusted_owner_ids,
                legacy_m_owner_ids=legacy_owner_ids,
                owner_weight=owner_weight,
                trajectories=trajectories,
                position_returns=position_returns,
            )
        )
    return _construct_trajectory_credit_ledger(
        source_sha256=acquisition.groups[0].acquisition_group.identity.source_sha256,
        manifest_sha256=manifest_sha,
        acquisition_sha256=acquisition.content_sha256,
        logical_image_count=len(image_ledgers),
        logical_k=acquisition.logical_k,
        images=tuple(image_ledgers),
    )


def _build_trajectory_credit_ledger_for_test(
    manifest: Human13KUnionManifest,
    acquisition: _TrajectoryCreditAcquisition,
) -> TrajectoryCreditLedger:
    """Build an unadmitted formula fixture that public loss must reject."""

    return _project_trajectory_credit_ledger(manifest, acquisition)


def build_trajectory_credit_ledger(
    manifest: Human13KUnionManifest,
    acquisition: TrajectoryCreditPanelAcquisition,
    *,
    tokenizer_adapter: CanonicalTokenizerDecodeAdapter,
) -> TrajectoryCreditLedger:
    """Build detached credit only from exact Task-2 admitted publications.

    Canonical decode, parse, and token alignment are rerun inside this public
    boundary.  No caller-supplied category, box, row order, or token span is an
    admissible input.
    """

    if not isinstance(manifest, Human13KUnionManifest):
        raise ValueError("manifest must be a Human13KUnionManifest")
    if type(acquisition) is not TrajectoryCreditPanelAcquisition:
        raise ValueError(
            "trajectory credit requires a panel of exact Task2 AdmittedPublication"
        )
    if manifest.binding.surface.parser != PARSER_POLICY:
        raise ValueError("manifest parser policy differs from canonical parser policy")
    admitted = TrajectoryCreditPanelAcquisition(acquisition.publications)
    manifest_image_ids = tuple(image.image_id for image in manifest.images)
    plan_image_ids = tuple(
        publication.execution.plan.image_id for publication in admitted.publications
    )
    if plan_image_ids != manifest_image_ids:
        raise ValueError(
            "Task2 plan image coverage/order differs from supplied manifest"
        )
    manifest_sha256 = _manifest_sha256(manifest)
    if any(
        publication.replayed_group.identity.manifest_sha256 != manifest_sha256
        for publication in admitted.publications
    ):
        raise ValueError("Task2 publication manifest lineage differs")

    receipts = tuple(
        build_canonical_parser_projection_receipt(
            manifest,
            publication,
            tokenizer_adapter=tokenizer_adapter,
        )
        for publication in admitted.publications
    )
    internal_groups: list[_AcquisitionGroupCreditInput] = []
    for publication, receipt in zip(admitted.publications, receipts, strict=True):
        parsed_trajectories = tuple(
            _ParsedTrajectoryCreditInput(
                request_id=trajectory.request_id,
                rows=tuple(
                    _ParsedCreditRow(
                        generated_order=event.event_order,
                        category=event.category,
                        bbox=event.bbox,
                        token_start=event.token_start,
                        token_end=event.token_end,
                        geometry_valid=event.kind == "prediction",
                    )
                    for event in trajectory.events
                    if event.kind in {"prediction", "invalid"}
                ),
                malformed_spans=tuple(
                    _MalformedRowSpan(
                        generated_order=event.event_order,
                        token_start=event.token_start,
                        token_end=event.token_end,
                    )
                    for event in trajectory.events
                    if event.kind == "malformed"
                ),
            )
            for trajectory in receipt.trajectories
        )
        internal_groups.append(
            _AcquisitionGroupCreditInput(
                image_id=publication.execution.plan.image_id,
                acquisition_group=publication.replayed_group,
                parsed_trajectories=parsed_trajectories,
            )
        )
    internal = _TrajectoryCreditAcquisition(
        groups=tuple(internal_groups),
        logical_k=FIXED_SCIENTIFIC_K,
    )
    ledger = _project_trajectory_credit_ledger(manifest, internal)
    images = tuple(
        replace(
            image,
            plan_sha256=publication.execution.plan_sha256,
            native_receipts_sha256=publication.binding.native_receipts_sha256,
            parity_receipt_sha256=publication.binding.parity_receipt_sha256,
            parser_projection_sha256=receipt.content_sha256,
        )
        for image, publication, receipt in zip(
            ledger.images,
            admitted.publications,
            receipts,
            strict=True,
        )
    )
    return _construct_trajectory_credit_ledger(
        source_sha256=ledger.source_sha256,
        manifest_sha256=ledger.manifest_sha256,
        acquisition_sha256=admitted.content_sha256,
        logical_image_count=ledger.logical_image_count,
        logical_k=ledger.logical_k,
        images=images,
        training_repetition_penalty=admitted.training_repetition_penalty,
        seed_group_id=admitted.seed_group_id,
        admit_scientific=True,
    )


def _selected_tokens(
    ledger: TrajectoryCreditLedger,
    token_indices: Sequence[int] | None,
) -> tuple[TokenCredit, ...]:
    scored = ledger.scored_tokens
    if token_indices is None:
        return scored
    indices = tuple(token_indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(scored)
        for index in indices
    ):
        raise ValueError("microstep token index is outside the scored ledger")
    if len(set(indices)) != len(indices):
        raise ValueError("microstep token indices must be unique")
    return tuple(scored[index] for index in indices)


def _trajectory_score_function_numerator_impl(
    policy_logprobs: Any,
    ledger: TrajectoryCreditLedger,
    *,
    token_indices: Sequence[int] | None = None,
) -> Any:
    """Return ``-sum(log_pi * detached_A)`` without any denominator.

    ``token_indices`` partitions the canonical scored-token list for physical
    packs or accumulation microsteps.  Callers sum these numerators, then apply
    ``ledger.logical_denominator`` once.
    """

    import torch

    selected = _selected_tokens(ledger, token_indices)
    tensors: Mapping[str, Any]
    if isinstance(policy_logprobs, Mapping):
        tensors = policy_logprobs
    elif torch.is_tensor(policy_logprobs):
        if policy_logprobs.ndim != 1:
            raise ValueError("flat policy_logprobs must be one-dimensional")
        offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for image in ledger.images:
            for trajectory in image.trajectories:
                offsets[trajectory.request_id] = (
                    offset,
                    offset + trajectory.token_count,
                )
                offset += trajectory.token_count
        if policy_logprobs.numel() != offset:
            raise ValueError("flat policy_logprobs length differs from ledger tokens")
        tensors = {
            request_id: policy_logprobs[start:end]
            for request_id, (start, end) in offsets.items()
        }
    else:
        raise ValueError("policy_logprobs must be a tensor or request-id mapping")

    token_counts = {
        trajectory.request_id: trajectory.token_count
        for image in ledger.images
        for trajectory in image.trajectories
    }
    for request_id in {token.request_id for token in selected}:
        value = tensors.get(request_id)
        if (
            not torch.is_tensor(value)
            or value.ndim != 1
            or value.numel() != token_counts[request_id]
        ):
            raise ValueError("policy logprob tensor differs from sealed trajectory")
        if not torch.isfinite(value).all():
            raise ValueError("policy logprobs must be finite")
    terms = []
    for token in selected:
        logprob = tensors[token.request_id][token.token_index]
        advantage = logprob.new_tensor(token.advantage).detach()
        terms.append(-(logprob * advantage))
    if terms:
        return torch.stack(terms).sum()
    first = next(iter(tensors.values()), None)
    if first is None:
        raise ValueError("policy_logprobs mapping is empty")
    return first.sum() * 0.0


def _trajectory_score_function_numerator_for_test(
    policy_logprobs: Any,
    ledger: TrajectoryCreditLedger,
    *,
    token_indices: Sequence[int] | None = None,
) -> Any:
    """Private formula-only seam; it is not scientific ledger admission."""

    if type(ledger) is not TrajectoryCreditLedger:
        raise ValueError("ledger must be a TrajectoryCreditLedger")
    return _trajectory_score_function_numerator_impl(
        policy_logprobs,
        ledger,
        token_indices=token_indices,
    )


def trajectory_score_function_numerator(
    policy_logprobs: Any,
    ledger: TrajectoryCreditLedger,
    *,
    token_indices: Sequence[int] | None = None,
) -> Any:
    """Return an unnormalized numerator from an admitted scientific ledger."""

    admitted = _require_scientific_ledger_admission(ledger)
    return _trajectory_score_function_numerator_impl(
        policy_logprobs,
        admitted,
        token_indices=token_indices,
    )


def _trajectory_score_function_loss_for_test(
    policy_logprobs: Any,
    ledger: TrajectoryCreditLedger,
) -> Any:
    """Private exact-formula loss for explicitly small-K CPU fixtures."""

    return (
        _trajectory_score_function_numerator_for_test(policy_logprobs, ledger)
        / ledger.logical_denominator
    )


def trajectory_score_function_loss(
    policy_logprobs: Any,
    ledger: TrajectoryCreditLedger,
) -> Any:
    """Apply the one exact logical ``N*K`` denominator to the global numerator."""

    admitted = _require_scientific_ledger_admission(ledger)
    return (
        _trajectory_score_function_numerator_impl(policy_logprobs, admitted)
        / admitted.logical_denominator
    )


__all__ = [
    "CanonicalParserProjectionReceipt",
    "CanonicalProjectionEvent",
    "CanonicalTokenizerDecodeAdapter",
    "CanonicalTrajectoryProjection",
    "FIXED_SCIENTIFIC_K",
    "PANEL_ACQUISITION_SCHEMA_VERSION",
    "PARSER_PROJECTION_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TrajectoryCreditPanelAcquisition",
    "TrajectoryCreditLedger",
    "build_canonical_parser_projection_receipt",
    "build_trajectory_credit_ledger",
    "trajectory_score_function_loss",
    "trajectory_score_function_numerator",
]
