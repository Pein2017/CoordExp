#!/usr/bin/env python3
"""Lazy production composition boundary for the Human-13 RP crossover vertical.

Import and factory inspection are standard-library-only. Model, Torch, vLLM,
and experiment owners are imported only after the authorized node runner calls
the returned runtime. The module deliberately exposes one typed composition
boundary for the still-missing live sampled-trajectory pack/logit materializer;
it never substitutes a second sampler, trainer, parser, or matcher.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol
import uuid


NODE_RUNTIME_FACTORY_CONTRACT = "human13_rp_crossover_node_runtime_factory.v1"
UNIT_ID = "2026-08-14-human13-k-trajectory-rp-crossover-screen"
QUALIFICATION_SEEDS = tuple(range(30001, 30017))
EVALUATION_RPS = (1.0, 1.10)
CANONICAL_IMAGE_IDS = (
    1584,
    2299,
    2685,
    4134,
    5001,
    6040,
    7511,
    10707,
    13348,
    13923,
    14038,
    14439,
    16228,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT_PATH = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
    "checkpoints/step-2444"
)
BASE_MODEL_PATH = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/"
    "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
ADAPTER_TENSOR_PATH = SOURCE_CHECKPOINT_PATH / "adapter/adapter_model.safetensors"
SPECIAL_EMBEDDING_TENSOR_PATH = (
    SOURCE_CHECKPOINT_PATH
    / "special_token_embeddings/special_token_embeddings.safetensors"
)
MANIFEST_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-12-human13-k-union-to-greedy-overfit-screen/manifest/"
    "human13-k-union-manifest.json"
)
PANEL_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
PROMPT_CONFIG_PATH = (
    REPO_ROOT / "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
)
C_LEAF_PATH_BY_RP = {
    1.0: REPO_ROOT
    / "configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/"
    "03_rp100_trajectory_compiler_preservation.yaml",
    1.10: REPO_ROOT
    / "configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/"
    "06_rp110_trajectory_compiler_preservation.yaml",
}

SOURCE_CHECKPOINT_PAYLOAD_SHA256 = (
    "99678ea954c4b37abbf704432dbf43a8df5ce37cd07ebcef4e11f263782dca47"
)
ADAPTER_SHA256 = "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da"
SPECIAL_EMBEDDING_SHA256 = (
    "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2"
)
BASE_CONFIG_SHA256 = "c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de"
TOKENIZER_SHA256 = "ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8"
PANEL_SHA256 = "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"
MANIFEST_SHA256 = "a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb"
PROMPT_CONFIG_SHA256 = (
    "d2217208bc3e419bc9d8c621b4842da262d216780eb1dfc88eade0edf1358f6b"
)
PROMPT_POLICY_FINGERPRINT = (
    "0b4fa411f289ccc29e3f6b59c65d32689ac04e6a014891cbe2dc2d976de634c1"
)
ALIAS_BANK_SHA256 = "3f64952f000380372f894443da5fe038265c1624d9449a5b0506b84cc57cc6ca"
C_LEAF_SHA256_BY_RP = {
    1.0: "b826fa9f70ec7a5772bdf5c023e31d48b5721d04bc92159c51f606c8e67d38e3",
    1.10: "d7c1a5de691a006a065e4b35ea212b7ea743b3748ae30d0ea739a4830aed0ddf",
}
ADAMW_CONFIG_SHA256 = "b938f99b3cb5145085eedd439e391e89348d58d96844180a73c0ff53e4dbf772"


class ProductionCompositionUnavailable(RuntimeError):
    """Raised when the existing owners cannot yet materialize a live node."""


@dataclass(frozen=True)
class FrozenProductionInputs:
    source_checkpoint_path: str
    source_checkpoint_payload_sha256: str
    base_model_path: str
    adapter_tensor_path: str
    adapter_sha256: str
    special_embedding_tensor_path: str
    special_embedding_sha256: str
    manifest_path: str
    manifest_sha256: str
    panel_path: str
    panel_sha256: str
    tokenizer_path: str
    tokenizer_sha256: str
    prompt_config_path: str
    prompt_config_sha256: str
    prompt_policy_fingerprint: str
    alias_bank_sha256: str
    c_leaf_path: str
    c_leaf_sha256: str


@dataclass(frozen=True)
class QualificationAcquisition:
    """Typed result returned by the existing acquisition/semantic owners."""

    cell_specs: tuple[Any, ...]
    source_baselines: tuple[Any, ...]
    training_rp: float
    seeds: tuple[int, ...]
    image_ids: tuple[int, ...]
    native_request_count: int
    native_batch_count: int
    manifest_sha256: str
    tokenizer_sha256: str
    prompt_policy_fingerprint: str
    alias_bank_sha256: str
    nested_objective_hashes: Mapping[str, tuple[tuple[str, str], ...]]
    streaming_mode: str


@dataclass(frozen=True)
class AcquisitionReleaseReceipt:
    engine_closed: bool
    model_released: bool
    panel_wide_logits_retained: bool


class LiveNodeComposition(Protocol):
    """One injected bridge over existing Task2-5 and live-model owners.

    ``acquire_qualification`` owns native batch-four sampling, streamed packed
    replay, canonical Task3/4 materialization, and typed publication. It must
    not return until all A/B/C objective identities exist. ``close_acquisition``
    must release the vLLM engine/model before ``services_for_cell`` assembles a
    fresh FA2/MRoPE training model and Task5/HF services.
    """

    def acquire_qualification(
        self, node: Mapping[str, Any], frozen: FrozenProductionInputs
    ) -> QualificationAcquisition: ...

    def close_acquisition(self) -> AcquisitionReleaseReceipt: ...

    def services_for_cell(self, spec: Any) -> Any: ...


class _UnavailableLiveComposition:
    def acquire_qualification(
        self, node: Mapping[str, Any], frozen: FrozenProductionInputs
    ) -> QualificationAcquisition:
        raise ProductionCompositionUnavailable(
            "no existing owner exposes the sampled-trajectory packed-logit "
            "materializer plus native batch receipt builder required to compose "
            "Task2 replay with the no-padding FA2/MRoPE backward; refusing to "
            "substitute a second sampler or trainer"
        )

    def close_acquisition(self) -> AcquisitionReleaseReceipt:
        return AcquisitionReleaseReceipt(
            engine_closed=True,
            model_released=True,
            panel_wide_logits_retained=False,
        )

    def services_for_cell(self, spec: Any) -> Any:
        raise ProductionCompositionUnavailable(
            "training services are unavailable before admitted acquisition"
        )


def _plan_contract() -> dict[str, Any]:
    return {
        "schema_version": NODE_RUNTIME_FACTORY_CONTRACT,
        "arms": ["A", "B", "C"],
        "evaluation_rps": list(EVALUATION_RPS),
        "max_updates": 1,
        "retry_policy": "none",
        "world_size": 1,
        "requires_user_model_gpu_authority": True,
        "actions": {
            "model_loads": 0,
            "gpu_allocations": 0,
            "subprocess_launches": 0,
            "output_roots_created": 0,
        },
    }


def _sha256_file(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    if _sha256_file(path, label=label) != expected:
        raise ValueError(f"{label} SHA-256 drifted")


def _require_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"{label} is not a regular directory: {path}")


def _validate_node(node: Mapping[str, Any]) -> float:
    if not isinstance(node, Mapping):
        raise ValueError("production node must be a sealed mapping")
    rp = node.get("training_rp")
    if isinstance(rp, bool) or not isinstance(rp, (int, float)):
        raise ValueError("qualification training RP must be numeric")
    training_rp = float(rp)
    expected_node_id = {
        1.0: "rp100:qualification",
        1.10: "rp110:qualification",
    }.get(training_rp)
    if (
        expected_node_id is None
        or node.get("node_id") != expected_node_id
        or node.get("phase") != "qualification"
        or tuple(node.get("seeds", ())) != QUALIFICATION_SEEDS
    ):
        raise ValueError("production factory accepts only an exact qualification node")
    acquisition = node.get("acquisition_key")
    if not isinstance(acquisition, Mapping) or (
        float(acquisition.get("training_rp", -1.0)),
        acquisition.get("seed_group_id"),
        acquisition.get("phase"),
        tuple(acquisition.get("seeds", ())),
    ) != (training_rp, "qualification", "qualification", QUALIFICATION_SEEDS):
        raise ValueError("qualification acquisition identity or seeds drifted")
    cells = node.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError("qualification node cells are malformed")
    if len(cells) != 1 or not isinstance(cells[0], Mapping):
        raise ValueError("qualification node must contain exactly one C cell")
    cell = cells[0]
    cell_key = cell.get("cell_key")
    if not isinstance(cell_key, Mapping) or cell_key.get("arm_id") != "C":
        raise ValueError("qualification node must execute preservation arm C only")
    if (
        cell_key.get("acquisition_key") != acquisition
        or tuple(cell.get("objective_components", ()))
        != ("trajectory", "compiler", "preservation")
        or tuple(cell.get("evaluation_rps", ())) != EVALUATION_RPS
        or cell.get("world_size") != 1
        or cell.get("max_updates") != 1
        or cell.get("retry_policy") != "none"
        or cell.get("source") != "fresh"
        or cell.get("optimizer") != "fresh_adamw"
        or cell.get("adamw_config_sha256") != ADAMW_CONFIG_SHA256
    ):
        raise ValueError("qualification C cell execution contract drifted")
    if Path(str(cell.get("output_root", ""))).exists():
        raise FileExistsError("refusing to reuse a qualification cell output root")
    return training_rp


def _validate_frozen_inputs(training_rp: float) -> FrozenProductionInputs:
    _require_directory(SOURCE_CHECKPOINT_PATH, label="Source checkpoint")
    _require_directory(BASE_MODEL_PATH, label="Source base model")
    from scripts.research.human13_live_eval import checkpoint_payload_sha256

    if (
        checkpoint_payload_sha256(SOURCE_CHECKPOINT_PATH)
        != SOURCE_CHECKPOINT_PAYLOAD_SHA256
    ):
        raise ValueError("Source checkpoint payload SHA-256 drifted")
    _require_hash(ADAPTER_TENSOR_PATH, ADAPTER_SHA256, label="Source adapter")
    _require_hash(
        SPECIAL_EMBEDDING_TENSOR_PATH,
        SPECIAL_EMBEDDING_SHA256,
        label="Source special-token embeddings",
    )
    _require_hash(
        BASE_MODEL_PATH / "config.json", BASE_CONFIG_SHA256, label="base config"
    )
    _require_hash(
        BASE_MODEL_PATH / "tokenizer.json", TOKENIZER_SHA256, label="tokenizer"
    )
    _require_hash(MANIFEST_PATH, MANIFEST_SHA256, label="Human-13 manifest")
    _require_hash(PANEL_PATH, PANEL_SHA256, label="Human-13 panel")
    _require_hash(
        PROMPT_CONFIG_PATH, PROMPT_CONFIG_SHA256, label="Source prompt config"
    )
    leaf_path = C_LEAF_PATH_BY_RP[training_rp]
    leaf_sha256 = C_LEAF_SHA256_BY_RP[training_rp]
    _require_hash(leaf_path, leaf_sha256, label="qualification C leaf config")
    return FrozenProductionInputs(
        source_checkpoint_path=str(SOURCE_CHECKPOINT_PATH),
        source_checkpoint_payload_sha256=SOURCE_CHECKPOINT_PAYLOAD_SHA256,
        base_model_path=str(BASE_MODEL_PATH),
        adapter_tensor_path=str(ADAPTER_TENSOR_PATH),
        adapter_sha256=ADAPTER_SHA256,
        special_embedding_tensor_path=str(SPECIAL_EMBEDDING_TENSOR_PATH),
        special_embedding_sha256=SPECIAL_EMBEDDING_SHA256,
        manifest_path=str(MANIFEST_PATH),
        manifest_sha256=MANIFEST_SHA256,
        panel_path=str(PANEL_PATH),
        panel_sha256=PANEL_SHA256,
        tokenizer_path=str(BASE_MODEL_PATH / "tokenizer.json"),
        tokenizer_sha256=TOKENIZER_SHA256,
        prompt_config_path=str(PROMPT_CONFIG_PATH),
        prompt_config_sha256=PROMPT_CONFIG_SHA256,
        prompt_policy_fingerprint=PROMPT_POLICY_FINGERPRINT,
        alias_bank_sha256=ALIAS_BANK_SHA256,
        c_leaf_path=str(leaf_path),
        c_leaf_sha256=leaf_sha256,
    )


def _validate_release(receipt: AcquisitionReleaseReceipt) -> None:
    if not isinstance(receipt, AcquisitionReleaseReceipt) or (
        receipt.engine_closed,
        receipt.model_released,
        receipt.panel_wide_logits_retained,
    ) != (True, True, False):
        raise RuntimeError(
            "acquisition must release vLLM and retain no panel-wide logits before training"
        )


def _validate_acquisition(
    acquired: QualificationAcquisition,
    *,
    node: Mapping[str, Any],
    training_rp: float,
) -> tuple[Any, ...]:
    if not isinstance(acquired, QualificationAcquisition):
        raise TypeError("live composition must return QualificationAcquisition")
    if (
        acquired.training_rp != training_rp
        or acquired.seeds != QUALIFICATION_SEEDS
        or acquired.image_ids != CANONICAL_IMAGE_IDS
        or acquired.native_request_count != 208
        or acquired.native_batch_count != 52
        or acquired.manifest_sha256 != MANIFEST_SHA256
        or acquired.tokenizer_sha256 != TOKENIZER_SHA256
        or acquired.prompt_policy_fingerprint != PROMPT_POLICY_FINGERPRINT
        or acquired.alias_bank_sha256 != ALIAS_BANK_SHA256
        or acquired.streaming_mode != "per_image_or_pack"
    ):
        raise ValueError(
            "qualification acquisition or frozen semantic identity drifted"
        )
    nested = acquired.nested_objective_hashes
    if not isinstance(nested, Mapping) or set(nested) != {"A", "B", "C"}:
        raise ValueError(
            "qualification must materialize exact A/B/C objective identities"
        )
    a = tuple(nested["A"])
    b = tuple(nested["B"])
    c = tuple(nested["C"])
    if (
        tuple(name for name, _ in a) != ("trajectory",)
        or tuple(name for name, _ in b) != ("trajectory", "compiler")
        or tuple(name for name, _ in c) != ("trajectory", "compiler")
        or b[:1] != a
        or c != b
    ):
        raise ValueError("qualification A/B/C objective bytes are not exactly nested")
    from scripts.research.human13_rp_crossover_matrix_contracts import (
        AcquisitionKey,
        CellSpec,
        SourceBaselineRef,
    )

    baselines = tuple(acquired.source_baselines)
    if (
        len(baselines) != 2
        or any(not isinstance(item, SourceBaselineRef) for item in baselines)
        or tuple(item.evaluation_rp for item in baselines) != EVALUATION_RPS
        or any(
            item.checkpoint_sha256 != SOURCE_CHECKPOINT_PAYLOAD_SHA256
            or item.image_ids != CANONICAL_IMAGE_IDS
            for item in baselines
        )
    ):
        raise ValueError(
            "qualification requires two fresh deterministic Source baselines per RP"
        )
    specs = tuple(acquired.cell_specs)
    if len(specs) != 1 or not isinstance(specs[0], CellSpec):
        raise ValueError("qualification acquisition must return exactly one C CellSpec")
    spec = specs[0]
    planned = node["cells"][0]
    if (
        spec.cell_key.acquisition_key
        != AcquisitionKey.from_dict(node["acquisition_key"])
        or spec.cell_key.arm_id != "C"
        or spec.objective_component_hashes != c
        or spec.leaf_config_sha256 != C_LEAF_SHA256_BY_RP[training_rp]
        or spec.source_checkpoint_sha256 != SOURCE_CHECKPOINT_PAYLOAD_SHA256
        or spec.shared_evidence.manifest_sha256 != MANIFEST_SHA256
        or spec.output_root != planned["output_root"]
    ):
        raise ValueError("qualification C CellSpec differs from frozen live evidence")
    return specs


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite production receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


class ProductionNodeRuntime:
    """Node-level phase owner used by the final launcher/runner contract."""

    def __init__(
        self, node: Mapping[str, Any], *, composition: LiveNodeComposition
    ) -> None:
        self._node = node
        self._composition = composition
        self._acquisition_released = False
        self._spec_output_roots: dict[str, str] = {}

    def acquire_cell_specs(self, node: Mapping[str, Any]) -> tuple[Any, ...]:
        if node is not self._node and dict(node) != dict(self._node):
            raise ValueError("node runtime received a different sealed node")
        training_rp = _validate_node(node)
        frozen = _validate_frozen_inputs(training_rp)
        acquired: QualificationAcquisition | None = None
        try:
            acquired = self._composition.acquire_qualification(node, frozen)
        finally:
            release = self._composition.close_acquisition()
            _validate_release(release)
            self._acquisition_released = True
        specs = _validate_acquisition(acquired, node=node, training_rp=training_rp)
        self._spec_output_roots = {
            spec.cell_key.content_sha256: spec.output_root for spec in specs
        }
        return specs

    def services_for_cell(self, spec: Any) -> Any:
        if not self._acquisition_released:
            raise RuntimeError("training model cannot load before acquisition release")
        if spec.cell_key.content_sha256 not in self._spec_output_roots:
            raise ValueError("cell service requested for an unacquired CellSpec")
        return self._composition.services_for_cell(spec)

    def write_cell_receipt(self, receipt: Any) -> None:
        key = receipt.cell_key.content_sha256
        try:
            output_root = self._spec_output_roots[key]
        except KeyError as exc:
            raise ValueError("cell receipt is not bound to this node") from exc
        _atomic_write_json(Path(output_root) / "cell-receipt.json", receipt.to_dict())

    def write_node_terminal_receipt(
        self, path: str, payload: Mapping[str, Any]
    ) -> None:
        _atomic_write_json(path, payload)


def _declare_runtime_factory(factory: Any) -> Any:
    """Declare the runner contract without importing its Torch-bearing module."""

    factory.node_runtime_factory_contract = NODE_RUNTIME_FACTORY_CONTRACT
    factory.plan_contract = _plan_contract
    return factory


@_declare_runtime_factory
def create_node_runtime(
    node: Mapping[str, Any],
    *,
    _composition: LiveNodeComposition | None = None,
) -> ProductionNodeRuntime:
    """Create one node runtime; all validation/live action remains execute-time."""

    composition = _composition or _UnavailableLiveComposition()
    return ProductionNodeRuntime(node, composition=composition)


__all__ = [
    "ALIAS_BANK_SHA256",
    "ADAMW_CONFIG_SHA256",
    "AcquisitionReleaseReceipt",
    "C_LEAF_SHA256_BY_RP",
    "FrozenProductionInputs",
    "MANIFEST_SHA256",
    "ProductionCompositionUnavailable",
    "ProductionNodeRuntime",
    "PROMPT_POLICY_FINGERPRINT",
    "QualificationAcquisition",
    "SOURCE_CHECKPOINT_PAYLOAD_SHA256",
    "TOKENIZER_SHA256",
    "create_node_runtime",
]
