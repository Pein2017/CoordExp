"""Canonical parser projection for same-session HF K16 replay evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence, cast

from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    HFSharedSurfaceIdentity,
    SampledHFRequest,
)
from src.artifacts.json_values import json_sha256


class HFNativeProjectionError(ValueError):
    """Tokenizer, replay, parser, or surface lineage is not canonical."""


def hf_native_request_evidence_sha256(
    sampled_request: SampledHFRequest,
    replay_group: GradientReplayGroup,
) -> str:
    if type(replay_group) is not GradientReplayGroup:
        raise HFNativeProjectionError("native request evidence requires replay")
    if sampled_request not in replay_group.sampled_group.requests:
        raise HFNativeProjectionError(
            "native request evidence is absent from its replay group"
        )
    return json_sha256(
        {
            "schema_version": "human13_hf_native_request_evidence.v1",
            "request_id": sampled_request.request_id,
            "sampled_request": sampled_request.to_dict(),
            "sampled_group_sha256": replay_group.sampled_group.content_sha256,
            "replay_group_sha256": replay_group.content_sha256,
        }
    )


@dataclass(frozen=True)
class HFNativeTokenizerAttestation:
    """Private tokenizers snapshot bound to the loaded Qwen components."""

    tokenizer_object_id: int
    processor_object_id: int
    tokenizer_sha256: str
    tokenizer_class: str
    tokenizer_module: str
    base_model_path: str
    tokenizer_config_sha256: str
    backend_tokenizer_sha256: str
    tokenizers_package_version: str
    decode_backend_id: str
    decode_policy: str
    _decode_backend: Any
    _tokenizer_json_bytes: bytes

    @classmethod
    def from_qwen_components(
        cls,
        components: object,
        *,
        identity: HFSharedSurfaceIdentity,
        manifest: object,
    ) -> HFNativeTokenizerAttestation:
        import transformers
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        from src.config.fingerprint import sha256_file
        from src.qwen.runtime_loading import QwenComponents, _processor_identity
        from src.qwen.tokens import (
            tokenizer_config_class_matches_runtime,
            validate_qwen_token_identity,
        )

        if type(components) is not QwenComponents:
            raise HFNativeProjectionError(
                "HF-native projector requires exact loaded QwenComponents"
            )
        tokenizer = components.tokenizer
        processor = components.processor
        if (
            getattr(processor, "tokenizer", None) is not tokenizer
            or _processor_identity(processor, tokenizer)
            != components.processor_identity
            or validate_qwen_token_identity(tokenizer) != components.token_identity
        ):
            raise HFNativeProjectionError("loaded Qwen tokenizer identity drifted")
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise HFNativeProjectionError(
                "HF-native projector requires a real fast tokenizer"
            )
        tokenizer_class = type(tokenizer)
        transformers_root = Path(transformers.__file__).resolve().parent
        try:
            class_path = Path(inspect.getfile(tokenizer_class)).resolve()
            decode_path = Path(inspect.getfile(tokenizer_class.decode)).resolve()
        except (OSError, TypeError) as error:
            raise HFNativeProjectionError(
                "tokenizer implementation cannot be inspected"
            ) from error
        if (
            not class_path.is_relative_to(transformers_root)
            or not decode_path.is_relative_to(transformers_root)
            or "decode" in vars(tokenizer)
        ):
            raise HFNativeProjectionError("tokenizer decode implementation differs")
        base = Path(components.base_model_path).expanduser().resolve(strict=True)
        if Path(str(tokenizer.name_or_path)).expanduser().resolve(strict=True) != base:
            raise HFNativeProjectionError("tokenizer path differs from loaded Source")
        tokenizer_path = base / "tokenizer.json"
        config_path = base / "tokenizer_config.json"
        tokenizer_bytes = tokenizer_path.read_bytes()
        tokenizer_sha256 = sha256_file(tokenizer_path)
        tokenizer_config_sha256 = sha256_file(config_path)
        tokenizer_payload = json.loads(tokenizer_bytes.decode("utf-8"))
        tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
        binding = getattr(manifest, "binding", None)
        surface = getattr(binding, "surface", None)
        if (
            tokenizer_sha256 != identity.tokenizer_sha256
            or tokenizer_sha256 != getattr(surface, "tokenizer_sha256", None)
            or not tokenizer_config_class_matches_runtime(
                tokenizer_config.get("tokenizer_class"),
                tokenizer_class,
            )
            or getattr(surface, "tokenizer_class", None) != tokenizer_class.__name__
        ):
            raise HFNativeProjectionError(
                "tokenizer snapshot differs from surface/manifest identity"
            )
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if type(backend) is not Tokenizer or json.loads(backend.to_str()) != tokenizer_payload:
            raise HFNativeProjectionError("loaded tokenizer backend differs from snapshot")
        decode_backend = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        decode_payload = json.loads(decode_backend.to_str())
        if decode_payload != tokenizer_payload:
            raise HFNativeProjectionError("private decoder differs from tokenizer.json")
        version = metadata.version("tokenizers")
        return cls(
            tokenizer_object_id=id(tokenizer),
            processor_object_id=id(processor),
            tokenizer_sha256=tokenizer_sha256,
            tokenizer_class=tokenizer_class.__name__,
            tokenizer_module=tokenizer_class.__module__,
            base_model_path=str(base),
            tokenizer_config_sha256=tokenizer_config_sha256,
            backend_tokenizer_sha256=json_sha256(decode_payload),
            tokenizers_package_version=version,
            decode_backend_id="tokenizers.Tokenizer.from_str(tokenizer.json)",
            decode_policy=(
                "tokenizers-snapshot-skip-special-false-serialized-spacing-concat.v3"
            ),
            _decode_backend=decode_backend,
            _tokenizer_json_bytes=tokenizer_bytes,
        )

    @property
    def content_sha256(self) -> str:
        return json_sha256(
            {
                "schema_version": "human13_hf_native_tokenizer_attestation.v1",
                "tokenizer_object_id": self.tokenizer_object_id,
                "processor_object_id": self.processor_object_id,
                "tokenizer_sha256": self.tokenizer_sha256,
                "tokenizer_class": self.tokenizer_class,
                "tokenizer_module": self.tokenizer_module,
                "base_model_path": self.base_model_path,
                "tokenizer_config_sha256": self.tokenizer_config_sha256,
                "backend_tokenizer_sha256": self.backend_tokenizer_sha256,
                "tokenizers_package_version": self.tokenizers_package_version,
                "decode_backend_id": self.decode_backend_id,
                "decode_policy": self.decode_policy,
            }
        )

    def decode(self, token_ids: Sequence[int]) -> tuple[str, tuple[str, ...]]:
        from tokenizers import Tokenizer

        if (
            hashlib.sha256(self._tokenizer_json_bytes).hexdigest()
            != self.tokenizer_sha256
            or type(self._decode_backend) is not Tokenizer
            or metadata.version("tokenizers") != self.tokenizers_package_version
        ):
            raise HFNativeProjectionError("private tokenizer attestation drifted")
        ids = tuple(int(token) for token in token_ids)
        token_texts = tuple(
            str(self._decode_backend.decode([token], skip_special_tokens=False))
            for token in ids
        )
        sequence = str(
            self._decode_backend.decode(list(ids), skip_special_tokens=False)
        )
        if "".join(token_texts) != sequence:
            raise HFNativeProjectionError(
                "per-token text differs from canonical sequence decode"
            )
        return sequence, token_texts


@dataclass(frozen=True)
class _ProjectionIdentity:
    request_id: str
    generated_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class _ProjectionEvidence:
    identity: _ProjectionIdentity
    terminal_kind: str
    content_sha256: str


def project_hf_native_replay_groups(
    *,
    replay_groups: Sequence[GradientReplayGroup],
    manifest: object,
    manifest_image: object,
    attestation: HFNativeTokenizerAttestation,
) -> tuple[object, ...]:
    """Rerun canonical decode/parser/span projection over exact HF K16 tokens."""

    from scripts.research import human13_trajectory_credit as credit
    from scripts.research.build_human13_k_union_manifest import (
        Human13KUnionManifest,
        ImageRecord,
    )

    groups = tuple(replay_groups)
    if (
        not isinstance(manifest, Human13KUnionManifest)
        or not isinstance(manifest_image, ImageRecord)
        or manifest_image.image_id != 1584
        or len(groups) != 4
        or any(type(group) is not GradientReplayGroup for group in groups)
        or tuple(group.sampled_group.group_index for group in groups) != (0, 1, 2, 3)
    ):
        raise HFNativeProjectionError("HF-native projection coverage differs")
    identities = {group.sampled_group.identity for group in groups}
    if len(identities) != 1:
        raise HFNativeProjectionError("HF-native projection surface differs")
    identity = identities.pop()
    if (
        identity.tokenizer_sha256 != attestation.tokenizer_sha256
        or getattr(manifest_image, "image_sha256", None) != identity.image_sha256
        or not any(image is manifest_image for image in manifest.images)
    ):
        raise HFNativeProjectionError("HF-native projection tokenizer differs")
    dimensions = credit._canonical_panel_dimensions(manifest)
    width, height = dimensions[1584]
    projections: list[object] = []
    for group in groups:
        for request in group.sampled_group.requests:
            generated = tuple(token.chosen_token_id for token in request.tokens)
            evidence = _ProjectionEvidence(
                identity=_ProjectionIdentity(request.request_id, generated),
                terminal_kind=(
                    "natural_stop" if request.stop_reason == "im_end" else "cap"
                ),
                content_sha256=hf_native_request_evidence_sha256(request, group),
            )
            projections.append(
                credit._trajectory_projection(
                    evidence,
                    image_width=width,
                    image_height=height,
                    tokenizer_adapter=cast(Any, attestation),
                )
            )
    if len(projections) != 16:
        raise HFNativeProjectionError("HF-native projection must contain K16")
    return tuple(projections)


__all__ = [
    "HFNativeProjectionError",
    "HFNativeTokenizerAttestation",
    "hf_native_request_evidence_sha256",
    "project_hf_native_replay_groups",
]
