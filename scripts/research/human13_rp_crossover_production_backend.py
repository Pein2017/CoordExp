"""Concrete execute-time backend for the Human-13 RP-crossover composition.

Importing or constructing this module is plan-only.  Every model, engine,
Torch, tokenizer, and filesystem action is delayed until a protocol method is
called by :mod:`human13_rp_crossover_live_composition`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import resource
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_UPDATED_ARM = (
    REPO_ROOT / "configs/coordexp_swift/research/human13_k_union/05_a4.yaml"
)
_CAP_STOP_REASONS = frozenset({"length", "max_tokens", "max_new_tokens"})
_NATURAL_STOP_REASONS = frozenset({"im_end"})
_PARSEABLE_STATUSES = frozenset({"accepted", "accepted_with_drops"})
_UNPARSEABLE_STATUSES = frozenset({"all_spans_dropped", "empty", "unsupported_format"})


class ProductionBackendError(RuntimeError):
    """Raised when live evidence cannot be projected through an existing seam."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_text(value: Any) -> str:
    return _sha256(value)


def _release_model_caches() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass


def _regular_output(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite live evidence: {path}")
    return path


@dataclass
class _SamplerHandle:
    session: Any
    base_requests: Mapping[int, Any]
    session_identity_sha256: str
    model_id: str
    model_identity_sha256: str
    tokenizer_id: str
    processor_id: str
    sampler_backend_id: str
    frozen: Any
    closed: bool = False


@dataclass
class _PackedHandle:
    assembly: Any
    skeletons: Mapping[int, Any]
    manifest: Any
    frozen: Any
    closed: bool = False


class _MarginSurface:
    """Fresh rank-16 Source DoRA on the canonical HF fp32/SDPA history seam."""

    def __init__(
        self,
        *,
        model: Any,
        components: Any,
        scorer: Any,
        session: Any,
        skeletons: Mapping[int, Any],
    ) -> None:
        self._model: Any = model
        self._components: Any = components
        self._scorer: Any = scorer
        self._session: Any = session
        self._skeletons = dict(skeletons)
        self._closed = False

    def named_trainable_parameters(self) -> tuple[tuple[str, Any], ...]:
        if self._closed:
            raise ProductionBackendError("witness margin surface is closed")
        bound = tuple(
            (name, parameter)
            for name, parameter in self._model.named_parameters()
            if parameter.requires_grad
        )
        if not bound:
            raise ProductionBackendError(
                "witness surface has no Source DoRA trainables"
            )
        return bound

    def raw_logit_rows(self, decode: Any, token_indices: Sequence[int]) -> Any:
        if self._closed:
            raise ProductionBackendError("witness margin surface is closed")
        from scripts.research.human13_live_census import _clone_skeleton

        skeleton = self._skeletons[int(decode.image_id)]
        prompt_count = len(decode.prompt_token_ids)
        if tuple(skeleton.input_ids[:prompt_count]) != tuple(decode.prompt_token_ids):
            raise ProductionBackendError(
                "witness prompt differs from canonical processor skeleton"
            )
        segment_id = f"rp-crossover-witness:{decode.surface_key}"
        encoded = _clone_skeleton(
            skeleton,
            segment_id=segment_id,
            image_id=int(decode.image_id),
            input_ids=(*decode.prompt_token_ids, *decode.generated_token_ids),
        )
        positions = tuple(prompt_count - 1 + int(index) for index in token_indices)
        result = self._scorer.score_causal_logits_with_grad(encoded, positions)
        if tuple(result.logits_position_ids) != positions:
            raise ProductionBackendError("witness HF causal positions differ")
        return result.logits[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._session.close()
        finally:
            self._scorer = None
            self._session = None
            self._skeletons.clear()
            self._components = None
            self._model = None
            _release_model_caches()


class Human13RPCrossoverProductionBackend:
    """Mechanical projection of the four established live owners."""

    def __init__(self, node: Mapping[str, Any]) -> None:
        if not isinstance(node, Mapping):
            raise TypeError("production backend requires one sealed node mapping")
        raw_cells = node.get("cells")
        if (
            not isinstance(raw_cells, Sequence)
            or isinstance(raw_cells, (str, bytes))
            or any(not isinstance(cell, Mapping) for cell in raw_cells)
        ):
            raise ValueError("production backend requires typed node cells")
        cells = tuple(raw_cells)
        if not cells:
            raise ValueError("production backend requires a node with cells")
        roots = {Path(str(cell["output_root"])).expanduser().parent for cell in cells}
        if len(roots) != 1:
            raise ValueError("production backend node cells do not share one root")
        self._node = node
        self._acquisition_root = roots.pop() / "acquisition"
        self._source_surfaces: dict[float, Any] = {}
        self._source_outputs: dict[float, tuple[Mapping[str, Any], ...]] = {}
        self._packed: _PackedHandle | None = None
        self._active_state: Any | None = None
        self._cell_metrics: dict[str, int | None] = {}

    # -- Source baselines/frontiers -------------------------------------

    def source_surface(self, frozen: Any, *, repetition_penalty: float) -> Any:
        from scripts.research.build_human13_k_union_manifest import load_manifest
        from scripts.research.build_human13_on_policy_frontier import (
            CheckpointIdentity,
            build_frontier_iteration,
            canonical_write,
        )
        from scripts.research.human13_live_eval import (
            current_decodes_from_outputs,
            evaluate_hf_checkpoint,
            hf_runtime_identity_from_outputs,
            write_outputs_jsonl,
        )
        from scripts.research.human13_rp_crossover_live_composition import (
            SourceSurfaceEvidence,
        )
        from scripts.research.human13_rp_crossover_matrix_contracts import (
            SourceBaselineRef,
        )

        rp = float(repetition_penalty)
        if rp in self._source_surfaces:
            raise ProductionBackendError("Source surface was requested more than once")
        manifest = load_manifest(frozen.manifest_path, require_full_panel=True)
        tag = "rp100" if rp == 1.0 else "rp110"
        root = self._acquisition_root / "source" / tag
        outputs: list[tuple[Mapping[str, Any], ...]] = []
        output_hashes: list[str] = []
        for serial in ("a", "b"):
            raw = evaluate_hf_checkpoint(
                manifest=manifest,
                manifest_sha256=frozen.manifest_sha256,
                checkpoint_path=frozen.source_checkpoint_path,
                arm_id=f"source-{tag}",
                milestone=0,
                run_id=f"human13-rp-crossover-source-{tag}",
                run_root=str(self._acquisition_root),
                resolved_arm_plan_sha256=frozen.c_leaf_sha256,
                resolved_config_sha256=frozen.c_leaf_sha256,
                source_config_path=frozen.prompt_config_path,
                repetition_penalty=rp,
            )
            hf_runtime_identity_from_outputs(raw)
            normalized = tuple({**dict(item), "runtime": {}} for item in raw)
            path = _regular_output(root / f"baseline-{serial}.jsonl")
            write_outputs_jsonl(path, normalized)
            outputs.append(normalized)
            output_hashes.append(_sha256_file(path))
        if outputs[0] != outputs[1] or output_hashes[0] != output_hashes[1]:
            raise ProductionBackendError(
                "repeated Source clean-greedy baselines are not byte-identical"
            )
        checkpoint = CheckpointIdentity(
            path=frozen.source_checkpoint_path,
            payload_sha256=frozen.source_checkpoint_payload_sha256,
        )
        current = current_decodes_from_outputs(
            manifest=manifest,
            manifest_sha256=frozen.manifest_sha256,
            outputs=outputs[0],
            checkpoint=checkpoint,
            repetition_penalty=rp,
        )
        frontier = build_frontier_iteration(
            manifest,
            manifest_path=frozen.manifest_path,
            iteration=0,
            checkpoint=checkpoint,
            decodes=current,
        )
        frontier_path = _regular_output(root / "frontier.json")
        canonical_write(frontier, frontier_path)
        decodes = self._sealed_source_decodes(
            manifest=manifest,
            outputs=outputs[0],
            frontier=frontier,
            repetition_penalty=rp,
        )
        baseline = SourceBaselineRef(
            evaluation_rp=rp,
            output_a_sha256=output_hashes[0],
            output_b_sha256=output_hashes[1],
            checkpoint_sha256=frozen.source_checkpoint_payload_sha256,
            image_ids=tuple(int(image.image_id) for image in manifest.images),
        )
        surface = SourceSurfaceEvidence(
            repetition_penalty=rp,
            baseline=baseline,
            decodes=decodes,
            frontier=frontier,
            frontier_path=str(frontier_path),
            outputs=outputs[0],
        )
        self._source_outputs[rp] = outputs[0]
        self._source_surfaces[rp] = surface
        return surface

    @staticmethod
    def _sealed_source_decodes(
        *,
        manifest: Any,
        outputs: Sequence[Mapping[str, Any]],
        frontier: Any,
        repetition_penalty: float,
    ) -> tuple[Any, ...]:
        from scripts.research.analyze_human13_k_union import _match_prefix
        from scripts.research.human13_adamw_proposal_preservation import (
            LEGACY_M_OWNER_CLASS,
            TRUSTED_OWNER_CLASS,
        )
        from scripts.research.human13_rp_crossover_witness import (
            SealedOwnerRow,
            SealedSourceDecode,
        )

        matcher = manifest.binding.matcher
        decodes = []
        for manifest_image, output, frontier_image in zip(
            manifest.images, outputs, frontier.images, strict=True
        ):
            predictions = output.get("predictions")
            prompt = output.get("prompt_token_ids")
            generated = output.get("generated_token_ids")
            if not isinstance(predictions, list) or not isinstance(prompt, list):
                raise ProductionBackendError(
                    "Source output lacks canonical prediction or prompt evidence"
                )
            if not isinstance(generated, list):
                raise ProductionBackendError("Source output lacks generated token IDs")
            matched = _match_prefix(
                manifest_image,
                predictions,
                duplicate_iou_threshold=matcher.duplicate_iou_threshold,
                owner_iou_threshold=matcher.owner_iou_threshold,
            )
            rows = {row.generated_order: row for row in frontier_image.rows}
            owners = {owner.owner_id: owner for owner in manifest_image.owners}
            owner_rows = []
            for owner_id, receipt in matched["owner_matches"].items():
                row = rows[int(receipt["generated_order"])]
                owner = owners[owner_id]
                owner_rows.append(
                    SealedOwnerRow(
                        owner_id=owner_id,
                        owner_class=(
                            LEGACY_M_OWNER_CLASS
                            if owner.stratum == "M"
                            else TRUSTED_OWNER_CLASS
                        ),
                        token_start=int(row.token_start),
                        token_end=int(row.token_end),
                    )
                )
            decodes.append(
                SealedSourceDecode(
                    image_id=int(frontier_image.image_id),
                    repetition_penalty=repetition_penalty,
                    prompt_token_ids=tuple(int(value) for value in prompt),
                    generated_token_ids=tuple(int(value) for value in generated),
                    owner_rows=tuple(
                        sorted(
                            owner_rows,
                            key=lambda row: (
                                row.token_start,
                                row.token_end,
                                row.owner_id,
                            ),
                        )
                    ),
                )
            )
        return tuple(decodes)

    # -- witness surface -------------------------------------------------

    def open_margin_surface(self, frozen: Any) -> _MarginSurface:
        from scripts.research.human13_hf_census import (
            Human13HFCensusScorer,
            _load_source_inputs,
        )
        from scripts.research.human13_live_model import (
            DefaultHuman13AssemblyBackend,
            build_human13_processor_skeletons,
        )
        from scripts.research.build_human13_k_union_manifest import load_manifest
        from src.inference.hf_backend import open_hf_backend_session

        plan = self._qualification_plan(
            frozen, learning_rate=frozen.default_qualification_learning_rate
        )
        witness_plan = replace(plan, mixed_precision="fp32", attn_implementation="sdpa")
        owner = DefaultHuman13AssemblyBackend()
        components = adapted = special = model = bound_components = None
        skeletons = session = None
        try:
            components = owner.load_qwen(witness_plan)
            adapted = owner.warm_start_language_dora(
                components.model, components, witness_plan, repo_root=REPO_ROOT
            )
            special = owner.load_and_freeze_special_token_delta(
                adapted.model, components, witness_plan, repo_root=REPO_ROOT
            )
            model = special.model
            model.eval()
            bound_components = replace(components, model=model)
            manifest = load_manifest(frozen.manifest_path, require_full_panel=True)
            skeletons = build_human13_processor_skeletons(
                manifest, bound_components, repo_root=REPO_ROOT
            )
            launch, requests = _load_source_inputs(REPO_ROOT)

            def load_components(_launch: Any) -> Any:
                return SimpleNamespace(
                    qwen=bound_components,
                    adapter_receipt={"mode": "warm_start_expand_dora", "rank": 16},
                    embedding_delta_receipt={"mode": "frozen_source_delta"},
                )

            session = open_hf_backend_session(launch, components_loader=load_components)
            scorer = Human13HFCensusScorer(
                session=session, requests_by_image=requests, launch=launch
            )
        except BaseException as primary:
            if session is not None:
                try:
                    session.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "margin-surface cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            session = skeletons = bound_components = model = None
            special = adapted = components = None
            _release_model_caches()
            raise
        return _MarginSurface(
            model=model,
            components=bound_components,
            scorer=scorer,
            session=session,
            skeletons=skeletons,
        )

    def close_margin_surface(self, surface: Any) -> None:
        if not isinstance(surface, _MarginSurface):
            raise ProductionBackendError("unknown witness margin surface")
        surface.close()

    # -- native vLLM acquisition ----------------------------------------

    def open_sampler(self, frozen: Any) -> _SamplerHandle:
        from dataclasses import replace as dc_replace

        from scripts.research.collect_human13_discovery import (
            K_CONFIG_PATH,
            _session_identity,
            load_exact_config,
        )
        from scripts.research.run_current_seeded_sampled_rollouts import (
            _build_requests,
            physical_image_id,
        )
        from scripts.research.human13_rp_crossover_matrix_contracts import (
            CANONICAL_IMAGE_IDS,
        )
        from src.config.fingerprint import sha256_json
        from src.data import load_raw_examples
        from src.inference.backend import GenerationPolicy
        from src.inference.pipeline import _resolve_execution_model_for_run
        from src.inference.runtime import assemble_frontend
        from src.inference.vllm_backend import open_vllm_backend_session

        resolved = load_exact_config("k", K_CONFIG_PATH)
        execution_model = _resolve_execution_model_for_run(resolved)
        if not isinstance(execution_model, Mapping):
            raise ProductionBackendError("K execution-model receipt is unavailable")
        frontend = assemble_frontend(
            resolved.config,
            generation_config_fingerprint=sha256_json(
                resolved.config.generation.model_dump(mode="json")
            ),
            execution_model=execution_model,
        )
        if frontend.qwen.tokenizer_sha256 != frozen.tokenizer_sha256:
            raise ProductionBackendError("sampler tokenizer differs from frozen input")
        examples = tuple(load_raw_examples(resolved.config.data.input_jsonl))
        requests, _ = _build_requests(resolved.config, frontend, examples)
        by_image = {}
        for example, request in zip(examples, requests, strict=True):
            image_id = int(physical_image_id(example))
            by_image[image_id] = dc_replace(
                request,
                request_id=f"human13:{image_id}:rp-crossover-base",
                generation_policy=GenerationPolicy(
                    max_new_tokens=512,
                    repetition_penalty=1.0,
                    temperature=0.0,
                    top_p=1.0,
                    include_raw_model_logprob=False,
                ),
            )
        if tuple(by_image) != CANONICAL_IMAGE_IDS:
            raise ProductionBackendError(
                "native sampler inputs differ from the canonical 13-image order"
            )
        session = open_vllm_backend_session(frontend.launch)
        try:
            identity = _session_identity(session.receipt)
            receipt = session.receipt.to_artifact_dict()
            model_identity = receipt["model_identity"]
            processor_identity = receipt["processor_identity"]
            return _SamplerHandle(
                session=session,
                base_requests=by_image,
                session_identity_sha256=str(identity["session_identity_sha256"]),
                model_id=frozen.source_checkpoint_path,
                model_identity_sha256=_identity_text(model_identity),
                tokenizer_id=frozen.base_model_path,
                processor_id=_identity_text(processor_identity),
                sampler_backend_id=(
                    f"vllm:{session.receipt.backend_version}:"
                    f"{session.receipt.backend_mode}"
                ),
                frozen=frozen,
            )
        except BaseException:
            session.close()
            raise

    def sample_batch(
        self, sampler: _SamplerHandle, batch: Any, params: tuple[Any, ...]
    ) -> Any:
        from dataclasses import replace as dc_replace

        from scripts.research.collect_human13_rp_crossover import (
            NativeBatchReceipt,
            NativeOutputReceipt,
            NativeRequestReceipt,
            PROCESSOR_ORDER,
            native_sampling_evidence,
            token_ids_sha256,
        )
        from src.inference.backend import GenerationPolicy
        from src.inference.vllm_backend import (
            _close_prompt_images,
            _restore_native_request_order,
        )

        if sampler.closed:
            raise ProductionBackendError("native sampler is already closed")
        if len(batch.requests) != 4 or len(params) != 4:
            raise ProductionBackendError("native sampler requires one exact batch four")
        base = sampler.base_requests[int(batch.image_id)]
        rp = float(batch.requests[0].sampling["repetition_penalty"])
        scientific = tuple(
            dc_replace(
                base,
                request_id=request.request_id,
                generation_policy=GenerationPolicy(
                    max_new_tokens=512,
                    repetition_penalty=rp,
                    temperature=0.0,
                    top_p=1.0,
                    include_raw_model_logprob=False,
                ),
            )
            for request in batch.requests
        )
        native_requests = tuple(
            dc_replace(request, request_id=str(index))
            for index, request in enumerate(scientific)
        )
        prompts, media_hashes = sampler.session._generation_prompts(native_requests)
        try:
            native_outputs = sampler.session._engine.generate(
                prompts, list(params), use_tqdm=False
            )
        finally:
            _close_prompt_images(prompts)
        ordered = _restore_native_request_order(native_outputs, 4)
        request_receipts = []
        output_receipts = []
        for (
            planned,
            native_request,
            scientific_request,
            native,
            media_hash,
            param,
        ) in zip(
            batch.requests,
            native_requests,
            scientific,
            ordered,
            media_hashes,
            params,
            strict=True,
        ):
            result = sampler.session._materialize_result(
                request=native_request,
                native_output=native,
                executed_media_sha256=media_hash,
                raw_logprobs=None,
            )
            result = dc_replace(result, request_id=scientific_request.request_id)
            result.validate_for_request(scientific_request, sampler.session.receipt)
            request_receipt = NativeRequestReceipt(
                request_id=planned.request_id,
                seed=int(planned.seed),
                physical_batch_index=int(planned.physical_batch_index),
                request_order_in_batch=int(planned.request_order_in_batch),
                sampling_params=native_sampling_evidence(param),
                prompt_token_ids_sha256=token_ids_sha256(
                    result.executed_prompt_token_ids
                ),
                model_id=sampler.model_id,
                model_identity_sha256=sampler.model_identity_sha256,
                session_identity_sha256=sampler.session_identity_sha256,
            )
            generated = tuple(int(value) for value in result.generated_token_ids)
            logprobs = tuple(
                float(item.likelihood.policy_logprob) for item in result.token_trace
            )
            output_receipt = NativeOutputReceipt(
                native_request_receipt_sha256=request_receipt.content_sha256,
                request_id=planned.request_id,
                seed=int(planned.seed),
                physical_batch_index=int(planned.physical_batch_index),
                request_order_in_batch=int(planned.request_order_in_batch),
                prompt_token_ids=tuple(result.executed_prompt_token_ids),
                source_sha256=sampler.frozen.source_checkpoint_payload_sha256,
                manifest_sha256=sampler.frozen.manifest_sha256,
                model_id=sampler.model_id,
                tokenizer_id=sampler.tokenizer_id,
                processor_id=sampler.processor_id,
                processor_order=PROCESSOR_ORDER,
                sampler_backend_id=sampler.sampler_backend_id,
                generated_token_ids=generated,
                processed_logprobs=logprobs,
                terminal_kind=(
                    "natural_stop" if result.stop_reason == "im_end" else "cap_stop"
                ),
            )
            request_receipts.append(request_receipt)
            output_receipts.append(output_receipt)
        return NativeBatchReceipt(
            requests=tuple(request_receipts), outputs=tuple(output_receipts)
        )

    def close_sampler(self, sampler: _SamplerHandle) -> None:
        if sampler.closed:
            return
        sampler.closed = True
        sampler.session.close()

    # -- packed replay/compiler -----------------------------------------

    def open_parity_surface(
        self,
        frozen: Any,
        *,
        image_id: int,
        _assembly_backend: Any | None = None,
        _skeleton_builder: Any | None = None,
    ) -> _PackedHandle:
        """Open the one-image inference-only fp32/SDPA replay surface."""

        from scripts.research.human13_live_model import (
            assemble_human13_parity_model,
            build_human13_parity_skeleton,
        )

        if image_id != 1584:
            raise ProductionBackendError("parity replay is reserved for image 1584")
        if self._packed is not None and not self._packed.closed:
            raise ProductionBackendError("packed Source model is already open")
        plan = self._qualification_plan(
            frozen, learning_rate=frozen.default_qualification_learning_rate
        )
        assembly = assemble_human13_parity_model(
            plan,
            repo_root=REPO_ROOT,
            backend=_assembly_backend,
        )
        builder = _skeleton_builder or build_human13_parity_skeleton
        skeleton = builder(
            image_id=image_id,
            components=assembly.components,
            repo_root=REPO_ROOT,
        )
        if (
            getattr(skeleton, "prompt_token_count", 0) <= 0
            or hasattr(skeleton, "owner_row_tokens")
        ):
            raise ProductionBackendError(
                "parity skeleton must expose one prompt without owner rows"
            )
        handle = _PackedHandle(assembly, {image_id: skeleton}, None, frozen)
        self._packed = handle
        return handle

    def open_packed_surface(self, frozen: Any) -> _PackedHandle:
        from scripts.research.build_human13_k_union_manifest import load_manifest
        from scripts.research.human13_live_model import (
            assemble_human13_qualification_model,
            build_human13_processor_skeletons,
        )

        if self._packed is not None and not self._packed.closed:
            raise ProductionBackendError("packed Source model is already open")
        plan = self._qualification_plan(
            frozen, learning_rate=frozen.default_qualification_learning_rate
        )
        assembly = assemble_human13_qualification_model(
            plan, pack_count=1, repo_root=REPO_ROOT
        )
        manifest = load_manifest(frozen.manifest_path, require_full_panel=True)
        skeletons = build_human13_processor_skeletons(
            manifest, assembly.components, repo_root=REPO_ROOT
        )
        handle = _PackedHandle(assembly, skeletons, manifest, frozen)
        self._packed = handle
        return handle

    def packed_raw_logits(self, packed: _PackedHandle, execution: Any) -> Any:
        import torch

        from scripts.research.human13_rp_crossover_live_packs import (
            materialize_live_packs,
            plan_live_packs,
        )

        self._require_packed(packed)
        plan = plan_live_packs(
            execution=execution,
            skeleton=packed.skeletons[int(execution.plan.image_id)],
        )
        with torch.no_grad():
            materialized = materialize_live_packs(
                plan=plan,
                expected_vocab_size=len(packed.assembly.components.tokenizer),
                model=packed.assembly.model,
                runtime=packed.assembly.runtime,
                tokenizer=packed.assembly.components.tokenizer,
            )
        try:
            return materialized.packed_raw_logits
        finally:
            materialized.release()

    def admit_compiler_panel(
        self, packed: _PackedHandle, *, acquisition: Any, surface: Any
    ) -> Any:
        from scripts.research.human13_greedy_compiler import (
            SourceBoundaryInput,
            admit_source_compiler_panel,
            admit_source_forward_runtime,
            admit_source_greedy_decode,
        )

        self._require_packed(packed)
        runtime = admit_source_forward_runtime(
            replace(packed.assembly.components, model=packed.assembly.model),
            source_checkpoint_path=packed.frozen.source_checkpoint_path,
        )
        publications = acquisition.publications
        source_decodes = {int(decode.image_id): decode for decode in surface.decodes}
        decodes = []
        for manifest_image, frontier_image, _publication in zip(
            packed.manifest.images,
            surface.frontier.images,
            publications,
            strict=True,
        ):
            boundary = SourceBoundaryInput(
                image=frontier_image,
                prompt_token_ids=source_decodes[
                    int(manifest_image.image_id)
                ].prompt_token_ids,
                repetition_penalty=acquisition.training_repetition_penalty,
                image_sha256=manifest_image.image_sha256,
            )
            decodes.append(
                admit_source_greedy_decode(
                    boundary,
                    runtime=runtime,
                    manifest_sha256=packed.frozen.manifest_sha256,
                    prompt_skeleton=packed.skeletons[int(manifest_image.image_id)],
                )
            )
        return admit_source_compiler_panel(
            packed.manifest,
            surface.frontier,
            frontier_path=surface.frontier_path,
            acquisition=acquisition,
            source_decodes=tuple(decodes),
        )

    def tokenizer_adapter(self, *, manifest: Any, publication: Any) -> Any:
        from scripts.research.human13_trajectory_credit import (
            CanonicalTokenizerDecodeAdapter,
        )

        packed = self._packed
        if packed is None or packed.closed:
            raise ProductionBackendError(
                "tokenizer attestation requires the open packed Source runtime"
            )
        return CanonicalTokenizerDecodeAdapter.from_qwen_components(
            packed.assembly.components,
            manifest=manifest,
            publication=publication,
        )

    def close_packed_surface(self, packed: _PackedHandle) -> None:
        self._require_packed(packed)
        packed.closed = True
        packed.skeletons = {}
        packed.manifest = None
        packed.assembly = None
        self._packed = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass

    @staticmethod
    def _require_packed(packed: _PackedHandle) -> None:
        if not isinstance(packed, _PackedHandle) or packed.closed:
            raise ProductionBackendError("packed Source model is not live")

    # -- fresh cells/backward/checkpoint/audit --------------------------

    def open_cell(self, spec: Any, frozen: Any) -> Any:
        from scripts.research.human13_live_model import (
            assemble_human13_live_model,
            assemble_human13_qualification_model,
            bind_human13_selected_rp_crossover_plan,
            build_human13_processor_skeletons,
        )
        from scripts.research.human13_rp_crossover_runtime import CellExecutionState
        from scripts.research.human13_training_transaction import (
            TrainingStateTransaction,
            UpdateCounter,
        )
        from scripts.research.build_human13_k_union_manifest import load_manifest

        if self._active_state is not None:
            raise ProductionBackendError("a previous fresh cell is still live")
        plan = self._qualification_plan(
            frozen,
            learning_rate=spec.learning_rate,
            arm_id=spec.cell_key.arm_id,
        )
        if spec.cell_key.acquisition_key.phase == "qualification":
            if (
                spec.cell_key.arm_id != "C"
                or spec.global_learning_rate_decision_sha256 is not None
            ):
                raise ProductionBackendError(
                    "qualification model requires one provisional C dose"
                )
            assembly = assemble_human13_qualification_model(
                plan, pack_count=1, repo_root=REPO_ROOT
            )
        else:
            decision = spec.global_learning_rate_decision_sha256
            node_decisions = {
                cell.get("global_learning_rate_decision_sha256")
                for cell in self._node["cells"]
            }
            if decision is None or node_decisions != {decision}:
                raise ProductionBackendError(
                    "matrix model differs from the node's selected LR decision"
                )
            plan = bind_human13_selected_rp_crossover_plan(
                plan, decision_sha256=decision
            )
            assembly = assemble_human13_live_model(
                plan, pack_count=1, repo_root=REPO_ROOT
            )
        manifest = load_manifest(frozen.manifest_path, require_full_panel=True)
        skeletons = build_human13_processor_skeletons(
            manifest, assembly.components, repo_root=REPO_ROOT
        )
        named = tuple(
            (name, parameter)
            for name, parameter in assembly.model.named_parameters()
            if parameter.requires_grad
        )
        counter = UpdateCounter()
        transaction = TrainingStateTransaction(
            named,
            optimizer=assembly.optimizer,
            scheduler=assembly.scheduler,
            update_counter=counter,
            runtime=assembly.runtime,
            capture_cuda=True,
        )
        state = CellExecutionState(
            named_trainable_parameters=named,
            optimizer=assembly.optimizer,
            scheduler=assembly.scheduler,
            update_counter=counter,
            transaction=transaction,
            world_size=1,
            source_checkpoint_sha256=frozen.source_checkpoint_payload_sha256,
            shared_evidence_sha256=spec.shared_evidence.content_sha256,
            adamw_config_sha256=spec.adamw_config_sha256,
            optimizer_identity_sha256=spec.fresh_optimizer_identity_sha256,
            fresh_source=True,
        )
        object.__setattr__(state, "_human13_assembly", assembly)
        object.__setattr__(state, "_human13_skeletons", skeletons)
        object.__setattr__(state, "_human13_frozen", frozen)
        object.__setattr__(state, "_human13_witness_surface", None)
        self._active_state = state
        self._cell_metrics = {
            "decode_token_count": 0,
            "packed_token_count": 0,
            "logical_token_count": 0,
            "forward_count": 0,
            "row_bytes": 0,
            "artifact_bytes": 0,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass
        if spec.cell_key.arm_id == "C":
            try:
                surface = self.open_margin_surface(frozen)
            except BaseException:
                self.close_cell(state)
                raise
            object.__setattr__(state, "_human13_witness_surface", surface)
        return state

    def backward_objective(
        self, state: Any, spec: Any, evidence: Any, packed: Any
    ) -> Any:
        del packed
        from scripts.research.human13_greedy_compiler import (
            PackedCompilerLineage,
            admit_compiler_compact_logits_from_packed_plan,
            greedy_compiler_numerator,
        )
        from scripts.research.human13_rp_crossover_live_packs import (
            StreamingObjectiveStep,
            backward_incremental_objectives,
            stream_panel_live_packs,
        )
        from scripts.research.human13_rp_crossover_runtime import (
            ObjectiveBackwardReceipt,
        )
        from scripts.research.human13_trajectory_credit import (
            trajectory_score_function_numerator,
        )

        self._require_active_state(state)
        assembly = state._human13_assembly
        include_compiler = spec.cell_key.arm_id != "A"
        segments = (
            self._compiler_segments(
                evidence.compiler_ledger,
                self._source_surfaces[spec.cell_key.acquisition_key.training_rp],
            )
            if include_compiler
            else {}
        )
        stream = stream_panel_live_packs(
            acquisition=evidence.acquisition,
            skeletons=state._human13_skeletons,
            expected_vocab_size=len(assembly.components.tokenizer),
            credit_ledger=evidence.credit_ledger,
            compiler_ledger=(evidence.compiler_ledger if include_compiler else None),
            compiler_segments_by_image=segments,
            model=assembly.model,
            runtime=assembly.runtime,
            tokenizer=assembly.components.tokenizer,
        )
        lineage = PackedCompilerLineage(
            acquisition_sha256=evidence.acquisition.content_sha256,
            trajectory_credit_sha256=evidence.credit_ledger.content_sha256,
            repetition_penalty=spec.cell_key.acquisition_key.training_rp,
        )
        compiler_images = (
            {image.image_id: image for image in evidence.compiler_ledger.images}
            if include_compiler
            else {}
        )

        def steps() -> Any:
            for materialized in stream:
                trajectory = trajectory_score_function_numerator(
                    materialized.policy_logprobs,
                    evidence.credit_ledger,
                    token_indices=materialized.scored_token_indices,
                )
                rows = tuple(materialized.compiler_rows.values())
                compiler = None
                compiler_absent_reason = None
                if include_compiler and rows:
                    compact = admit_compiler_compact_logits_from_packed_plan(
                        materialized.plan.packed_plan,
                        evidence.compiler_ledger,
                        rows=rows,
                        lineage=lineage,
                    )
                    compiler = greedy_compiler_numerator(
                        compact, evidence.compiler_ledger
                    )
                elif include_compiler:
                    compiler_image = compiler_images.get(materialized.receipt.image_id)
                    if compiler_image is None:
                        raise ProductionBackendError(
                            "compiler ledger omitted a streamed image"
                        )
                    compiler_absent_reason = compiler_image.absent_reason
                receipt = materialized.receipt
                self._cell_metrics["packed_token_count"] = (
                    int(self._cell_metrics["packed_token_count"] or 0)
                    + receipt.packed_token_count
                )
                self._cell_metrics["logical_token_count"] = (
                    int(self._cell_metrics["logical_token_count"] or 0)
                    + receipt.logical_token_count
                )
                self._cell_metrics["forward_count"] = (
                    int(self._cell_metrics["forward_count"] or 0)
                    + receipt.forward_count
                )
                self._cell_metrics["row_bytes"] = (
                    int(self._cell_metrics["row_bytes"] or 0)
                    + receipt.compact_row_bytes
                )
                yield StreamingObjectiveStep(
                    image_id=receipt.image_id,
                    trajectory_numerator=trajectory,
                    compiler_numerator=compiler,
                    release=materialized.release,
                    compiler_absent_reason=compiler_absent_reason,
                )

        incremental = backward_incremental_objectives(
            steps(),
            trajectory_denominator=evidence.credit_ledger.logical_denominator,
            compiler_image_denominator=(
                evidence.credit_ledger.logical_image_count if include_compiler else None
            ),
            include_compiler=include_compiler,
            backward=assembly.accelerator.backward,
        )
        return ObjectiveBackwardReceipt(
            proposal_components=spec.expected_objective_components,
            component_hashes=spec.objective_component_hashes,
            objective_ledger_sha256=_sha256(
                {
                    "shared_evidence_sha256": spec.shared_evidence.content_sha256,
                    "components": [
                        list(item) for item in spec.objective_component_hashes
                    ],
                }
            ),
            shared_evidence_sha256=spec.shared_evidence.content_sha256,
            trajectory_credit_acquisition_sha256=(
                evidence.credit_ledger.acquisition_sha256
            ),
            compiler_ledger_sha256=evidence.compiler_ledger.content_sha256,
            image_ids=incremental.image_ids,
            backward_count=incremental.backward_count,
            optimizer_step_count=0,
            trajectory_denominator=incremental.trajectory_denominator,
            compiler_image_denominator=incremental.compiler_image_denominator,
            released_graph_count=incremental.released_graph_count,
        )

    @staticmethod
    def _compiler_segments(ledger: Any, surface: Any) -> Mapping[int, tuple[Any, ...]]:
        from scripts.research.human13_rp_crossover_live_packs import (
            CompilerSegmentRequest,
        )

        decodes = {decode.image_id: decode for decode in surface.decodes}
        result = {}
        for image in ledger.images:
            if image.site is None:
                result[image.image_id] = ()
                continue
            site = image.site
            decode = decodes[image.image_id]
            prefix = decode.generated_token_ids[: site.source_prefix_token_count]
            result[image.image_id] = (
                CompilerSegmentRequest(
                    site_id=site.site_id,
                    image_id=image.image_id,
                    segment_id=site.packed_segment_id,
                    token_ids=(*decode.prompt_token_ids, *prefix),
                    local_causal_position=site.local_causal_position,
                ),
            )
        return result

    def write_private_checkpoint(self, state: Any, spec: Any, stack: ExitStack) -> Any:
        from scripts.research.human13_live_eval import checkpoint_payload_sha256
        from scripts.research.human13_live_model import (
            build_human13_checkpoint_kwargs,
            build_human13_checkpoint_writer,
            readback_human13_checkpoint,
        )
        from scripts.research.human13_proposal_checkpoint import (
            private_proposal_checkpoint,
        )
        from scripts.research.human13_rp_crossover_runtime import PrivateCheckpointRef

        self._require_active_state(state)
        assembly = state._human13_assembly

        def writer(run_dir: Path) -> Path:
            checkpoint_writer = build_human13_checkpoint_writer(run_dir)
            written = checkpoint_writer.write_checkpoint(
                step=1,
                model=assembly.model,
                **build_human13_checkpoint_kwargs(assembly),
            )
            readback_human13_checkpoint(
                written.checkpoint_dir, expected_step=1, assembly=assembly
            )
            return written.checkpoint_dir

        checkpoint = stack.enter_context(
            private_proposal_checkpoint(
                Path(spec.output_root).expanduser().resolve() / "private",
                writer=writer,
            )
        )
        self._cell_metrics["artifact_bytes"] = int(
            self._cell_metrics["artifact_bytes"] or 0
        ) + sum(
            path.stat().st_size
            for path in Path(checkpoint).rglob("*")
            if path.is_file()
        )
        return PrivateCheckpointRef(
            path=str(checkpoint),
            checkpoint_sha256=checkpoint_payload_sha256(checkpoint),
            private=True,
        )

    def audit_checkpoint(
        self, spec: Any, checkpoint: Any, repetition_penalty: float
    ) -> Any:
        from scripts.research.human13_live_eval import (
            checkpoint_payload_sha256,
            evaluate_hf_checkpoint,
            hf_runtime_identity_from_outputs,
            write_outputs_jsonl,
        )
        from scripts.research.human13_rp_crossover_live_composition import AuditOutcome
        from scripts.research.human13_rp_crossover_matrix_contracts import AuditRef

        rp = float(repetition_penalty)
        if checkpoint_payload_sha256(checkpoint.path) != checkpoint.checkpoint_sha256:
            raise ProductionBackendError(
                "private checkpoint payload drifted before audit"
            )
        state = self._active_state
        if state is None:
            raise ProductionBackendError("audit requested without an active cell")
        assembly = state._human13_assembly
        from scripts.research.build_human13_k_union_manifest import load_manifest

        frozen = state._human13_frozen
        bound_manifest = load_manifest(
            frozen.manifest_path,
            require_full_panel=True,
        )
        outputs = evaluate_hf_checkpoint(
            manifest=bound_manifest,
            manifest_sha256=self._source_surfaces[rp].frontier.manifest_sha256,
            checkpoint_path=checkpoint.path,
            arm_id=spec.cell_key.arm_id,
            milestone=1,
            run_id=spec.cell_key.content_sha256,
            run_root=spec.output_root,
            resolved_arm_plan_sha256=(
                assembly.plan.resolved_plan_sha256 or spec.resolved_leaf_config_sha256
            ),
            resolved_config_sha256=spec.resolved_leaf_config_sha256,
            source_config_path=frozen.prompt_config_path,
            repetition_penalty=rp,
        )
        typed_outputs = cast(tuple[Mapping[str, Any], ...], outputs)
        runtime_identity = hf_runtime_identity_from_outputs(typed_outputs)
        tag = "rp100" if rp == 1.0 else "rp110"
        output_path = _regular_output(
            Path(spec.output_root).expanduser().resolve() / "audits" / f"{tag}.jsonl"
        )
        write_outputs_jsonl(output_path, typed_outputs)
        deltas = self._burden_delta(typed_outputs, self._source_outputs[rp])
        self._cell_metrics["decode_token_count"] = int(
            self._cell_metrics["decode_token_count"] or 0
        ) + sum(len(item["generated_token_ids"]) for item in typed_outputs)
        self._cell_metrics["artifact_bytes"] = (
            int(self._cell_metrics["artifact_bytes"] or 0) + output_path.stat().st_size
        )
        policy_sha = _sha256(
            {
                "schema_version": "human13_rp_crossover_audit_policy.v1",
                "repetition_penalty": rp,
                "decode_mode": "original_prompt_clean_greedy",
                "observed_runtime_identity": runtime_identity,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            }
        )
        audit = AuditRef(
            evaluation_rp=rp,
            evaluated_checkpoint_sha256=checkpoint.checkpoint_sha256,
            output_path=str(output_path),
            output_sha256=_sha256_file(output_path),
            row_count=len(typed_outputs),
            image_ids=tuple(int(item["image_id"]) for item in typed_outputs),
            generation_policy_receipt_sha256=policy_sha,
        )
        return AuditOutcome(
            audit=audit,
            malformed_delta=deltas["malformed"],
            cap_terminated_delta=deltas["cap_terminated"],
            unparseable_delta=deltas["unparseable"],
        )

    @staticmethod
    def _burden_rows(
        outputs: Sequence[Mapping[str, Any]],
    ) -> dict[int, tuple[int, bool, bool]]:
        image_ids = tuple(int(item["image_id"]) for item in outputs)
        if len(image_ids) != 13 or len(set(image_ids)) != 13:
            raise ProductionBackendError(
                "audit burden requires the exact 13-image panel"
            )
        rows = {}
        for image_id, item in zip(image_ids, outputs, strict=True):
            malformed = item["malformed_row_count"]
            if (
                isinstance(malformed, bool)
                or not isinstance(malformed, int)
                or malformed < 0
            ):
                raise ProductionBackendError(
                    "audit malformed-row burden must be nonnegative"
                )
            stop_reason = str(item["stop_reason"])
            if stop_reason not in _NATURAL_STOP_REASONS | _CAP_STOP_REASONS:
                raise ProductionBackendError("audit stop reason is not canonical")
            parser_status = str(item["parser_status"])
            if parser_status not in _PARSEABLE_STATUSES | _UNPARSEABLE_STATUSES:
                raise ProductionBackendError("audit parser status is not canonical")
            rows[image_id] = (
                malformed,
                stop_reason in _CAP_STOP_REASONS,
                parser_status in _UNPARSEABLE_STATUSES,
            )
        return rows

    @classmethod
    def _burdens(cls, outputs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        rows = tuple(cls._burden_rows(outputs).values())
        return {
            "malformed": sum(row[0] for row in rows),
            "cap_terminated": sum(row[1] for row in rows),
            "unparseable": sum(row[2] for row in rows),
        }

    @classmethod
    def _burden_delta(
        cls,
        proposal: Sequence[Mapping[str, Any]],
        source: Sequence[Mapping[str, Any]],
    ) -> dict[str, int]:
        proposal_rows = cls._burden_rows(proposal)
        source_rows = cls._burden_rows(source)
        if tuple(proposal_rows) != tuple(source_rows):
            raise ProductionBackendError(
                "proposal audit panel differs from its sealed Source surface"
            )
        pairs = tuple(
            (proposal_rows[image_id], source_rows[image_id])
            for image_id in proposal_rows
        )
        return {
            "malformed": sum(
                max(0, current[0] - baseline[0]) for current, baseline in pairs
            ),
            "cap_terminated": sum(
                current[1] and not baseline[1] for current, baseline in pairs
            ),
            "unparseable": sum(
                current[2] and not baseline[2] for current, baseline in pairs
            ),
        }

    def close_cell(self, state: Any) -> None:
        self._require_active_state(state)
        close_error: BaseException | None = None
        surface = getattr(state, "_human13_witness_surface", None)
        if surface is not None:
            try:
                self.close_margin_surface(surface)
            except BaseException as error:
                close_error = error
            finally:
                object.__setattr__(state, "_human13_witness_surface", None)
        try:
            import torch

            if torch.cuda.is_available():
                self._cell_metrics["cuda_peak_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated()
                )
                self._cell_metrics["cuda_peak_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved()
                )
        except (ImportError, RuntimeError):
            pass
        try:
            state.transaction.release()
        except BaseException as error:
            if close_error is None:
                close_error = error
        state.named_trainable_parameters = ()
        state.optimizer = None
        state.scheduler = None
        object.__setattr__(state, "_human13_skeletons", {})
        object.__setattr__(state, "_human13_witness_measurement", None)
        object.__setattr__(state, "_human13_witness_bank", None)
        object.__setattr__(state, "_human13_assembly", None)
        self._active_state = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass
        if close_error is not None:
            raise close_error

    def resource_snapshot(self) -> Any:
        from scripts.research.human13_rp_crossover_live_composition import (
            ResourceSnapshot,
        )

        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        try:
            import torch

            if torch.cuda.is_available():
                self._cell_metrics["cuda_peak_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated()
                )
                self._cell_metrics["cuda_peak_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved()
                )
        except (ImportError, RuntimeError):
            pass
        return ResourceSnapshot(
            measurement_scope="live",
            peak_host_rss_bytes=peak_rss,
            cuda_peak_allocated_bytes=self._cell_metrics.get(
                "cuda_peak_allocated_bytes"
            ),
            cuda_peak_reserved_bytes=self._cell_metrics.get("cuda_peak_reserved_bytes"),
            decode_token_count=int(self._cell_metrics.get("decode_token_count") or 0),
            packed_token_count=int(self._cell_metrics.get("packed_token_count") or 0),
            logical_token_count=int(self._cell_metrics.get("logical_token_count") or 0),
            forward_count=int(self._cell_metrics.get("forward_count") or 0),
            row_bytes=int(self._cell_metrics.get("row_bytes") or 0),
            artifact_bytes=int(self._cell_metrics.get("artifact_bytes") or 0),
        )

    def _require_active_state(self, state: Any) -> None:
        if state is not self._active_state:
            raise ProductionBackendError(
                "cell state is not the active fresh transaction"
            )

    @staticmethod
    def _qualification_plan(
        frozen: Any, *, learning_rate: float, arm_id: str = "C"
    ) -> Any:
        from scripts.research.human13_live_model import (
            RP_CROSSOVER_MILESTONES,
            RP_CROSSOVER_UNIT_ID,
            build_human13_live_model_plan,
        )
        from scripts.research.launch_human13_k_trajectory_rp_crossover import (
            load_leaf_config,
        )
        from scripts.research.materialize_human13_k_union_configs import (
            load_arm_config,
        )

        if learning_rate not in frozen.qualification_learning_rate_ray:
            raise ProductionBackendError("cell LR is outside the qualification ray")
        leaf = load_leaf_config(frozen.c_leaf_path)
        base = load_arm_config(_BASE_UPDATED_ARM)
        if base.optimizer is None:
            raise ProductionBackendError("base updated arm has no optimizer")
        config = replace(
            base,
            unit_id=RP_CROSSOVER_UNIT_ID,
            arm_id=arm_id,
            optimizer=replace(base.optimizer, learning_rate=learning_rate),
            milestones=RP_CROSSOVER_MILESTONES,
        )
        plan = build_human13_live_model_plan(config)
        if (
            leaf.arm_id != "C"
            or float(leaf.training_rp) not in {1.0, 1.10}
            or plan.arm_id != arm_id
            or plan.learning_rate != learning_rate
        ):
            raise ProductionBackendError("qualification leaf/model projection differs")
        return plan


__all__ = [
    "Human13RPCrossoverProductionBackend",
    "ProductionBackendError",
]
