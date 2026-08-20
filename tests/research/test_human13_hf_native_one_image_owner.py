from __future__ import annotations

import importlib.util
from dataclasses import replace
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast
import weakref

import pytest

from scripts.research.human13_greedy_compiler import (
    admit_hf_native_compiler_compact_logits,
    construct_hf_native_one_image_compiler_ledger,
)

from scripts.research.human13_hf_native_one_image_owner import (
    HFNativeAdmissionRequest,
    HFNativeCompilerAdmission,
    HFNativeOneImageAdmission,
    HFNativeOneImageOwnerError,
    HFNativeTrajectoryAdmission,
    PreAcquisitionSourceOwners,
    RepositoryHFNativeOneImageOwner,
    SourceOwnerRequest,
    build_repository_hf_native_owner,
    construct_hf_native_one_image_trajectory_ledger,
    hf_native_request_evidence_sha256,
)
from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    SampledHFGroup,
    admit_gradient_replay,
    admit_sampled_group,
)
from scripts.research.human13_one_image_services import (
    AdmittedTask5RuntimeEvidence,
    ExistingOwnersProductionBackend,
    ProductionAcquisition,
)
from scripts.research.run_human13_all_hf_shared_surface_vertical import (
    DualGPUResourceReceipt,
    EntryConfig,
    GPUResource,
)
from src.artifacts.json_values import json_sha256


_MANIFEST_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-12-human13-k-union-to-greedy-overfit-screen/manifest/"
    "human13-k-union-manifest.json"
)
_FRONTIER_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-14-human13-k-trajectory-rp-crossover-screen/"
    "vertical-dose-qualification-v4/rp100/qualification/c/acquisition/"
    "source/rp100/frontier.json"
)


def _shared_groups() -> tuple[
    tuple[SampledHFGroup, ...], tuple[GradientReplayGroup, ...]
]:
    fixture_path = Path(__file__).with_name("test_human13_hf_shared_surface.py")
    spec = importlib.util.spec_from_file_location("_hf_owner_fixture", fixture_path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    sampled: list[SampledHFGroup] = []
    replayed: list[GradientReplayGroup] = []
    for index, seeds in enumerate(fixture.plan_image1584_k16().seed_groups):
        requests = tuple(fixture.valid_request(seed) for seed in seeds)
        group = fixture.admit_sampled_group(
            **fixture.valid_group_kwargs(
                group_index=index,
                requests=requests,
                active_batch_steps=fixture.valid_active_batch_steps(requests),
            )
        )
        replay = fixture.admit_gradient_replay(
            sampled_group=group,
            replay_identity=group.identity,
            replayed_tokens=fixture.valid_replayed(group),
            replay_processor_order=("repetition_penalty", "temperature", "top_p"),
            causal_gathers=fixture.valid_gathers(group),
        )
        sampled.append(group)
        replayed.append(replay)
    return tuple(sampled), tuple(replayed)


def _source(*, session: object, model: object) -> PreAcquisitionSourceOwners:
    return PreAcquisitionSourceOwners(
        session_object_id=id(session),
        model_object_id=id(model),
        parameter_state_sha256="a" * 64,
        manifest_sha256="b" * 64,
        image_sha256="c" * 64,
        source_audit_sha256s=(
            (1.0, json_sha256({"rp": 1.0})),
            (1.1, json_sha256({"rp": 1.1})),
        ),
        compiler_source_context=object(),
        witness_bank=object(),
        realized_margin_probe=lambda: {"owner": 1.0},
        frozen_before_acquisition=True,
        sample_group_count_at_freeze=0,
        replay_group_count_at_freeze=0,
    )


def test_owner_rejects_late_witness_and_old_vllm_surrogate() -> None:
    with pytest.raises(HFNativeOneImageOwnerError, match="after acquisition"):
        PreAcquisitionSourceOwners(
            **{
                **_source(session=object(), model=object()).__dict__,
                "frozen_before_acquisition": False,
            }
        )
    with pytest.raises(HFNativeOneImageOwnerError, match="rejects old"):
        HFNativeAdmissionRequest(
            assembly=SimpleNamespace(),
            session=object(),
            manifest=object(),
            manifest_image=SimpleNamespace(image_id=1584),
            config=object(),
            sampled_groups=(object(),) * 4,
            replay_groups=(object(),) * 4,
            replay_logprob_tensors={"graph": object()},
        )


def test_public_owner_seam_preserves_exact_live_objects_and_phase_order() -> None:
    events: list[str] = []
    model = object()
    assembly = SimpleNamespace(model=model)
    session = object()
    manifest = object()
    image = SimpleNamespace(image_id=1584)
    config = object()
    source = _source(session=session, model=model)
    admitted_evidence = object()

    def prepare(request: SourceOwnerRequest) -> PreAcquisitionSourceOwners:
        events.append("prepare_source")
        assert request.assembly is assembly
        assert request.session is session
        assert tuple(request.source_audits) == (1.0, 1.1)
        return source

    def admit(
        request: HFNativeAdmissionRequest, frozen: PreAcquisitionSourceOwners
    ) -> HFNativeOneImageAdmission:
        events.append("admit_after_replay")
        assert frozen is source
        assert request.session is session
        assert request.assembly is assembly
        sampled_live = cast(tuple[SampledHFGroup, ...], request.sampled_groups)
        replay_live = cast(tuple[GradientReplayGroup, ...], request.replay_groups)
        return HFNativeOneImageAdmission(
            source_owner_sha256=frozen.content_sha256,
            session_object_id=id(session),
            model_object_id=id(model),
            sampled_group_sha256s=tuple(
                group.content_sha256 for group in sampled_live
            ),
            replay_group_sha256s=tuple(
                group.content_sha256 for group in replay_live
            ),
            runtime_evidence=admitted_evidence,
        )

    owner = RepositoryHFNativeOneImageOwner(
        prepare_source_owner=prepare,
        admit_hf_owners=admit,
    )
    prepared = owner.prepare_source(
        SourceOwnerRequest(
            assembly=assembly,
            session=session,
            manifest=manifest,
            manifest_image=image,
            config=config,
            source_audits={1.0: {"rp": 1.0}, 1.1: {"rp": 1.1}},
        )
    )
    events.append("sample_and_replay")
    sampled, replayed = _shared_groups()
    result = owner.admit_after_replay(
        HFNativeAdmissionRequest(
            assembly=assembly,
            session=session,
            manifest=manifest,
            manifest_image=image,
            config=config,
            sampled_groups=sampled,
            replay_groups=replayed,
            replay_logprob_tensors={"graph": object()},
        ),
        prepared,
    )

    assert result.runtime_evidence is admitted_evidence
    assert events == ["prepare_source", "sample_and_replay", "admit_after_replay"]


def test_public_backend_freezes_source_owner_before_sampling(tmp_path: Path) -> None:
    events: list[str] = []
    config = EntryConfig.from_yaml(
        Path(
            "configs/coordexp_swift/research/"
            "human13_all_hf_shared_surface_vertical/01_image1584.yaml"
        )
    )
    resources = DualGPUResourceReceipt(
        cards=(GPUResource(0, 80 << 30, 70 << 30), GPUResource(1, 80 << 30, 70 << 30))
    )
    sampled, replayed = _shared_groups()
    model = object()
    assembly = SimpleNamespace(components=object(), model=model, optimizer=object())

    class Session:
        _live_replay_tensors = {"graph": object()}
        sample_index = 0
        replay_index = 0

        def sample_group(self, _seeds: tuple[int, ...]) -> object:
            events.append("sample")
            result = sampled[self.sample_index]
            self.sample_index += 1
            return result

        def replay_group(self, group: object) -> object:
            events.append("replay")
            result = replayed[self.replay_index]
            assert result.sampled_group is group
            self.replay_index += 1
            return result

    session = Session()
    image = SimpleNamespace(image_id=1584, image_sha256="c" * 64)
    manifest = SimpleNamespace(binding=object(), images=(image,))
    evidence = AdmittedTask5RuntimeEvidence(
        trajectory_ledger=object(),
        compiler_ledger=object(),
        compiler_compact_logits=object(),
        witness_bank=object(),
        realized_margin_probe=lambda: {},
    )

    def prepare(request: SourceOwnerRequest) -> PreAcquisitionSourceOwners:
        events.append("prepare_source")
        assert request.session is session
        return PreAcquisitionSourceOwners(
            session_object_id=id(session),
            model_object_id=id(model),
            parameter_state_sha256="a" * 64,
            manifest_sha256=config.manifest_sha256 or "b" * 64,
            image_sha256=image.image_sha256,
            source_audit_sha256s=tuple(
                (rp, json_sha256(output))
                for rp, output in request.source_audits.items()
            ),
            compiler_source_context=object(),
            witness_bank=object(),
            realized_margin_probe=lambda: {},
            frozen_before_acquisition=True,
            sample_group_count_at_freeze=0,
            replay_group_count_at_freeze=0,
        )

    def admit(
        request: HFNativeAdmissionRequest, _source: PreAcquisitionSourceOwners
    ) -> HFNativeOneImageAdmission:
        events.append("admit_after_replay")
        assert request.sampled_groups == sampled
        assert request.replay_groups == replayed
        sampled_live = cast(tuple[SampledHFGroup, ...], request.sampled_groups)
        replay_live = cast(tuple[GradientReplayGroup, ...], request.replay_groups)
        return HFNativeOneImageAdmission(
            source_owner_sha256=_source.content_sha256,
            session_object_id=id(session),
            model_object_id=id(model),
            sampled_group_sha256s=tuple(
                group.content_sha256 for group in sampled_live
            ),
            replay_group_sha256s=tuple(
                group.content_sha256 for group in replay_live
            ),
            runtime_evidence=evidence,
        )

    owner = RepositoryHFNativeOneImageOwner(
        prepare_source_owner=prepare, admit_hf_owners=admit
    )

    def runtime_factory(**kwargs: Any) -> ProductionAcquisition:
        events.append("runtime_factory")
        proposal = SimpleNamespace(
            sampled_groups=tuple(kwargs["sampled_groups"]),
            replay_groups=tuple(kwargs["replay_groups"]),
            replay_logprob_tensors=kwargs["replay_logprob_tensors"],
        )
        return ProductionAcquisition(
            parity_passed=True,
            trusted_h_owner_ids=("h",),
            cuda_proposal_input=proposal,
            task2_resource_sha256="1" * 64,
            trajectory_ledger_sha256="2" * 64,
            compiler_ledger_sha256="3" * 64,
        )

    backend = ExistingOwnersProductionBackend(
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
        repo_root=Path.cwd(),
        source_config_path=tmp_path / "source.yaml",
        runtime_factory=runtime_factory,
        hf_native_owner=owner,
        assemble_model=lambda *_args, **_kwargs: assembly,
        build_skeletons=lambda *_args, **_kwargs: {1584: object()},
        open_surface=lambda *_args, **_kwargs: session,
        evaluate_checkpoint=lambda **kwargs: (
            {"image_id": 1584, "rp": kwargs["repetition_penalty"]},
        ),
        checkpoint_writer_factory=lambda _root: object(),
        checkpoint_readback=lambda *_args, **_kwargs: object(),
        checkpoint_hasher=lambda _path: "e" * 64,
    )
    training = backend.open_training(config, resources)
    audit = backend.open_audit(config, resources)
    backend.source_audit(audit, 1.0)
    backend.source_audit(audit, 1.1)
    backend.acquire_and_replay(training, config)

    assert events == [
        "prepare_source",
        "sample",
        "sample",
        "sample",
        "sample",
        "replay",
        "replay",
        "replay",
        "replay",
        "admit_after_replay",
        "runtime_factory",
    ]


def test_pre_acquisition_owner_failure_stops_before_k16(tmp_path: Path) -> None:
    config = EntryConfig.from_yaml(
        Path(
            "configs/coordexp_swift/research/"
            "human13_all_hf_shared_surface_vertical/01_image1584.yaml"
        )
    )
    resources = DualGPUResourceReceipt(
        cards=(
            GPUResource(0, 80 << 30, 70 << 30),
            GPUResource(1, 80 << 30, 70 << 30),
        )
    )
    model = object()
    assembly = SimpleNamespace(components=object(), model=model, optimizer=object())

    class Session:
        sample_calls = 0
        replay_calls = 0

        def sample_group(self, _seeds: tuple[int, ...]) -> object:
            self.sample_calls += 1
            raise AssertionError("K16 sampling must not start")

        def replay_group(self, _group: object) -> object:
            self.replay_calls += 1
            raise AssertionError("K16 replay must not start")

    session = Session()

    def reject_source(_request: SourceOwnerRequest) -> PreAcquisitionSourceOwners:
        raise HFNativeOneImageOwnerError(
            "typed pre-acquisition HOLD: compiler graph is absent"
        )

    owner = RepositoryHFNativeOneImageOwner(
        prepare_source_owner=reject_source,
        admit_hf_owners=lambda *_args: (_ for _ in ()).throw(AssertionError()),
    )
    image = SimpleNamespace(image_id=1584, image_sha256="c" * 64)
    backend = ExistingOwnersProductionBackend(
        manifest=SimpleNamespace(binding=object(), images=(image,)),
        manifest_path=tmp_path / "manifest.json",
        repo_root=Path.cwd(),
        source_config_path=tmp_path / "source.yaml",
        runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError()),
        hf_native_owner=owner,
        assemble_model=lambda *_args, **_kwargs: assembly,
        build_skeletons=lambda *_args, **_kwargs: {1584: object()},
        open_surface=lambda *_args, **_kwargs: session,
        evaluate_checkpoint=lambda **kwargs: (
            {"image_id": 1584, "rp": kwargs["repetition_penalty"]},
        ),
        checkpoint_writer_factory=lambda _root: object(),
        checkpoint_readback=lambda *_args, **_kwargs: object(),
        checkpoint_hasher=lambda _path: "e" * 64,
    )
    backend.open_training(config, resources)
    audit = backend.open_audit(config, resources)
    backend.source_audit(audit, 1.0)

    with pytest.raises(HFNativeOneImageOwnerError, match="pre-acquisition HOLD"):
        backend.source_audit(audit, 1.1)

    assert session.sample_calls == 0
    assert session.replay_calls == 0


def _complete_owner_fixture() -> Any:
    fixture_path = Path(__file__).with_name("test_human13_all_hf_vertical.py")
    spec = importlib.util.spec_from_file_location("_complete_owner_fixture", fixture_path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    return fixture._fixture()


def test_repository_builder_joins_exact_native_trajectory_and_compiler_graph() -> None:
    complete = _complete_owner_fixture()
    sampled = complete.sampled_groups
    replayed = complete.replay_groups
    identity = sampled[0].identity
    session = object()
    assembly = SimpleNamespace(model=complete.model, optimizer=complete.optimizer)
    manifest = SimpleNamespace()
    manifest_image = SimpleNamespace(image_id=1584, image_sha256=identity.image_sha256)
    boundary = SimpleNamespace(
        source_decode_sha256=complete.compiler_ledger.images[0].source_decode_sha256
    )
    source = PreAcquisitionSourceOwners(
        session_object_id=id(session),
        model_object_id=id(complete.model),
        parameter_state_sha256=identity.parameter_state_sha256,
        manifest_sha256=complete.trajectory_ledger.manifest_sha256,
        image_sha256=identity.image_sha256,
        source_audit_sha256s=(
            (1.0, json_sha256({"rp": 1.0})),
            (1.1, json_sha256({"rp": 1.1})),
        ),
        compiler_source_context=(boundary,),
        witness_bank=complete.witness_bank,
        realized_margin_probe=lambda: {"owner-g": 1.0},
        frozen_before_acquisition=True,
        sample_group_count_at_freeze=0,
        replay_group_count_at_freeze=0,
    )
    trajectory = HFNativeTrajectoryAdmission(
        source_owner_sha256=source.content_sha256,
        session_object_id=id(session),
        model_object_id=id(complete.model),
        manifest_object_id=id(manifest),
        manifest_sha256=source.manifest_sha256,
        image_id=1584,
        image_sha256=source.image_sha256,
        source_checkpoint_sha256=identity.checkpoint_payload_sha256,
        parameter_state_sha256=identity.parameter_state_sha256,
        sampled_group_sha256s=tuple(group.content_sha256 for group in sampled),
        replay_group_sha256s=tuple(group.content_sha256 for group in replayed),
        parser_projection_sha256=complete.trajectory_ledger.images[
            0
        ].parser_projection_sha256,
        trajectory_ledger=complete.trajectory_ledger,
    )
    compiler = HFNativeCompilerAdmission(
        source_owner_sha256=source.content_sha256,
        trajectory_admission_sha256=trajectory.content_sha256,
        session_object_id=id(session),
        model_object_id=id(complete.model),
        manifest_object_id=id(manifest),
        manifest_sha256=source.manifest_sha256,
        image_id=1584,
        image_sha256=source.image_sha256,
        source_checkpoint_sha256=identity.checkpoint_payload_sha256,
        source_boundary_sha256s=(boundary.source_decode_sha256,),
        compiler_ledger=complete.compiler_ledger,
        compiler_compact_logits=complete.compact_logits,
    )
    owner = build_repository_hf_native_owner(
        prepare_source_owner=lambda _request: source,
        admit_trajectory=lambda _request, _source: trajectory,
        admit_compiler=lambda _request, _source, _trajectory: compiler,
    )
    assert owner is not None
    request = HFNativeAdmissionRequest(
        assembly=assembly,
        session=session,
        manifest=manifest,
        manifest_image=manifest_image,
        config=SimpleNamespace(manifest_sha256=source.manifest_sha256),
        sampled_groups=sampled,
        replay_groups=replayed,
        replay_logprob_tensors=complete.replay_tensors,
    )

    admitted = owner.admit_after_replay(request, source)

    evidence = cast(AdmittedTask5RuntimeEvidence, admitted.runtime_evidence)
    assert evidence.trajectory_ledger is complete.trajectory_ledger
    assert evidence.compiler_ledger is complete.compiler_ledger
    assert evidence.compiler_compact_logits is complete.compact_logits
    assert evidence.witness_bank is complete.witness_bank


def test_repository_builder_registers_fail_closed_default_producers() -> None:
    assert build_repository_hf_native_owner() is not None


def test_native_admissions_reject_parser_or_compact_registry_substitution() -> None:
    complete = _complete_owner_fixture()
    sampled = complete.sampled_groups
    replayed = complete.replay_groups
    identity = sampled[0].identity
    trajectory = HFNativeTrajectoryAdmission(
        source_owner_sha256="a" * 64,
        session_object_id=1,
        model_object_id=id(complete.model),
        manifest_object_id=2,
        manifest_sha256=complete.trajectory_ledger.manifest_sha256,
        image_id=1584,
        image_sha256=identity.image_sha256,
        source_checkpoint_sha256=identity.checkpoint_payload_sha256,
        parameter_state_sha256=identity.parameter_state_sha256,
        sampled_group_sha256s=tuple(group.content_sha256 for group in sampled),
        replay_group_sha256s=tuple(group.content_sha256 for group in replayed),
        parser_projection_sha256=complete.trajectory_ledger.images[
            0
        ].parser_projection_sha256,
        trajectory_ledger=complete.trajectory_ledger,
    )
    with pytest.raises(HFNativeOneImageOwnerError, match="native parser"):
        replace(trajectory, parser_projection_sha256="f" * 64)

    boundary_sha256 = complete.compiler_ledger.images[0].source_decode_sha256
    detached = {
        site_id: tensor.detach()
        for site_id, tensor in complete.compact_logits._raw_logits.items()
    }
    object.__setattr__(complete.compact_logits, "_raw_logits", detached)
    with pytest.raises(
        HFNativeOneImageOwnerError,
        match="graph-bearing raw logits",
    ):
        HFNativeCompilerAdmission(
            source_owner_sha256="a" * 64,
            trajectory_admission_sha256=trajectory.content_sha256,
            session_object_id=1,
            model_object_id=id(complete.model),
            manifest_object_id=2,
            manifest_sha256=complete.trajectory_ledger.manifest_sha256,
            image_id=1584,
            image_sha256=identity.image_sha256,
            source_checkpoint_sha256=identity.checkpoint_payload_sha256,
            source_boundary_sha256s=(boundary_sha256,),
            compiler_ledger=complete.compiler_ledger,
            compiler_compact_logits=complete.compact_logits,
        )


def test_native_adapters_build_one_image_registry_ledgers_and_graph_compact() -> None:
    import torch

    from scripts.research.build_human13_k_union_manifest import load_manifest
    from scripts.research.build_human13_on_policy_frontier import (
        load_frontier_iteration,
    )
    from scripts.research.human13_greedy_compiler import SourceBoundaryInput
    from scripts.research.human13_trajectory_credit import (
        CanonicalTrajectoryProjection,
        _require_scientific_ledger_admission,
    )

    complete = _complete_owner_fixture()
    manifest = load_manifest(_MANIFEST_PATH, require_full_panel=True)
    image = next(item for item in manifest.images if item.image_id == 1584)
    projections = []
    for group in complete.replay_groups:
        for request in group.sampled_group.requests:
            generated = tuple(token.chosen_token_id for token in request.tokens)
            projections.append(
                CanonicalTrajectoryProjection(
                    request_id=request.request_id,
                    acquisition_trajectory_sha256=hf_native_request_evidence_sha256(
                        request, group
                    ),
                    generated_token_ids_sha256=json_sha256(
                        {"generated_token_ids": list(generated)}
                    ),
                    decoded_text_sha256=json_sha256({"decoded": ""}),
                    parse_status="accepted",
                    valid_prediction_count=0,
                    dropped_prediction_count=0,
                    canonical_predictions_json="[]",
                    canonical_drops_json="[]",
                    events=(),
                )
            )
    trajectory = construct_hf_native_one_image_trajectory_ledger(
        manifest=manifest,
        manifest_image=image,
        replay_groups=complete.replay_groups,
        canonical_projections=tuple(projections),
    )
    assert _require_scientific_ledger_admission(trajectory) is trajectory
    assert trajectory.logical_image_count == 1
    assert trajectory.logical_k == 16

    frontier = load_frontier_iteration(_FRONTIER_PATH)
    frontier_image = next(item for item in frontier.images if item.image_id == 1584)
    boundary = SourceBoundaryInput(
        image=frontier_image,
        prompt_token_ids=(1, 2),
        repetition_penalty=1.0,
        image_sha256=cast(str, image.image_sha256),
    )
    compiler = construct_hf_native_one_image_compiler_ledger(
        manifest, image, boundary, trajectory
    )
    site = compiler.images[0].site
    assert site is not None
    raw_logits = torch.zeros(
        (1, max(site.compact_token_ids) + 1), requires_grad=True
    )
    compact = admit_hf_native_compiler_compact_logits(
        compiler, boundary, raw_logits=raw_logits
    )
    assert compact._raw_logits[site.site_id].requires_grad
    assert compiler.logical_image_count == 1
    assert compiler.manifest_sha256 == trajectory.manifest_sha256


def test_default_native_callbacks_use_same_session_source_graph(tmp_path: Path) -> None:
    import torch

    from scripts.research.build_human13_k_union_manifest import load_manifest
    from scripts.research.build_human13_on_policy_frontier import (
        load_frontier_iteration,
    )
    from scripts.research.human13_greedy_compiler import SourceBoundaryInput
    from scripts.research.human13_hf_native_projection import (
        project_hf_native_replay_groups,
    )
    from scripts.research.human13_rp_crossover_witness import SealedSourceDecode
    from scripts.research.human13_trajectory_credit import (
        CanonicalTrajectoryProjection,
        _manifest_sha256,
    )

    complete = _complete_owner_fixture()
    manifest = load_manifest(_MANIFEST_PATH, require_full_panel=True)
    image = next(item for item in manifest.images if item.image_id == 1584)
    sampled: list[SampledHFGroup] = []
    replayed: list[GradientReplayGroup] = []
    replay_tensors: dict[str, object] = {}
    for old_sample, old_replay in zip(
        complete.sampled_groups, complete.replay_groups, strict=True
    ):
        identity = replace(old_sample.identity, image_sha256=image.image_sha256)
        sample = admit_sampled_group(
            plan=old_sample.plan,
            group_index=old_sample.group_index,
            expected_identity=identity,
            identity=identity,
            policy=old_sample.policy,
            requests=old_sample.requests,
            active_batch_steps=old_sample.active_batch_steps,
        )
        replay = admit_gradient_replay(
            sampled_group=sample,
            replay_identity=identity,
            replayed_tokens=old_replay.replayed_tokens,
            replay_processor_order=old_replay.replay_processor_order,
            causal_gathers=old_replay.causal_gathers,
        )
        sampled.append(sample)
        replayed.append(replay)
        replay_tensors[replay.content_sha256] = complete.replay_tensors[
            old_replay.content_sha256
        ]

    class EmptyCanonicalDecoder:
        tokenizer_sha256 = sampled[0].identity.tokenizer_sha256

        @staticmethod
        def decode(token_ids: tuple[int, ...]) -> tuple[str, tuple[str, ...]]:
            return "", tuple("" for _token in token_ids)

    projections = project_hf_native_replay_groups(
        replay_groups=tuple(replayed),
        manifest=manifest,
        manifest_image=image,
        attestation=cast(Any, EmptyCanonicalDecoder()),
    )
    assert all(type(item) is CanonicalTrajectoryProjection for item in projections)
    frontier = load_frontier_iteration(_FRONTIER_PATH)
    frontier_image = next(item for item in frontier.images if item.image_id == 1584)
    boundary = SourceBoundaryInput(
        image=frontier_image,
        prompt_token_ids=(1, 2),
        repetition_penalty=1.0,
        image_sha256=cast(str, image.image_sha256),
    )
    decode = SealedSourceDecode(
        image_id=1584,
        repetition_penalty=1.0,
        prompt_token_ids=(1, 2),
        generated_token_ids=tuple(frontier_image.generated_token_ids),
        owner_rows=(),
    )

    class Session:
        _live_replay_tensors: dict[str, object]
        calls = 0
        sample_index = 0
        replay_index = 0

        def raw_logit_rows(
            self, observed_decode: object, token_indices: tuple[int, ...]
        ) -> torch.Tensor:
            assert observed_decode is decode
            assert len(token_indices) == 1
            self.calls += 1
            return complete.model.weight[0] * torch.ones((1, 200_000))

        def canonical_replay_projections(
            self,
            observed_replays: object,
            observed_manifest: object,
            observed_image: object,
        ) -> tuple[object, ...]:
            assert observed_replays == tuple(replayed)
            assert observed_manifest is manifest
            assert observed_image is image
            return projections

        def sample_group(self, _seeds: tuple[int, ...]) -> SampledHFGroup:
            result = sampled[self.sample_index]
            self.sample_index += 1
            return result

        def replay_group(self, observed: object) -> GradientReplayGroup:
            result = replayed[self.replay_index]
            assert result.sampled_group is observed
            self.replay_index += 1
            return result

    session = Session()
    session._live_replay_tensors = replay_tensors
    frozen_compiler_row = session.raw_logit_rows(decode, (0,))
    identity = sampled[0].identity
    source = PreAcquisitionSourceOwners(
        session_object_id=id(session),
        model_object_id=id(complete.model),
        parameter_state_sha256=identity.parameter_state_sha256,
        manifest_sha256=_manifest_sha256(manifest),
        image_sha256=cast(str, image.image_sha256),
        source_audit_sha256s=(
            (1.0, json_sha256({"rp": 1.0})),
            (1.1, json_sha256({"rp": 1.1})),
        ),
        compiler_source_context=(boundary,),
        witness_bank=complete.witness_bank,
        realized_margin_probe=lambda: {"owner-g": 1.0},
        frozen_before_acquisition=True,
        sample_group_count_at_freeze=0,
        replay_group_count_at_freeze=0,
        source_decodes=(decode,),
        compiler_raw_logits=frozen_compiler_row,
    )
    owner = build_repository_hf_native_owner(
        prepare_source_owner=lambda _request: source,
    )
    assert owner is not None
    request = HFNativeAdmissionRequest(
        assembly=SimpleNamespace(
            components=object(),
            model=complete.model,
            optimizer=complete.optimizer,
        ),
        session=session,
        manifest=manifest,
        manifest_image=image,
        config=SimpleNamespace(manifest_sha256=_manifest_sha256(manifest)),
        sampled_groups=tuple(sampled),
        replay_groups=tuple(replayed),
        replay_logprob_tensors=replay_tensors,
    )

    admitted = owner.admit_after_replay(request, source)

    evidence = cast(AdmittedTask5RuntimeEvidence, admitted.runtime_evidence)
    assert session.calls == 1
    assert evidence.compiler_compact_logits is not None
    assert cast(Any, evidence.compiler_compact_logits)._raw_logits

    missing_projector_session = object()
    missing_source = replace(
        source,
        session_object_id=id(missing_projector_session),
    )
    with pytest.raises(
        HFNativeOneImageOwnerError,
        match="lacks the HF-native canonical projector",
    ):
        owner.admit_after_replay(
            replace(request, session=missing_projector_session),
            missing_source,
        )

    class FailingProjectionSession(Session):
        def canonical_replay_projections(
            self,
            observed_replays: object,
            observed_manifest: object,
            observed_image: object,
        ) -> tuple[object, ...]:
            del observed_replays, observed_manifest, observed_image
            raise ValueError("injected canonical replay projection failure")

    failing_projection_session = FailingProjectionSession()
    failing_projection_session._live_replay_tensors = replay_tensors
    failing_source = replace(
        source,
        session_object_id=id(failing_projection_session),
    )
    with pytest.raises(HFNativeOneImageOwnerError) as projection_failure:
        owner.admit_after_replay(
            replace(request, session=failing_projection_session),
            failing_source,
        )
    assert projection_failure.value.reason == (
        "canonical replay projection failed: "
        "ValueError: injected canonical replay projection failure"
    )
    assert projection_failure.value.disposition == "canonical_projection_failure"
    assert isinstance(projection_failure.value.__cause__, ValueError)

    config = EntryConfig.from_yaml(
        Path(
            "configs/coordexp_swift/research/"
            "human13_all_hf_shared_surface_vertical/01_image1584.yaml"
        )
    )
    assert config.manifest_sha256 == _manifest_sha256(manifest)
    resources = DualGPUResourceReceipt(
        cards=(
            GPUResource(0, 80 << 30, 70 << 30),
            GPUResource(1, 80 << 30, 70 << 30),
        )
    )
    runtime_evidence: list[AdmittedTask5RuntimeEvidence] = []

    def runtime_factory(**kwargs: Any) -> ProductionAcquisition:
        observed = cast(AdmittedTask5RuntimeEvidence, kwargs["runtime_evidence"])
        runtime_evidence.append(observed)
        proposal = SimpleNamespace(
            sampled_groups=tuple(kwargs["sampled_groups"]),
            replay_groups=tuple(kwargs["replay_groups"]),
            replay_logprob_tensors=kwargs["replay_logprob_tensors"],
        )
        return ProductionAcquisition(
            parity_passed=True,
            trusted_h_owner_ids=("owner-h",),
            cuda_proposal_input=proposal,
            task2_resource_sha256="1" * 64,
            trajectory_ledger_sha256=cast(
                Any, observed.trajectory_ledger
            ).content_sha256,
            compiler_ledger_sha256=cast(
                Any, observed.compiler_ledger
            ).content_sha256,
        )

    backend = ExistingOwnersProductionBackend(
        manifest=manifest,
        manifest_path=_MANIFEST_PATH,
        repo_root=Path.cwd(),
        source_config_path=tmp_path / "source.yaml",
        runtime_factory=runtime_factory,
        hf_native_owner=owner,
        assemble_model=lambda *_args, **_kwargs: request.assembly,
        build_skeletons=lambda *_args, **_kwargs: {1584: object()},
        open_surface=lambda *_args, **_kwargs: session,
        evaluate_checkpoint=lambda **kwargs: (
            {"image_id": 1584, "rp": kwargs["repetition_penalty"]},
        ),
        checkpoint_writer_factory=lambda _root: object(),
        checkpoint_readback=lambda *_args, **_kwargs: object(),
        checkpoint_hasher=lambda _path: "e" * 64,
    )
    training = backend.open_training(config, resources)
    audit = backend.open_audit(config, resources)
    backend.source_audit(audit, 1.0)
    backend.source_audit(audit, 1.1)
    backend.acquire_and_replay(training, config)

    assert len(runtime_evidence) == 1
    assert cast(Any, runtime_evidence[0].trajectory_ledger).logical_k == 16
    assert cast(Any, runtime_evidence[0].compiler_ledger).logical_image_count == 1
    assert runtime_evidence[0].compiler_compact_logits is not None
    assert session.calls == 1


def test_frozen_bridge_jacobians_and_post_apply_probe_survive_apply_rollback() -> None:
    from scripts.research import human13_hf_native_one_image_owner as owner_module
    from scripts.research.human13_cuda_cpu_adapter import CudaHFVerticalAdapter

    fixture_path = Path(__file__).with_name("test_human13_cuda_cpu_adapter.py")
    spec = importlib.util.spec_from_file_location("_bridge_apply_fixture", fixture_path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    surface, _ = fixture._task2_surface(module_name="_bridge_apply_surface")

    source_available = True
    source_provider = surface.witness_bank.jacobian_provider

    def guarded_source_provider(witness: object) -> object:
        assert source_available, "Source Jacobians were invoked after acquisition"
        return source_provider(witness)

    source_bank = replace(
        surface.witness_bank,
        jacobian_provider=guarded_source_provider,
    )
    frozen_bank = owner_module.freeze_witness_bank_for_post_acquisition(source_bank)
    source_available = False
    callback_states: list[tuple[str, str]] = []
    update_authorities: list[bool] = []
    original_margin_probe = surface.realized_margin_probe

    class Session:
        def named_trainable_parameters(
            self, *, allow_parameter_update: bool = False
        ) -> object:
            update_authorities.append(allow_parameter_update)
            if not allow_parameter_update:
                raise RuntimeError("applied proposal requires update authority")
            return surface.named_trainable_parameters

        def proposal_realized_margin_values(
            self,
            _source_decodes: object,
            _witness_sites: object,
            *,
            source_parameter_state_sha256: str,
            proposal_parameter_state_sha256: str,
        ) -> object:
            callback_states.append(
                (source_parameter_state_sha256, proposal_parameter_state_sha256)
            )
            return original_margin_probe()

    session = Session()
    realized_probe = owner_module.AdmittedPostApplyMarginProbe(
        session=session,
        source_parameter_state_sha256=surface.surface_identity.parameter_state_sha256,
        source_decodes=(),
        witness_sites=frozen_bank.constraints,
    )
    adapter = CudaHFVerticalAdapter(
        replace(
            surface,
            witness_bank=frozen_bank,
            realized_margin_probe=realized_probe,
        )
    )

    applied = adapter.apply_private_proposal()
    rollback = adapter.rollback_private_proposal()

    assert applied.update_count_after == 1
    assert rollback.rollback_decision == "rejected_restored"
    assert update_authorities == [True]
    assert len(callback_states) == 1
    assert callback_states[0][0] == surface.surface_identity.parameter_state_sha256
    assert callback_states[0][1]


@pytest.mark.parametrize("close_fails", [False, True])
def test_backend_close_releases_training_and_pre_acquisition_owner_graph(
    tmp_path: Path,
    close_fails: bool,
) -> None:
    config = EntryConfig.from_yaml(
        Path(
            "configs/coordexp_swift/research/"
            "human13_all_hf_shared_surface_vertical/01_image1584.yaml"
        )
    )
    resources = DualGPUResourceReceipt(
        cards=(
            GPUResource(0, 80 << 30, 70 << 30),
            GPUResource(1, 80 << 30, 70 << 30),
        )
    )
    model = object()
    assembly = SimpleNamespace(components=object(), model=model, optimizer=object())
    source_refs: list[weakref.ReferenceType[PreAcquisitionSourceOwners]] = []

    class Session:
        resource_receipt = object()

        def close(self) -> object:
            if close_fails:
                raise RuntimeError("injected close failure")
            return self.resource_receipt

    session = Session()
    session_ref = weakref.ref(session)
    session_queue = [session]

    def prepare(request: SourceOwnerRequest) -> PreAcquisitionSourceOwners:
        source = PreAcquisitionSourceOwners(
            session_object_id=id(request.session),
            model_object_id=id(model),
            parameter_state_sha256="a" * 64,
            manifest_sha256=config.manifest_sha256 or "b" * 64,
            image_sha256="c" * 64,
            source_audit_sha256s=tuple(
                (rp, json_sha256(output)) for rp, output in request.source_audits.items()
            ),
            compiler_source_context=object(),
            witness_bank=object(),
            realized_margin_probe=lambda: {"session": float(id(request.session))},
            frozen_before_acquisition=True,
            sample_group_count_at_freeze=0,
            replay_group_count_at_freeze=0,
        )
        source_refs.append(weakref.ref(source))
        return source

    owner = RepositoryHFNativeOneImageOwner(
        prepare_source_owner=prepare,
        admit_hf_owners=lambda *_args: (_ for _ in ()).throw(AssertionError()),
    )
    image = SimpleNamespace(image_id=1584, image_sha256="c" * 64)
    backend = ExistingOwnersProductionBackend(
        manifest=SimpleNamespace(binding=object(), images=(image,)),
        manifest_path=tmp_path / "manifest.json",
        repo_root=Path.cwd(),
        source_config_path=tmp_path / "source.yaml",
        runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError()),
        hf_native_owner=owner,
        assemble_model=lambda *_args, **_kwargs: assembly,
        build_skeletons=lambda *_args, **_kwargs: {1584: object()},
        open_surface=lambda *_args, **_kwargs: session_queue.pop(),
        evaluate_checkpoint=lambda **kwargs: (
            {"image_id": 1584, "rp": kwargs["repetition_penalty"]},
        ),
        checkpoint_writer_factory=lambda _root: object(),
        checkpoint_readback=lambda *_args, **_kwargs: object(),
        checkpoint_hasher=lambda _path: "e" * 64,
    )
    training = backend.open_training(config, resources)
    audit = backend.open_audit(config, resources)
    backend.source_audit(audit, 1.0)
    backend.source_audit(audit, 1.1)

    if close_fails:
        with pytest.raises(Exception, match="injected close failure"):
            backend.close_training(training)
    else:
        backend.close_training(training)

    assert backend._active_training_handle is None
    assert backend._pre_acquisition_source is None
    assert backend._source_owner_audits == {}
    del training, session
    gc.collect()
    assert session_ref() is None
    assert source_refs and source_refs[0]() is None
