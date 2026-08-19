"""Public training facade for the V1 supervised smoke.

Design decision 1 of ``decompose-coordexp-swift-training-orchestration``: this
module is the compatibility facade and nothing else.  It builds the immutable
model-free plan, opens and binds the bounded rank control plane, initializes
the current run owner, admits the cache workflow, constructs exactly one
``TrainingSession``, returns its result, and closes resources in ``finally``.

Every implementation it used to own now has a narrower owner:
``execution_plan.py`` (frozen model-free decisions), ``control_plane.py`` (rank
convergence and close), ``cache_workflow.py`` (prepare/admit/hydrate), and
``session.py`` (model/runtime/trainer lifetime).  The facade imports them; they
never import the facade.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.training import cache_workflow, control_plane, execution_plan, session


def run_training_pipeline(
    config_path: str | Path,
    *,
    measurement_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = execution_plan.build_training_execution_plan(
        config_path,
        measurement_context=measurement_context,
    )
    rank_control_plane: control_plane.RankControlPlane | None = None
    try:
        rank_control_plane = control_plane.RankControlPlane.open(
            rank=plan.launch_rank,
            world_size=plan.launch_world_size,
        )
        runtime_determinism = cache_workflow._establish_converged_runtime_determinism(
            plan.resolved_config.config.runtime,
            rank=plan.launch_rank,
            world_size=plan.launch_world_size,
            rank_report_gatherer=rank_control_plane.gatherer,
            phase="pipeline_entry",
        )
        run_identity = session._initialize_model_free_run_owner(
            config=plan.resolved_config.config,
            resolved_config=plan.resolved_config,
            repo_root=plan.repo_root,
            launch_rank=plan.launch_rank,
            launch_world_size=plan.launch_world_size,
            preflight_gatherer=rank_control_plane.gatherer,
            measurement_context=plan.measurement_context,
            entry_started_at=plan.entry_started_at,
        )
    except BaseException:
        if rank_control_plane is not None:
            rank_control_plane.close()
        raise
    writer = run_identity.writer
    lifecycle = session.new_training_lifecycle(plan=plan, run_identity=run_identity)
    training_session: session.TrainingSession | None = None
    profile_sync_policy_configured = False
    try:
        pinned_runtime_baseline = session.admit_runtime_baseline_policies(
            plan=plan,
            rank_control_plane=rank_control_plane,
            run_identity=run_identity,
            lifecycle=lifecycle,
            runtime_determinism=runtime_determinism,
        )
        profile_sync_timings = session.apply_converged_profile_sync_policy(
            plan=plan,
            rank_control_plane=rank_control_plane,
        )
        profile_sync_policy_configured = True
        resolved_forward_input_provider = session.admit_forward_input_provider_policy(
            plan=plan,
            rank_control_plane=rank_control_plane,
            writer=writer,
            lifecycle=lifecycle,
            profile_sync_timings=profile_sync_timings,
        )
        cache_preflight = session.admit_training_cache_workflow(
            plan=plan,
            rank_control_plane=rank_control_plane,
            writer=writer,
            lifecycle=lifecycle,
        )
        rank_control_plane.close()
        accelerator = session.open_admitted_accelerator(
            plan=plan,
            writer=writer,
            lifecycle=lifecycle,
        )
        rank_control_plane.bind_accelerator(accelerator)
        session.admit_accelerator_runtime(
            plan=plan,
            rank_control_plane=rank_control_plane,
            writer=writer,
            lifecycle=lifecycle,
            accelerator=accelerator,
        )
        training_session = session.TrainingSession(
            plan=plan,
            control_plane=rank_control_plane,
            writer=writer,
            run_identity=run_identity,
            cache_preflight=cache_preflight,
            admitted_policies={
                "pinned_runtime_baseline": pinned_runtime_baseline,
                "profile_sync_timings": profile_sync_timings,
                "forward_input_provider": resolved_forward_input_provider,
            },
            lifecycle=lifecycle,
        )
        return training_session.run()
    except BaseException as error:
        if training_session is None:
            session.publish_training_entry_failure(
                plan=plan,
                writer=writer,
                lifecycle=lifecycle,
                error=error,
            )
        else:
            training_session.fail(error)
        raise
    finally:
        if training_session is not None:
            training_session.close()
        elif profile_sync_policy_configured:
            session.reset_profile_sync_timing_policies()
        rank_control_plane.close()


def prepare_training_pack_caches(
    config_path: str | Path,
    *,
    require_all_hit: bool = False,
) -> dict[str, Any]:
    """Compatibility entry point for the cache-workflow preparation command.

    `src/training/cache_workflow.py` is the canonical owner.  This facade entry
    is the one documented compatibility surface of this decomposition: the
    published `src.training.pipeline.prepare_training_pack_caches` name, its
    module identity, and its result are frozen by the Wave-0 compatibility
    ledger, so the facade keeps the name rather than re-exporting the owner's
    function object under a different `__module__`.
    """

    return cache_workflow.prepare_training_pack_caches(
        config_path,
        require_all_hit=require_all_hit,
    )
