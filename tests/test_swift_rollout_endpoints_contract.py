from fastapi import FastAPI


def test_swift_rollout_server_registers_expected_endpoints() -> None:
    # CPU-only contract test against the local ms-swift install.
    from swift.llm.infer.rollout import SwiftRolloutDeploy

    deploy = SwiftRolloutDeploy.__new__(SwiftRolloutDeploy)
    deploy.app = FastAPI()

    deploy._register_rl_rollout_app()

    paths = {getattr(r, "path", None) for r in deploy.app.routes}

    # Launcher health/readiness polling.
    assert "/health/" in paths
    assert "/get_world_size/" in paths

    # Inference endpoint used for rollouts.
    assert "/infer/" in paths

    # Weight sync / communicator surface used by VLLMClient.
    assert "/init_communicator/" in paths
    assert "/close_communicator/" in paths
    assert "/update_named_param/" in paths
