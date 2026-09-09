"""Keep the actual Accelerate boundary independent of historical probe helpers."""

import subprocess
import sys
import textwrap


def test_real_accelerate_keeps_raw_optimizer_and_executes_one_step() -> None:
    # A fresh process avoids sharing Accelerate's singleton precision state
    # with unrelated tests in the full suite.
    program = textwrap.dedent(
        """
        import torch
        from accelerate import Accelerator
        from accelerate.optimizer import AcceleratedOptimizer
        from src.config.models import RuntimeBatchResolution, RuntimeConfig
        from src.runtime.train_runtime import TrainRuntime

        model = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
        original_parameters = tuple(model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)
        accelerator = Accelerator(cpu=True, mixed_precision="bf16")
        runtime = TrainRuntime(
            runtime_config=RuntimeConfig(seed=17),
            runtime_batch=RuntimeBatchResolution(
                world_size=1, effective_batch_size=1, resolved_grad_accum_steps=1,
            ),
            model=model, optimizer=optimizer, scheduler=None,
            expected_mixed_precision="bf16", accelerator=accelerator,
        )
        assert isinstance(runtime.optimizer, AcceleratedOptimizer)
        assert runtime.optimizer is not optimizer
        assert runtime.optimizer.optimizer is optimizer
        assert tuple(map(id, runtime.model.parameters())) == tuple(map(id, original_parameters))
        loss = runtime.model(torch.ones(1, 3, dtype=torch.bfloat16)).float().square().mean()
        runtime.backward(loss, planned_step_id=0)
        runtime.optimizer_step(planned_step_id=0)
        assert runtime.optimizer_step_count == 1
        assert all(state["step"].item() == 1 for state in optimizer.state.values())
        assert len(optimizer.state) == len(original_parameters)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
