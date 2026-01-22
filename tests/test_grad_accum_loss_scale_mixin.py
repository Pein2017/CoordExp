import torch


def test_grad_accum_loss_scale_mixin_scales_train_loss() -> None:
    from collections import defaultdict

    from src.metrics.dataset_metrics import GradAccumLossScaleMixin

    class DummyBase:
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss = torch.tensor(8.0)
            outputs = object()
            return (loss, outputs) if return_outputs else loss

    class DummyModel:
        def __init__(self, training: bool) -> None:
            self.training = training

    class DummyMetric:
        def __init__(self) -> None:
            self.last = None

        def update(self, value: float) -> None:
            self.last = float(value)

    class DummyTrainer(GradAccumLossScaleMixin, DummyBase):
        def __init__(self, training: bool, gas: int) -> None:
            self.model = DummyModel(training)
            self.model_accepts_loss_kwargs = True
            self.compute_loss_func = None
            self.current_gradient_accumulation_steps = int(gas)

            class Args:
                gradient_accumulation_steps = int(gas)

            self.args = Args()
            self.custom_metrics = {
                "train": defaultdict(DummyMetric),
                "eval": defaultdict(DummyMetric),
            }

            # ms-swift uses this for logging; ensure the helper exists.
            self._get_learning_rate = lambda: 1e-5  # noqa: E731

    trainer = DummyTrainer(training=True, gas=4)
    loss = trainer.compute_loss(None, {}, return_outputs=False, num_items_in_batch=123)
    assert torch.is_tensor(loss)
    assert loss.item() == 2.0


def test_grad_accum_loss_scale_mixin_does_not_scale_eval_loss() -> None:
    from collections import defaultdict

    from src.metrics.dataset_metrics import GradAccumLossScaleMixin

    class DummyBase:
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss = torch.tensor(8.0)
            outputs = object()
            return (loss, outputs) if return_outputs else loss

    class DummyModel:
        def __init__(self, training: bool) -> None:
            self.training = training

    class DummyMetric:
        def __init__(self) -> None:
            self.last = None

        def update(self, value: float) -> None:
            self.last = float(value)

    class DummyTrainer(GradAccumLossScaleMixin, DummyBase):
        def __init__(self, training: bool, gas: int) -> None:
            self.model = DummyModel(training)
            self.model_accepts_loss_kwargs = True
            self.compute_loss_func = None
            self.current_gradient_accumulation_steps = int(gas)

            class Args:
                gradient_accumulation_steps = int(gas)

            self.args = Args()
            self.custom_metrics = {
                "train": defaultdict(DummyMetric),
                "eval": defaultdict(DummyMetric),
            }
            self._get_learning_rate = lambda: 1e-5  # noqa: E731

    trainer = DummyTrainer(training=False, gas=4)
    loss = trainer.compute_loss(None, {}, return_outputs=False, num_items_in_batch=123)
    assert torch.is_tensor(loss)
    assert loss.item() == 8.0
