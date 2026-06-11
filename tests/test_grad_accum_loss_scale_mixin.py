import torch
import pytest


def test_grad_accum_loss_scale_mixin_does_not_scale_train_loss() -> None:
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
    assert loss.item() == 8.0


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


def test_grad_accum_loss_scale_mixin_validates_packed_batch_contract() -> None:
    from collections import defaultdict

    from src.metrics.dataset_metrics import GradAccumLossScaleMixin

    class DummyBase:
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            raise AssertionError("batch contract must fail before model forward")

    class DummyModel:
        training = True

    class DummyMetric:
        def update(self, value: float) -> None:
            pass

    class DummyTrainer(GradAccumLossScaleMixin, DummyBase):
        def __init__(self) -> None:
            self.model = DummyModel()
            self.model_accepts_loss_kwargs = True
            self.compute_loss_func = None
            self.current_gradient_accumulation_steps = 1
            self.args = type("Args", (), {"gradient_accumulation_steps": 1})()
            self.custom_metrics = {
                "train": defaultdict(DummyMetric),
                "eval": defaultdict(DummyMetric),
            }
            self.template = None

    bad_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3, -100, 5, 6]], dtype=torch.long),
        "text_position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
        "cu_seq_lens_q": torch.tensor([0, 2, 6], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2, 6], dtype=torch.int32),
        "max_length_q": 4,
        "max_length_k": 4,
        "pack_num_samples": torch.tensor([2], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="reset points mismatch|boundaries mismatch"):
        DummyTrainer().compute_loss(
            None,
            bad_inputs,
            return_outputs=False,
            num_items_in_batch=torch.tensor(4),
        )
