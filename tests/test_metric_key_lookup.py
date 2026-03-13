from __future__ import annotations

import types

from transformers.trainer_utils import SaveStrategy

from src.callbacks.save_delay_callback import SaveDelayCallback
from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer
from src.utils.metric_key_lookup import (
    metric_lookup_candidates,
    metric_name_matches_key,
    stage2_eval_metric_key,
)


def test_metric_lookup_candidates_bridge_stage2_eval_aliases() -> None:
    assert metric_lookup_candidates("detection/mAP") == (
        "detection/mAP",
        "eval_detection/mAP",
        "eval/detection/mAP",
    )
    assert metric_lookup_candidates("rollout/f1") == (
        "rollout/f1",
        "eval_rollout/f1",
        "eval/detection/f1",
    )
    assert metric_lookup_candidates("eval_detection/mAP") == (
        "eval_detection/mAP",
        "eval/detection/mAP",
    )


def test_metric_name_matches_stage2_map_target() -> None:
    metric_key = stage2_eval_metric_key("eval", "rollout/mAP")
    assert metric_key == "eval/detection/mAP"
    assert metric_name_matches_key("detection/mAP", metric_key)
    assert metric_name_matches_key("eval_detection/mAP", metric_key)
    assert metric_name_matches_key("eval/detection/mAP", metric_key)


def test_stage2_trainer_best_metric_resolves_detection_alias() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer.args = types.SimpleNamespace(
        metric_for_best_model="detection/mAP",
        greater_is_better=True,
        save_strategy=SaveStrategy.STEPS,
    )
    trainer.state = types.SimpleNamespace(
        best_metric=None,
        best_global_step=0,
        global_step=17,
    )

    metrics = {"eval/detection/mAP": 0.25}

    assert trainer._determine_best_metric(metrics, trial=None) is True
    assert trainer.state.best_metric == 0.25
    assert trainer.state.best_global_step == 17


def test_stage2_trainer_best_metric_resolves_rollout_alias() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer.args = types.SimpleNamespace(
        metric_for_best_model="rollout/f1",
        greater_is_better=True,
        save_strategy=SaveStrategy.STEPS,
    )
    trainer.state = types.SimpleNamespace(
        best_metric=None,
        best_global_step=0,
        global_step=23,
    )

    metrics = {"eval/detection/f1": 0.6}

    assert trainer._determine_best_metric(metrics, trial=None) is True
    assert trainer.state.best_metric == 0.6
    assert trainer.state.best_global_step == 23


def test_save_delay_callback_resolves_stage2_metric_alias() -> None:
    callback = SaveDelayCallback(save_delay_steps=10)
    args = types.SimpleNamespace(
        save_strategy=SaveStrategy.BEST,
        metric_for_best_model="detection/mAP",
        greater_is_better=True,
    )
    state = types.SimpleNamespace(
        global_step=3,
        epoch=0.0,
        best_metric=None,
        best_model_checkpoint="checkpoint-1",
        best_global_step=12,
    )
    control = types.SimpleNamespace(should_save=True)

    callback.on_evaluate(
        args,
        state,
        control,
        metrics={"eval/detection/mAP": 0.5},
    )

    assert control.should_save is False
    assert state.best_metric is not None
    assert state.best_metric > 0.5
    assert state.best_model_checkpoint is None
    assert state.best_global_step == 0
