import json
from pathlib import Path
import time

import pytest

from probes.training_set_completion.coordinate_codebook_alignment import execute
from probes.training_set_completion.coordinate_codebook_alignment.evaluation import _claim_queue
from probes.training_set_completion.coordinate_codebook_alignment.scale_execute import remaining, select_condition, run


def test_queue_parent_counterexample_and_fixed_isolation(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    assert _claim_queue(first, 3)[0] == 0
    assert _claim_queue(second, 3)[0] == 1  # Filename alone does not isolate claims.
    launch = {"evaluation": {}}
    for name in ("source", "epoch4", "epoch16", "epoch32"):
        parent = tmp_path / name
        parent.mkdir()
        queue = parent / "queue.json"
        queue.write_text(json.dumps({"specs": [{}, {}]}))
        assert _claim_queue(queue, 2)[0] == 0
        assert remaining(queue) == 1
        launch["evaluation"][name] = {"queue": str(queue), "checkpoint": "source" if name == "source" else str(parent / "checkpoint")}
    assert select_condition(launch, False) == "source"
    Path(launch["evaluation"]["epoch4"]["checkpoint"]).mkdir()
    assert select_condition(launch, False) == "epoch4"
    Path(launch["evaluation"]["epoch32"]["checkpoint"]).mkdir()
    assert select_condition(launch, True) == "epoch32"


def test_supervisor_new_root_cache_and_reserve(tmp_path, monkeypatch):
    captured = {}
    class Process:
        pid = 123456789
        def __init__(self, command, **kwargs):
            captured.update(kwargs)
        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(execute.subprocess, "Popen", Process)
    root = tmp_path / "new"
    cache = tmp_path / "accepted-cache"
    assert execute.run("cpu-fake", [0], ["fake"], root, cache_root=cache, reserve_seconds=900) == 0
    assert captured["env"]["coordexp_infras_PACK_CACHE_ROOT"] == str(cache)
    data = json.loads((root / "cost.json").read_text())
    assert data["jobs"][0]["state"] == "terminal"
    assert data["jobs"][0]["outstanding_reservation_gpu_seconds"] == 900
    data["wall_start"] = time.time() - 27901
    (root / "cost.json").write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="reserve"):
        execute.run("rejected", [1], ["fake"], root, reserve_seconds=900)


def test_continuation_does_not_replace_live_training_or_reset_clock(tmp_path):
    cost = {"wall_start": 123, "jobs": [{"name": "existing-fit", "gpus": [0, 1, 2, 3]}]}
    path = tmp_path / "cost.json"
    path.write_text(json.dumps(cost))
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"root": str(tmp_path)}))
    with pytest.raises(ValueError, match="training GPUs"):
        run(launch, continue_existing=True)
    assert json.loads(path.read_text()) == cost


def test_three_loss_condition_order_has_no_source_launch(tmp_path):
    names = ('ce_only_epoch16', 'three_loss_epoch8', 'three_loss_epoch16')
    launch = {'evaluation': {}, 'condition_order_during': list(names),
              'condition_order_after': [names[2], names[0], names[1]]}
    for name in names:
        p = tmp_path / name
        p.mkdir()
        q = p / 'queue.json'
        q.write_text(json.dumps({'specs': [{}]}))
        launch['evaluation'][name] = {'queue': str(q), 'checkpoint': str(p / 'checkpoint')}
    Path(launch['evaluation'][names[0]]['checkpoint']).mkdir()
    assert select_condition(launch, False) == names[0]
    Path(launch['evaluation'][names[2]]['checkpoint']).mkdir()
    assert select_condition(launch, True) == names[2]
    assert _claim_queue(Path(launch['evaluation'][names[0]]['queue']), 1)[0] == 0
    assert _claim_queue(Path(launch['evaluation'][names[2]]['queue']), 1)[0] == 0
    assert select_condition(launch, True) is None
