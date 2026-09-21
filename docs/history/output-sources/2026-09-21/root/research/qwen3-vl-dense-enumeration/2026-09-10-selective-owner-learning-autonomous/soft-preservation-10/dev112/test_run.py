import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('locked_dev112',Path(__file__).with_name('run.py'))
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)


def test_exact_complement_numeric_round_robin_and_no_backfill():
    selected,shards=run.partition(list(reversed(range(128))),range(16),[1000,2000])
    assert selected==list(range(16,128))
    assert shards==[list(range(16+i,128,4)) for i in range(4)]
    assert all(len(s)==28 for s in shards)
    for dev,guard,train in [(list(range(127))+[0],range(16),[1000]),(range(128),[0]*16,[1000]),
                            (range(128),range(120,136),[1000]),(range(128),range(16),[20])]:
        with pytest.raises(ValueError):run.partition(dev,guard,train)


def test_merge_requires_all_complete_unique_correct_gpu_receipts():
    packet={'frozen':'identity'}
    terminals=[dict(shard=i,gpu=run.GPUS[i],status='completed',manifest_sha256=run.digest(packet),continuations=28,
        model_loads=1,score_forwards=0,new_tokens=100,elapsed_seconds=10.) for i in range(4)]
    run.require_complete_shards(terminals,packet)
    for changes in [dict(status='running'),dict(continuations=27),dict(gpu=7),dict(model_loads=2),
                    dict(score_forwards=1),dict(new_tokens=86353),dict(elapsed_seconds=3600),dict(manifest_sha256='wrong')]:
        wrong=[dict(t) for t in terminals];wrong[0].update(changes)
        with pytest.raises(ValueError):run.require_complete_shards(wrong,packet)
    with pytest.raises(ValueError):run.require_complete_shards(terminals[:3],packet)
    with pytest.raises(ValueError):run.require_complete_shards(terminals[:3]+[terminals[0]],packet)


def test_shard_selection_exact28_ids_and_invalid_index():
    packet=dict(selection={'shards':[{'image_ids':list(range(i,112,4))} for i in range(4)]},
        records=[dict(image_id=i,example_id=f'e{i}') for i in range(112)])
    assert [r['image_id'] for r in run.shard_records(packet,2)]==list(range(2,112,4))
    with pytest.raises(ValueError):run.shard_records(packet,4)
    packet['records'][0]['example_id']=packet['records'][4]['example_id']
    with pytest.raises(ValueError):run.shard_records(packet,0)
