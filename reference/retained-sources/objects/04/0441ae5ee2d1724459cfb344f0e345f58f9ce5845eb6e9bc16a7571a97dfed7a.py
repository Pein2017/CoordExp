import copy
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('expanded_train254',Path(__file__).with_name('run.py'))
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)


def test_exact254_complement_strata_and_roundrobin_no_outcome_filter():
    train=[368,7116]+list(range(1,255));support=list(range(1,49));dev=list(range(10000,10128))
    selected,outside,shards=run.partition(list(reversed(train)),support,dev)
    assert selected==list(range(1,255)) and outside==list(range(49,255))
    assert [len(s) for s in shards]==[32]*6+[31]*2
    assert shards==[selected[i::8] for i in range(8)]
    for tr,su,de in [(train[:-1],support,dev),(train[:-1]+[1],support,dev),(train,[368]+support[1:],dev),
                     (train,support[:-1]+[1],dev),(train,support,[1]+dev[1:])]:
        with pytest.raises(ValueError):run.partition(tr,su,de)


def test_all_eight_exit_bound_receipts_and_fixed_candidate_required():
    packet={'frozen':'packet'};ts=[dict(shard=i,gpu=i,status='completed',manifest_sha256=run.digest(packet),continuations=run.COUNTS[i],
        model_loads=1,score_forwards=0,new_tokens=100,elapsed_seconds=1.,training_receipt_sha256=run.RECEIPT_SHA) for i in range(8)]
    run.require_complete_shards(ts,packet)
    for change in [dict(status='running'),dict(continuations=31),dict(gpu=7),dict(score_forwards=1),dict(training_receipt_sha256='wrong'),
        dict(model_loads=2),dict(new_tokens=98689),dict(elapsed_seconds=3600),dict(manifest_sha256='wrong')]:
        bad=copy.deepcopy(ts);bad[0].update(change)
        with pytest.raises(ValueError):run.require_complete_shards(bad,packet)
    with pytest.raises(ValueError):run.require_complete_shards(ts[:7],packet)
    with pytest.raises(ValueError):run.require_complete_shards(ts[:7]+[ts[0]],packet)


def test_exact_shard_unique_ids():
    ids=list(range(254));packet=dict(selection={'shards':[{'image_ids':ids[i::8]} for i in range(8)]},records=[{'image_id':i,'example_id':f'e{i}'} for i in ids])
    assert len(run.shard_records(packet,7))==31
    packet['records'][8]['example_id']='e0'
    with pytest.raises(ValueError):run.shard_records(packet,0)


def test_native_consumer_rejects_forced_context_and_missing_eos():
    from probes.dora_owner_learning.tests.test_entrance_ce_eval import natural_fixture
    tok,frozen,row=natural_fixture();run.validate_natural(row,frozen,tok)
    for change in [dict(prefix_ids=[1]),dict(forced_ids=[1]),dict(action_ids=[1]),dict(remaining_budget=3085)]:
        with pytest.raises(ValueError):run.validate_natural(dict(row,**change),frozen,tok)
