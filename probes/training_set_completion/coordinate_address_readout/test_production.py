"""Production persistence and global allocation guards."""
import json
import time
import pytest
from probes.training_set_completion.coordinate_address_readout import runtime


def test_publication_is_create_only_and_budget_counts_outstanding_jobs(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    path=tmp_path/'value.json'
    runtime.write_once(path,{'value':1})
    with pytest.raises(FileExistsError): runtime.write_once(path,{'value':2})
    assert json.loads(path.read_text())=={'value':1}
    runtime.write_once(tmp_path/'wall-start.json',{'started_epoch':time.time()})
    runtime.write_once(tmp_path/'other/process.json',{'started_epoch':time.time()-114250,'allocated_gpus':1})
    ledger=runtime.RunLedger(tmp_path/'job','cuda:0',production=True)
    with pytest.raises(RuntimeError,match='allocated GPU'): ledger.check_budget()
    ledger.finish('test')
    runtime.write_once(tmp_path/'other/cost.json',{'terminal_epoch':time.time()-114245})
    second=runtime.RunLedger(tmp_path/'job2','cuda:0',production=True)
    second.check_budget()
    second.finish('test')
