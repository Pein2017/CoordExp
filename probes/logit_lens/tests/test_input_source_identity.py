from pathlib import Path
from types import SimpleNamespace

from probes.logit_lens import base, runtime
from src.config import fingerprint


def test_fresh_logit_identity_records_changed_input_dependency(monkeypatch):
    original = fingerprint.sha256_file
    before = runtime.input_source_hashes()
    monkeypatch.setattr(fingerprint, "sha256_file", lambda path: "changed-input-source" if Path(path).name == "inputs.py" else original(path))
    after = runtime.input_source_hashes()
    assert {key for key in before if before[key] != after[key]} == {"src/inference/inputs.py"}
    known = {
        base.SOURCE_ADAPTER / "adapter_model.safetensors": base.SOURCE_ADAPTER_SHA256,
        base.OVERFIT_ADAPTER / "adapter_model.safetensors": base.OVERFIT_ADAPTER_SHA256,
        base.SOURCE_DELTA / "special_token_embeddings.safetensors": base.SOURCE_DELTA_SHA256,
    }
    monkeypatch.setattr(base, "sha256_file", lambda path: known.get(Path(path), "unchanged-profile-source"))
    receipt = base._identity_receipt(SimpleNamespace(to_artifact_dict=lambda: {}), source_gate={})
    assert receipt["executed_input_sources"] == after
    assert receipt["source_adapter"]["tensor_sha256"] == base.SOURCE_ADAPTER_SHA256
