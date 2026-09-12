from pathlib import Path
from types import SimpleNamespace

from probes.human13 import runtime
from src.config import fingerprint


def test_actual_runtime_identity_records_changed_input_dependency(monkeypatch):
    original = fingerprint.sha256_file
    before = runtime.input_source_hashes()
    monkeypatch.setattr(fingerprint, "sha256_file", lambda path: "changed-input-source" if Path(path).name == "inputs.py" else original(path))
    after = runtime.input_source_hashes()
    assert {key for key in before if before[key] != after[key]} == {"src/inference/inputs.py"}
    launch = SimpleNamespace(backend="hf", model_dtype="fp32", batch_size=1,
        backend_options={"hf": {"attn_implementation": "sdpa"}})
    receipt = SimpleNamespace(backend="hf", backend_mode="native", backend_version="fixture",
        effective_settings={"batch_size": 1, "observed_model_dtype": {"parameter_dtype_names": ["torch.float32"]},
            "observed_attn_implementation": "sdpa"},
        generation_config_fingerprint="unchanged", model_identity={}, tokenizer_identity={},
        processor_identity={}, validate_for_launch=lambda _: None)
    result = runtime.validate_hf_fp32_sdpa_batch_one(launch, receipt)
    assert result["executed_input_sources"] == after
    assert result["observed_attn_implementation"] == "sdpa"
