import json

import pytest
import torch
from safetensors.torch import save_file

from probes.row_feedback import teacher


def test_loader_rejects_runtime_binding_or_non_normalized_tensor(tmp_path, monkeypatch):
    tensor_path = tmp_path / "one.safetensors"
    log_probs = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32), dim=-1).contiguous()
    save_file({"log_probs": log_probs}, str(tensor_path))
    source = tmp_path / "protection.json"
    source.write_text("{}")
    manifest = {
        "schema": teacher.SCHEMA, "status": "complete", "source_protection_records": {"path": str(source), "sha256": teacher.data.file_hash(source)},
        "native_n16_teacher": {"fingerprint": "n16"}, "normal_keys": ["r"],
        "records": [{"key": "r", "prompt_token_ids_sha256": "p", "action_ids_sha256": "a",
                     "kl_positions": [0], "kl_positions_sha256": teacher._positions_sha256([0]),
                     "visible_target_ordinals": [0], "log_probs": {"path": str(tensor_path), "sha256": teacher.data.file_hash(tensor_path),
                     "tensor_key": "log_probs", "dtype": "float32", "shape": [1, 3], "tensor_sha256": teacher._tensor_digest(log_probs)},
                     "normalization_max_abs_logsumexp": 0.0}],
        "denominators": {"records": 1, "protected_positions": 1},
    }
    manifest["manifest_sha256"] = teacher._json_digest(manifest)
    monkeypatch.setattr(teacher, "validate_manifest", lambda value: None)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    loaded = teacher.load_teacher_record(path, "r", expected_prompt_sha256="p", expected_action_sha256="a", expected_positions=[0])
    assert loaded.dtype == torch.float32 and loaded.device.type == "cpu" and loaded.is_contiguous()
    with pytest.raises(ValueError, match="runtime action binding"):
        teacher.load_teacher_record(path, "r", expected_prompt_sha256="p", expected_action_sha256="other", expected_positions=[0])
