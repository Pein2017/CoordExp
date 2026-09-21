from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import torch

from src.adapters import build_adapter_setup_plan, load_default_adapter_source_gate_evidence
from src.adapters.dora import inspect_dora_adapter_payload
from src.config.loader import load_train_config
from src.config.resolve import resolve_qwen_runtime_controls
from src.qwen import build_default_special_token_selection, load_qwen_components
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
    _validate_source_gate,
    inspect_special_token_embedding_delta_payload,
    load_default_special_token_embedding_source_gate_evidence,
)

repo = Path('/data/CoordExp/.worktrees/dora-prox-linear-n2').resolve()
config_path = repo / 'research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-sft256-dev128-baseline/qualification-v2.yaml'
source = Path('/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444').resolve()
retained = Path('/data/CoordExp/.worktrees/research-probes/outputs/probes/coordexp_swift').resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

resolved = load_train_config(config_path)
config = resolved.config
assert resolved.fingerprint == 'f12db0371d2039200fb4939d4ac49fd3d297d38885b55fedc0fcf234615f5066'
assert config.training.mode == 'supervised'
assert config.adapter.seed_mode == 'warm_start_expand_dora'
assert tuple(config.adapter.target_towers) == ('language',)
assert config.adapter.target_modules == 'all_linear'
assert config.model.special_token_embeddings.trainable is False
assert config.annotated_owner is None
assert config.rollout_calibration is None
assert config.model.attn_implementation == 'flash_attention_2'
assert config.model.fa2_branch_proof == 'first_micro_step'

components = load_qwen_components(config, load_model=False)
resolve_qwen_runtime_controls(
    config,
    tokenizer_vocab_size=components.token_identity.tokenizer_vocab_size,
    model_logits_dtype=config.training.precision,
)
selection = build_default_special_token_selection(
    config.model.special_token_embeddings,
    components.token_identity,
)
assert len(selection) == 1004

adapter_evidence = load_default_adapter_source_gate_evidence(repo)
adapter_plan = build_adapter_setup_plan(
    config.adapter,
    adapter_evidence,
    base_model_path=components.base_model_path,
)
assert adapter_plan.mode == 'warm_start_expand_dora'

special_evidence = load_default_special_token_embedding_source_gate_evidence(
    repo,
    selection,
)
_validate_source_gate(special_evidence, selection)

source_adapter = inspect_dora_adapter_payload(
    config.adapter.source_adapter_path,
    expected_base_model_path=components.base_model_path,
)
source_delta = inspect_special_token_embedding_delta_payload(
    config.adapter.repaired_embedding_payload_path,
    expected_base_model_path=components.base_model_path,
    expected_base_config_sha256=components.base_config_sha256,
    expected_tokenizer_sha256=components.tokenizer_sha256,
)
probe_adapter = inspect_dora_adapter_payload(
    retained / 'dora_roundtrip',
    expected_base_model_path=components.base_model_path,
)
probe_delta = inspect_special_token_embedding_delta_payload(
    retained / 'special_token_embeddings_roundtrip',
    expected_base_model_path=components.base_model_path,
    expected_base_config_sha256=components.base_config_sha256,
    expected_tokenizer_sha256=components.tokenizer_sha256,
)

runtime_strings = list(selection.token_strings)
runtime_ids = list(selection.token_ids)
for identity in (source_delta, probe_delta):
    semantic = identity['semantic_identity']
    assert semantic['semantics'] == SPECIAL_TOKEN_EMBEDDING_SEMANTICS
    assert semantic['tensor_key'] == DEFAULT_EMBED_DELTA_TENSOR_KEY
    assert semantic['tie_word_embeddings'] is True
    assert semantic['token_strings'] == runtime_strings
    assert semantic['token_ids'] == runtime_ids
    assert semantic['tensor_shape'][0] == 1004
assert source_delta['semantic_identity']['tensor_shape'] == [1004, 2048]
assert source_adapter['semantic_identity']['r'] == 16
assert source_adapter['semantic_identity']['lora_alpha'] == 32.0
assert source_adapter['semantic_identity']['lora_A_count'] == 196
assert source_adapter['semantic_identity']['lora_B_count'] == 196
assert source_adapter['semantic_identity']['lora_magnitude_vector_count'] == 196

manifest_path = source / 'training_state/manifest.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['metadata']['completed_step'] == 2444
assert Path(manifest['metadata']['base_model_path']).resolve() == components.base_model_path
assert manifest['metadata']['base_config_sha256'] == components.base_config_sha256
assert manifest['metadata']['tokenizer_sha256'] == components.tokenizer_sha256
for item in manifest['inference_files']:
    assert sha256(source / item['path']) == item['sha256']

live_versions = {
    'torch': torch.__version__,
    **{name: metadata.version(name) for name in ('transformers', 'peft', 'safetensors')},
}
dora_receipt_path = repo / 'outputs/probes/coordexp_swift/dora_roundtrip/receipt.json'
special_receipt_path = repo / 'outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json'
dora_receipt = json.loads(dora_receipt_path.read_text())
special_receipt = json.loads(special_receipt_path.read_text())
for receipt in (dora_receipt, special_receipt):
    for name, recorded in receipt['versions'].items():
        assert live_versions[name] == recorded, (name, live_versions[name], recorded)
assert metadata.version('flash_attn') == '2.8.3'

print(json.dumps({
    'status': 'NATIVE_CPU_ADMISSION_PASS',
    'qualification_config': str(config_path),
    'config_fingerprint': resolved.fingerprint,
    'runtime_shape': {
        'training_mode': config.training.mode,
        'adapter_seed_mode': adapter_plan.mode,
        'adapter_target': 'language/all_linear',
        'special_token_delta_trainable': config.model.special_token_embeddings.trainable,
        'annotated_owner': None,
        'rollout_calibration': None,
        'attention': config.model.attn_implementation,
        'fa2_branch_proof': config.model.fa2_branch_proof,
    },
    'selected_tokens': {
        'count': len(selection),
        'wrapper_count': 4,
        'coordinate_count': 1000,
        'selection_sha256': hashlib.sha256(json.dumps({
            'token_strings': runtime_strings,
            'token_ids': runtime_ids,
        }, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'source_and_probe_exact_match': True,
        'semantics': SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
        'tensor_key': DEFAULT_EMBED_DELTA_TENSOR_KEY,
        'frozen_by_config': True,
    },
    'artifacts': {
        'dora_probe_receipt': {'sha256': sha256(dora_receipt_path), 'payload_fingerprint': probe_adapter['fingerprint']},
        'special_embedding_probe_receipt': {'sha256': sha256(special_receipt_path), 'payload_fingerprint': probe_delta['fingerprint']},
        'source_adapter': {'tensor_sha256': sha256(source / 'adapter/adapter_model.safetensors'), 'payload_fingerprint': source_adapter['fingerprint'], 'tensor_counts': [196, 196, 196]},
        'source_embedding_delta': {'tensor_sha256': sha256(source / 'special_token_embeddings/special_token_embeddings.safetensors'), 'payload_fingerprint': source_delta['fingerprint'], 'shape': [1004, 2048]},
        'source_checkpoint_manifest': {'sha256': sha256(manifest_path), 'inference_files_verified': len(manifest['inference_files'])},
    },
    'versions': {**live_versions, 'flash_attn': metadata.version('flash_attn')},
}, sort_keys=True, indent=2))
