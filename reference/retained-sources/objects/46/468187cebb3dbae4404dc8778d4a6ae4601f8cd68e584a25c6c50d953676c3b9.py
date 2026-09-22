"""Frozen production untied loader reused with research-native inputs; no checkpoint edits."""
import importlib.util,json,sys
from pathlib import Path
import torch
from src.adapters.dora import attach_dora_adapter
from src.qwen.runtime_loading import QwenLoadOptions,load_qwen_components_from_options
from src.config.inference import InferConfig
from probes.training_set_completion.readout_norm_fresh import _binding,_tensor_hash
ROOT=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural')
TIED=Path('/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444')
UNTIED=Path('/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444')
BASE='/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent'
SOURCE=ROOT/'sources/infras_special_token_embeddings.py'
spec=importlib.util.spec_from_file_location('frozen_untied_payload',SOURCE);payload=importlib.util.module_from_spec(spec);sys.modules[spec.name]=payload;spec.loader.exec_module(payload)
def config_for(model_key):
 p=json.loads((ROOT/'panel.json').read_text());return InferConfig.model_validate(p['configs'][model_key])
def load_model(model_key,device):
 assert model_key in ['tied','untied'];c=config_for(model_key)
 q=load_qwen_components_from_options(QwenLoadOptions(base_model=c.model.base_model,dtype='fp32',attn_implementation='sdpa',patch_embed_linearization=c.backend.hf.patch_embed_linearization,load_model=True))
 adapter=attach_dora_adapter(q.model,adapter_path=c.adapter.path,base_model_path=q.base_model_path,adapter_name=c.adapter.name)
 embedding=payload.load_inference_embedding_delta(config=c,qwen=q)
 q.model.to(device).eval();e=q.model.get_input_embeddings();h=q.model.get_output_embeddings();shared=e.shared_embed_delta is h.shared_embed_delta
 assert shared==(model_key=='tied');assert h.bias is None
 for v in q.model.parameters():v.requires_grad_(False)
 ids=h.selected_token_ids;E=e(ids).detach();U=h.base.weight[ids].detach()+h.shared_embed_delta.detach()
 assert ids.numel()==1004 and torch.isfinite(E).all() and torch.isfinite(U).all()
 identity=dict(model=model_key,adapter=adapter,embedding=embedding,loader_source=_binding(SOURCE),shared_delta=shared,selected_ids_sha256=_tensor_hash(ids),input_rows_sha256=_tensor_hash(E),output_rows_sha256=_tensor_hash(U),input_delta_sha256=_tensor_hash(e.shared_embed_delta),output_delta_sha256=_tensor_hash(h.shared_embed_delta),dtype='fp32',attention='sdpa')
 return q,identity
