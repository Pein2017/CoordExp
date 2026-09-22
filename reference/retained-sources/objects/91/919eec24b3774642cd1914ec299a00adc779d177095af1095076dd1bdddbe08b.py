"""CPU-only input, source-gate, and frozen predecessor evidence acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probes.logit_lens import base, causal, radius, natural


def check(*, output_root: Path) -> dict:
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.inference.runtime import assemble_frontend
    from src.qwen.special_token_embeddings import load_default_special_token_embedding_source_gate_evidence

    output_root.mkdir(parents=True, exist_ok=False)
    resolved = load_research_infer_config(base.CONFIG)
    frontend = assemble_frontend(resolved.config, generation_config_fingerprint=sha256_json(
        resolved.config.generation.model_dump(mode='json')))
    base.require(frontend.qwen.model is None, 'CPU preflight unexpectedly loaded a model')
    gate_root, gate = base._stage_source_gate(output_root)
    gate_status = load_default_special_token_embedding_source_gate_evidence(gate_root)
    base.require(gate_status.source_study_passed and gate_status.roundtrip_probe_passed, 'source gate is not passed')
    bindings = {}
    for path, expected in (
        (base.SOURCE_ADAPTER/'adapter_model.safetensors', base.SOURCE_ADAPTER_SHA256),
        (base.OVERFIT_ADAPTER/'adapter_model.safetensors', base.OVERFIT_ADAPTER_SHA256),
        (base.SOURCE_DELTA/'special_token_embeddings.safetensors', base.SOURCE_DELTA_SHA256),
        (causal.PARENT_RECEIPT, causal.PARENT_RECEIPT_SHA256),
        (radius.STAGE_A_ACCEPTED, radius.STAGE_A_ACCEPTED_SHA256),
        (radius.STAGE_B_ACCEPTED, radius.STAGE_B_ACCEPTED_SHA256),
        (natural.RADIUS_RECEIPT, natural.RADIUS_RECEIPT_SHA256),
    ):
        actual = base.sha256_file(path)
        base.require(actual == expected, f'frozen binding changed: {path}')
        bindings[str(path)] = actual
    images = []
    for image_id in natural.IMAGE_IDS:
        _request, native, prompt, identity = causal.request_and_inputs_for_image(
            components=frontend.qwen, frontend=frontend, config=resolved.config,
            image_id=image_id, request_id=f'preflight-{image_id}',
        )
        prior = radius._load_prior(image_id)
        anchor = natural._load_anchor(image_id)
        base.require(identity == prior['baseline']['input'], f'fresh input differs from saved baseline: {image_id}')
        base.require(base.sha256_json(prompt) == identity['prompt_token_ids_sha256'], 'prompt identity mismatch')
        images.append({
            'image_id': image_id, 'input': identity,
            'prior_site_count': len(prior['sites']),
            'prior_current_anchor_count': len(prior['prior_current']),
            'trace_reduction': causal.reduce_trace([json.loads(line) for line in prior['paths']['trace'].read_text().splitlines()]),
            'natural_prefix_count': anchor['generated_prefix_count'],
            'natural_suffix_cap': anchor['suffix_cap'],
            'prior_artifact_hashes': prior['identities'],
            'natural_artifact_hashes': anchor['hashes'],
        })
        del native, prior, anchor
    report = {
        'schema_version': 'logit_lens_package_preflight.v1', 'status': 'passed',
        'scope': 'CPU input/predecessor evidence only; no model forwards',
        'model_load_count': 0, 'model_forward_count': 0,
        'producer_sources': {str(Path(module.__file__).resolve()): base.sha256_file(Path(module.__file__))
                             for module in (base, causal, radius, natural)},
        'config': resolved.to_artifact_dict(), 'source_gate': gate,
        'bindings': bindings, 'images': images,
    }
    base.atomic_json(output_root/'preflight.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    report = check(output_root=args.output_root)
    print(json.dumps({'status': report['status'], 'image_count': len(report['images']),
                      'output': str(args.output_root/'preflight.json')}))


if __name__ == '__main__':
    main()
