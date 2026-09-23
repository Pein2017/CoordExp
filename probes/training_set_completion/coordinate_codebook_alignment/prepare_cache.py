"""CPU-only admission through the ordinary packed-training cache owner."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from src.config.loader import load_train_config
from src.losses import build_token_vocabulary_groups
from src.qwen import load_qwen_components
from src.training.pipeline import _resolve_or_build_train_pack_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.parent
    os.environ['coordexp_infras_PACK_CACHE_ROOT'] = str(root / 'packing-cache')
    os.environ['coordexp_infras_PACK_CACHE_MATERIALIZATION_WORKERS'] = '4'
    receipts = []
    for path in args.config:
        resolved = load_train_config(path)
        components = load_qwen_components(resolved.config, load_model=False)
        assert components.model is None
        vocab = build_token_vocabulary_groups(components.token_identity, tokenizer=components.tokenizer)
        cache = _resolve_or_build_train_pack_cache(
            resolved.config, components, vocab, repo_root=Path(__file__).resolve().parents[3],
            accelerator=SimpleNamespace(is_main_process=True, num_processes=1), rank=0,
        )
        cache = {key: str(value) if isinstance(value, Path) else value for key, value in cache.items()}
        receipts.append({'config': str(path.resolve()), 'fingerprint': resolved.fingerprint, 'cache': cache})
    args.output.write_text(json.dumps({'model_calls': 0, 'receipts': receipts}, indent=2) + '\n')
    print(json.dumps({'model_calls': 0, 'output': str(args.output),
                      'pack_counts': [x['cache']['micro_step_count'] for x in receipts]}))


if __name__ == '__main__':
    main()
