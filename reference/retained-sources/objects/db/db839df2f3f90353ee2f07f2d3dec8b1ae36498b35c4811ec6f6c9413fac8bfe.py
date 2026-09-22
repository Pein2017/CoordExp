"""Load the explicit local judge profile without a runtime registry."""
from pathlib import Path

import yaml


def load_profile(path):
    data=yaml.safe_load(Path(path).read_text())
    if not isinstance(data,dict) or data.get('schema_version')!=1:
        raise ValueError('Expected judge profile schema_version=1')
    for key in ('name','model_path','dtype','system_prompt','user_prompt_template'):
        if not isinstance(data.get(key),str) or not data[key]:
            raise ValueError(f'Missing profile string: {key}')
    if data['dtype']!='bfloat16' or data.get('temperature')!=0:
        raise ValueError('This profile requires bfloat16 deterministic greedy inference')
    for key in ('max_model_len','max_new_tokens','min_pixels','max_pixels','max_num_seqs'):
        if type(data.get(key)) is not int or data[key]<=0:
            raise ValueError(f'Invalid positive integer: {key}')
    if not 0<data.get('gpu_memory_utilization',0)<1:
        raise ValueError('gpu_memory_utilization must be in (0,1)')
    if type(data.get('seed')) is not int or type(data.get('enforce_eager')) is not bool:
        raise ValueError('Explicit seed and enforce_eager are required')
    if not data['min_pixels']<=data['max_pixels'] or data['max_new_tokens']>=data['max_model_len']:
        raise ValueError('Inconsistent profile bounds')
    if '{category}' not in data['user_prompt_template']:
        raise ValueError('Question must name the candidate category')
    model=Path(data['model_path'])
    if not model.is_dir() or not (model/'config.json').is_file():
        raise ValueError('model_path must reference an existing local model')
    return data
