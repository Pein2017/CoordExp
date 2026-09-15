"""CPU-only real frozen source regression; no model load or inference."""
import copy
from pathlib import Path

from probes.native_owner_scale import evaluation as e
from probes.source_rweak_row_cross.run import build_requests
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.native import prepare_native_inputs

root = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/evaluation')
packet = e.read(root / 'candidate-panel-bound-v10.json')
qwen = load_qwen_components_from_options(QwenLoadOptions(base_model=packet['config']['model']['base_model'], dtype='fp32', attn_implementation='sdpa', load_model=False))
qwen.processor.image_processor.do_resize = False
try:
    build_requests(qwen, packet['config'], [packet['records'][0]['case']])
except Exception as exc:
    assert 'data.image_missing' in str(exc), str(exc)
    red = str(exc)
else:
    raise AssertionError('legacy entry unexpectedly succeeded')
before = e.digest(packet)
media_ids = []
for index, frozen in enumerate(packet['records']):
    requests, _ = build_requests(qwen, packet['config'], [e._candidate_materialized_case(frozen['case'], packet['config'])])
    assert list(requests[0].expected_token_ids) == frozen['prompt_token_ids']
    if index < 8:
        batch = prepare_native_inputs(qwen.processor, requests, device='cpu', record_media_identity=True)
        plan = frozen['case']['image_plan']
        assert list(batch.prompt_token_ids[0]) == frozen['prompt_token_ids']
        assert batch.media_sha256[0] == plan['executed_media_sha256']
        assert list(batch.image_grids[0]) == plan['observed_image_grid_thw']
        media_ids.append(frozen['image_id'])
assert e.digest(packet) == before
e.publish(root / 'materialization-check-v11.json', {'status': 'passed', 'packet': e.binding(root / 'candidate-panel-bound-v10.json'), 'producer': e.binding(Path(e.__file__)), 'legacy_red': red, 'bound_request_prompt_and_file_bytes_checked': len(packet['records']), 'cpu_executed_media_grid_prompt_ids': media_ids, 'frozen_packet_unchanged': True, 'model_loads': 0, 'model_forwards': 0})
print('passed: 640 real requests; 8 CPU media projections; no model loads/forwards')
