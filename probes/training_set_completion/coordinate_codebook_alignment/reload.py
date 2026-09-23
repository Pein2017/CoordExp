"""Fresh-process checkpoint composition against independent training assembly."""
import argparse
import json
from pathlib import Path

import torch
from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_codebook_alignment import parity, evaluation
from src.adapters.dora import normalize_dora_state_key
from src.config.loader import load_train_config
from src.qwen.coordinate_codebook import load_coordinate_codebook


def run(args):
    original = load_train_config(args.config).config
    config = original.model_copy(update={'adapter': original.adapter.model_copy(update={
        'source_adapter_path': str(args.checkpoint / 'adapter'),
        'repaired_embedding_payload_path': str(args.checkpoint / 'special_token_embeddings')})})
    admission = json.loads(args.admission.read_text())
    result = {'status': 'running', 'bindings': [binding(args.config), binding(args.training_receipt),
              binding(args.admission), binding(args.dataset)], 'cases': []}
    counters, handles = [], []
    try:
        training, assembly = parity._load_expanded(config, torch.device('cuda:0'), require_new_targets=False)
        assert assembly['warm_start_surface']['device_initialization'] == []
        codebook_path = args.checkpoint / 'coordinate_codebook'
        if config.model.coordinate_codebook is None:
            assert not codebook_path.exists()
            assert getattr(training.model, 'coordinate_codebook', None) is None
        else:
            load_coordinate_codebook(training.model, codebook_path).enabled = True
        reloaded, identity = evaluation._load_runtime(str(args.checkpoint), 'cuda:0', admission)
        from src.adapters.dora import finalize_dora_initialization
        assert finalize_dora_initialization(training.model) == []
        assert finalize_dora_initialization(reloaded.model) == []
        result['repeated_finalization'] = 'no pending targets; no writes to restored tensors'
        for qwen in (training, reloaded):
            counter, hooks = evaluation._hook_counters(qwen.model)
            counters.append(counter); handles.extend(hooks)
        expected = json.loads(args.training_receipt.read_text())['trainable_parameter_hashes_after']
        expected = {normalize_dora_state_key(k, adapter_name='default'): v for k, v in expected.items()}
        observed = identity['parameter_hashes']
        assert expected == observed, {'missing': sorted(expected.keys()-observed.keys()),
                                      'extra': sorted(observed.keys()-expected.keys()),
                                      'changed': [k for k in expected.keys() & observed.keys() if expected[k] != observed[k]]}
        result['training_parameter_identity'] = 'exact'
        result['reload_identity'] = identity
        inference = parity._infer_config(config, args.dataset)
        for row in parity._read_rows(args.dataset):
            case = parity._plan_case(training, row, inference, args.dataset)
            item = parity._compare_case(training, reloaded, case, inference, args.dataset)
            result['cases'].append(item)
            assert item['max_abs_logit_difference_fp32'] <= parity.PARITY_TOLERANCE
            assert item['short_greedy_equal']
        native_rows = [row for row in evaluation._load_rows(args.dataset) if int(row['image_id']) == 13004]
        paths = evaluation.evaluate_cases(reloaded, admission=admission, dataset=args.dataset,
                    cases=native_rows, output_dir=args.output.parent,
                    output_path=args.output.with_name(args.output.stem+'-native.json'),
                    condition='qualification-live-reload', checkpoint=identity, include_teacher=True)
        assert all(json.loads(path.read_text())['status'] == 'complete' for path in paths)
        result['native_cells'] = [binding(path) for path in paths]
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        for handle in handles: handle.remove()
        result['counters'] = counters
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'checkpoint', 'training-receipt', 'admission', 'dataset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
