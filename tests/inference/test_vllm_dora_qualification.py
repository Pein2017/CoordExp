"""Actual technical entry/receipts, with only model and CUDA compute substituted."""
from __future__ import annotations

import copy
from contextlib import nullcontext
from datetime import timedelta
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import torch

from src.qwen.generation import ContinuationResult, ContinuationTrace
from src.qwen.native import NativeBatch, NativeRequest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT/'scripts/probes/coordexp_infras/vllm_dora_rollout.py'
CPU_OUTPUT = ROOT/'outputs/runtime-optimization/unify-resident-rollout-policy/qualification-cpu'
MEDIA = 151655
VOCAB = 152000


def _entry():
    spec = importlib.util.spec_from_file_location('qualification_entry', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _continuation(request_id, budget, sample):
    length = min(budget, 34 if sample else 20)
    tokens = [0, MEDIA, 100, 5] * ((length+3)//4)
    tokens = tokens[:length]
    reason = 'length'
    if length < budget:
        tokens[-1], reason = 9, 'im_end'
    raw = tuple(-1-j/100 for j in range(length))
    policy = tuple(x-.1 for x in raw)
    return ContinuationResult(request_id, tuple(tokens), reason,
                              ContinuationTrace(tuple(tokens), policy, raw))


class _Compute:
    """Synthetic tensors only: no model construction/loading or CUDA calls."""
    def __init__(self):
        self.language = torch.nn.Parameter(torch.tensor([.2]))
        self.input_delta = torch.nn.Parameter(torch.tensor([.1]))
        self.output_delta = torch.nn.Parameter(torch.tensor([.3]))
        self.config = SimpleNamespace(image_token_id=MEDIA, vision_config=SimpleNamespace(spatial_merge_size=2))
        self.forward_calls = 0

    def parameters(self):
        return (self.language, self.input_delta, self.output_delta)

    def named_parameters(self):
        return zip(('language', 'input_delta', 'output_delta'), self.parameters())

    def eval(self): return self
    def train(self): return self

    def get_rope_index(self, input_ids, mm_token_type_ids, **kwargs):
        assert bool((mm_token_type_ids[:, -32:] == 0).all()) or input_ids.shape[1] < 32
        return torch.arange(input_ids.shape[1])[None, None].expand(3, 1, -1), torch.zeros(1, 1)

    def get_placeholder_mask(self, ids, embeds, **kwargs):
        return (ids == MEDIA)[..., None], None

    def __call__(self, **inputs):
        self.forward_calls += 1
        ids = inputs['input_ids']
        prompt_length = ids.shape[1]-len(inputs['logits_to_keep'])
        image_mask, _ = self.get_placeholder_mask(ids, torch.zeros(*ids.shape, 1))
        assert not bool(image_mask[:, prompt_length:].any())
        assert inputs['logits_to_keep'].tolist() == list(range(prompt_length-1, ids.shape[1]-1))
        n = len(inputs['logits_to_keep'])
        gain = inputs['pixel_values'].reshape(-1)[0]
        scores = torch.stack((self.language*gain, self.input_delta*gain, self.output_delta*gain)).squeeze(-1)
        return SimpleNamespace(logits=torch.zeros(1, n, VOCAB).index_copy(
            -1, torch.tensor([5, MEDIA, 100]), scores.expand(1, n, -1)))


class _Norm:
    def __init__(self, model): self.model = model
    def generation_transform(self): return lambda ids, scores: scores
    def transform_replay(self, logits):
        selected = logits[..., 100:1100] * (1+self.model.output_delta)
        return logits.index_copy(-1, torch.arange(100, 1100), selected)


def _resident(connection):
    from src.qwen.vllm_rollout import _process_identity
    connection.send(_process_identity())
    while True:
        operation, value = connection.recv()
        if operation == 'close':
            break
        if operation == 'generate':
            requests, budgets, sample = value
            connection.send(tuple(_continuation(request, budget, sample)
                for request, budget in zip(requests, budgets, strict=True)))
        else:
            connection.send(value)
    connection.close()


class _Resident:
    def __init__(self, *, identity, device, trainer_rank, log_path, **kwargs):
        from src.qwen.vllm_rollout import _process_identity
        self.identity, self.receipts = identity, []
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_resident, args=(child,))
        self.process.start(); child.close()
        child_identity = self.connection.recv()
        physical = dict(uuid='GPU-'+str(UUID(int=trainer_rank+1)), pci_domain_id=0,
                        pci_bus_id=100+trainer_rank, pci_device_id=0)
        self.device_request = dict(schema='coordexp-vllm-device-1', rank=trainer_rank, device=device,
            physical_token=str(device), parent=dict(_process_identity(), visibility=None, physical=physical))
        self.startup = dict(identity=identity, startup_seconds=0,
            device=dict(requested=self.device_request, child=child_identity, inherited_visibility=str(device),
                        effective_visibility=str(device), logical_device=0, physical=physical))
        Path(log_path).write_text('CPU model/device substitution; real resident child ownership\n')

    def configure_coordinate_output_norm(self, mode, ids, *, identity):
        return dict(mode=mode, identity=identity)

    def generate(self, requests, *, budgets, policy, seeds, identity, **kwargs):
        assert identity == self.identity and not policy.use_model_defaults and kwargs['trace']
        assert kwargs['allow_pad_tokens']
        if policy.temperature:
            assert all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
        self.connection.send(('generate', ([r.request_id for r in requests], budgets, policy.temperature == 1)))
        results = self.connection.recv()
        self.receipts.append(dict(operation='generate', identity=identity, seconds=0,
            coordinate_output_norm=dict(mode='median', identity=identity),
            paired_trace=dict(snapshot_id=identity, begin_seconds=0, finalize_seconds=0,
                requests=[dict(request_id=r.request_id, emitted_actions=len(r.token_ids),
                    excluded_async_suffix=0, discarded_prefill_actions=0, dropped_budget_actions=0) for r in results])))
        return results

    def refresh(self, model, delta, *, identity):
        self.connection.send(('refresh', identity))
        self.identity = self.connection.recv()
        self.receipts.append(dict(operation='refresh', identity=identity, seconds=0,
            snapshot_materialization_seconds=0, adapter_bytes=4, embedding_bytes=8,
            coordinate_output_norm=dict(mode='median', identity=identity)))

    def close(self):
        self.connection.send(('close', None)); self.connection.close()
        self.process.join(5)
        if self.process.is_alive():
            self.process.terminate(); self.process.join(5)
        self.shutdown = dict(pid=self.process.pid, exitcode=self.process.exitcode,
                             settled=not self.process.is_alive(), terminated=False, seconds=0)


def _runtime(args, items, rank, local_rank, world, receipt):
    import torch.distributed as dist
    from src.qwen.vllm_rollout import validate_device_assignments
    dist.init_process_group('gloo', init_method=os.environ['QUALIFICATION_RENDEZVOUS'],
                            rank=rank, world_size=world, timeout=timedelta(seconds=30))
    model = _Compute()
    q = SimpleNamespace(model=model, base_model_path='CPU-model-compute-substitution',
        tokenizer=SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda text: 9))
    receipt['cpu_substitution'] = 'model/device compute only; real validated inputs, entry, Gloo SUM and children'
    batches = [NativeBatch(dict(input_ids=torch.tensor([row['prompt_token_ids']]),
        image_grid_thw=torch.tensor([row['image_grid_thw']]),
        mm_token_type_ids=(torch.tensor([row['prompt_token_ids']]) == MEDIA).long(),
        pixel_values=torch.tensor([float((1584, 2299, 2685).index(row['image_id'])+1)])),
        (row['request_id'],)) for row in items]
    return SimpleNamespace(q=q, delta=SimpleNamespace(delta_tensors=lambda:
        {'input': model.input_delta, 'output': model.output_delta}), norm=_Norm(model),
        coordinate_ids=tuple(range(100, 1100)), batches=batches,
        requests=[NativeRequest(row['request_id'], 'fake encoding', row['image_path']) for row in items],
        dist=dist, sync=lambda: None, autocast=nullcontext, engine_factory=_Resident,
        validate_devices=validate_device_assignments, verify_source=lambda: None,
        peak_memory=lambda: dict(cpu_model_forward_calls=model.forward_calls,
            final_parameters=[float(p.detach()) for p in model.parameters()]))


def _rank_main(rank, root, rendezvous, fault):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      QUALIFICATION_RENDEZVOUS='file://'+str(rendezvous))
    module = _entry()
    import src.qwen.generation as generation
    generation.generate_continuations = lambda model, batch, *, budgets, policy, **kwargs: (
        _continuation(batch.request_ids[0], budgets[0], policy.temperature == 1),)
    if fault:
        original = _Resident.generate
        def broken(self, *args, **kwargs):
            results = original(self, *args, **kwargs)
            return tuple(ContinuationResult(r.request_id, r.token_ids, r.stop_reason,
                ContinuationTrace(r.token_ids, (), r.raw_logprobs)) for r in results)
        _Resident.generate = broken
    module.main(['--qualification', '--checkpoint', str(module.ANCHOR), '--output', str(root)],
                runtime_factory=_runtime)


def _invoke(fault=False):
    CPU_OUTPUT.mkdir(parents=True, exist_ok=True)
    root = CPU_OUTPUT/(('entry-failure-' if fault else 'entry-success-')+uuid4().hex)
    rendezvous = CPU_OUTPUT/('gloo-'+uuid4().hex)
    context = mp.get_context('spawn')
    processes = [context.Process(target=_rank_main, args=(rank, root, rendezvous, fault)) for rank in range(2)]
    for process in processes: process.start()
    for process in processes:
        process.join(45)
        if process.is_alive():
            process.terminate(); process.join(5)
            pytest.fail('CPU entry exceeded its bounded observation deadline')
    receipts = [json.loads((root/f'rank-{rank}/receipt.json').read_text()) for rank in range(2)]
    return root, receipts, [process.exitcode for process in processes]


@pytest.fixture(scope='module')
def qualified():
    root, receipts, exits = _invoke()
    assert exits == [0, 0]
    return root, receipts


def test_entry_uneven_collectives_single_graph_update_and_terminal_consumer(qualified):
    root, receipts = qualified
    summary = json.loads((root/'receipt.json').read_text())
    assert summary['generation_requests'] == 21 and summary['replay_target_actions'] == 96
    assert summary['replay_forwards'] == summary['replay_backwards'] == 3
    assert [r['image_ids'] for r in receipts] == [[1584, 2685], [2299]]
    assert [r['peak_memory']['cpu_model_forward_calls'] for r in receipts] == [2, 1]
    assert all(r['distributed_settlement'] == 'destroyed' for r in receipts)
    assert all(not (Path('/proc')/str(r['child_settlement']['pid'])).exists() for r in receipts)
    # Independent single-process image mean catches /local_count even if a receipt claims /3.
    model = _Compute(); norm = _Norm(model)
    optimizer = torch.optim.AdamW([dict(params=[model.language], lr=1e-5),
        dict(params=[model.input_delta, model.output_delta], lr=5e-6)],
        betas=(.9, .999), eps=1e-8, weight_decay=0)
    for gain in (1., 2., 3.):
        target = torch.tensor(_continuation('reference', 64, True).token_ids[:32])
        scores = torch.stack((model.language*gain, model.input_delta*gain, model.output_delta*gain)).squeeze(-1)
        logits = torch.zeros(1, 32, VOCAB).index_copy(-1, torch.tensor([5, MEDIA, 100]), scores.expand(1, 32, -1))
        loss = -norm.transform_replay(logits)[0].log_softmax(-1).gather(-1, target[:, None]).mean()
        (loss/3).backward()
    total_norm = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1))
    optimizer.step()
    assert receipts[0]['learning_step']['total_norm'] == pytest.approx(total_norm, rel=1e-6)
    assert receipts[0]['peak_memory']['final_parameters'] == pytest.approx([float(p.detach()) for p in model.parameters()])
    assert receipts[1]['peak_memory']['final_parameters'] == receipts[0]['peak_memory']['final_parameters']
    assert summary['backend_acquisition_totals']['vllm']['critical_path_seconds'] == max(
        sum(stage['parent_seconds'] for name, stage in r['acquisitions'].items()
            if name.startswith('vllm') and 'restore' not in name) for r in receipts)


@pytest.mark.parametrize('fault', ['missing_rank', 'reordered_rank', 'missing_image', 'reordered_request',
    'wrong_scale', 'wrong_sequence', 'stale_snapshot', 'missing_channel', 'duplicate_channels', 'unsettled_child'])
def test_receipt_consumer_rejects_decision_bearing_counterexamples(qualified, fault):
    receipts = copy.deepcopy(qualified[1])
    first = receipts[0]
    if fault == 'missing_rank': receipts.pop()
    elif fault == 'reordered_rank': receipts.reverse()
    elif fault == 'missing_image': first['image_ids'].pop()
    elif fault == 'reordered_request': first['acquisitions']['vllm_v0_sample']['rows'].reverse()
    elif fault == 'wrong_scale': first['replay'][0]['backward_scale'] = 1/2
    elif fault == 'wrong_sequence': first['phase_order'].reverse()
    elif fault == 'stale_snapshot': first['updated_refresh']['rpc']['identity'] = first['initial_snapshot']
    elif fault == 'missing_channel': first['acquisitions']['vllm_v0_sample']['rows'][0]['policy_logprobs'] = None
    elif fault == 'duplicate_channels':
        for receipt in receipts:
            for phase, stage in receipt['acquisitions'].items():
                if phase.startswith('vllm'):
                    for row in stage['rows']: row['policy_logprobs'] = list(row['raw_logprobs'])
    elif fault == 'unsettled_child': first['child_settlement']['settled'] = False
    with pytest.raises(ValueError): _entry().validate_qualification(receipts)


def test_failed_entry_preserves_partial_evidence_and_closes_owned_children():
    root, receipts, exits = _invoke(fault=True)
    assert exits == [1, 1]
    assert not (root/'receipt.json').exists()
    for receipt in receipts:
        assert receipt['status'] == 'failed' and receipt['actual_exit_status'] == 1
        assert 'policy_logprobs' in receipt['error']
        assert receipt['acquisitions']['vllm_v0_greedy']['rows']
        assert receipt['child_settlement']['settled'] and receipt['child_settlement']['exitcode'] == 0
        assert receipt['distributed_settlement'] == 'destroyed'
        assert not (Path('/proc')/str(receipt['child_settlement']['pid'])).exists()


def test_qualification_static_source_owners_exist_in_selected_stack() -> None:
    import importlib.util
    import runpy
    from pathlib import Path

    probe_path = Path(__file__).resolve().parents[2] / "scripts/probes/coordexp_infras/vllm_qualification.py"
    probe = runpy.run_path(str(probe_path))
    package_root = Path(importlib.util.find_spec("vllm").origin).parent
    missing = [relative for relative in probe["STATIC_SOURCE_FILES"].values() if not (package_root / relative).is_file()]
    assert missing == []
    assert probe["STATIC_SOURCE_FILES"]["prompt_inputs"] == "inputs/llm.py"
