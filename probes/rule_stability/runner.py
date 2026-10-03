"""Shared eight-rank lifecycle, with CPU model substitution and native lead release."""
from __future__ import annotations

import os
import resource
import time
from pathlib import Path

from . import artifacts as a


def seed_for(version, image_id):
    import hashlib
    return int.from_bytes(hashlib.sha256(f'92711:{version}:{image_id}'.encode()).digest()[:4], 'big')


def synchronize_gradients(parameters):
    """Each image backward is /18; SUM makes uneven rank shards the image mean."""
    import torch
    import torch.distributed as dist
    parameters = list(parameters)
    if not parameters:
        raise ValueError('no trainable parameters')
    buffer = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                        for p in parameters])
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    if not torch.isfinite(buffer).all():
        raise ValueError('nonfinite global gradient')
    offset = 0
    for parameter in parameters:
        parameter.grad = buffer[offset:offset + parameter.numel()].view_as(parameter).clone()
        offset += parameter.numel()


def backward_branch(loss, groups):
    """Record each branch's incoming gradient, preserving accumulated gradients."""
    import math
    totals = {name: 0. for name in groups}
    hooks = []
    for name, parameters in groups.items():
        def capture(grad, name=name):
            totals[name] += float(grad.detach().float().square().sum())
        hooks.extend(p.register_hook(capture) for p in parameters)
    try:
        (loss / 18).backward()
    finally:
        for hook in hooks:
            hook.remove()
    return {name: math.sqrt(value) for name, value in totals.items()}


class CPUFixtureEngine:
    """Only model computations are substituted; real encoding and consumers remain."""
    kind = 'cpu_fixture'

    def __init__(self, images, inputs):
        import torch
        from probes.rollout_row_credit import frontend
        from .data import full_label_sequence
        from src.losses.vocab import build_token_vocabulary_groups
        from .policy import MedianPolicy
        from types import SimpleNamespace
        self.q = frontend()
        if self.q.model is not None:
            raise ValueError('CPU lifecycle accidentally loaded a model')
        self.inputs = inputs
        self.positive = {image['image_id']: full_label_sequence(image, self.q) for image in images}
        selected = a.load(a.ANCHOR / 'special_token_embeddings/special_token_embeddings.json')['token_ids']
        self.groups = {name: [torch.nn.Parameter(torch.full((1,) if name == 'language' else (len(selected), 1), .1))]
                       for name in ('language', 'input_delta', 'output_delta')}
        self.vocab = build_token_vocabulary_groups(self.q.token_identity, tokenizer=self.q.tokenizer)
        self.selected = torch.tensor(selected, dtype=torch.long)
        self.profile = torch.linspace(.5, 1.5, len(self.q.tokenizer))
        head = SimpleNamespace(weight=self.profile[:, None].to(torch.bfloat16), bias=None,
            selected_token_ids=self.selected, shared_embed_delta=self.groups['output_delta'][0])
        input_head = SimpleNamespace(shared_embed_delta=self.groups['input_delta'][0])
        policy_model = SimpleNamespace(get_output_embeddings=lambda: head, get_input_embeddings=lambda: input_head)
        self.norm = MedianPolicy(policy_model, list(self.vocab.coordinate))
        self.parameters = [p for values in self.groups.values() for p in values]
        self.optimizer = torch.optim.AdamW([dict(params=values, lr=1e-5 if name == 'language' else 5e-6)
            for name, values in self.groups.items()], betas=(.9, .999), eps=1e-8, weight_decay=0)
        self.calls = dict(greedy=0, sample=0, positive_replay=0, geometry_replay=0, duplicate_replay=0)
        self.composition = dict(kind=self.kind, model_loaded=False,
                               substitution='synthetic model decisions/logits only',
                               frontend=self.q.to_artifact_dict())

    def generate(self, image, version, channel):
        self.calls[channel] += 1
        encoded, sequence, _ = self.positive[image['image_id']]
        prompt = self.inputs[image['image_id']]['prompt_token_ids']
        if list(sequence.input_ids[:len(prompt)]) != prompt:
            raise ValueError('real encoding prompt differs from frozen input')
        tokens = list(sequence.input_ids[len(prompt):])
        eos = self.q.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if eos not in tokens:
            raise ValueError('full-label rendering lost actual EOS')
        tokens = tokens[:tokens.index(eos) + 1]
        # A synthetic sampled duplicate supplies a nonempty event and EOS path.
        if channel == 'sample':
            box_end = self.q.tokenizer.convert_tokens_to_ids('<|box_end|>')
            end = tokens.index(box_end) + 1
            tokens = tokens[:end] + tokens[:end] + tokens[end:]
        return dict(token_ids=tokens, text=self.q.tokenizer.decode(tokens, skip_special_tokens=False),
                    stop_reason='im_end', raw_logprobs=[-1.] * len(tokens),
                    policy_logprobs=[-1.25] * len(tokens), generation_seconds=0.)

    def train_image(self, image, greedy, sample, analyses, arm):
        # Same objective/replay/backward caller as native; only score computation differs.
        result = NativeEngine.train_image(self, image, greedy, sample, analyses, arm)
        for value in result.values():
            value['computation'] = 'CPU synthetic scores; actual full-vocabulary objectives'
        return result

    def replay(self, image_id, history, positions):
        import torch
        if any(position < 0 or position >= len(history) - 1 for position in positions):
            raise ValueError('CPU synthetic forward received noncausal target positions')
        scores = self.profile * self.groups['language'][0] + self.profile.cos() * self.groups['input_delta'][0].mean()
        scores = scores.index_add(0, self.selected, self.groups['output_delta'][0][:, 0])
        raw = scores.reshape(1, 1, -1).expand(1, len(positions), -1)
        return raw, self.norm.transform_replay(raw)

    def checkpoint(self, directory, arm, version):
        import torch
        from safetensors.torch import save_file
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'adapter').mkdir()
        (directory / 'special_token_embeddings').mkdir()
        save_file({'fixture_language': self.groups['language'][0].detach().cpu()},
                  str(directory / 'adapter/adapter_model.safetensors'))
        save_file({name: self.groups[group][0].detach().cpu()
                   for name, group in [('input_embed_delta', 'input_delta'), ('output_embed_delta', 'output_delta')]},
                  str(directory / 'special_token_embeddings/special_token_embeddings.safetensors'))
        a.write(directory / 'special_token_embeddings/special_token_embeddings.json',
                dict(tie_word_embeddings=False, tensor_shape=list(self.groups['output_delta'][0].shape), tensor_dtype='float32'))
        torch.save(dict(completed_updates=version, arm=arm, state_dict=self.optimizer.state_dict()), directory / 'optimizer.pt')
        schema = [dict(name=name, role=name, shape=list(parameters[0].shape), dtype=str(parameters[0].dtype))
                  for name, parameters in self.groups.items()]
        a.seal_checkpoint(directory, arm=arm, version=version, engine=self.kind, parameter_schema=schema)


def native_components(checkpoint):
    """Use maintained loaders; the parent qualifies base bytes once per invocation."""
    import torch
    from dataclasses import replace
    from probes import iterative_positive as p
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    from src.adapters.dora import load_live_dora_adapter
    from src.qwen.untied_embeddings import (SpecialTokenSelection, install_special_token_embedding_deltas,
                                           load_special_token_embedding_deltas)
    policy = p.load(p.POLICY)
    q = load_qwen_components_from_options(QwenLoadOptions(policy['base_model'], 'bf16',
                                                         'flash_attention_2', load_model=True))
    model, adapter = load_live_dora_adapter(q.model, adapter_path=checkpoint / 'adapter', base_model_path=q.base_model_path)
    q = replace(q, model=model)
    metadata = a.load(checkpoint / 'special_token_embeddings/special_token_embeddings.json')
    if metadata['tie_word_embeddings'] is not False:
        raise ValueError('native checkpoint needs independent deltas')
    delta = install_special_token_embedding_deltas(model,
        SpecialTokenSelection(token_strings=metadata['token_strings'], token_ids=metadata['token_ids']),
        tie_word_embeddings=False)
    payload = load_special_token_embedding_deltas(delta, checkpoint / 'special_token_embeddings',
        expected_base_model_path=q.base_model_path, expected_base_config_sha256=q.base_config_sha256,
        expected_tokenizer_sha256=q.tokenizer_sha256)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('lora_' in name or name in delta.receipt.delta_parameter_names)
    model.to('cuda')
    if any('visual' in name for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise ValueError('unexpected visual trainable')
    trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainables) != 590 or any(parameter.dtype != torch.float32 for parameter in trainables):
        raise ValueError('native trainable schema differs')
    return q, delta, dict(adapter=adapter, delta=payload.to_artifact_dict(), frontend=q.to_artifact_dict(),
                         base='BF16', trainable='FP32', attention='flash_attention_2')


class NativeEngine:
    kind = 'native'

    def __init__(self, images, inputs, checkpoint, *, prepare_positive=True):
        import torch
        from probes import iterative_positive as p
        from probes.online_row_credit import set_checkpointing
        from src.losses.vocab import build_token_vocabulary_groups
        from .data import full_label_sequence
        from .policy import MedianPolicy
        self.q, self.delta, self.composition = native_components(checkpoint)
        self.inputs = inputs
        self.batches = {image['image_id']: p.native_request(inputs[image['image_id']], p.load(p.POLICY), self.q.processor)
                        for image in images}
        for image_id, batch in self.batches.items():
            bound = inputs[image_id]
            if (list(batch.prompt_token_ids[0]) != bound['prompt_token_ids'] or
                    list(batch.image_grids[0]) != bound['image_grid_thw'] or
                    batch.media_sha256[0] != bound['media_sha256']):
                raise ValueError('native prepared input differs')
        self.positive = ({image['image_id']: full_label_sequence(image, self.q) for image in images}
                         if prepare_positive else {})
        self.vocab = build_token_vocabulary_groups(self.q.token_identity, tokenizer=self.q.tokenizer)
        self.norm = MedianPolicy(self.q.model, list(self.vocab.coordinate))
        deltas = self.delta.delta_tensors()
        delta_ids = {id(x) for x in deltas.values()}
        self.groups = dict(language=[x for x in self.q.model.parameters() if x.requires_grad and id(x) not in delta_ids],
                           input_delta=[deltas['input_embed_delta']], output_delta=[deltas['output_embed_delta']])
        self.parameters = [p for group in self.groups.values() for p in group]
        self.optimizer = torch.optim.AdamW([dict(params=group, lr=1e-5 if name == 'language' else 5e-6)
            for name, group in self.groups.items()], betas=(.9, .999), eps=1e-8, weight_decay=0)
        if self.q.model.config.text_config.attention_dropout != 0:
            raise ValueError('attention dropout changes policy')
        if any(getattr(module, 'p', 0) != 0 for name, module in self.q.model.named_modules() if 'lora_dropout' in name):
            raise ValueError('adapter dropout changes policy')
        set_checkpointing(self.q.model, True)
        self.calls = dict(greedy=0, sample=0, positive_replay=0, geometry_replay=0, duplicate_replay=0)

    def generate(self, image, version, channel):
        import torch
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        self.q.model.eval()
        begin = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = generate_continuations(self.q.model, self.batches[image['image_id']], extensions=[()],
                budgets=[3084], eos_token_id=self.q.tokenizer.convert_tokens_to_ids('<|im_end|>'),
                pad_token_id=self.q.tokenizer.pad_token_id,
                policy=NativeGenerationPolicy(temperature=1. if channel == 'sample' else 0.,
                    top_p=1., top_k=0, repetition_penalty=1., use_model_defaults=False),
                trace='raw_and_policy', seed=seed_for(version, image['image_id']) if channel == 'sample' else None,
                allow_pad_tokens=True,
                logits_processor=[self.norm.generation_transform()])[0]
        self.calls[channel] += 1
        return dict(token_ids=list(result.token_ids), text=self.q.tokenizer.decode(result.token_ids, skip_special_tokens=False),
                    stop_reason=result.stop_reason, raw_logprobs=list(result.raw_logprobs),
                    policy_logprobs=list(result.policy_logprobs), generation_seconds=time.monotonic() - begin)

    def replay(self, image_id, history, positions, *, record_media_masks=False):
        import torch
        from src.qwen.native import exact_history_inputs
        from .replay import prompt_only_placeholder_masks
        if not positions:
            raise ValueError('empty replay positions')
        kwargs = exact_history_inputs(self.q.model, self.batches[image_id].inputs, [tuple(history)],
                                      pad_token_id=self.q.tokenizer.pad_token_id, prompt_only_media=True)
        kwargs['logits_to_keep'] = torch.tensor(positions, device='cuda')
        prompt = self.batches[image_id].prompt_token_ids[0]
        with prompt_only_placeholder_masks(self.q.model, prompt,
                record_masks=record_media_masks) as binding, torch.autocast('cuda', dtype=torch.bfloat16):
            raw = self.q.model(**kwargs).logits
        if record_media_masks:
            self.last_replay_media = dict(binding, input_ids=kwargs['input_ids'].detach().cpu().tolist(),
                attention_mask=kwargs['attention_mask'].detach().cpu().tolist(),
                mm_token_type_ids=kwargs['mm_token_type_ids'].detach().cpu().tolist(),
                position_ids=kwargs['position_ids'].detach().cpu().tolist(),
                causal_positions=list(positions))
        return raw, self.norm.transform_replay(raw)

    def technical_suffix_diagnostic(self, image_id=1584):
        """One qualification-only cached/replay counterexample, without training."""
        import torch
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        from src.losses.token_scores import aligned_token_logprobs
        from .policy import TechnicalSuffixSelection, replay_difference
        if image_id != 1584 or image_id not in self.batches:
            raise ValueError('technical diagnostic must use resident rank0/image1584')
        tokenizer, config = self.q.tokenizer, self.q.model.config
        names = ('<|endoftext|>', '<|image_pad|>', '<|video_pad|>',
                 '<|vision_start|>', '<|vision_end|>', '<|im_end|>')
        actions = tuple(tokenizer.convert_tokens_to_ids(name) for name in names)
        if actions[0] != tokenizer.pad_token_id:
            raise ValueError('technical PAD differs from bound tokenizer')
        for index, field in ((1, 'image_token_id'), (2, 'video_token_id'), (3, 'vision_start_token_id')):
            if actions[index] != getattr(config, field):
                raise ValueError(f'technical token/config identity differs: {field}')
        if getattr(config, 'vision_end_token_id', actions[4]) != actions[4]:
            raise ValueError('technical vision-end token/config identity differs')
        batch = self.batches[image_id]
        prompt = batch.prompt_token_ids[0]
        select = TechnicalSuffixSelection(actions, len(prompt))
        self.q.model.eval()
        begin = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated = generate_continuations(self.q.model, batch, extensions=[()], budgets=[6],
                eos_token_id=actions[-1], pad_token_id=actions[0],
                policy=NativeGenerationPolicy(temperature=0., top_p=1., top_k=0,
                    repetition_penalty=1., use_model_defaults=False), trace='raw_and_policy',
                allow_pad_tokens=True, logits_processor=[self.norm.generation_transform(), select])[0]
        generation_seconds = time.monotonic() - begin
        if (generated.token_ids != actions or generated.stop_reason != 'im_end'
                or len(select.unforced_policy_logprobs) != 6):
            raise ValueError('technical cached suffix did not preserve the exact six actions')
        positions = tuple(range(len(prompt) - 1, len(prompt) + 5))
        begin = time.monotonic()
        with torch.inference_mode():
            raw, normalized = self.replay(image_id, [*prompt, *actions], positions, record_media_masks=True)
            targets = torch.tensor(actions, device=raw.device)
            comparison = replay_difference(request_id=f'rule-stability:technical:0:{image_id}',
                token_ids=actions, behavior_raw_logprobs=generated.raw_logprobs,
                behavior_policy_logprobs=select.unforced_policy_logprobs,
                replay_raw_logprobs=aligned_token_logprobs(raw[0], targets),
                replay_policy_logprobs=aligned_token_logprobs(normalized[0], targets))
        media = self.last_replay_media
        if (media['input_ids'] != [[*prompt, *actions]] or
                media['attention_mask'] != [[1] * (len(prompt) + 6)] or
                media['mm_token_type_ids'][0][-6:] != [0] * 6 or
                not media['calls'] or not any(call['prompt_image_positions'] > 0 for call in media['calls']) or
                any(call['suffix_image_true'] or call['suffix_video_true'] for call in media['calls'])):
            raise ValueError('technical replay media/action binding differs')
        return dict(status='complete', kind='teacher_forced_technical_evidence', image_id=image_id,
            anchor_version=0, action_names=list(names), token_ids=list(actions), actions=6,
            stop_reason=generated.stop_reason, comparison=comparison,
            forced_selection_logprobs=list(generated.policy_logprobs),
            likelihood_definition='comparison.policy is unforced median-normalized conditional likelihood; forced selection is separate',
            replay_media=media, finite=True, action_alignment=True, prompt_media_preserved=True,
            suffix_media_masks_false=True, generation_seconds=generation_seconds,
            replay_seconds=time.monotonic() - begin, training_contribution=False, backward_calls=0,
            scientific_metric=False, work=dict(generation_requests=1, generated_actions=6, replay_requests=1))

    def train_image(self, image, greedy, sample, analyses, arm):
        import torch
        from .data import normalized_positive_objective
        from .objectives import geometry_objective, duplicate_advantages, duplicate_loss
        from .policy import replay_difference
        if self.q.model is not None:
            self.q.model.train()
        image_id = image['image_id']
        prompt = self.inputs[image_id]['prompt_token_ids']
        _, sequence, row_ids = self.positive[image_id]
        if list(sequence.input_ids[:len(prompt)]) != prompt:
            raise ValueError('positive/native prompt differs')
        positions = tuple(a.causal_logits_position for a in sequence.atoms)
        begin = time.monotonic()
        raw, normalized = self.replay(image_id, sequence.input_ids, positions)
        loss, rows = normalized_positive_objective(normalized, sequence, row_ids, self.vocab, positions)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite positive objective')
        result = dict(positive=dict(loss=float(loss.detach()), weighted_loss=float(loss.detach()), rows=rows,
            gradient_l2_per_image_over18=backward_branch(loss, self.groups), seconds=time.monotonic() - begin))
        self.calls['positive_replay'] += 1
        del raw, normalized, loss
        sites = analyses['greedy']['geometry_sites']
        if sites:
            positions = tuple(sorted({len(prompt) + site['position'] - 1 for site in sites}))
            begin = time.monotonic()
            raw, normalized = self.replay(image_id, prompt + greedy['token_ids'], positions)
            loss, details = geometry_objective(normalized, positions, sites, len(prompt), self.vocab.coordinate)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite geometry objective')
            result['geometry'] = dict(loss=float(loss.detach()), weighted_loss=float(.1 * loss.detach()),
                details=details, gradient_l2_per_image_over18=backward_branch(.1 * loss, self.groups),
                seconds=time.monotonic() - begin)
            self.calls['geometry_replay'] += 1
            del raw, normalized, loss
        else:
            result['geometry'] = dict(loss=0., weighted_loss=0., sites=0,
                                     gradient_l2_per_image_over18={name: 0. for name in self.groups})
        if arm == 'B':
            begin = time.monotonic()
            tokens = sample['token_ids']
            if not tokens:
                raise ValueError('generation omitted even EOS action')
            positions = tuple(range(len(prompt) - 1, len(prompt) + len(tokens) - 1))
            raw, normalized = self.replay(image_id, prompt + tokens, positions)
            advantage = duplicate_advantages(analyses['sample']['event_positions'], analyses['greedy']['event_positions'],
                                             len(tokens), len(greedy['token_ids']))
            loss = duplicate_loss(normalized, tokens, advantage)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite duplicate objective')
            from src.losses.token_scores import aligned_token_logprobs
            targets = torch.tensor(tokens, device=raw.device)
            comparison = replay_difference(request_id=sample['request_id'], token_ids=tokens,
                behavior_raw_logprobs=sample['raw_logprobs'], behavior_policy_logprobs=sample['policy_logprobs'],
                replay_raw_logprobs=aligned_token_logprobs(raw[0], targets),
                replay_policy_logprobs=aligned_token_logprobs(normalized[0], targets))
            result['duplicate'] = dict(loss=float(loss.detach()), weighted_loss=float(loss.detach()),
                advantages=advantage.detach().cpu().tolist(), nonzero_advantages=int((advantage != 0).sum()),
                replay_comparison=comparison, gradient_l2_per_image_over18=backward_branch(loss, self.groups),
                seconds=time.monotonic() - begin)
            self.calls['duplicate_replay'] += 1
            del raw, normalized, loss
        return result

    def checkpoint(self, directory, arm, version):
        import torch
        from probes.iterative_positive import save_checkpoint
        save_checkpoint(self.q, self.delta, directory)
        torch.save(dict(completed_updates=version, arm=arm, state_dict=self.optimizer.state_dict()), directory / 'optimizer.pt')
        names = {id(p): name for name, p in self.q.model.named_parameters()}
        schema = [dict(name=names[id(p)], role=role, shape=list(p.shape), dtype=str(p.dtype))
                  for role, group in self.groups.items() for p in group]
        a.seal_checkpoint(directory, arm=arm, version=version, engine=self.kind, parameter_schema=schema)


def run_rank(config_path, output):
    """The same rank entry publishes CPU/native trajectories, steps and terminal receipt."""
    import torch
    import torch.distributed as dist
    from probes.online_row_credit import seal
    from .data import load_inputs, request_records, balanced_layout
    from .objectives import trajectory_analysis
    config = a.load(config_path)
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8 or config['world_size'] != world or config['engine'] not in ('cpu_fixture', 'native'):
        raise ValueError('rank/world/engine differs')
    output = Path(output)
    if config['output'] != str(output.resolve()):
        raise ValueError('output does not match invocation')
    begin = time.monotonic()
    directory = output / f'rank-{rank}'
    directory.mkdir(parents=True, exist_ok=False)
    native = config['engine'] == 'native'
    if native:
        # Parent verified a released packet before starting any device process.
        binding = a.load(output / 'invocation.json')
        if binding['config_sha256'] != a.digest(config_path) or binding['native_released'] is not True:
            raise ValueError('native parent binding absent')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    a.write(directory / 'entry.json', dict(schema=a.SCHEMA, rank=rank, world_size=world, pid=os.getpid(),
        engine=config['engine'], source=config['source'], start=time.time(), config_sha256=a.digest(config_path)))
    dist.init_process_group('nccl' if native else 'gloo')
    try:
        images, manifest = load_inputs()
        records = request_records(images, manifest)
        if config['inputs']['manifest_sha256'] != a.digest(a.ROOT / config['inputs']['manifest_path']):
            raise ValueError('input manifest changed')
        by_input = {record['image_id']: record for record in records}
        assignment = balanced_layout(records)
        # Static assignment is shared for learning/acquisition; SUM gradients handles unequal work.
        local_ids = assignment['groups'][rank] if 'groups' in assignment else assignment['ranks'][rank]['image_ids']
        local_images = [image for image in images if image['image_id'] in local_ids]
        torch.manual_seed(92711)
        engine = NativeEngine(local_images, by_input, Path(config['anchor']['path'])) if native else CPUFixtureEngine(local_images, by_input)
        a.write(directory / 'composition.json', engine.composition)
        a.write(directory / 'assignment.json', dict(image_ids=local_ids, assignment=assignment,
            backward_denominator=18, collective='gradient_SUM_once_per_update', local_images=len(local_images)))
        if rank == 0:
            engine.checkpoint(output / 'checkpoint-0', config['arm'], 0)
        dist.barrier()
        if native and config['mode'] == 'qualification':
            if rank == 0:
                diagnostic = engine.technical_suffix_diagnostic()
                diagnostic.update(source=config['source'], config_sha256=a.digest(config_path))
                a.write(directory / 'technical-suffix-1584.json', diagnostic)
            dist.barrier()
        for version in range(config['updates'] + 1):
            acquired = []
            analyses = {}
            for image in local_images:
                image_id = image['image_id']
                channels = ('greedy', 'sample') if version < config['updates'] else ('greedy',)
                analyses[image_id] = {}
                for channel in channels:
                    producer = dict(kind='rule_stability', engine=engine.kind, arm=config['arm'], version=version,
                                    channel=channel, source=config['source'], config_sha256=a.digest(config_path))
                    result = engine.generate(image, version, channel)
                    record = seal(dict(by_input[image_id], **result, arm='greedy' if channel == 'greedy' else 'sample',
                        request_id=f"rule-stability:{config['arm']}:{version}:{channel}:{image_id}",
                        seed=seed_for(version, image_id) if channel == 'sample' else None,
                        temperature=1. if channel == 'sample' else 0., generated_tokens=len(result['token_ids']),
                        coordinate_norm='median', generation_rank=rank), producer)
                    analysis = trajectory_analysis(record, engine.q.tokenizer)
                    analyses[image_id][channel] = analysis
                    a.write(directory / f'version-{version}/{channel}-{image_id}.json', record)
                    a.write(directory / f'version-{version}/{channel}-{image_id}-analysis.json', analysis)
                    acquired.append(record)
            a.write(directory / f'version-{version}/acquisition.json', dict(status='complete',
                records=[record['request_id'] for record in acquired], image_ids=local_ids))
            if version == config['updates']:
                break
            engine.optimizer.zero_grad(set_to_none=True)
            updates = []
            for image in local_images:
                image_id = image['image_id']
                greedy = next(r for r in acquired if r['image_id'] == image_id and r['arm'] == 'greedy')
                sample = next(r for r in acquired if r['image_id'] == image_id and r['arm'] == 'sample')
                branches = engine.train_image(image, greedy, sample, analyses[image_id], config['arm'])
                updates.append(dict(image_id=image_id, branches=branches, denominator=18))
            synchronize_gradients(engine.parameters)
            norm = torch.nn.utils.clip_grad_norm_(engine.parameters, 1., error_if_nonfinite=True)
            engine.optimizer.step()
            if any(not torch.isfinite(p).all() for p in engine.parameters):
                raise ValueError('nonfinite optimizer parameter')
            a.write(directory / f'update-{version + 1}.json', dict(status='complete', update=version + 1,
                image_contributions=updates, global_gradient_l2_before_clip=float(norm),
                optimizer_steps=version + 1, learning_rates=[g['lr'] for g in engine.optimizer.param_groups]))
            if version + 1 in config['checkpoint_versions'] and rank == 0:
                engine.checkpoint(output / f'checkpoint-{version + 1}', config['arm'], version + 1)
            dist.barrier()
        a.write(directory / 'complete.json', dict(status='complete', schema=a.SCHEMA, rank=rank,
            engine=engine.kind, arm=config['arm'], updates=config['updates'], calls=engine.calls,
            seconds=time.monotonic() - begin, rss_max_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if native else 0,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if native else 0,
            files=a.file_manifest(directory)))
    finally:
        dist.destroy_process_group()


def readback(config_path, output):
    """Fresh offline consumer; sealed raw records never get rewritten."""
    from .data import load_inputs, request_records
    from .consumer import evaluate
    from probes.online_row_credit import seal
    config = a.load(config_path)
    images, manifest = load_inputs()
    bound = {r['image_id']: r for r in request_records(images, manifest)}
    versions = {version: [] for version in range(config['updates'] + 1)}
    checkpoints = []
    rank_receipts = []
    for rank in range(8):
        directory = Path(output) / f'rank-{rank}'
        receipt = a.load(directory / 'complete.json')
        if (receipt['status'] != 'complete' or receipt['rank'] != rank or receipt['updates'] != config['updates']
                or receipt['engine'] != config['engine'] or receipt['arm'] != config['arm']):
            raise ValueError('rank completion identity differs')
        a.verify_manifest(directory, receipt['files'], excluded=('complete.json',))
        assigned = a.load(directory / 'assignment.json')['image_ids']
        for version in versions:
            for image_id in assigned:
                for channel in (('greedy', 'sample') if version < config['updates'] else ('greedy',)):
                    record = a.load(directory / f'version-{version}/{channel}-{image_id}.json')
                    if record['raw_identity'] != seal(record, record['producer'])['raw_identity']:
                        raise ValueError('trajectory raw seal changed')
                    if any(record[key] != value for key, value in bound[image_id].items() if key != 'request_id'):
                        raise ValueError('trajectory input changed')
                    producer = record['producer']
                    if any(producer[k] != v for k, v in dict(kind='rule_stability', engine=config['engine'],
                            arm=config['arm'], version=version, channel=channel, source=config['source'],
                            config_sha256=a.digest(config_path)).items()):
                        raise ValueError('trajectory producer differs')
                    tokens = record['token_ids']
                    if (len(tokens) > 3084 or record['generated_tokens'] != len(tokens) or
                            len(record['raw_logprobs']) != len(tokens) or len(record['policy_logprobs']) != len(tokens)):
                        raise ValueError('trajectory actions/likelihoods differ')
                    if channel == 'greedy':
                        versions[version].append(record)
        rank_receipts.append(receipt)
    for version in config['checkpoint_versions']:
        checkpoints.append(a.checkpoint_readback(Path(output) / f'checkpoint-{version}', arm=config['arm'],
                                                version=version, engine=config['engine']))
    diagnostic = None
    if config['engine'] == 'native' and config['mode'] == 'qualification':
        diagnostic = a.load(Path(output) / 'rank-0/technical-suffix-1584.json')
        if (diagnostic['status'] != 'complete' or diagnostic['image_id'] != 1584 or
                diagnostic['anchor_version'] != 0 or diagnostic['actions'] != 6 or
                diagnostic['training_contribution'] is not False or diagnostic['backward_calls'] != 0 or
                diagnostic['scientific_metric'] is not False or diagnostic['source'] != config['source'] or
                diagnostic['config_sha256'] != a.digest(config_path)):
            raise ValueError('qualification technical suffix binding differs')
    metrics = evaluate(images, versions)
    a.write(Path(output) / 'metrics.json', metrics)
    a.write(Path(output) / 'readback.json', dict(schema=a.SCHEMA, status='complete', engine=config['engine'],
        scientific_evidence=config['engine'] == 'native', arm=config['arm'], updates=config['updates'],
        greedy_versions=list(versions), images=18, labels=570, checkpoints=checkpoints,
        calls={key: sum(r['calls'][key] for r in rank_receipts) for key in rank_receipts[0]['calls']},
        max_rank_seconds=max(r['seconds'] for r in rank_receipts),
        max_rank_rss_kib=max(r['rss_max_kib'] for r in rank_receipts),
        technical_diagnostic=diagnostic, metrics_sha256=a.digest(Path(output) / 'metrics.json')))
    return metrics


def reload_rank(config_path, output):
    """Fresh native checkpoint load and one greedy per image; qualification only."""
    import torch
    import torch.distributed as dist
    from .data import load_inputs, request_records, balanced_layout
    from probes.online_row_credit import seal
    config = a.load(config_path)
    parent = Path(output).parent
    invocation = a.load(parent / 'invocation.json')
    if (config['mode'] != 'qualification' or config['released'] is not True or
            invocation['native_released'] is not True or invocation['config_sha256'] != a.digest(config_path)):
        raise ValueError('fresh reload lacks qualification release')
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 8:
        raise ValueError('reload rank layout differs')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    begin = time.monotonic()
    directory = Path(output) / f'rank-{rank}'
    directory.mkdir(parents=True, exist_ok=False)
    try:
        images, manifest = load_inputs()
        inputs = request_records(images, manifest)
        assigned = balanced_layout(inputs)['ranks'][rank]['image_ids']
        local = [image for image in images if image['image_id'] in assigned]
        engine = NativeEngine(local, {r['image_id']: r for r in inputs}, parent / 'checkpoint-1', prepare_positive=False)
        optimizer = torch.load(parent / 'checkpoint-1/optimizer.pt', map_location='cpu', weights_only=True)
        engine.optimizer.load_state_dict(optimizer['state_dict'])
        if optimizer['completed_updates'] != 1 or any(int(s['step']) != 1 for s in engine.optimizer.state.values()):
            raise ValueError('fresh optimizer reload lost continuity')
        rows = []
        for image in local:
            image_id = image['image_id']
            result = engine.generate(image, 1, 'greedy')
            original = a.load(parent / f'rank-{rank}/version-1/greedy-{image_id}.json')
            producer = dict(kind='fresh_checkpoint_reload', source=config['source'], arm=config['arm'], version=1,
                            original_raw_identity=original['raw_identity'], config_sha256=a.digest(config_path))
            record = seal(dict(engine.inputs[image_id], **result, arm='greedy',
                request_id=f"rule-stability:{config['arm']}:1:reload:{image_id}", generated_tokens=len(result['token_ids'])), producer)
            a.write(directory / f'greedy-{image_id}.json', record)
            rows.append(dict(image_id=image_id, original_raw_identity=original['raw_identity'],
                reload_raw_identity=record['raw_identity'], token_ids_equal=record['token_ids'] == original['token_ids'],
                stop_reason_equal=record['stop_reason'] == original['stop_reason']))
        a.write(directory / 'complete.json', dict(status='complete', rank=rank, checkpoint_version=1,
            fresh_model_load=True, optimizer_updates=1, comparisons=rows, calls=engine.calls,
            seconds=time.monotonic() - begin, rss_max_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), files=a.file_manifest(directory)))
    finally:
        dist.destroy_process_group()


def reload_readback(config_path, output):
    """Consume fresh qualification behavior without repeating original readback."""
    from .data import load_inputs
    from .consumer import evaluate
    config = a.load(config_path)
    images, _ = load_inputs()
    original, reloaded, comparisons = [], [], []
    for rank in range(8):
        directory = Path(output) / f'rank-{rank}'
        receipt = a.load(directory / 'complete.json')
        if receipt['status'] != 'complete' or receipt['rank'] != rank or not receipt['fresh_model_load']:
            raise ValueError('fresh reload completion differs')
        a.verify_manifest(directory, receipt['files'], excluded=('complete.json',))
        for comparison in receipt['comparisons']:
            image_id = comparison['image_id']
            original.append(a.load(Path(output).parent / f'rank-{rank}/version-1/greedy-{image_id}.json'))
            reloaded.append(a.load(directory / f'greedy-{image_id}.json'))
            comparisons.append(comparison)
    metrics = evaluate(images, {0: original, 1: reloaded})
    a.write(Path(output) / 'readback.json', dict(status='complete', scope='fresh checkpoint1 versus saved version1',
        new_greedy_requests=18, comparisons=comparisons, metrics=metrics,
        token_equal_images=sum(row['token_ids_equal'] for row in comparisons)))
