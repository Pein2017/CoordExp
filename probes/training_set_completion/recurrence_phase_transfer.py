"""Transfer the accepted historical-key phase contrast to train269858 row20 x1."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import transformers
from transformers import DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_census.prepare import _rows_from_tokens
from probes.training_set_completion.recurrence_donor_tracking import probability_readback, require, write
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest, check_cache
from probes.training_set_completion.recurrence_key_phase import make_keys, phase_qualification
from probes.training_set_completion.recurrence_written_content import hooks
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.config.inference import InferConfig
from src.data.examples import raw_example_from_jsonl_row
from src.inference.inputs import plan_examples
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-phase-transfer")
OUT = ROOT / "attempt-002"
PREVIOUS = ROOT / "attempt-001"
PREVIOUS_CAPTURE = PREVIOUS / "post-failure-source-capture.json"
INPUT_ORACLE = ROOT / "lead-checks/native-input-oracle.json"
CENSUS_PRODUCER = Path("probes/training_set_completion/recurrence_census/natural.py")
CENSUS_CURRENT_SHA256 = "c8304e6a726c057c46b493e09fb34ea04933819df61d75ed3979991fcaa34274"
SELECTION = ROOT / "selection/selected-case-metadata-v1.json"
SELECTION_SHA256 = "7e5d4b6fa308e1a10a47592e2c88d4f1a3c31fe5effb5dc3e512f059115e96aa"
UNIT = Path("research/experiments/2026-09-22-recurrence-phase-transfer/unit.md")
TARGET, OFFSET, WIDTH, FULL_WIDTH, SUFFIX = 1, 184, 1542, 1546, 4
SOURCE, DEST = (1524, 1533), (1533, 1542)
ROW = [151646, 8987, 151647, 151648, 152020, 151851, 152039, 151900, 151649]
S = ROW[:4]
OLD, NEW = 152020, 152018  # x1 coordinate bins 350 and 348.
LAYERS, KV_HEADS, HEAD_DIM = 28, 8, 128
MAX_FORWARDS, MAX_SECONDS, ATOL = 8, 900, 2e-4


def binding_matches(path, expected):
    actual = literal_binding(Path(path))
    return all(actual[key] == expected[key] for key in ("path", "sha256", "size_bytes"))


@contextmanager
def patch_key(cache, candidate, native_digest, *, target=TARGET, width=WIDTH, dest=DEST):
    require(target == TARGET and dest == (width - 9, width) and len(candidate) == len(cache.layers),
            "wrong batch/destination or layer count")
    saved = []
    try:
        for layer, block in zip(cache.layers, candidate, strict=True):
            require(block.shape == (KV_HEADS, 9, HEAD_DIM), "wrong candidate K block shape")
            old = layer.keys[target, :, dest[0]:dest[1], :].clone()
            saved.append((layer, old))
            layer.keys[target, :, dest[0]:dest[1], :].copy_(block)
        yield
    finally:
        cache.crop(width)
        for layer, old in saved:
            layer.keys[target, :, dest[0]:dest[1], :].copy_(old)
        require(cache_digest(cache) == native_digest, "historical cache did not restore exactly")


def block_hashes(cache, *, target=TARGET, source=SOURCE, dest=DEST):
    return {i: {part: {axis: tensor_hash(getattr(layer, axis)[target, :, start:end, :])
                      for axis in ("keys", "values")}
                for part, (start, end) in {"source": source, "dest": dest}.items()}
            for i, layer in enumerate(cache.layers)}


def suffix_observers(model, cache, expected_blocks, fixed_history):
    """Attest the four-token x1 suffix at each decoder attention."""
    seen, phase, handles = {}, [], []
    text = model.model.language_model
    def rotary(_module, _args, output):
        cos, sin = output
        require(cos.shape == sin.shape == (4, SUFFIX, HEAD_DIM), "current x1 phase shape changed")
        phase.append((cos.detach().clone(), sin.detach().clone()))
    handles.append(text.rotary_emb.register_forward_hook(rotary))
    for i, layer in enumerate(text.layers):
        def before(_module, _args, kwargs, *, index=i):
            require(kwargs.get("past_key_values") is cache and cache.get_seq_length(index) == WIDTH,
                    "attention consumed wrong cache or prefix width")
            mask, slots, embedded = (kwargs.get(k) for k in ("attention_mask", "cache_position", "position_embeddings"))
            require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and
                    mask.shape == (4, 1, SUFFIX, FULL_WIDTH) and isinstance(slots, torch.Tensor) and
                    torch.equal(slots, torch.arange(WIDTH, FULL_WIDTH, device=slots.device)) and
                    len(phase) == 1 and isinstance(embedded, tuple) and len(embedded) == 2 and
                    torch.equal(embedded[0], phase[0][0]) and torch.equal(embedded[1], phase[0][1]),
                    "attention mask/physical slots/current phase changed")
            current = block_hashes_one(cache.layers[index])
            require(current == expected_blocks[index], "attention consumed wrong historical K/V blocks")
            require(fixed_history_matches(cache.layers[index], fixed_history[index]),
                    "attention consumed changed companion or pre-source history")
            seen[index] = {"prefix_length": cache.get_seq_length(index), "mask_hash": tensor_hash(mask),
                           "cache_position_hash": tensor_hash(slots),
                           "phase_hashes": [tensor_hash(x) for x in embedded], "blocks": current,
                           "fixed_history_equal": True}
        handles.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
    return seen, phase, handles


def block_hashes_one(layer):
    return {part: {axis: tensor_hash(getattr(layer, axis)[TARGET, :, start:end, :])
                   for axis in ("keys", "values")}
            for part, (start, end) in {"source": SOURCE, "dest": DEST}.items()}


def fixed_history_snapshot(cache, *, source_start=SOURCE[0]):
    return [{axis: {"companions": getattr(layer, axis)[[0, 2, 3]].clone(),
                    "target_before_source": getattr(layer, axis)[TARGET, :, :source_start, :].clone()}
             for axis in ("keys", "values")} for layer in cache.layers]


def fixed_history_matches(layer, saved, *, source_start=SOURCE[0]):
    return all(torch.equal(getattr(layer, axis)[[0, 2, 3]], saved[axis]["companions"]) and
               torch.equal(getattr(layer, axis)[TARGET, :, :source_start, :],
                           saved[axis]["target_before_source"]) for axis in ("keys", "values"))


def selfcheck():
    sample = DynamicCache(((torch.zeros(4, KV_HEADS, 18, HEAD_DIM),
                            torch.ones(4, KV_HEADS, 18, HEAD_DIM)),))
    digest = cache_digest(sample)
    snapshot = fixed_history_snapshot(sample, source_start=9)
    old = sample.layers[0].keys.clone()
    block = [torch.full((KV_HEADS, 9, HEAD_DIM), 2.0)]
    with torch.inference_mode():
        with patch_key(sample, block, digest, width=18, dest=(9, 18)):
            require(torch.all(sample.layers[0].keys[TARGET, :, 9:18] == 2) and
                    torch.equal(sample.layers[0].keys[[0, 2, 3]], old[[0, 2, 3]]) and
                    torch.all(sample.layers[0].values == 1) and
                    fixed_history_matches(sample.layers[0], snapshot[0], source_start=9),
                    "target-only K patch failed")
            sample.update(torch.zeros(4, KV_HEADS, SUFFIX, HEAD_DIM),
                          torch.ones(4, KV_HEADS, SUFFIX, HEAD_DIM), 0)
        require(cache_digest(sample) == digest, "post-append restoration failed")
        sample.layers[0].keys[TARGET, 0, 0, 0] = 1
        require(not fixed_history_matches(sample.layers[0], snapshot[0], source_start=9),
                "wrong pre-source history escaped")
        sample.layers[0].keys[TARGET, 0, 0, 0] = 0
        sample.layers[0].values[0, 0, 0, 0] = 2
        require(not fixed_history_matches(sample.layers[0], snapshot[0], source_start=9),
                "wrong companion history escaped")
        sample.layers[0].values[0, 0, 0, 0] = 1
        for target, dest, candidate in ((0, (9, 18), block), (TARGET, (8, 17), block),
                                        (TARGET, (9, 18), [block[0][:, :-1]])):
            try:
                with patch_key(sample, candidate, digest, target=target, width=18, dest=dest):
                    pass
            except ValueError:
                pass
            else:
                raise AssertionError("wrong batch/block/shape passed")
        try:
            with patch_key(sample, block, digest, width=18, dest=(9, 18)):
                sample.layers[0].keys[0, 0, 0, 0] = 1
        except ValueError:
            pass
        else:
            raise AssertionError("companion corruption escaped exact restoration")
    angle = torch.linspace(0.1, 1.0, HEAD_DIM // 2)
    old_angle, new_angle = torch.cat((angle, angle)), torch.cat((angle + .2, angle + .2))
    oc, osin = old_angle.cos().repeat(9, 1), old_angle.sin().repeat(9, 1)
    nc, nsin = new_angle.cos().repeat(9, 1), new_angle.sin().repeat(9, 1)
    pre_old, pre_new = torch.randn(KV_HEADS, 9, HEAD_DIM), torch.randn(KV_HEADS, 9, HEAD_DIM)
    from probes.training_set_completion.recurrence_key_phase import rotate
    keys, errors = make_keys(pre_old, pre_new, oc, osin, nc, nsin,
                             rotate(pre_old, oc, osin), rotate(pre_new, nc, nsin))
    require(phase_qualification(pre_old, pre_new, keys["OO"], keys["NN"], nc, nsin,
                                oc, osin, keys, errors)["qualified"], "valid phase failed")
    wrong = dict(keys); wrong["ON"] = keys["OO"]
    require(not phase_qualification(pre_old, pre_new, keys["OO"], keys["NN"], nc, nsin,
                                    oc, osin, wrong, errors)["qualified"], "wrong phase passed")
    fake_raw = ROW * 2 + S
    require(_rows_from_tokens(fake_raw)[1]["start"] == 9 and fake_raw[18:22] == S,
            "x1 S boundary selfcheck failed")


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda") and not OUT.exists(),
            "CUDA and unused attempt path required")
    prior_receipt = json.loads((PREVIOUS / "receipt.json").read_text())
    prior_capture = json.loads(PREVIOUS_CAPTURE.read_text())
    require(prior_receipt["status"] == "technical_invalid" and prior_receipt["error"] == "KeyError('image_plan')" and
            prior_receipt["model_forwards"] == prior_receipt["vision_forwards"] == 0 and
            prior_capture["status"] == "captured_after_zero_forward_failure" and
            prior_capture["failure_receipt"] == literal_binding(PREVIOUS / "receipt.json") and
            prior_capture["captured_producer"]["sha256"] ==
            "71605ee2c76b52b224cb442de02497b7fae574b66b554adf92335127b16922b3" and
            binding_matches(prior_capture["captured_producer"]["path"], prior_capture["captured_producer"]),
            "zero-forward failed attempt/capture changed")
    input_oracle = json.loads(INPUT_ORACLE.read_text())
    require(input_oracle["status"] == "independent-CPU-native-input-construction" and
            input_oracle["width"] == FULL_WIDTH and input_oracle["target_block_valid"],
            "independent input oracle changed")
    require(hashlib.sha256(CENSUS_PRODUCER.read_bytes()).hexdigest() == CENSUS_CURRENT_SHA256,
            "maintained census request route changed")
    require(hashlib.sha256(SELECTION.read_bytes()).hexdigest() == SELECTION_SHA256,
            "frozen selection packet changed")
    packet = json.loads(SELECTION.read_text())
    require(packet["status"] == "candidate", "selection packet status changed")
    chosen = packet["root_selected"]
    require(chosen["image_key"] == "train:269858" and chosen["source_group"] == "new-14" and
            chosen["target_batch_index"] == TARGET and chosen["generated_action_offset_S"] == OFFSET and
            chosen["old_block_range"] == [162, 171] and chosen["new_block_range"] == [171, 180] and
            chosen["common_current_row_prefix_tokens"] == S, "frozen selection alignment changed")
    require(all(binding_matches(chosen[key]["path"], chosen[key]) for key in
                ("image", "raw", "trace", "runtime_receipt", "source_panel")),
            "selected source binding changed")
    raw = json.loads(Path(chosen["raw"]["path"]).read_text())["rows"]
    trace = json.loads(Path(chosen["trace"]["path"]).read_text())["steps"]
    receipt = json.loads(Path(chosen["runtime_receipt"]["path"]).read_text())
    panel = json.loads(Path(chosen["source_panel"]["path"]).read_text())
    group = next(x for x in panel["groups"] if x["key"] == "new-14")
    tokens = raw[TARGET]["token_ids"]
    rows = _rows_from_tokens(tokens)
    require(raw[TARGET]["image_id"] == 269858 and len(rows) > 20 and
            [(rows[i]["start"], rows[i]["end"]) for i in (18, 19, 20)] ==
            [(162, 171), (171, 180), (180, 189)] and
            tokens[162:171] == tokens[171:180] == ROW and tokens[180:184] == S and
            tokens[184] == NEW and trace[OFFSET]["offset"] == OFFSET and
            trace[OFFSET]["raw_winners"][TARGET] == NEW and
            trace[OFFSET]["raw_runnerups"][TARGET] == OLD,
            "saved target/source/trace offset changed")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device,
             "model_forwards": 0, "vision_forwards": 0, "started_unix": time.time(), "calls": [],
             "source": {"frozen_unit": literal_binding(UNIT), "selection": literal_binding(SELECTION),
                        "failed_previous_receipt": literal_binding(PREVIOUS / "receipt.json"),
                        "postfailure_producer_capture": literal_binding(PREVIOUS_CAPTURE),
                        "independent_native_input_oracle": literal_binding(INPUT_ORACLE),
                        "historical_census_producer": chosen["source_producer_expected"],
                        "current_census_producer": literal_binding(CENSUS_PRODUCER),
                        **{key: chosen[key] for key in ("image", "raw", "trace", "runtime_receipt", "source_panel")}}}
    write(OUT / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        actual, saved = dict(identity), dict(receipt["identity"])
        loader, prior_loader = actual.pop("loader_source"), saved.pop("loader_source")
        require(actual == saved and (loader["sha256"], loader["size_bytes"]) ==
                (prior_loader["sha256"], prior_loader["size_bytes"]) and
                (loader["sha256"], loader["size_bytes"]) ==
                (chosen["loader_source_binding"]["sha256"], chosen["loader_source_binding"]["size_bytes"]),
                "effective model/loader source identity changed")
        config_model = InferConfig.model_validate(panel["configs"]["untied"])
        raw_examples = [raw_example_from_jsonl_row(case["input_record"],
                        jsonl_path=Path(group["input_jsonl"]), row_number=int(case["row_index"]) + 1,
                        raw_line=json.dumps(case["input_record"])) for case in group["cases"]]
        planned = plan_examples(raw_examples, config=config_model, components=q,
                                row_indices=[int(case["row_index"]) for case in group["cases"]])
        requests = [item.request for item in planned]
        batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
        require(input_identity(batch) == receipt["input_identity"] and
                list(batch.request_ids) == chosen["batch_request_ids"], "selected native batch/image identity changed")
        tails = _prefix_tokens(raw, OFFSET, int(q.tokenizer.pad_token_id))
        histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
        native = exact_history_inputs(q.model, batch.inputs, histories,
                                      pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
        require(native["input_ids"].shape == (4, FULL_WIDTH) and
                native["position_ids"].shape == (3, 4, FULL_WIDTH) and
                native["input_ids"][TARGET, SOURCE[0]:SOURCE[1]].tolist() == ROW and
                native["input_ids"][TARGET, DEST[0]:DEST[1]].tolist() == ROW and
                native["input_ids"][TARGET, WIDTH:FULL_WIDTH].tolist() == S and
                len(batch.prompt_token_ids[TARGET]) + OFFSET == FULL_WIDTH,
                "actual physical source/destination/S alignment changed")
        require(tensor_hash(native["input_ids"]) == input_oracle["input_ids"] and
                tensor_hash(native["attention_mask"]) == input_oracle["attention_mask"] and
                tensor_hash(native["input_ids"][:, WIDTH:FULL_WIDTH]) == input_oracle["suffix_ids"] and
                [len(x) for x in histories] == input_oracle["history_lengths"],
                "reconstructed native histories differ from independent CPU oracle")
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        source_capture = preserve_source(Path(__file__), run_root=OUT,
                                         relative_name=f"recurrence_phase_transfer-{source_hash[:12]}.py")
        dependency_paths = [Path(p) for p in ("probes/training_set_completion/recurrence_key_phase.py",
                            "probes/training_set_completion/recurrence_census/natural.py",
                            "probes/training_set_completion/recurrence_census/prepare.py",
                            "probes/training_set_completion/recurrence_history_cache_partition.py",
                            "probes/training_set_completion/recurrence_written_content.py",
                            "probes/training_set_completion/recurrence_donor_tracking.py",
                            "probes/training_set_completion/untied_shared.py",
                            "probes/training_set_completion/numerical_feedback/runtime.py", "src/qwen/native.py",
                            "src/data/examples.py", "src/inference/inputs.py", "src/config/inference.py",
                            "src/qwen/untied_embeddings.py",
                            "src/qwen/input_identity.py", "src/artifacts/source_provenance.py",
                            "probes/training_set_completion/artifacts.py",
                            inspect.getfile(cache_utils), inspect.getfile(modeling_qwen3_vl), inspect.getfile(sdpa_attention))]
        captures = [preserve_source(p, run_root=OUT,
                                    relative_name=str(p) if not p.is_absolute() else f"transformers/{p.name}")
                    for p in dependency_paths]
        require(literal_binding(captures[12])["sha256"] == loader["sha256"],
                "current loader source capture changed")
        manifest = {"schema": "recurrence_phase_transfer.cells.v1", "status": "frozen_before_forward",
                    "source": state["source"], "producer": literal_binding(Path(__file__)),
                    "producer_capture": literal_binding(source_capture),
                    "dependency_captures": [literal_binding(p) for p in captures],
                    "transformers_version": transformers.__version__, "model_identity": identity,
                    "native_batch_identity": input_identity(batch),
                    "batch_index": TARGET, "raw_target_offset": OFFSET, "current_S": S,
                    "source_physical": list(SOURCE), "destination_physical": list(DEST),
                    "suffix_physical": [WIDTH, FULL_WIDTH],
                    "native_input_hashes": {k: tensor_hash(native[k]) for k in
                                            ("input_ids", "attention_mask", "position_ids")},
                    "suffix_input_hashes": {k: tensor_hash(v) for k, v in
                                            {"input_ids": native["input_ids"][:, WIDTH:FULL_WIDTH],
                                             "attention_mask": native["attention_mask"],
                                             "position_ids": native["position_ids"][:, :, WIDTH:FULL_WIDTH],
                                             "cache_position": torch.arange(WIDTH, FULL_WIDTH, device=device)}.items()},
                    "conditions": {"NN": ["new", "new"], "OO": ["old", "old"],
                                   "ON": ["old", "new"], "NO": ["new", "old"]},
                    "max_forwards": MAX_FORWARDS, "max_seconds": MAX_SECONDS,
                    "anchor_parity_atol": ATOL, "cached_NN_fullvector_atol": ATOL,
                    "winner_margin_guard": 0.001}
        write(OUT / "source-to-cell.json", manifest)
        state.update(status="executing", manifest=literal_binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)
        clock = time.monotonic()
        def count(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(state["model_forwards"] <= MAX_FORWARDS and time.monotonic() - clock <= MAX_SECONDS,
                    "model forward/time cap exceeded")
        def vision(*_):
            state["vision_forwards"] += 1
            require(state["vision_forwards"] <= 2, "unexpected vision forward")
        counters = [q.model.register_forward_pre_hook(count, with_kwargs=True),
                    q.model.model.visual.register_forward_pre_hook(vision)]
        def forward(tag, inputs):
            state["current_forward"] = tag
            begin = time.monotonic()
            try:
                output = q.model(**inputs)
            except BaseException:
                state["last_forward_seconds"] = time.monotonic() - begin
                write(OUT / "receipt.json", state)
                raise
            state["calls"].append({"tag": tag, "model_forward": state["model_forwards"],
                                   "vision_forwards_so_far": state["vision_forwards"],
                                   "seconds": time.monotonic() - begin})
            state.pop("current_forward", None)
            write(OUT / "receipt.json", state)
            return output
        try:
            with torch.inference_mode():
                anchor_seen, anchor_handles = hooks(q.model, native["input_ids"], native["position_ids"])
                try:
                    anchor_out = forward("full_history_anchor", native)
                finally:
                    for h in anchor_handles:
                        h.remove()
                anchor = anchor_out.logits[TARGET, -1].detach().float().cpu()
                torch.save(anchor, OUT / "native-anchor.pt")
                top = torch.topk(anchor, 2)
                saved_top = trace[OFFSET]["raw_top2"][TARGET]
                anchor_parity = {"argmax_token": int(top.indices[0]),
                                 "runnerup_token": int(top.indices[1]),
                                 "top2_max_abs_vs_trace": max(abs(float(top.values[j]) - saved_top[j]) for j in (0, 1)),
                                 "full_logits": literal_binding(OUT / "native-anchor.pt"),
                                 "consumed": anchor_seen}
                write(OUT / "anchor-readback.json", anchor_parity)
                state["anchor_readback"] = literal_binding(OUT / "anchor-readback.json")
                write(OUT / "receipt.json", state)
                require(anchor_parity["argmax_token"] == NEW and anchor_parity["runnerup_token"] == OLD and
                        anchor_parity["top2_max_abs_vs_trace"] <= ATOL and
                        len(anchor_seen["embedding_inputs"]) == len(anchor_seen["rotary_positions"]) == 1 and
                        len(anchor_seen["masks"]) == len(anchor_seen["cache_slots"]) == 2,
                        "full-history anchor differs from original trace/consumption")

                cache = DynamicCache()
                prefill = dict(native)
                prefill.update(input_ids=native["input_ids"][:, :WIDTH],
                               attention_mask=native["attention_mask"][:, :WIDTH],
                               position_ids=native["position_ids"][:, :, :WIDTH],
                               cache_position=torch.arange(WIDTH, device=device),
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                phase_seen, pre_norm = [], [dict() for _ in range(LAYERS)]
                handles = []
                def rotary(_module, _args, output):
                    cos, sin = output
                    require(cos.shape == sin.shape == (4, WIDTH, HEAD_DIM), "historical phase shape changed")
                    phase_seen.append({"old_cos": cos[TARGET, SOURCE[0]:SOURCE[1]].detach().clone(),
                                       "old_sin": sin[TARGET, SOURCE[0]:SOURCE[1]].detach().clone(),
                                       "new_cos": cos[TARGET, DEST[0]:DEST[1]].detach().clone(),
                                       "new_sin": sin[TARGET, DEST[0]:DEST[1]].detach().clone()})
                handles.append(q.model.model.language_model.rotary_emb.register_forward_hook(rotary))
                for i, layer in enumerate(q.model.model.language_model.layers):
                    def normalized(_module, _args, output, *, index=i):
                        require(output.shape == (4, WIDTH, KV_HEADS, HEAD_DIM), "historical pre-K shape changed")
                        pre_norm[index] = {"old": output[TARGET, SOURCE[0]:SOURCE[1]].transpose(0, 1).detach().clone(),
                                           "new": output[TARGET, DEST[0]:DEST[1]].transpose(0, 1).detach().clone()}
                    handles.append(layer.self_attn.k_norm.register_forward_hook(normalized))
                pre_seen, standard = hooks(q.model, prefill["input_ids"], prefill["position_ids"])
                handles.extend(standard)
                try:
                    prefill_out = forward("historical_prefill", prefill)
                finally:
                    for h in handles:
                        h.remove()
                require(prefill_out.past_key_values is cache and state["vision_forwards"] == 2 and
                        len(phase_seen) == 1 and all(set(x) == {"old", "new"} for x in pre_norm) and
                        len(pre_seen["embedding_inputs"]) == len(pre_seen["rotary_positions"]) == 1 and
                        len(pre_seen["masks"]) == len(pre_seen["cache_slots"]) == 2,
                        "native prefill/phase/pre-K capture invalid")
                check_cache(cache, width=WIDTH)
                native_digest = cache_digest(cache)
                native_blocks = block_hashes(cache)
                fixed_history = fixed_history_snapshot(cache)
                phase = phase_seen[0]
                candidates = {name: [] for name in ("NN", "OO", "ON", "NO")}
                phase_packet = {"actual_phase": {k: v.cpu() for k, v in phase.items()}, "layers": []}
                qualifications = []
                for i, layer in enumerate(cache.layers):
                    old_post = layer.keys[TARGET, :, SOURCE[0]:SOURCE[1], :]
                    new_post = layer.keys[TARGET, :, DEST[0]:DEST[1], :]
                    keys, errors = make_keys(pre_norm[i]["old"], pre_norm[i]["new"],
                                             phase["old_cos"], phase["old_sin"],
                                             phase["new_cos"], phase["new_sin"], old_post, new_post)
                    qualification = phase_qualification(pre_norm[i]["old"], pre_norm[i]["new"],
                                                        old_post, new_post, phase["new_cos"], phase["new_sin"],
                                                        phase["old_cos"], phase["old_sin"], keys, errors)
                    qualifications.append(qualification)
                    for name in candidates:
                        candidates[name].append(keys[name])
                    phase_packet["layers"].append({"pre_old": pre_norm[i]["old"].cpu(),
                                                   "pre_new": pre_norm[i]["new"].cpu(),
                                                   "post_old": old_post.cpu(), "post_new": new_post.cpu(),
                                                   "native_V_dest": layer.values[TARGET, :, DEST[0]:DEST[1], :].cpu(),
                                                   "candidate_keys": {name: keys[name].cpu() for name in candidates},
                                                   "qualification": qualification})
                torch.save(phase_packet, OUT / "phase-and-prekeys.pt")
                write(OUT / "prefill-readback.json", {"consumed": pre_seen, "cache_digest": native_digest,
                                                       "block_hashes": native_blocks,
                                                       "phase_and_prekeys": literal_binding(OUT / "phase-and-prekeys.pt"),
                                                       "phase_qualifications": qualifications})
                state["prefill_readback"] = literal_binding(OUT / "prefill-readback.json")
                write(OUT / "receipt.json", state)
                require(all(x["qualified"] for x in qualifications), "observed pre/post RoPE qualification failed")
                require(not (torch.equal(phase["old_cos"], phase["new_cos"]) and
                             torch.equal(phase["old_sin"], phase["new_sin"])),
                        "old/new phase unexpectedly identical")

                cells = {}
                for name in ("NN", "OO", "ON", "NO"):
                    with patch_key(cache, candidates[name], native_digest):
                        expected_blocks = block_hashes(cache)
                        require(all(expected_blocks[i]["source"] == native_blocks[i]["source"] and
                                    expected_blocks[i]["dest"]["values"] == native_blocks[i]["dest"]["values"] and
                                    expected_blocks[i]["dest"]["keys"] == tensor_hash(candidates[name][i])
                                    for i in range(LAYERS)), "historical K/V patch changed other source blocks")
                        suffix = {"input_ids": native["input_ids"][:, WIDTH:FULL_WIDTH],
                                  "attention_mask": native["attention_mask"],
                                  "position_ids": native["position_ids"][:, :, WIDTH:FULL_WIDTH],
                                  "cache_position": torch.arange(WIDTH, FULL_WIDTH, device=device),
                                  "past_key_values": cache, "use_cache": True, "return_dict": True,
                                  "logits_to_keep": SUFFIX}
                        consumed, standard = hooks(q.model, suffix["input_ids"], suffix["position_ids"])
                        layer_seen, actual_phase, observers = suffix_observers(q.model, cache, expected_blocks,
                                                                             fixed_history)
                        try:
                            output = forward(name, suffix)
                            logits = output.logits[TARGET, -1].detach().float().cpu()
                            companion_hash = tensor_hash(output.logits[[0, 2, 3], -1].detach())
                        finally:
                            for h in standard + observers:
                                h.remove()
                        require(torch.isfinite(logits).all() and cache.get_seq_length() == FULL_WIDTH and
                                state["vision_forwards"] == 2 and len(actual_phase) == 1 and
                                len(layer_seen) == LAYERS and
                                len(consumed["embedding_inputs"]) == len(consumed["rotary_positions"]) == 1 and
                                len(consumed["masks"]) == len(consumed["cache_slots"]) == 2,
                                "cached S consumption/output incomplete")
                        readback = {"consumed": consumed, "layers": layer_seen,
                                    "actual_S_phase_hashes": [tensor_hash(x) for x in actual_phase[0]],
                                    "companion_final_logits_hash": companion_hash,
                                    "cache_length_after": cache.get_seq_length()}
                        if name != "NN":
                            base = cells["NN"]["readback"]
                            require(consumed == base["consumed"] and
                                    readback["actual_S_phase_hashes"] == base["actual_S_phase_hashes"] and
                                    companion_hash == base["companion_final_logits_hash"] and
                                    all(layer_seen[i]["mask_hash"] == base["layers"][i]["mask_hash"] and
                                        layer_seen[i]["phase_hashes"] == base["layers"][i]["phase_hashes"] and
                                        layer_seen[i]["blocks"]["source"] == base["layers"][i]["blocks"]["source"] and
                                        layer_seen[i]["blocks"]["dest"]["values"] ==
                                        base["layers"][i]["blocks"]["dest"]["values"] for i in range(LAYERS)),
                                    "cell changed suffix mask/phase/companions/native V")
                        readback_path, vector_path = OUT / f"{name}-readback.json", OUT / f"{name}.pt"
                        write(readback_path, readback)
                        torch.save(logits, vector_path)
                        top = torch.topk(logits, 5)
                        cells[name] = {"full_logits": literal_binding(vector_path),
                                       "readback_binding": literal_binding(readback_path), "readback": readback,
                                       "argmax_token": int(top.indices[0]),
                                       "top1_top2_gap": float(top.values[0] - top.values[1]),
                                       "top5": [{"token": int(t), "logit": float(v)} for t, v in
                                                zip(top.indices, top.values, strict=True)],
                                       "z350_minus_z348": float(logits[OLD] - logits[NEW]),
                                       "absolute_coordinates": probability_readback(logits, (348, 350))}
                        state["last_cell"] = name
                        write(OUT / "partial-results.json", {"status": "running", "cells": cells})
                        write(OUT / "receipt.json", state)
                    if name == "NN":
                        cells[name]["max_abs_vs_full_history"] = float((logits - anchor).abs().max())
                        require(cells[name]["max_abs_vs_full_history"] <= ATOL and
                                cells[name]["argmax_token"] == NEW,
                                "cached NN full-vocabulary anchor parity failed")
                require(state["model_forwards"] == 6 and state["vision_forwards"] == 2 and
                        time.monotonic() - clock <= MAX_SECONDS, "frozen six-call plan incomplete")
                d = {name: cell["z350_minus_z348"] for name, cell in cells.items()}
                contrast = {"phase_new_minus_old_old_prekey": d["ON"] - d["OO"],
                            "phase_new_minus_old_new_prekey": d["NN"] - d["NO"],
                            "old_minus_new_prekey_old_phase": d["OO"] - d["NO"],
                            "old_minus_new_prekey_new_phase": d["ON"] - d["NN"],
                            "factorial_interaction": d["ON"] - d["OO"] - d["NN"] + d["NO"]}
                predicted = {"NN": NEW, "ON": NEW, "OO": OLD, "NO": OLD}
                full_transfer = all(cells[name]["argmax_token"] == token and
                                    cells[name]["top1_top2_gap"] > .001 and abs(d[name]) > .001
                                    for name, token in predicted.items())
                result = {"schema": "recurrence_phase_transfer.result.v1", "status": "candidate",
                          "cells": cells, "anchor": anchor_parity, "contrast": contrast,
                          "full_directional_transfer": full_transfer,
                          "phase_and_prekeys": literal_binding(OUT / "phase-and-prekeys.pt"),
                          "prefill_readback": literal_binding(OUT / "prefill-readback.json"),
                          "source_manifest": literal_binding(OUT / "source-to-cell.json"),
                          "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                          "model_seconds": time.monotonic() - clock}
                write(OUT / "result.json", result)
                state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"),
                             model_seconds=result["model_seconds"], elapsed_seconds=time.monotonic() - started,
                             peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                write(OUT / "receipt.json", state)
                print(json.dumps({"status": state["status"], "full_transfer": full_transfer,
                                  "forwards": state["model_forwards"], "margins": d}))
        finally:
            for h in counters:
                h.remove()
    except BaseException as error:
        state.update(status="technical_invalid", error=repr(error), elapsed_seconds=time.monotonic() - started)
        write(OUT / "receipt.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(); print("selfcheck ok")
    else:
        run(args.device)
