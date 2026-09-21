"""Lane-A original-policy natural producer, reusing the qualified native path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from probes.training_set_completion import untied_shared as shared
from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from src.data.examples import raw_example_from_jsonl_row
from src.inference.bound_requests import build_bound_native_requests
from src.inference.inputs import plan_examples
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.native import prepare_native_inputs


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census")
EOS = 151645
CAP = 3084


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _prefix_hash(row: list[int]) -> str:
    return hashlib.sha256(json.dumps(row, separators=(",", ":")).encode()).hexdigest()


def run(condition: str, keys: list[str], device: str, qualify: bool = False) -> None:
    # load_model/config_for resolve their panel through untied_shared globals.
    shared.ROOT = OUT
    model_key, policy = condition.split("-", 1)
    assert policy == "original"
    panel = json.loads((OUT / "panel.json").read_text())
    torch_device = torch.device(device)
    started = time.monotonic()
    q, identity = shared.load_model(model_key, torch_device)
    m = q.model
    e = m.get_input_embeddings()
    h = m.get_output_embeddings()
    ids = h.selected_token_ids
    coords = ids[4:]
    assert coords.tolist() == list(range(151670, 152670))
    E = e(ids).detach()
    U = h.base.weight[ids].detach() + h.shared_embed_delta.detach()
    norms = U[4:].double().norm(dim=1)
    factors = norms.median() / norms
    weight_dir = OUT / "weights" / condition / keys[0]
    weight_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_rows": E.cpu(),
            "output_rows": U.cpu(),
            "input_delta": e.shared_embed_delta.detach().cpu(),
            "output_delta": h.shared_embed_delta.detach().cpu(),
            "base_rows": h.base.weight[ids].detach().cpu(),
            "selected_ids": ids.cpu(),
            "factors": factors.cpu(),
            "final_norm": m.model.language_model.norm.weight.detach().cpu(),
        },
        weight_dir / "weights.pt",
    )
    write(weight_dir / "identity.json", identity)

    for key in keys:
        group = next(g for g in panel["groups"] if g["key"] == key)
        out = OUT / "runtime" / condition / key
        out.mkdir(parents=True, exist_ok=False)
        receipt = {
            "status": "running",
            "pid": os.getpid(),
            "condition": condition,
            "model": model_key,
            "policy": "original",
            "device": device,
            "group": key,
            "model_forwards": 0,
            "vision_forwards": 0,
            "panel": _binding(OUT / "panel.json"),
            "producer": _binding(Path(__file__)),
            "identity": identity,
            "contract": panel["generation_contract"],
        }
        write(out / "receipt.json", receipt)
        group_started = time.monotonic()
        hooks = []
        try:
            config = dict(panel["configs"][model_key])
            config["data"] = {"input_jsonl": group["input_jsonl"]}
            # Recreate and bind the true native image/prompt plan at this entry;
            # the selected cohort was frozen before weights were loaded.
            config_model = shared.config_for(model_key)
            raw_examples = [
                raw_example_from_jsonl_row(
                    case["input_record"],
                    jsonl_path=Path(group["input_jsonl"]),
                    row_number=int(case["row_index"]) + 1,
                    raw_line=json.dumps(case["input_record"]),
                )
                for case in group["cases"]
            ]
            planned = plan_examples(raw_examples, config=config_model, components=q, row_indices=[int(c["row_index"]) for c in group["cases"]])
            for case, item in zip(group["cases"], planned, strict=True):
                case["image_path"] = item.image.image_path
                case["image_plan"] = item.image.to_artifact_dict()
            requests, _ = build_bound_native_requests(q, config, group["cases"])
            batch = prepare_native_inputs(q.processor, requests, device=torch_device, record_media_identity=True)
            receipt["input_identity"] = _input_identity(batch)
            width = batch.inputs["input_ids"].shape[1]
            traces: list[dict[str, object]] = []
            last: dict[str, torch.Tensor] = {}

            def count(_module, _args, _kwargs):
                receipt["model_forwards"] += 1
                if receipt["model_forwards"] > CAP + 220:
                    raise RuntimeError("native forward budget exceeded")

            def vision(_module, _args):
                receipt["vision_forwards"] += 1

            def capture(_module, args):
                last["h"] = args[0].detach()

            hooks = [
                m.register_forward_pre_hook(count, with_kwargs=True),
                m.model.visual.register_forward_pre_hook(vision),
                h.register_forward_pre_hook(capture),
            ]

            class OriginalPolicy(LogitsProcessor):
                def __call__(self, tokens, scores):
                    chosen = scores.argmax(-1)
                    top = scores.topk(2)
                    traces.append(
                        {
                            "offset": tokens.shape[1] - width,
                            "raw_winners": top.indices[:, 0].tolist(),
                            "raw_top2": top.values.tolist(),
                            "raw_runnerups": top.indices[:, 1].tolist(),
                            "chosen": chosen.tolist(),
                            "eos_logits": scores[:, EOS].tolist(),
                            "logsumexp": scores.logsumexp(-1).tolist(),
                            "chosen_raw_logits": scores.gather(1, chosen[:, None]).squeeze(1).tolist(),
                        }
                    )
                    return scores

            def generate(budget: int = CAP, instrument: bool = True):
                original = m.generate

                def wrapped(**kwargs):
                    processors = [OriginalPolicy()] if instrument else []
                    return original(**kwargs, logits_processor=LogitsProcessorList(processors))

                m.generate = wrapped
                try:
                    return generate_continuations(
                        m,
                        batch,
                        extensions=[[] for _ in requests],
                        budgets=[budget for _ in requests],
                        eos_token_id=EOS,
                        pad_token_id=q.tokenizer.pad_token_id,
                        policy=NativeGenerationPolicy(
                            temperature=0,
                            top_p=1,
                            top_k=0,
                            repetition_penalty=1,
                            use_model_defaults=False,
                        ),
                        trace="none",
                        seed=None,
                    )
                finally:
                    m.generate = original

            with torch.inference_mode():
                if qualify:
                    logits = m(**batch.inputs, use_cache=False, logits_to_keep=1).logits.detach()
                    hidden = last["h"]
                    computed = hidden @ U.T
                    assert torch.allclose(logits[:, :, ids], computed, atol=0.0002, rtol=1e-5)
                    explicit = hidden.double() @ (U[4:].double() * factors[:, None]).T
                    assert torch.allclose(explicit, logits[:, :, coords].double() * factors, atol=0.0002, rtol=1e-5)
                    ones = torch.ones_like(factors)
                    assert torch.equal((logits[:, :, coords].double() * ones).to(logits.dtype), logits[:, :, coords])
                    before_e = e(ids).clone()
                    before_u = h(hidden).clone()
                    input_delta = e.shared_embed_delta
                    output_delta = h.shared_embed_delta
                    original_in = input_delta[4, 0].item()
                    original_out = output_delta[5, 0].item()
                    input_delta[4, 0] += 0.01
                    changed_e = e(ids)
                    changed_u = h(hidden)
                    assert not torch.equal(changed_e, before_e)
                    if model_key == "untied":
                        assert torch.equal(changed_u, before_u)
                    input_delta[4, 0] = original_in
                    output_delta[5, 0] += 0.01
                    assert not torch.equal(h(hidden), before_u)
                    if model_key == "untied":
                        assert torch.equal(e(ids), before_e)
                    output_delta[5, 0] = original_out
                    assert torch.equal(e(ids), before_e) and torch.equal(h(hidden), before_u)
                    plain = generate(16, False)
                    traces.clear()
                    identity_run = generate(16, True)
                    assert [list(v.token_ids) for v in plain] == [list(v.token_ids) for v in identity_run]
                    traces.clear()
                    receipt["qualification"] = {
                        "effective_row_reconstruction": True,
                        "independent_delta_paths": model_key == "untied",
                        "temporary_values_exactly_restored": True,
                        "no_op_tokens_exact": True,
                        "identity_coefficients_exact": True,
                        "max_abs": float((logits[:, :, ids] - computed).abs().max()),
                    }
                values = generate(CAP, True)
            rows = []
            prompt_ids = batch.prompt_token_ids
            for i, (case, result) in enumerate(zip(group["cases"], values)):
                rows.append(
                    {
                        "image_id": int(case["input_record"]["image_id"]),
                        "split": str(case["input_record"]["metadata"]["split"]),
                        "row_id": case["row_id"],
                        "source_row": case["row_index"],
                        "prefix_hash": _prefix_hash(list(prompt_ids[i])),
                        "token_ids": list(result.token_ids),
                        "text": q.tokenizer.decode(list(result.token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False),
                        "stop": result.stop_reason,
                    }
                )
            assert {r["image_id"] for r in rows} == {int(c["input_record"]["image_id"]) for c in group["cases"]}
            assert receipt["model_forwards"] <= CAP + 220
            write(out / "raw.json", {"rows": rows})
            write(out / "trace.json", {"steps": traces})
            readback = json.loads((out / "raw.json").read_text())
            assert readback["rows"] == rows
            receipt.update(
                status="candidate_complete",
                raw=_binding(out / "raw.json"),
                trace=_binding(out / "trace.json"),
                elapsed_seconds=time.monotonic() - group_started,
                active_tokens=sum(len(r["token_ids"]) for r in rows),
                padded_token_work=len(traces) * len(rows),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(torch_device),
                no_parameter_mutation=True,
            )
            write(out / "receipt.json", receipt)
        except BaseException as exc:
            receipt.update(status="technical_invalid", error=repr(exc), elapsed_seconds=time.monotonic() - group_started)
            write(out / "receipt.json", receipt)
            raise
        finally:
            for hook in hooks:
                hook.remove()
        qualify = False
    write(OUT / f"worker-{condition}-{keys[0]}.json", {"status": "complete", "pid": os.getpid(), "condition": condition, "device": device, "groups": keys, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    run(args.condition, args.groups, args.device, args.qualify)
