"""Independent CPU check of saved original-batch inputs and own greedy histories."""

import json
from pathlib import Path

import torch

from src.qwen.saved_prefix import prefix_tokens as _prefix_tokens
from probes.recurrence_dynamics.recurrence_native_row_completion.run import ADMISSION, ARMS, NATIVE, OFFSET, OUT, TARGET, UNIT, bind, require, write_new
from probes.recurrence_dynamics.coordinate_continuity.runtime import tensor_hash


def main():
    dest = UNIT / "supporting/first-case-cold-source-v1.json"
    require(not dest.exists(), "cold source record exists")
    a = json.loads(ADMISSION.read_text())
    receipt = json.loads((OUT/"receipt.json").read_text())
    prior = json.loads((OUT/"readback.json").read_text())
    pre = json.loads((UNIT/"supporting/first-case-preflight-v1.json").read_text())
    require(receipt["status"] == "candidate_complete" and
            prior["status"] == "candidate_cold_readback_passed" and
            receipt["preflight"] == bind(UNIT/"supporting/first-case-preflight-v1.json"),
            "bound attempt/readback changed")
    raw_path = Path(a["source_bindings"]["raw"]["path"])
    source_receipt_path = Path(a["source_bindings"]["runtime_receipt"]["path"])
    require(bind(raw_path)["sha256"] == a["source_bindings"]["raw"]["sha256"] and
            bind(source_receipt_path)["sha256"] == a["source_bindings"]["runtime_receipt"]["sha256"],
            "original source changed")
    raw = json.loads(raw_path.read_text())["rows"]
    source_receipt = json.loads(source_receipt_path.read_text())
    prompts = source_receipt["input_identity"]["prompt_token_ids"]
    require(pre["input_identity"] == source_receipt["input_identity"] and len(prompts) == 4,
            "full source request identity changed")
    pad = 151643
    cases = []
    for t in range(5):
        native_inputs = None
        for arm_rec in receipt["arms"]:
            arm = arm_rec["arm"]
            if t >= len(arm_rec["steps"]):
                continue
            entry = arm_rec["steps"][t]
            path = Path(entry["inputs"]["path"])
            require(bind(path) == entry["inputs"], "saved input bytes changed")
            saved = json.loads(path.read_text())
            ids = torch.tensor(saved["input_ids"])
            mask = torch.tensor(saved["attention_mask"])
            pos = torch.tensor(saved["position_ids"])
            cache = torch.tensor(saved["cache_position"])
            width = 1387+t
            require(ids.shape == mask.shape == (4,width) and pos.shape == (3,4,width) and
                    cache.tolist() == list(range(width)), "saved shape/positions changed")
            tails = _prefix_tokens(raw, OFFSET+t, pad)
            tails[TARGET] = raw[TARGET]["token_ids"][:OFFSET] + arm_rec["emitted"][:t]
            for i,(prompt,tail) in enumerate(zip(prompts,tails,strict=True)):
                history = prompt+tail
                left = width-len(history)
                require(left >= 0 and ids[i].tolist() == [pad]*left+history and
                        mask[i].tolist() == [0]*left+[1]*len(history),
                        f"original full-batch/own-history mismatch: {arm}:{t}:{i}")
            require(tails[2][10] == 151645 and tails[2][11:] == [pad]*(14+t) and
                    ids[TARGET, 1362+OFFSET:1362+OFFSET+t].tolist() == arm_rec["emitted"][:t],
                    "ended companion or actual previous greedy token changed")
            hashes = {name:tensor_hash(x) for name,x in
                      (("input_ids",ids),("attention_mask",mask),
                       ("position_ids",pos),("cache_position",cache))}
            if arm == "native":
                native_inputs = (ids,mask,pos,cache)
                require(hashes == pre["native_source_hashes"][t] and
                        arm_rec["emitted"][:t] == NATIVE[:t],
                        "native source prefix changed")
            else:
                require(native_inputs is not None and
                        torch.equal(mask,native_inputs[1]) and
                        torch.equal(pos,native_inputs[2]) and
                        torch.equal(cache,native_inputs[3]) and
                        torch.equal(ids[[0,1,2]],native_inputs[0][[0,1,2]]) and
                        torch.equal(ids[TARGET,:1362+OFFSET],native_inputs[0][TARGET,:1362+OFFSET]),
                        "companion/historical content or positions changed")
                if arm == "identity-mask-sham":
                    require(torch.equal(ids,native_inputs[0]), "identity sham own history changed")
            cases.append({"arm":arm,"step":t,"input":entry["inputs"],"input_hashes":hashes,
                          "ended_companion_pad_count":14+t,
                          "consumed_own_previous_tokens":arm_rec["emitted"][:t]})
    require(len(cases) == receipt["counts"]["model_forwards"] == 15 and
            [x["arm"] for x in receipt["arms"]] == list(ARMS), "saved-step denominator changed")
    result = {"schema":"recurrence_native_row_completion.cold_source.v1",
              "status":"independent_cpu_source_passed",
              "admission":bind(ADMISSION),"receipt":bind(OUT/"receipt.json"),
              "readback":bind(OUT/"readback.json"),
              "original_raw":bind(raw_path),"original_runtime_receipt":bind(source_receipt_path),
              "checks":cases,"model_loads":0,"vision_forwards":0,"GPU_seconds":0}
    write_new(dest,result)
    print(json.dumps({"status":result["status"],"checked_steps":len(cases)}))


if __name__ == "__main__":
    main()
