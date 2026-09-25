"""Two bounded non-model device-boundary checks for attempt 002."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from probes.training_set_completion.recurrence_history_read_stage import run as r


RULE = r.UNIT / "lead-repair-attempt-002.json"
OLD_SHA = "9577b772377baa87703ed503c46a65f730315e929a973083cf0569e5d93f30a6"


class DeviceRope(r.base.ConfigOnlyRope):
    def __init__(self, device):
        super().__init__()
        self.anchor = torch.empty(0, device=device)

    def parameters(self):
        yield self.anchor


def fixture(a, q, device, arm, t):
    batch, raw, _, source, _ = r.source(q, a, device)
    emitted = list(r.old.ROW1[:t]) if t <= 9 else [151646]+[8987]*(t-1)
    model = DeviceRope(device)
    full = r.step_inputs(model, batch, raw, int(q.tokenizer.pad_token_id),
                         a, arm, emitted, previous=emitted[-1] if t else None)
    mask, native, selected = r.mask_for(full, a, arm)
    actual = native if r.spec(a, arm)["mask_mode"] == "native" else mask
    chosen = 151646
    logits = torch.zeros((4, 152670));logits[r.TARGET, chosen] = 1
    seen = {"expected_mask":actual,"layers":list(range(28)),
            "layer_mask_hashes":[r.base.tensor_hash(actual)]*28}
    entry = r.entry_for(full,mask,native,selected,seen,chosen,t,Path(__file__),Path(__file__))
    stored = {k:full[k].detach().cpu() for k in r.KEYS}
    payload = {"arm":arm,"step":t,"actual_mask":actual.detach().cpu(),"logits":logits}
    return model,batch,raw,full,entry,stored,payload,selected


def reject(call,label):
    try:call()
    except ValueError as exc:
        assert str(exc)=="serialized mask/input/own prefix/vector differs", (label,str(exc))
        return label
    raise AssertionError(f"actual serialized reader accepted {label}")


def child(phase):
    rule=r.bind(RULE)
    assert rule["sha256"]=="ac8d04b9a85e5dd8c7e74b6153126555d40b3c48889651feef9178702363f1bc"
    takeover=r.bind(r.UNIT / "lead-device-takeover-v3.json")
    assert takeover["sha256"]=="eb95a7ec41dc1979e2be94076db71d2a901d83491601497d4acf7d787779fb5e"
    source=r.bind(Path(r.__file__))
    if phase=="red":assert source["sha256"]==OLD_SHA
    a,_,_=r.contract()
    q=r.base.load_qwen_components_from_options(r.base.QwenLoadOptions(
        base_model=str(r.base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    assert q.model is None
    torch.cuda.set_device(0)
    results=[]
    if phase=="red":
        model,batch,raw,full,entry,stored,payload,selected=fixture(a,q,torch.device("cuda:0"),"native_A",0)
        pad=int(q.tokenizer.pad_token_id)
        mask,native,_=r.mask_for(full,a,"native_A")
        actual=native
        assert all(full[k].device.type=="cuda" and stored[k].device.type=="cpu" for k in r.KEYS)
        assert actual.device.type=="cuda" and payload["actual_mask"].device.type=="cpu"
        try:r.verify_serialized(model,batch,raw,pad,a,"native_A",[],entry,stored,payload)
        except RuntimeError as exc:
            first=traceback.format_exc()
            assert "same device" in str(exc) and "in verify_serialized" in first
            assert "torch.equal(stored[k],expected[k])" in first
        else:raise AssertionError("old input comparison did not fail")
        cuda_stored={k:full[k] for k in r.KEYS}
        assert all(torch.equal(cuda_stored[k],full[k]) for k in r.KEYS)
        try:r.verify_serialized(model,batch,raw,pad,a,"native_A",[],entry,cuda_stored,payload)
        except RuntimeError as exc:
            second=traceback.format_exc()
            assert "same device" in str(exc) and "in verify_serialized" in second
            assert 'torch.equal(payload["actual_mask"],actual)' in second
        else:raise AssertionError("old saved-mask comparison did not fail")
        results=[{"case":"native_A_t0","cpu_saved_cuda_expected_input_error":first,
                  "cuda_saved_input_cpu_saved_mask_error":second,
                  "reconstructed_devices":{k:str(full[k].device) for k in r.KEYS},
                  "actual_mask_device":str(actual.device),
                  "serialized_devices":{k:str(stored[k].device) for k in r.KEYS},
                  "saved_mask_device":str(payload["actual_mask"].device)}]
    else:
        for arm,t in (("native_A",0),("header_A",5),("coordinate_F",5),
                      ("header_F",15),("coordinate_A",15)):
            model,batch,raw,full,entry,stored,payload,selected=fixture(a,q,torch.device("cuda:0"),arm,t)
            pad=int(q.tokenizer.pad_token_id)
            mask,native,_=r.mask_for(full,a,arm)
            actual=native if r.spec(a,arm)["mask_mode"]=="native" else mask
            assert all(full[k].device.type=="cuda" and stored[k].device.type=="cpu" for k in r.KEYS)
            assert actual.device.type=="cuda" and payload["actual_mask"].device.type=="cpu"
            returned=r.verify_serialized(model,batch,raw,pad,a,arm,
                list(r.old.ROW1[:t]) if t<=9 else [151646]+[8987]*(t-1),entry,stored,payload)
            assert returned["input_ids"].device.type=="cuda" and full["input_ids"].device.type=="cuda"
            assert stored["input_ids"].device.type==payload["actual_mask"].device.type=="cpu"
            failures=[]
            def check(label,badstored=stored,badpayload=payload):
                failures.append(reject(lambda:r.verify_serialized(model,batch,raw,pad,a,arm,
                    list(r.old.ROW1[:t]) if t<=9 else [151646]+[8987]*(t-1),
                    entry,badstored,badpayload),label))
            for label,key,index in (("history","input_ids",(r.TARGET,r.PROMPT+5)),
                                    ("companion","input_ids",(0,r.PROMPT)),
                                    ("position","position_ids",(0,r.TARGET,r.PROMPT)),
                                    ("own_prefix","input_ids",(r.TARGET,-1))):
                changed={**stored,key:stored[key].clone()};changed[key][index]+=1
                check(label,badstored=changed)
            complement=payload["actual_mask"].clone();complement[0,0,-1,0]=~complement[0,0,-1,0]
            check("complement",badpayload={**payload,"actual_mask":complement})
            if t:
                current=payload["actual_mask"].clone();current[r.TARGET,0,-1,-1]=False
                check("current_to_current",badpayload={**payload,"actual_mask":current})
            if selected.any():
                changed=payload["actual_mask"].clone();changed[selected.cpu()]=True
                check("selected",badpayload={**payload,"actual_mask":changed})
            if arm.startswith(("header_","coordinate_")):
                other="coordinate_" if arm.startswith("header_") else "header_"
                wrong=r.mask_for(full,a,other+r.spec(a,arm)["history"])[0].detach().cpu()
                check("wrong_rectangle",badpayload={**payload,"actual_mask":wrong})
            results.append({"case":f"{arm}_t{t}","selected_cells":int(selected.sum()),
                            "input_device":str(full["input_ids"].device),
                            "saved_devices":[str(stored["input_ids"].device),str(payload["actual_mask"].device)],
                            "mutations_rejected":failures})
        # The production cold reader reconstructs both sides on CPU.
        model,batch,raw,full,entry,stored,payload,selected=fixture(a,q,torch.device("cpu"),"coordinate_F",5)
        returned=r.verify_serialized(model,batch,raw,int(q.tokenizer.pad_token_id),
                                     a,"coordinate_F",list(r.old.ROW1[:5]),entry,stored,payload)
        assert returned["input_ids"].device.type=="cpu"
        results.append({"case":"coordinate_F_t5_cold_cpu","selected_cells":int(selected.sum()),"passed":True})
    torch.cuda.synchronize(0)
    return {"phase":phase,"producer":source,"fixture":r.bind(Path(__file__)),"ruling":rule,"takeover":takeover,
            "source_input_identity":a["source_bindings"]["input_identity_sha256"],
            "cases":results,"cuda_used":True,"model_loaded":False,
            "model_forwards":0,"vision_forwards":0,"generated_tokens":0}


def main():
    arg=argparse.ArgumentParser();arg.add_argument("phase",choices=("red","green","red-child","green-child"))
    phase=arg.parse_args().phase
    if phase.endswith("-child"):
        print(json.dumps(child(phase.removesuffix("-child"))))
        return
    output=r.UNIT / "supporting" / f"attempt-002-{phase}-v3.json"
    assert not output.exists()
    command=[sys.executable,"-B","-m","probes.training_set_completion.recurrence_history_read_stage.boundary_fixture",phase+"-child"]
    started=time.monotonic();run=subprocess.run(command,text=True,capture_output=True,check=False)
    packet={"phase":phase,"command":command,"outer_seconds":time.monotonic()-started,
            "exit_code":run.returncode,"terminal":True,"stderr":run.stderr,
            "child_result":json.loads(run.stdout.splitlines()[-1]) if run.returncode==0 else None}
    output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps({"phase":phase,"exit_code":run.returncode,"outer_seconds":packet["outer_seconds"],
                      "cases":len(packet["child_result"]["cases"]) if packet["child_result"] else 0,
                      "receipt":str(output)}))
    if run.returncode:raise RuntimeError("device fixture failed; see receipt; stop")


if __name__=="__main__":main()
