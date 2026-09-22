"""Exact-history EOS versus row-open teacher-forced score diagnostic for image 210457."""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, resource, signal, time, traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

EOS, ROW_OPEN = 151645, 151646
SCHEMA = "training_set_completion.image210457_stop_logits.v1"

def require(c: bool, m: str) -> None:
    if not c: raise ValueError(m)
def canonical(v: Any) -> bytes: return (json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
def digest(v: Any) -> str: return hashlib.sha256(canonical(v)).hexdigest()
def file_hash(path: str|Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def binding(path: str|Path) -> dict[str,Any]:
    p=Path(path).resolve(strict=True); require(p.is_file(),f"not file: {p}")
    return {"path":str(p),"sha256":file_hash(p),"size_bytes":p.stat().st_size}
def tree_binding(path: str|Path) -> dict[str,Any]:
    p=Path(path).resolve(strict=True); require(p.is_dir(),f"not dir: {p}")
    fs=[{"relative_path":str(q.relative_to(p)),"sha256":file_hash(q),"size_bytes":q.stat().st_size} for q in sorted(p.rglob("*")) if q.is_file()]
    require(bool(fs),f"empty dir: {p}")
    return {"root":str(p),"file_count":len(fs),"files":fs,"fingerprint":digest(fs)}
def publish(path: str|Path,v: Any)->None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);require(not p.exists(),f"refusing overwrite: {p}")
    data=canonical(v)
    with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    require(p.read_bytes()==data,f"readback: {p}")
def save_npy(path:Path,a:np.ndarray)->None:
    path.parent.mkdir(parents=True,exist_ok=True);require(not path.exists(),f"refusing overwrite: {path}")
    with path.open("xb") as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
def verify_manifest(m:Mapping[str,Any],path:Path)->dict[str,Any]:
    require(m.get("schema")==SCHEMA and m.get("status")=="frozen_ready","manifest schema/status")
    require(m.get("content_sha256")==digest({k:v for k,v in m.items() if k!="content_sha256"}),"manifest content")
    for b in m["source_bindings"].values(): require(binding(b["path"])==b,f"source changed: {b['path']}")
    for spec in m["models"]:
        require(tree_binding(spec["adapter"]["root"])==spec["adapter"],f"adapter changed: {spec['name']}")
    require([h["temperature"] for h in m["histories"]]==[0.0,0.1,0.3,0.7],"history order")
    for h in m["histories"]:
        require(len(h["token_ids_0_47"])==47 and digest(h["token_ids_0_47"])==h["history_sha256"],"history identity")
        require(EOS not in h["token_ids_0_47"],"historical EOS")
    require(m["targets"]=={"candidate_target_appended_for_scoring":EOS,"eos_token_id":EOS,"row_open_token_id":ROW_OPEN},"targets")
    require(m["execution"]=={"physical_gpu":2,"cuda_visible_devices":"2","max_wall_seconds":600,"expected_model_loads":2,"expected_teacher_forced_forwards":8,"expected_image_forwards":8,"gradients":"forbidden"},"execution")
    require(Path(path).resolve()==Path(m["manifest_path"]).resolve(),"manifest path")
    return dict(m)
def score_array(values:np.ndarray)->dict[str,Any]:
    require(values.ndim==1 and values.shape[0]>ROW_OPEN and np.isfinite(values).all(),"next logits")
    top=int(np.argmax(values));mx=float(np.max(values));lse=mx+math.log(float(np.exp(values.astype(np.float64)-mx).sum()))
    return {"eos_logit":float(values[EOS]),"row_open_logit":float(values[ROW_OPEN]),"eos_logprob":float(values[EOS])-lse,"row_open_logprob":float(values[ROW_OPEN])-lse,"margin_eos_minus_row_open":float(values[EOS]-values[ROW_OPEN]),"top_next_id":top,"top_next_logit":float(values[top]),"top_next_logprob":float(values[top])-lse}
def run(manifest_path:Path)->None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.native_owner_scale.evaluation import _candidate_materialized_case
    from probes.source_rweak_row_cross.run import build_requests
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.native import prepare_native_inputs,prepare_replay
    root=manifest_path.resolve().parent;m=verify_manifest(json.loads(manifest_path.read_text()),manifest_path)
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="2" and torch.cuda.device_count()==1,"GPU2-only visibility")
    terminal_path=root/"terminal.json";require(not terminal_path.exists(),"terminal exists")
    started=time.monotonic();old=signal.getsignal(signal.SIGALRM)
    counters={"model_loads":0,"teacher_forced_forwards":0,"image_forwards":0,"scored_histories":0,"gradient_enabled_forwards":0}
    result_rows=[];phase="preflight";peak_alloc=peak_reserved=0
    try:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("600 second diagnostic wall bound")));signal.alarm(600)
        torch.cuda.set_device(torch.device("cuda:0")); torch.cuda.reset_peak_memory_stats()
        config0=InferConfig.model_validate(m["model_config"]); record=m["record"]
        for model_spec in m["models"]:
            phase=f"load_{model_spec['name']}"
            observed=inspect_dora_adapter_payload(model_spec["adapter"]["root"],config0.model.base_model)
            require(digest(observed)==model_spec["adapter_inspection_sha256"],"adapter inspection identity")
            cfg=checkpoint_config(config0,model_spec["adapter"]["root"])
            qwen,identity=load_policy(cfg,device=torch.device("cuda:0"));counters["model_loads"]+=1
            require(identity["model_identity"]["adapter"]["adapter_path"]==model_spec["adapter"]["root"] and identity["model_identity"]["adapter"]["merged_adapters"]==[],"live adapter")
            require(identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"]==["torch.float32"] and identity["effective_settings"]["observed_attn_implementation"]=="sdpa","FP32 SDPA")
            require(qwen.token_identity.im_end_token_ids==(EOS,) and qwen.tokenizer.decode([ROW_OPEN],skip_special_tokens=False)=="<|object_ref_start|>","target token identity")
            publish(root/f"model-{model_spec['name']}.json",identity);publish(root/f"config-{model_spec['name']}.json",cfg.model_dump(mode="json"))
            model=qwen.model;model.eval()
            for p in model.parameters():p.requires_grad_(False);p.grad=None
            require(sum(int(p.requires_grad) for p in model.parameters())==0,"requires-grad surface")
            handles=[model.register_forward_pre_hook(lambda *_:counters.__setitem__("teacher_forced_forwards",counters["teacher_forced_forwards"]+1))]
            visual=[mod for name,mod in model.named_modules() if name.endswith("visual")];require(len(visual)==1,"visual module identity")
            handles.append(visual[0].register_forward_pre_hook(lambda *_:counters.__setitem__("image_forwards",counters["image_forwards"]+1)))
            case=_candidate_materialized_case(record["case"],m["model_config"])
            requests,_=build_requests(qwen,m["model_config"],[case]);require(len(requests)==1,"one native request")
            batch=prepare_native_inputs(qwen.processor,requests,device="cuda:0",record_media_identity=True)
            require(list(batch.prompt_token_ids[0])==record["prompt_token_ids"],"live prompt identity")
            require(batch.media_sha256[0]==record["executed_media_sha256"] and list(batch.image_grids[0])==record["observed_image_grid_thw"],"live media/grid identity")
            for h in m["histories"]:
                phase=f"score_{model_spec['name']}_{h['request_id']}";continuation=list(h["token_ids_0_47"])+[EOS]
                tick=time.monotonic()
                with torch.inference_mode():
                    require(not torch.is_grad_enabled(),"grad enabled inside forward")
                    replay=prepare_replay(model,batch.inputs,prompt_token_ids=record["prompt_token_ids"],continuation_token_ids=continuation)
                    require(replay.target_ids.tolist()==continuation and replay.target_ids[:-1].tolist()==h["token_ids_0_47"] and int(replay.target_ids[-1])==EOS,"replay history/target identity")
                    aligned=replay.aligned_logits(model(**replay.inputs).logits);require(aligned.shape[0]==48,"aligned length")
                    values=aligned[-1].detach().float().cpu().numpy().copy();del aligned,replay
                torch.cuda.synchronize();require(all(p.grad is None for p in model.parameters()),"gradient materialized")
                fname=f"next-logits-{model_spec['name']}-{h['request_id'].split(':')[-1]}.npy";save_npy(root/"logits"/fname,values)
                s=score_array(values);result_rows.append({"model":model_spec["name"],"request_id":h["request_id"],"temperature":h["temperature"],"history_sha256":h["history_sha256"],"historical_token_span":[0,47],"appended_scoring_target_id":EOS,"next_logit_array":binding(root/"logits"/fname),"elapsed_seconds":time.monotonic()-tick,**s})
                counters["scored_histories"]+=1
            for handle in handles:handle.remove()
            require(all(p.grad is None and not p.requires_grad for p in model.parameters()),"post-model gradient identity")
            del batch,requests,model,qwen;gc.collect();torch.cuda.empty_cache()
        require(counters=={"model_loads":2,"teacher_forced_forwards":8,"image_forwards":8,"scored_histories":8,"gradient_enabled_forwards":0},"execution counters")
        by={(r["model"],r["request_id"]):r for r in result_rows};comparisons=[]
        for h in m["histories"]:
            a=by[("n16_anchor",h["request_id"])];b=by[("after_two_updates",h["request_id"])]
            comparisons.append({"request_id":h["request_id"],"temperature":h["temperature"],"history_sha256":h["history_sha256"],"anchor_margin_eos_minus_row_open":a["margin_eos_minus_row_open"],"after_two_updates_margin_eos_minus_row_open":b["margin_eos_minus_row_open"],"delta_margin_after_minus_anchor":b["margin_eos_minus_row_open"]-a["margin_eos_minus_row_open"],"anchor_top_next_id":a["top_next_id"],"after_two_updates_top_next_id":b["top_next_id"]})
        result={"schema":SCHEMA+".result","status":"candidate_completed","manifest":binding(manifest_path),"score_count":len(result_rows),"scores":result_rows,"comparisons":comparisons,"counters":counters,"interpretation_limit":"Teacher-forced margin changes at exact saved histories show next-token preference changes, not a KV mechanism or natural-generation causal path. Sampling EOS versus greedy argmax remains a sufficient alternative explanation for the original route split."}
        publish(root/"result.json",result)
        terminal={"schema":SCHEMA+".terminal","status":"completed","manifest":binding(manifest_path),"result":binding(root/"result.json"),"phase":"complete","counters":counters,"elapsed_seconds":time.monotonic()-started}
    except BaseException as exc:
        terminal={"schema":SCHEMA+".terminal","status":"failed","manifest":binding(manifest_path),"phase":phase,"counters":counters,"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-started}
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        if torch.cuda.is_initialized():peak_alloc=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
        terminal.update(peak_cuda_allocated_bytes=peak_alloc,peak_cuda_reserved_bytes=peak_reserved,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,artifact_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()))
        publish(terminal_path,terminal)
def verify(manifest_path:Path,out:Path)->dict[str,Any]:
    root=manifest_path.resolve().parent;m=verify_manifest(json.loads(manifest_path.read_text()),manifest_path)
    t=json.loads((root/"terminal.json").read_text());r=json.loads((root/"result.json").read_text())
    require(t["status"]=="completed" and r["status"]=="candidate_completed","completion")
    require(t["counters"]==r["counters"]=={"model_loads":2,"teacher_forced_forwards":8,"image_forwards":8,"scored_histories":8,"gradient_enabled_forwards":0},"counters")
    require(len(r["scores"])==8 and len(r["comparisons"])==4,"denominators")
    require({(z["model"],z["request_id"]) for z in r["scores"]}=={(s["name"],h["request_id"]) for s in m["models"] for h in m["histories"]},"score coverage")
    max_delta=0.
    for z in r["scores"]:
        b=z["next_logit_array"];require(binding(b["path"])==b,"logit array bytes")
        observed=score_array(np.load(b["path"],allow_pickle=False))
        for k,v in observed.items():
            if isinstance(v,float):max_delta=max(max_delta,abs(v-z[k]))
            else:require(v==z[k],f"score {k}")
    require(max_delta<=2e-6,"saved score replay tolerance")
    value={"schema":SCHEMA+".verification","status":"candidate_cpu_verified","manifest":binding(manifest_path),"terminal":binding(root/"terminal.json"),"result":binding(root/"result.json"),"score_count":8,"comparison_count":4,"max_numeric_replay_delta":max_delta,"adapter_fingerprints_distinct":m["models"][0]["adapter"]["fingerprint"]!=m["models"][1]["adapter"]["fingerprint"]}
    publish(out,value);return value
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("run","verify"));p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.command=="run":run(a.manifest)
    else: require(a.output is not None,"verify output");print(json.dumps(verify(a.manifest,a.output),indent=2))
if __name__=="__main__":main()
