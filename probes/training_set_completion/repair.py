"""Bounded stage-03 forced-history single-owner repair acquisition."""
from __future__ import annotations
import argparse, hashlib, json, os, resource, signal, subprocess, time, traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.training_set_completion.acquisition import IMAGE_IDS, binding, digest, publish, read, require
from probes.training_set_completion.refresh import _checked_terminal

B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
ROOT = B / "stage03-single-owner-repair-v1"; PLAN = ROOT / "owner-plan.json"
ACQ = B / "stage01-acquisition-v1-retry1-config-batch2"; PARENT = B / "first-fit-v1/training/checkpoints/step-00016/adapter"
STEP16 = B / "first-fit-v1/readback-step-16.json"; STEP32 = B / "first-fit-v1/readback-step-32.json"
CAP, EOS, SEED_ROOT, WORKER_SECONDS = 3084, 151645, 2026091403, 1800
TEMPS, GPUS, SCHEMA = (0.1, .3, .7), tuple(range(8)), "training_set_completion.stage03_single_owner_repair.v1"

def request_id(image: int, temp: float | None) -> str:
    return f"stage03:image-{image:012d}:{'greedy' if temp is None else 'sample-t'+str(temp).replace('.','p')}"
def seed(image: int, temp: float) -> int: return int(digest({"root":SEED_ROOT,"image":image,"temperature":temp})[:8],16)&0x7fffffff
def row_path(output: Path, request: Mapping[str,Any]) -> Path: return output/"rows"/(hashlib.sha256(request["request_id"].encode()).hexdigest()+".json")

def forced_row_tokens(category_ids: Sequence[int], bins: Sequence[int], coordinate_ids: Sequence[int], specials: Mapping[str,int]) -> list[int]:
    require(len(bins)==4 and all(type(x) is int and 0 <= x < 1000 for x in bins) and bins[0]<bins[2] and bins[1]<bins[3], "invalid forced reference axes")
    require(category_ids and all(type(x) is int and x>=0 for x in category_ids), "invalid category token IDs")
    return [specials["object_ref_start"],*category_ids,specials["object_ref_end"],specials["box_start"],*[coordinate_ids[x] for x in bins],specials["box_end"]]

def _source_by_image(path: Path) -> dict[int,dict[str,Any]]:
    rows={int(x["image_id"]):x for x in read(path)["rows"]}; require(set(rows)==set(IMAGE_IDS), "source readback image denominator"); return rows

def _load_prefixes(plan: Mapping[str,Any], records: Mapping[int,Mapping[str,Any]], tokenizer: Any, coord: Sequence[int]) -> dict[int,dict[str,Any]]:
    from probes.dora_owner_learning.geometric_dedup import _character_span_to_token_interval, _exact_token_text_frame
    from probes.source_rweak_row_cross.run import native_record
    s16,s32=_source_by_image(STEP16),_source_by_image(STEP32); special_names=("object_ref_start","object_ref_end","box_start","box_end")
    specials={name:tokenizer.convert_tokens_to_ids(f"<|{name}|>") for name in special_names}
    require(all(type(x)is int and x>=0 and tokenizer.convert_ids_to_tokens(x)==f"<|{n}|>" for n,x in specials.items()),"special token binding")
    values={}
    for action in plan["rows"]:
        image=int(action["image_id"]); kind=action["action"]
        if kind=="retain_step32_complete_route":
            values[image]={"condition":"retained_step32_complete_route","continuation":s32[image]["generated_token_ids"],"source":binding(STEP32),"forced_prefix":[],"forced_owner":None}; continue
        source=s16[image]
        if kind=="replace_step32_last_row_geometry_and_release":
            source=s32[image]; text,spans=_exact_token_text_frame(source["generated_token_ids"],tokenizer)
            parsed=native_record(text,records[image]["case"],records[image]["golden"],source["decode_stop_reason"])
            require(len(parsed["pred"])>=6,"step32 lacks six rows")
            end=int(parsed["pred"][5]["char_end"]); _,right=_character_span_to_token_interval(0,end,text=text,token_spans=spans)
            base=source["generated_token_ids"][:right]; source_binding=binding(STEP32); prefix_rule="exact_step32_first_six_raw_rows"
        else:
            require(source["generated_token_ids"][-1]==EOS,"step16 source needs final EOS")
            base=source["generated_token_ids"][:-1]; source_binding=binding(STEP16); prefix_rule="exact_step16_greedy_without_final_eos"
        cat=tokenizer.encode(action["category"],add_special_tokens=False); require(tokenizer.decode(cat,skip_special_tokens=False)==action["category"],"category token binding")
        appended=forced_row_tokens(cat,action["reference_bins"],coord,specials); prefix=[*base,*appended]
        require(EOS not in prefix and len(prefix)<CAP,"forced prefix EOS/cap")
        values[image]={"condition":"forced_history_exploration_not_native_completion","continuation":None,"forced_prefix":prefix,"forced_prefix_sha256":digest(prefix),"source":source_binding,"prefix_rule":prefix_rule,"forced_owner":{"owner_id":action["owner_id"],"category":action["category"],"reference_bins":action["reference_bins"],"appended_row_token_ids":appended,"appended_row_sha256":digest(appended)}}
    require(set(values)==set(IMAGE_IDS),"owner plan cohort")
    # JSON object keys are strings; normalize before hashing the manifest so a
    # published reload has the same canonical identity as its in-memory source.
    return {str(image): value for image, value in values.items()}

def prepare(output: Path=ROOT)->dict[str,Any]:
    from probes.training_set_completion.training import coordinate_token_table
    from src.adapters.dora import inspect_dora_adapter_payload
    require(not (output/"manifest.json").exists(),"repair manifest collision")
    plan=read(PLAN); require(plan.get("status")=="frozen_root_repair_plan" and len(plan.get("rows",[]))==11 and plan["generator_parent"]==str(PARENT),"frozen owner plan")
    acq=read(ACQ/"manifest.json"); records={int(x["image_id"]):x for x in acq["records"]}; require(set(records)==set(IMAGE_IDS),"original acquisition records")
    table=coordinate_token_table(acq["model"]["config"]["model"]["base_model"])
    tokenizer=None
    from probes.training_set_completion.route_bank import _load_tokenizer
    tokenizer=_load_tokenizer(Path(acq["model"]["config"]["model"]["base_model"]))
    prefixes=_load_prefixes(plan,records,tokenizer,table["ids"])
    jobs=[]
    for image in IMAGE_IDS:
        if prefixes[str(image)]["condition"]=="retained_step32_complete_route": continue
        jobs.append({"request_id":request_id(image,None),"image_id":image,"kind":"greedy","temperature":0.,"seed":SEED_ROOT})
        jobs.extend({"request_id":request_id(image,t),"image_id":image,"kind":"sample","temperature":t,"seed":seed(image,t)} for t in TEMPS)
    require(len(jobs)==40 and len({x["request_id"] for x in jobs})==40,"repair request denominator")
    m={"schema":SCHEMA,"status":"candidate_ready","owner_plan":binding(PLAN),"parent_adapter":inspect_dora_adapter_payload(PARENT,acq["model"]["config"]["model"]["base_model"]),"sources":{"acquisition_manifest":binding(ACQ/"manifest.json"),"step16":binding(STEP16),"step32":binding(STEP32),"producer":binding(Path(__file__))},"model_config":acq["model"]["config"],"records":list(records.values()),"coordinate_token_ids":table["ids"],"coordinate_token_spellings":table["spellings"],"prefixes":prefixes,"requests":jobs,"runtime":{"cap":CAP,"eos":EOS,"seed_root":SEED_ROOT,"worker_seconds":WORKER_SECONDS,"gpus":list(GPUS),"fp32_sdpa":True,"empty_assistant_prefix":True,"repetition_penalty":1.,"top_p":1.,"top_k":0},"content_sha256":None};m["content_sha256"]=digest({k:v for k,v in m.items() if k!="content_sha256"});validate_manifest(m);publish(output/"manifest.json",m);return m

def validate_manifest(m:Mapping[str,Any])->None:
    require(m.get("schema")==SCHEMA and m.get("content_sha256")==digest({k:v for k,v in m.items() if k!="content_sha256"}),"manifest hash")
    require(len(m.get("requests",[]))==40 and len(m.get("prefixes",{}))==11,"repair denominator")
    for key in ("owner_plan","parent_adapter") : require(key in m,"repair parent binding")
    for key in ("acquisition_manifest","step16","step32","producer"): require(binding(m["sources"][key]["path"])==m["sources"][key],f"source changed {key}")
    for image,info in m["prefixes"].items():
        if info["condition"]!="retained_step32_complete_route": require(EOS not in info["forced_prefix"] and len(info["forced_prefix"])<CAP and info["forced_prefix_sha256"]==digest(info["forced_prefix"]),"prefix identity/cap/eos")

def _jobs(m:Mapping[str,Any],shard:int)->list[dict[str,Any]]: require(0<=shard<8,"shard");return m["requests"][shard::8]

def worker(*,manifest_path:Path,output:Path,shard:int,gpu:int)->None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.native_owner_scale.evaluation import _candidate_materialized_case
    from probes.source_rweak_row_cross.run import build_requests,native_record
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy,generate_continuations
    from src.qwen.native import prepare_native_inputs
    m=read(manifest_path);validate_manifest(m);require(os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu) and torch.cuda.device_count()==1,"GPU isolation"); jobs=_jobs(m,shard);term={"schema":SCHEMA+".terminal","status":"running","shard":shard,"gpu":gpu,"manifest":binding(manifest_path),"expected":len(jobs),"completed":0,"model_forwards":0,"image_forwards":0,"new_tokens":0};path=output/"terminals"/f"shard-{shard}.json";require(not path.exists(),"terminal collision");started=time.monotonic();handles=[];old=signal.getsignal(signal.SIGALRM)
    try:
      signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("repair worker wall")));signal.alarm(WORKER_SECONDS);config=checkpoint_config(InferConfig.model_validate(m["model_config"]),PARENT);torch.cuda.set_device("cuda:0");qwen,identity=load_policy(config,device=torch.device("cuda:0"));require(identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"]==["torch.float32"] and identity["effective_settings"]["observed_attn_implementation"]=="sdpa","FP32 SDPA");publish(output/"model"/f"shard-{shard}.json",identity);qwen.model.eval();handles.append(qwen.model.register_forward_pre_hook(lambda *_:term.__setitem__("model_forwards",term["model_forwards"]+1)));visual=[x for n,x in qwen.model.named_modules() if n.endswith("visual")];require(len(visual)==1,"visual");handles.append(visual[0].register_forward_pre_hook(lambda *_:term.__setitem__("image_forwards",term["image_forwards"]+1)));records={int(x["image_id"]):x for x in m["records"]}
      for job in jobs:
       rec=records[job["image_id"]];info=m["prefixes"][str(job["image_id"])] if str(job["image_id"]) in m["prefixes"] else m["prefixes"][job["image_id"]];prefix=info["forced_prefix"];case=_candidate_materialized_case(rec["case"],m["model_config"]);requests,_=build_requests(qwen,m["model_config"],[case]);batch=prepare_native_inputs(qwen.processor,requests,device="cuda:0",record_media_identity=True);plan=rec["case"]["image_plan"];require(list(batch.prompt_token_ids[0])==rec["prompt_token_ids"] and batch.media_sha256[0]==plan["executed_media_sha256"] and list(batch.image_grids[0])==plan["observed_image_grid_thw"],"original prompt/media");pol=NativeGenerationPolicy(temperature=job["temperature"],top_p=1.,top_k=0,repetition_penalty=1.,use_model_defaults=False);tick=time.monotonic();
       with torch.inference_mode(): gen,=generate_continuations(qwen.model,batch,extensions=[prefix],budgets=[CAP-len(prefix)],eos_token_id=EOS,pad_token_id=qwen.tokenizer.pad_token_id,policy=pol,trace="none",seed=job["seed"])
       suffix=list(gen.token_ids);full=[*prefix,*suffix];_checked_terminal(full,gen.stop_reason);text=qwen.tokenizer.decode(full,skip_special_tokens=False);row={"schema":SCHEMA+".row","request":job,"manifest_sha256":m["content_sha256"],"image_id":job["image_id"],"example_id":rec["example_id"],"forced_history_condition":info["condition"],"forced_prefix_token_ids":prefix,"forced_prefix_sha256":digest(prefix),"generated_suffix_token_ids":suffix,"generated_suffix_sha256":digest(suffix),"continuation_token_ids":full,"continuation_token_ids_sha256":digest(full),"prefix_token_count":len(prefix),"suffix_token_count":len(suffix),"assistant_token_cap":CAP,"decode_stop_reason":gen.stop_reason,"raw_decode_text":text,"parsed":native_record(text,rec["case"],rec["golden"],gen.stop_reason),"forced_owner":info["forced_owner"],"source_prefix":info["source"],"timing":{"generation_seconds":time.monotonic()-tick},"model_receipt":binding(output/"model"/f"shard-{shard}.json")};publish(row_path(output,job),row);term["completed"]+=1;term["new_tokens"]+=len(suffix)
      term.update(status="completed",exit_code=0)
    except BaseException as exc: term.update(status="failed",exit_code=1,error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc());raise
    finally:
      signal.alarm(0);signal.signal(signal.SIGALRM,old);[h.remove() for h in handles];term.update(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);publish(path,term)

def collect(manifest_path:Path,output:Path)->dict[str,Any]:
 m=read(manifest_path);validate_manifest(m);rows=[read(row_path(output,j)) for j in m["requests"]];require(len(rows)==40 and all(len(x["continuation_token_ids"])<=CAP for x in rows),"output cap denominator");require(all(x["continuation_token_ids"][:x["prefix_token_count"]]==x["forced_prefix_token_ids"] and EOS not in x["forced_prefix_token_ids"] for x in rows),"prefix preservation");value={"schema":SCHEMA+".result","status":"candidate_ready","manifest":binding(manifest_path),"request_count":40,"generated_suffix_tokens":sum(x["suffix_token_count"] for x in rows),"stop_counts":{s:sum(x["decode_stop_reason"]==s for x in rows) for s in ("im_end","length")},"parser":{"valid_predictions":sum(x["parsed"]["valid_prediction_count"] for x in rows),"dropped_predictions":sum(x["parsed"]["dropped_prediction_count"] for x in rows)},"forced_history_not_native_completion":True};publish(output/"result.json",value);return value

def controller(manifest_path:Path,output:Path)->None:
 m=read(manifest_path);validate_manifest(m);term={"schema":SCHEMA+".controller","status":"running","manifest":binding(manifest_path)};started=time.monotonic()
 try:
  ps=[]
  for shard,gpu in enumerate(GPUS):
   log=output/"logs"/f"shard-{shard}.log";log.parent.mkdir(parents=True,exist_ok=True);f=log.open("x");cmd=["python","-m","probes.training_set_completion.repair","worker","--manifest",str(manifest_path),"--output",str(output),"--shard",str(shard),"--gpu",str(gpu)];ps.append((subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"OMP_NUM_THREADS":"2","TOKENIZERS_PARALLELISM":"false"}),f,cmd))
  exits=[]
  for p,f,cmd in ps: exits.append({"pid":p.pid,"exit_code":p.wait(timeout=WORKER_SECONDS),"command":cmd});f.close()
  publish(output/"exits.json",{"schema":SCHEMA+".exits","exits":exits});require(all(x["exit_code"]==0 for x in exits),"worker failed");term.update(status="completed",result=binding(output/"result.json") if (output/"result.json").exists() else None);result=collect(manifest_path,output);term["result"]=binding(output/"result.json")
 except BaseException as e:term.update(status="failed",error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc())
 finally:term["elapsed_seconds"]=time.monotonic()-started;publish(output/"terminal.json",term)
 if term["status"]!="completed":raise RuntimeError(term["error"])

def launch(manifest_path:Path,output:Path)->dict[str,Any]:
 require(not (output/"launch.json").exists(),"launch collision");session="coordexp-stage03-repair-v1";cmd=f"cd {Path(__file__).resolve().parents[2]} && exec python -m probes.training_set_completion.repair controller --manifest {manifest_path} --output {output}";subprocess.run(["tmux","new-session","-d","-s",session,cmd],check=True);x={"schema":SCHEMA+".launch","status":"launched","tmux_session":session,"manifest":binding(manifest_path),"command":cmd,"gpus":list(GPUS),"per_worker_seconds":WORKER_SECONDS};publish(output/"launch.json",x);return x

def main()->None:
 p=argparse.ArgumentParser();p.add_argument("command",choices=("prepare","worker","controller","launch"));p.add_argument("--output",type=Path,default=ROOT);p.add_argument("--manifest",type=Path);p.add_argument("--shard",type=int);p.add_argument("--gpu",type=int);a=p.parse_args()
 if a.command=="prepare":x=prepare(a.output)
 elif a.command=="launch":x=launch(a.manifest or a.output/"manifest.json",a.output)
 elif a.command=="controller":controller(a.manifest or a.output/"manifest.json",a.output);return
 else:require(a.manifest and a.shard is not None and a.gpu is not None,"worker args");worker(manifest_path=a.manifest,output=a.output,shard=a.shard,gpu=a.gpu);return
 print(json.dumps({"schema":x["schema"],"status":x["status"]}))
if __name__=="__main__":main()
