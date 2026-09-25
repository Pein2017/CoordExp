"""First admitted four-request, all-layer latest-y1 V discriminator."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import resource
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.recurrence_dynamics.recurrence_cross_image_phase import scale as old

REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-23-recurrence-y1-value-phase"
PROTOCOL = UNIT / "unit.md"
MANIFEST = UNIT / "manifest.json"
PROTOCOL_SHA = "b665c1ad61ce8930f59a278ac43f9f30104be89a3df2641263232795bcca4190"
MANIFEST_SHA = "68854cbc6c1c81e122d00631b7724ae5891993ea83086014ee9dcffd87d8e015"
CASE_ID = "mature:1584:2"
ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/first-case-v1")
LAYERS, HEADS, DIM, SUFFIX, TOL = old.LAYERS, old.HEADS, old.DIM, old.SUFFIX, old.TOL
KINDS = ("native", "phase", "v_sham", "donor_native", "donor_phase")


def require(ok, message):
    old._require(bool(ok), message)


def binding(path):
    return old.literal_binding(Path(path))


def contract():
    require(binding(PROTOCOL)["sha256"] == PROTOCOL_SHA and
            binding(MANIFEST)["sha256"] == MANIFEST_SHA, "frozen value protocol/manifest changed")
    m = json.loads(MANIFEST.read_text())
    require(m["status"] == "frozen-three-cases-first-only-admitted" and
            m["admitted_case_ids"] == [CASE_ID] and
            m["held_case_ids"] == ["mature:2299:2", "mature:4134:7"] and
            [x["id"] for x in m["cases"]] == [CASE_ID, *m["held_case_ids"]] and
            m["decision"] == {"margin":"z(earlier_y1)-z(latest_y1)",
                              "shifted_donor_effect_min_nats":1.0,
                              "phase_by_V_interaction_min_nats":0.5,
                              "numerical_guard_nats":0.001,
                              "shared_prediction":"all3 frozen cases pass; technical unknown remains unanswered"} and
            m["budget"]["first_case_gpu_hours_cap"] == .1 and
            m["budget"]["package_gpu_hours_cap"] == .25 and
            m["budget"]["sequence_gpu_hours_cap"] == 8,
            "value admission, scope, thresholds, or caps changed")
    for name in ("predecessor_acceptance", "CPU_readout_acceptance", "feasibility"):
        old.bound(m[name])
    c = m["cases"][0]
    _, source, receipt, _ = old.contract(CASE_ID)
    require(c["frozen_source_case"] == source and c["V_row_offset"] == 5 and
            c["donor_V_physical"] == source["geometry"]["physical_full_batch_padded"]["earlier"][0]+5 and
            c["destination_V_physical"] == source["geometry"]["physical_full_batch_padded"]["latest"][0]+5,
            "case source or V slot changed")
    for name in ("prefill_blocks", "reference_reduction", "reference_native_vector", "reference_phase_vector"):
        old.bound(c[name])
    return m, c, source, receipt


def snapshots(cache, target, earlier_slot, latest_slot):
    require(len(cache.layers) == LAYERS, "wrong snapshot layer count")
    result = []
    for layer in cache.layers:
        donor = layer.values[target, :, earlier_slot, :].detach().clone()
        native = layer.values[target, :, latest_slot, :].detach().clone()
        require(donor.shape == native.shape == (HEADS, DIM) and
                torch.isfinite(donor).all() and torch.isfinite(native).all(),
                "nonfinite or malformed fresh V snapshot")
        result.append({"donor":donor, "native":native,
                       "donor_sha256":old.tensor_hash(donor),
                       "native_sha256":old.tensor_hash(native)})
    return result


@contextmanager
def joint_patch(cache, kind, latest_keys, saved_v, native_digest, *,
                target, expected_target, latest_span, latest_slot, expected_slot, width):
    """The complete suffix caller stays inside inference mode through restoration."""
    with torch.inference_mode():
        require(kind in KINDS and target == expected_target and target in range(4) and
                latest_slot == expected_slot == latest_span[0]+5 and
                latest_span[1]-latest_span[0] == 9 and latest_span[1] <= width and
                cache.get_seq_length() == width and len(cache.layers) == LAYERS and
                len(latest_keys) == len(saved_v) == LAYERS and
                old.cache_digest(cache) == native_digest, "wrong patch entry/slot/target/cache")
        saved = []
        try:
            for i, layer in enumerate(cache.layers):
                a,b = latest_span
                key = latest_keys[i]
                donor,native = saved_v[i]["donor"],saved_v[i]["native"]
                require(key.shape == (HEADS,9,DIM) and donor.shape == native.shape == (HEADS,DIM) and
                        torch.isfinite(key).all() and torch.isfinite(donor).all() and
                        torch.isfinite(native).all() and
                        old.tensor_hash(donor) == saved_v[i]["donor_sha256"] and
                        old.tensor_hash(native) == saved_v[i]["native_sha256"] and
                        old.tensor_hash(layer.values[target,:,latest_slot,:]) == saved_v[i]["native_sha256"],
                        "wrong candidate K or donor/native V snapshot")
                prior_k = layer.keys[target,:,a:b,:].clone()
                prior_v = layer.values[target,:,latest_slot,:].clone()
                saved.append((layer,prior_k,prior_v))
                if kind in ("phase","v_sham","donor_phase"):
                    layer.keys[target,:,a:b,:].copy_(key.to(device=prior_k.device,dtype=prior_k.dtype))
                if kind in ("v_sham","donor_native","donor_phase"):
                    value = native if kind == "v_sham" else donor
                    layer.values[target,:,latest_slot,:].copy_(value.to(device=prior_v.device,dtype=prior_v.dtype))
            yield
        finally:
            try:
                cache.crop(width)
            finally:
                for layer,prior_k,prior_v in saved:
                    layer.keys[target,:,latest_span[0]:latest_span[1],:].copy_(prior_k)
                    layer.values[target,:,latest_slot,:].copy_(prior_v)
            require(cache.get_seq_length() == width and old.cache_digest(cache) == native_digest,
                    "historical K/V or suffix failed finally restoration")


def patch_fixture(case):
    """Exercise the actual context with inference tensors; no model or CUDA."""
    width,span,slot,target = 20,(6,15),11,case["frozen_source_case"]["batch_index"]
    with torch.inference_mode():
        cache=DynamicCache()
        for i in range(LAYERS):
            key=torch.arange(4*HEADS*width*DIM,dtype=torch.float32).reshape(4,HEADS,width,DIM)/10000+i
            val=(key/7).clone()
            cache.update(key,val,i)
        before=old.cache_digest(cache)
        snap=snapshots(cache,target,2,slot)
        keys=[layer.keys[target,:,span[0]:span[1],:].clone()+1 for layer in cache.layers]
        args=dict(target=target,expected_target=target,latest_span=span,
                  latest_slot=slot,expected_slot=slot,width=width)
        def call(kind="donor_phase", snap_arg=snap, **changes):
            passed={**args,**changes}
            return joint_patch(cache,kind,keys,snap_arg,before,**passed)
        with call():
            require(torch.is_inference_mode_enabled(),"patch body left inference mode")
            for i,layer in enumerate(cache.layers):
                require(old.tensor_hash(layer.values[target,:,slot,:])==snap[i]["donor_sha256"] and
                        torch.equal(layer.keys[target,:,span[0]:span[1],:],keys[i]),
                        "CPU fixture patch not consumed")
                cache.update(torch.zeros((4,HEADS,SUFFIX,DIM)),
                             torch.zeros((4,HEADS,SUFFIX,DIM)),i)
        require(old.cache_digest(cache)==before,"normal fixture restoration failed")
        try:
            with call():
                for i in range(LAYERS):
                    cache.update(torch.zeros((4,HEADS,SUFFIX,DIM)),
                                 torch.zeros((4,HEADS,SUFFIX,DIM)),i)
                raise RuntimeError("forced-body-exception")
        except RuntimeError as exc:
            require(str(exc)=="forced-body-exception", "wrong forced exception")
        require(old.cache_digest(cache)==before,"exception fixture restoration failed")
        rejected=[]
        bad=dict(snap[0]);bad["donor"]=bad["donor"].clone()+1
        tests=(("wrong_target",lambda:call(target=(target+1)%4)),
               ("wrong_slot",lambda:call(latest_slot=slot+1)),
               ("wrong_donor",lambda:call(snap_arg=[bad,*snap[1:]])))
        for name,make in tests:
            try:
                with make():pass
            except ValueError: rejected.append(name)
            else:raise ValueError("CPU fixture missed "+name)
        for axis in ("keys","values"):
            untouched=getattr(cache.layers[0],axis)[(target+1)%4,:,0,:].clone()
            try:
                with call():
                    getattr(cache.layers[0],axis)[(target+1)%4,:,0,:].add_(1)
            except ValueError:
                rejected.append("unrelated_"+axis)
                getattr(cache.layers[0],axis)[(target+1)%4,:,0,:].copy_(untouched)
            else:raise ValueError("CPU fixture missed unrelated "+axis)
            require(old.cache_digest(cache)==before,"fixture failed cleanup after mutation")
    return {"status":"passed","layers":LAYERS,"inference_tensor_cache":True,
            "normal_and_forced_exception_restored":True,"mutation_rejections":rejected}


def preflight(root):
    require(not root.exists(),"value-phase output already exists")
    m,c,source,receipt=contract()
    q=old.load_qwen_components_from_options(old.QwenLoadOptions(
        base_model=str(old.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preflight loaded model")
    raw=json.loads(Path(source["source_bindings"]["raw"]["path"]).read_text())["rows"]
    panel=json.loads((Path(source["source_bindings"]["raw"]["path"]).parents[3]/"panel.json").read_text())
    batch,raw,trace,group,planning=old._source(old.source_boundary(source,raw),"untied",panel,q,torch.device("cpu"))
    require(len(raw)==len(group["cases"])==4 and
            old.input_identity(batch)==receipt["input_identity"] and
            old.digest(old.input_identity(batch))==source["source_input_identity_sha256"],
            "CPU original batch identity changed")
    full=old.source_inputs(old._ConfigOnlyRope(),batch,raw,source,int(q.tokenizer.pad_token_id))
    checks=old.cpu_mutations(full,source,raw,batch.prompt_token_ids,int(q.tokenizer.pad_token_id))
    phase=old.phase_cpu_gate(source)
    fixture=patch_fixture(c)
    prior=json.loads(Path(c["reference_reduction"]["path"]).read_text())
    prior_seconds=float(prior["cost"]["allocated_gpu_seconds"])
    forecast=2*prior_seconds
    require(prior["cost"]["model_forwards"]==7 and prior["cost"]["vision_forwards"]==2 and
            forecast<360*m["budget"]["first_case_gpu_hours_cap"]/.1 and
            forecast<3600*m["budget"]["package_gpu_hours_cap"] and
            m["budget"]["sequence_cumulative_prior_gpu_hours"]+forecast/3600<8,
            "first-case shape-aware forecast exceeds cap")
    root.mkdir(parents=True)
    sources=[Path(__file__)]+[REPO/x for x in old.IMPORTS]
    sources += [Path(inspect.getfile(x)) for x in (cache_utils,modeling_qwen3_vl,sdpa_attention)]
    captures=[]
    for path in dict.fromkeys(sources):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved=old.preserve_source(path,run_root=root,relative_name=rel)
        captures.append({"maintained":binding(path),"capture":binding(saved)})
    output=root/"d1-1584-2"
    prefix=["python","-B","-m","probes.recurrence_dynamics.recurrence_y1_value_phase.run"]
    commands={mode:prefix+[mode,"--output",str(output)]+(["--device","cuda:0"] if mode=="run" else [])
              for mode in ("run","readback","reduce")}
    record={"schema":"recurrence_y1_value_phase.preflight.v1","status":"cpu_qualified_before_gpu",
            "protocol":binding(PROTOCOL),"manifest":binding(MANIFEST),
            "case_id":CASE_ID,"source":source["source_bindings"],"effective_identity_expected":receipt["identity"],
            "batch_request_ids":list(batch.request_ids),"target_index":source["batch_index"],
            "input_identity_sha256":old.digest(old.input_identity(batch)),
            "full_source_input_hashes":{k:old.tensor_hash(full[k]) for k in
                                        ("input_ids","attention_mask","position_ids","cache_position")},
            "full_source_shape":{"batch_size":4,"width":int(full["input_ids"].shape[1]),
                                 "pixel_elements":int(batch.inputs["pixel_values"].numel()),
                                 "image_grids":[list(g) for g in batch.image_grids]},
            "source_step_offset":source["geometry"]["first_y1_raw_offset"],
            "ended_companions":[len(row["token_ids"])<=source["geometry"]["first_y1_raw_offset"] for row in raw],
            "cpu_checks":checks,"cpu_phase":phase,"cpu_patch_fixture":fixture,
            "source_planning_replanned":planning.get("replanned_image_plan"),
            "direct_source_captures":captures,"commands":commands,
            "cost_forecast":{"prior_accepted_seconds":prior_seconds,"planning_multiplier":2,
                             "first_case_seconds_2x":forecast,"first_case_cap_seconds":360,
                             "package_cap_seconds":900,"sequence_prior_gpu_hours":
                             m["budget"]["sequence_cumulative_prior_gpu_hours"],
                             "artifact_plan_bytes":m["budget"]["artifact_planning_bytes"]}}
    old._write_new(root/"preflight.json",record)
    print(json.dumps({"status":record["status"],"fixture":fixture,
                      "forecast_seconds_2x":forecast,"captures":len(captures)}))


def other_hash(tensor, start, end):
    return old.tensor_hash(torch.cat((tensor[..., :start, :],tensor[..., end:, :]),dim=-2))


def run(out, device_name):
    started=time.monotonic()
    counts={"model_forwards":0,"vision_forwards":0}
    handles=[]
    device=None
    done=[]
    require(not out.exists(),"first-case output already exists; no retry")
    out.mkdir(parents=True)
    try:
        m,c,source,receipt=contract()
        pre=json.loads((out.parent/"preflight.json").read_text())
        require(pre["status"]=="cpu_qualified_before_gpu" and
                pre["protocol"]==binding(PROTOCOL) and pre["manifest"]==binding(MANIFEST) and
                pre["case_id"]==CASE_ID and pre["cpu_patch_fixture"]["status"]=="passed" and
                pre["cost_forecast"]["first_case_seconds_2x"]<360 and
                not any(out.parent.glob("*/receipt.json")),
                "CPU preflight, one-job scope, or budget changed")
        for item in pre["direct_source_captures"]:
            old.bound(item["maintained"]);old.bound(item["capture"])
        require(pre["commands"]["run"][-2:]==["--device",device_name],"frozen GPU command changed")
        device=torch.device(device_name)
        torch.cuda.set_device(device);torch.empty(1,device=device);torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        q,identity=old.load_model("untied",device)
        expected_identity=receipt["identity"]
        require({k:v for k,v in identity.items() if k!="loader_source"}==
                {k:v for k,v in expected_identity.items() if k!="loader_source"} and
                all(identity["loader_source"][k]==expected_identity["loader_source"][k]
                    for k in ("sha256","size_bytes")),"effective model identity changed")
        model=q.model.eval();target=source["batch_index"]
        raw=json.loads(Path(source["source_bindings"]["raw"]["path"]).read_text())["rows"]
        panel=json.loads((Path(source["source_bindings"]["raw"]["path"]).parents[3]/"panel.json").read_text())
        batch,raw,trace,group,planning=old._source(old.source_boundary(source,raw),"untied",panel,q,device)
        require(len(raw)==len(group["cases"])==4 and
                old.input_identity(batch)==receipt["input_identity"] and
                old.digest(old.input_identity(batch))==pre["input_identity_sha256"] and
                list(batch.request_ids)==pre["batch_request_ids"],"GPU original batch changed")
        full=old.source_inputs(model,batch,raw,source,int(q.tokenizer.pad_token_id))
        require({k:old.tensor_hash(full[k]) for k in pre["full_source_input_hashes"]}==
                pre["full_source_input_hashes"],"GPU full-source inputs changed")
        end=full["input_ids"].shape[1];width=end-SUFFIX
        g=source["geometry"];spans={name:tuple(g["physical_full_batch_padded"][name])
                                      for name in ("earlier","latest")}
        earlier_slot,latest_slot=c["donor_V_physical"],c["destination_V_physical"]
        require(spans["latest"][1]==width and earlier_slot==spans["earlier"][0]+5 and
                latest_slot==spans["latest"][0]+5,"physical V/cache slots changed")
        cache=DynamicCache()
        full.update(use_cache=False,return_dict=True,logits_to_keep=1)
        prefill=dict(full)
        prefill.update(input_ids=full["input_ids"][:,:width],
                       attention_mask=full["attention_mask"][:,:width],
                       position_ids=full["position_ids"][:,:,:width],
                       cache_position=torch.arange(width,device=device),
                       past_key_values=cache,use_cache=True)
        require(prefill["input_ids"].shape==(4,width) and
                torch.equal(prefill["attention_mask"],full["attention_mask"][:,:width]),
                "full-batch prefill split changed")
        active=None

        def before_model(_module,_args,kwargs):
            counts["model_forwards"]+=1
            require(active is not None and counts["model_forwards"]<=7 and
                    time.monotonic()-started<360,"unexpected/over-cap model forward")
            for name in ("input_ids","attention_mask","position_ids","cache_position"):
                require(torch.equal(kwargs.get(name),active["input"][name]),
                        "actual model consumer "+name+" changed")
            media=active["name"] in ("full_native","prefill")
            require(kwargs.get("use_cache") is (not (active["name"]=="full_native")) and
                    kwargs.get("past_key_values") is (None if active["name"]=="full_native" else cache),
                    "actual cache route changed")
            for name in ("pixel_values","image_grid_thw"):
                require((name in kwargs)==media,"actual media route changed")
                if media:require(old.tensor_hash(kwargs[name])==old.tensor_hash(active["input"][name]),
                                 "actual media values changed")
            active["observed"]={name:kwargs[name].detach().cpu().tolist() for name in
                                ("input_ids","attention_mask","position_ids","cache_position")}
            active["observed"].update(media_present=media,use_cache=kwargs["use_cache"],
                                      past_cache_present=kwargs.get("past_key_values") is not None)

        def before_vision(_module,_args):
            counts["vision_forwards"]+=1
            require(counts["vision_forwards"]<=2,"vision forward ceiling reached")
        handles.extend((model.register_forward_pre_hook(before_model,with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(before_vision)))

        # Call 1: exact original full batch/source step.
        active={"name":"full_native","input":full,"observed":{}}
        with torch.inference_mode():source_logits=model(**full).logits[:,-1,:].detach().float().cpu()
        torch.cuda.synchronize(device)
        require(source_logits.shape[0]==4 and counts=={"model_forwards":1,"vision_forwards":1},
                "full source call incomplete")
        offset=g["first_y1_raw_offset"]
        trace_parity=[]
        for i in range(4):
            if offset<len(raw[i]["token_ids"]):
                check=old._trace_compare(logits=source_logits[i],trace=trace,batch_index=i,
                    absolute_offset=offset,token_id=int(raw[i]["token_ids"][offset]),
                    role="first_y1" if i==target else "original_companion_step",atol=TOL)
                require(check["passed"],f"source trace parity row {i} failed")
                trace_parity.append({"batch_index":i,"status":"active_trace_parity",**check})
            else:
                trace_parity.append({"batch_index":i,"status":"ended_before_source_step"})
        require([x["status"]=="ended_before_source_step" for x in trace_parity]==
                pre["ended_companions"] and not pre["ended_companions"][target],
                "source active/ended companion state changed")
        full_dir=out/"cells/01-full-native";full_dir.mkdir(parents=True)
        torch.save(source_logits[target].clone(),full_dir/"vocabulary.pt")
        torch.save(source_logits,full_dir/"full-batch-vocabulary.pt")
        full_record=old._write_new(full_dir/"cell.json",{
            "kind":"full_native","vector":binding(full_dir/"vocabulary.pt"),
            "full_batch_vector":binding(full_dir/"full-batch-vocabulary.pt"),
            "consumer":old._write_new(full_dir/"consumer-raw.json",active["observed"]),
            "trace_parity":trace_parity,"counts_after":dict(counts),
            "allocated_gpu_seconds_after":time.monotonic()-started})
        done.append({"kind":"full_native","record":full_record})

        # Call 2: fresh native four-request historical prefill and block capture.
        rotary=model.model.language_model.rotary_emb
        seen_phase=[];pre_norm=[{} for _ in range(LAYERS)];capture=[]
        def on_prefill_rotary(_module,_args,output):
            cos,sin=output
            require(cos.shape==sin.shape==(4,width,DIM),"prefill rotary shape changed")
            seen_phase.append({name:{"cos":cos[target,a:b].detach().clone(),
                                     "sin":sin[target,a:b].detach().clone()}
                               for name,(a,b) in spans.items()})
        capture.append(rotary.register_forward_hook(on_prefill_rotary))
        for index,layer in enumerate(model.model.language_model.layers):
            def on_pre_k(_module,_args,output,i=index):
                require(output.shape==(4,width,HEADS,DIM),"pre-K shape changed")
                pre_norm[i]={name:output[target,a:b].transpose(0,1).detach().clone()
                             for name,(a,b) in spans.items()}
            capture.append(layer.self_attn.k_norm.register_forward_hook(on_pre_k))
        active={"name":"prefill","input":prefill,"observed":{}}
        try:
            with torch.inference_mode():pref_out=model(**prefill)
        finally:
            for hook in capture:hook.remove()
        torch.cuda.synchronize(device)
        require(pref_out.past_key_values is cache and len(seen_phase)==1 and
                all(set(x)==set(spans) for x in pre_norm) and
                counts=={"model_forwards":2,"vision_forwards":2} and
                len(cache.layers)==LAYERS and all(
                    getattr(layer,axis).shape==(4,HEADS,width,DIM) and
                    getattr(layer,axis).dtype==torch.float32
                    for layer in cache.layers for axis in ("keys","values")),
                "full-batch prefill/cache incomplete")
        native_digest=old.cache_digest(cache)
        snap=snapshots(cache,target,earlier_slot,latest_slot)
        native_segments=[]
        for i,layer in enumerate(cache.layers):
            native_segments.append({
                "target_other_K":other_hash(layer.keys[target],*spans["latest"]),
                "target_other_V":other_hash(layer.values[target],latest_slot,latest_slot+1),
                "latest_K":old.tensor_hash(layer.keys[target,:,spans["latest"][0]:spans["latest"][1],:]),
                "selected_V":snap[i]["native_sha256"],
                "companion_K":old.tensor_hash(layer.keys[[j for j in range(4) if j!=target]]),
                "companion_V":old.tensor_hash(layer.values[[j for j in range(4) if j!=target]])})
        native_positions={name:full["position_ids"][:,target,a:b].cpu()
                          for name,(a,b) in spans.items()}
        destination,dest_record=old.destination_phase(
            rotary,native_positions,device,g["rotary_position_ids"])
        earlier_error=max(float((destination["destination_earlier"][axis]-
                                 seen_phase[0]["latest"][axis].cpu()).abs().max())
                          for axis in ("cos","sin"))
        require(earlier_error<=TOL,"destination/observed latest phase differs")
        dest_record["earlier_matches_latest_max_abs_error"]=earlier_error
        torch.save(destination,out/"destination-phase.pt")
        old._write_new(out/"destination-phase-raw.json",{
            **dest_record,"vector":binding(out/"destination-phase.pt")})
        seen_phase[0].update(destination)
        blocks={"phase":{name:{axis:value.detach().float().cpu() for axis,value in rec.items()}
                         for name,rec in seen_phase[0].items()},"layers":[]}
        for i,layer in enumerate(cache.layers):
            blocks["layers"].append({name:{
                "pre_k":pre_norm[i][name].float().cpu(),
                "native_k":layer.keys[target,:,a:b,:].detach().float().cpu().clone(),
                "native_v":layer.values[target,:,a:b,:].detach().float().cpu().clone()}
                for name,(a,b) in spans.items()})
        torch.save(blocks,out/"prefill-blocks.pt")
        old._write_new(out/"prefill-raw.json",{
            "vector":binding(out/"prefill-blocks.pt"),"consumer":active["observed"],
            "cache_digest":native_digest,"native_segments":native_segments,
            "donor_V_sha256":[x["donor_sha256"] for x in snap],
            "native_latest_V_sha256":[x["native_sha256"] for x in snap],
            "counts_after":dict(counts),"allocated_gpu_seconds_after":time.monotonic()-started})
        phase_records,candidates=old.qualify_phase(blocks)
        old._write_new(out/"phase-qualification.json",{
            "records":phase_records,"prefill_blocks":binding(out/"prefill-blocks.pt")})
        require(all(v["qualified"] for row in phase_records for k,v in row.items() if k!="layer"),
                "all-layer phase reconstruction failed")
        active=None

        native_vector=None;phase_vector=None;native_companions=None
        for step,kind in enumerate(KINDS,3):
            require(step==3 or native_vector is not None,"native anchor not qualified")
            require(step<=4 or phase_vector is not None,"phase anchor not qualified")
            require(step<=5 or sham_passed,"identity V sham not qualified")
            suffix={"input_ids":full["input_ids"][:,width:end],
                    "attention_mask":full["attention_mask"],
                    "position_ids":full["position_ids"][:,:,width:end],
                    "cache_position":torch.arange(width,end,device=device),
                    "past_key_values":cache,"use_cache":True,"return_dict":True,"logits_to_keep":1}
            require(suffix["input_ids"].shape==(4,SUFFIX) and
                    torch.equal(suffix["input_ids"],full["input_ids"][:,width:end]) and
                    torch.equal(suffix["position_ids"],full["position_ids"][:,:,width:end]) and
                    torch.equal(suffix["attention_mask"],full["attention_mask"]),
                    "native S/companions changed")
            with joint_patch(cache,kind,candidates["latest"],snap,native_digest,
                             target=target,expected_target=target,latest_span=spans["latest"],
                             latest_slot=latest_slot,expected_slot=c["destination_V_physical"],width=width):
                expected=old.cache_digest(cache)
                want_k=kind in ("phase","v_sham","donor_phase")
                want_donor=kind in ("donor_native","donor_phase")
                for i,layer in enumerate(cache.layers):
                    require(other_hash(layer.keys[target],*spans["latest"])==native_segments[i]["target_other_K"] and
                            other_hash(layer.values[target],latest_slot,latest_slot+1)==native_segments[i]["target_other_V"] and
                            old.tensor_hash(layer.keys[target,:,spans["latest"][0]:spans["latest"][1],:])==
                            (old.tensor_hash(candidates["latest"][i]) if want_k else native_segments[i]["latest_K"]) and
                            old.tensor_hash(layer.values[target,:,latest_slot,:])==
                            (snap[i]["donor_sha256"] if want_donor else snap[i]["native_sha256"]) and
                            old.tensor_hash(layer.keys[[j for j in range(4) if j!=target]])==native_segments[i]["companion_K"] and
                            old.tensor_hash(layer.values[[j for j in range(4) if j!=target]])==native_segments[i]["companion_V"],
                            "historical K/V changed outside declared cell")
                active={"name":kind,"input":suffix,"observed":{}}
                before_rows={};after_rows={};rotary_record=[];rotary_values=[];local=[]
                def on_suffix_rotary(_module,args,output):
                    require(torch.equal(args[1],suffix["position_ids"]),"actual S rotary input changed")
                    cos,sin=output
                    require(cos.shape==sin.shape==(4,SUFFIX,DIM),"S rotary shape changed")
                    rotary_record.append({"cos":old.tensor_hash(cos),"sin":old.tensor_hash(sin)})
                    rotary_values.append((cos,sin))
                local.append(rotary.register_forward_hook(on_suffix_rotary))
                for index,layer in enumerate(model.model.language_model.layers):
                    def before_attention(_module,_args,kwargs,i=index):
                        require(kwargs.get("past_key_values") is cache and
                                cache.get_seq_length(i)==width and
                                torch.equal(kwargs.get("cache_position"),suffix["cache_position"]),
                                "actual attention cache slots changed")
                        mask=kwargs.get("attention_mask")
                        require(isinstance(mask,torch.Tensor) and mask.dtype==torch.bool and
                                mask.shape==(4,1,SUFFIX,end),"actual attention mask changed")
                        causal=torch.arange(end,device=mask.device)[None,:] <= (
                            width+torch.arange(SUFFIX,device=mask.device)[:,None])
                        allowed=causal[None,:,:] & suffix["attention_mask"][:,:,None].transpose(1,2).bool()
                        require(torch.equal(mask[:,0],allowed),"actual causal/companion mask changed")
                        embed=kwargs.get("position_embeddings")
                        require(len(rotary_record)==1 and isinstance(embed,tuple) and len(embed)==2 and
                                torch.equal(embed[0],rotary_values[0][0]) and
                                torch.equal(embed[1],rotary_values[0][1]),
                                "actual rotary consumer changed")
                        current=cache.layers[i]
                        observed={axis:old.tensor_hash(getattr(current,axis)) for axis in ("keys","values")}
                        require(observed==expected[i],"attention consumed wrong historical cache")
                        explicit={"target_other_K_sha256":other_hash(current.keys[target],*spans["latest"]),
                                  "target_other_V_sha256":other_hash(current.values[target],latest_slot,latest_slot+1),
                                  "latest_K_sha256":old.tensor_hash(current.keys[target,:,spans["latest"][0]:spans["latest"][1],:]),
                                  "selected_V_sha256":old.tensor_hash(current.values[target,:,latest_slot,:]),
                                  "companion_K_sha256":old.tensor_hash(current.keys[[j for j in range(4) if j!=target]]),
                                  "companion_V_sha256":old.tensor_hash(current.values[[j for j in range(4) if j!=target]])}
                        require(explicit["target_other_K_sha256"]==native_segments[i]["target_other_K"] and
                                explicit["target_other_V_sha256"]==native_segments[i]["target_other_V"] and
                                explicit["latest_K_sha256"]==
                                (old.tensor_hash(candidates["latest"][i]) if want_k else native_segments[i]["latest_K"]) and
                                explicit["selected_V_sha256"]==
                                (snap[i]["donor_sha256"] if want_donor else snap[i]["native_sha256"]) and
                                explicit["companion_K_sha256"]==native_segments[i]["companion_K"] and
                                explicit["companion_V_sha256"]==native_segments[i]["companion_V"],
                                "actual attention consumed wrong selected/unselected K/V")
                        before_rows[i]={"key_sha256":observed["keys"],"value_sha256":observed["values"],
                                        "mask_sha256":old.tensor_hash(mask),**explicit}
                    def before_output(_module,_args,i=index):
                        require(cache.get_seq_length(i)==end,"S K/V not appended before attention output")
                        current=cache.layers[i]
                        require({axis:old.tensor_hash(getattr(current,axis)[:,:,:width,:])
                                 for axis in ("keys","values")}==expected[i],
                                "historical cache mutated during suffix")
                        after_rows[i]={"companion_suffix_K_sha256":old.tensor_hash(
                                           current.keys[[j for j in range(4) if j!=target],:,width:end,:]),
                                       "companion_suffix_V_sha256":old.tensor_hash(
                                           current.values[[j for j in range(4) if j!=target],:,width:end,:]),
                                       "historical_digest":expected[i]}
                    local.extend((layer.self_attn.register_forward_pre_hook(before_attention,with_kwargs=True),
                                  layer.self_attn.o_proj.register_forward_pre_hook(before_output)))
                try:
                    with torch.inference_mode():logits=model(**suffix).logits[:,-1,:].detach().float().cpu()
                finally:
                    for hook in local:hook.remove()
                torch.cuda.synchronize(device)
                require(logits.shape[0]==4 and len(rotary_record)==1 and
                        len(before_rows)==len(after_rows)==LAYERS and counts["vision_forwards"]==2,
                        "full-batch actual suffix consumer incomplete")
                if kind=="native":
                    err=max(float((rotary_values[0][j][target].detach().cpu()-
                                   destination["destination_latest"][axis][:SUFFIX]).abs().max())
                            for j,axis in enumerate(("cos","sin")))
                    require(err<=TOL,"destination first five differ from native S")
                    old._write_new(out/"destination-s-check.json",{
                        "max_abs_error":err,"destination":binding(out/"destination-phase-raw.json")})
                companion=[{name:after_rows[i][name] for name in
                            ("companion_suffix_K_sha256","companion_suffix_V_sha256")}
                           for i in range(LAYERS)]
                if kind=="native":native_companions=companion
                else:require(companion==native_companions,"companion suffix K/V changed")
                cell_dir=out/"cells"/f"{step:02d}-{kind}";cell_dir.mkdir(parents=True)
                vec=logits[target].clone();torch.save(vec,cell_dir/"vocabulary.pt")
                if kind=="native":torch.save(logits,cell_dir/"full-batch-vocabulary.pt")
                consumer=old._write_new(cell_dir/"consumer-raw.json",{
                    **active["observed"],"rotary_output_hashes":rotary_record,
                    "before_attention":before_rows,"after_attention":after_rows,
                    "expected_historical_digest":expected})
                require(vec.ndim==1 and torch.isfinite(vec).all(),"invalid vocabulary vector")
                top=torch.topk(vec,2)
                cell={"kind":kind,"vector":binding(cell_dir/"vocabulary.pt"),"consumer":consumer,
                      "top2_ids":top.indices.tolist(),"top2_logits":top.values.tolist(),
                      "top2_gap":float(top.values[0]-top.values[1]),
                      "logsumexp":float(torch.logsumexp(vec,-1)),
                      "counts_after":dict(counts),"allocated_gpu_seconds_after":time.monotonic()-started}
                if kind=="native":
                    cell["full_batch_vector"]=binding(cell_dir/"full-batch-vocabulary.pt")
                    errors=(logits-source_logits).abs().amax(dim=1).tolist()
                    require(max(errors)<=TOL,"cached/full native four-row mismatch")
                    cell["full_native_max_abs_error_by_batch_index"]=errors
                    old_reference=torch.load(old.bound(c["reference_native_vector"]),map_location="cpu",weights_only=True)
                    ref_error=float((vec-old_reference).abs().max())
                    require(ref_error<=TOL,"native anchor differs from accepted vector")
                    cell["accepted_native_max_abs_error"]=ref_error
                    native_vector=vec.clone()
                elif kind=="phase":
                    old_reference=torch.load(old.bound(c["reference_phase_vector"]),map_location="cpu",weights_only=True)
                    ref_error=float((vec-old_reference).abs().max())
                    require(ref_error<=TOL,"K+9 anchor differs from accepted vector")
                    cell["accepted_phase_max_abs_error"]=ref_error
                    phase_vector=vec.clone()
                elif kind=="v_sham":
                    error=float((vec-phase_vector).abs().max())
                    require(error<=TOL,"K+9 identity V sham differs from phase anchor")
                    cell["phase_sham_max_abs_error"]=error;sham_passed=True
                done.append({"kind":kind,"record":old._write_new(cell_dir/"cell.json",cell)})
                old._write_new(out/f"checkpoint-{step:02d}.json",{
                    "completed":done,"counts":dict(counts),
                    "allocated_gpu_seconds":time.monotonic()-started})
            active=None
            require(time.monotonic()-started<360,"first-case cap including setup reached")
        require(counts=={"model_forwards":7,"vision_forwards":2} and len(done)==6,
                "finite first-case calls incomplete")
        cost={"allocated_gpu_seconds":time.monotonic()-started,
              "rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
              "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),**counts}
        require(cost["allocated_gpu_seconds"]<360 and cost["allocated_gpu_seconds"]<900 and
                m["budget"]["sequence_cumulative_prior_gpu_hours"]+
                cost["allocated_gpu_seconds"]/3600<8,"first-case/package/sequence cap reached")
        pilot=old._write_new(out/"pilot.json",{
            "schema":"recurrence_y1_value_phase.pilot.v1","status":"candidate_complete",
            "protocol":binding(PROTOCOL),"manifest":binding(MANIFEST),
            "preflight":binding(out.parent/"preflight.json"),
            "producer":binding(Path(__file__)),"effective_identity":identity,
            "input_identity_sha256":old.digest(old.input_identity(batch)),
            "source_planning_replanned":planning.get("replanned_image_plan"),
            "full_native":done[0]["record"],"prefill":binding(out/"prefill-raw.json"),
            "destination_phase":binding(out/"destination-phase-raw.json"),
            "destination_s_check":binding(out/"destination-s-check.json"),
            "phase_qualification":binding(out/"phase-qualification.json"),
            "completed":done,"cost":cost,"case_id":CASE_ID,"target_index":target})
        terminal={"status":"candidate_complete","terminal":True,"pilot":pilot,
                  "completed_cells":len(done),"cost":cost,"case_id":CASE_ID}
    except BaseException as exc:
        terminal={"status":"technical_invalid","terminal":True,"error":repr(exc),
                  "traceback":traceback.format_exc(),"completed":done,
                  "cost":{"allocated_gpu_seconds":time.monotonic()-started,
                          "rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)) if device is not None else 0,
                          "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)) if device is not None else 0,
                          **counts}}
    finally:
        for hook in handles:hook.remove()
    terminal["artifact_bytes_before_receipt"]=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    old._write_new(out/"receipt.json",terminal)
    if terminal["status"]!="candidate_complete":raise RuntimeError(terminal["error"])
    print(json.dumps({"status":terminal["status"],"cost":terminal["cost"]}))


def cold(out):
    m,c,source,receipt=contract();target=source["batch_index"]
    pre=json.loads((out.parent/"preflight.json").read_text())
    require(pre["status"]=="cpu_qualified_before_gpu" and
            pre["protocol"]==binding(PROTOCOL) and pre["manifest"]==binding(MANIFEST) and
            pre["cpu_patch_fixture"]["status"]=="passed", "cold CPU preflight changed")
    for item in pre["direct_source_captures"]:
        old.bound(item["maintained"]);old.bound(item["capture"])
    term=json.loads((out/"receipt.json").read_text())
    require(term["status"]=="candidate_complete" and term["terminal"] and
            term["case_id"]==CASE_ID and term["completed_cells"]==6 and
            term["cost"]["model_forwards"]==7 and term["cost"]["vision_forwards"]==2 and
            term["cost"]["allocated_gpu_seconds"]<360,"cold terminal/cost invalid")
    pilot=json.loads(old.bound(term["pilot"]).read_text())
    identity,saved=pilot["effective_identity"],receipt["identity"]
    require(pilot["producer"]==binding(Path(__file__)) and
            pilot["preflight"]==binding(out.parent/"preflight.json") and
            pilot["protocol"]==binding(PROTOCOL) and pilot["manifest"]==binding(MANIFEST) and
            pilot["case_id"]==CASE_ID and pilot["target_index"]==target and
            [x["kind"] for x in pilot["completed"]]==["full_native",*KINDS] and
            {k:v for k,v in identity.items() if k!="loader_source"}==
            {k:v for k,v in saved.items() if k!="loader_source"} and
            all(identity["loader_source"][k]==saved["loader_source"][k]
                for k in ("sha256","size_bytes")),"cold producer/model/queue changed")
    raw=json.loads(Path(source["source_bindings"]["raw"]["path"]).read_text())["rows"]
    offset=source["geometry"]["first_y1_raw_offset"]
    pad=int(old.AutoTokenizer.from_pretrained(old.BASE,local_files_only=True).pad_token_id)
    tails=old._prefix_tokens(raw,offset,pad)
    histories=[list(prompt)+tail for prompt,tail in zip(
        receipt["input_identity"]["prompt_token_ids"],tails,strict=True)]
    from src.qwen.native import padded_histories
    ids,mask=padded_histories(histories,pad_token_id=pad)
    grid=torch.tensor(receipt["input_identity"]["image_grids"],dtype=torch.long)
    positions,_=old._ConfigOnlyRope().get_rope_index(ids,grid,None,mask)
    require(ids.shape==mask.shape==(4,source["geometry"]["source_step_full_batch_width"]) and
            positions.shape==(3,4,ids.shape[1]) and
            {"input_ids":old.tensor_hash(ids),"attention_mask":old.tensor_hash(mask),
             "position_ids":old.tensor_hash(positions),
             "cache_position":old.tensor_hash(torch.arange(ids.shape[1]))}==
            pre["full_source_input_hashes"],"cold original full-source consumer differs")
    full=json.loads(old.bound(pilot["full_native"]).read_text())
    full_obs=json.loads(old.bound(full["consumer"]).read_text())
    full_vectors=torch.load(old.bound(full["full_batch_vector"]),map_location="cpu",weights_only=True)
    require(full_vectors.ndim==2 and full_vectors.shape[0]==4 and
            torch.equal(full_vectors[target],torch.load(old.bound(full["vector"]),map_location="cpu",weights_only=True)) and
            all(x["passed"] for x in full["trace_parity"] if x["status"]=="active_trace_parity") and
            [x["status"]=="ended_before_source_step" for x in full["trace_parity"]]==pre["ended_companions"] and
            full_obs["input_ids"]==ids.tolist() and full_obs["attention_mask"]==mask.tolist() and
            full_obs["position_ids"]==positions.tolist() and
            full_obs["cache_position"]==list(range(ids.shape[1])) and
            full_obs["media_present"] and not full_obs["use_cache"] and
            not full_obs["past_cache_present"],"cold original source vector/consumer differs")
    prefill=json.loads(old.bound(pilot["prefill"]).read_text())
    blocks=torch.load(old.bound(prefill["vector"]),map_location="cpu",weights_only=True)
    phase=json.loads(old.bound(pilot["phase_qualification"]).read_text())
    rebuilt,candidates=old.qualify_phase(blocks)
    require(rebuilt==phase["records"] and len(rebuilt)==LAYERS and
            all(v["qualified"] for row in rebuilt for k,v in row.items() if k!="layer"),
            "cold all-layer phase oracle differs")
    dest=json.loads(old.bound(pilot["destination_phase"]).read_text())
    old.bound(dest["vector"])
    scheck=json.loads(old.bound(pilot["destination_s_check"]).read_text())
    require(dest["fp64_full_destination_max_abs_error"]<=TOL and
            dest["earlier_matches_latest_max_abs_error"]<=TOL and
            scheck["max_abs_error"]<=TOL,"cold actual destination/S phase differs")
    expected_donor=[old.tensor_hash(layer["earlier"]["native_v"][:,5,:]) for layer in blocks["layers"]]
    expected_native=[old.tensor_hash(layer["latest"]["native_v"][:,5,:]) for layer in blocks["layers"]]
    require(expected_donor==prefill["donor_V_sha256"] and
            expected_native==prefill["native_latest_V_sha256"] and
            all(layer["earlier"]["native_v"].shape==layer["latest"]["native_v"].shape==(HEADS,9,DIM)
                for layer in blocks["layers"]),"cold saved donor/native V differs")
    cells={x["kind"]:json.loads(old.bound(x["record"]).read_text()) for x in pilot["completed"]}
    vectors={kind:torch.load(old.bound(cell["vector"]),map_location="cpu",weights_only=True)
             for kind,cell in cells.items()}
    native_full=torch.load(old.bound(cells["native"]["full_batch_vector"]),map_location="cpu",weights_only=True)
    errors=(native_full-full_vectors).abs().amax(dim=1).tolist()
    require(max(errors)<=TOL and errors==cells["native"]["full_native_max_abs_error_by_batch_index"] and
            torch.equal(native_full[target],vectors["native"]) and
            float((vectors["native"]-torch.load(old.bound(c["reference_native_vector"]),map_location="cpu",weights_only=True)).abs().max())<=TOL and
            float((vectors["phase"]-torch.load(old.bound(c["reference_phase_vector"]),map_location="cpu",weights_only=True)).abs().max())<=TOL and
            float((vectors["v_sham"]-vectors["phase"]).abs().max())<=TOL,
            "cold four-row/native/phase/sham endpoint mismatch")
    width=ids.shape[1]-SUFFIX;span=source["geometry"]["physical_full_batch_padded"]["latest"]
    native_companions=None
    for kind in KINDS:
        cell=cells[kind];obs=json.loads(old.bound(cell["consumer"]).read_text());vec=vectors[kind]
        require(vec.ndim==1 and vec.numel()==full_vectors.shape[-1] and torch.isfinite(vec).all() and
                torch.topk(vec,2).indices.tolist()==cell["top2_ids"] and
                math.isclose(float(torch.logsumexp(vec,-1)),cell["logsumexp"],abs_tol=1e-6) and
                obs["input_ids"]==ids[:,width:].tolist() and
                obs["attention_mask"]==mask.tolist() and
                obs["position_ids"]==positions[:,:,width:].tolist() and
                obs["cache_position"]==list(range(width,ids.shape[1])) and
                not obs["media_present"] and obs["use_cache"] and obs["past_cache_present"] and
                len(obs["before_attention"])==len(obs["after_attention"])==len(obs["expected_historical_digest"])==LAYERS and
                len(obs["rotary_output_hashes"])==1,"cold suffix vector/input/consumer differs")
        companions=[{name:obs["after_attention"][str(i)][name] for name in
                     ("companion_suffix_K_sha256","companion_suffix_V_sha256")}
                    for i in range(LAYERS)]
        if kind=="native":native_companions=companions
        else:require(companions==native_companions,"cold companion suffix K/V changed")
        want_k=kind in ("phase","v_sham","donor_phase")
        want_donor=kind in ("donor_native","donor_phase")
        for i in range(LAYERS):
            before=obs["before_attention"][str(i)];after=obs["after_attention"][str(i)]
            base=prefill["native_segments"][i];digest=obs["expected_historical_digest"][i]
            require(before["key_sha256"]==after["historical_digest"]["keys"]==digest["keys"] and
                    before["value_sha256"]==after["historical_digest"]["values"]==digest["values"] and
                    before["target_other_K_sha256"]==base["target_other_K"] and
                    before["target_other_V_sha256"]==base["target_other_V"] and
                    before["latest_K_sha256"]==
                    (old.tensor_hash(candidates["latest"][i]) if want_k else base["latest_K"]) and
                    before["selected_V_sha256"]==
                    (expected_donor[i] if want_donor else expected_native[i]) and
                    before["companion_K_sha256"]==base["companion_K"] and
                    before["companion_V_sha256"]==base["companion_V"],
                    f"cold all-layer actual {kind} K/V consumer mismatch at layer {i}")
    return {"status":"passed","pilot":binding(out/"pilot.json"),
            "terminal":binding(out/"receipt.json"),"full_batch_native_max_abs_errors":errors,
            "actual_consumer_layers":LAYERS,"cells":list(KINDS),
            "model_forwards":7,"vision_forwards":2,
            "allocated_gpu_seconds":term["cost"]["allocated_gpu_seconds"]}


def tv(p,q):
    return float((p-q).abs().sum()/2)


def reduce(out):
    m,c,source,_=contract()
    readback=cold(out)
    pilot=json.loads((out/"pilot.json").read_text())
    cells={x["kind"]:json.loads(Path(x["record"]["path"]).read_text()) for x in pilot["completed"]}
    logits={kind:torch.load(Path(cell["vector"]["path"]),map_location="cpu",weights_only=True).double()
            for kind,cell in cells.items()}
    ids={name:c["roles"][name]["coordinate_token_id"] for name in ("earlier","latest","current")}
    margin={kind:float(vec[ids["earlier"]]-vec[ids["latest"]])
            for kind,vec in logits.items() if kind in KINDS}
    delta0=margin["donor_native"]-margin["native"]
    delta1=margin["donor_phase"]-margin["phase"]
    interaction=delta1-delta0
    guard=m["decision"]["numerical_guard_nats"]
    near=abs(delta1-m["decision"]["shifted_donor_effect_min_nats"])<=guard or abs(
         interaction-m["decision"]["phase_by_V_interaction_min_nats"])<=guard
    category=("numerical_HOLD" if near else "directional_amplification" if
              delta1>m["decision"]["shifted_donor_effect_min_nats"] and
              interaction>m["decision"]["phase_by_V_interaction_min_nats"] else
              "shared_prediction_nonpass")
    summaries={}
    for kind,vec in logits.items():
        logp=torch.log_softmax(vec,-1);p=logp.exp();top=torch.topk(vec,2)
        roles={name:{"token_id":int(token),"logit":float(vec[token]),
                     "probability":float(p[token]),"log_probability":float(logp[token]),
                     "rank":int((vec>vec[token]).sum())+1}
               for name,token in ids.items()}
        summaries[kind]={"vector":cells[kind]["vector"],"winner":int(top.indices[0]),
                         "runner":int(top.indices[1]),"gap":float(top.values[0]-top.values[1]),
                         "roles":roles}
    prob={kind:torch.softmax(vec,-1) for kind,vec in logits.items()}
    summary={"schema":"recurrence_y1_value_phase.first_case_reduction.v1",
             "status":"candidate","case_id":CASE_ID,"category":category,
             "decision":m["decision"],"margin_nats":margin,
             "delta0_native_K_nats":delta0,"delta1_phase_K_nats":delta1,
             "interaction_nats":interaction,
             "donor_tv_to_same_K":{"donor_native":tv(prob["donor_native"],prob["native"]),
                                   "donor_phase":tv(prob["donor_phase"],prob["phase"])},
             "cells":summaries,"cold_readback":readback,"cost":pilot["cost"],
             "held_case_ids":m["held_case_ids"],
             "protocol":binding(PROTOCOL),"manifest":binding(MANIFEST)}
    old._write_new(out/"reduction.json",summary)
    print(json.dumps({"category":category,"margin_nats":margin,
                      "delta0":delta0,"delta1":delta1,"interaction":interaction,
                      "gpu_seconds":pilot["cost"]["allocated_gpu_seconds"]}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","run","readback","reduce"))
    parser.add_argument("--output",type=Path,default=ROOT)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.mode=="preflight":preflight(args.output)
    elif args.mode=="run":run(args.output,args.device)
    elif args.mode=="readback":print(json.dumps(cold(args.output)))
    else:reduce(args.output)


if __name__=="__main__":main()
