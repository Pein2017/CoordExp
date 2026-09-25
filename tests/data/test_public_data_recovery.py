"""Recovery capability tests, not historical producer existence tests."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

from PIL import Image
import pytest

from public_data import recover_coco as recovery
from public_data.coco_records import apply_annotation_edit, pixel_record, smart_resize

REAL_ROOT = recovery.ROOT
MROOT = recovery.MANIFEST_ROOT
BASE = recovery.BASE
VIEW = recovery.VIEW


def encoded(value, *, compact=False):
    kwargs = {"separators": (",", ":")} if compact else {}
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, **kwargs)+"\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Two real ZIP/JPEG inputs with independently specified expected rows."""
    root = tmp_path/"repo"; (root/"public_data").mkdir(parents=True)
    for name in ("recover_coco.py", "coco_records.py"):
        (root/"public_data"/name).write_bytes((REAL_ROOT/"public_data"/name).read_bytes())
    raw = tmp_path/"raw"; raw.mkdir()
    jpeg = io.BytesIO(); Image.new("RGB", (32,32), (32,96,144)).save(jpeg, format="JPEG")
    with Image.open(io.BytesIO(jpeg.getvalue())) as im:
        resized = io.BytesIO(); im.convert("RGB").resize((1024,1024), Image.Resampling.LANCZOS).save(resized, format="JPEG")
    docs, pixels, views, canaries, edits = {}, {}, {}, [], []
    for split in ("train", "val"):
        member = f"{split}2017/000000000001.jpg"
        with zipfile.ZipFile(raw/(split+"2017.zip"), "w") as z:
            z.writestr(member, jpeg.getvalue())
        docs[split] = {"images": [{"id":1,"width":32,"height":32,"file_name":"000000000001.jpg"}],
            "categories":[{"id":1,"name":"person"}], "annotations":[
                {"id":100,"image_id":1,"category_id":1,"bbox":[1,2,7,8],"iscrowd":0},
                {"id":101,"image_id":1,"category_id":1,"bbox":[1,2,7,8],"iscrowd":1},
                {"id":102,"image_id":1,"category_id":1,"bbox":[-3,2,7,8],"iscrowd":0},
                {"id":103,"image_id":1,"category_id":1,"bbox":[1,2,0,8],"iscrowd":0}]}
        row={"images":["images/"+member],"objects":[{"bbox_2d":[32,64,256,320],"desc":"person",
             "category_id":1,"category_name":"person","coco_ann_id":100}],"width":1024,"height":1024,
             "image_id":1,"file_name":"images/"+member,"metadata":{"source":"coco2017","split":split}}
        pixels[f"{BASE}/{split}.jsonl"] = encoded(row)
        p=copy.deepcopy(row);p["images"]=["../rescale_32_1024_bbox/images/"+member]
        n=copy.deepcopy(p);n["objects"][0]["bbox_2d"]=[31,62,250,313]
        if split=="train":
            before=digest(encoded(n)); n["objects"][0]["bbox_2d"][0]=30
            edits.append({"split":split,"image_id":1,"surface":"norm1000","before_sha256":before,
                "remove_annotation_ids":[],"upsert_objects":copy.deepcopy(n["objects"]),"object_order":[100],"after_sha256":digest(encoded(n))})
        c=copy.deepcopy(n);c["objects"][0]["bbox_2d"]=[f"<|coord_{v}|>" for v in n["objects"][0]["bbox_2d"]]
        views.update({f"{VIEW}/{split}.jsonl":encoded(p), f"{VIEW}/{split}.norm.jsonl":encoded(n,compact=True),
                      f"{VIEW}/{split}.coord.jsonl":encoded(c,compact=True)})
        canaries.append({"split":split,"member":member,"width":1024,"height":1024,
            "raw_sha256":digest(jpeg.getvalue()),"resized_sha256":digest(resized.getvalue())})
    with zipfile.ZipFile(raw/"annotations_trainval2017.zip", "w") as z:
        for split, doc in docs.items():z.writestr(f"annotations/instances_{split}2017.json",json.dumps(doc))
    delta=root/"public_data/coco_annotation_delta.json";delta.write_bytes(encoded({"schema_version":1,"edits":edits}))
    inputs=[{"path":p.name,"sha256":digest(p.read_bytes()),"size_bytes":p.stat().st_size,
             "url":"http://images.cocodataset.org/zips/"+p.name} for p in sorted(raw.iterdir())]
    paths={}
    for recipe, outputs in [("pixel-v1",pixels),("curated-len12000-v1",views)]:
        value=json.loads((MROOT/"coco"/("rescale_32_1024_bbox.json" if recipe=="pixel-v1" else "rescale_32_1024_bbox_len12000.json")).read_text())
        value["checksums"]={"scope":"jsonl_training_samples_only","algorithm":"sha256","files":[
            {"path":name,"sha256":digest(data),"size_bytes":len(data),"records":1} for name,data in outputs.items()]}
        value["recovery"].update(raw_archives=inputs,image_canaries=canaries)
        if recipe!="pixel-v1":
            value["recovery"]["annotation_delta"]={"path":"public_data/coco_annotation_delta.json",
                "sha256":digest(delta.read_bytes()),"size_bytes":delta.stat().st_size}
        p=root/(recipe+".json");p.write_bytes(encoded(value));paths[recipe]=p
    monkeypatch.setattr(recovery,"ROOT",root)
    return root,raw,paths,pixels,views


@pytest.mark.parametrize("recipe",["pixel-v1","curated-len12000-v1"])
def test_complete_restoration_has_exact_bytes_and_real_reader(bundle,tmp_path,recipe):
    root,raw,paths,pixels,views=bundle;dest=tmp_path/"restored"
    result=recovery.recover(paths[recipe],raw,dest)
    assert result["status"]=="restored" and result["images_written"]==2
    expected=pixels if recipe=="pixel-v1" else views
    for name,data in expected.items():assert (dest/name).read_bytes()==data
    assert not (dest/".recovery-incomplete").exists()
    verified=recovery.verify_materialization(paths[recipe],dest)
    assert verified["jsonl_files_verified"]==len(expected)
    assert verified["sampled_image_reads"]==len(expected)
    if recipe!="pixel-v1":assert verified["reader_rows"]==2


@pytest.mark.parametrize("recipe",["pixel-v1","curated-len12000-v1"])
def test_dry_run_validates_inputs_and_writes_nothing(bundle,tmp_path,recipe):
    _,raw,paths,_,_=bundle;dest=tmp_path/"must-not-exist"
    result=recovery.recover(paths[recipe],raw,dest,dry_run=True)
    assert result["writes"]==0 and result["raw_archives_verified"]==3
    assert result["image_canaries_verified"]==2 and not dest.exists()


def test_jsonl_only_is_not_image_qualification(bundle):
    _,raw,paths,_,_=bundle
    (raw/"train2017.zip").unlink()
    result=recovery.recover(paths["curated-len12000-v1"],raw,None,jsonl_only=True)
    assert result["status"]=="jsonl_reconstruction_passed"
    assert result["raw_archives_verified"]==1 and result["images_written"]==0
    assert "not qualified" in result["scope"]


@pytest.mark.parametrize("case",["missing_raw","raw_tamper","delta_tamper","wrong_environment","wrong_canary","missing_producer","occupied","symlink_raw"])
def test_preflight_failures_publish_nothing(bundle,tmp_path,case):
    root,raw,paths,_,_=bundle;path=paths["curated-len12000-v1"];dest=tmp_path/"destination"
    if case=="missing_raw":(raw/"train2017.zip").unlink()
    elif case=="raw_tamper":(raw/"train2017.zip").write_bytes(b"changed")
    elif case=="delta_tamper":(root/"public_data/coco_annotation_delta.json").write_text("{}")
    elif case=="missing_producer":(root/"public_data/coco_records.py").unlink()
    elif case=="occupied":dest.mkdir();(dest/"mine").write_text("preserve")
    elif case=="symlink_raw":
        p=raw/"train2017.zip";original=tmp_path/"original.zip";p.rename(original);p.symlink_to(original)
    else:
        v=json.loads(path.read_text())
        if case=="wrong_environment":v["recovery"]["environment"]["pillow"]="0"
        else:v["recovery"]["image_canaries"][0]["resized_sha256"]="0"*64
        path.write_bytes(encoded(v))
    with pytest.raises(ValueError):recovery.recover(path,raw,dest)
    if case=="occupied":assert (dest/"mine").read_text()=="preserve"
    else:assert not dest.exists()


def test_output_mismatch_keeps_incomplete_marker_without_success(bundle,tmp_path):
    _,raw,paths,_,_=bundle;path=paths["pixel-v1"];dest=tmp_path/"partial"
    v=json.loads(path.read_text());v["checksums"]["files"][0]["sha256"]="0"*64;path.write_bytes(encoded(v))
    with pytest.raises(ValueError,match="reconstructed JSONL"):
        recovery.recover(path,raw,dest)
    assert (dest/".recovery-incomplete").exists() and not (dest/"recovery.json").exists()
    with pytest.raises(ValueError,match="incomplete"):
        recovery.verify_materialization(path,dest)


@pytest.mark.parametrize("value",["",".","..","../escape","/absolute","a/../b","a//b","./a","a\nb","a\\b"])
def test_unsafe_path_rejected(tmp_path,value):
    with pytest.raises(ValueError):recovery.local(tmp_path,value,must_exist=False)


def test_edit_cannot_silently_change_context(bundle):
    _,_,_,_,_=bundle
    with pytest.raises(ValueError,match="input identity"):
        apply_annotation_edit({"objects":[]},{"before_sha256":"0"*64})


def test_historical_manifests_fail_before_input_access(tmp_path):
    historical=[]
    for p in (MROOT/"coco").rglob("*.json"):
        if recovery.load_manifest(p)["support"]=="historical":historical.append(p)
    assert len(historical)==8
    for p in historical:
        with pytest.raises(ValueError,match="historical identity only"):
            recovery.recover(p,tmp_path/"missing-raw",tmp_path/"no-output")
    assert not (tmp_path/"no-output").exists()


def test_all_manifest_origins_are_exact_and_historical_hashes_unchanged():
    for p in (MROOT/"coco").rglob("*.json"):
        value=recovery.load_manifest(p);origin=value["origin"]
        original=subprocess.check_output(["git","-C",str(REAL_ROOT),"show",origin["commit"]+":"+origin["manifest_path"]])
        assert digest(original)==origin["manifest_sha256"]
        if value["support"]=="historical":
            assert value["checksums"]==json.loads(original)["checksums"]
            assert "producer_script" not in value and "command" not in value
    current=json.loads((MROOT/"coco/rescale_32_1024_bbox_len12000.json").read_text())
    assert current["asset_version"]=="curated-observed-2026-09-25"


def test_current_coco_config_inputs_have_current_recovery_owner():
    current=[recovery.load_manifest(p) for p in (MROOT/"coco").rglob("*.json")]
    prefixes=["/data/CoordExp/"+v["relative_path"]+"/" for v in current if v["support"]=="current"]
    assert len(prefixes)==2
    inputs=set()
    for p in (REAL_ROOT/"configs").rglob("*.yaml"):
        inputs.update(re.findall(r"/data/CoordExp/public_data/coco/[^\s'\"]+\.jsonl",p.read_text()))
    assert inputs
    assert all(any(s.startswith(prefix) for prefix in prefixes) for s in inputs)


def test_retired_operational_routes_are_not_current_guidance():
    for name in ("REPO_HYGIENE.md","OUTPUT_SYNC_AND_DATA_PROVENANCE.md"):
        assert not (REAL_ROOT/"docs/standards"/name).exists()
    text="\n".join(p.read_text() for p in (REAL_ROOT/"docs").rglob("*.md"))
    assert "absorb_output_remote_into_outputs.py" not in text
    assert "are preserved under `openspec/changes/archive/`" not in text
    assert "| Superseded documentation/provenance | `docs/history/`" not in text
    assert not (REAL_ROOT/"public_data/run.sh").exists()


def test_fixed_resize_and_border_geometry_are_explicit():
    assert smart_resize(height=32,width=32)==(1024,1024)
    row={"images":["images/train2017/a.jpg"],"width":32,"height":32,
         "objects":[{"bbox_2d":[31.99,31.99,32.,32.]}]}
    assert pixel_record(row)["objects"][0]["bbox_2d"]==[1022,1022,1023,1023]



def test_manifest_change_during_recovery_cannot_publish_success(bundle,tmp_path,monkeypatch):
    _,raw,paths,_,_=bundle;path=paths["pixel-v1"];dest=tmp_path/"raced"
    original=recovery.pixel_record
    def changed(row):
        path.write_text(path.read_text()+"\n")
        return original(row)
    monkeypatch.setattr(recovery,"pixel_record",changed)
    with pytest.raises(ValueError,match="manifest changed"):
        recovery.recover(path,raw,dest)
    assert (dest/".recovery-incomplete").exists() and not (dest/"recovery.json").exists()


def test_unexpected_receipt_occupancy_is_not_overwritten(bundle,tmp_path,monkeypatch):
    _,raw,paths,_,_=bundle;dest=tmp_path/"occupied-receipt"
    original=recovery.pixel_record
    def occupy(row):
        target=dest/"recovery.json"
        if not target.exists():target.write_text("another writer")
        return original(row)
    monkeypatch.setattr(recovery,"pixel_record",occupy)
    with pytest.raises(FileExistsError):recovery.recover(paths["pixel-v1"],raw,dest)
    assert (dest/"recovery.json").read_text()=="another writer"
    assert (dest/".recovery-incomplete").exists()



def test_current_guides_have_no_dangling_local_command_targets():
    """Inspect fenced/inline commands, not just Markdown hyperlink existence."""
    documents=[REAL_ROOT/"README.md",REAL_ROOT/"public_data/README.md",MROOT/"README.md",
               *sorted((REAL_ROOT/"docs").rglob("*.md"))]
    checked=0
    for document in documents:
        for line in document.read_text().splitlines():
            for match in re.finditer(r"\bpython(?:3)?(?:\s+-B)?\s+(-m\s+)?((?:src|scripts|public_data|probes)[\w./-]+)",line):
                module,name=match.groups()
                if module:
                    path=REAL_ROOT/(name.replace(".","/")+".py")
                    assert path.is_file() or path.with_suffix("").joinpath("__main__.py").is_file(),(document,name)
                else:
                    assert (REAL_ROOT/name).is_file(),(document,name)
                checked+=1
            for name in re.findall(r"(?:\./|\bbash\s+|\bsh\s+)((?:scripts|public_data)/[\w/.-]+\.sh)",line):
                assert (REAL_ROOT/name).is_file(),(document,name)
                checked+=1
    assert checked>=10
