"""Recover the retained fixed COCO assets from bound official ZIP inputs.

This is not a general dataset factory or an old receipt continuation path.
Historical-only manifests cannot execute. No downloads, model loads, overwrites,
legacy config imports or subprocess commands selected by a manifest occur here.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import zipfile
from typing import Any

from PIL import Image, features, __version__ as PILLOW_VERSION
from public_data.coco_records import (raw_records, pixel_record, curated_triplet,
                                      jsonl_bytes, compact_jsonl_bytes)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "manifests/public_data_provenance"
BASE = "public_data/coco/rescale_32_1024_bbox"
VIEW = "public_data/coco/rescale_32_1024_bbox_len12000"
MODULE = "public_data.recover_coco"


def local(root: Path, name: str, *, must_exist: bool = True) -> Path:
    if not isinstance(name, str) or any(c in name for c in "\0\r\n\\"):
        raise ValueError("invalid relative asset path")
    rel = PurePosixPath(name)
    if not name or rel == PurePosixPath(".") or rel.is_absolute() or ".." in rel.parts or str(rel) != name:
        raise ValueError("asset path must be normalized and relative")
    path = root.joinpath(*rel.parts)
    cursor = root
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("asset path must not traverse a symlink")
    if must_exist and not path.is_file():
        raise ValueError(f"required asset file is absent: {name}")
    return path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_file(root: Path, record: dict[str, Any]) -> Path:
    path = local(root, record["path"])
    if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"asset content identity differs: {record['path']}")
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    import jsonschema
    value = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads((MANIFEST_ROOT / "schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(value)
    if value["support"] == "current":
        recovery = value["recovery"]
        wanted = BASE if recovery["recipe"] == "pixel-v1" else VIEW
        if value["relative_path"] != wanted or recovery["module"] != MODULE:
            raise ValueError("manifest does not identify a supported fixed recipe")
        if not local(ROOT, "public_data/recover_coco.py").is_file() or not local(ROOT, "public_data/coco_records.py").is_file():
            raise ValueError("recovery implementation is absent")
        names = [x["path"] for x in recovery["raw_archives"]]
        if set(names) != {"annotations_trainval2017.zip", "train2017.zip", "val2017.zip"} or len(names) != 3:
            raise ValueError("the exact three raw archives are required")
        output_names = {x["path"] for x in value["checksums"]["files"]}
        expected = {f"{wanted}/{s}.{suffix}" for s in ("train", "val")
                    for suffix in (["jsonl"] if wanted == BASE else ["jsonl", "norm.jsonl", "coord.jsonl"])}
        if output_names != expected or len(output_names) != len(value["checksums"]["files"]):
            raise ValueError("manifest must bind the complete output JSONL set")
        if wanted == VIEW:
            if recovery.get("annotation_delta") is None:
                raise ValueError("current curated view requires its annotation delta")
            check_file(ROOT, recovery["annotation_delta"])
        for row in value["checksums"]["files"]:
            local(ROOT, row["path"], must_exist=False)
    return value


def require_current(manifest: dict[str, Any]) -> None:
    if manifest["support"] != "current":
        raise ValueError("historical identity only: no supported recovery or continuation")


def environment(manifest: dict[str, Any]) -> dict[str, str]:
    actual = {"python": ".".join(platform.python_version_tuple()[:2]),
              "pillow": PILLOW_VERSION, "libjpeg_turbo": features.version_feature("libjpeg_turbo")}
    if actual != manifest["recovery"]["environment"]:
        raise ValueError(f"image recovery environment differs: expected {manifest['recovery']['environment']}, observed {actual}")
    return actual


def load_edits(manifest: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    binding = manifest["recovery"].get("annotation_delta")
    if binding is None:
        return {}
    path = local(ROOT, binding["path"])
    content = path.read_bytes()
    if len(content) != binding["size_bytes"] or hashlib.sha256(content).hexdigest() != binding["sha256"]:
        raise ValueError("annotation delta identity differs")
    value = json.loads(content)
    if value.get("schema_version") != 1 or not isinstance(value.get("edits"), list):
        raise ValueError("unsupported annotation delta")
    result = {}
    for row in value["edits"]:
        key = row["split"], row["image_id"], row["surface"]
        if key in result or key[0] not in {"train", "val"} or key[2] not in {"pixel", "norm1000"}:
            raise ValueError("duplicate or unsupported annotation edit")
        result[key] = row
    return result


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def open_archives(stack: ExitStack, manifest: dict[str, Any], raw_root: Path,
                  *, annotations_only: bool = False) -> tuple[dict[str, zipfile.ZipFile], dict[Path, tuple]]:
    archives, stamps = {}, {}
    for binding in manifest["recovery"]["raw_archives"]:
        if annotations_only and binding["path"] != "annotations_trainval2017.zip":
            continue
        path = local(raw_root, binding["path"])
        start = _stamp(path)
        stream = stack.enter_context(path.open("rb"))
        if start[2] != binding["size_bytes"] or hashlib.file_digest(stream, "sha256").hexdigest() != binding["sha256"]:
            raise ValueError(f"raw archive checksum differs: {path.name}")
        if _stamp(path) != start:
            raise ValueError("raw archive changed during verification")
        stream.seek(0)
        archive = stack.enter_context(zipfile.ZipFile(stream))
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("raw archive contains ambiguous duplicate names")
        archives[path.name], stamps[path] = archive, start
    return archives, stamps


def source_unchanged(stamps: dict[Path, tuple]) -> None:
    if any(_stamp(p) != s for p, s in stamps.items()):
        raise ValueError("raw inputs changed during recovery")


def image_bytes(raw_bytes: bytes, width: int, height: int) -> bytes:
    with Image.open(io.BytesIO(raw_bytes)) as image:
        rgb = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        rgb.save(output, format="JPEG")
        return output.getvalue()


def check_canaries(manifest: dict[str, Any], archives: dict[str, zipfile.ZipFile]) -> int:
    for c in manifest["recovery"]["image_canaries"]:
        data = archives[c["split"] + "2017.zip"].read(c["member"])
        if hashlib.sha256(data).hexdigest() != c["raw_sha256"]:
            raise ValueError("raw image canary differs")
        out = image_bytes(data, c["width"], c["height"])
        if hashlib.sha256(out).hexdigest() != c["resized_sha256"]:
            raise ValueError("resized-image canary differs")
    return len(manifest["recovery"]["image_canaries"])


def recover(manifest_path: Path, raw_root: Path, destination: Path | None,
            *, dry_run: bool = False, jsonl_only: bool = False) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_manifest(manifest_path)
    if manifest_path.read_bytes() != manifest_bytes or json.loads(manifest_bytes) != manifest:
        raise ValueError("manifest changed during validation")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    require_current(manifest)
    runtime = environment(manifest)
    edits = load_edits(manifest)
    if destination is not None:
        destination = destination.absolute()
        if destination.exists() or destination.is_symlink():
            raise ValueError("destination already exists; recovery never overwrites")
        if not destination.parent.is_dir() or destination.parent.resolve() != destination.parent:
            raise ValueError("destination requires an existing non-symlink parent")
        if destination.is_relative_to(raw_root.resolve()):
            raise ValueError("destination must be separate from raw inputs")
    elif not (jsonl_only or dry_run):
        raise ValueError("regeneration requires an absent destination workspace")
    expected = {x["path"]: x for x in manifest["checksums"]["files"]}
    with ExitStack() as stack:
        archives, stamps = open_archives(stack, manifest, raw_root, annotations_only=jsonl_only)
        canaries = 0 if jsonl_only else check_canaries(manifest, archives)
        if dry_run:
            source_unchanged(stamps)
            if manifest_path.read_bytes() != manifest_bytes:
                raise ValueError("manifest changed during recovery")
            return {"status": "dry_run_passed", "writes": 0, "manifest_sha256": manifest_hash,
                    "raw_archives_verified": len(archives), "image_canaries_verified": canaries,
                    "recipe": manifest["recovery"]["recipe"], "environment": runtime,
                    "destination": str(destination) if destination else None,
                    "outputs": sorted(expected), "images": "all retained COCO train/val images from the bound raw archives"}
        if not jsonl_only:
            destination.mkdir()  # Exclusive reservation; a failed destination stays incomplete.
            with (destination / ".recovery-incomplete").open("x") as marker:
                marker.write("No success receipt: do not use partial data.\n")
        hashes = {p: hashlib.sha256() for p in expected}
        sizes, counts = dict.fromkeys(expected, 0), dict.fromkeys(expected, 0)
        used_edits, image_count = set(), 0
        handles = {}
        if not jsonl_only:
            for name in expected:
                p = local(destination, name, must_exist=False)
                p.parent.mkdir(parents=True, exist_ok=True)
                handles[name] = stack.enter_context(p.open("xb"))
        for split in ("train", "val"):
            with archives["annotations_trainval2017.zip"].open(f"annotations/instances_{split}2017.json") as stream:
                document = json.load(stream)
            for raw in raw_records(document, split):
                pixel = pixel_record(raw)
                image_count += 1
                if manifest["recovery"]["recipe"] == "pixel-v1":
                    records = [(f"{BASE}/{split}.jsonl", jsonl_bytes(pixel))]
                else:
                    p, n, c = curated_triplet(pixel, edits)
                    records = [(f"{VIEW}/{split}.jsonl", jsonl_bytes(p)),
                               (f"{VIEW}/{split}.norm.jsonl", compact_jsonl_bytes(n)),
                               (f"{VIEW}/{split}.coord.jsonl", compact_jsonl_bytes(c))]
                    used_edits.update((split, raw["image_id"], surface) for surface in ("pixel", "norm1000")
                                      if (split, raw["image_id"], surface) in edits)
                for name, content in records:
                    hashes[name].update(content); sizes[name] += len(content); counts[name] += 1
                    if not jsonl_only:
                        handles[name].write(content)
                if not jsonl_only:
                    member = raw["images"][0].removeprefix("images/")
                    encoded = archives[split+"2017.zip"].read(member)
                    with Image.open(io.BytesIO(encoded)) as image:
                        if image.size != (raw["width"], raw["height"]):
                            raise ValueError("raw annotation/image dimensions differ")
                    image_path = local(destination, BASE+"/"+raw["images"][0], must_exist=False)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    with image_path.open("xb") as output:
                        output.write(image_bytes(encoded, pixel["width"], pixel["height"]))
            del document
        if set(edits) != used_edits:
            raise ValueError("not every bound annotation edit was consumed")
        for name, record in expected.items():
            if (hashes[name].hexdigest(), sizes[name], counts[name]) != (record["sha256"], record["size_bytes"], record["records"]):
                raise ValueError(f"reconstructed JSONL differs from expected identity: {name}")
        source_unchanged(stamps)
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("manifest changed during recovery")
        for stream in handles.values():
            stream.flush(); os.fsync(stream.fileno())
        result = {"status": "jsonl_reconstruction_passed" if jsonl_only else "restored",
                  "manifest_sha256": manifest_hash, "recipe": manifest["recovery"]["recipe"],
                  "raw_archives_verified": len(archives), "jsonl_files_verified": len(expected),
                  "records": image_count, "annotation_edits": len(used_edits),
                  "images_written": 0 if jsonl_only else image_count,
                  "image_canaries_verified": canaries, "environment": runtime,
                  "scope": "JSONL bytes only; image archives not qualified" if jsonl_only else "all JSONL hashes and image generation completed"}
        if not jsonl_only:
            with local(destination, "recovery.json", must_exist=False).open("x") as receipt:
                receipt.write(json.dumps(result, indent=2)+"\n")
                receipt.flush(); os.fsync(receipt.fileno())
            (destination / ".recovery-incomplete").unlink()
        return result


def verify_materialization(manifest_path: Path, workspace: Path, *, smoke_rows: int = 2) -> dict[str, Any]:
    manifest = load_manifest(manifest_path); require_current(manifest)
    if smoke_rows <= 0:
        raise ValueError("smoke_rows must be positive")
    if (workspace / ".recovery-incomplete").exists():
        raise ValueError("materialization has incomplete recovery marker")
    examples = images = 0
    from src.data.examples import raw_example_from_jsonl_row
    for binding in manifest["checksums"]["files"]:
        path = check_file(workspace, binding)
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if number > smoke_rows:
                    break
                row = json.loads(line)
                if path.name.endswith(".coord.jsonl"):
                    parsed = raw_example_from_jsonl_row(row, jsonl_path=path, row_number=number, raw_line=line)
                    if parsed.image.path != (path.parent/row["images"][0]).resolve():
                        raise ValueError("reader image resolution differs")
                    examples += 1
                image = (path.parent/row["images"][0]).resolve()
                if not image.is_relative_to((workspace/BASE/"images").resolve()):
                    raise ValueError("image reference escapes the declared image store")
                with Image.open(image) as im:
                    im.load()
                    if im.size != (row["width"], row["height"]):
                        raise ValueError("restored image dimensions differ")
                images += 1
    for canary in manifest["recovery"]["image_canaries"]:
        path = local(workspace, BASE+"/images/"+canary["member"])
        if sha256(path) != canary["resized_sha256"]:
            raise ValueError("materialized image canary differs")
    return {"status": "verified", "jsonl_files_verified": len(manifest["checksums"]["files"]),
            "reader_rows": examples, "sampled_image_reads": images,
            "image_canaries_verified": len(manifest["recovery"]["image_canaries"]),
            "scope": "all declared JSONL bytes, bounded reader/image smoke; not a full image checksum census"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "regenerate", "reconstruct-jsonl", "verify"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--raw-archives", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "check":
            paths = [args.manifest] if args.manifest else sorted((MANIFEST_ROOT/"coco").rglob("*.json"))
            values = [load_manifest(p) for p in paths]
            result = {"status": "contracts_valid", "current": sum(v["support"] == "current" for v in values),
                      "historical": sum(v["support"] == "historical" for v in values),
                      "scope": "contract/dependency check, not raw input availability or restoration"}
        elif args.action == "verify":
            if args.manifest is None or args.workspace is None:
                parser.error("verify requires --manifest and --workspace")
            result = verify_materialization(args.manifest, args.workspace.resolve())
        else:
            if args.manifest is None or args.raw_archives is None:
                parser.error("recovery requires --manifest and --raw-archives")
            if args.action == "reconstruct-jsonl" and args.destination is not None:
                parser.error("reconstruct-jsonl is read-only and accepts no destination")
            result = recover(args.manifest, args.raw_archives.resolve(), args.destination,
                             dry_run=args.dry_run, jsonl_only=args.action == "reconstruct-jsonl")
        print(json.dumps(result, sort_keys=True)); return 0
    except Exception as exc:
        print(json.dumps({"status": "HOLD", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
