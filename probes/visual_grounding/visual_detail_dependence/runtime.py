"""Prepare and qualify the fixed Lane B visual-detail contrast.

The admission manifest supplies the current pre-rescaled image and its raw
original counterpart.  Preparation is CPU-only.  Qualification is opt-in and
uses the maintained tied loader with the same prompt and greedy budget for all
three image settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

BASELINE_MAX_PIXELS = 1_048_576
HIGH_MAX_PIXELS = 2_097_152
GRID_FACTOR = 32
RESAMPLE = Image.Resampling.LANCZOS
SETTINGS = ("baseline", "high_detail", "baseline_upsampled")
ASPECT_TOLERANCE = 0.03


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    try:
        return _sha256_bytes(
            json.dumps(
                {"size": list(rgb.size), "mode": rgb.mode},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + rgb.tobytes()
        )
    finally:
        rgb.close()


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_file():
        return {"path": str(path), "kind": "file", "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    raise FileNotFoundError(path)


def _record_path(record: dict[str, Any], *names: str) -> Path | None:
    for name in names:
        value = record.get(name)
        if value:
            return Path(str(value)).expanduser().resolve()
    return None


def _fit_size(width: int, height: int, *, max_pixels: int, factor: int) -> tuple[int, int]:
    """Fit without upsampling, then floor both dimensions to the model grid."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
    fitted_width = max(factor, int(width * scale) // factor * factor)
    fitted_height = max(factor, int(height * scale) // factor * factor)
    while fitted_width * fitted_height > max_pixels:
        if fitted_width >= fitted_height:
            fitted_width -= factor
        else:
            fitted_height -= factor
        if fitted_width < factor or fitted_height < factor:
            raise ValueError("pixel budget cannot fit one model grid cell")
    return fitted_width, fitted_height


def _resize_rgb(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    rgb = source.convert("RGB")
    if rgb.size == size:
        return rgb
    resized = rgb.resize(size, RESAMPLE)
    rgb.close()
    return resized


def _save_png(image: Image.Image, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=6)
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "pixel_sha256": _pixel_hash(image),
        "width": int(image.width),
        "height": int(image.height),
        "grid": [int(image.height // GRID_FACTOR), int(image.width // GRID_FACTOR)],
        "raw_patch_rows": int((image.height // 16) * (image.width // 16)),
        "merged_visual_tokens": int((image.height // GRID_FACTOR) * (image.width // GRID_FACTOR)),
    }


def _image_entry(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    image_id = str(record.get("image_id", record.get("id", record.get("row_id", "unknown"))))
    baseline_path = _record_path(record, "baseline_image_path", "image_path", "image")
    original_path = _record_path(record, "original_image_path", "raw_image_path", "original_path")
    result: dict[str, Any] = {"image_id": image_id, "source_record": record}
    if baseline_path is None:
        result.update({"status": "HOLD", "reason": "missing_baseline_image_path"})
        return result
    if original_path is None:
        result.update({"status": "HOLD", "reason": "missing_original_image_path"})
        return result
    if not baseline_path.is_file() or not original_path.is_file():
        result.update({"status": "HOLD", "reason": "source_image_missing", "baseline_path": str(baseline_path), "original_path": str(original_path)})
        return result

    with Image.open(baseline_path) as opened_baseline, Image.open(original_path) as opened_original:
        baseline = opened_baseline.convert("RGB")
        original = opened_original.convert("RGB")
        baseline_size = baseline.size
        original_size = original.size
        aspect_delta = abs(baseline_size[0] / baseline_size[1] - original_size[0] / original_size[1])
        result.update({
            "baseline_source": _binding(baseline_path),
            "original_source": _binding(original_path),
            "baseline_original_dimensions": list(baseline_size),
            "raw_original_dimensions": list(original_size),
            "source_aspect_delta": aspect_delta,
            "source_aspect_match": aspect_delta <= ASPECT_TOLERANCE,
        })
        if any(value % GRID_FACTOR for value in baseline_size):
            result.update({"status": "HOLD", "reason": "baseline_dimensions_not_grid_aligned"})
            baseline.close(); original.close()
            return result
        if not result["source_aspect_match"]:
            result.update({"status": "HOLD", "reason": "baseline_original_aspect_mismatch"})
            baseline.close(); original.close()
            return result
        high_size = _fit_size(original.width, original.height, max_pixels=HIGH_MAX_PIXELS, factor=GRID_FACTOR)
        genuine = _resize_rgb(original, high_size)
        control_low = _resize_rgb(baseline, baseline_size)
        control = _resize_rgb(control_low, high_size)
        control_low.close()
        try:
            baseline_info = {
                "path": str(baseline_path),
                "file_sha256": _sha256_file(baseline_path),
                "pixel_sha256": _pixel_hash(baseline),
                "width": baseline.width,
                "height": baseline.height,
                "grid": [baseline.height // GRID_FACTOR, baseline.width // GRID_FACTOR],
                "raw_patch_rows": (baseline.height // 16) * (baseline.width // 16),
                "merged_visual_tokens": (baseline.height // GRID_FACTOR) * (baseline.width // GRID_FACTOR),
            }
            high_path = output_dir / "images" / image_id / "high_detail.png"
            control_path = output_dir / "images" / image_id / "baseline_upsampled.png"
            high_info = _save_png(genuine, high_path)
            control_info = _save_png(control, control_path)
            eligible = (
                original.width * original.height > baseline.width * baseline.height
                and high_size[0] * high_size[1] > baseline.width * baseline.height
                and high_size != baseline_size
            )
            result.update({
                "status": "eligible" if eligible else "HOLD",
                "reason": None if eligible else "raw_original_does_not_provide_more_detail_at_fixed_budget",
                "settings": {
                    "baseline": baseline_info,
                    "high_detail": high_info,
                    "baseline_upsampled": control_info,
                },
                "geometry": {
                    "annotation_to_baseline": {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0},
                    "annotation_to_high_detail": {"scale_x": high_size[0] / baseline.width, "scale_y": high_size[1] / baseline.height, "offset_x": 0.0, "offset_y": 0.0},
                    "compositor": "RGB; preserve aspect; PIL.Image.Resampling.LANCZOS; no crop; no pad",
                },
                "attention_length_cost": {
                    name: int(info["merged_visual_tokens"])
                    for name, info in (("baseline", baseline_info), ("high_detail", high_info), ("baseline_upsampled", control_info))
                },
            })
        finally:
            genuine.close(); control.close(); baseline.close(); original.close()
    return result


def prepare_manifest(
    admission_path: Path,
    output_dir: Path,
    *,
    baseline_max_pixels: int = BASELINE_MAX_PIXELS,
    high_max_pixels: int = HIGH_MAX_PIXELS,
) -> dict[str, Any]:
    del baseline_max_pixels  # Baseline pixels are the exact admitted pre-rescaled input.
    if high_max_pixels != HIGH_MAX_PIXELS:
        raise ValueError("Lane B high-detail max_pixels is frozen at 2097152")
    admission = json.loads(admission_path.read_text())
    records = admission.get("images", admission.get("cases", []))
    if not isinstance(records, list) or not records:
        raise ValueError("admission manifest must contain a nonempty images list")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [_image_entry(dict(record), output_dir) for record in records]
    manifest = {
        "schema": "address_readout_pilot.lane_b.v1",
        "status": "prepared",
        "admission": _binding(admission_path),
        "lane": "visual_detail_dependence",
        "settings": SETTINGS,
        "frozen_processing": {
            "baseline": "exact admitted pre-rescaled image; do_resize=false",
            "high_detail": {"max_pixels": HIGH_MAX_PIXELS, "grid_factor": GRID_FACTOR, "resample": "LANCZOS", "upsample": False},
            "baseline_upsampled": "exact admitted baseline resized to the high-detail dimensions",
            "baseline_max_pixels": BASELINE_MAX_PIXELS,
        },
        "entries": entries,
        "counts": {
            "total": len(entries),
            "eligible": sum(entry.get("status") == "eligible" for entry in entries),
            "hold": sum(entry.get("status") == "HOLD" for entry in entries),
        },
    }
    manifest_path = output_dir / "lane-b-manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def qualify_one(
    manifest_path: Path,
    output_dir: Path,
    *,
    ledger: Any | None = None,
    device: str = "cuda:0",
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Run one short greedy model qualification through all three settings."""
    if ledger is None:
        raise RuntimeError("model qualification requires the parent RunLedger")
    if Path(ledger.output).resolve() != output_dir.resolve():
        raise ValueError("qualification output must equal RunLedger.output")
    import torch
    from probes.model_profiles.mature_tied_untied import load_model
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import NativeRequest, prepare_native_inputs

    manifest = json.loads(manifest_path.read_text())
    entry = next((item for item in manifest["entries"] if item.get("status") == "eligible"), None)
    if entry is None:
        raise RuntimeError("Lane B qualification requires one eligible entry")
    record = entry["source_record"]
    chat_text = record.get("chat_text") or record.get("prompt")
    if not isinstance(chat_text, str) or not chat_text:
        raise ValueError("eligible admission record requires chat_text for qualification")
    started = time.monotonic()
    finished = False
    outputs: dict[str, Any] = {}
    try:
        qwen, identity = load_model("tied", torch.device(device))
        ledger.capture_sources()
        ledger.attach(qwen.model)
        model = qwen.model
        for setting in SETTINGS:
            info = entry["settings"][setting]
            request = NativeRequest(
                request_id=f"{entry['image_id']}:{setting}",
                chat_text=chat_text,
                image=info["path"],
                expected_image_grid=(1, info["height"] // 16, info["width"] // 16),
                expected_image_size=(info["width"], info["height"]),
                image_sha256=info["file_sha256"],
            )
            batch = prepare_native_inputs(qwen.processor, [request], device=device, record_media_identity=True)
            eos = int(qwen.token_identity.im_end_token_ids[0])
            with torch.inference_mode():
                result, = generate_continuations(
                    model, batch, extensions=[[]], budgets=[max_new_tokens],
                    eos_token_id=eos, pad_token_id=qwen.tokenizer.pad_token_id,
                    policy=NativeGenerationPolicy(temperature=0, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False),
                    trace="none", seed=None,
                )
            outputs[setting] = {
                "prompt_token_count": len(batch.prompt_token_ids[0]),
                "observed_grid": list(batch.image_grids[0]),
                "executed_media_sha256": batch.media_sha256[0] if batch.media_sha256 else None,
                "token_ids": list(result.token_ids),
                "stop_reason": result.stop_reason,
            }
        result = {
            "schema": "address_readout_pilot.lane_b.qualification.v1",
            "status": "candidate_complete",
            "manifest": _binding(manifest_path),
            "image_id": entry["image_id"],
            "settings": outputs,
            "model_identity": identity,
            "model_forwards": ledger.counts["model_forwards"],
            "vision_forwards": ledger.counts["vision_forwards"],
            "max_new_tokens": max_new_tokens,
            "elapsed_seconds": time.monotonic() - started,
            "qualification_only": True,
        }
        qualification_path = output_dir / "lane-b-qualification.json"
        with qualification_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        ledger.finish("candidate_complete")
        finished = True
        return result
    except BaseException as exc:
        if not finished:
            ledger.finish("technical_invalid", repr(exc))
            finished = True
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--admission", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    qualify = sub.add_parser("qualify")
    qualify.add_argument("--manifest", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--device", default="cuda:0")
    qualify.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_manifest(args.admission, args.output)
    else:
        qualify_one(args.manifest, args.output, device=args.device, max_new_tokens=args.max_new_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
