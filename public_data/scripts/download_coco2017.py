import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


COCO2017_URLS: Dict[str, str] = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


@dataclass(frozen=True)
class DownloadResult:
    filename: str
    url: str
    sha256: str
    bytes_written: int


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst_path: Path, *, force: bool) -> DownloadResult:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists() and not force:
        sha256 = _sha256_file(dst_path)
        return DownloadResult(
            filename=dst_path.name,
            url=url,
            sha256=sha256,
            bytes_written=dst_path.stat().st_size,
        )

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": "CoordExp/public_data"})
    with urllib.request.urlopen(req) as resp, tmp_path.open("wb") as f:
        h = hashlib.sha256()
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            total += len(chunk)

    tmp_path.replace(dst_path)

    return DownloadResult(
        filename=dst_path.name,
        url=url,
        sha256=h.hexdigest(),
        bytes_written=total,
    )


def _extract_zip(zip_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def _write_checksums(results: Iterable[DownloadResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in sorted(results, key=lambda x: x.filename):
        lines.append(f"{r.sha256}  {r.filename}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(results: Iterable[DownloadResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "filename": r.filename,
            "url": r.url,
            "sha256": r.sha256,
            "bytes_written": r.bytes_written,
        }
        for r in sorted(results, key=lambda x: x.filename)
    ]
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download COCO 2017 (images + annotations) into public_data/coco/raw/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Canonical COCO 2017 artifacts (hosted on images.cocodataset.org):
  - train2017.zip
  - val2017.zip
  - annotations_trainval2017.zip

Expected output structure (after extraction):
  public_data/coco/raw/
    images/train2017/*.jpg
    images/val2017/*.jpg
    annotations/instances_train2017.json
    annotations/instances_val2017.json

This script writes checksums for downloaded zips to:
  public_data/coco/raw/downloads/SHA256SUMS.txt
""",
    )

    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to public_data/coco/raw (created if missing)",
    )

    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download/extract only annotations (no image zips)",
    )

    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download only; do not extract zips",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if zip already exists",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    downloads_dir = raw_dir / "downloads"
    images_dir = raw_dir / "images"

    to_download: List[Tuple[str, str]] = []
    to_download.append(("annotations_trainval2017.zip", COCO2017_URLS["annotations_trainval2017.zip"]))
    if not args.annotations_only:
        to_download.append(("train2017.zip", COCO2017_URLS["train2017.zip"]))
        to_download.append(("val2017.zip", COCO2017_URLS["val2017.zip"]))

    print("=" * 70)
    print("COCO 2017 downloader")
    print("=" * 70)
    print(f"  raw_dir: {raw_dir}")
    print(f"  downloads_dir: {downloads_dir}")
    print(f"  annotations_only: {args.annotations_only}")
    print(f"  skip_extract: {args.skip_extract}")
    print(f"  force: {args.force}")
    print("=" * 70)

    results: List[DownloadResult] = []
    for filename, url in to_download:
        dst_path = downloads_dir / filename
        print(f"\n[download] {filename}")
        print(f"  url: {url}")
        r = _download(url, dst_path, force=bool(args.force))
        results.append(r)
        print(f"  bytes: {r.bytes_written}")
        print(f"  sha256: {r.sha256}")

        if args.skip_extract:
            continue

        if filename == "annotations_trainval2017.zip":
            print(f"[extract] {filename} -> {raw_dir}")
            _extract_zip(dst_path, raw_dir)
        else:
            print(f"[extract] {filename} -> {images_dir}")
            _extract_zip(dst_path, images_dir)

    _write_checksums(results, downloads_dir / "SHA256SUMS.txt")
    _write_manifest(results, downloads_dir / "manifest.json")

    print("\n✓ Done")
    print(f"  checksums: {downloads_dir / 'SHA256SUMS.txt'}")
    print(f"  manifest:  {downloads_dir / 'manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise
