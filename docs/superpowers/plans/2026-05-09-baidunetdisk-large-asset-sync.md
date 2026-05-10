# Baidu Netdisk Large Asset Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manifest-driven large-asset sync tool that keeps `public_data/**`, `model_cache/**`, and `output/**` aligned across two non-networked A100 nodes and canonical Baidu Netdisk paths under `/CoordExp/**` without requiring full cloud re-downloads for sync verification.

**Architecture:** Keep `git` as the truth source through tracked manifest files under `manifests/large_assets/`, implement pure local scan and diff logic in `src/utils/`, isolate BaiduPCS-Go subprocess interaction in a thin adapter module, and expose report-first `scan-local`, `scan-remote`, `publish`, and `align-local` commands through one YAML-first Python entrypoint. The first implementation is metadata-first for speed, hash-strong on publish and enrollment, and preserves repo-relative paths exactly.

**Tech Stack:** Python, standard library (`argparse`, `dataclasses`, `pathlib`, `hashlib`, `json`, `fnmatch`, `subprocess`), BaiduPCS-Go CLI, pytest, repo docs, `rtk conda run -n ms python -m pytest`.

---

## Scope Boundary

This plan implements the first production-ready large-asset sync tool described in [2026-05-09-baidunetdisk-large-asset-sync-design.md](/data/home/xiaoyan/AIteam/data/CoordExp/docs/superpowers/specs/2026-05-09-baidunetdisk-large-asset-sync-design.md). It does not add a background daemon, does not add direct node-to-node transfer, does not turn remote storage into a content-addressed store, and does not require live cloud smoke as a merge gate. Remote integrity remains a path-and-size contract because the current BaiduPCS-Go workflow does not expose a trustworthy remote hash surface.

The implementation must preserve these hard rules from the approved design:

- Managed identity is always repo-relative path.
- Canonical remote root is `/CoordExp/**`.
- `git`-tracked manifests are the durable truth source.
- `align-local` is report-first and dry-run by default.
- `output/**` is broadly managed but filtered through explicit ignore rules.
- Temporary scan/report artifacts live under `temp/large_asset_sync/`, not `git`.

## File Structure

- Create: `manifests/large_assets/policy.yaml`
  - Canonical managed roots, remote root, hash policy, report directory, and ignore file location.
- Create: `manifests/large_assets/ignore.txt`
  - Initial glob-style exclusions for obvious transient files.
- Create: `manifests/large_assets/public_data.manifest.json`
  - Seeded empty manifest for `public_data/**`.
- Create: `manifests/large_assets/model_cache.manifest.json`
  - Seeded empty manifest for `model_cache/**`.
- Create: `manifests/large_assets/output.manifest.json`
  - Seeded empty manifest for `output/**`.
- Create: `src/utils/large_asset_sync.py`
  - Policy parsing, ignore matching, manifest I/O, local scan, diff classification, report writing, and high-level publish/align planning.
- Create: `src/utils/large_asset_sync_baidupcs.py`
  - Thin subprocess adapter for BaiduPCS-Go `ls`, `mkdir`, `upload`, `download`, and `pwd`/`quota` validation as needed.
- Create: `scripts/large_asset_sync.py`
  - User-facing CLI with `scan-local`, `scan-remote`, `publish`, and `align-local`.
- Create: `tests/test_large_asset_sync_manifest.py`
  - Unit tests for policy loading, ignore rules, manifest roundtrip, path mapping, and diff classification.
- Create: `tests/test_large_asset_sync_remote.py`
  - Unit tests for BaiduPCS-Go output parsing and remote scan behavior via mocked subprocess output.
- Create: `tests/test_large_asset_sync_cli.py`
  - CLI-level dry-run and report-path tests using temporary fixtures and monkeypatched adapters.
- Create: `docs/standards/LARGE_ASSET_SYNC.md`
  - Stable operator contract for manifests, remote root, and node-switch workflow.
- Modify: `docs/standards/README.md`
  - Route users to the new large-asset sync standard.
- Modify: `docs/ARTIFACTS.md`
  - Document `manifests/large_assets/**` and `temp/large_asset_sync/**` as reproducibility-supporting infrastructure artifacts.
- Modify: `scripts/README.md`
  - Add `scripts/large_asset_sync.py` as a stable YAML-first operational entrypoint.

## Task 1: Seed Policy, Ignore Rules, And Empty Manifests

**Files:**
- Create: `manifests/large_assets/policy.yaml`
- Create: `manifests/large_assets/ignore.txt`
- Create: `manifests/large_assets/public_data.manifest.json`
- Create: `manifests/large_assets/model_cache.manifest.json`
- Create: `manifests/large_assets/output.manifest.json`
- Test later: `tests/test_large_asset_sync_manifest.py`

- [ ] **Step 1: Add the managed-root policy file**

Create `manifests/large_assets/policy.yaml` with the approved root contract and conservative verification defaults:

```yaml
schema_version: 1
repo_root: "."
remote_root: "/CoordExp"
report_dir: "temp/large_asset_sync"
ignore_file: "manifests/large_assets/ignore.txt"
managed_roots:
  - name: "public_data"
    relative_root: "public_data"
    manifest_path: "manifests/large_assets/public_data.manifest.json"
    hash_policy: "full"
  - name: "model_cache"
    relative_root: "model_cache"
    manifest_path: "manifests/large_assets/model_cache.manifest.json"
    hash_policy: "full"
  - name: "output"
    relative_root: "output"
    manifest_path: "manifests/large_assets/output.manifest.json"
    hash_policy: "sampled"
```

- [ ] **Step 2: Add the ignore rules file**

Create `manifests/large_assets/ignore.txt` with only the exclusions approved in the design:

```text
**/*.tmp
**/*.part
**/*.lock
**/.DS_Store
**/tmp/**
**/cache/**
**/*.uploading
**/*.downloading
**/wandb/latest-run/**
```

- [ ] **Step 3: Seed the three empty manifest files**

Create `manifests/large_assets/public_data.manifest.json`, `manifests/large_assets/model_cache.manifest.json`, and `manifests/large_assets/output.manifest.json` with the same empty schema shape:

```json
{
  "schema_version": 1,
  "remote_root": "/CoordExp",
  "managed_root": "public_data",
  "generated_at_utc": "",
  "files": []
}
```

Use `managed_root: "model_cache"` and `managed_root: "output"` in the other two files.

- [ ] **Step 4: Verify the seed files are present and machine-readable**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in [
    Path("manifests/large_assets/public_data.manifest.json"),
    Path("manifests/large_assets/model_cache.manifest.json"),
    Path("manifests/large_assets/output.manifest.json"),
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert isinstance(payload["files"], list)
print("manifest seeds ok")
PY
```

Expected: `manifest seeds ok`

- [ ] **Step 5: Commit the seed contract**

```bash
git add manifests/large_assets/policy.yaml manifests/large_assets/ignore.txt manifests/large_assets/*.manifest.json
git commit -m "Add large asset sync policy seeds"
```

## Task 2: Implement Manifest Models, Ignore Matching, And Local Diff Logic

**Files:**
- Create: `src/utils/large_asset_sync.py`
- Create: `tests/test_large_asset_sync_manifest.py`

- [ ] **Step 1: Write the failing manifest and ignore-rule tests**

Create `tests/test_large_asset_sync_manifest.py` with targeted unit tests for:

- policy loading and relative-root preservation
- ignore glob matching
- repo-relative path mapping to `/CoordExp/<relative_path>`
- empty manifest roundtrip
- diff classification for `synced`, `missing_local`, `missing_remote`, and metadata drift

Start with tests in this shape:

```python
from pathlib import Path

from src.utils.large_asset_sync import (
    classify_local_against_manifest,
    load_policy,
    remote_path_for_relative_path,
    should_ignore_relative_path,
)


def test_remote_path_for_relative_path_preserves_repo_layout() -> None:
    assert remote_path_for_relative_path("/CoordExp", "output/stage1/run-a/file.bin") == (
        "/CoordExp/output/stage1/run-a/file.bin"
    )


def test_should_ignore_relative_path_matches_globs() -> None:
    patterns = ["**/*.tmp", "**/tmp/**"]
    assert should_ignore_relative_path("output/run-a/model.tmp", patterns) is True
    assert should_ignore_relative_path("output/run-a/checkpoint-100/adapter_model.safetensors", patterns) is False


def test_classify_local_against_manifest_marks_missing_and_synced() -> None:
    manifest_files = {
        "output/run-a/adapter_model.safetensors": {"size_bytes": 12, "sha256": "abc"},
        "output/run-a/summary.json": {"size_bytes": 2, "sha256": "def"},
    }
    local_files = {
        "output/run-a/adapter_model.safetensors": {"size_bytes": 12, "sha256": "abc"},
    }
    diff = classify_local_against_manifest(manifest_files, local_files)
    assert diff["synced"] == ["output/run-a/adapter_model.safetensors"]
    assert diff["missing_local"] == ["output/run-a/summary.json"]
```

- [ ] **Step 2: Run the manifest tests and confirm they fail first**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_large_asset_sync_manifest.py -q
```

Expected: `ImportError` or missing-symbol failures from `src.utils.large_asset_sync`.

- [ ] **Step 3: Add the minimal manifest/policy/local-diff module**

Create `src/utils/large_asset_sync.py` with these first-class building blocks before any remote integration:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import fnmatch
import hashlib
import json
from typing import Any

import yaml


@dataclass(frozen=True)
class ManagedRootPolicy:
    name: str
    relative_root: str
    manifest_path: str
    hash_policy: str


@dataclass(frozen=True)
class SyncPolicy:
    schema_version: int
    repo_root: str
    remote_root: str
    report_dir: str
    ignore_file: str
    managed_roots: tuple[ManagedRootPolicy, ...]


def load_policy(path: str | Path) -> SyncPolicy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    roots = tuple(ManagedRootPolicy(**item) for item in payload["managed_roots"])
    return SyncPolicy(
        schema_version=int(payload["schema_version"]),
        repo_root=str(payload["repo_root"]),
        remote_root=str(payload["remote_root"]),
        report_dir=str(payload["report_dir"]),
        ignore_file=str(payload["ignore_file"]),
        managed_roots=roots,
    )


def should_ignore_relative_path(relative_path: str, patterns: list[str]) -> bool:
    normalized = relative_path.strip("/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def remote_path_for_relative_path(remote_root: str, relative_path: str) -> str:
    return f"{remote_root.rstrip('/')}/{relative_path.strip('/')}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_local_against_manifest(manifest_files: dict[str, dict[str, Any]], local_files: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    result = {
        "synced": [],
        "missing_local": [],
        "missing_remote": [],
        "drift_local_metadata": [],
    }
    for relative_path, expected in manifest_files.items():
        actual = local_files.get(relative_path)
        if actual is None:
            result["missing_local"].append(relative_path)
            continue
        if actual["size_bytes"] != expected["size_bytes"]:
            result["drift_local_metadata"].append(relative_path)
            continue
        if expected.get("sha256") and actual.get("sha256") and actual["sha256"] != expected["sha256"]:
            result["drift_local_metadata"].append(relative_path)
            continue
        result["synced"].append(relative_path)
    return result
```

Keep the first implementation pure and deterministic. Do not call BaiduPCS-Go from this file yet.

- [ ] **Step 4: Extend the module with manifest roundtrip and local scan helpers**

Add these functions to `src/utils/large_asset_sync.py` before re-running tests:

```python
def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scan_local_root(repo_root: Path, managed_root: ManagedRootPolicy, ignore_patterns: list[str], include_hash: bool) -> dict[str, dict[str, Any]]:
    root = repo_root / managed_root.relative_root
    rows: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(repo_root).as_posix()
        if should_ignore_relative_path(relative_path, ignore_patterns):
            continue
        stat = path.stat()
        rows[relative_path] = {
            "size_bytes": int(stat.st_size),
            "local_mtime_utc": int(stat.st_mtime),
            "sha256": sha256_file(path) if include_hash else "",
            "managed_root": managed_root.name,
        }
    return rows
```

- [ ] **Step 5: Re-run the manifest tests and make them pass**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_large_asset_sync_manifest.py -q
```

Expected: all tests in `tests/test_large_asset_sync_manifest.py` pass.

- [ ] **Step 6: Commit the local manifest core**

```bash
git add src/utils/large_asset_sync.py tests/test_large_asset_sync_manifest.py
git commit -m "Add local large asset sync core"
```

## Task 3: Add BaiduPCS-Go Remote Listing And Transfer Adapters

**Files:**
- Create: `src/utils/large_asset_sync_baidupcs.py`
- Create: `tests/test_large_asset_sync_remote.py`

- [ ] **Step 1: Write the failing remote parsing and adapter tests**

Create `tests/test_large_asset_sync_remote.py` with mocked subprocess coverage for:

- parsing `BaiduPCS-Go ls` directory output
- mapping remote rows back to repo-relative paths
- partial remote-scan failure becoming `unknown_remote_state`
- file upload/download command construction preserving relative paths

Start with tests like:

```python
from pathlib import Path

from src.utils.large_asset_sync_baidupcs import (
    parse_baidupcs_ls_output,
    remote_rows_to_manifest_index,
)


def test_parse_baidupcs_ls_output_extracts_dirs_and_files() -> None:
    text = """
当前目录: /CoordExp/output
----
  #  文件大小       修改日期               文件(目录)
  0     12.00MB  2026-05-09 12:00:00  adapter_model.safetensors
  1          -  2026-05-09 12:00:00  stage1_2b/
----
"""
    rows = parse_baidupcs_ls_output(text, current_dir="/CoordExp/output")
    assert rows[0]["path"] == "/CoordExp/output/adapter_model.safetensors"
    assert rows[0]["is_dir"] is False
    assert rows[1]["path"] == "/CoordExp/output/stage1_2b"
    assert rows[1]["is_dir"] is True


def test_remote_rows_to_manifest_index_drops_dirs() -> None:
    rows = [
        {"path": "/CoordExp/output/run-a", "is_dir": True, "size_bytes": 0},
        {"path": "/CoordExp/output/run-a/summary.json", "is_dir": False, "size_bytes": 42},
    ]
    index = remote_rows_to_manifest_index(rows, remote_root="/CoordExp")
    assert list(index) == ["output/run-a/summary.json"]
```

- [ ] **Step 2: Run the remote tests and confirm they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_large_asset_sync_remote.py -q
```

Expected: missing-module or missing-symbol failures from `src.utils.large_asset_sync_baidupcs`.

- [ ] **Step 3: Add the thin BaiduPCS-Go adapter module**

Create `src/utils/large_asset_sync_baidupcs.py` with subprocess wrappers and a parser separated from business logic:

```python
from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Any


def run_baidupcs(bin_path: str | Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(bin_path), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def parse_baidupcs_ls_output(text: str, current_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*\d+\s+(.+?)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(.+?)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        size_text, raw_name = match.groups()
        is_dir = raw_name.endswith("/")
        clean_name = raw_name.rstrip("/")
        rows.append(
            {
                "path": f"{current_dir.rstrip('/')}/{clean_name}".rstrip("/"),
                "is_dir": is_dir,
                "size_bytes": 0 if is_dir else size_text,
            }
        )
    return rows


def remote_rows_to_manifest_index(rows: list[dict[str, Any]], remote_root: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    prefix = remote_root.rstrip("/") + "/"
    for row in rows:
        if row["is_dir"]:
            continue
        remote_path = row["path"]
        if not remote_path.startswith(prefix):
            continue
        relative_path = remote_path[len(prefix):]
        index[relative_path] = {
            "size_bytes": row["size_bytes"],
            "remote_path": remote_path,
        }
    return index
```

Keep command building explicit. Do not bury subprocess calls inside the local manifest module.

- [ ] **Step 4: Add remote scan, mkdir, upload, and download command helpers**

Extend `src/utils/large_asset_sync_baidupcs.py` with focused helpers:

```python
def list_remote_path(bin_path: str | Path, remote_path: str) -> str:
    return run_baidupcs(bin_path, "ls", remote_path).stdout


def mkdir_remote_path(bin_path: str | Path, remote_path: str) -> None:
    run_baidupcs(bin_path, "mkdir", remote_path)


def upload_file(bin_path: str | Path, local_path: Path, remote_parent: str) -> None:
    run_baidupcs(
        bin_path,
        "upload",
        str(local_path),
        remote_parent,
        "--policy",
        "overwrite",
        "-p",
        "1",
        "-l",
        "1",
        "--retry",
        "8",
        "--norapid",
    )


def download_path(bin_path: str | Path, remote_path: str, save_dir: Path) -> None:
    run_baidupcs(bin_path, "config", "set", "-savedir", str(save_dir))
    run_baidupcs(
        bin_path,
        "download",
        remote_path,
        "--fullpath",
        "--mode",
        "locate",
        "-p",
        "8",
        "-l",
        "4",
        "--retry",
        "8",
        "--ow",
        "--mtime",
    )
```

These defaults should intentionally mirror the proven BaiduPCS-Go workflow already used in this repo.

- [ ] **Step 5: Re-run the remote adapter tests and make them pass**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_large_asset_sync_remote.py -q
```

Expected: all tests in `tests/test_large_asset_sync_remote.py` pass.

- [ ] **Step 6: Commit the remote adapter**

```bash
git add src/utils/large_asset_sync_baidupcs.py tests/test_large_asset_sync_remote.py
git commit -m "Add BaiduPCS-Go sync adapter"
```

## Task 4: Build The Report-First CLI For Scan, Publish, And Align

**Files:**
- Modify: `src/utils/large_asset_sync.py`
- Modify: `src/utils/large_asset_sync_baidupcs.py`
- Create: `scripts/large_asset_sync.py`
- Create: `tests/test_large_asset_sync_cli.py`

- [ ] **Step 1: Write the failing CLI and report tests**

Create `tests/test_large_asset_sync_cli.py` with dry-run coverage for:

- `scan-local` writing `local_scan.json`
- `scan-remote` writing `remote_scan.json`
- `publish --plan` reporting changed files without mutating manifests
- `align-local --plan` reporting `missing_local` without downloading

Start with tests like:

```python
import json
from pathlib import Path

from scripts.large_asset_sync import main


def test_scan_local_writes_report(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "output/run-a").mkdir(parents=True)
    (repo_root / "output/run-a/summary.json").write_text("{}", encoding="utf-8")
    report_path = tmp_path / "local_scan.json"

    exit_code = main(
        [
            "scan-local",
            "--repo-root",
            str(repo_root),
            "--policy",
            "manifests/large_assets/policy.yaml",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert "output/run-a/summary.json" in payload["local_files"]
```

- [ ] **Step 2: Run the CLI tests and confirm they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_large_asset_sync_cli.py -q
```

Expected: import or missing-entrypoint failures from `scripts/large_asset_sync.py`.

- [ ] **Step 3: Add high-level planning and report helpers to the core module**

Extend `src/utils/large_asset_sync.py` with:

- `load_ignore_patterns(...)`
- `build_local_scan_report(...)`
- `classify_remote_against_manifest(...)`
- `write_report(path, payload)`
- `plan_publish(...)`
- `plan_align_local(...)`

Use a shape like:

```python
def build_local_scan_report(*, repo_root: Path, policy: SyncPolicy, include_hash: bool) -> dict[str, Any]:
    ignore_patterns = load_ignore_patterns(repo_root / policy.ignore_file)
    local_files: dict[str, dict[str, Any]] = {}
    for managed_root in policy.managed_roots:
        local_files.update(
            scan_local_root(
                repo_root=repo_root,
                managed_root=managed_root,
                ignore_patterns=ignore_patterns,
                include_hash=include_hash or managed_root.hash_policy == "full",
            )
        )
    return {
        "status": "ok",
        "local_files": local_files,
        "total_files": len(local_files),
    }


def plan_publish(manifest_files: dict[str, dict[str, Any]], local_files: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    changed: list[str] = []
    new: list[str] = []
    for relative_path, actual in local_files.items():
        expected = manifest_files.get(relative_path)
        if expected is None:
            new.append(relative_path)
            continue
        if expected["size_bytes"] != actual["size_bytes"] or expected.get("sha256") != actual.get("sha256"):
            changed.append(relative_path)
    return {"new": sorted(new), "changed": sorted(changed)}
```

- [ ] **Step 4: Add the user-facing CLI entrypoint**

Create `scripts/large_asset_sync.py` in the same thin-wrapper style as `scripts/postop_confidence.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.large_asset_sync import (
    build_local_scan_report,
    load_policy,
    write_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large asset sync helper for repo-relative Baidu Netdisk mirroring.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("scan-local", "scan-remote", "publish", "align-local"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--policy", type=Path, required=True)
        sub.add_argument("--repo-root", type=Path, default=Path("."))
        sub.add_argument("--report", type=Path, required=True)
        sub.add_argument("--execute", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy = load_policy(args.policy)
    repo_root = args.repo_root.resolve()
    if args.command == "scan-local":
        report = build_local_scan_report(repo_root=repo_root, policy=policy, include_hash=False)
        write_report(args.report, report)
        return 0
    raise NotImplementedError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
```

Then fill in `scan-remote`, `publish`, and `align-local` using the already-tested helpers instead of embedding logic directly in the script.

- [ ] **Step 5: Add report-first behavior for publish and align**

Implement the remaining CLI branches with these guarantees:

- `publish` without `--execute` only writes a plan report.
- `publish --execute` updates manifest payloads in memory, uploads missing/changed files, verifies remote existence and size, then writes updated manifest files.
- `align-local` without `--execute` only writes a missing/drift plan.
- `align-local --execute` downloads only the planned files into the repo root.

Use high-level orchestration in this shape:

```python
if args.command == "publish":
    plan = build_publish_plan(...)
    write_report(args.report, plan)
    if not args.execute:
        return 0
    execute_publish_plan(...)
    return 0

if args.command == "align-local":
    plan = build_align_plan(...)
    write_report(args.report, plan)
    if not args.execute:
        return 0
    execute_align_plan(...)
    return 0
```

Do not let `--execute` be the default for any subcommand.

- [ ] **Step 6: Run the targeted CLI tests and then the combined sync suite**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_large_asset_sync_cli.py -q
rtk conda run -n ms python -m pytest tests/test_large_asset_sync_manifest.py tests/test_large_asset_sync_remote.py tests/test_large_asset_sync_cli.py -q
```

Expected: all three test files pass.

- [ ] **Step 7: Commit the CLI and orchestration layer**

```bash
git add src/utils/large_asset_sync.py src/utils/large_asset_sync_baidupcs.py scripts/large_asset_sync.py tests/test_large_asset_sync_cli.py
git commit -m "Add large asset sync CLI"
```

## Task 5: Document The Operator Workflow And Add A Local Smoke

**Files:**
- Create: `docs/standards/LARGE_ASSET_SYNC.md`
- Modify: `docs/standards/README.md`
- Modify: `docs/ARTIFACTS.md`
- Modify: `scripts/README.md`

- [ ] **Step 1: Add the stable operator guide**

Create `docs/standards/LARGE_ASSET_SYNC.md` with the stable operational contract:

```markdown
# Large Asset Sync

This workflow keeps repo-relative large assets aligned across local nodes and Baidu Netdisk.

## Truth Source

- `git` manifests under `manifests/large_assets/`

## Canonical Remote Root

- `/CoordExp`

## Managed Roots

- `public_data/**`
- `model_cache/**`
- `output/**`

## Daily Workflow

1. Produce or update files locally under a managed root.
2. Run `scan-local`.
3. Run `publish --report ...` and inspect the plan.
4. Run `publish --execute ...` only when the plan is correct.
5. Commit the manifest updates.
6. On another node, `git pull`, run `scan-local`, `scan-remote`, and `align-local --report ...`.
7. Run `align-local --execute ...` only for the planned missing/drifted files.
```

- [ ] **Step 2: Route the new guide from standards and scripts docs**

Modify `docs/standards/README.md` to add:

```markdown
- [LARGE_ASSET_SYNC.md](LARGE_ASSET_SYNC.md)
  - repo-relative Baidu Netdisk backup and cross-node restore workflow
```

Modify `scripts/README.md` to add the new stable entrypoint:

```markdown
- Large asset sync / cross-node restore planning (YAML-first policy): `scripts/large_asset_sync.py`.
```

- [ ] **Step 3: Document the new artifact surfaces**

Modify `docs/ARTIFACTS.md` with a short new section describing:

- `manifests/large_assets/*.manifest.json`
- `manifests/large_assets/policy.yaml`
- `manifests/large_assets/ignore.txt`
- `temp/large_asset_sync/*.json`

Use wording in this shape:

```markdown
## Large Asset Sync Artifacts

The large-asset sync workflow writes git-tracked desired-state manifests under
`manifests/large_assets/` and ephemeral local reports under
`temp/large_asset_sync/`.

- `public_data.manifest.json`, `model_cache.manifest.json`, `output.manifest.json`
  - desired-state file inventories keyed by repo-relative path
- `policy.yaml`
  - managed-root and canonical remote-root contract
- `ignore.txt`
  - transient-file exclusions for broad `output/**` management
- `local_scan.json`, `remote_scan.json`, `publish_report.json`, `align_plan.json`
  - disposable operator reports; these do not enter `git`
```

- [ ] **Step 4: Run a local smoke with a temporary managed-root fixture**

Create a temporary local-only smoke tree and verify report generation without cloud mutation:

```bash
mkdir -p temp/large_asset_sync/smoke_repo/output/demo-run
printf '{}' > temp/large_asset_sync/smoke_repo/output/demo-run/summary.json
python scripts/large_asset_sync.py scan-local \
  --repo-root temp/large_asset_sync/smoke_repo \
  --policy manifests/large_assets/policy.yaml \
  --report temp/large_asset_sync/local_scan_smoke.json
python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("temp/large_asset_sync/local_scan_smoke.json").read_text(encoding="utf-8"))
assert payload["status"] == "ok"
print(payload["total_files"])
PY
```

Expected: the report exists and prints a positive file count.

- [ ] **Step 5: Run the full targeted verification set**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_large_asset_sync_manifest.py tests/test_large_asset_sync_remote.py tests/test_large_asset_sync_cli.py -q
python scripts/large_asset_sync.py --help
```

Expected:

- pytest passes
- CLI help shows `scan-local`, `scan-remote`, `publish`, and `align-local`

- [ ] **Step 6: Commit docs and smoke-ready workflow**

```bash
git add docs/standards/LARGE_ASSET_SYNC.md docs/standards/README.md docs/ARTIFACTS.md scripts/README.md
git commit -m "Document large asset sync workflow"
```

## Execution Notes

- Keep implementation repo-relative from end to end. Never write absolute machine paths into manifest payloads.
- Prefer standard-library JSON and dataclasses; do not pull in new dependencies for this tool.
- Keep BaiduPCS-Go interaction mockable. Every subprocess boundary should be behind a helper that tests can monkeypatch.
- Do not make live cloud access a unit-test requirement.
- Do not silently rewrite or backfill manifest state on `scan-local` or `scan-remote`; only `publish --execute` should advance tracked manifests.
- If command ergonomics drift during implementation, preserve the contract names from the design and document the final CLI in `scripts/README.md`.
