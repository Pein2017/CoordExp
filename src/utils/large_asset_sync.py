from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ManagedRootPolicy:
    name: str
    relative_root: str
    manifest_path: str
    hash_policy: str


@dataclass(frozen=True)
class LargeAssetSyncPolicy:
    schema_version: int
    repo_root: str
    remote_root: str
    report_dir: str
    ignore_file: str
    managed_roots: list[ManagedRootPolicy]


@dataclass(frozen=True)
class LargeAssetManifestFile:
    relative_path: str
    size_bytes: int
    sha256: str
    remote_path: str = ""


@dataclass(frozen=True)
class LargeAssetManifest:
    schema_version: int
    remote_root: str
    managed_root: str
    relative_root: str
    hash_policy: str
    generated_at_utc: str
    files: list[LargeAssetManifestFile]


def generated_at_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_relative_path(relative_path: str | Path) -> str:
    raw_text = str(relative_path).replace("\\", "/")
    pure_path = PurePosixPath(raw_text)
    if pure_path.is_absolute():
        raise ValueError(f"Expected a repo-relative path, got absolute path: {relative_path!r}")

    normalized_parts: list[str] = []
    for part in pure_path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(
                f"Expected a repo-relative path without parent traversal, got: {relative_path!r}"
            )
        normalized_parts.append(part)

    if not normalized_parts:
        raise ValueError(f"Expected a non-empty repo-relative path, got: {relative_path!r}")

    return PurePosixPath(*normalized_parts).as_posix()


def load_policy(path: str | Path) -> LargeAssetSyncPolicy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    managed_roots_raw = payload.get("managed_roots") or []
    managed_roots = [
        ManagedRootPolicy(
            name=str(entry["name"]),
            relative_root=_normalize_relative_path(entry["relative_root"]),
            manifest_path=str(entry["manifest_path"]),
            hash_policy=str(entry["hash_policy"]),
        )
        for entry in managed_roots_raw
    ]
    return LargeAssetSyncPolicy(
        schema_version=int(payload["schema_version"]),
        repo_root=str(payload.get("repo_root", ".")),
        remote_root=str(payload["remote_root"]),
        report_dir=str(payload["report_dir"]),
        ignore_file=str(payload["ignore_file"]),
        managed_roots=managed_roots,
    )


def load_ignore_patterns(path: str | Path) -> list[str]:
    ignore_path = Path(path)
    if not ignore_path.exists():
        return []
    patterns: list[str] = []
    for raw_line in ignore_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _pattern_variants(pattern: str) -> list[str]:
    variants = [pattern]
    trimmed = pattern
    while trimmed.startswith("**/"):
        trimmed = trimmed[3:]
        variants.append(trimmed)
    if pattern.endswith("/**"):
        variants.append(pattern[: -len("/**")])
    return variants


def should_ignore_relative_path(
    relative_path: str | Path,
    patterns: Sequence[str],
) -> bool:
    normalized = _normalize_relative_path(relative_path)
    pure_path = PurePosixPath(normalized)
    for pattern in patterns:
        for variant in _pattern_variants(str(pattern)):
            if fnmatch.fnmatchcase(normalized, variant):
                return True
            if pure_path.match(variant):
                return True
    return False


def remote_path_for_relative_path(remote_root: str, relative_path: str | Path) -> str:
    normalized_root = str(PurePosixPath(str(remote_root).replace("\\", "/")))
    normalized_rel = _normalize_relative_path(relative_path)
    return str(PurePosixPath(normalized_root) / PurePosixPath(normalized_rel))


def sha256_for_path(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _local_observable_file_metadata(file_info: Mapping[str, Any]) -> tuple[int | None, str | None]:
    size_raw = file_info.get("size_bytes")
    sha_raw = file_info.get("sha256")
    size_value = None if size_raw is None else int(size_raw)
    sha_value = None if sha_raw is None else str(sha_raw)
    return size_value, sha_value


def classify_local_against_manifest(
    manifest_files: Mapping[str, Mapping[str, Any]],
    local_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    manifest_keys = set(manifest_files)
    local_keys = set(local_files)

    synced: list[str] = []
    metadata_drift: list[str] = []
    for relative_path in sorted(manifest_keys & local_keys):
        manifest_observed = _local_observable_file_metadata(manifest_files[relative_path])
        local_observed = _local_observable_file_metadata(local_files[relative_path])
        if manifest_observed == local_observed:
            synced.append(relative_path)
        else:
            metadata_drift.append(relative_path)

    return {
        "synced": synced,
        "missing_local": sorted(manifest_keys - local_keys),
        "missing_remote": sorted(local_keys - manifest_keys),
        "metadata_drift": metadata_drift,
    }


def classify_remote_against_manifest(
    manifest_files: Mapping[str, Mapping[str, Any]],
    remote_files: Mapping[str, Mapping[str, Any]],
    *,
    unknown_remote_state: Sequence[str] = (),
) -> dict[str, list[str]]:
    manifest_keys = set(manifest_files)
    remote_keys = set(remote_files)
    unknown_keys = set(unknown_remote_state)

    synced: list[str] = []
    metadata_drift: list[str] = []
    for relative_path in sorted((manifest_keys & remote_keys) - unknown_keys):
        manifest_size = manifest_files[relative_path].get("size_bytes")
        remote_size = remote_files[relative_path].get("size_bytes")
        if manifest_size is not None and remote_size is not None and int(manifest_size) == int(remote_size):
            synced.append(relative_path)
        else:
            metadata_drift.append(relative_path)

    known_manifest_keys = manifest_keys - unknown_keys
    return {
        "synced_remote": synced,
        "missing_remote": sorted(known_manifest_keys - remote_keys),
        "metadata_drift_remote": metadata_drift,
        "unknown_remote_state": sorted(unknown_keys & manifest_keys),
    }


def load_manifest(path: str | Path) -> LargeAssetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload["schema_version"]) != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported large asset manifest schema_version={payload['schema_version']!r}"
        )

    files = [
        LargeAssetManifestFile(
            relative_path=_normalize_relative_path(entry["relative_path"]),
            size_bytes=int(entry["size_bytes"]),
            sha256=str(entry["sha256"]),
            remote_path=str(entry.get("remote_path", "")),
        )
        for entry in payload.get("files", [])
    ]
    return LargeAssetManifest(
        schema_version=int(payload["schema_version"]),
        remote_root=str(payload["remote_root"]),
        managed_root=str(payload["managed_root"]),
        relative_root=_normalize_relative_path(payload["relative_root"]),
        hash_policy=str(payload["hash_policy"]),
        generated_at_utc=str(payload.get("generated_at_utc", "")),
        files=files,
    )


def write_manifest(path: str | Path, manifest: LargeAssetManifest) -> None:
    payload = asdict(manifest)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def empty_manifest_for_managed_root(
    policy_remote_root: str,
    managed_root: ManagedRootPolicy,
) -> LargeAssetManifest:
    return LargeAssetManifest(
        schema_version=1,
        remote_root=policy_remote_root,
        managed_root=managed_root.name,
        relative_root=managed_root.relative_root,
        hash_policy=managed_root.hash_policy,
        generated_at_utc="",
        files=[],
    )


def scan_local_root(
    repo_root: str | Path,
    relative_root: str | Path,
    ignore_patterns: Sequence[str],
    *,
    include_hash: bool = True,
) -> dict[str, dict[str, Any]]:
    repo_root_path = Path(repo_root)
    normalized_root = _normalize_relative_path(relative_root)
    managed_root_path = repo_root_path / Path(normalized_root)
    scanned: dict[str, dict[str, Any]] = {}
    if not managed_root_path.exists():
        return scanned

    for path in sorted(managed_root_path.rglob("*")):
        if not path.is_file():
            continue
        relative_path = _normalize_relative_path(path.relative_to(repo_root_path))
        if should_ignore_relative_path(relative_path, ignore_patterns):
            continue
        file_info: dict[str, Any] = {"size_bytes": int(path.stat().st_size)}
        if include_hash:
            file_info["sha256"] = sha256_for_path(path)
        scanned[relative_path] = file_info
    return scanned


def manifest_to_file_index(manifest: LargeAssetManifest) -> dict[str, dict[str, Any]]:
    return {
        entry.relative_path: {
            "size_bytes": entry.size_bytes,
            "sha256": entry.sha256,
            "remote_path": entry.remote_path,
        }
        for entry in manifest.files
    }


def load_manifests_for_policy(
    repo_root: Path,
    policy: LargeAssetSyncPolicy,
) -> tuple[dict[str, LargeAssetManifest], dict[str, Path]]:
    manifests: dict[str, LargeAssetManifest] = {}
    manifest_paths: dict[str, Path] = {}
    for managed_root in policy.managed_roots:
        manifest_path = repo_root / managed_root.manifest_path
        manifest_paths[managed_root.name] = manifest_path
        if manifest_path.exists():
            manifests[managed_root.name] = load_manifest(manifest_path)
        else:
            manifests[managed_root.name] = empty_manifest_for_managed_root(
                policy.remote_root,
                managed_root,
            )
    return manifests, manifest_paths


def managed_root_for_relative_path(
    policy: LargeAssetSyncPolicy,
    relative_path: str,
) -> ManagedRootPolicy:
    normalized_relative_path = _normalize_relative_path(relative_path)
    for managed_root in policy.managed_roots:
        prefix = f"{managed_root.relative_root}/"
        if (
            normalized_relative_path == managed_root.relative_root
            or normalized_relative_path.startswith(prefix)
        ):
            return managed_root
    raise ValueError(
        f"No managed root configured for relative_path={relative_path!r}"
    )


def rewrite_manifest_files(
    manifest: LargeAssetManifest,
    manifest_files: Mapping[str, Mapping[str, Any]],
    *,
    generated_at_utc: str | None = None,
) -> LargeAssetManifest:
    files = [
        LargeAssetManifestFile(
            relative_path=relative_path,
            size_bytes=int(file_info["size_bytes"]),
            sha256=str(file_info.get("sha256", "")),
            remote_path=str(file_info.get("remote_path", "")),
        )
        for relative_path, file_info in sorted(manifest_files.items())
    ]
    return LargeAssetManifest(
        schema_version=manifest.schema_version,
        remote_root=manifest.remote_root,
        managed_root=manifest.managed_root,
        relative_root=manifest.relative_root,
        hash_policy=manifest.hash_policy,
        generated_at_utc=(
            manifest.generated_at_utc
            if generated_at_utc is None
            else str(generated_at_utc)
        ),
        files=files,
    )


def apply_publish_manifest_updates(
    *,
    policy: LargeAssetSyncPolicy,
    manifests: Mapping[str, LargeAssetManifest],
    local_files: Mapping[str, Mapping[str, Any]],
    planned_paths: Sequence[str],
) -> dict[str, LargeAssetManifest]:
    updated_manifests = dict(manifests)
    touched_roots: set[str] = set()
    for relative_path in planned_paths:
        managed_root = managed_root_for_relative_path(policy, relative_path)
        manifest_files = manifest_to_file_index(updated_manifests[managed_root.name])
        manifest_files[relative_path] = {
            "size_bytes": local_files[relative_path]["size_bytes"],
            "sha256": local_files[relative_path]["sha256"],
            "remote_path": remote_path_for_relative_path(policy.remote_root, relative_path),
        }
        updated_manifests[managed_root.name] = rewrite_manifest_files(
            updated_manifests[managed_root.name],
            manifest_files,
        )
        touched_roots.add(managed_root.name)

    timestamp = generated_at_utc_now()
    for managed_root_name in touched_roots:
        updated_manifests[managed_root_name] = rewrite_manifest_files(
            updated_manifests[managed_root_name],
            manifest_to_file_index(updated_manifests[managed_root_name]),
            generated_at_utc=timestamp,
        )
    return updated_manifests


def write_manifests(
    manifest_paths: Mapping[str, Path],
    manifests: Mapping[str, LargeAssetManifest],
) -> None:
    for managed_root_name, manifest in manifests.items():
        write_manifest(manifest_paths[managed_root_name], manifest)


def build_local_scan_report(
    *,
    repo_root: Path,
    policy: LargeAssetSyncPolicy,
    include_hash: bool,
) -> dict[str, Any]:
    ignore_patterns = load_ignore_patterns(repo_root / policy.ignore_file)
    local_files: dict[str, dict[str, Any]] = {}
    for managed_root in policy.managed_roots:
        local_files.update(
            scan_local_root(
                repo_root=repo_root,
                relative_root=managed_root.relative_root,
                ignore_patterns=ignore_patterns,
                include_hash=include_hash or managed_root.hash_policy == "full",
            )
        )
    return {
        "status": "ok",
        "local_files": local_files,
        "total_files": len(local_files),
    }


def write_report(path: str | Path, payload: Mapping[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def plan_publish(
    manifest_files: Mapping[str, Mapping[str, Any]],
    local_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    changed: list[str] = []
    new: list[str] = []
    for relative_path in sorted(local_files):
        actual = local_files[relative_path]
        expected = manifest_files.get(relative_path)
        if expected is None:
            new.append(relative_path)
            continue
        actual_size = actual.get("size_bytes")
        expected_size = expected.get("size_bytes")
        actual_sha = actual.get("sha256")
        expected_sha = expected.get("sha256")
        if actual_size != expected_size or (
            actual_sha is not None and expected_sha is not None and actual_sha != expected_sha
        ):
            changed.append(relative_path)
    return {"new": new, "changed": changed}


def plan_align_local(
    manifest_files: Mapping[str, Mapping[str, Any]],
    local_files: Mapping[str, Mapping[str, Any]],
    remote_files: Mapping[str, Mapping[str, Any]],
    *,
    unknown_remote_state: Sequence[str] = (),
) -> dict[str, list[str]]:
    local_diff = classify_local_against_manifest(manifest_files, local_files)
    remote_diff = classify_remote_against_manifest(
        manifest_files,
        remote_files,
        unknown_remote_state=unknown_remote_state,
    )
    remote_restorable = set(remote_diff["synced_remote"])

    missing_local = [
        relative_path
        for relative_path in local_diff["missing_local"]
        if relative_path in remote_restorable
    ]
    metadata_drift_local = [
        relative_path
        for relative_path in local_diff["metadata_drift"]
        if relative_path in remote_restorable
    ]
    unavailable_remote = sorted(
        (
            set(local_diff["missing_local"]) | set(local_diff["metadata_drift"])
        )
        - remote_restorable
    )
    return {
        "missing_local": missing_local,
        "metadata_drift_local": metadata_drift_local,
        "unavailable_remote": unavailable_remote,
        "unknown_remote_state": remote_diff["unknown_remote_state"],
    }
