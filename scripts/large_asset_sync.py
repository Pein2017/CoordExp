#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.large_asset_sync import (
    apply_publish_manifest_updates,
    build_local_scan_report,
    classify_remote_against_manifest,
    load_manifests_for_policy,
    load_policy,
    managed_root_for_relative_path,
    manifest_to_file_index,
    write_manifests,
    plan_align_local,
    plan_publish,
    remote_path_for_relative_path,
    write_report,
)
from src.utils.large_asset_sync_baidupcs import (
    download_file_to_local_path,
    mkdir_remote_path,
    scan_remote_manifest_index,
    upload_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Large asset sync helper for repo-relative Baidu Netdisk mirroring."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("scan-local", "scan-remote", "publish", "align-local"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--policy", type=Path, required=True)
        sub.add_argument("--repo-root", type=Path, default=Path("."))
        sub.add_argument("--report", type=Path, required=True)
        sub.add_argument("--execute", action="store_true")
        sub.add_argument("--baidupcs-bin", default="BaiduPCS-Go")

    return parser.parse_args(argv)


def _build_manifest_file_index(
    manifests: dict[str, object],
) -> dict[str, dict[str, object]]:
    manifest_files: dict[str, dict[str, object]] = {}
    for manifest in manifests.values():
        manifest_files.update(manifest_to_file_index(manifest))
    return manifest_files


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy = load_policy(args.policy)
    repo_root = args.repo_root.resolve()

    if args.command == "scan-local":
        report = build_local_scan_report(repo_root=repo_root, policy=policy, include_hash=False)
        write_report(args.report, report)
        return 0

    manifests, manifest_paths = load_manifests_for_policy(repo_root, policy)
    manifest_files = _build_manifest_file_index(manifests)

    if args.command == "scan-remote":
        remote_scan = scan_remote_manifest_index(
            args.baidupcs_bin,
            manifest_files,
            remote_root=policy.remote_root,
        )
        report = {
            "status": "ok",
            **remote_scan,
            "diff": classify_remote_against_manifest(
                manifest_files,
                remote_scan["remote_files"],
                unknown_remote_state=remote_scan["unknown_remote_state"],
            ),
        }
        write_report(args.report, report)
        return 0

    local_scan = build_local_scan_report(repo_root=repo_root, policy=policy, include_hash=True)
    local_files = local_scan["local_files"]

    if args.command == "publish":
        plan = plan_publish(manifest_files, local_files)
        report = {"status": "ok", "plan": plan}
        write_report(args.report, report)
        if not args.execute:
            return 0

        planned_paths = sorted(plan["new"] + plan["changed"])
        if not planned_paths:
            return 0

        for relative_path in planned_paths:
            remote_path = remote_path_for_relative_path(policy.remote_root, relative_path)
            mkdir_remote_path(args.baidupcs_bin, str(Path(remote_path).parent).replace("\\", "/"))
            upload_file(args.baidupcs_bin, repo_root / relative_path, str(Path(remote_path).parent).replace("\\", "/"))

        manifests = apply_publish_manifest_updates(
            policy=policy,
            manifests=manifests,
            local_files=local_files,
            planned_paths=planned_paths,
        )

        verification_index = {
            relative_path: manifests[managed_root_for_relative_path(policy, relative_path).name]
            for relative_path in planned_paths
        }
        verification_manifest_files: dict[str, dict[str, object]] = {}
        for relative_path, manifest in verification_index.items():
            verification_manifest_files[relative_path] = manifest_to_file_index(manifest)[relative_path]

        remote_scan = scan_remote_manifest_index(
            args.baidupcs_bin,
            verification_manifest_files,
            remote_root=policy.remote_root,
        )
        remote_diff = classify_remote_against_manifest(
            verification_manifest_files,
            remote_scan["remote_files"],
            unknown_remote_state=remote_scan["unknown_remote_state"],
        )
        if remote_diff["missing_remote"] or remote_diff["metadata_drift_remote"] or remote_diff["unknown_remote_state"]:
            raise RuntimeError(f"Remote verification failed after publish: {remote_diff}")

        write_manifests(manifest_paths, manifests)
        return 0

    remote_scan = scan_remote_manifest_index(
        args.baidupcs_bin,
        manifest_files,
        remote_root=policy.remote_root,
    )
    plan = plan_align_local(
        manifest_files,
        local_files,
        remote_scan["remote_files"],
        unknown_remote_state=remote_scan["unknown_remote_state"],
    )
    report = {"status": "ok", "plan": plan}
    write_report(args.report, report)
    if not args.execute:
        return 0

    planned_paths = sorted(plan["missing_local"] + plan["metadata_drift_local"])
    for relative_path in planned_paths:
        remote_path = str(manifest_files[relative_path].get("remote_path") or remote_path_for_relative_path(policy.remote_root, relative_path))
        download_file_to_local_path(
            args.baidupcs_bin,
            remote_path,
            remote_root=policy.remote_root,
            local_path=repo_root / relative_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
