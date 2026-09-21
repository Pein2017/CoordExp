from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.label_studio_coco_refinement.project as project_module  # noqa: E402
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY  # noqa: E402
from src.label_studio_coco_refinement.geometry import (  # noqa: E402
    norm1000_bbox_to_label_studio_xywh,
)
from src.label_studio_coco_refinement.label_config import (  # noqa: E402
    build_label_config,
    label_config_fingerprint,
)
from src.label_studio_coco_refinement.project import (  # noqa: E402
    BootstrapAction,
    LiveProjectAttestation,
    LiveTaskSetAttestation,
    RuntimeLayout,
    SOURCE_CONTRACTS,
    Split,
    build_split_project_plan,
    fingerprint_json,
    plan_instance_bootstrap,
)


VENDOR_REVISION = "pinned-vendor-revision"
ARTIFACT_ROOT = (
    REPO_ROOT
    / "outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000"
)


def _rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _file_stat(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "mode": oct(stat.S_IMODE(metadata.st_mode)),
        "owner_uid": metadata.st_uid,
        "is_regular": stat.S_ISREG(metadata.st_mode),
        "is_symlink": stat.S_ISLNK(metadata.st_mode),
    }


def _artifact_inventory(schema_version: int) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for split in (Split.TRAIN, Split.VAL):
        directory = ARTIFACT_ROOT / split.value / "bootstrap-task-index"
        for prefix in (f"build-v{schema_version}-", f"task-index-v{schema_version}-"):
            for path in sorted(directory.glob(prefix + "*")):
                inventory[str(path)] = _file_stat(path)
    return inventory


def _build_split(split: Split, label_config: str, fingerprint: str):
    contract = SOURCE_CONTRACTS[split]
    return build_split_project_plan(
        contract.path(REPO_ROOT),
        repo_root=REPO_ROOT,
        split=split,
        vendor_revision=VENDOR_REVISION,
        label_config=label_config,
        label_config_fingerprint=fingerprint,
        registry=COCO80_REGISTRY,
        bbox_converter=norm1000_bbox_to_label_studio_xywh,
    )


class _CreateAdapter:
    def attest_bootstrap_manifest(self) -> None:
        return None

    def attest_project(self, _desired) -> None:
        return None


def _live_attestation(desired, project_id: int) -> LiveProjectAttestation:
    task_count = desired.task_manifest.task_count
    return LiveProjectAttestation(
        split=desired.split,
        project_id=project_id,
        project_identity=desired.manifest.project_identity,
        saved_manifest=desired.manifest.to_dict(),
        vendor_revision=desired.manifest.vendor_revision,
        label_config=desired.label_config,
        controls=desired.controls,
        storage_manifest=desired.storage_manifest,
        managed_link_is_symlink=True,
        managed_link_resolved_target=desired.storage_manifest.managed_link_target,
        task_set=LiveTaskSetAttestation(
            expected_task_manifest_fingerprint=desired.task_manifest.fingerprint,
            observed_task_count=task_count,
            missing_task_count=0,
            content_fingerprint=fingerprint_json(
                {
                    "benchmark": "synthetic-reuse-attestation",
                    "project_id": project_id,
                    "task_count": task_count,
                }
            ),
        ),
    )


class _ReuseAdapter:
    def __init__(self, desired: Mapping[Split, Any]) -> None:
        self._manifest = project_module._build_instance_bootstrap_manifest(
            desired,
            layout=RuntimeLayout.for_repo(REPO_ROOT),
        ).to_dict()
        self._projects = {
            Split.TRAIN: _live_attestation(desired[Split.TRAIN], 101),
            Split.VAL: _live_attestation(desired[Split.VAL], 102),
        }

    def attest_bootstrap_manifest(self) -> dict[str, Any]:
        return self._manifest

    def attest_project(self, desired) -> LiveProjectAttestation:
        return self._projects[desired.split]


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, int] = {}
    by_path_and_stage: dict[str, int] = {}
    for event in events:
        stage = event["stage"]
        by_stage[stage] = by_stage.get(stage, 0) + 1
        key = f"{event['path']}::{stage}"
        by_path_and_stage[key] = by_path_and_stage.get(key, 0) + 1
    return {
        "by_stage": by_stage,
        "by_path_and_stage": by_path_and_stage,
        "events": events,
    }


def _receipt(
    *,
    mode: str,
    started_ns: int,
    elapsed_seconds: float,
    phases: dict[str, float],
    plans: Mapping[Split, Any],
    bootstrap,
    events: list[dict[str, Any]],
    rss_start_kib: int,
    artifacts_before_v1: dict[str, dict[str, Any]],
    artifacts_after_v1: dict[str, dict[str, Any]],
    artifacts_before_v2: dict[str, dict[str, Any]],
    artifacts_after_v2: dict[str, dict[str, Any]],
    parser_forbidden: bool,
) -> dict[str, Any]:
    source_receipts = {}
    index_receipts = {}
    for split in (Split.TRAIN, Split.VAL):
        source = plans[split].source_inspection
        index = plans[split].task_manifest.task_index
        source_receipts[split.value] = {
            "path": source.source_path,
            "sha256": source.sha256,
            "row_count": source.row_count,
            "box_count": source.box_count,
            "source_identity_fingerprint": source.task_identity_fingerprint,
            "stat": _file_stat(Path(source.source_path)),
        }
        index_receipts[split.value] = {
            **index.to_dict(),
            "stat": _file_stat(Path(index.path)),
        }
    return {
        "benchmark_contract": "coordexp-label-studio-bootstrap-v2",
        "mode": mode,
        "pid": os.getpid(),
        "process_started_monotonic_ns": started_ns,
        "python": sys.executable,
        "repo_root": str(REPO_ROOT),
        "command": f"conda run -n ms python {Path(__file__).resolve()} {mode}",
        "elapsed_seconds": elapsed_seconds,
        "phase_seconds": phases,
        "rss": {
            "start_kib": rss_start_kib,
            "max_kib": _rss_kib(),
            "increase_kib": _rss_kib() - rss_start_kib,
        },
        "actions": {
            action.project.split.value: action.action.value
            for action in bootstrap.projects
        },
        "source_receipts": source_receipts,
        "task_index_receipts": index_receipts,
        "stable_attestation": _summarize_events(events),
        "full_record_parser_forbidden": parser_forbidden,
        "artifact_inventory": {
            "schema_v1_before": artifacts_before_v1,
            "schema_v1_after": artifacts_after_v1,
            "schema_v1_unchanged": artifacts_before_v1 == artifacts_after_v1,
            "schema_v2_before": artifacts_before_v2,
            "schema_v2_after": artifacts_after_v2,
            "schema_v2_unchanged": artifacts_before_v2 == artifacts_after_v2,
        },
        "code_sha256": {
            "project.py": hashlib.sha256(
                (REPO_ROOT / "src/label_studio_coco_refinement/project.py").read_bytes()
            ).hexdigest(),
            "test_project.py": hashlib.sha256(
                (
                    REPO_ROOT
                    / "tests/label_studio_coco_refinement/test_project.py"
                ).read_bytes()
            ).hexdigest(),
            "benchmark_bootstrap.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "scope": {
            "adapter": "synthetic read-only attestation adapter",
            "database_touched": False,
            "network_touched": False,
            "source_or_v1_deleted": False,
        },
    }


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"create", "reuse"}:
        raise SystemExit("usage: benchmark_bootstrap.py {create|reuse}")
    mode = sys.argv[1]
    before_v1 = _artifact_inventory(1)
    before_v2 = _artifact_inventory(2)
    if mode == "create" and before_v2:
        raise RuntimeError("CREATE benchmark requires an absent schema-v2 artifact set")
    if mode == "reuse" and len(before_v2) != 4:
        raise RuntimeError("REUSE benchmark requires two v2 anchors and two v2 sidecars")

    events: list[dict[str, Any]] = []

    def hook(path: Path, stage: str, byte_count: int) -> None:
        events.append(
            {
                "path": str(path),
                "stage": stage,
                "byte_count": byte_count,
            }
        )

    project_module._clear_byte_attestation_cache()
    project_module._STABLE_FILE_TEST_HOOK = hook
    parser_forbidden = mode == "reuse"
    if parser_forbidden:
        def forbidden_record_parser(*_args, **_kwargs):
            raise AssertionError("warm startup attempted full task-index parsing")

        project_module._iter_task_index_records = forbidden_record_parser

    config = build_label_config()
    config_fingerprint = label_config_fingerprint(config)
    phases: dict[str, float] = {}
    started_ns = time.monotonic_ns()
    rss_start_kib = _rss_kib()
    overall_start = time.perf_counter()

    phase_start = time.perf_counter()
    train = _build_split(Split.TRAIN, config, config_fingerprint)
    phases["build_train"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    val = _build_split(Split.VAL, config, config_fingerprint)
    phases["build_val"] = time.perf_counter() - phase_start

    desired = {Split.TRAIN: train, Split.VAL: val}
    adapter = _CreateAdapter() if mode == "create" else _ReuseAdapter(desired)
    phase_start = time.perf_counter()
    bootstrap = plan_instance_bootstrap(desired, adapter)
    phases["plan_instance_bootstrap"] = time.perf_counter() - phase_start
    elapsed_seconds = time.perf_counter() - overall_start

    expected_action = (
        BootstrapAction.CREATE if mode == "create" else BootstrapAction.REUSE
    )
    if any(action.action is not expected_action for action in bootstrap.projects):
        raise AssertionError("bootstrap action drifted from benchmark mode")

    after_v1 = _artifact_inventory(1)
    after_v2 = _artifact_inventory(2)
    if before_v1 != after_v1:
        raise AssertionError("schema-v1 artifact was rewritten")
    if len(after_v2) != 4:
        raise AssertionError("expected exactly two v2 anchors and two v2 sidecars")
    if mode == "reuse" and before_v2 != after_v2:
        raise AssertionError("warm startup rewrote a schema-v2 artifact")

    print(
        json.dumps(
            _receipt(
                mode=mode,
                started_ns=started_ns,
                elapsed_seconds=elapsed_seconds,
                phases=phases,
                plans=desired,
                bootstrap=bootstrap,
                events=events,
                rss_start_kib=rss_start_kib,
                artifacts_before_v1=before_v1,
                artifacts_after_v1=after_v1,
                artifacts_before_v2=before_v2,
                artifacts_after_v2=after_v2,
                parser_forbidden=parser_forbidden,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
