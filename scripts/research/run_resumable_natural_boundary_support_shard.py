#!/usr/bin/env python3
"""Run a bounded or full physical support-completion slot durably.

This is adoption glue for the exact active natural-boundary consumer.  It
hash-binds that consumer and every declared model/config input before opening a
journal or loading a model, then reuses the consumer's validated candidate bank
and exact scalar scorer one logical context at a time.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import resumable_natural_boundary_support_completion as adapter  # noqa: E402
from src.artifacts.json_values import json_sha256  # noqa: E402


EXPECTED_CONSUMER_SHA256 = adapter.EXPECTED_CONSUMER_SHA256
EXPECTED_CONFIG_SHA256 = "d2217208bc3e419bc9d8c621b4842da262d216780eb1dfc88eade0edf1358f6b"
EXPECTED_PLAN_SHA256 = "1b7e97af291b58aec50849ae09aa883148d55610c281a9edb543fce46ec21d4c"
EXPECTED_CENSUS_SHA256 = "dd1c61abb9acff7f4fc42380365439ee931db604525749f297bbc3e191a26a2e"
EXPECTED_ADAPTER_TENSOR_SHA256 = "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da"
EXPECTED_EMBEDDING_DELTA_SHA256 = "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2"
EXPECTED_RUNTIME_SOURCE_SHA256 = {
    "src/config/inference.py": "3e9b40f139610641700a4af2c60a38990a70b7280c9d1dc51c49d47042e6e229",
    "src/config/models.py": "8c4b79e2337f747354e6b58673b309f760b2845858d78325c37fc4a78cc68666",
    "src/inference/backend.py": "d3cd3b4d6c643fa3ffe4b1b57fbb7da377a4738a615103ac996b6b57f1ea4b36",
    "src/inference/runtime.py": "f529cd0a4e57352608d6122e5cfbd2c16c823d9dbf60a6c057036ee0be5239c3",
}
EXPECTED_BASE_MODEL = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
EXPECTED_BASE_MODEL_FILES = {
    "README.md": "5fc5be1ca9a3910399bd6239ee5086ab5d82a2a59c5d2b00e887a8835cc110e4",
    "added_tokens.json": "66eb4ae4a6c85b3aff4cd9f55dd63283ed70a4f8dee0ab6d32a5af2d3a077d45",
    "chat_template.jinja": "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4",
    "chat_template.json": "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
    "config.json": "c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de",
    "configuration.json": "2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0",
    "coord_init.json": "778d7859bc5bff0400a0abbf36d138c6b0856171d8e8402d22c7dfc38a2887d0",
    "coord_tokens.json": "93893c8bdeff19d487f2cca0941d0acda525f643401ab4867550bfcb5a63a3d7",
    "generation_config.json": "4d9818c3d27895c0058828a5f68bc7a4de80c3ae1bcdac90936ce24178063f59",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00002.safetensors": "2dff41296f9d817f9698bef43e31ce58fd06b5080978d31279d06f6766a695a4",
    "model-00002-of-00002.safetensors": "915ce8d4baabd778e76d32bd3d57c6c65c806c877a4fda47fc4b6f6f9e1510a5",
    "model.safetensors.index.json": "7ab471424a936028921a6c952e662457bb4fa6aedb1fb13232a0993a0ad97d61",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "special_tokens_map.json": "c0a3e7f5a114a881dc4ec4fe7fb0f88df406bf208befcc147655c47769e169aa",
    "tokenizer.json": "ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8",
    "tokenizer_config.json": "d30ea64dbc6941e97cefa9de22d913364cd8126a16ef1957f400e84cd4bcaf60",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
EXPECTED_SOURCE_GATE_ROOT = Path("/data/CoordExp")
EXPECTED_SOURCE_GATE_FILES = {
    "docs/history/architecture/proposals/2026-06-27-coordexp-swift/source-studies/special-token-embeddings.md": "e024f8f9754475cfa6ed81136eae6c72becd2aca53b7b03b9fa3047e8da7d193",
    "outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json": "4bd4b18464d2d2b29bcff534a8e40381c42da760b4ed0f4ff1e11d1327fbb348",
}


class ResumableWorkerError(ValueError):
    """Raised before or around the exact active consumer boundary."""


def _require_digest(path: str | Path, expected: str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    observed = adapter.file_sha256(candidate)
    if observed != expected:
        raise ResumableWorkerError(f"{label} SHA-256 differs from the admitted identity")
    return candidate.resolve(strict=True)


def _require_model_root(path: str | Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ResumableWorkerError("base model is not a regular non-symlink directory")
    resolved = candidate.resolve(strict=True)
    if resolved != EXPECTED_BASE_MODEL.resolve(strict=True):
        raise ResumableWorkerError("base model path differs from the admitted S checkpoint substrate")
    entries = sorted(resolved.iterdir(), key=lambda item: item.name)
    if [item.name for item in entries] != sorted(EXPECTED_BASE_MODEL_FILES):
        raise ResumableWorkerError("base model file denominator differs from the admitted checkpoint")
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ResumableWorkerError("base model contains a non-regular or symlink entry")
    files: dict[str, Any] = {}
    for name, expected in EXPECTED_BASE_MODEL_FILES.items():
        source = _require_digest(resolved / name, expected, label=f"base model {name}")
        files[name] = {"path": str(source), "sha256": expected}
    return resolved, {
        "path": str(resolved),
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": json_sha256(
            [{"name": name, "sha256": value["sha256"]} for name, value in sorted(files.items())]
        ),
    }


def _bind_source_gate_root(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ResumableWorkerError("embedding source-gate root is not a non-symlink directory")
    resolved = candidate.resolve(strict=True)
    if resolved != EXPECTED_SOURCE_GATE_ROOT.resolve(strict=True):
        raise ResumableWorkerError("embedding source-gate root differs from the admitted authority")
    files: dict[str, Any] = {}
    for relative, expected in EXPECTED_SOURCE_GATE_FILES.items():
        source = _require_digest(resolved / relative, expected, label=f"source gate {relative}")
        files[relative] = {"path": str(source), "sha256": expected}
    return {"root": str(resolved), "files": files}


def _import_bound_consumer(path: Path) -> Any:
    import scripts.research

    research_dir = str(path.parent)
    if research_dir not in scripts.research.__path__:
        scripts.research.__path__.append(research_dir)
    module_name = "scripts.research.run_natural_boundary_support_completion"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    loaded = Path(module.__file__).resolve(strict=True)
    if loaded != path:
        raise ResumableWorkerError("imported support consumer is not the digest-bound source")
    return module


def _bind_consumer_runtime_source(consumer: Path) -> dict[str, Any]:
    """Prefer the active consumer's transitive runtime source over this infra tree.

    The sibling has an uncommitted research-required ``geo_sorted_xy`` config
    schema that is absent from this infra branch.  Binding and selecting it is
    necessary to execute the named consumer without copying or editing it.
    """

    active_root = consumer.parents[2]
    observed: dict[str, Any] = {}
    for relative, expected in EXPECTED_RUNTIME_SOURCE_SHA256.items():
        path = _require_digest(active_root / relative, expected, label=relative)
        observed[relative] = {"path": str(path), "sha256": expected}
    import src

    active_src = str((active_root / "src").resolve(strict=True))
    if active_src in src.__path__:
        src.__path__.remove(active_src)
    src.__path__.insert(0, active_src)
    for module_name in tuple(sys.modules):
        if module_name.startswith("src.") and not module_name.startswith(
            ("src.artifacts", "src.common")
        ):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    entries: list[dict[str, str]] = []
    for path in sorted((active_root / "src").rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ResumableWorkerError("consumer runtime source tree contains a non-regular Python file")
        entries.append(
            {
                "path": path.relative_to(active_root).as_posix(),
                "sha256": adapter.file_sha256(path),
            }
        )
    if not entries:
        raise ResumableWorkerError("consumer runtime source tree contains no Python files")
    return {
        "critical_files": observed,
        "tree": {
            "root": str((active_root / "src").resolve(strict=True)),
            "algorithm": "sorted_relative_path_and_sha256_json.v1",
            "python_file_count": len(entries),
            "aggregate_sha256": json_sha256(entries),
        },
    }


class _LiveContextObserver:
    def __init__(
        self,
        *,
        runner: Any,
        plan: Mapping[str, Any],
        bank: Any,
        scorer: Any,
        device: str,
        runtime_receipt: Path,
        runtime_binding: Mapping[str, Any],
        source_gate_root: Path = EXPECTED_SOURCE_GATE_ROOT,
    ) -> None:
        self.runner = runner
        self.plan = plan
        self.bank = bank
        self.scorer = scorer
        self.device = device
        self.runtime_receipt = runtime_receipt
        self.runtime_binding = dict(runtime_binding)
        self.source_gate_root = source_gate_root
        self.opened = False
        self.runtime_identity: Mapping[str, Any] | None = None

    def _open(self) -> None:
        if self.opened:
            return
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.source_gate_root)
            self.scorer.open()
        finally:
            os.chdir(previous_cwd)
        self.opened = True
        runtime = self.runner.support_probe.validate_live_runtime_identity(
            self.scorer._session,
            expected_checkpoint=self.runner.CHECKPOINT,
            expected_config_fingerprint=self.plan["h0_lineage"]["config_fingerprint"],
        )
        if runtime.get("normalized_device") != self.device:
            raise ResumableWorkerError("opened scorer device differs from the bound logical device")
        receipt: dict[str, Any] = {
            "schema_version": adapter.MECHANICS_SCHEMA_VERSION,
            "kind": "live_worker_runtime_identity",
            "binding": self.runtime_binding,
            "runtime_identity": dict(runtime),
            "claim_boundary": "live_runtime_mechanics_only_no_support_or_model_mechanism_claim",
        }
        receipt["content_sha256"] = json_sha256(receipt)
        adapter.write_once_json(self.runtime_receipt, receipt)
        self.runtime_identity = runtime

    def close(self) -> None:
        if self.opened:
            self.scorer.close()
            self.opened = False

    def __call__(self, context: Mapping[str, Any]) -> dict[str, Any]:
        self._open()
        context_id = str(context["context_id"])
        score_map: dict[str, float] = {}
        failures = 0
        h0_row = self.bank.h0_by_owner[str(context["gt_owner_id"])]
        for candidate_id in context["candidate_ids"]:
            candidate = self.bank.by_id[str(candidate_id)]
            try:
                value = self.scorer.score(h0_row, candidate)
                score_map[str(candidate_id)] = self.runner._finite(
                    value, f"candidate score {context_id}.{candidate_id}"
                )
            except Exception:  # the exact consumer defines this as a durable failed context
                failures += 1
        scores = dict(sorted(score_map.items()))
        common: dict[str, Any] = {
            "context_id": context_id,
            "stable_key": context["stable_key"],
            "gt_owner_id": context["gt_owner_id"],
            "image_id": int(context["image_id"]),
            "candidate_ids": list(context["candidate_ids"]),
            "candidate_scores": scores,
            "candidate_score_count": len(scores),
            "candidate_scores_sha256": json_sha256(scores) if scores else None,
        }
        if failures or len(scores) != len(context["candidate_ids"]):
            return {
                **common,
                "status": "failed" if failures else "partial",
                "support_features": None,
                "failure_count": failures,
            }
        group = self.bank.groups[
            (int(context["image_id"]), str(context["category_name"]).lower())
        ]
        try:
            features = self.runner.support_probe.support_features(
                scores, group, owner_id=str(context["gt_owner_id"])
            )
        except Exception:
            return {
                **common,
                "status": "failed",
                "support_features": None,
                "failure_count": 1,
            }
        return {
            **common,
            "status": "measured",
            "support_features": dict(features),
            "failure_count": 0,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer", required=True, type=Path)
    parser.add_argument("--consumer-sha256", default=EXPECTED_CONSUMER_SHA256)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", default=EXPECTED_PLAN_SHA256)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--census-sha256", default=EXPECTED_CENSUS_SHA256)
    parser.add_argument("--infer-config", required=True, type=Path)
    parser.add_argument("--infer-config-sha256", default=EXPECTED_CONFIG_SHA256)
    parser.add_argument("--adapter-tensor", required=True, type=Path)
    parser.add_argument("--adapter-tensor-sha256", default=EXPECTED_ADAPTER_TENSOR_SHA256)
    parser.add_argument("--embedding-delta", required=True, type=Path)
    parser.add_argument("--embedding-delta-sha256", default=EXPECTED_EMBEDDING_DELTA_SHA256)
    parser.add_argument("--base-model", type=Path, default=EXPECTED_BASE_MODEL)
    parser.add_argument("--source-gate-root", type=Path, default=EXPECTED_SOURCE_GATE_ROOT)
    parser.add_argument("--journal-root", required=True, type=Path)
    parser.add_argument("--runtime-receipt", required=True, type=Path)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--physical-slot-index", type=int, default=0)
    parser.add_argument("--physical-slot-count", type=int, default=1)
    parser.add_argument("--context-id", action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    admitted = {
        "consumer": (args.consumer_sha256, EXPECTED_CONSUMER_SHA256),
        "plan": (args.plan_sha256, EXPECTED_PLAN_SHA256),
        "census": (args.census_sha256, EXPECTED_CENSUS_SHA256),
        "inference config": (args.infer_config_sha256, EXPECTED_CONFIG_SHA256),
        "adapter tensor": (args.adapter_tensor_sha256, EXPECTED_ADAPTER_TENSOR_SHA256),
        "embedding delta": (args.embedding_delta_sha256, EXPECTED_EMBEDDING_DELTA_SHA256),
    }
    drifted = [label for label, (supplied, expected) in admitted.items() if supplied != expected]
    if drifted:
        raise ResumableWorkerError(
            "CLI identity differs from the admitted source binding: " + ", ".join(drifted)
        )
    consumer = _require_digest(args.consumer, args.consumer_sha256, label="consumer")
    plan_path = _require_digest(args.plan, args.plan_sha256, label="plan")
    census = _require_digest(args.census, args.census_sha256, label="census")
    infer_config = _require_digest(
        args.infer_config, args.infer_config_sha256, label="inference config"
    )
    adapter_tensor = _require_digest(
        args.adapter_tensor, args.adapter_tensor_sha256, label="adapter tensor"
    )
    embedding_delta = _require_digest(
        args.embedding_delta, args.embedding_delta_sha256, label="embedding delta"
    )
    base_model, base_model_identity = _require_model_root(args.base_model)
    source_gate = _bind_source_gate_root(args.source_gate_root)
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_device or "," in visible_device or visible_device == "-1":
        raise ResumableWorkerError("worker requires exactly one caller-bound CUDA_VISIBLE_DEVICES token")
    if args.device != "cuda:0":
        raise ResumableWorkerError("single-visible-device worker must use logical cuda:0")

    runtime_source = _bind_consumer_runtime_source(consumer)
    runner = _import_bound_consumer(consumer)
    logical_plan = adapter.validate_sealed_consumer_plan(
        plan_path=plan_path,
        expected_plan_file_sha256=args.plan_sha256,
        census_path=census,
        runner_validate_execution_plan=runner.validate_execution_plan,
    )
    validated_plan, _ = runner.validate_execution_plan(
        plan_path,
        expected_plan_sha256=args.plan_sha256,
        census_path=census,
    )
    projected = adapter.project_logical_contexts(logical_plan, context_ids=args.context_id)
    schedule = adapter.plan_physical_slots(projected, slot_count=args.physical_slot_count)
    if not 0 <= args.physical_slot_index < args.physical_slot_count:
        raise ResumableWorkerError("physical slot is outside the bounded schedule")
    source_identity = {
        "consumer": {"path": str(consumer), "sha256": args.consumer_sha256},
        "consumer_runtime_source": runtime_source,
        "adapter": {
            "path": str(Path(adapter.__file__).resolve(strict=True)),
            "sha256": adapter.file_sha256(adapter.__file__),
        },
        "worker": {
            "path": str(Path(__file__).resolve(strict=True)),
            "sha256": adapter.file_sha256(__file__),
        },
        "logical_plan": {
            "path": str(plan_path),
            "file_sha256": logical_plan.file_sha256,
            "content_sha256": logical_plan.content_sha256,
        },
        "bounded_context_ids": list(args.context_id),
        "schedule_sha256": schedule["content_sha256"],
        "census": {"path": str(census), "sha256": args.census_sha256},
        "infer_config": {"path": str(infer_config), "sha256": args.infer_config_sha256},
        "model": {
            "base_model": base_model_identity,
            "adapter_tensor": {"path": str(adapter_tensor), "sha256": args.adapter_tensor_sha256},
            "embedding_delta": {"path": str(embedding_delta), "sha256": args.embedding_delta_sha256},
        },
        "embedding_source_gate": source_gate,
        "runtime_policy": {
            "logical_device": args.device,
            "cuda_visible_devices": visible_device,
            "physical_slot_index": args.physical_slot_index,
            "physical_slot_count": args.physical_slot_count,
        },
    }

    # Bind the root before candidate-bank construction or scorer/model work.
    journal, _ = adapter.open_slot_journal(
        root=args.journal_root,
        execution_id=args.execution_id,
        execution_identity=source_identity,
        plan=projected,
        schedule=schedule,
        slot_index=args.physical_slot_index,
        continuation=args.resume,
    )
    journal.close()
    bank = runner.build_candidate_bank(validated_plan)
    scorer = runner.make_hf_scorer(validated_plan, bank, infer_config)
    observer = _LiveContextObserver(
        runner=runner,
        plan=validated_plan,
        bank=bank,
        scorer=scorer,
        device=args.device,
        runtime_receipt=args.runtime_receipt,
        runtime_binding=source_identity,
        source_gate_root=Path(source_gate["root"]),
    )
    attempt_id, accepted = adapter.execute_slot(
        root=args.journal_root,
        execution_id=args.execution_id,
        execution_identity=source_identity,
        plan=projected,
        schedule=schedule,
        slot_index=args.physical_slot_index,
        observe_context=observer,
        continuation=True,
        cleanup=observer.close,
        install_sigterm_handler=True,
    )
    return {
        "attempt_id": attempt_id,
        "accepted_context_ids": list(accepted),
        "schedule": schedule,
        "execution_identity": source_identity,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run(args)
    except adapter.WorkerSIGTERM:
        # execute_slot has already best-effort recorded the attempt failure and
        # closed the scorer/journal.  Re-emit the real signal so the parent owns
        # an authoritative negative return code rather than a Python exit code.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)
        return 143  # pragma: no cover - SIG_DFL terminates first
    except (ResumableWorkerError, adapter.SupportShardAdapterError, OSError, ValueError) as exc:
        print(f"resumable-natural-boundary-support-worker: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
