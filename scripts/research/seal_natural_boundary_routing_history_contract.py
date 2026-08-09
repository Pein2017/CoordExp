#!/usr/bin/env python3
"""Seal the CPU-only contract for the natural-boundary routing study.

This command is deliberately an administrative sealer.  It reads the three
contract documents and immutable evidence from the prior study, but it never
loads a checkpoint, starts a runner, or changes an upstream artifact.  The
result is a canonical JSON document whose self hash binds the exact contract
and source identities used to create it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "natural_boundary_routing_history_contract.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_PATH = REPO_ROOT / (
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    f"{UNIT_ID}/unit.md"
)
TASKS_PATH = UNIT_PATH.with_name("tasks.md")
LAUNCH_GATE_PATH = UNIT_PATH.with_name("launch-gate.md")
TEST_PATH = REPO_ROOT / "tests/research/test_seal_natural_boundary_routing_history_contract.py"
PRIOR_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover"
)
DEFAULT_OUTPUT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    f"{UNIT_ID}/contract.json"
)

# The upstream evidence count is part of this sealer's contract.  These are
# not frozen byte hashes: the hashes are read from unit.md at seal time so a
# lead can make final contract edits without changing this source.
UPSTREAM_ARTIFACT_COUNT = 7
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TABLE_HASH_RE = re.compile(
    r"^\s*\|\s*(?P<label>[^|]+?)\s*\|\s*`?(?P<sha>[0-9a-f]{64})`?\s*\|\s*$",
    re.IGNORECASE,
)

BOUNDARY_CONTRACT: dict[str, Any] = {
    "primary": {
        "checkpoint": "S",
        "step": 2444,
        "role": "decision_owner",
        "substrate": "four-coordinate geo_sorted_xy",
    },
    "secondary": {
        "checkpoint": "A3",
        "step": 2445,
        "role": "optional_sensitized_contrast",
        "promotion_authority": False,
    },
    "training": {
        "authorized": False,
        "mode": "no_training",
        "optimizer_steps": False,
        "weight_updates": False,
    },
    "mutations": {
        "old_artifacts_overwrite": False,
        "wrapper_or_special_token_change": False,
        "production_launch": False,
    },
}

# These are a conservative fixed contract when no review receipt is supplied.
# They state that review is required/pending; they do not claim that a review
# passed.  A caller may replace this mapping with a reviewed, provenance-bearing
# mapping via build_contract or --review-dispositions-json.
FIXED_REVIEW_DISPOSITIONS: dict[str, Any] = {
    "independent_contract_review": {
        "disposition": "required_before_implementation",
        "status": "pending",
        "source": "launch-gate.md",
    },
    "scientific_scope_review": {
        "disposition": "required_before_gpu",
        "status": "pending",
        "source": "launch-gate.md",
    },
}


class ContractSealError(ValueError):
    """Raised when the natural-boundary contract is not exact or complete."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractSealError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractSealError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def document_self_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("self_sha256", None)
    return sha256_json(payload)


def _resolved_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ContractSealError(f"{label} must be a regular non-symlink file: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ContractSealError(f"{label} does not exist: {candidate}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise ContractSealError(f"{label} must be a regular non-symlink file: {candidate}")
    return resolved


def file_ref(path: str | Path, label: str) -> dict[str, Any]:
    resolved = _resolved_file(path, label)
    payload = resolved.read_bytes()
    return {
        "label": label,
        "path": str(resolved),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContractSealError(f"cannot read {label} {path}: {exc}") from exc


def _read_json(path: Path, label: str) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractSealError(f"cannot read {label} {path}: {exc}") from exc
    canonical_json_bytes(payload)
    return payload


def _parse_frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", text, flags=re.DOTALL)
    if match is None:
        raise ContractSealError("unit.md is missing a YAML frontmatter block")
    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _validate_boundaries(unit_text: str, tasks_text: str, launch_gate_text: str) -> None:
    frontmatter = _parse_frontmatter(unit_text)
    if frontmatter.get("unit_id") != UNIT_ID:
        raise ContractSealError(
            f"unit_id mismatch: expected {UNIT_ID!r}, observed {frontmatter.get('unit_id')!r}"
        )

    # Keep these checks semantic but tolerant of harmless prose edits.  They
    # protect the authority split without freezing the whole unit's bytes.
    folded = re.sub(r"[_-]+", " ", unit_text).lower()
    if "s step 2444" not in folded:
        raise ContractSealError("unit does not bind S step-2444")
    if not re.search(r"s[^\n]{0,100}primary", folded):
        raise ContractSealError("unit does not declare S as primary")
    if "a3 step 2445" not in folded:
        raise ContractSealError("unit does not bind A3 step-2445")
    if not re.search(r"a3[^\n]{0,140}secondary", folded):
        raise ContractSealError("unit does not declare A3 as secondary")
    if not re.search(r"no\s+training|without\s+training|performs\s+no\s+training", folded):
        raise ContractSealError("unit does not preserve the no-training boundary")
    if re.search(r"training_promotion_status\s*:\s*promoted", unit_text, flags=re.IGNORECASE):
        raise ContractSealError("unit promotes training despite the no-training boundary")
    if "run s" not in tasks_text.lower() or "a3" not in tasks_text.lower():
        raise ContractSealError("tasks.md does not bind S work and the A3 secondary boundary")
    if "no training" not in re.sub(r"[-_]", " ", launch_gate_text.lower()):
        raise ContractSealError("launch-gate.md does not preserve the no-training boundary")


def parse_declared_upstream_hashes(unit_text: str) -> list[dict[str, str]]:
    """Read the immutable evidence table from unit.md, without path guesses."""
    rows: list[dict[str, str]] = []
    for line in unit_text.splitlines():
        match = TABLE_HASH_RE.match(line)
        if match is None:
            continue
        label = match.group("label").strip()
        digest = match.group("sha").lower()
        rows.append({"label": label, "sha256": digest})
    if len(rows) != UPSTREAM_ARTIFACT_COUNT:
        raise ContractSealError(
            "unit.md must declare exactly "
            f"{UPSTREAM_ARTIFACT_COUNT} immutable upstream artifact hashes; observed {len(rows)}"
        )
    digests = [row["sha256"] for row in rows]
    if len(set(digests)) != len(digests):
        raise ContractSealError("unit.md declares duplicate upstream artifact hashes")
    return rows


def _discover_artifact(root: Path, digest: str, label: str) -> Path:
    resolved_root = Path(root).expanduser().resolve(strict=True)
    if not resolved_root.is_dir():
        raise ContractSealError(f"prior output root is not a directory: {root}")
    matches: list[Path] = []
    for candidate in sorted(resolved_root.rglob("*"), key=lambda item: str(item)):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            observed = sha256_file(candidate)
        except ContractSealError:
            continue
        if observed == digest:
            matches.append(candidate.resolve())
    if len(matches) != 1:
        detail = ", ".join(str(path) for path in matches[:4]) or "none"
        raise ContractSealError(
            f"{label} hash {digest} must resolve to exactly one file under {resolved_root}; "
            f"observed {len(matches)} ({detail})"
        )
    return matches[0]


def discover_upstream_artifacts(
    unit_path: str | Path,
    prior_output_root: str | Path,
) -> list[dict[str, Any]]:
    unit = _resolved_file(unit_path, "unit.md")
    rows = parse_declared_upstream_hashes(_read_text(unit, "unit.md"))
    root = Path(prior_output_root).expanduser().resolve(strict=True)
    artifacts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        path = _discover_artifact(root, row["sha256"], row["label"])
        observed = file_ref(path, f"upstream[{index}]")
        if observed["sha256"] != row["sha256"]:
            raise ContractSealError(f"upstream[{index}] changed while sealing")
        relative = path.relative_to(root)
        artifacts.append(
            {
                "index": index,
                "label": row["label"],
                "declared_sha256": row["sha256"],
                "observed_sha256": observed["sha256"],
                "path": str(path),
                "relative_path": relative.as_posix(),
                "size_bytes": observed["size_bytes"],
            }
        )
    return artifacts


def _review_mapping(review_dispositions: Mapping[str, Any] | None) -> tuple[dict[str, Any], str]:
    if review_dispositions is None:
        return json.loads(json.dumps(FIXED_REVIEW_DISPOSITIONS)), "fixed_contract"
    if not isinstance(review_dispositions, Mapping) or not review_dispositions:
        raise ContractSealError("caller-supplied review dispositions must be a non-empty object")
    # Round-trip through canonical JSON to reject non-JSON values and aliases.
    payload = json.loads(canonical_json_bytes(dict(review_dispositions)).decode("utf-8"))
    return payload, "caller_supplied"


def _source_identity(
    unit_path: Path,
    tasks_path: Path,
    launch_gate_path: Path,
    sealer_path: Path | None,
    tests_path: Path | None,
) -> dict[str, Any]:
    contract_files = {
        "unit": file_ref(unit_path, "unit.md"),
        "tasks": file_ref(tasks_path, "tasks.md"),
        "launch_gate": file_ref(launch_gate_path, "launch-gate.md"),
    }
    if sealer_path is None:
        sealer_path = Path(__file__).resolve()
    if tests_path is None:
        tests_path = TEST_PATH
    source_files = {
        "sealer": file_ref(sealer_path, "sealer source"),
        "focused_tests": file_ref(tests_path, "focused test source"),
    }
    return {"contract_files": contract_files, "source_files": source_files}


def build_contract(
    *,
    unit_path: str | Path = UNIT_PATH,
    tasks_path: str | Path = TASKS_PATH,
    launch_gate_path: str | Path = LAUNCH_GATE_PATH,
    prior_output_root: str | Path = PRIOR_OUTPUT_ROOT,
    review_dispositions: Mapping[str, Any] | None = None,
    sealer_path: str | Path | None = None,
    tests_path: str | Path | None = TEST_PATH,
) -> dict[str, Any]:
    unit = _resolved_file(unit_path, "unit.md")
    tasks = _resolved_file(tasks_path, "tasks.md")
    launch_gate = _resolved_file(launch_gate_path, "launch-gate.md")
    unit_text = _read_text(unit, "unit.md")
    tasks_text = _read_text(tasks, "tasks.md")
    launch_gate_text = _read_text(launch_gate, "launch-gate.md")
    _validate_boundaries(unit_text, tasks_text, launch_gate_text)

    identities = _source_identity(
        unit,
        tasks,
        launch_gate,
        None if sealer_path is None else _resolved_file(sealer_path, "sealer source"),
        None if tests_path is None else _resolved_file(tests_path, "focused test source"),
    )
    upstream = discover_upstream_artifacts(unit, prior_output_root)
    reviews, review_source = _review_mapping(review_dispositions)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed",
        "unit_id": UNIT_ID,
        "boundary_contract": BOUNDARY_CONTRACT,
        "review_dispositions": reviews,
        "review_disposition_source": review_source,
        "contract_files": identities["contract_files"],
        "source_files": identities["source_files"],
        "prior_output_root": str(Path(prior_output_root).expanduser().resolve(strict=True)),
        "upstream_artifacts": upstream,
    }
    document["self_sha256"] = document_self_sha256(document)
    return document


def validate_contract(document: Mapping[str, Any]) -> None:
    if not isinstance(document, Mapping):
        raise ContractSealError("contract must be a JSON object")
    required = {
        "schema_version",
        "status",
        "unit_id",
        "boundary_contract",
        "review_dispositions",
        "review_disposition_source",
        "contract_files",
        "source_files",
        "prior_output_root",
        "upstream_artifacts",
        "self_sha256",
    }
    if set(document) != required:
        raise ContractSealError("contract top-level schema is not exact")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ContractSealError("contract schema mismatch")
    if document.get("status") != "sealed" or document.get("unit_id") != UNIT_ID:
        raise ContractSealError("contract status or unit_id is not sealed for this unit")
    if document.get("boundary_contract") != BOUNDARY_CONTRACT:
        raise ContractSealError("boundary contract drift")
    if not isinstance(document.get("review_dispositions"), Mapping) or not document["review_dispositions"]:
        raise ContractSealError("review dispositions are missing")
    if document.get("review_disposition_source") not in {"fixed_contract", "caller_supplied"}:
        raise ContractSealError("review disposition source is invalid")
    for field in ("contract_files", "source_files"):
        refs = document.get(field)
        if not isinstance(refs, Mapping) or not refs:
            raise ContractSealError(f"{field} are missing")
        expected_keys = (
            {"unit", "tasks", "launch_gate"}
            if field == "contract_files"
            else {"sealer", "focused_tests"}
        )
        if set(refs) != expected_keys:
            raise ContractSealError(f"{field} key set is not exact")
        for label, ref in refs.items():
            if not isinstance(ref, Mapping):
                raise ContractSealError(f"{field}.{label} is not an object")
            raw_path = ref.get("path")
            if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).expanduser().is_absolute():
                raise ContractSealError(f"{field}.{label} path is not absolute")
            digest = ref.get("sha256")
            size_bytes = ref.get("size_bytes")
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise ContractSealError(f"{field}.{label} has an invalid SHA-256")
            if not isinstance(size_bytes, int) or size_bytes < 0:
                raise ContractSealError(f"{field}.{label} has an invalid size")
            # unit.md is the immutable authority for this receipt.  The task
            # checklist and launch gate are captured snapshots: their live
            # files may legitimately advance after sealing.  Runner/test
            # source identities remain strict so a changed implementation or
            # focused verifier cannot silently validate an old receipt.
            strict = field == "source_files" or (field == "contract_files" and label == "unit")
            if not strict:
                continue
            path = _resolved_file(str(ref.get("path")), f"{field}.{label}")
            observed = sha256_file(path)
            if ref.get("sha256") != observed or ref.get("size_bytes") != path.stat().st_size:
                raise ContractSealError(f"{field}.{label} bytes differ from the sealed identity")
    artifacts = document.get("upstream_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != UPSTREAM_ARTIFACT_COUNT:
        raise ContractSealError("upstream artifact count drift")
    prior_root = Path(str(document.get("prior_output_root"))).expanduser()
    try:
        prior_root = prior_root.resolve(strict=True)
    except OSError as exc:
        raise ContractSealError("prior output root is unavailable") from exc
    if not prior_root.is_dir():
        raise ContractSealError("prior output root is not a directory")
    seen: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ContractSealError(f"upstream artifact {index} is not an object")
        digest = artifact.get("declared_sha256")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None or digest in seen:
            raise ContractSealError(f"upstream artifact {index} has invalid or duplicate digest")
        seen.add(digest)
        path = _resolved_file(str(artifact.get("path")), f"upstream[{index}]")
        try:
            relative = path.relative_to(prior_root)
        except ValueError as exc:
            raise ContractSealError(f"upstream artifact {index} escaped prior output root") from exc
        if artifact.get("relative_path") != relative.as_posix():
            raise ContractSealError(f"upstream artifact {index} relative path drift")
        observed = sha256_file(path)
        if artifact.get("observed_sha256") != digest or observed != digest:
            raise ContractSealError(f"upstream artifact {index} bytes differ from declared hash")
        if artifact.get("size_bytes") != path.stat().st_size:
            raise ContractSealError(f"upstream artifact {index} size differs")
    self_hash = document.get("self_sha256")
    if not isinstance(self_hash, str) or SHA256_RE.fullmatch(self_hash) is None:
        raise ContractSealError("contract self_sha256 is invalid")
    if self_hash != document_self_sha256(document):
        raise ContractSealError("contract self_sha256 mismatch")


def _serialized_document(document: Mapping[str, Any]) -> bytes:
    validate_contract(document)
    return canonical_json_bytes(document) + b"\n"


def write_immutable(document: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    content = _serialized_document(document)
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise FileExistsError(f"output collision: {target} is not a regular file")
        existing = target.read_bytes()
        if existing != content:
            raise FileExistsError(f"output collision: refusing to overwrite {target}")
        return {
            "status": "sealed",
            "path": str(target),
            "sha256": sha256_bytes(existing),
            "byte_identical": True,
        }
    try:
        with target.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        # A concurrent writer may have won the race.  It is safe only when the
        # winner wrote the exact same immutable bytes.
        existing = target.read_bytes()
        if existing != content:
            raise FileExistsError(f"output collision: refusing to overwrite {target}")
        return {
            "status": "sealed",
            "path": str(target),
            "sha256": sha256_bytes(existing),
            "byte_identical": True,
        }
    return {
        "status": "sealed",
        "path": str(target),
        "sha256": sha256_bytes(content),
        "byte_identical": False,
    }


def seal(
    output: str | Path = DEFAULT_OUTPUT,
    **kwargs: Any,
) -> dict[str, Any]:
    document = build_contract(**kwargs)
    result = write_immutable(document, output)
    persisted = _read_json(Path(result["path"]), "sealed contract")
    if not isinstance(persisted, Mapping):
        raise ContractSealError("sealed contract is not a JSON object")
    validate_contract(persisted)
    return {**result, "self_sha256": persisted["self_sha256"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", "--unit-path", dest="unit_path", type=Path, default=UNIT_PATH)
    parser.add_argument("--tasks", "--tasks-path", dest="tasks_path", type=Path, default=TASKS_PATH)
    parser.add_argument(
        "--launch-gate",
        "--launch-gate-path",
        dest="launch_gate_path",
        type=Path,
        default=LAUNCH_GATE_PATH,
    )
    parser.add_argument(
        "--prior-output-root",
        "--upstream-root",
        dest="prior_output_root",
        type=Path,
        default=PRIOR_OUTPUT_ROOT,
    )
    parser.add_argument("--output", "--output-path", dest="output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tests", "--tests-path", dest="tests_path", type=Path, default=TEST_PATH)
    parser.add_argument("--sealer", "--sealer-path", dest="sealer_path", type=Path, default=Path(__file__))
    parser.add_argument(
        "--review-dispositions-json",
        "--review-disposition-json",
        type=Path,
        help="JSON object containing caller-supplied review dispositions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reviews = None
        if args.review_dispositions_json is not None:
            value = _read_json(_resolved_file(args.review_dispositions_json, "review dispositions"), "review dispositions")
            if not isinstance(value, Mapping):
                raise ContractSealError("review dispositions JSON must be an object")
            reviews = value
        result = seal(
            args.output,
            unit_path=args.unit_path,
            tasks_path=args.tasks_path,
            launch_gate_path=args.launch_gate_path,
            prior_output_root=args.prior_output_root,
            tests_path=args.tests_path,
            sealer_path=args.sealer_path,
            review_dispositions=reviews,
        )
    except (ContractSealError, FileExistsError, OSError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
