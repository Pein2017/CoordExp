from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from .parser import SessionRecord, parse_rollout
from .pricing import load_rates
from .attempts import (
    annotate_attempts,
    load_outcomes,
    summarize_attempt_routes,
    summarize_attempts,
    summarize_route_pairs,
)
from .report import enrich_record, summarize, summarize_totals


_DATE_IN_FILENAME = re.compile(r"rollout-(\d{4}-\d{2}-\d{2})T")


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def _file_date(path: Path) -> date | None:
    match = _DATE_IN_FILENAME.search(path.name)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def discover_files(
    root: Path, since: date | None, until: date | None, limit: int | None
) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("rollout-*.jsonl"):
        file_date = _file_date(path)
        if since and file_date and file_date < since:
            continue
        if until and file_date and file_date > until:
            continue
        paths.append(path)
    paths.sort()
    return paths[-limit:] if limit else paths


def _scan(
    paths: Iterable[Path],
    include_root: bool,
    *,
    thread_id: str | None = None,
    session_id: str | None = None,
    root_thread_id: str | None = None,
) -> tuple[list[SessionRecord], list[SessionRecord], dict[str, Any]]:
    all_records: list[SessionRecord] = []
    files_seen = 0
    for path in paths:
        files_seen += 1
        record = parse_rollout(path)
        all_records.append(record)
    records, scope = filter_records(
        all_records,
        include_root,
        thread_id=thread_id,
        session_id=session_id,
        root_thread_id=root_thread_id,
    )
    return records, all_records, {
        "files_seen": files_seen,
        **scope,
        "records_emitted": len(records),
        "skipped_non_subagents": scope["scope_records"] - len(records),
        "parse_errors": sum(record.parse_errors for record in records),
    }


def filter_records(
    records: Iterable[SessionRecord],
    include_root: bool,
    *,
    thread_id: str | None = None,
    session_id: str | None = None,
    root_thread_id: str | None = None,
) -> tuple[list[SessionRecord], dict[str, Any]]:
    """Select an exact thread, a persisted session, or a root thread subtree."""

    scope_args = [
        value
        for value in (thread_id, session_id, root_thread_id)
        if value is not None
    ]
    if len(scope_args) > 1:
        raise ValueError(
            "--thread-id, --session-id, and --root-thread-id are mutually exclusive"
        )

    materialized = list(records)
    if thread_id is not None:
        scoped = [record for record in materialized if record.thread_id == thread_id]
        scope_filter = "thread"
    elif session_id is not None:
        scoped = [record for record in materialized if record.session_id == session_id]
        scope_filter = "session"
    elif root_thread_id is not None:
        children_by_parent: dict[str, set[str]] = {}
        for record in materialized:
            if record.thread_id is None or record.parent_thread_id is None:
                continue
            children_by_parent.setdefault(record.parent_thread_id, set()).add(
                record.thread_id
            )
        subtree = {root_thread_id}
        frontier = [root_thread_id]
        while frontier:
            parent = frontier.pop()
            for child in children_by_parent.get(parent, set()):
                if child not in subtree:
                    subtree.add(child)
                    frontier.append(child)
        scoped = [
            record for record in materialized if record.thread_id in subtree
        ]
        scope_filter = "root_thread_subtree"
    else:
        scoped = materialized
        scope_filter = "all"

    if scope_args and not scoped:
        requested = scope_args[0]
        raise ValueError(f"scope did not match any rollout record: {requested}")

    emitted = [record for record in scoped if include_root or record.is_subagent]
    return emitted, {
        "scope_filter": scope_filter,
        "scope_records": len(scoped),
    }


def _write_json(path: Path, value: Any, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2 if pretty else None, sort_keys=pretty
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(stream: Any, items: list[dict[str, Any]]) -> None:
    fields = [
        "file",
        "thread_id",
        "parent_thread_id",
        "task_label",
        "agent_role",
        "model_provider",
        "model",
        "effort",
        "status",
        "started_at",
        "ended_at",
        "attempt_id",
        "disposition",
        "disposition_source",
        "followup_count",
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
        "pricing_status",
        "estimated_cost",
        "currency",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for item in items:
        usage = item.get("measured_usage") or item.get("latest_total_usage") or {}
        pricing = item.get("pricing") or {}
        writer.writerow(
            {
                **item,
                "attempt_id": (item.get("attempt") or {}).get("attempt_id"),
                "disposition": (item.get("attempt") or {}).get("disposition"),
                "disposition_source": (item.get("attempt") or {}).get(
                    "disposition_source"
                ),
                "followup_count": (item.get("attempt") or {}).get("followup_count"),
                **usage,
                "pricing_status": pricing.get("status"),
                "estimated_cost": pricing.get("amount"),
                "currency": pricing.get("currency"),
            }
        )


def build_parser() -> argparse.ArgumentParser:
    default_root = (
        Path(os.environ["CODEX_HOME"]) / "sessions"
        if os.environ.get("CODEX_HOME")
        else None
    )
    parser = argparse.ArgumentParser(
        description="Build an offline token and approximate cost ledger from Codex rollout JSONL files."
    )
    parser.add_argument(
        "--sessions",
        type=Path,
        default=default_root,
        help="CODEX_HOME/sessions directory (defaults to $CODEX_HOME/sessions)",
    )
    parser.add_argument(
        "--prices", type=Path, help="TOML file containing [[rates]] entries"
    )
    parser.add_argument(
        "--since", type=_parse_date, help="inclusive rollout date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--until", type=_parse_date, help="inclusive rollout date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--max-files", type=int, help="limit the newest matching rollout files"
    )
    parser.add_argument(
        "--include-root", action="store_true", help="include non-subagent root sessions"
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--thread-id",
        help="emit one exact rollout thread (root requires --include-root)",
    )
    scope.add_argument(
        "--session-id",
        help="emit rollout records whose persisted session_id matches exactly",
    )
    scope.add_argument(
        "--root-thread-id",
        help=(
            "emit a root thread and all descendants; omit --include-root for "
            "children-only accounting"
        ),
    )
    parser.add_argument(
        "--output", type=Path, help="write the session report to this file"
    )
    parser.add_argument(
        "--summary-out", type=Path, help="write grouped route/task summary JSON"
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        help="optional JSONL dispositions keyed by attempt_id or thread_id",
    )
    parser.add_argument(
        "--disposition-policy",
        choices=("strict", "completed", "followup_aware"),
        default="strict",
        help="acceptance labels: explicit only, completion proxy, or followup-aware proxy",
    )
    parser.add_argument("--format", choices=("jsonl", "json", "csv"), default="jsonl")
    parser.add_argument(
        "--pretty", action="store_true", help="pretty-print JSON output"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sessions is None:
        print("set CODEX_HOME or pass --sessions", file=sys.stderr)
        return 2
    if not args.sessions.is_dir():
        print(f"sessions directory does not exist: {args.sessions}", file=sys.stderr)
        return 2
    if args.max_files is not None and args.max_files <= 0:
        print("--max-files must be positive", file=sys.stderr)
        return 2

    try:
        rates = load_rates(args.prices)
        outcomes = load_outcomes(args.outcomes)
        paths = discover_files(args.sessions, args.since, args.until, args.max_files)
        records, all_records, scan = _scan(
            paths,
            args.include_root,
            thread_id=args.thread_id,
            session_id=args.session_id,
            root_thread_id=args.root_thread_id,
        )
    except (OSError, ValueError) as exc:
        print(f"scan failed: {exc}", file=sys.stderr)
        return 2

    items = [enrich_record(record, rates) for record in records]
    items = annotate_attempts(
        items, all_records, outcomes=outcomes, policy=args.disposition_policy
    )
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "sessions_root": str(args.sessions),
        "filters": {
            "since": args.since.isoformat() if args.since else None,
            "until": args.until.isoformat() if args.until else None,
            "max_files": args.max_files,
            "include_root": args.include_root,
            "thread_id": args.thread_id,
            "session_id": args.session_id,
            "root_thread_id": args.root_thread_id,
            "outcomes": str(args.outcomes) if args.outcomes else None,
            "disposition_policy": args.disposition_policy,
        },
        "totals": summarize_totals(items),
        "attempts": summarize_attempts(items, args.disposition_policy),
        "attempt_routes": summarize_attempt_routes(items),
        "route_pairs": summarize_route_pairs(items),
        "scan": scan,
        "groups": summarize(items),
    }

    if args.summary_out:
        _write_json(args.summary_out, summary, args.pretty)

    output_stream = (
        open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    )
    try:
        if args.format == "jsonl":
            for item in items:
                output_stream.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
        elif args.format == "json":
            _write_target = {**summary, "sessions": items}
            output_stream.write(
                json.dumps(
                    _write_target,
                    ensure_ascii=False,
                    indent=2 if args.pretty else None,
                    sort_keys=args.pretty,
                )
                + "\n"
            )
        else:
            _write_csv(output_stream, items)
    finally:
        if args.output:
            output_stream.close()
    return 0
