#!/usr/bin/env python3
"""Create, inspect, or release the standalone editor's temporary Focus Queue."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any


DEFAULT_SERVER = "http://127.0.0.1:53662"


def _server_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port is None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise argparse.ArgumentTypeError(
            "server must be canonical loopback HTTP with an explicit port"
        )
    return f"http://{parsed.hostname}:{parsed.port}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the one persistent COCO refinement Focus Queue."
    )
    parser.add_argument("--server", type=_server_url, default=DEFAULT_SERVER)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "create", help="Create one ordered queue from exact image paths."
    )
    create.add_argument("image_paths", type=Path, nargs="+")
    commands.add_parser("status", help="Print the active queue and work status.")
    commands.add_parser(
        "release", help="Delete only terminal queue metadata and keep annotations."
    )
    commands.add_parser(
        "retry-publication",
        help="Retry a failed training-pair publish without recapturing Drafts.",
    )
    return parser


class _Client:
    def __init__(self, server: str) -> None:
        self.server = server.rstrip("/")
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        self.csrf_token = ""

    def bootstrap(self) -> None:
        payload = self.request("GET", "/api/session", mutation=False)
        token = payload.get("csrf_token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise RuntimeError("session response lacks a CSRF token")
        self.csrf_token = token

    def request(
        self,
        method: str,
        path: str,
        *,
        body: object | None = None,
        mutation: bool = False,
    ) -> dict[str, Any]:
        encoded = None
        headers = {"Accept": "application/json"}
        if body is not None:
            encoded = json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if mutation:
            if not self.csrf_token:
                raise RuntimeError("local session is not initialized")
            headers["Origin"] = self.server
            headers["x-csrf-token"] = self.csrf_token
        request = urllib.request.Request(
            self.server + path, data=encoded, headers=headers, method=method
        )
        try:
            with self.opener.open(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = error.get("message") if isinstance(error, dict) else None
            details = error.get("details") if isinstance(error, dict) else None
            if message and details:
                message = f"{message}: {json.dumps(details, ensure_ascii=False)}"
            raise RuntimeError(message or f"server rejected request ({exc.code})") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach {self.server}; start the COCO refinement service first"
            ) from exc
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("server returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("server returned a non-object JSON response")
        return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    client = _Client(args.server)
    try:
        client.bootstrap()
        if args.command == "create":
            payload = client.request(
                "POST",
                "/api/focus",
                body={"image_paths": [str(path) for path in args.image_paths]},
                mutation=True,
            )
        elif args.command == "release":
            payload = client.request("DELETE", "/api/focus", mutation=True)
        elif args.command == "retry-publication":
            payload = client.request(
                "POST", "/api/focus/publication/retry", body={}, mutation=True
            )
        else:
            payload = client.request("GET", "/api/focus")
    except RuntimeError as exc:
        print(f"coco-refinement-focus: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
