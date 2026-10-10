"""CPU-only tests for Pi Web peer-session transport using its real HTTP wire shape."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import runpy
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


SCRIPT = Path(__file__).with_name("pi_worker_turn.py")
LEAD = "00000000-0000-4000-8000-000000000001"
WORKER = "00000000-0000-4000-8000-000000000002"
CREATED = "00000000-0000-4000-8000-000000000003"
PEER = "00000000-0000-4000-8000-000000000004"


def session(session_id: str, cwd: str, *, loaded: bool = True,
            provider: str = "openai", model_id: str = "gpt-6.1-sol",
            effort: str = "xhigh", streaming: bool = False) -> dict[str, object]:
    return {
        "id": session_id,
        "cwd": cwd,
        "name": "fixture",
        "messageCount": 2,
        "relation": None,
        "parentSessionId": None,
        "transient": False,
        "loaded": loaded,
        "model": {"provider": provider, "modelId": model_id},
        "effort": effort,
        "streaming": streaming,
        "prompt_count": 0,
        "name_updates": 0,
    }


class PiWebFixture:
    def __init__(self, cwd: Path) -> None:
        self.cwd = str(cwd)
        self.sessions = {
            LEAD: session(LEAD, self.cwd, loaded=True, effort="low"),
            WORKER: session(WORKER, self.cwd, loaded=False, effort="xhigh"),
        }
        self.calls: list[tuple[str, str, dict[str, object] | None]] = []
        self.mutations: list[dict[str, object]] = []
        self.receipt_path: Path | None = None
        self.disconnect_create = False
        self.disconnect_prompt = False
        self.http_error_after_create = False
        self.redirect_on_get_state = False
        self.redirect_hits = 0
        self.runtime_override: dict[str, object] = {}
        self.info_overrides: dict[str, dict[str, object]] = {}
        self.catalog = {
            "modelList": [{"provider": "openai", "id": "gpt-6-sol", "name": "Sol"}],
            "thinkingLevels": {"openai:gpt-6-sol": ["off", "high"]},
        }

    def new_server(self) -> ThreadingHTTPServer:
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                return

            def _body(self) -> dict[str, object]:
                size = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(size).decode("utf-8")) if size else {}
                return value if isinstance(value, dict) else {}

            def _json(self, status: int, value: dict[str, object]) -> None:
                raw = json.dumps(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _disconnect(self) -> None:
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()

            def _record_mutation(self, path: str, body: dict[str, object]) -> None:
                receipt = None
                if fixture.receipt_path is not None and fixture.receipt_path.exists():
                    try:
                        receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        receipt = "unreadable"
                fixture.mutations.append({"path": path, "body": body, "receipt": receipt})

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlsplit(self.path)
                path = parsed.path
                fixture.calls.append(("GET", self.path, None))
                if path == "/api/models":
                    if parse_qs(parsed.query).get("cwd") != [fixture.cwd]:
                        self._json(400, {"error": "wrong cwd"})
                    else:
                        self._json(200, fixture.catalog)
                    return
                if path == "/redirected":
                    fixture.redirect_hits += 1
                    self._json(200, {"unexpected": True})
                    return
                prefix = "/api/sessions/"
                if path.startswith(prefix):
                    ident = path[len(prefix):]
                    item = fixture.sessions.get(ident)
                    if item is None:
                        self._json(404, {"error": "Session not found"})
                        return
                    info = {k: item[k] for k in (
                        "id", "cwd", "name", "messageCount", "parentSessionId", "relation", "transient",
                    )}
                    info.update(fixture.info_overrides.get(ident, {}))
                    context = {
                        # Prove the CLI does not save or print even the bounded transcript field.
                        "messages": [{"role": "assistant", "content": "TRANSCRIPT_SECRET"}],
                        "entryIds": ["fixture-entry"],
                        "oldestEntryId": "fixture-entry",
                        "hasMore": False,
                        "thinkingLevel": item["effort"],
                        "model": item["model"],
                    }
                    self._json(200, {
                        "sessionId": ident,
                        "filePath": "/fixture/session.jsonl",
                        "info": info,
                        "tree": {"summary": "TRANSCRIPT_SECRET"},
                        "context": context,
                    })
                    return
                prefix = "/api/agent/"
                if path.startswith(prefix):
                    ident = path[len(prefix):]
                    item = fixture.sessions.get(ident)
                    if item is None:
                        self._json(200, {"running": False})
                    elif item["loaded"]:
                        self._json(200, {"running": True, "state": fixture.state(item)})
                    else:
                        self._json(200, {"running": False})
                    return
                self._json(404, {"error": "unknown path"})

            def do_POST(self) -> None:  # noqa: N802
                body = self._body()
                path = urlsplit(self.path).path
                fixture.calls.append(("POST", path, body))
                self._record_mutation(path, body)
                if path == "/api/agent/new":
                    created = session(CREATED, str(body.get("cwd")), loaded=True,
                                      provider=str(body.get("provider")), model_id=str(body.get("modelId")),
                                      effort=str(body.get("thinkingLevel")))
                    created["messageCount"] = 0
                    created["transient"] = True
                    fixture.sessions[CREATED] = created
                    if fixture.disconnect_create:
                        self._disconnect()
                        return
                    if fixture.http_error_after_create:
                        self._json(500, {"error": "creation response failed", "accepted": False})
                        return
                    if body.get("type") != "ensure_session":
                        self._json(400, {"error": "wrong create type"})
                        return
                    self._json(200, {
                        "success": True,
                        "sessionId": CREATED,
                        "data": None,
                        "model": {"provider": created["model"]["provider"],
                                  "modelId": created["model"]["modelId"]},
                        "thinkingLevel": created["effort"],
                    })
                    return
                prefix = "/api/agent/"
                if path.startswith(prefix):
                    ident = path[len(prefix):]
                    item = fixture.sessions.get(ident)
                    if item is None:
                        self._json(404, {"error": "Session not found", "accepted": False})
                        return
                    command_type = body.get("type")
                    if command_type == "get_state":
                        if fixture.redirect_on_get_state:
                            self.send_response(302)
                            self.send_header("Location", "/redirected")
                            self.send_header("Content-Length", "0")
                            self.end_headers()
                            return
                        item["loaded"] = True
                        state = fixture.state(item)
                        self._json(200, {"success": True, "data": state})
                        return
                    if command_type == "set_session_name":
                        item["name"] = body.get("name")
                        item["name_updates"] = int(item["name_updates"]) + 1
                        self._json(200, {"success": True, "data": None})
                        return
                    if command_type == "prompt":
                        item["prompt_count"] = int(item["prompt_count"]) + 1
                        item["messageCount"] = int(item["messageCount"]) + 1
                        item["transient"] = False
                        fixture.calls.append(("PROMPT_BODY", path, body))
                        if fixture.disconnect_prompt:
                            self._disconnect()
                            return
                        self._json(200, {"success": True, "data": None})
                        return
                    self._json(400, {"error": "unexpected command"})
                    return
                self._json(404, {"error": "unknown path"})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        return server

    @staticmethod
    def state(item: dict[str, object]) -> dict[str, object]:
        model = item["model"]
        assert isinstance(model, dict)
        data: dict[str, object] = {
            "sessionId": item["id"],
            "sessionFile": "/fixture/session.jsonl",
            "isStreaming": item["streaming"],
            "isPromptRunning": item["streaming"],
            "model": {"id": model["modelId"], "provider": model["provider"]},
            "thinkingLevel": item["effort"],
        }
        data.update(item.get("runtime_override", {}))
        return data


class PiWorkerTurnTest(unittest.TestCase):
    def test_cwd_arguments_require_unambiguous_expected_directories(self) -> None:
        parse = runpy.run_path(str(SCRIPT))["parse_args"]
        with self.assertRaises(SystemExit):
            parse(["--lead-session", LEAD, "--worker-session", WORKER,
                   "--lead-cwd", str(self.root)])
        with self.assertRaises(SystemExit):
            parse(["--lead-session", LEAD, "--worker-session", WORKER,
                   "--cwd", str(self.root), "--lead-cwd", str(self.root),
                   "--worker-cwd", str(self.worker_root)])

    def test_default_endpoint_and_delivery_arguments(self) -> None:
        parse = runpy.run_path(str(SCRIPT))["parse_args"]
        args = parse(["--cwd", str(Path.cwd()), "--list-models"])
        self.assertEqual(args.base_url, "http://127.0.0.1:12345")
        with self.assertRaises(SystemExit):
            parse(["--cwd", str(Path.cwd()), "--list-models", "--delivery", "steering"])

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "lead-worktree"
        self.root.mkdir()
        self.worker_root = Path(self.temp.name) / "worker-worktree"
        self.worker_root.mkdir()
        self.peer_root = Path(self.temp.name) / "third-worktree"
        self.peer_root.mkdir()
        self.fixture = PiWebFixture(self.root)
        self.fixture.sessions[PEER] = session(PEER, str(self.peer_root), loaded=False,
                                              effort="medium")
        self.server = self.fixture.new_server()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def message_file(self, content: str = "Inspect the assigned artifact and report evidence.") -> Path:
        path = self.root / "message.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def run_command(self, command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        # The CLI must use the explicit local endpoint even when ambient proxy variables exist.
        env.update({"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1",
                    "HTTPS_PROXY": "http://127.0.0.1:1", "https_proxy": "http://127.0.0.1:1",
                    "NO_PROXY": "", "no_proxy": ""})
        return subprocess.run(command, capture_output=True, text=True, env=env, cwd=cwd, timeout=10)

    def run_cli(self, args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return self.run_command([sys.executable, str(SCRIPT), *args], cwd=cwd)

    def create_args(self, receipt: Path, message: Path) -> list[str]:
        return ["--base-url", self.base, "--lead-session", LEAD, "--cwd", str(self.root),
                "--create", "--provider", "openai", "--model", "gpt-6-sol",
                "--effort", "high", "--name", "bounded-peer", "--send",
                "--message", str(message), "--receipt", str(receipt)]

    def pair_args(self, receipt: Path | None = None, message: Path | None = None,
                  to: str = "worker") -> list[str]:
        values = ["--base-url", self.base, "--lead-session", LEAD, "--worker-session", WORKER,
                  "--cwd", str(self.root), "--to", to]
        if receipt is not None and message is not None:
            values += ["--send", "--message", str(message), "--receipt", str(receipt)]
        return values

    def split_cwd_pair_args(self, to: str = "worker") -> list[str]:
        return ["--base-url", self.base, "--lead-session", LEAD, "--worker-session", WORKER,
                "--lead-cwd", str(self.root), "--worker-cwd", str(self.worker_root), "--to", to]

    def test_create_verifies_names_then_sends_with_identity_and_direct_return(self) -> None:
        message = self.message_file("Inspect only the assigned artifact; do not expand scope.")
        receipt = self.root / "create.json"
        self.fixture.receipt_path = receipt
        result = self.run_cli(self.create_args(receipt, message))
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(output["status"], "accepted_or_queued")
        catalog_index = next(i for i, call in enumerate(self.fixture.calls)
                             if call[1].startswith("/api/models?"))
        create_index = next(i for i, call in enumerate(self.fixture.calls)
                            if call[:2] == ("POST", "/api/agent/new"))
        self.assertLess(catalog_index, create_index)
        self.assertEqual(saved["worker_session_id"], CREATED)
        self.assertEqual(saved["worker"]["name"], "bounded-peer")
        self.assertTrue(self.fixture.sessions[CREATED]["transient"] is False)
        self.assertNotIn("TRANSCRIPT_SECRET", result.stdout)
        self.assertNotIn("TRANSCRIPT_SECRET", receipt.read_text(encoding="utf-8"))

        paths = [path for method, path, _ in self.fixture.calls if method == "POST"]
        self.assertEqual(paths, ["/api/agent/new", f"/api/agent/{CREATED}", f"/api/agent/{CREATED}"])
        bodies = [body for method, path, body in self.fixture.calls if method == "POST"]
        self.assertEqual(bodies[0], {
            "cwd": str(self.root), "type": "ensure_session", "provider": "openai",
            "modelId": "gpt-6-sol", "thinkingLevel": "high",
        })
        self.assertEqual(bodies[1], {"type": "set_session_name", "name": "bounded-peer"})
        self.assertEqual(bodies[2]["type"], "prompt")
        self.assertEqual(bodies[2]["streamingBehavior"], "followUp")
        prompt = str(bodies[2]["message"])
        self.assertIn("not a user instruction, new authorization", prompt)
        self.assertIn(f"sender_session_id={LEAD}", prompt)
        self.assertIn(f"recipient_session_id={CREATED}", prompt)
        self.assertIn(f"dispatch_id={saved['dispatch_id']}", prompt)
        self.assertIn("Inspect only the assigned artifact; do not expand scope.", prompt)
        self.assertIn("--to lead", prompt)
        self.assertIn("--lead-session " + LEAD, prompt)
        self.assertIn("--worker-session " + CREATED, prompt)
        self.assertEqual([m["receipt"]["phase"] for m in self.fixture.mutations], [
            "create_session", "set_session_name", "send_first_prompt",
        ])
        self.assertEqual([m["receipt"]["status"] for m in self.fixture.mutations], [
            "creation_unknown", "created", "delivery_unknown",
        ])

        # The generated return command must execute the retained public script from another cwd.
        return_command = next(line for line in prompt.splitlines() if "--to lead" in line)
        argv = shlex.split(return_command)
        self.assertEqual(Path(argv[1]), SCRIPT.resolve())
        report = self.peer_root / "return-report.txt"
        report.write_text("Return report body from the worker.", encoding="utf-8")
        return_receipt = self.peer_root / "return-receipt.json"
        argv[argv.index("REPORT_FILE")] = str(report)
        argv[argv.index("NEW_RECEIPT_FILE")] = str(return_receipt)
        returned = self.run_command(argv, cwd=self.peer_root)
        self.assertEqual(returned.returncode, 0, returned.stderr)
        return_output = json.loads(returned.stdout)
        return_saved = json.loads(return_receipt.read_text(encoding="utf-8"))
        self.assertEqual(return_output["target_session_id"], LEAD)
        self.assertEqual(return_saved["reply_to_dispatch_id"], saved["dispatch_id"])
        return_prompt = next(body for kind, _, body in self.fixture.calls
                             if kind == "PROMPT_BODY" and f"sender_session_id={CREATED}" in body["message"])
        self.assertEqual(return_prompt["streamingBehavior"], "followUp")

    def test_pair_send_supports_distinct_worktrees_and_inline_text(self) -> None:
        self.fixture.sessions[WORKER]["cwd"] = str(self.worker_root)
        receipt = self.root / "cross-worktree-send.json"
        message = "Quick peer check: report whether the target is reachable."
        reply_to = "00000000-0000-4000-8000-000000000099"
        self.fixture.receipt_path = receipt
        result = self.run_cli(self.split_cwd_pair_args() + [
            "--send", "--text", message, "--reply-to", reply_to, "--receipt", str(receipt),
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["lead"]["cwd"], str(self.root))
        self.assertEqual(output["worker"]["cwd"], str(self.worker_root))
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(saved["lead_cwd"], str(self.root))
        self.assertEqual(saved["worker_cwd"], str(self.worker_root))
        self.assertNotIn("cwd", saved)
        self.assertEqual(saved["reply_to_dispatch_id"], reply_to)
        prompt = next(body for kind, _, body in self.fixture.calls
                      if kind == "PROMPT_BODY")["message"]
        self.assertIn(f"reply_to_dispatch_id={reply_to}", prompt)
        self.assertIn(f"sender_session_id={LEAD}", prompt)
        self.assertIn(f"recipient_session_id={WORKER}", prompt)
        self.assertIn(message, prompt)

        self.fixture.calls.clear()
        self.fixture.info_overrides[WORKER] = {"cwd": str(self.root)}
        rejected = self.run_cli(self.split_cwd_pair_args() + [
            "--send", "--text", message, "--receipt", str(self.root / "wrong-peer-cwd.json"),
        ])
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))

    def test_legacy_send_keeps_message_file_body_and_report_pointer_separate(self) -> None:
        message = self.message_file("Only this selected file is the brief body.")
        pointer = self.root / "unavailable-report.md"
        receipt = self.root / "legacy-reference.json"
        result = self.run_cli(self.pair_args(receipt, message) + [
            "--report-path", str(pointer), "--report-section", "Findings",
            "--revision", "v3", "--supersedes", "one rendered chart",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        prompt = next(body for kind, _, body in self.fixture.calls
                      if kind == "PROMPT_BODY")["message"]
        self.assertEqual(saved["report_reference"], {
            "path": str(pointer), "section": "Findings", "revision": "v3",
            "supersedes": "one rendered chart",
        })
        self.assertEqual(saved["message_sha256"], hashlib.sha256(
            b"Only this selected file is the brief body.").hexdigest())
        self.assertIn("Only this selected file is the brief body.", prompt)
        self.assertIn(str(pointer), prompt)
        self.assertFalse(pointer.exists())
        self.assertEqual(json.loads(result.stdout)["delivery"], "queue")

        for reference_args in (("--report-section", "Findings"),
                               ("--report-path", "relative.md")):
            with self.subTest(reference_args=reference_args, mode="legacy"):
                self.fixture.calls.clear()
                invalid = self.run_cli(self.pair_args() + [
                    "--send", "--text", "brief", "--receipt", str(self.root / "invalid-ref.json"),
                    *reference_args,
                ])
                self.assertEqual(invalid.returncode, 2)
                self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))
            with self.subTest(reference_args=reference_args, mode="peer"):
                self.fixture.calls.clear()
                invalid = self.run_cli([
                    "--base-url", self.base,
                    "--sender", LEAD, "--sender-cwd", str(self.root),
                    "--recipient", PEER, "--recipient-cwd", str(self.peer_root),
                    "--send", "--text", "brief", "--receipt", str(self.root / "invalid-peer-ref.json"),
                    *reference_args,
                ])
                self.assertEqual(invalid.returncode, 2)
                self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))

    def test_create_can_target_a_distinct_worker_worktree_and_generates_return_route(self) -> None:
        self.fixture.cwd = str(self.worker_root)
        message = self.message_file("Inspect the assigned artifact and report the result.")
        receipt = self.root / "cross-worktree-create.json"
        self.fixture.receipt_path = receipt
        result = self.run_cli([
            "--base-url", self.base, "--lead-session", LEAD,
            "--lead-cwd", str(self.root), "--worker-cwd", str(self.worker_root),
            "--create", "--provider", "openai", "--model", "gpt-6-sol",
            "--effort", "high", "--name", "cross-worktree-peer", "--send",
            "--message", str(message), "--receipt", str(receipt),
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(saved["lead_cwd"], str(self.root))
        self.assertEqual(saved["worker_cwd"], str(self.worker_root))
        create_body = next(body for method, path, body in self.fixture.calls
                           if method == "POST" and path == "/api/agent/new")
        self.assertEqual(create_body["cwd"], str(self.worker_root))
        prompt = next(body for method, path, body in self.fixture.calls
                      if method == "POST" and path.endswith(f"/{CREATED}")
                      and body.get("type") == "prompt")["message"]
        self.assertIn("--lead-cwd", prompt)
        self.assertIn(str(self.root), prompt)
        self.assertIn("--worker-cwd", prompt)
        self.assertIn(str(self.worker_root), prompt)
        self.assertIn(f"--reply-to {saved['dispatch_id']}", prompt)

    def test_list_models_needs_no_session_and_never_mutates(self) -> None:
        result = self.run_cli(["--base-url", self.base, "--cwd", str(self.root), "--list-models"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["models"], [
            {"provider": "openai", "model": "gpt-6-sol", "efforts": ["off", "high"]},
        ])
        self.assertEqual(len(self.fixture.calls), 1)
        self.assertEqual(self.fixture.calls[0][0], "GET")

    def test_list_models_can_use_worker_cwd_without_session_pair(self) -> None:
        self.fixture.cwd = str(self.worker_root)
        result = self.run_cli([
            "--base-url", self.base, "--worker-cwd", str(self.worker_root), "--list-models",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["cwd"], str(self.worker_root))
        self.assertEqual(output["worker_cwd"], str(self.worker_root))
        self.assertEqual(parse_qs(urlsplit(self.fixture.calls[0][1]).query)["cwd"], [str(self.worker_root)])
        self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))

    def test_create_catalog_rejections_happen_before_any_mutation(self) -> None:
        for catalog in (
            {"modelList": [], "thinkingLevels": {}},
            {"modelList": [{"provider": "other", "id": "gpt-6-sol"}],
             "thinkingLevels": {"other:gpt-6-sol": ["high"]}},
            {"modelList": [{"provider": "openai", "id": "gpt-6-sol"}],
             "thinkingLevels": {"openai:gpt-6-sol": ["off"]}},
            {"modelList": [{"provider": "openai", "id": "gpt-6-sol"}], "thinkingLevels": {}},
            {"modelError": "PRIVATE_DETAIL", "modelList": [], "thinkingLevels": {}},
        ):
            with self.subTest(catalog=catalog):
                self.fixture.catalog = catalog
                self.fixture.calls.clear()
                receipt = self.root / "rejected.json"
                result = self.run_cli(self.create_args(receipt, self.message_file()))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(receipt.exists())
                self.assertFalse(any(c[0] == "POST" for c in self.fixture.calls))
                self.assertNotIn("PRIVATE_DETAIL", result.stderr)

    def test_inspection_is_read_only_and_rejects_relation_cwd_and_loaded_settings_drift(self) -> None:
        inspected = self.run_cli(self.pair_args())
        self.assertEqual(inspected.returncode, 0, inspected.stderr)
        self.assertEqual(json.loads(inspected.stdout)["status"], "inspected_only")
        self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))
        self.assertNotIn("TRANSCRIPT_SECRET", inspected.stdout)

        for ident in (LEAD, WORKER):
            with self.subTest(relation=ident):
                self.fixture.info_overrides[ident] = {
                    "parentSessionId": LEAD,
                    "relation": {"kind": "subagent", "parentSessionId": LEAD},
                }
                receipt = self.root / f"relation-{ident}.json"
                result = self.run_cli(self.pair_args(receipt, self.message_file(), "worker"))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(receipt.exists())
                self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))
                self.fixture.info_overrides.pop(ident)
                self.fixture.calls.clear()

        self.fixture.info_overrides[WORKER] = {"cwd": "/wrong/cwd"}
        result = self.run_cli(self.pair_args(self.root / "wrong-cwd.json", self.message_file()))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))
        self.fixture.info_overrides.clear()
        self.fixture.calls.clear()

        self.fixture.runtime_override = {"model": {"id": "changed", "provider": "openai"}}
        self.fixture.sessions[WORKER]["runtime_override"] = dict(self.fixture.runtime_override)
        receipt = self.root / "settings-drift.json"
        sent = self.run_cli(self.pair_args(receipt, self.message_file()))
        self.assertNotEqual(sent.returncode, 0)
        self.assertFalse(any(body and body.get("type") == "prompt"
                             for _, _, body in self.fixture.calls))
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "prepared")
        self.assertEqual(saved["phase"], "load_target_state")

    def test_existing_send_loads_unloaded_target_and_routes_in_both_directions(self) -> None:
        message = self.message_file("Return the concrete finding to the named peer.")
        receipt = self.root / "worker-send.json"
        self.fixture.receipt_path = receipt
        result = self.run_cli(self.pair_args(receipt, message, "worker"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.fixture.sessions[WORKER]["loaded"])
        post_types = [body.get("type") for method, _, body in self.fixture.calls
                      if method == "POST" and body]
        self.assertEqual(post_types, ["get_state", "prompt"])
        get_state = next(m for m in self.fixture.mutations if m["body"].get("type") == "get_state")
        prompt_mutation = next(m for m in self.fixture.mutations if m["body"].get("type") == "prompt")
        self.assertEqual(self.fixture.mutations[0]["body"]["type"], "get_state")
        self.assertIsNotNone(get_state["receipt"], "receipt must exist before target loading")
        self.assertEqual(get_state["receipt"]["phase"], "load_target_state")
        self.assertEqual(prompt_mutation["receipt"]["status"], "delivery_unknown")
        worker_prompt = next(body for kind, _, body in self.fixture.calls if kind == "PROMPT_BODY")
        self.assertIn(f"sender_session_id={LEAD}", str(worker_prompt["message"]))
        self.assertIn(f"recipient_session_id={WORKER}", str(worker_prompt["message"]))

        self.fixture.calls.clear()
        self.fixture.mutations.clear()
        self.fixture.receipt_path = self.root / "lead-return.json"
        self.fixture.sessions[LEAD]["streaming"] = True
        lead_receipt = self.fixture.receipt_path
        returned = self.run_cli(self.pair_args(lead_receipt, message, "lead"))
        self.assertEqual(returned.returncode, 0, returned.stderr)
        self.assertEqual(json.loads(returned.stdout)["target_session_id"], LEAD)
        lead_prompt = next(body for kind, _, body in self.fixture.calls if kind == "PROMPT_BODY")
        self.assertIn(f"sender_session_id={WORKER}", str(lead_prompt["message"]))
        self.assertIn(f"recipient_session_id={LEAD}", str(lead_prompt["message"]))
        self.assertEqual(lead_prompt["streamingBehavior"], "followUp")

    def test_steering_routes_both_directions_busy_and_idle_with_durable_mode(self) -> None:
        for to, target in (("worker", WORKER), ("lead", LEAD)):
            for busy in (False, True):
                with self.subTest(to=to, busy=busy):
                    self.fixture.calls.clear()
                    self.fixture.mutations.clear()
                    self.fixture.sessions[target]["streaming"] = busy
                    receipt = self.root / f"steer-{to}-{busy}.json"
                    self.fixture.receipt_path = receipt
                    sent = self.run_cli(self.pair_args(receipt, self.message_file(), to)
                                        + ["--delivery", "steering"])
                    self.assertEqual(sent.returncode, 0, sent.stderr)
                    prompt = next(c for c in self.fixture.calls if c[0] == "PROMPT_BODY")
                    self.assertEqual(prompt[1], f"/api/agent/{target}")
                    self.assertEqual(prompt[2]["streamingBehavior"], "steer")
                    self.assertEqual(json.loads(sent.stdout)["delivery"], "steering")
                    self.assertEqual(json.loads(receipt.read_text())["delivery"], "steering")
                    self.assertTrue(all(m["receipt"]["delivery"] == "steering"
                                        for m in self.fixture.mutations))

    def test_steering_unknown_delivery_does_not_resend(self) -> None:
        receipt = self.root / "steer-unknown.json"
        self.fixture.receipt_path = receipt
        self.fixture.disconnect_prompt = True
        args = self.pair_args(receipt, self.message_file(), "lead") + ["--delivery", "steering"]
        self.assertNotEqual(self.run_cli(args).returncode, 0)
        saved = json.loads(receipt.read_text())
        self.assertEqual(saved["status"], "delivery_unknown")
        self.assertEqual(saved["delivery"], "steering")
        self.assertNotEqual(self.run_cli(args).returncode, 0)
        self.assertEqual(sum(c[0] == "PROMPT_BODY" for c in self.fixture.calls), 1)

    def test_unknown_create_send_redirect_and_reused_receipts_never_duplicate(self) -> None:
        message = self.message_file()
        create_receipt = self.root / "unknown-create.json"
        self.fixture.receipt_path = create_receipt
        self.fixture.disconnect_create = True
        created = self.run_cli(self.create_args(create_receipt, message))
        self.assertNotEqual(created.returncode, 0)
        self.assertEqual(json.loads(create_receipt.read_text())["status"], "creation_unknown")
        create_posts = sum(method == "POST" and path == "/api/agent/new"
                           for method, path, _ in self.fixture.calls)
        retried = self.run_cli(self.create_args(create_receipt, message))
        self.assertNotEqual(retried.returncode, 0)
        self.assertEqual(sum(method == "POST" and path == "/api/agent/new"
                             for method, path, _ in self.fixture.calls), create_posts)

        self.fixture.disconnect_create = False
        self.fixture.calls.clear()
        send_receipt = self.root / "unknown-send.json"
        self.fixture.receipt_path = send_receipt
        self.fixture.disconnect_prompt = True
        sent = self.run_cli(self.pair_args(send_receipt, message))
        self.assertNotEqual(sent.returncode, 0)
        self.assertEqual(json.loads(send_receipt.read_text())["status"], "delivery_unknown")
        prompt_count = sum(kind == "PROMPT_BODY" for kind, _, _ in self.fixture.calls)
        retried = self.run_cli(self.pair_args(send_receipt, message))
        self.assertNotEqual(retried.returncode, 0)
        self.assertEqual(sum(kind == "PROMPT_BODY" for kind, _, _ in self.fixture.calls), prompt_count)

        self.fixture.disconnect_prompt = False
        self.fixture.calls.clear()
        self.fixture.mutations.clear()
        redirect_receipt = self.root / "redirect.json"
        self.fixture.receipt_path = redirect_receipt
        self.fixture.redirect_on_get_state = True
        redirected = self.run_cli(self.pair_args(redirect_receipt, message))
        self.assertNotEqual(redirected.returncode, 0)
        self.assertEqual(self.fixture.redirect_hits, 0)
        self.assertFalse(any(kind == "PROMPT_BODY" for kind, _, _ in self.fixture.calls))
        self.assertEqual(json.loads(redirect_receipt.read_text())["status"], "prepared")

    def test_peer_route_targets_one_independent_session_and_references_report_without_reading_it(self) -> None:
        self.fixture.sessions[WORKER]["cwd"] = str(self.worker_root)
        sender_cwd_link = self.root / "sender-cwd-link"
        sender_cwd_link.symlink_to(self.worker_root, target_is_directory=True)
        secret_report = self.peer_root / "private-report.txt"
        secret_report.write_text("REPORT_SECRET_MUST_NOT_BE_SENT", encoding="utf-8")
        receipt = self.root / "peer-route.json"
        self.fixture.receipt_path = receipt
        reply_to = "00000000-0000-4000-8000-000000000099"
        args = [
            "--base-url", self.base,
            "--sender", WORKER, "--sender-cwd", str(sender_cwd_link),
            "--recipient", PEER, "--recipient-cwd", str(self.peer_root),
            "--send", "--text", "Quick update: see the current report pointer.",
            "--reply-to", reply_to,
            "--report-path", str(secret_report), "--report-section", "Results",
            "--revision", "rev-2", "--supersedes", "table only; source rows remain",
            "--receipt", str(receipt),
        ]
        result = self.run_cli(args, cwd=self.peer_root)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(output["sender"]["id"], WORKER)
        self.assertEqual(output["recipient"]["id"], PEER)
        self.assertEqual(saved["sender_session_id"], WORKER)
        self.assertEqual(saved["recipient_session_id"], PEER)
        self.assertEqual(saved["sender_cwd"], str(self.worker_root.resolve()))
        self.assertEqual(saved["recipient_cwd"], str(self.peer_root.resolve()))
        self.assertEqual(saved["reply_to_dispatch_id"], reply_to)
        self.assertEqual(saved["delivery"], "queue")
        self.assertEqual(saved["report_reference"], {
            "path": str(secret_report), "section": "Results", "revision": "rev-2",
            "supersedes": "table only; source rows remain",
        })
        first_mutation = self.fixture.mutations[0]
        self.assertEqual(first_mutation["body"]["type"], "get_state")
        self.assertEqual(first_mutation["receipt"]["phase"], "load_target_state")
        prompt_mutations = [m for m in self.fixture.mutations if m["body"].get("type") == "prompt"]
        self.assertEqual(len(prompt_mutations), 1)
        self.assertEqual(prompt_mutations[0]["path"], f"/api/agent/{PEER}")
        prompt = prompt_mutations[0]["body"]["message"]
        self.assertIn(f"sender_cwd={json.dumps(str(self.worker_root.resolve()))}\n", prompt)
        self.assertIn(f"recipient_cwd={json.dumps(str(self.peer_root.resolve()))}\n", prompt)
        self.assertIn("Quick update: see the current report pointer.", prompt)
        self.assertIn(str(secret_report), prompt)
        self.assertNotIn("REPORT_SECRET_MUST_NOT_BE_SENT", prompt)
        self.assertEqual(saved["message_sha256"], hashlib.sha256(
            b"Quick update: see the current report pointer.").hexdigest())
        self.assertNotEqual(saved["sent_sha256"], saved["message_sha256"])
        self.assertIn("not a user instruction, new authorization", prompt)
        self.assertEqual(prompt_mutations[0]["body"]["streamingBehavior"], "followUp")

        missing_pointer = self.peer_root / "not-created-report.md"
        missing_receipt = self.root / "missing-pointer.json"
        missing = self.run_cli([
            "--base-url", self.base,
            "--sender", WORKER, "--sender-cwd", str(self.worker_root),
            "--recipient", PEER, "--recipient-cwd", str(self.peer_root),
            "--send", "--text", "The report is not present locally yet.",
            "--report-path", str(missing_pointer), "--revision", "draft-1",
            "--delivery", "steering", "--receipt", str(missing_receipt),
        ])
        self.assertEqual(missing.returncode, 0, missing.stderr)
        self.assertEqual(json.loads(missing_receipt.read_text())["report_reference"], {
            "path": str(missing_pointer), "revision": "draft-1",
        })
        self.assertEqual(json.loads(missing.stdout)["delivery"], "steering")
        steering_prompt = [body for kind, _, body in self.fixture.calls if kind == "PROMPT_BODY"][-1]
        self.assertEqual(steering_prompt["streamingBehavior"], "steer")
        self.assertFalse(missing_pointer.exists())

    def test_alias_route_is_read_only_and_invalid_peer_routes_make_no_api_mutation(self) -> None:
        self.fixture.sessions[WORKER]["cwd"] = str(self.worker_root)
        state_path = self.root / "task-state.json"
        state = {
            "science": {"metric": "uninterpreted", "private": "never print"},
            "pi_peer_transport": {
                "version": 1,
                "participants": {
                    "gradient-worker": {"session_id": WORKER, "cwd": str(self.worker_root)},
                    "runtime-worker": {"session_id": PEER, "cwd": str(self.peer_root)},
                },
            },
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        original_state = state_path.read_bytes()
        inspected = self.run_cli([
            "--base-url", self.base, "--participants-state", str(state_path),
            "--sender", "gradient-worker", "--recipient", "runtime-worker",
        ])
        self.assertEqual(inspected.returncode, 0, inspected.stderr)
        self.assertEqual(json.loads(inspected.stdout)["status"], "inspected_only")
        self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))
        self.assertEqual(state_path.read_bytes(), original_state)
        self.fixture.calls.clear()
        self.fixture.mutations.clear()

        receipt = self.root / "alias-route.json"
        self.fixture.receipt_path = receipt
        sent = self.run_cli([
            "--base-url", self.base, "--participants-state", str(state_path),
            "--sender", "gradient-worker", "--recipient", "runtime-worker",
            "--send", "--text", "Alias route check.", "--receipt", str(receipt),
        ])
        self.assertEqual(sent.returncode, 0, sent.stderr)
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(saved["sender_alias"], "gradient-worker")
        self.assertEqual(saved["recipient_alias"], "runtime-worker")
        self.assertEqual(state_path.read_bytes(), original_state)
        self.assertNotIn("uninterpreted", sent.stdout)
        self.assertNotIn("never print", sent.stdout)
        self.assertEqual(sum(m["body"].get("type") == "prompt" for m in self.fixture.mutations), 1)

        invalid_state_documents = [
            json.dumps({"pi_peer_transport": {"version": 2, "participants": {}}}).encode(),
            json.dumps({"pi_peer_transport": {"version": 1, "participants": {
                "gradient-worker": {"session_id": WORKER, "cwd": str(self.worker_root)},
                "runtime-worker": {"session_id": WORKER, "cwd": str(self.peer_root)},
            }}}).encode(),
            json.dumps({"pi_peer_transport": {"version": 1, "participants": {}, "extra": True}}).encode(),
            b'{"pi_peer_transport":{"version":1,"participants":{},"version":1}}',
        ]
        for raw_state in invalid_state_documents:
            state_path.write_bytes(raw_state)
            before_failure = state_path.read_bytes()
            self.fixture.calls.clear()
            self.fixture.mutations.clear()
            invalid_alias = self.run_cli([
                "--base-url", self.base, "--participants-state", str(state_path),
                "--sender", "gradient-worker", "--recipient", "runtime-worker",
            ])
            self.assertNotEqual(invalid_alias.returncode, 0)
            self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))
            self.assertEqual(state_path.read_bytes(), before_failure)
        state_path.write_bytes(original_state)

        invalid_routes = [
            ["--sender", WORKER, "--sender-cwd", str(self.worker_root),
             "--recipient", WORKER, "--recipient-cwd", str(self.worker_root)],
            ["--sender", WORKER, "--sender-cwd", str(self.worker_root),
             "--recipient", PEER, "--recipient", LEAD, "--recipient-cwd", str(self.peer_root)],
            ["--sender", WORKER, "--recipient", PEER, "--participants-state", str(state_path),
             "--sender-cwd", str(self.worker_root), "--recipient-cwd", str(self.peer_root)],
            ["--sender", WORKER, "--sender-cwd", str(self.worker_root),
             "--recipient", PEER, "--recipient-cwd", str(self.peer_root),
             "--lead-session", LEAD, "--worker-session", WORKER],
            ["--sender", WORKER, "--sender-cwd", str(self.worker_root),
             "--recipient", PEER, "--recipient-cwd", str(self.peer_root / "missing")],
        ]
        for index, route_args in enumerate(invalid_routes):
            with self.subTest(route=route_args):
                self.fixture.calls.clear()
                self.fixture.mutations.clear()
                invalid = self.run_cli(["--base-url", self.base, *route_args])
                self.assertNotEqual(invalid.returncode, 0)
                self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))

    def test_peer_route_settings_drift_and_lost_prompt_are_not_retried(self) -> None:
        self.fixture.sessions[WORKER]["cwd"] = str(self.worker_root)
        route = [
            "--base-url", self.base,
            "--sender", WORKER, "--sender-cwd", str(self.worker_root),
            "--recipient", PEER, "--recipient-cwd", str(self.peer_root),
            "--send", "--text", "Do not duplicate this message.",
        ]
        self.fixture.sessions[PEER]["loaded"] = False
        self.fixture.sessions[PEER]["runtime_override"] = {
            "model": {"id": "changed-after-inspection", "provider": "openai"},
        }
        drift_receipt = self.root / "peer-drift.json"
        self.fixture.receipt_path = drift_receipt
        drift = self.run_cli(route + ["--receipt", str(drift_receipt)])
        self.assertNotEqual(drift.returncode, 0)
        self.assertEqual(json.loads(drift_receipt.read_text())["phase"], "load_target_state")
        self.assertFalse(any(body and body.get("type") == "prompt"
                             for _, _, body in self.fixture.calls))
        self.assertIsNotNone(self.fixture.mutations[0]["receipt"])

        self.fixture.calls.clear()
        self.fixture.mutations.clear()
        self.fixture.sessions[PEER]["runtime_override"] = {}
        self.fixture.sessions[PEER]["loaded"] = False
        self.fixture.disconnect_prompt = True
        receipt = self.root / "peer-unknown.json"
        self.fixture.receipt_path = receipt
        sent = self.run_cli(route + ["--receipt", str(receipt)])
        self.assertNotEqual(sent.returncode, 0)
        self.assertEqual(json.loads(receipt.read_text())["status"], "delivery_unknown")
        self.assertEqual(sum(kind == "PROMPT_BODY" for kind, _, _ in self.fixture.calls), 1)
        prompt_attempts = sum(kind == "PROMPT_BODY" for kind, _, _ in self.fixture.calls)
        reused = self.run_cli(route + ["--receipt", str(receipt)])
        self.assertNotEqual(reused.returncode, 0)
        self.assertEqual(sum(kind == "PROMPT_BODY" for kind, _, _ in self.fixture.calls),
                         prompt_attempts)
        self.assertEqual(sum(method == "POST" for method, _, _ in self.fixture.calls), 2)

    def test_runpy_public_entrypoint_loads_args_from_an_unrelated_cwd(self) -> None:
        code = "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')"
        result = self.run_command([
            sys.executable, "-c", code, str(SCRIPT), "--base-url", self.base,
            "--sender", LEAD, "--sender-cwd", str(self.root),
            "--recipient", PEER, "--recipient-cwd", str(self.peer_root),
        ], cwd=self.peer_root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "inspected_only")
        self.assertFalse(any(method == "POST" for method, _, _ in self.fixture.calls))

    def test_creation_http_failure_after_side_effect_stays_unknown_and_arguments_fail_closed(self) -> None:
        message = self.message_file()
        receipt = self.root / "http-error-create.json"
        self.fixture.receipt_path = receipt
        self.fixture.http_error_after_create = True
        result = self.run_cli(self.create_args(receipt, message))
        self.assertNotEqual(result.returncode, 0)
        saved = json.loads(receipt.read_text())
        self.assertEqual(saved["status"], "creation_unknown")
        self.assertIsNone(saved["worker_session_id"])
        self.assertEqual(sum(method == "POST" and path == "/api/agent/new"
                             for method, path, _ in self.fixture.calls), 1)

        for base in ("http://example.com:30141", "https://127.0.0.1:30141",
                     "http://user:secret@127.0.0.1:30141"):
            with self.subTest(base=base):
                invalid = self.run_cli(["--base-url", base, "--lead-session", LEAD,
                                        "--worker-session", WORKER, "--cwd", str(self.root)])
                self.assertEqual(invalid.returncode, 2)

        same = self.run_cli(["--base-url", self.base, "--lead-session", LEAD,
                             "--worker-session", LEAD, "--cwd", str(self.root)])
        self.assertEqual(same.returncode, 2)


if __name__ == "__main__":
    unittest.main()
