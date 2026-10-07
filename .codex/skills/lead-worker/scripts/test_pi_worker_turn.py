"""CPU-only tests for Pi Web peer-session transport using its real HTTP wire shape."""

from __future__ import annotations

import json
import os
import runpy
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


SCRIPT = Path(__file__).with_name("pi_worker_turn.py")
LEAD = "00000000-0000-4000-8000-000000000001"
WORKER = "00000000-0000-4000-8000-000000000002"
CREATED = "00000000-0000-4000-8000-000000000003"


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
    def test_default_endpoint_and_delivery_arguments(self) -> None:
        parse = runpy.run_path(str(SCRIPT))["parse_args"]
        args = parse(["--cwd", str(Path.cwd()), "--list-models"])
        self.assertEqual(args.base_url, "http://127.0.0.1:12345")
        with self.assertRaises(SystemExit):
            parse(["--cwd", str(Path.cwd()), "--list-models", "--delivery", "steering"])

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = PiWebFixture(self.root)
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

    def run_cli(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        # The CLI must use the explicit local endpoint even when ambient proxy variables exist.
        env.update({"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1",
                    "HTTPS_PROXY": "http://127.0.0.1:1", "https_proxy": "http://127.0.0.1:1",
                    "NO_PROXY": "", "no_proxy": ""})
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                              text=True, env=env, timeout=10)

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

    def test_list_models_needs_no_session_and_never_mutates(self) -> None:
        result = self.run_cli(["--base-url", self.base, "--cwd", str(self.root), "--list-models"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["models"], [
            {"provider": "openai", "model": "gpt-6-sol", "efforts": ["off", "high"]},
        ])
        self.assertEqual(len(self.fixture.calls), 1)
        self.assertEqual(self.fixture.calls[0][0], "GET")

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
