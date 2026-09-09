#!/usr/bin/env python3
"""Exercise real native-child completion using an entirely local Responses fixture."""

import argparse
import asyncio
import http.server
import json
from pathlib import Path
import threading
import time

import psutil

import idle_mcp_acceptance as h


class SpawnFixture:
    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()
        self.server = None
        self.thread = None

    def start(self):
        fixture = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                with fixture.lock:
                    fixture.requests.append(request)
                    number = len(fixture.requests)
                if number == 1:
                    item = {
                        "type": "function_call",
                        "id": "spawn-item",
                        "call_id": "spawn-call",
                        "name": "spawn_agent",
                        "namespace": "collaboration",
                        "arguments": json.dumps(
                            {
                                "task_name": "idle_child",
                                "fork_turns": "none",
                                "message": "Return the fixed local fixture completion.",
                            }
                        ),
                    }
                elif not any(
                    t.get("name") == "collaboration" for t in request.get("tools", [])
                ) and not any(
                    x.get("call_id") == "child-identity"
                    for x in request.get("input", [])
                ):
                    item = {
                        "type": "function_call",
                        "id": "child-identity-item",
                        "call_id": "child-identity",
                        "name": "identity",
                        "namespace": "mcp__eligible",
                        "arguments": "{}",
                    }
                else:
                    item = {
                        "id": f"fixture-message-{number}",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "local child fixture complete",
                            }
                        ],
                    }
                events = [
                    {
                        "type": "response.created",
                        "response": {"id": f"fixture-{number}"},
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": f"fixture-{number}",
                            "status": "completed",
                            "output": [item],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    },
                ]
                body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.server.server_port

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


async def run(binary, root):
    root.mkdir(parents=True)
    report = {
        "name": "native_child_idle",
        "binary": h._binary_info(binary),
        "network": "local Responses fixture only",
        "assertions": {},
    }
    main_idle, child_idle = 20.0, 2.0
    spec = h._fixture_spec(
        "eligible", root, idle_timeout=main_idle, child_idle_timeout=child_idle
    )
    fixture = SpawnFixture()
    port = fixture.start()
    app = None
    try:
        workspace, home = root / "workspace", root / "core-home"
        h._write_workspace(workspace)
        config = h._render_config(home, [spec], responses_port=port)
        config.write_text(
            config.read_text().replace(
                "[features]\n",
                "[features]\nmulti_agent = true\nmulti_agent_v2 = true\n",
            )
        )
        app = h.AppServer(binary, home, workspace, root / "core.stderr.log")
        await app.start()
        await app.rpc.request(
            "initialize",
            {
                "clientInfo": {"name": "native-child-idle-acceptance", "version": "1"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await app.rpc.notify("initialized")
        started = h._response_result(
            await app.rpc.request(
                "thread/start",
                {
                    "cwd": str(workspace),
                    "ephemeral": False,
                    "model": "probe-model",
                    "modelProvider": "probe",
                    "approvalPolicy": "never",
                },
            )
        )
        parent = started["thread"]["id"]
        h._response_result(await h._call_tool(app, parent, "eligible", "identity", {}))
        parent_identity = h._latest_identity(spec)
        h._response_result(
            await app.rpc.request(
                "turn/start",
                {
                    "threadId": parent,
                    "input": [
                        {"type": "text", "text": "spawn one local fixture child"}
                    ],
                },
            )
        )
        await app.rpc.wait_notification(
            "turn/completed",
            predicate=lambda msg: msg.get("params", {}).get("threadId") == parent,
        )
        listed = h._response_result(await app.rpc.request("thread/loaded/list", {}))
        children = [item for item in listed.get("data", []) if item != parent]
        report["child_list"] = children
        h._record_check(report, "one_native_child_created", len(children) == 1, listed)
        if len(children) != 1:
            raise RuntimeError("Expected exactly one actual native child")
        child = children[0]
        # The real child executes the MCP call through its model/tool route;
        # v2 intentionally rejects direct app-server input to children.
        deadline = time.monotonic() + 10
        while True:
            read = h._response_result(
                await app.rpc.request(
                    "thread/read",
                    {
                        "threadId": child,
                        "includeTurns": True,
                    },
                )
            )
            if "local child fixture complete" in json.dumps(read):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Child did not complete its local fixture turn")
            await asyncio.sleep(0.1)
        items = [item for turn in read["thread"]["turns"] for item in turn["items"]]
        call = next(item for item in items if item.get("id") == "child-identity")
        call_text = "\n".join(c.get("text", "") for c in call["result"]["content"])
        _, child_pid = h._fixture_identity_from_text(call_text)
        child_identity = h._process_identity(psutil.Process(child_pid))
        h._record_check(
            report,
            "bounded_servers_and_calls",
            len(h._start_events(spec)) == 2 and len(h._call_events(spec)) == 2,
        )
        h._record_check(
            report,
            "distinct_child_server",
            child_identity != parent_identity,
            {"parent": parent_identity, "child": child_identity},
        )
        report.update(
            parent_id=parent,
            child_id=child,
            parent_process=parent_identity,
            child_process=child_identity,
            intervals=[main_idle, child_idle],
        )
        dead, elapsed = await h._wait_until(
            lambda: not h._same_process(child_identity), 10
        )
        h._record_check(
            report,
            "completed_child_uses_short_grace",
            dead and elapsed < main_idle,
            {"elapsed": elapsed},
        )
        h._record_check(
            report,
            "parent_retained_during_child_grace",
            h._same_process(parent_identity),
        )
        read = h._response_result(
            await app.rpc.request(
                "thread/read",
                {
                    "threadId": child,
                    "includeTurns": True,
                },
            )
        )
        h._record_check(
            report,
            "completed_child_history_retained",
            "local child fixture complete" in json.dumps(read),
            read,
        )
    except Exception as error:
        h._record_check(report, "scenario_execution", False, repr(error))
    finally:
        await h._finish_scenario(report, app, fixture)
        (root / "requests.json").write_text(json.dumps(fixture.requests, indent=2))
    (root / "receipt.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {"status": report.get("status"), "receipt": str(root / "receipt.json")}
        )
    )
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir / time.strftime("native-child-%Y%m%dT%H%M%SZ")
    raise SystemExit(asyncio.run(run(args.binary.resolve(), root)))
