from __future__ import annotations

import csv
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from src.analysis.manual_audit_reviewer import (
    ManualAuditSession,
    build_index_html,
    make_reviewer_handler,
)
from scripts.analysis.run_manual_audit_reviewer import _resolve_audit_csv


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def test_manual_audit_session_roundtrip(tmp_path: Path) -> None:
    overlay = tmp_path / "overlays" / "sample.png"
    crop = tmp_path / "crops" / "sample.png"
    _write_png(overlay)
    _write_png(crop)
    csv_path = tmp_path / "manual_audit_recommended96.csv"
    csv_path.write_text(
        "\n".join(
            [
                "package_audit_id,proposal_uid,desc,package_overlay_path,package_crop_path,audit_label,audit_notes",
                f"A000,proposal-0000,bus,{overlay},{crop},,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    session = ManualAuditSession.from_csv(csv_path)
    rows = session.export_rows()
    assert len(rows) == 1
    assert rows[0]["audit_id"] == "proposal-0000"
    assert rows[0]["display_id"] == "A000"
    assert rows[0]["overlay_url"].endswith("/proposal-0000")
    assert rows[0]["crop_url"].endswith("/proposal-0000")

    session.update_label(
        audit_id="proposal-0000",
        audit_label="real_visible_object",
        audit_notes="clear foreground bus",
    )

    updated_rows = list(csv.DictReader(csv_path.open()))
    assert updated_rows[0]["audit_label"] == "real_visible_object"
    assert updated_rows[0]["audit_notes"] == "clear foreground bus"

    labels_path = tmp_path / "manual_audit_labels.jsonl"
    label_rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
    assert label_rows == [
        {
            "audit_id": "proposal-0000",
            "audit_label": "real_visible_object",
            "audit_notes": "clear foreground bus",
        }
    ]


def test_manual_audit_session_loads_existing_labels(tmp_path: Path) -> None:
    overlay = tmp_path / "overlays" / "sample.png"
    crop = tmp_path / "crops" / "sample.png"
    _write_png(overlay)
    _write_png(crop)
    csv_path = tmp_path / "manual_audit_recommended96.csv"
    csv_path.write_text(
        "\n".join(
            [
                "proposal_uid,desc,package_overlay_path,package_crop_path,audit_label,audit_notes",
                f"proposal-0001,cup,{overlay},{crop},,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    labels_path = tmp_path / "manual_audit_labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "audit_id": "proposal-0001",
                "audit_label": "wrong_location",
                "audit_notes": "same desc, wrong region",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    session = ManualAuditSession.from_csv(csv_path, labels_jsonl_path=labels_path)
    rows = session.export_rows()
    assert rows[0]["audit_label"] == "wrong_location"
    assert rows[0]["audit_notes"] == "same desc, wrong region"


def test_build_index_html_mentions_shortcuts_and_labels() -> None:
    page = build_index_html()
    assert "Manual Audit Reviewer" in page
    assert "real_visible_object" in page
    assert "Next Unlabeled U" in page
    assert "addEventListener('blur'" in page
    assert "addEventListener('input'" in page
    assert "beforeunload" in page


def test_resolve_audit_csv_prefers_recommended_pack(tmp_path: Path) -> None:
    (tmp_path / "manual_audit_priority48.csv").write_text("a\n", encoding="utf-8")
    expected = tmp_path / "manual_audit_recommended96.csv"
    expected.write_text("a\n", encoding="utf-8")
    assert _resolve_audit_csv(tmp_path) == expected


def test_reviewer_http_handler_serves_state_and_html(tmp_path: Path) -> None:
    overlay = tmp_path / "overlays" / "sample.png"
    crop = tmp_path / "crops" / "sample.png"
    _write_png(overlay)
    _write_png(crop)
    csv_path = tmp_path / "manual_audit_recommended96.csv"
    csv_path.write_text(
        "\n".join(
            [
                "package_audit_id,proposal_uid,desc,package_overlay_path,package_crop_path,audit_label,audit_notes",
                f"A000,proposal-0000,bus,{overlay},{crop},,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    session = ManualAuditSession.from_csv(csv_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_reviewer_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base_url = f"http://127.0.0.1:{server.server_port}"
        payload = json.loads(opener.open(f"{base_url}/api/state").read().decode("utf-8"))
        assert payload["progress"]["total"] == 1
        assert payload["rows"][0]["display_id"] == "A000"
        html = opener.open(f"{base_url}/").read().decode("utf-8")
        assert "Manual Audit Reviewer" in html
    finally:
        server.shutdown()
        server.server_close()
