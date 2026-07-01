from __future__ import annotations

import csv
import json
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse


_LABEL_OPTIONS = (
    "real_visible_object",
    "duplicate_like",
    "wrong_location",
    "dead_or_hallucinated",
    "uncertain",
)
_VALID_LABELS = frozenset(_LABEL_OPTIONS)
_IMAGE_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _load_label_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    loaded: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            audit_id = _coerce_optional_text(row.get("audit_id"))
            if not audit_id:
                continue
            loaded[audit_id] = dict(row)
    return loaded


def _read_csv_rows(path: Path) -> tuple[List[str], List[Dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _write_csv_rows(
    path: Path,
    *,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    resolved_fieldnames = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in resolved_fieldnames})


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _coerce_optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_asset_path(csv_path: Path, row: Mapping[str, Any], key: str) -> Path:
    explicit_candidates = []
    for field in (
        f"{key}_abs_path",
        f"package_{key}_path",
        key,
    ):
        raw = _coerce_optional_text(row.get(field))
        if raw:
            explicit_candidates.append(Path(raw))
    raw_relative = _coerce_optional_text(row.get(f"{key}_path"))
    relative_candidates = []
    if raw_relative:
        relative_candidates.extend(
            [
                csv_path.parent / raw_relative,
                csv_path.parent / "manual_audit" / raw_relative,
                csv_path.parent / "audit_pack" / raw_relative,
            ]
        )
    for candidate in explicit_candidates + relative_candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve {key!r} path for audit row "
        f"{row.get('proposal_uid') or row.get('package_audit_id') or row.get('audit_id')}"
    )


def _coerce_label(value: Any) -> Optional[str]:
    label = _coerce_optional_text(value)
    if not label:
        return None
    if label not in _VALID_LABELS:
        raise ValueError(f"Unknown audit label: {label}")
    return label


@dataclass
class ManualAuditSession:
    csv_path: Path
    labels_jsonl_path: Path
    fieldnames: List[str]
    rows: List[Dict[str, Any]]
    rows_by_id: Dict[str, Dict[str, Any]]
    overlay_paths: Dict[str, Path]
    crop_paths: Dict[str, Path]
    lock: threading.Lock

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        *,
        labels_jsonl_path: Optional[Path] = None,
    ) -> "ManualAuditSession":
        fieldnames, raw_rows = _read_csv_rows(csv_path)
        if "audit_id" not in fieldnames:
            fieldnames.append("audit_id")
        if "display_id" not in fieldnames:
            fieldnames.append("display_id")
        if "audit_label" not in fieldnames:
            fieldnames.append("audit_label")
        if "audit_notes" not in fieldnames:
            fieldnames.append("audit_notes")
        rows: List[Dict[str, Any]] = []
        rows_by_id: Dict[str, Dict[str, Any]] = {}
        overlay_paths: Dict[str, Path] = {}
        crop_paths: Dict[str, Path] = {}
        for index, row in enumerate(raw_rows):
            audit_id = (
                _coerce_optional_text(row.get("audit_id"))
                or _coerce_optional_text(row.get("proposal_uid"))
                or _coerce_optional_text(row.get("package_audit_id"))
                or f"row-{index:04d}"
            )
            display_id = (
                _coerce_optional_text(row.get("package_audit_id"))
                or _coerce_optional_text(row.get("audit_id"))
                or _coerce_optional_text(row.get("proposal_uid"))
                or audit_id
            )
            normalized = dict(row)
            normalized["audit_id"] = audit_id
            normalized["display_id"] = display_id
            normalized["audit_label"] = _coerce_optional_text(row.get("audit_label"))
            normalized["audit_notes"] = _coerce_optional_text(row.get("audit_notes"))
            overlay_paths[audit_id] = _resolve_asset_path(csv_path, row, "overlay")
            crop_paths[audit_id] = _resolve_asset_path(csv_path, row, "crop")
            rows.append(normalized)
            rows_by_id[audit_id] = normalized
        labels_path = labels_jsonl_path or (csv_path.parent / "manual_audit_labels.jsonl")
        session = cls(
            csv_path=csv_path,
            labels_jsonl_path=labels_path,
            fieldnames=fieldnames,
            rows=rows,
            rows_by_id=rows_by_id,
            overlay_paths=overlay_paths,
            crop_paths=crop_paths,
            lock=threading.Lock(),
        )
        existing_labels = _load_label_rows(labels_path)
        for audit_id, label_row in existing_labels.items():
            row = session.rows_by_id.get(audit_id)
            if row is None:
                continue
            row["audit_label"] = _coerce_optional_text(label_row.get("audit_label"))
            row["audit_notes"] = _coerce_optional_text(label_row.get("audit_notes"))
        session.persist()
        return session

    def export_rows(self) -> List[Dict[str, Any]]:
        exported: List[Dict[str, Any]] = []
        for row in self.rows:
            exported.append(
                {
                    "audit_id": row["audit_id"],
                    "display_id": row["display_id"],
                    "checkpoint": _coerce_optional_text(row.get("checkpoint")),
                    "temperature": _coerce_optional_text(row.get("temperature")),
                    "score_quantile_bucket": _coerce_optional_text(
                        row.get("score_quantile_bucket")
                    ),
                    "nearest_gt_overlap_bucket": _coerce_optional_text(
                        row.get("nearest_gt_overlap_bucket")
                    ),
                    "desc": _coerce_optional_text(row.get("desc")),
                    "nearest_gt_desc": _coerce_optional_text(row.get("nearest_gt_desc")),
                    "nearest_gt_iou": _coerce_optional_text(row.get("nearest_gt_iou")),
                    "commitment": _coerce_optional_text(row.get("commitment")),
                    "counterfactual": _coerce_optional_text(row.get("counterfactual")),
                    "combined_linear": _coerce_optional_text(row.get("combined_linear")),
                    "duplicate_like_any_desc_iou90": _coerce_optional_text(
                        row.get("duplicate_like_any_desc_iou90")
                    ),
                    "image_path": _coerce_optional_text(row.get("image_path")),
                    "proposal_uid": _coerce_optional_text(row.get("proposal_uid")),
                    "pred_count": _coerce_optional_text(row.get("pred_count")),
                    "audit_label": _coerce_optional_text(row.get("audit_label")),
                    "audit_notes": _coerce_optional_text(row.get("audit_notes")),
                    "overlay_url": f"/api/asset/overlay/{row['audit_id']}",
                    "crop_url": f"/api/asset/crop/{row['audit_id']}",
                }
            )
        return exported

    def progress_summary(self) -> Dict[str, Any]:
        label_counts: Dict[str, int] = {label: 0 for label in _LABEL_OPTIONS}
        labeled = 0
        for row in self.rows:
            label = _coerce_optional_text(row.get("audit_label"))
            if not label:
                continue
            labeled += 1
            label_counts[label] = label_counts.get(label, 0) + 1
        return {
            "total": len(self.rows),
            "labeled": labeled,
            "unlabeled": len(self.rows) - labeled,
            "label_counts": label_counts,
        }

    def persist(self) -> None:
        csv_rows = []
        for row in self.rows:
            row_out = {
                fieldname: _coerce_optional_text(row.get(fieldname))
                for fieldname in self.fieldnames
            }
            row_out["audit_label"] = _coerce_optional_text(row.get("audit_label"))
            row_out["audit_notes"] = _coerce_optional_text(row.get("audit_notes"))
            csv_rows.append(row_out)
        _write_csv_rows(self.csv_path, fieldnames=self.fieldnames, rows=csv_rows)
        jsonl_rows = []
        for row in self.rows:
            label = _coerce_optional_text(row.get("audit_label"))
            notes = _coerce_optional_text(row.get("audit_notes"))
            if not label and not notes:
                continue
            jsonl_rows.append(
                {
                    "audit_id": row["audit_id"],
                    "audit_label": label or None,
                    "audit_notes": notes or None,
                }
            )
        _write_jsonl(self.labels_jsonl_path, jsonl_rows)

    def update_label(
        self,
        *,
        audit_id: str,
        audit_label: Optional[str],
        audit_notes: Optional[str],
    ) -> Dict[str, Any]:
        with self.lock:
            if audit_id not in self.rows_by_id:
                raise KeyError(audit_id)
            row = self.rows_by_id[audit_id]
            row["audit_label"] = audit_label or ""
            row["audit_notes"] = _coerce_optional_text(audit_notes)
            self.persist()
            return {
                "ok": True,
                "audit_id": audit_id,
                "progress": self.progress_summary(),
            }

    def asset_path(self, *, audit_id: str, kind: str) -> Path:
        if kind == "overlay":
            return self.overlay_paths[audit_id]
        if kind == "crop":
            return self.crop_paths[audit_id]
        raise KeyError(kind)


def build_index_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Manual Audit Reviewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f6f7f9; color: #111; }
    .topbar { position: sticky; top: 0; background: white; border-bottom: 1px solid #ddd; padding: 12px 16px; z-index: 10; }
    .row { display: flex; gap: 16px; padding: 16px; }
    .panel { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
    .meta { width: 420px; }
    .viewer { flex: 1; min-width: 0; }
    .images { display: grid; grid-template-columns: 1fr 320px; gap: 12px; align-items: start; }
    img { width: 100%; border: 1px solid #ccc; background: #fff; }
    .buttons { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px; }
    button { padding: 8px 10px; border-radius: 6px; border: 1px solid #aaa; background: #fff; cursor: pointer; }
    button.selected { border-color: #0b6; background: #e9fff4; }
    button.nav { min-width: 110px; }
    textarea { width: 100%; min-height: 96px; font: inherit; }
    table { width: 100%; border-collapse: collapse; }
    td { padding: 4px 6px; border-bottom: 1px solid #eee; vertical-align: top; }
    td.key { width: 150px; color: #555; }
    .status { color: #333; font-size: 14px; }
    .hint { color: #666; font-size: 13px; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="topbar">
    <div id="progress" class="status">Loading…</div>
    <div class="buttons">
      <button class="nav" id="prevBtn">Prev [</button>
      <button class="nav" id="nextBtn">Next ]</button>
      <button class="nav" id="nextUnlabeledBtn">Next Unlabeled U</button>
      <button class="nav" id="saveBtn">Save S</button>
      <label><input type="checkbox" id="unlabeledOnly"> show unlabeled only</label>
    </div>
    <div class="hint">Shortcuts: 1-5 assign label, S save, ] next, [ prev, U next unlabeled.</div>
  </div>
  <div class="row">
    <div class="panel meta">
      <h2 id="title">Audit Row</h2>
      <div class="buttons" id="labelButtons"></div>
      <div><strong>Notes</strong></div>
      <textarea id="notesBox" placeholder="Optional audit notes"></textarea>
      <table id="metaTable"></table>
    </div>
    <div class="panel viewer">
      <div class="images">
        <div>
          <div><strong>Overlay</strong></div>
          <img id="overlayImg" alt="overlay">
        </div>
        <div>
          <div><strong>Crop</strong></div>
          <img id="cropImg" alt="crop">
        </div>
      </div>
    </div>
  </div>
  <script>
    const LABELS = ["real_visible_object", "duplicate_like", "wrong_location", "dead_or_hallucinated", "uncertain"];
    let state = { rows: [], filteredIndices: [], currentIndex: 0, dirty: false };

    async function fetchState() {
      const res = await fetch('/api/state');
      const payload = await res.json();
      state.rows = payload.rows;
      state.progress = payload.progress;
      applyFilter();
    }

    function applyFilter() {
      const unlabeledOnly = document.getElementById('unlabeledOnly').checked;
      state.filteredIndices = [];
      for (let i = 0; i < state.rows.length; i += 1) {
        const row = state.rows[i];
        if (!unlabeledOnly || !row.audit_label) {
          state.filteredIndices.push(i);
        }
      }
      if (state.filteredIndices.length === 0) {
        state.currentIndex = 0;
      } else if (state.currentIndex >= state.filteredIndices.length) {
        state.currentIndex = state.filteredIndices.length - 1;
      }
      render();
    }

    function currentRow() {
      if (!state.filteredIndices.length) return null;
      return state.rows[state.filteredIndices[state.currentIndex]];
    }

    function setLabel(label) {
      const row = currentRow();
      if (!row) return;
      row.audit_label = label;
      state.dirty = true;
    }

    function renderProgress() {
      const labeled = state.rows.filter(r => r.audit_label).length;
      const counts = {};
      for (const label of LABELS) counts[label] = 0;
      for (const row of state.rows) {
        if (row.audit_label) counts[row.audit_label] += 1;
      }
      document.getElementById('progress').textContent =
        `Rows: ${state.rows.length} | labeled: ${labeled} | unlabeled: ${state.rows.length - labeled} | ` +
        LABELS.map(label => `${label}=${counts[label]}`).join(' | ');
    }

    function render() {
      renderProgress();
      const row = currentRow();
      if (!row) {
        document.getElementById('title').textContent = 'No rows match the current filter';
        document.getElementById('metaTable').innerHTML = '';
        return;
      }
      document.getElementById('title').textContent =
        `${row.display_id} (${state.currentIndex + 1}/${state.filteredIndices.length})`;
      const labelButtons = document.getElementById('labelButtons');
      labelButtons.innerHTML = '';
      LABELS.forEach((label, idx) => {
        const btn = document.createElement('button');
        btn.textContent = `${idx + 1}. ${label}`;
        btn.className = row.audit_label === label ? 'selected' : '';
        btn.onclick = () => { setLabel(label); saveCurrent(); };
        labelButtons.appendChild(btn);
      });
      const meta = [
        ['checkpoint', row.checkpoint],
        ['temperature', row.temperature],
        ['desc', row.desc],
        ['nearest_gt_desc', row.nearest_gt_desc],
        ['nearest_gt_iou', row.nearest_gt_iou],
        ['quantile', row.score_quantile_bucket],
        ['overlap_bucket', row.nearest_gt_overlap_bucket],
        ['dup_any', row.duplicate_like_any_desc_iou90],
        ['pred_count', row.pred_count],
        ['commitment', row.commitment],
        ['counterfactual', row.counterfactual],
        ['combined', row.combined_linear],
        ['image_path', row.image_path],
        ['proposal_uid', row.proposal_uid],
      ];
      document.getElementById('metaTable').innerHTML = meta.map(
        ([k, v]) => `<tr><td class="key">${k}</td><td>${String(v ?? '')}</td></tr>`
      ).join('');
      document.getElementById('notesBox').value = row.audit_notes || '';
      document.getElementById('overlayImg').src = row.overlay_url;
      document.getElementById('cropImg').src = row.crop_url;
    }

    async function saveCurrent({ advance = false, unlabeled = false } = {}) {
      const row = currentRow();
      if (!row) return;
      row.audit_notes = document.getElementById('notesBox').value;
      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audit_id: row.audit_id,
          audit_label: row.audit_label || null,
          audit_notes: row.audit_notes || null
        })
      });
      if (!res.ok) {
        alert('Save failed');
        return;
      }
      state.dirty = false;
      if (advance) {
        move(unlabeled ? 'next-unlabeled' : 'next');
      } else {
        render();
      }
    }

    async function move(mode) {
      if (!state.filteredIndices.length) return;
      if (state.dirty) {
        await saveCurrent();
      }
      if (mode === 'prev') {
        state.currentIndex = Math.max(0, state.currentIndex - 1);
      } else if (mode === 'next') {
        state.currentIndex = Math.min(state.filteredIndices.length - 1, state.currentIndex + 1);
      } else if (mode === 'next-unlabeled') {
        for (let i = state.currentIndex + 1; i < state.filteredIndices.length; i += 1) {
          const row = state.rows[state.filteredIndices[i]];
          if (!row.audit_label) {
            state.currentIndex = i;
            render();
            return;
          }
        }
      }
      render();
    }

    document.getElementById('prevBtn').onclick = () => move('prev');
    document.getElementById('nextBtn').onclick = () => move('next');
    document.getElementById('nextUnlabeledBtn').onclick = () => move('next-unlabeled');
    document.getElementById('saveBtn').onclick = () => saveCurrent();
    document.getElementById('unlabeledOnly').onchange = () => applyFilter();
    document.getElementById('notesBox').addEventListener('input', (event) => {
      const row = currentRow();
      if (!row) return;
      row.audit_notes = event.target.value;
      state.dirty = true;
    });
    document.getElementById('notesBox').addEventListener('blur', () => {
      if (state.dirty) {
        saveCurrent();
      }
    });
    window.addEventListener('beforeunload', (event) => {
      if (state.dirty) {
        event.preventDefault();
        event.returnValue = '';
      }
    });
    document.addEventListener('keydown', async (event) => {
      if (event.target.tagName === 'TEXTAREA') {
        if (event.key.toLowerCase() === 's' && (event.ctrlKey || event.metaKey)) {
          event.preventDefault();
          await saveCurrent();
        }
        return;
      }
      if (['1','2','3','4','5'].includes(event.key)) {
        setLabel(LABELS[Number(event.key) - 1]);
        await saveCurrent();
      } else if (event.key === ']') {
        await move('next');
      } else if (event.key === '[') {
        await move('prev');
      } else if (event.key.toLowerCase() === 'u') {
        await move('next-unlabeled');
      } else if (event.key.toLowerCase() === 's') {
        await saveCurrent();
      }
    });

    fetchState();
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler, *, body: str, content_type: str = "text/html; charset=utf-8"
) -> None:
    payload = body.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _error_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, message: str) -> None:
    payload = json.dumps({"ok": False, "error": message}, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def make_reviewer_handler(session: ManualAuditSession) -> type[BaseHTTPRequestHandler]:
    class AuditHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/" or path == "/index.html":
                _text_response(self, body=build_index_html())
                return
            if path == "/api/state":
                _json_response(
                    self,
                    {
                        "rows": session.export_rows(),
                        "progress": session.progress_summary(),
                        "label_options": list(_LABEL_OPTIONS),
                        "labels_jsonl_path": str(session.labels_jsonl_path),
                        "audit_csv_path": str(session.csv_path),
                    },
                )
                return
            if path.startswith("/api/asset/"):
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    _error_response(self, HTTPStatus.NOT_FOUND, "unknown asset path")
                    return
                _, _, kind, audit_id = parts
                try:
                    asset_path = session.asset_path(audit_id=audit_id, kind=kind)
                except KeyError:
                    _error_response(self, HTTPStatus.NOT_FOUND, "asset not found")
                    return
                if not asset_path.exists():
                    _error_response(self, HTTPStatus.NOT_FOUND, "asset missing on disk")
                    return
                content_type = _IMAGE_CONTENT_TYPES.get(asset_path.suffix.lower(), "application/octet-stream")
                payload = asset_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            _error_response(self, HTTPStatus.NOT_FOUND, "unknown path")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                _error_response(self, HTTPStatus.NOT_FOUND, "unknown path")
                return
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                payload = json.loads(self.rfile.read(content_length) or b"{}")
            except json.JSONDecodeError:
                _error_response(self, HTTPStatus.BAD_REQUEST, "invalid json body")
                return
            audit_id = _coerce_optional_text(payload.get("audit_id"))
            if not audit_id:
                _error_response(self, HTTPStatus.BAD_REQUEST, "audit_id is required")
                return
            try:
                label = _coerce_label(payload.get("audit_label"))
            except ValueError as exc:
                _error_response(self, HTTPStatus.BAD_REQUEST, str(exc))
                return
            notes = payload.get("audit_notes")
            try:
                result = session.update_label(
                    audit_id=audit_id,
                    audit_label=label,
                    audit_notes=notes,
                )
            except KeyError:
                _error_response(self, HTTPStatus.NOT_FOUND, "unknown audit_id")
                return
            _json_response(self, result)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return AuditHandler


def serve_manual_audit_reviewer(
    *,
    audit_csv: Path,
    labels_jsonl: Optional[Path] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    session = ManualAuditSession.from_csv(audit_csv, labels_jsonl_path=labels_jsonl)
    server = ThreadingHTTPServer((host, port), make_reviewer_handler(session))
    url = f"http://{host}:{port}/"
    print(f"[manual-audit-reviewer] audit csv: {audit_csv}")
    print(f"[manual-audit-reviewer] labels jsonl: {session.labels_jsonl_path}")
    print(f"[manual-audit-reviewer] serving: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[manual-audit-reviewer] stopping")
    finally:
        server.server_close()
