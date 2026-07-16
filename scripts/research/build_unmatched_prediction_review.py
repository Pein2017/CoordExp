#!/usr/bin/env python3
"""Build a self-contained, no-server HTML reviewer for unmatched predictions.

The reviewer is intentionally small and artifact-oriented.  It consumes a
manual-review manifest, an unmatched-prediction JSON Lines queue, and an
accepted/audit ledger JSON Lines file, then embeds the selected images and all
review state in one deterministic HTML file.  The browser never needs access
to the filesystem or a local HTTP server.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import mimetypes
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the repository's normal environment has Pillow.
    Image = None  # type: ignore[assignment,misc]


SCHEMA_VERSION = "coordexp.unmatched_prediction_review.v1"
DEFAULT_ARM = "FULL_BAG_K"
DEFAULT_ONTOLOGY = "COCO-80"
VERDICTS = ("approve", "reject", "unknown")
SEMANTIC_STATUSES = ("exact", "wrong_category", "ambiguous", "not_applicable")
GEOMETRY_STATUSES = ("acceptable", "localization_error", "fragment", "multiple_entities", "ambiguous")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON input does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON Lines input does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON Lines at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value: Any, default: str = "") -> str:
    return default if value is None else str(value).strip()


def _image_key(value: Any) -> str:
    return _text(value)


def _coerce_bbox(value: Any, *, context: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Invalid bbox for {context}: expected four xyxy numbers")
    result: list[float] = []
    for component in value:
        if isinstance(component, bool):
            raise ValueError(f"Invalid bbox for {context}: boolean component")
        try:
            number = float(component)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid bbox for {context}: {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Invalid bbox for {context}: non-finite component")
        result.append(number)
    x1, y1, x2, y2 = result
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox for {context}: expected 0 <= x1 < x2 and 0 <= y1 < y2")
    return result


def _resolve_image_path(raw_path: Any, *, manifest_path: Path) -> Path:
    raw = _text(raw_path)
    if not raw:
        raise ValueError("Selected image has no image_path")
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [manifest_path.parent / path, Path.cwd() / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image path does not exist: {raw_path}")


def _image_data_uri(path: Path) -> str:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _image_dimensions(path: Path) -> tuple[int | None, int | None]:
    if Image is None:
        return None, None
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _manifest_items(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_items = manifest.get("items") or manifest.get("images") or []
    if not isinstance(raw_items, list):
        raise ValueError("Manifest items/images must be a list")
    items: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict) or "image_id" not in item:
            raise ValueError("Every manifest item must be an object with image_id")
        items.append(item)
    return items


def _selected_ids(
    manifest: Mapping[str, Any], image_ids: Sequence[str] | None,
) -> list[str]:
    if image_ids:
        selected = [_image_key(value) for value in image_ids]
    else:
        selected = [_image_key(value) for value in manifest.get("selected_image_ids", [])]
        if not selected:
            selected = [_image_key(item["image_id"]) for item in _manifest_items(manifest)]
    if not selected or any(not value for value in selected):
        raise ValueError("At least one non-empty selected image_id is required")
    # Preserve caller/manifest order, while making duplicate ids harmless.
    return list(dict.fromkeys(selected))


def _accepted_ledger_row(row: Mapping[str, Any]) -> bool:
    values = [
        _text(row.get("final_state")),
        _text(row.get("decision_outcome")),
        _text(row.get("operational_status")),
        _text(row.get("status")),
    ]
    if not any(values):
        return True
    return any(
        value.lower() in {"accepted", "approved", "approve", "confirmed"}
        or value.lower().startswith(("accepted-", "accept-"))
        for value in values
    )


def _manifest_arm_summary(item: Mapping[str, Any], arm: str) -> dict[str, Any]:
    arms = item.get("arms")
    if not isinstance(arms, dict):
        return {}
    summary = arms.get(arm)
    return dict(summary) if isinstance(summary, dict) else {}


def _is_unmatched_prediction(row: Mapping[str, Any]) -> bool:
    """Allow a mixed queue while keeping the canonical queue semantics explicit."""

    if "is_unmatched" in row:
        value = row.get("is_unmatched")
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "matched"}
        return bool(value)
    status = _text(row.get("match_status")).lower()
    return status not in {"matched", "ignored", "accepted"}


def build_review_payload(
    *,
    manifest_path: Path,
    queue_path: Path,
    accepted_ledger_path: Path,
    image_ids: Sequence[str] | None = None,
    arm: str = DEFAULT_ARM,
    review_set_id: str | None = None,
    reviewer: str = "",
    ontology: Any = DEFAULT_ONTOLOGY,
) -> dict[str, Any]:
    """Load and validate inputs, returning the JSON payload embedded in HTML."""

    manifest = _read_json(manifest_path)
    queue = _read_jsonl(queue_path)
    ledger = _read_jsonl(accepted_ledger_path)
    selected = _selected_ids(manifest, image_ids)
    selected_set = set(selected)
    items_by_id = {_image_key(item["image_id"]): item for item in _manifest_items(manifest)}
    missing_manifest = [image_id for image_id in selected if image_id not in items_by_id]
    if missing_manifest:
        raise ValueError(f"Selected image ids missing from manifest: {missing_manifest}")

    image_records: list[dict[str, Any]] = []
    for image_id in selected:
        item = items_by_id[image_id]
        image_path = _resolve_image_path(item.get("image_path") or item.get("image"), manifest_path=manifest_path)
        discovered_width, discovered_height = _image_dimensions(image_path)
        image_records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "image_data_uri": _image_data_uri(image_path),
                "width": item.get("width") or discovered_width,
                "height": item.get("height") or discovered_height,
                "arm_summary": _manifest_arm_summary(item, arm),
            }
        )

    candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(queue):
        image_id = _image_key(row.get("image_id"))
        if image_id not in selected_set or _text(row.get("arm"), arm) != arm or not _is_unmatched_prediction(row):
            continue
        candidate_id = _text(row.get("candidate_id")) or _text(row.get("prediction_id"))
        if not candidate_id:
            candidate_id = f"{image_id}:{arm}:{row.get('prediction_index', row_index)}"
        bbox = row.get("bbox_xyxy", row.get("bbox"))
        candidates.append(
            {
                "candidate_id": candidate_id,
                "image_id": image_id,
                "category": _text(row.get("category") or row.get("normalized_category_name"), "unknown"),
                "bbox_xyxy": _coerce_bbox(bbox, context=f"candidate {candidate_id}"),
                "verdict": _text(row.get("manual_label")) or None,
                "comment": _text(row.get("manual_note")),
                "hidden_provenance": dict(row),
            }
        )
    candidates.sort(key=lambda row: (selected.index(row["image_id"]), row["candidate_id"]))

    accepted_boxes: list[dict[str, Any]] = []
    for row_index, row in enumerate(ledger):
        image_id = _image_key(row.get("image_id"))
        if image_id not in selected_set or not _accepted_ledger_row(row):
            continue
        bbox = row.get("source_canvas_box_xyxy", row.get("bbox_xyxy", row.get("bbox")))
        if bbox is None:
            continue
        object_id = _text(row.get("object_identifier")) or f"ledger:{image_id}:{row_index}"
        accepted_boxes.append(
            {
                "object_id": object_id,
                "image_id": image_id,
                "category": _text(row.get("normalized_category_name") or row.get("category"), "accepted"),
                "bbox_xyxy": _coerce_bbox(bbox, context=f"ledger {object_id}"),
                "hidden_provenance": dict(row),
            }
        )
    accepted_boxes.sort(key=lambda row: (selected.index(row["image_id"]), row["object_id"]))

    resolved_review_set_id = review_set_id or _text(manifest.get("review_set_id")) or manifest_path.parent.name
    source_digests = {
        "manifest_sha256": _sha256(manifest_path),
        "queue_sha256": _sha256(queue_path),
        "accepted_ledger_sha256": _sha256(accepted_ledger_path),
        "manifest_receipt_sha256": _text(manifest.get("receipt_sha256")),
        "manifest_ledger_sha256": _text(manifest.get("ledger_sha256")),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "review_set_id": resolved_review_set_id,
        "arm": arm,
        "reviewer": reviewer,
        "ontology": ontology,
        "selected_image_ids": selected,
        "images": image_records,
        "candidates": candidates,
        "accepted_ledger": accepted_boxes,
        "hidden_provenance": {
            "manifest_path": str(manifest_path.resolve()),
            "queue_path": str(queue_path.resolve()),
            "accepted_ledger_path": str(accepted_ledger_path.resolve()),
            "manifest_schema_version": manifest.get("schema_version"),
            "selection_policy": manifest.get("selection_policy"),
        },
        "source_digests": source_digests,
    }


def _safe_json_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def build_html(payload: Mapping[str, Any]) -> str:
    """Render one deterministic standalone HTML document."""

    embedded = _safe_json_script(payload)
    title = html.escape(_text(payload.get("review_set_id"), "Unmatched prediction review"))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f5f6f8;color:#17202a}} header{{position:sticky;top:0;z-index:4;background:#fff;border-bottom:1px solid #ccd2d9;padding:10px 14px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}} main{{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:14px;padding:14px}} section{{background:#fff;border:1px solid #d7dce2;border-radius:8px;padding:12px}} #stage{{position:relative;max-width:100%;display:inline-block;background:#111;line-height:0}} #stage img{{display:block;max-width:100%;max-height:78vh}} #overlay{{position:absolute;inset:0;width:100%;height:100%}} .candidate{{stroke:#ef3340;stroke-width:4;fill:#ef334033}} .accepted{{stroke:#1d8a4a;stroke-width:2;fill:#1d8a4a18}} .approvedCandidate{{stroke:#3478c8;stroke-width:2;fill:#3478c818}} .candidate.active{{stroke:#ffb000;stroke-width:6;fill:#ffb00033}} button{{padding:7px 12px;border:1px solid #aab2bb;border-radius:5px;background:#fff;cursor:pointer}} button.active{{background:#1f7a4d;color:#fff}} textarea,input,select{{font:inherit;width:100%;box-sizing:border-box;padding:6px}} label{{display:block;margin-top:8px;font-size:13px;color:#4d5965}} #cropCanvas{{max-width:100%;border:1px solid #ccd2d9;background:#fafafa}} .muted{{color:#68737d;font-size:12px}} .rownav{{display:flex;gap:6px;align-items:center}} .badge{{padding:2px 6px;border-radius:10px;background:#e8edf3;font-size:12px}} .gate{{padding:8px;background:#fff8dd;border:1px solid #e4cb65;border-radius:6px;font-size:13px}}
</style></head><body>
<script id="review-payload" type="application/json">{embedded}</script>
<header><strong>Unmatched prediction reviewer</strong><span id="counter" class="badge"></span><span id="cohort" class="muted"></span><div class="rownav"><button id="prev">Prev [</button><button id="next">Next ]</button><button id="consolidate">Consolidation</button><button id="export">Export JSON</button><label style="display:inline-block;width:auto"><input id="import" type="file" accept="application/json,.json" hidden><button id="importButton">Import JSON</button></label></div></header>
<main><section><div id="stage"><img id="image" alt="selected image"><svg id="overlay" aria-label="prediction and accepted boxes"></svg></div><h3>Client-side crop</h3><canvas id="cropCanvas"></canvas></section>
<section><div class="gate">Approve means a real reportable COCO-80 entity, even if phrase or geometry is imperfect. Approved decisions require an entity reference.</div><div id="imageMeta" class="muted"></div><h2 id="category"></h2><div id="bbox" class="muted"></div>
<label>Verdict (keyboard: A approve, R reject, U unknown)<select id="verdict"><option value="">Unreviewed</option><option value="approve">Approve</option><option value="reject">Reject</option><option value="unknown">Unknown</option></select></label>
<label>Entity reference (required for approve)<input id="entityRef" list="entityOptions" placeholder="accepted-ledger ID or human:image:NNNN"><datalist id="entityOptions"></datalist></label><button id="newEntity" type="button">Assign new human entity</button>
<label>Semantic status<select id="semantic"><option value="">Unreviewed</option><option value="exact">Exact</option><option value="wrong_category">Wrong category</option><option value="ambiguous">Ambiguous</option><option value="not_applicable">Not applicable</option></select></label>
<label>Geometry status<select id="geometry"><option value="">Unreviewed</option><option value="acceptable">Acceptable</option><option value="localization_error">Localization error</option><option value="fragment">Fragment</option><option value="multiple_entities">Multiple entities</option><option value="ambiguous">Ambiguous</option></select></label>
<label>Comment<textarea id="comment" placeholder="Optional review note"></textarea></label><div class="rownav"><button data-v="approve">Approve A</button><button data-v="reject">Reject R</button><button data-v="unknown">Unknown U</button></div><p id="status" class="muted"></p><h3>Accepted ledger boxes</h3><div id="ledgerList" class="muted"></div></section></main>
<script>
(() => {{
  const P = JSON.parse(document.getElementById('review-payload').textContent);
  const candidates = P.candidates || [], images = P.images || [], ledger = P.accepted_ledger || [];
  const imageMap = Object.fromEntries(images.map(x => [String(x.image_id), x]));
  const key = 'coordexp.unmatched-review.' + P.review_set_id + '.' + P.arm;
  let reviews = {{}}; let index = 0; let consolidationMode = false;
  try {{ reviews = JSON.parse(localStorage.getItem(key) || '{{}}'); }} catch (e) {{ reviews = {{}}; }}
  const el = id => document.getElementById(id);
  const visibleCandidates = () => consolidationMode ? candidates.filter(c => reviews[c.candidate_id] && reviews[c.candidate_id].verdict === 'approve') : candidates;
  const current = () => visibleCandidates()[index];
  function save() {{ localStorage.setItem(key, JSON.stringify(reviews)); }}
  (ledger || []).forEach(x => {{ if (x.object_id) {{ const option=document.createElement('option'); option.value=x.object_id; el('entityOptions').appendChild(option); }} }});
  function countText() {{ const visible = visibleCandidates(); const done = candidates.filter(c => reviews[c.candidate_id] && reviews[c.candidate_id].verdict).length; return `${{visible.length ? index + 1 : 0}} / ${{visible.length || 0}} · reviewed ${{done}}${{consolidationMode ? ' · consolidation' : ''}}`; }}
  function updateForm() {{
    const c = current(); if (!c) {{ el('counter').textContent = `0 / 0`; el('category').textContent = 'No unmatched candidates'; return; }}
    const r = reviews[c.candidate_id] || {{}}; const im = imageMap[c.image_id];
    el('counter').textContent = countText(); el('cohort').textContent = `${{P.selected_image_ids.length}} selected images · ${{images.filter(x => !(candidates.some(y => y.image_id === x.image_id))).length}} zero-candidate image(s)`;
    el('imageMeta').textContent = `Image ${{c.image_id}} · candidate ${{index + 1}} of ${{candidates.length}}`;
    el('category').textContent = c.category; el('bbox').textContent = `bbox: ${{c.bbox_xyxy.map(v => Number(v).toFixed(1)).join(', ')}}`;
    el('verdict').value = r.verdict || ''; el('entityRef').value = r.entity_ref || ''; el('semantic').value = r.semantic_status || ''; el('geometry').value = r.geometry_status || ''; el('comment').value = r.comment || '';
    el('image').src = im.image_data_uri; el('image').dataset.imageId = c.image_id;
    const [x1,y1,x2,y2] = c.bbox_xyxy; const width = im.width || x2 || 1; const height = im.height || y2 || 1; const svg = el('overlay'); svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`); const approvedForImage = consolidationMode ? candidates.filter(x => x.image_id === c.image_id && reviews[x.candidate_id] && reviews[x.candidate_id].verdict === 'approve') : []; svg.innerHTML = ledger.filter(x => x.image_id === c.image_id).map(x => `<rect class="accepted" x="${{x.bbox_xyxy[0]}}" y="${{x.bbox_xyxy[1]}}" width="${{x.bbox_xyxy[2]-x.bbox_xyxy[0]}}" height="${{x.bbox_xyxy[3]-x.bbox_xyxy[1]}}"><title>${{escapeHtml(x.category)}}</title></rect>`).join('') + approvedForImage.map(x => `<rect class="approvedCandidate" x="${{x.bbox_xyxy[0]}}" y="${{x.bbox_xyxy[1]}}" width="${{x.bbox_xyxy[2]-x.bbox_xyxy[0]}}" height="${{x.bbox_xyxy[3]-x.bbox_xyxy[1]}}"><title>${{escapeHtml(x.category)}} approved</title></rect>`).join('') + `<rect class="candidate active" x="${{x1}}" y="${{y1}}" width="${{x2-x1}}" height="${{y2-y1}}"><title>${{escapeHtml(c.category)}}</title></rect>`;
    el('ledgerList').textContent = ledger.filter(x => x.image_id === c.image_id).map(x => `${{x.category}} [${{x.bbox_xyxy.map(v => Number(v).toFixed(0)).join(', ')}}]`).join(' · ') || 'none';
    drawCrop();
  }}
  function escapeHtml(s) {{ return String(s).replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m])); }}
  function drawCrop() {{ const c = current(), im = imageMap[c.image_id], image = el('image'), canvas = el('cropCanvas'); if (!image.complete) {{ image.onload = drawCrop; return; }} const [x1,y1,x2,y2]=c.bbox_xyxy; const w=Math.max(1,Math.round(x2-x1)), h=Math.max(1,Math.round(y2-y1)); canvas.width=w; canvas.height=h; canvas.getContext('2d').drawImage(image,x1,y1,w,h,0,0,w,h); }}
  function writeField() {{ const c=current(); if (!c) return; const old=reviews[c.candidate_id] || {{}}; reviews[c.candidate_id]={{...old, candidate_id:c.candidate_id, image_id:c.image_id, category:c.category, bbox_xyxy:c.bbox_xyxy, hidden_provenance:c.hidden_provenance || {{}}, verdict:el('verdict').value || null, entity_ref:el('entityRef').value.trim(), semantic_status:el('semantic').value || '', geometry_status:el('geometry').value || '', comment:el('comment').value, timestamp:new Date().toISOString()}}; save(); el('counter').textContent=countText(); el('status').textContent='Autosaved locally'; }}
  function setVerdict(v) {{ el('verdict').value=v; writeField(); }}
  function move(delta) {{ writeField(); const visible = visibleCandidates(); if (visible.length) index=Math.max(0,Math.min(visible.length-1,index+delta)); updateForm(); }}
  const SEMANTIC_STATUSES=['exact','wrong_category','ambiguous','not_applicable']; const GEOMETRY_STATUSES=['acceptable','localization_error','fragment','multiple_entities','ambiguous'];
  function validEntityRef(r) {{ if (!r.entity_ref) return false; const prefix=`human:${{r.image_id}}:`; if (r.entity_ref.startsWith(prefix) && r.entity_ref.length > prefix.length) return true; return ledger.some(x => x.image_id === r.image_id && x.object_id === r.entity_ref); }}
  function exported() {{ const rows=Object.values(reviews).filter(r=>r.verdict).map(r=>{{ const source=candidates.find(c=>c.candidate_id===r.candidate_id) || {{}}; return {{...r, hidden_provenance:source.hidden_provenance || r.hidden_provenance || {{}}, source_candidate_id:source.candidate_id || r.candidate_id, source_image_id:source.image_id || r.image_id, source_category:source.category || r.category, source_bbox_xyxy:source.bbox_xyxy || r.bbox_xyxy}}; }}); const approved=rows.filter(r=>r.verdict==='approve'); const missingEntity=approved.filter(r=>!r.entity_ref); const invalidEntity=approved.filter(r=>!validEntityRef(r)); const invalidStatus=approved.filter(r=>!SEMANTIC_STATUSES.includes(r.semantic_status)||!GEOMETRY_STATUSES.includes(r.geometry_status)); const validApproved=approved.filter(r=>validEntityRef(r)&&!invalidStatus.includes(r)); const unreviewed=candidates.filter(c=>!reviews[c.candidate_id] || !reviews[c.candidate_id].verdict); const gate={{total_candidates:candidates.length, reviewed:rows.length, unreviewed:unreviewed.length, approved:approved.length, approved_valid:validApproved.length, rejected:rows.filter(r=>r.verdict==='reject').length, unknown:rows.filter(r=>r.verdict==='unknown').length, approved_missing_entity_ref:missingEntity.length, approved_invalid_entity_ref:invalidEntity.length, approved_missing_status:invalidStatus.length, approved_invalid_status:invalidStatus.length, zero_candidate_cohort:candidates.length===0, annotation_gate_passed:candidates.length===0 || (unreviewed.length===0 && invalidEntity.length===0 && invalidStatus.length===0), gate_reason:candidates.length===0 ? 'zero_candidates' : (unreviewed.length ? 'unreviewed_candidates' : (invalidEntity.length || invalidStatus.length ? 'invalid_approved_decisions' : 'complete'))}}; return {{schema_version:'coordexp.unmatched_prediction_review.export.v1', review_set_id:P.review_set_id, source_digests:P.source_digests, reviewer:P.reviewer, ontology:P.ontology, selected_image_ids:P.selected_image_ids, hidden_provenance:P.hidden_provenance, arm:P.arm, candidates:rows, annotation_gate_summary:gate, exported_at:new Date().toISOString()}}; }}
  function download() {{ const blob=new Blob([JSON.stringify(exported(),null,2)],{{type:'application/json'}}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=P.review_set_id+'.review.json'; a.click(); URL.revokeObjectURL(a.href); }}
  function showConsolidation() {{ consolidationMode=!consolidationMode; el('consolidate').textContent=consolidationMode ? 'Exit consolidation' : 'Consolidation'; index=0; const data=exported(); const missing=data.annotation_gate_summary.approved_missing_entity_ref; const text=`Consolidation: ${{data.annotation_gate_summary.reviewed}} reviewed / ${{data.annotation_gate_summary.total_candidates}} candidates; approved=${{data.annotation_gate_summary.approved}}; rejected=${{data.annotation_gate_summary.rejected}}; unknown=${{data.annotation_gate_summary.unknown}}; approved missing entity reference=${{missing}}`; el('status').textContent=text; updateForm(); }}
  function assignNewEntity() {{ const c=current(); if (!c) return; const ordinal=candidates.filter(x=>x.image_id===c.image_id).findIndex(x=>x.candidate_id===c.candidate_id)+1; el('entityRef').value=`human:${{c.image_id}}:${{String(ordinal).padStart(4,'0')}}`; writeField(); }}
  function sameObject(a,b) {{ const ak=Object.keys(a||{{}}).sort(), bk=Object.keys(b||{{}}).sort(); return ak.length===bk.length && ak.every((k,i)=>k===bk[i] && String(a[k])===String(b[k])); }}
  el('prev').onclick=()=>move(-1); el('next').onclick=()=>move(1); el('consolidate').onclick=showConsolidation; el('newEntity').onclick=assignNewEntity; el('export').onclick=download; document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>setVerdict(b.dataset.v)); ['verdict','entityRef','semantic','geometry','comment'].forEach(id=>el(id).addEventListener('change',writeField)); el('comment').addEventListener('input',writeField); el('importButton').onclick=()=>el('import').click(); el('import').onchange=event=>{{ const file=event.target.files[0]; if(!file)return; const reader=new FileReader(); reader.onload=()=>{{ try {{ const data=JSON.parse(reader.result); if(data.review_set_id!==P.review_set_id)throw new Error('review_set_id mismatch'); if(!sameObject(data.source_digests,P.source_digests))throw new Error('source_digests mismatch'); reviews=Object.fromEntries((data.candidates||[]).filter(r=>candidates.some(c=>c.candidate_id===r.candidate_id)).map(r=>[r.candidate_id,r])); save(); updateForm(); }} catch(e) {{ alert('Import failed: '+e.message); }} }}; reader.readAsText(file); }};
  document.addEventListener('keydown',e=>{{ if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return; if(e.key.toLowerCase()==='a')setVerdict('approve'); else if(e.key.toLowerCase()==='r')setVerdict('reject'); else if(e.key.toLowerCase()==='u')setVerdict('unknown'); else if(e.key==='ArrowRight'||e.key===']')move(1); else if(e.key==='ArrowLeft'||e.key==='[')move(-1); }});
  updateForm();
}})();
</script></body></html>'''


def _flatten_image_ids(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in str(value).split(",") if part.strip())
    return result or None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--review-manifest", dest="manifest", type=Path, required=True, help="Manual-review manifest JSON")
    parser.add_argument("--queue", "--unmatched-queue", dest="queue", type=Path, required=True, help="Unmatched-prediction queue JSONL")
    parser.add_argument("--accepted-ledger", "--audit-ledger", "--accepted-ledger-jsonl", dest="accepted_ledger", type=Path, required=True, help="Accepted/audit ledger JSONL")
    parser.add_argument("--image-id", "--images", action="append", default=None, help="Selected image id; repeat or comma-separate")
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--review-set-id", default=None)
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY)
    parser.add_argument("--output", "--output-html", dest="output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_review_payload(
        manifest_path=args.manifest,
        queue_path=args.queue,
        accepted_ledger_path=args.accepted_ledger,
        image_ids=_flatten_image_ids(args.image_id),
        arm=args.arm,
        review_set_id=args.review_set_id,
        reviewer=args.reviewer,
        ontology=args.ontology,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "images": len(payload["images"]), "candidates": len(payload["candidates"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
