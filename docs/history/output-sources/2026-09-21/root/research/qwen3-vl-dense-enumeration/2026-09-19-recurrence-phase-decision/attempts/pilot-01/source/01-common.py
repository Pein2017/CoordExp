"""Bound native source and prefix helpers for this one frozen unit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.numerical_feedback.runtime import full_prefix
from probes.training_set_completion.recurrence_phase_decision.prepare import PANEL, binding


def read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    if plan.get('status') != 'frozen' or plan.get('panel') != binding(PANEL):
        raise ValueError('phase plan or shared panel binding changed')
    if plan.get('sources') != binding(PANEL.with_name('shared-sources.json')):
        raise ValueError('source snapshot binding changed')
    return plan


def bound_source(plan: dict[str, Any], panel: dict[str, Any], boundary_id: str,
                 q: Any, device: Any) -> tuple[Any, list[dict], dict, dict]:
    matches = [b for b in panel['all_boundaries'] if b['id'] == boundary_id]
    summaries = [s for s in plan['source_summaries'] if s['boundary_id'] == boundary_id]
    if len(matches) != 1 or len(summaries) != 1:
        raise ValueError(f'nonunique source {boundary_id}')
    boundary, summary = matches[0], summaries[0]
    for key in ('raw', 'trace', 'receipt'):
        if binding(Path(boundary[f'{key}_path']))['sha256'] != summary['source'][key]['sha256']:
            raise ValueError(f'source {key} binding changed for {boundary_id}')
    source_panel_path = Path(summary['source_panel']['path'])
    if binding(source_panel_path) != summary['source_panel']:
        raise ValueError('source panel binding changed')
    source_panel = json.loads(source_panel_path.read_text())
    batch, raw, trace, _, planning = _source(boundary, boundary['model'], source_panel, q, device)
    return batch, raw, trace, {'boundary': boundary, 'summary': summary, 'planning': planning,
                               'source_panel': str(source_panel_path)}


def prefix(batch: Any, raw: list[dict], boundary: dict, end: int, q: Any,
           device: Any, site: tuple[int, int, int] | None = None) -> dict[str, Any]:
    if end < 0 or end > len(boundary['native_tokens']):
        raise ValueError('prefix outside native tokens')
    mutation = None
    if site is not None:
        offset, old, new = site
        if offset >= end or boundary['native_tokens'][offset] != old:
            raise ValueError('edited site does not match consumed native prefix')
        mutation = (int(boundary['batch_index']), offset, old, new)
    inputs = full_prefix(batch, raw, end, int(q.tokenizer.pad_token_id), device, mutation)
    target = int(boundary['batch_index'])
    prompt_width = int(batch.inputs['input_ids'].shape[1])
    expected = list(boundary['native_tokens'][:end])
    if site is not None:
        expected[site[0]] = site[2]
    actual = inputs['input_ids'][target, prompt_width:].tolist()
    if actual != expected:
        raise ValueError('prefix position readback differs from native/edited tokens')
    return inputs


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
