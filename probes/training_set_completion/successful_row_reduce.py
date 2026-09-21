"""Saved-output accounting for the fixed successful-row mechanism package."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from probes.training_set_completion.artifacts import literal_binding as _binding, write_pretty_json as _write
from probes.training_set_completion import row_scoring as scorer

SCORER = Path(scorer.__file__)
ARTIFACT_BASE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
PREVIOUS = ARTIFACT_BASE / '2026-09-17-repetition-history-mechanism'
BANK_PANEL = PREVIOUS.parent / '2026-09-17-readout-norm-fresh128/panel.json'


def reduce(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    cells = {}
    tokenizer = None
    base_model = None
    for entry in manifest['cells']:
        panel_path = Path(entry['panel'])
        panel = json.loads(panel_path.read_text())
        case = panel['cases'][0]
        target = case['target_position']
        native_case = case['group']['cases'][target]
        if tokenizer is None:
            base_model = panel['config']['model']['base_model']
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        assert panel['config']['model']['base_model'] == base_model
        raw_path = Path(entry['raw'])
        raw = json.loads(raw_path.read_text())
        row = raw['rows'][target]
        assert row['image_id'] == 309264
        tokens = row['token_ids']
        assert tokens[:63] == case['target_prefix_token_ids'][:63]
        bank = json.loads(BANK_PANEL.read_text())['banks']['309264']

        def view(ids, stop):
            text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            result = scorer.score(dict(token_ids=ids, text=text, stop=stop), native_case, bank)
            if ids in ([], [tokenizer.eos_token_id], [151645]):
                result['burden']['malformed'] = 0
            return result

        # A patched decision supplies part of its complete row, not only one token.
        intervention_end = 63
        if entry.get('intervention_offset') is not None:
            offset = entry['intervention_offset']
            assert offset == 67
            intervention_end = next((i + 1 for i in range(offset, len(tokens)) if tokens[i] in (151649, 151645)), len(tokens))
        full = view(tokens, row['stop'])
        supplied = view(tokens[54:intervention_end], 'supplied')
        free = view(tokens[intervention_end:], row['stop'])
        prefix = view(tokens[:54], 'prefix')
        # Exact exemplar rows transfer prior human evidence, never infer identities by IoU.
        prior_raw = json.loads((PREVIOUS/'runtime/C/T0.7-seed24/raw.json').read_text())
        prior_tokens = next(r['token_ids'] for r in prior_raw['rows'] if r['image_id'] == 309264)
        exemplars = view(prior_tokens[63:], 'im_end')['complete_rows']
        ledger = json.loads((PREVIOUS/'physical-review.json').read_text())['stage_c']['positive']['physical_ledger']
        exact_owners = {}
        exact_supplied = {}
        for owner in ledger:
            if not owner['status'].startswith('credible'):
                continue
            fingerprints = {(exemplars[i-1]['description'], tuple(exemplars[i-1]['box'])) for i in owner['free_rows']}
            matches = [r['row'] for r in free['complete_rows'] if (r['description'], tuple(r['box'])) in fingerprints]
            if matches:
                exact_owners[owner['id']] = matches
            supplied_matches = [r['row'] for r in supplied['complete_rows'] if (r['description'], tuple(r['box'])) in fingerprints]
            if supplied_matches:
                exact_supplied[owner['id']] = supplied_matches
        cells[entry['id']] = dict(entry=entry, raw=_binding(raw_path), panel=_binding(panel_path),
            release=intervention_end, full=full, supplied=supplied, free=free, prefix=prefix,
            exact_prior_physical_exemplars=exact_owners,
            exact_supplied_physical_exemplars=exact_supplied,
            physical_limit='Only identical reviewed rows transfer identity; other rows remain HOLD pending bounded review.',
            free_token_ids=tokens[intervention_end:])
    groups = {}
    physical_unions = {}
    for name, cell in cells.items():
        group = cell['entry']['comparison_group']
        groups.setdefault(group, set()).update(cell['supplied']['matches']['covered_owner_ids'])
        physical_unions.setdefault(group, set()).update(cell['exact_supplied_physical_exemplars'])
    for name, cell in cells.items():
        entry = cell['entry']
        baseline = cells[entry['baseline']]
        union = groups[entry['comparison_group']]
        free = set(cell['free']['matches']['covered_owner_ids']) - union
        base = set(baseline['free']['matches']['covered_owner_ids']) - union
        prefix = set(cell['prefix']['matches']['covered_owner_ids'])
        cell['symmetric_known_accounting'] = dict(excluded_supplied_union=sorted(union),
            free=sorted(free), new_relative_prefix=sorted(free-prefix),
            gained=sorted(free-base), lost=sorted(base-free), retained=sorted(free&base),
            retention_denominator=len(base), baseline=entry['baseline'])
        cell['individual_known_accounting'] = dict(
            supplied=cell['supplied']['matches']['covered_owner_ids'],
            free=cell['free']['matches']['covered_owner_ids'],
            full=cell['full']['matches']['covered_owner_ids'])
        physical_union = physical_unions[entry['comparison_group']]
        physical_free = set(cell['exact_prior_physical_exemplars']) - physical_union
        physical_base = set(baseline['exact_prior_physical_exemplars']) - physical_union
        cell['exact_exemplar_physical_accounting'] = dict(excluded_supplied_union=sorted(physical_union),
            free=sorted(physical_free), gained=sorted(physical_free-physical_base),
            lost_exact_exemplars=sorted(physical_base-physical_free), retained=sorted(physical_free&physical_base),
            limitation='Missing exact exemplar is not physical disappearance; changed extents and unresolved supplied identities require a sidecar.')
    return dict(status='candidate', manifest=_binding(manifest_path), consumer=_binding(Path(__file__)),
                scorer=_binding(SCORER), bank_source=_binding(BANK_PANEL), physical_evidence=_binding(PREVIOUS/'physical-review.json'), cells=cells)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    _write(args.out, reduce(args.manifest))
