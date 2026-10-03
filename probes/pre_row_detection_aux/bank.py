"""Immutable accepted v0 bank, causal row binding and CPU input inventory."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from probes import online_row_credit as o
from probes.full_label_fit import experiment as fit
UNIT = Path(__file__).resolve().parents[2] / 'research/experiments/2026-10-03-pre-row-detection-aux'
PREDECESSOR = UNIT.parent / '2026-10-02-full-label-self-rollout-fit'
OUT = Path(__file__).resolve().parents[2] / 'outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12'

def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def checked(path, expected):
    require(fit.sha(Path(path)) == expected, f'input identity drift: {path}')
    return read(path)


def positive_rows(image, record, plan, tokenizer, job):
    """The exact positive rows, coefficients and histories consumed by o.forward."""
    branch = job['branch']
    if branch == 'trace':
        selected = o.bridge_trace_plan(plan, 'chain') if plan['completion_arm'] == 'treatment' else plan
        rows = selected['M']
        sequences = [o.r.positive_sequence(image, record, row, tokenizer) for row in rows]
        weights = [1 / len(rows)] * len(rows) if rows else []
        kinds = ['trace_M'] * len(rows)
    elif branch == 'bridge':
        _, rows, weights, nb = o.bridge_rows(record, plan)
        sequences = o.bridge_sequences(image, record, plan, tokenizer)
        kinds = ['CHAIN_B'] * nb + ['relocated_M'] * (len(rows) - nb)
    else:
        require(branch == 'redirect', 'unexpected bank branch')
        rows = [plan['redirects'][job['branch_index']]]
        sequences = [o.redirect_sequence(image, record, rows[0], tokenizer)]
        weights = [rows[0]['event_weight']]
        kinds = ['redirect']
    return list(zip(rows, sequences, weights, kinds))


def opener_position(sequence, row, prompt_length, tokenizer):
    position = prompt_length + row['positions'][0]
    require(sequence.input_ids[position] == tokenizer.convert_tokens_to_ids('<|object_ref_start|>'), 'opener position drift')
    atoms = sequence.atoms
    require(atoms[0].target_position == position and atoms[0].token_type == 'schema', 'opener atom drift')
    require(all(a.target_position > position for a in atoms if a.token_type in ('desc_text', 'coordinate')), 'target leakage')
    return position


def exclude_conflicts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['image_id'], row['image_identity'], tuple(row['prefix']))].append(row)
    for group in groups.values():
        targets = {(r['class'], tuple(r['box'])) for r in group}
        for row in group:
            row['eligible'] = len(targets) == 1
            row['conflict_targets'] = len(targets)
    return rows






def verify_bank(path, expected):
    bank = checked(path, expected)
    require(bank['schema'] == 'pre-row-fixed-bank-v1', 'bank schema drift')
    for source, digest in bank['input_sha256'].items():
        require(fit.sha(Path(source)) == digest, f'persisted input identity drift: {source}')
    if 'recipe' in bank:
        images,manifest=fit.verify_inputs(PREDECESSOR)
        require(images==bank['images'] and manifest['full_labels']['sha256']==bank['recipe']['training_sha256'],'full-label snapshot binding drift')
    return bank


def execution_jobs(bank,tokenizer):
    records={r['image_id']:r for r in bank['records']};plans={p['image_id']:p for p in bank['plans']};images={i['image_id']:i for i in bank['images']}
    jobs=[]
    for original in bank['jobs']:
        job=dict(original);i=job['image_id'];raw=records[i];plan=plans[i]
        if job['branch']=='trace':
            full=raw['prompt_token_ids']+raw['token_ids'];positions=o.trace_positions(o.bridge_trace_plan(plan,'chain'),raw)
        elif job['branch']=='bridge':
            sequences=o.bridge_sequences(images[i],raw,plan,tokenizer);full=list(sequences[0].input_ids)
            positions=sorted({a.causal_logits_position for seq in sequences for a in seq.atoms})
        else:
            sequence=o.redirect_sequence(images[i],raw,plan['redirects'][job['branch_index']],tokenizer)
            full=list(sequence.input_ids);positions=[a.causal_logits_position for a in sequence.atoms]
        job.update(tokens=len(full),selected_logits=len(positions),input_sha256=o.identity(full),positions=list(positions),visual_tokens=math.prod(raw['image_grid_thw'])//4)
        jobs.append(job)
    return jobs


def selection_cases(path,digest,bank,bank_sha256):
    selection=checked(path,digest)
    require(selection['schema']=='pre-row-conditional-selection-v1' and selection['bank_sha256']==bank_sha256,'selection bank drift')
    require(selection['max_new_tokens']==64 and selection['max_requests']==24 and selection['max_generated_tokens']==1536,'selection dose drift')
    cases=selection['cases']
    require(len(cases)==8 and len({r['row_id'] for r in cases})==8,'selection row coverage drift')
    require(Counter(r['history_kind'] for r in cases)==dict(actual=4,synthetic=4),'history split drift')
    for case in cases:
        row=bank['rows'][case['row_id']]
        require(row['eligible'] and all(row[k]==v for k,v in case.items()),'selection target or prefix drift')
    return [bank['rows'][case['row_id']] for case in cases]


def bank_costs(bank,jobs,conditional,updates,hidden_size):
    selected_max=max(j['selected_logits'] for j in jobs)
    head_parameters=(hidden_size+1)*(len(bank['classes'])+4)
    return dict(training=dict(arms=2,updates_per_arm=updates,forwards_per_arm=updates*len(jobs),
        forwards_total=2*updates*len(jobs),input_tokens_total=2*updates*sum(j['tokens'] for j in jobs),
        visual_tokens_total=2*updates*sum(j['visual_tokens'] for j in jobs),context_max=max(j['tokens'] for j in jobs),
        selected_logits_max=selected_max,selected_logit_elements_per_largest_forward=selected_max*152670,
        BF16_selected_logits_bytes=selected_max*152670*2,FP32_selected_logits_bytes=selected_max*152670*4,
        hidden_rows_max=max(sum(r['eligible'] and r['job_index']==k for r in bank['rows']) for k in range(len(jobs))),
        hidden_width=hidden_size,head_parameters=head_parameters,head_FP32_parameter_gradient_Adam_bytes=head_parameters*16),
        evaluation=dict(checkpoints=3,ordinary_requests=54,conditional_requests=3*len(conditional),
            request_total=54+3*len(conditional),generated_token_ceiling=54*3084+3*len(conditional)*64,
            ordinary_context_ceiling=max(len(r['prompt_token_ids']) for r in bank['records'])+3084,
            conditional_context_ceiling=max(len(r['prefix']) for r in conditional)+64),
        estimates=dict(whole_owner_wall_seconds=1800 if updates==1 else 5400,
            execution_cutoff_seconds=1770 if updates==1 else 5370,cleanup_reserve_seconds=30,
            aggregate_RSS_ceiling_GiB=160,artifact_ceiling_GiB=8,
            basis='proposed bounds, not measured native qualification; includes all loads, checks, exports, evaluation and cleanup'))
