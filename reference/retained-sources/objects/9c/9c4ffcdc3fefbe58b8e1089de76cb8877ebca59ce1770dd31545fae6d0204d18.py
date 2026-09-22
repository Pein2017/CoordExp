"""Root-authorized untouched second cell only; never repeat Source(H_new)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
API = runpy.run_path(str(OUT/'run.py'))
read, require, publish = API['read'], API['require'], API['publish']
file_hash, digest, rows = API['file_hash'], API['digest'], API['rows']
THRESHOLDS = API['THRESHOLDS']


def first_cell():
    partial = read(OUT/'partial-receipt.json')
    for name, expected in partial['artifact_sha256'].items():
        require(file_hash(OUT/name) == expected, 'sealed first invocation changed: '+name)
    terminal = read(OUT/'terminal.json')
    raw = rows(OUT/'raw.jsonl')
    require(terminal['status'] == 'failed' and terminal['error'] ==
            "ValueError('previous model remains live before next load')", 'unexpected first failure')
    require(terminal['model_loads'] == terminal['model_load_attempts'] == terminal['continuations'] ==
            terminal['continuation_attempts'] == 1 and len(raw) == 1, 'first-cell execution count')
    require(raw[0]['history'] == 'H_new' and raw[0]['checkpoint'] == 'Source', 'completed first cell identity')
    return terminal, raw[0]


def execute():
    import torch
    first, _ = first_cell()
    packet = read(OUT/'packet.json')
    API['assert_files'](packet['input_files'])
    require(not (OUT/'continuation-admission.json').exists(), 'second cell already attempted')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '2' and torch.cuda.device_count() == 1, 'GPU2 only')
    require(not Path(f"/proc/{first['pid']}").exists(), 'first process still present')
    gpu = subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    gpu_fields = next(csv.reader([gpu],skipinitialspace=True))
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    require(gpu_fields[0] == '2' and int(gpu_fields[2]) == 0 and gpu_fields[1] not in apps, 'GPU2 not released')
    admission = dict(status='authorized_untouched_second_cell',checkpoint='old_CE23',history='H_good',
        parent_packet_sha256=digest(packet),first_terminal_sha256=file_hash(OUT/'terminal.json'),
        first_raw_sha256=file_hash(OUT/'raw.jsonl'),first_partial_receipt_sha256=file_hash(OUT/'partial-receipt.json'),
        continuation_code_sha256=file_hash(__file__),first_process_absent=True,gpu2_before=gpu,
        aggregate_limits=packet['limits'],remaining_seconds=1800-first['elapsed_seconds'],
        root_ruling='Preserve failed invocation and completed first cell; one fresh GPU2 process for untouched oldCE23(H_good) only, no rerun or expansion.')
    publish(OUT/'continuation-admission.json',admission)
    terminal = dict(status='running',pid=os.getpid(),model_load_attempts=0,model_loads=0,
                    continuation_attempts=0,continuations=0,new_tokens=0,model_forwards=0,image_forwards=0,
                    admission_sha256=file_hash(OUT/'continuation-admission.json'))
    publish(OUT/'continuation-launch.json',dict(**terminal,time=time.time(),visible_devices='2'))
    started = time.monotonic()
    def expired(*_):raise TimeoutError('remaining cumulative1800-second envelope')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(max(1,math.floor(admission['remaining_seconds'])))
    try:
        row, consumed, resources = API['generate_one'](packet,dict(checkpoint='old_CE23',history='H_good'),terminal)
        with (OUT/'continuation-raw.jsonl').open('x') as stream:
            stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        publish(OUT/'old_CE23__H_good-consumer.json',consumed)
        terminal['resources'] = resources
        require(terminal['model_loads'] == terminal['continuations'] == 1, 'one second cell only')
        API['assert_files'](packet['input_files'])
        terminal['status'] = 'completed'
    except BaseException as exc:
        terminal.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        terminal['elapsed_seconds'] = time.monotonic()-started
        terminal['cumulative_seconds'] = first['elapsed_seconds']+terminal['elapsed_seconds']
        publish(OUT/'continuation-terminal.json',terminal)


def verify():
    first, first_raw = first_cell()
    packet = read(OUT/'packet.json')
    API['assert_files'](packet['input_files'])
    admission, second = read(OUT/'continuation-admission.json'),read(OUT/'continuation-terminal.json')
    require(admission['parent_packet_sha256'] == digest(packet) and
            admission['continuation_code_sha256'] == file_hash(__file__), 'continuation packet/code identity')
    require(admission['first_terminal_sha256'] == file_hash(OUT/'terminal.json') and
            admission['first_raw_sha256'] == file_hash(OUT/'raw.jsonl'), 'continuation first-cell binding')
    require(second['status'] == 'completed' and second['model_loads'] == second['model_load_attempts'] ==
            second['continuations'] == second['continuation_attempts'] == 1, 'second cell incomplete')
    require((OUT/'exit-code.txt').read_text().strip() == '1' and
            (OUT/'continuation-exit-code.txt').read_text().strip() == '0', 'invocation exit statuses')
    require(first['elapsed_seconds']+second['elapsed_seconds'] == second['cumulative_seconds'] <= 1800,
            'cumulative time bound')
    second_rows = rows(OUT/'continuation-raw.jsonl')
    require(len(second_rows) == 1, 'extra second-cell raw outputs')
    fresh = [first_raw,second_rows[0]]
    require([(r['checkpoint'],r['history']) for r in fresh] == [('Source','H_new'),('old_CE23','H_good')],
            'exact new two-cell set')
    require(all(r['packet_sha256'] == digest(packet) for r in fresh), 'raw parent packet identity')
    tokenizer = API['tokenizer_only'](packet['base_model']).tokenizer
    diags = rows(OUT/'retained-diagonals.jsonl')
    require([(r['checkpoint'],r['history']) for r in diags] == [('Source','H_good'),('old_CE23','H_new')],
            'exact retained diagonals')
    consumed = [API['consume'](r,packet,tokenizer) for r in diags+fresh]
    require(consumed[:2] == read(OUT/'diagonal-consumer.json'), 'diagonal native consumer drift')
    for row, result, term in zip(fresh,consumed[2:],[first,second],strict=True):
        label = row['checkpoint']+'__'+row['history']
        require(result == read(OUT/(label+'-consumer.json')), 'fresh consumer readback drift')
        require(len(row['suffix_ids']) == term['new_tokens'] == term['model_forwards'] and term['image_forwards'] == 1,
                'per-cell raw forward accounting')
        cold = read(OUT/(label+'-cold.json'))
        training = read(API['OLD']/'training/receipt.json')
        expected = training['adapter_tensor_hash_before' if row['checkpoint'] == 'Source' else 'adapter_tensor_hash_after']
        require(cold['tensor_hashes'] == dict(adapter=expected,frozen=training['frozen_tensor_hash_before']), 'cold tensor identities')
        require(cold['model_forwards_before_generation'] == 0, 'unregistered score forwards')
    total_tokens = sum(t['new_tokens'] for t in (first,second))
    require(total_tokens <= 6168, 'cumulative new-token bound')
    contrasts = {}
    for history in ('H_good','H_new'):
        cells = {r['checkpoint']:r for r in consumed if r['history'] == history}
        source,candidate = cells['Source'],cells['old_CE23']
        contrasts[history] = dict(source=source,candidate=candidate,
            owner_changes={t:API['owner_change'](candidate['full_score'][t]['owners'],source['full_score'][t]['owners']) for t in THRESHOLDS},
            metric_deltas={t:{k:candidate['full_score'][t][k]-source['full_score'][t][k] for k in ('tp','fp','fn','f1')} for t in THRESHOLDS},
            suffix_burden_deltas={k:candidate['suffix'][k]-source['suffix'][k] for k in ('valid_predictions','parser_drops','new_strict_repeats','tokens')})
    payload = dict(schema='boat_cross.consumer.v1',cells=consumed,contrasts=contrasts,
                   limitation=packet['conditioning_scope'],suffix_accounting='Only new suffix repetition is attributed to continuation; given-prefix rows are fixed.')
    if (OUT/'consumer.json').exists():require(read(OUT/'consumer.json') == payload,'consumer changed')
    else:publish(OUT/'consumer.json',payload)
    receipt = dict(status='candidate',diagnostic_execution='completed_exact_two_new_cells',
        first_invocation='failed after completed first cell at zero-CUDA-allocation cleanup assertion; preserved, not rerun',
        second_invocation='root-authorized untouched cell completed in fresh process',
        model_loads=2,continuations=2,diagonal_reruns=0,new_tokens=total_tokens,model_forwards=total_tokens,image_forwards=2,
        cumulative_seconds=second['cumulative_seconds'],gpu_hours=second['cumulative_seconds']/3600,
        peak_cuda_allocated_bytes=max(first['per_cell'][0]['peak_cuda_allocated_bytes'],second['resources']['peak_cuda_allocated_bytes']),
        rss_peak_bytes=max(first['rss_peak_bytes'],second['resources']['rss_peak_bytes']),
        artifact_sha256={name:file_hash(OUT/name) for name in ('packet.json','raw.jsonl','terminal.json','partial-receipt.json',
            'continuation-admission.json','continuation-raw.jsonl','continuation-terminal.json','consumer.json')},
        acceptance='Scientific acceptance remains lead-owned; no new learning/checkpoint/architecture claim.')
    if (OUT/'receipt.json').exists():require(read(OUT/'receipt.json') == receipt,'receipt changed')
    else:publish(OUT/'receipt.json',receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('execute','verify'))
    globals()[parser.parse_args().command]()
