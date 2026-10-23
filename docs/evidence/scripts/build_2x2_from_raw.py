"""Pure factorial functions from the original publication analysis; see verify_and_analyze.py for CLI."""
from __future__ import annotations
import argparse
import csv
import json
import os
CELLS = ['A', 'B', 'C', 'D']
CELL_FACTORS = {'A': (8192, 64), 'B': (8192, 32), 'C': (1024, 64), 'D': (1024, 32)}

def _median(xs):
    xs = sorted((x for x in xs if x is not None))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def _cv_pct(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    if m == 0:
        return 0.0
    var = sum(((x - m) ** 2 for x in xs)) / len(xs)
    return 100.0 * var ** 0.5 / m

def _cell_name_to_key(dirname):
    return dirname.split('_')[1]

def collect(root):
    two = os.path.join(root, 'm12_a100_validation', 'twobytwo')
    per_block = []
    for entry in sorted(os.listdir(two)):
        cell_dir = os.path.join(two, entry)
        rf = os.path.join(cell_dir, 'result.json')
        if not os.path.isfile(rf):
            continue
        key = _cell_name_to_key(entry)
        block = int(entry.split('_')[0].replace('block', ''))
        with open(rf, 'r', encoding='utf-8') as fh:
            j = json.load(fh)
        run = j['levels'][0]['runs'][0]
        m = run['metrics']
        ee = run['execution_evidence']['request_stage']
        norm = (run.get('snapshot') or {}).get('normalized') or {}
        tok, conc = CELL_FACTORS[key]

        def mv(key):
            v = (m.get(key) or {}).get('value')
            return v if v is not None else norm.get(key)

        def q(fam, p):
            return ee.get(fam, {}).get('quantiles', {}).get(p)
        per_block.append({'cell': key, 'block': block, 'tok': tok, 'conc': conc, 'completed': run['counts']['completed'], 'incomplete': run['counts']['incomplete'], 'timeout': run['counts']['timeout'], 'error': run['counts']['error'], 'drained': run['counts']['drained'], 'req_s': mv('throughput_req_s'), 'tok_s': mv('output_tokens_per_s'), 'ttft_p50': mv('ttft_p50_s'), 'ttft_p95': mv('ttft_p95_s'), 'ttft_p99': mv('ttft_p99_s'), 'e2e_p50': mv('e2e_p50_s'), 'e2e_p95': mv('e2e_p95_s'), 'queue_p95': mv('queue_time_p95_s'), 'prefill_p50': q('prefill_time', 'p50'), 'decode_p50': q('decode_time', 'p50'), 'tpot_p50': q('tpot', 'p50'), 'running': mv('requests_running'), 'waiting': mv('requests_waiting'), 'kv': mv('kv_usage'), 'preempt': run['execution_evidence']['engine_level']['num_preemptions']['value'], 'ttft_n': ee.get('ttft', {}).get('sample_count')})
    return per_block

def summarise(per_block):
    summary = {}
    for key in CELLS:
        rows = [r for r in per_block if r['cell'] == key]
        req = [r['req_s'] for r in rows]
        tok, conc = CELL_FACTORS[key]
        summary[key] = {'tok': tok, 'conc': conc, 'req_s_mean': _mean(req), 'req_s_cv_pct': _cv_pct(req), 'tok_s_mean': _mean([r['tok_s'] for r in rows]), 'ttft_p50_mean': _mean([r['ttft_p50'] for r in rows]), 'ttft_p95_mean': _mean([r['ttft_p95'] for r in rows]), 'ttft_p99_mean': _mean([r['ttft_p99'] for r in rows]), 'e2e_p50_mean': _mean([r['e2e_p50'] for r in rows]), 'e2e_p95_mean': _mean([r['e2e_p95'] for r in rows]), 'queue_p95_mean': _mean([r['queue_p95'] for r in rows]), 'running_mean': _mean([r['running'] for r in rows]), 'waiting_mean': _mean([r['waiting'] for r in rows]), 'kv_mean': _mean([r['kv'] for r in rows]), 'preempt_mean': _mean([r['preempt'] for r in rows]), 'completed_total': sum((r['completed'] for r in rows)), 'repeat_range_req_s_pct': 100.0 * (max(req) - min(req)) / _mean(req) if _mean(req) else 0.0}
    a, b = (summary['A']['req_s_mean'], summary['B']['req_s_mean'])
    c, d = (summary['C']['req_s_mean'], summary['D']['req_s_mean'])

    def pct(num, den):
        return 100.0 * num / den if den else None
    tok64 = c - a
    tok32 = d - b
    main_token = (c + d) / 2 - (a + b) / 2
    main_conc = (b + d) / 2 - (a + c) / 2
    interaction = tok64 - tok32
    summary['_effects'] = {'main_token_pct': pct(main_token, (a + b) / 2), 'main_conc_pct': pct(main_conc, (a + c) / 2), 'token_eff_conc64_pct': pct(tok64, a), 'token_eff_conc32_pct': pct(tok32, b), 'interaction_abs_req_s': interaction, 'interaction_pct': pct(interaction, a)}
    return summary
