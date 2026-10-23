"""Validate historical captures and recompute summaries. Python stdlib; no engines/GPU."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
from build_2x2_from_raw import collect, summarise

EVIDENCE = Path(__file__).resolve().parents[1]
REPO = EVIDENCE.parents[1]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(name):
    with (EVIDENCE / 'data' / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, label, tolerance=1e-10):
    check(a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance),
          f'{label}: {a!r} != {b!r}')


def compare(a, b, label=''):
    if isinstance(a, dict):
        check(set(a) == set(b), label + ': keys differ')
        for key in a:
            compare(a[key], b[key], label + '/' + key)
    elif isinstance(a, list):
        check(len(a) == len(b), label + ': lengths differ')
        for index, (x, y) in enumerate(zip(a, b)):
            compare(x, y, label + '/' + str(index))
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        close(a, b, label)
    else:
        check(a == b, label + ': values differ')


def subset(projected, full):
    if isinstance(projected, dict):
        return isinstance(full, dict) and all(k in full and subset(v, full[k]) for k, v in projected.items())
    if isinstance(projected, list):
        return isinstance(full, list) and len(projected) == len(full) and all(subset(x, y) for x, y in zip(projected, full))
    return projected == full


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=REPO / 'build/evidence')
    ap.add_argument('--raw-dir', type=Path, help='Extracted infengine-measurements-v1 directory (optional).')
    args = ap.parse_args()
    output = args.output.resolve()
    check(not output.is_relative_to(EVIDENCE), 'Output must be outside the immutable evidence directory.')
    output.mkdir(parents=True, exist_ok=True)
    inputs = EVIDENCE / 'inputs'
    root = inputs / 'm12_a100_validation'
    provenance = load(EVIDENCE / 'PROVENANCE.json')
    checks = []
    manifest = EVIDENCE / 'CHECKSUMS.sha256'
    check(manifest.exists(), 'Missing CHECKSUMS.sha256')
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (EVIDENCE / name).resolve()
        check(path.is_relative_to(EVIDENCE), 'Unsafe manifest path')
        check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Checksum mismatch: ' + name)
        listed.add(name)
    actual = {p.relative_to(EVIDENCE).as_posix() for p in EVIDENCE.rglob('*') if p.is_file() and p.name != 'CHECKSUMS.sha256' and '__pycache__' not in p.parts}
    check(actual == listed, 'Checksum manifest coverage differs from evidence files')
    for item in provenance['projections']:
        compact = inputs / item['path']
        check(hashlib.sha256(compact.read_bytes()).hexdigest() == item['projection_sha256'], 'Projection hash mismatch')
        if args.raw_dir:
            full = args.raw_dir / item['path']
            check(hashlib.sha256(full.read_bytes()).hexdigest() == item['sanitized_record_sha256'], 'Archive record hash mismatch: ' + item['path'])
            check(subset(load(compact), load(full)), 'Compact/full measurement mismatch: ' + item['path'])
    checks.append(f"{len(provenance['projections'])} projection hashes and complete evidence checksums")
    if args.raw_dir:
        for line in (args.raw_dir / 'CHECKSUMS.sha256').read_text().splitlines():
            digest, name = line.split('  ', 1)
            path = (args.raw_dir / name).resolve()
            check(path.is_relative_to(args.raw_dir.resolve()), 'Unsafe archive manifest path')
            check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Archive checksum mismatch: ' + name)
        checks.append('Full sanitized archive checksums and every compact/full measurement projection')
    arm_rows = rows('ab_per_arm.csv')
    check(len(arm_rows) == 48, 'Expected 48 A/B arm records')
    columns = {'req_s': 'throughput_req_s', 'tok_s': 'output_tokens_per_s', 'ttft_p95': 'ttft_p95_s',
               'e2e_p50': 'e2e_p50_s', 'e2e_p95': 'e2e_p95_s', 'queue_p95': 'queue_time_p95_s',
               'running': 'requests_running', 'waiting': 'requests_waiting', 'kv': 'kv_usage'}
    hist = {}
    for row in arm_rows:
        matches = list((root / row['experiment']).glob(f"pair{row['pair']}_{row['arm']}_v*/result.json"))
        check(len(matches) == 1, 'Ambiguous/missing arm input')
        run = load(matches[0])['levels'][0]['runs'][0]
        for column, key in columns.items():
            close(row[column], run['metrics'][key]['value'], row['experiment'] + '/' + column)
        close(row['completed'], run['counts']['completed'], 'completed')
        close(row['ttft_p50_ev'], run['execution_evidence']['request_stage']['ttft']['quantiles']['p50'], 'ttft_p50_ev')
        close(row['preempt'], run['execution_evidence']['engine_level']['num_preemptions']['value'], 'preempt')
        if row['pair'] == '1' and row['experiment'] in ('runtime_concurrency', 'runtime_max_scheduled_tokens'):
            hist.setdefault(row['experiment'], {})[row['arm']] = {'histograms': run['histograms']}
        if row['kind'] == 'confounded':
            counters = load(matches[0].with_name('cache_counters.json'))
            delta = {k: v - counters['before'].get(k, 0) for k, v in counters['after'].items()}
            hits = sum(v for k, v in delta.items() if 'hits' in k)
            queries = sum(v for k, v in delta.items() if 'queries' in k)
            close(row['cache_frac'], hits / queries, 'token cache hit fraction', .000051)
    checks.append('48 A/B arms: throughput, latency, completion count, scheduler gauges, KV, preemptions; 18 native cache fractions')
    pilots = {p: load(root / 'pilot' / p / 'result.json') for p in ('interactive', 'prefill', 'mixed')}
    pilot_rows = rows('pilot_profiles.csv')
    check(len(pilot_rows) == 15, 'Expected 15 pilot points')
    for row in pilot_rows:
        level = next(x for x in pilots[row['profile']]['levels'] if float(x['offered_load']) == float(row['offered_concurrency']))
        for column, key in [('req_s', 'throughput_req_s'), ('ttft_p95_s', 'ttft_p95_s')]:
            close(row[column], level['aggregate'][key]['median'], 'pilot/' + column)
    pressure = load(root / 'pilot/prefill_m12_1_result.json')
    published_pressure = load(EVIDENCE / 'data/pilot_prefill_m12_1.json')
    # This publication JSON is a selected capture: compare every aggregate and run metric it retains.
    check(subset(pressure['levels'], published_pressure['levels']), 'Pressure pilot measurement projection differs')
    checks.append('15 initial pilot points and complete M12.1 pressure-pilot levels')
    per_block = collect(str(inputs))
    check(len(per_block) == 12, 'Expected 12 factorial records')
    computed = summarise(per_block)
    factorial = {'cells': {c: computed[c] for c in 'ABCD'}, 'effects': computed['_effects']}
    published = load(EVIDENCE / 'data/2x2_summary.json')
    for key in factorial:
        compare(factorial[key], published[key], 'factorial/' + key)
    for raw, table in zip(per_block, rows('2x2_per_block.csv')):
        for key, value in raw.items():
            if value is None:
                check(table[key] == '', 'Expected missing table value')
            elif isinstance(value, (int, float)):
                close(value, table[key], 'factorial/' + key)
            else:
                check(value == table[key], 'Factorial label mismatch')
    checks.append('12 factorial records reproduce per-block CSV and summary JSON within 1e-10')
    reconciliation = load(EVIDENCE / 'support/ab-ratio-reconciliation.json')
    archived_medians = {r['experiment']: r for r in rows('ab_medians.csv')}
    ratios = {}
    medians = []
    for exp in sorted({r['experiment'] for r in arm_rows}):
        selected = [r for r in arm_rows if r['experiment'] == exp]
        ratios[exp] = {}
        record = {'experiment': exp}
        for metric in columns:
            for arm in 'AB':
                record[metric + '_' + arm] = statistics.median(float(r[metric]) for r in selected if r['arm'] == arm)
                close(record[metric + '_' + arm], archived_medians[exp][metric + '_' + arm], 'rounded marginal median', .000051)
            if metric in ('req_s', 'tok_s', 'ttft_p95', 'e2e_p95', 'queue_p95'):
                paired = [float(next(r[metric] for r in selected if r['pair'] == str(p) and r['arm'] == 'B')) /
                          float(next(r[metric] for r in selected if r['pair'] == str(p) and r['arm'] == 'A')) for p in (1, 2, 3)]
                ratios[exp][metric] = statistics.median(paired)
                compare(paired, reconciliation['experiments'][exp][metric]['per_pair_B_over_A'], 'paired ratios')
                close(ratios[exp][metric], reconciliation['experiments'][exp][metric]['recomputed_median'], 'ratio median')
                close(archived_medians[exp][metric + '_B_over_A'], reconciliation['experiments'][exp][metric]['archived_summary'], 'archived ratio provenance')
        medians.append(record)
    # Validate the cache-order table, including AB/BA ordering, against per-arm measurements.
    for row in rows('cache_confound.csv'):
        order = row['order'].split('->')
        exp = row['experiment']
        for which, arm in zip(('first', 'second'), order):
            arm_row = next(r for r in arm_rows if r['experiment'] == exp and r['pair'] == row['pair'] and r['arm'] == arm)
            close(row[which + '_arm_cache_frac'], arm_row['cache_frac'], 'cache-order table', .000051)
    # Adjacent differences of cumulative histogram buckets, not a second cumulative distribution.
    bins = {}
    for exp, family in [('runtime_concurrency', 'ttft'), ('runtime_max_scheduled_tokens', 'e2e')]:
        bins[exp] = {}
        for arm in 'AB':
            h = hist[exp][arm]['histograms']['families'][family]
            disjoint = [c - (h['counts'][i-1] if i else 0) for i, c in enumerate(h['counts'])]
            check(all(c >= 0 for c in disjoint) and sum(disjoint) == h['total'], 'Invalid cumulative histogram')
            bins[exp][arm] = {'family': family, 'upper_bounds_s': h['le'], 'counts': disjoint, 'total': h['total']}
    smoke = load(root / 'deployment/smoke/smoke.json')
    values = [smoke[k]['policies']['runtime_max_running_seqs'] for k in ('raw controls before', 'raw controls after set', 'raw controls after restore')]
    check(values == [64, 16, 64], 'Smoke native readback differs')
    pids = [smoke[k]['api_pid'] for k in ('identity before', 'identity during (must equal before)', 'identity after (must equal before)')]
    check(len(set(pids)) == 1, 'API PID changed')
    check(all(smoke[k]['status'] == 'CONVERGED' for k in ('set runtime_concurrency 16', 'restore runtime_concurrency 64')), 'Smoke convergence absent')
    check(all(smoke[k]['object'] == 'text_completion' and smoke[k]['usage']['completion_tokens'] > 0 for k in ('inference while concurrency=16', 'inference after restore')), 'Successful completion payload absent')
    checks.append('40 paired-ratio summaries; cache-order table; four native histogram windows; live readback/restore and API PID continuity')
    result = {'checks': checks, 'factorial': factorial, 'factorial_rows': per_block, 'histogram_inputs': hist,
              'disjoint_histograms': bins, 'paired_B_over_A_medians': ratios,
              'smoke': {'native_values': values, 'api_pid': pids[0], 'http_status_explicitly_captured': False}}
    (output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (output / '2x2_summary.json').write_text(json.dumps(factorial, indent=2) + '\n', encoding='utf-8')
    for name, records in [('2x2_per_block.csv', per_block), ('ab_medians_recomputed.csv', medians)]:
        with (output / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({'passed': checks, 'interaction_req_s': factorial['effects']['interaction_abs_req_s']}, indent=2))


if __name__ == '__main__':
    main()
