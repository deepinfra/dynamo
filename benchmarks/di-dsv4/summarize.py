import sys, statistics as st

GPUS_DISAGG = 13

def load(path):
    rows = []
    for line in open(path):
        if not line.startswith('W'):
            continue
        d = {}
        for kv in line.split():
            if '=' in kv:
                k, v = kv.split('=', 1)
                d[k] = v
        try:
            rows.append({k: float(v) for k, v in d.items() if k != 't'})
        except ValueError:
            pass
    return rows

def med(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] > 0]
    return st.median(vals) if vals else float('nan')

def report(label, path):
    rows = load(path)
    if not rows:
        print(f"  {label}: no windows"); return None
    eng = med(rows, 'eng')
    out = {
        'n': len(rows),
        'hitD': med(rows, 'hitD'), 'hitP': med(rows, 'hitP'),
        't50D': med(rows, 't50D') * 1000, 't90D': med(rows, 't90D') * 1000, 't99D': med(rows, 't99D') * 1000,
        't50P': med(rows, 't50P') * 1000, 't90P': med(rows, 't90P') * 1000, 't99P': med(rows, 't99P') * 1000,
        'genD_gpu': med(rows, 'genD') / GPUS_DISAGG, 'genP_gpu': med(rows, 'genP') / eng,
        'reqD_gpu': med(rows, 'reqD') / GPUS_DISAGG, 'reqP_gpu': med(rows, 'reqP') / eng,
        'eng': eng,
    }
    print(f"\n  === {label} (n={out['n']} windows, prod engines={eng:.0f}) ===")
    print(f"    {'metric':<22} {'disagg':>10} {'prod':>10} {'ratio':>9}")
    def row(name, d, p, better_low=False, fmt="{:.0f}"):
        r = (p / d) if better_low else (d / p)
        mark = "  <-- win" if r > 1.0 else ""
        print(f"    {name:<22} {fmt.format(d):>10} {fmt.format(p):>10} {r:>8.2f}x{mark}")
    row('cache hit', out['hitD'], out['hitP'], fmt="{:.3f}")
    row('TTFT p50 (ms)', out['t50D'], out['t50P'], better_low=True)
    row('TTFT p90 (ms)', out['t90D'], out['t90P'], better_low=True)
    row('TTFT p99 (ms)', out['t99D'], out['t99P'], better_low=True)
    row('gen tok/s per GPU', out['genD_gpu'], out['genP_gpu'])
    row('req/s per GPU', out['reqD_gpu'], out['reqP_gpu'])
    return out

a = report('A: queue gate ON', sys.argv[1])
b = report('B: queue gate OFF', sys.argv[2]) if len(sys.argv) > 2 else None

if a and b:
    print("\n  === A vs B (disagg only, median of windows) ===")
    for k, name, low in (('t50D','TTFT p50',True), ('t90D','TTFT p90',True), ('t99D','TTFT p99',True),
                         ('hitD','cache hit',False), ('genD_gpu','gen tok/s/GPU',False), ('reqD_gpu','req/s/GPU',False)):
        delta = (a[k] / b[k]) if low else (b[k] / a[k])
        print(f"    {name:<16} A={a[k]:>10.3f}  B={b[k]:>10.3f}   B is {delta:.2f}x {'better' if delta>1 else 'worse'}")
