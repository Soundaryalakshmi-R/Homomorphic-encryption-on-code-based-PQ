"""Run from the project root:  python -m experiments.run_benchmark
Optional: python -m experiments.run_benchmark --sizes 100 1000 10000 100000 --cap 1000

Schemes slower than --cap items are measured on `cap` items and the totals
that scale with n (enc, agg, dec_each) are extrapolated linearly. Extrapolated
rows are marked with '*'.  Aggregate decryption (dec) is constant in n.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse, csv
from src.benchmarks import (make_values, bench_paper_scheme, bench_paillier,
                            bench_bfv_per_client, bench_bfv_batched)

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000, 100000])
ap.add_argument("--cap", type=int, default=1000, help="max items actually run for slow schemes")
args = ap.parse_args()

SCHEMES = [  # (label, function, needs_cap, secure?)
    ("Paper scheme (insecure)", bench_paper_scheme, False),
    ("Paillier-2048 (not PQ)",  bench_paillier,     True),
    ("BFV per-client (PQ)",     bench_bfv_per_client, True),
    ("BFV batched (PQ)",        bench_bfv_batched,  False),
]

def scale(r, n):
    m = r["measured_n"]
    if m == n:
        return r, False
    f = n / m
    r = dict(r)
    for key in ("enc", "agg", "dec_each"):
        if key in r:
            r[key] *= f
    return r, True

rows = []
for n in args.sizes:
    for label, fn, needs_cap in SCHEMES:
        run_n = min(n, args.cap) if needs_cap else n
        if label.startswith("BFV batched") and n > 20000:
            run_n = 20000                       # keep memory modest; scaled below
        r = fn(make_values(run_n))
        assert r["correct"], f"{label} decrypted incorrectly at n={run_n}"
        r, est = scale(r, n)
        rows.append(dict(n=n, scheme=label + ("*" if est else ""),
                         enc_ms=1e3 * r["enc"], agg_ms=1e3 * r["agg"],
                         dec_aggregate_ms=1e3 * r["dec"],
                         dec_each_ms=1e3 * r["dec_each"] if "dec_each" in r else None,
                         ct_bits=r["ct_bits"]))
        print(f"n={n:<7} {rows[-1]['scheme']:<28} enc={rows[-1]['enc_ms']:>12.3f} ms  "
              f"agg={rows[-1]['agg_ms']:>10.3f} ms  dec(agg)={rows[-1]['dec_aggregate_ms']:>8.3f} ms  "
              f"ct={rows[-1]['ct_bits']} bits")

with open("experiments/benchmark_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nsaved experiments/benchmark_results.csv   (* = extrapolated from --cap items)")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.2))
    for label, _, _ in SCHEMES:
        pts = [(r["n"], r["enc_ms"]) for r in rows if r["scheme"].rstrip("*") == label]
        plt.plot(*zip(*pts), "o-", label=label)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("number of clients n"); plt.ylabel("total encryption time (ms)")
    plt.title("Encryption time vs n"); plt.legend(); plt.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig("experiments/benchmark_enc.png", dpi=150)
    print("saved experiments/benchmark_enc.png")
except ImportError:
    print("matplotlib not installed: plot skipped")
