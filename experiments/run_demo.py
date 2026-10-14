"""Run from the project root:  python -m experiments.run_demo
Options: --households 20 --hours 24 --seed 0"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse, time
import numpy as np
from src.app_demo import make_readings, run_paper_scheme, run_paillier, run_bfv

ap = argparse.ArgumentParser()
ap.add_argument("--households", type=int, default=20)
ap.add_argument("--hours", type=int, default=24)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

R = make_readings(a.households, a.hours, a.seed)
print(f"{a.households} households x {a.hours} hours, readings 0..15")
print("Household 0 true daily profile:", R[0].tolist(), "\n")

results = {}
for name, fn in (("Paper scheme", run_paper_scheme),
                 ("Paillier-2048", run_paillier),
                 ("BFV", run_bfv)):
    t0 = time.perf_counter(); r = fn(R); dt = time.perf_counter() - t0
    results[name] = r
    sum_ok = bool((r["decrypted"] == r["true_sum"]).all())
    print(f"[{name}]  time {dt:.2f}s")
    print(f"  utility's hourly totals correct (equal true SUM)?  {sum_ok}")
    if name == "Paper scheme":
        xor_ok = bool((r["decrypted"] == r["true_xor"]).all())
        print(f"  ...but equal to the XOR of readings?              {xor_ok}  (XOR is not a billing total)")
        print(f"  server recovers individual readings:              {100*r['leak_rate']:.1f}%  (key-less attack)")
        print("  household 0 as seen by the server:", r["recovered"][0].tolist())
    else:
        print(f"  server's success if it can only guess:            {100*r['leak_rate']:.1f}%  (chance = 6.25%)")
    print()
