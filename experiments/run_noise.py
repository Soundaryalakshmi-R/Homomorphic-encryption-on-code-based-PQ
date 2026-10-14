"""Run from the project root:  python -m experiments.run_noise"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import numpy as np
from src.noise import empirical_success, exact_success
from src.mceliece_he import generate_keys, encrypt_noisy
from src.attack import recover_plaintext_noisy

MAX_N, TRIALS = 30, 1000
exact = exact_success(MAX_N)
rows = []
print(f"{'n':>3} {'empirical':>10} {'exact':>8}")
for n in range(1, MAX_N + 1):
    emp = empirical_success(n, TRIALS, seed=n)
    rows.append((n, emp, exact[n - 1]))
    print(f"{n:>3} {emp:>10.3f} {exact[n-1]:>8.3f}")

with open("experiments/noise_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["n", "empirical", "exact"]); w.writerows(rows)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = [r[0] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(ns, [r[1] for r in rows], "o", label="empirical")
    plt.plot(ns, [r[2] for r in rows], "-", label="exact (Markov chain)")
    plt.xlabel("number of aggregated ciphertexts n")
    plt.ylabel("decryption success rate")
    plt.title("Noise growth under XOR aggregation, Hamming(7,4)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("experiments/noise_vs_n.png", dpi=150)
    print("saved experiments/noise_vs_n.png")
except ImportError:
    print("matplotlib not installed: skipped plot (pip install matplotlib)")

# Is the repaired scheme secure at this size? No: brute-force nearest codeword.
rng = np.random.default_rng(0)
ok = tot = 0
for seed in range(500):
    k = generate_keys(seed)
    for x in rng.integers(0, 2, (10, 4)):
        tot += 1
        ok += int((recover_plaintext_noisy(encrypt_noisy(x, k.Psi, rng), k.Psi) == x).all())
print(f"\nKey-less attack on repaired scheme (single ciphertext): {ok}/{tot} = {100*ok/tot:.1f}%")
