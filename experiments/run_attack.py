"""Run from the project root:  python -m experiments.run_attack"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import numpy as np
from src.mceliece_he import paper_keys, generate_keys, encrypt, aggregate
from src.attack import recover_plaintext

# 1) Paper example
k = paper_keys()
x1, x2 = np.array([0, 1, 0, 0]), np.array([1, 0, 0, 0])
c1, c2 = encrypt(x1, k.Psi), encrypt(x2, k.Psi)
print("Paper example (attacker knows only Psi):")
print("  c1 =", c1, "-> recovered x1 =", recover_plaintext(c1, k.Psi))
print("  c2 =", c2, "-> recovered x2 =", recover_plaintext(c2, k.Psi))
print("  sum =", aggregate([c1, c2]), "-> recovered =",
      recover_plaintext(aggregate([c1, c2]), k.Psi))

# 2) Success rate over many random keys / plaintexts
trials, ok = 0, 0
t0 = time.perf_counter()
for seed in range(1000):
    kk = generate_keys(seed)
    for x in np.random.default_rng(seed).integers(0, 2, (10, 4)):
        trials += 1
        ok += int((recover_plaintext(encrypt(x, kk.Psi), kk.Psi) == x).all())
dt = time.perf_counter() - t0
print(f"\nAttack success: {ok}/{trials} = {100*ok/trials:.1f}%")
print(f"Avg attack time: {1000*dt/trials:.3f} ms per ciphertext (no private key used)")
