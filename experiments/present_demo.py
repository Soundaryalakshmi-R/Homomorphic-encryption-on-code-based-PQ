"""Demo output for the mentor. Run from the project root:

    python -m experiments.present_demo              # pauses between stages (Enter)
    python -m experiments.present_demo --no-pause
    python -m experiments.present_demo --stage 2    # only one stage (1-6)

The script prints results only; the explanation is in report/presentation_guide.md
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import numpy as np
from src import gf2
from src.mceliece_he import (paper_keys, generate_keys, encrypt, decrypt, aggregate,
                             encrypt_noisy, correct_errors)
from src.attack import recover_plaintext, recover_plaintext_noisy
from src.noise import exact_success, empirical_success

ap = argparse.ArgumentParser()
ap.add_argument("--no-pause", action="store_true")
ap.add_argument("--stage", type=int, default=0)
args = ap.parse_args()


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def pause():
    if not args.no_pause:
        input("\n[Enter to continue] ")


def show(name, M):
    M = np.atleast_2d(np.asarray(M))
    pad = " " * (len(name) + 3)
    for i, row in enumerate(M):
        print((name + " = " if i == 0 else pad) + "[" + " ".join(map(str, row)) + "]")


def v(x):
    return "[" + " ".join(str(int(b)) for b in x) + "]"


def stage1():
    banner("STAGE 1: Paper's scheme")
    k = paper_keys()
    show("S", k.S); show("G", k.G); show("P", k.P)
    show("Psi = SGP", k.Psi)
    paper_psi = np.array([[0,1,1,0,1,0,1],[1,0,0,0,1,0,1],[1,0,1,0,1,1,0],[1,0,1,1,0,0,1]])
    print("Psi matches paper Eq.16:", bool((k.Psi == paper_psi).all()))

    x1, x2 = np.array([0,1,0,0]), np.array([1,0,0,0])
    c1, c2 = encrypt(x1, k.Psi), encrypt(x2, k.Psi)
    print("\nEncryption")
    print("x1 =", v(x1), " c1 =", v(c1))
    print("x2 =", v(x2), " c2 =", v(c2))

    phi = aggregate([c1, c2])
    print("\nHomomorphic addition")
    print("Phi = c1 XOR c2 =", v(phi))

    print("\nDecryption of Phi")
    s1 = gf2.mat_mul(phi, gf2.perm_inverse(k.P))
    print("Phi P^-1       =", v(s1))
    print("syndrome       =", v(gf2.mat_mul(s1, k.H)))
    xS = gf2.mat_mul(s1, k.R)
    print("decode (R)     =", v(xS))
    out = gf2.mat_mul(xS, gf2.inverse(k.S))
    print("undo S         =", v(out))

    expected = np.bitwise_xor(x1, x2)
    print("\nx1 XOR x2      =", v(expected))
    print("decrypted      =", v(out), "| correct:", bool((out == expected).all()))

    print("\nRandom keys, n ciphertexts aggregated (50 keys each)")
    for n in (10, 100, 1000):
        ok = 0
        for seed in range(50):
            kk = generate_keys(seed)
            X = np.random.default_rng(seed).integers(0, 2, (n, 4))
            p = aggregate([encrypt(x, kk.Psi) for x in X])
            ok += int((decrypt(p, kk) == np.bitwise_xor.reduce(X, axis=0)).all())
        print(f"n = {n:<5} correct: {ok}/50")


def stage2():
    banner("STAGE 2: Attack on the paper's scheme")
    k = paper_keys()
    Psi = k.Psi
    x1 = np.array([0,1,0,0]); c1 = encrypt(x1, Psi)
    print("Attacker knows only")
    show("Psi", Psi)
    print("c1 =", v(c1))

    aug = np.hstack([Psi.T, c1.reshape(-1, 1)])
    print(); show("[Psi^T | c1^T]", aug)
    R, piv = gf2.rref(aug)
    show("after elimination", R)
    print("pivot columns:", piv, "| rank:", len(piv))
    rec = recover_plaintext(c1, Psi)
    print("\nrecovered x =", v(rec), "| true x1 =", v(x1), "| match:", bool((rec == x1).all()))

    x2 = np.array([1,0,0,0]); c2 = encrypt(x2, Psi)
    phi = aggregate([c1, c2])
    print()
    for name, c, truth in (("c2", c2, x2), ("Phi", phi, np.bitwise_xor(x1, x2))):
        r = recover_plaintext(c, Psi)
        print(f"{name:<4}= {v(c)} -> recovered {v(r)} | true {v(truth)} | match: {bool((r == truth).all())}")

    print("\nEnc([0 1 0 0]) twice:", v(encrypt(x1, Psi)), v(encrypt(x1, Psi)))

    ok = tot = 0
    for seed in range(1000):
        kk = generate_keys(seed)
        for x in np.random.default_rng(seed).integers(0, 2, (10, 4)):
            tot += 1
            ok += int((recover_plaintext(encrypt(x, kk.Psi), kk.Psi) == x).all())
    print(f"\nAttack success over random keys: {ok}/{tot} = {100*ok/tot:.1f}%")


def stage3():
    banner("STAGE 3: Error vector restored")
    k = paper_keys()
    rng = np.random.default_rng(5)
    x = np.array([1,0,1,1])
    c = encrypt_noisy(x, k.Psi, rng)
    e = np.bitwise_xor(c, encrypt(x, k.Psi))
    print("x       =", v(x))
    print("x Psi   =", v(encrypt(x, k.Psi)))
    print("e       =", v(e))
    print("y       =", v(c))
    s1 = gf2.mat_mul(c, gf2.perm_inverse(k.P))
    print("\ny P^-1  =", v(s1), "| syndrome =", v(gf2.mat_mul(s1, k.H)))
    print("corrected =", v(correct_errors(s1, k.H)))
    print("decrypted =", v(decrypt(c, k)), "| correct:", bool((decrypt(c, k) == x).all()))

    for seed in range(200):
        r = np.random.default_rng(seed)
        X = r.integers(0, 2, (3, 4))
        cts = [encrypt_noisy(xx, k.Psi, r) for xx in X]
        es = [np.bitwise_xor(cc, encrypt(xx, k.Psi)) for cc, xx in zip(cts, X)]
        etot = np.bitwise_xor.reduce(es, axis=0)
        want = np.bitwise_xor.reduce(X, axis=0)
        got = decrypt(aggregate(cts), k)
        if etot.sum() >= 2 and not (got == want).all():
            break
    print("\nAggregating 3 noisy ciphertexts")
    for i, ee in enumerate(es, 1):
        print(f"e{i} = {v(ee)}  weight {int(ee.sum())}")
    print(f"aggregate e = {v(etot)}  weight {int(etot.sum())}")
    print("expected XOR of plaintexts =", v(want))
    print("decrypted aggregate        =", v(got), "| correct:", bool((got == want).all()))

    ex = exact_success(30)
    print("\nSuccess rate of decrypting an aggregate of n ciphertexts")
    print(f"{'n':>3} {'exact':>8} {'simulated':>10}")
    for n in (1, 2, 3, 4, 5, 10, 11, 20, 30):
        print(f"{n:>3} {ex[n-1]:>8.3f} {empirical_success(n, 500, seed=n):>10.3f}")

    ok = tot = 0
    r = np.random.default_rng(0)
    for seed in range(300):
        kk = generate_keys(seed)
        for xx in r.integers(0, 2, (10, 4)):
            tot += 1
            ok += int((recover_plaintext_noisy(encrypt_noisy(xx, kk.Psi, r), kk.Psi) == xx).all())
    print(f"\nNearest-codeword attack on noisy ciphertexts: {ok}/{tot} = {100*ok/tot:.1f}%")


def stage4():
    banner("STAGE 4: Benchmark (n = 200 clients)")
    from src.benchmarks import (make_values, bench_paper_scheme, bench_paillier,
                                bench_bfv_per_client, bench_bfv_batched)
    vals = make_values(200)
    rows = [("Paper scheme (insecure)", bench_paper_scheme(vals)),
            ("Paillier-2048 (not PQ)", bench_paillier(vals)),
            ("BFV per-client (PQ)", bench_bfv_per_client(vals)),
            ("BFV batched (PQ)", bench_bfv_batched(vals))]
    print(f"{'scheme':<26}{'encrypt all':>13}{'aggregate':>12}{'decrypt agg':>13}{'ct size':>12}  correct")
    for name, r in rows:
        print(f"{name:<26}{1e3*r['enc']:>10.2f} ms{1e3*r['agg']:>9.2f} ms{1e3*r['dec']:>10.2f} ms"
              f"{r['ct_bits']:>8} bits  {r['correct']}")


def stage5():
    banner("STAGE 5: Smart-meter aggregation (8 households x 12 hours)")
    from src.app_demo import make_readings, run_paper_scheme, run_paillier, run_bfv
    R = make_readings(8, 12, seed=0)
    print("household 0, true usage :", R[0].tolist())
    print("true hourly totals      :", R.sum(axis=0).tolist())
    p = run_paper_scheme(R)
    print("\n[Paper scheme]")
    print("utility decrypts        :", p["decrypted"].tolist())
    print("server's view of hh 0   :", p["recovered"][0].tolist())
    print(f"readings recovered      : {100*p['leak_rate']:.0f}%")
    for name, fn in (("Paillier-2048", run_paillier), ("BFV", run_bfv)):
        r = fn(R)
        print(f"\n[{name}]")
        print("utility decrypts        :", r["decrypted"].tolist())
        print("totals correct          :", bool((r["decrypted"] == r["true_sum"]).all()))


def stage6():
    banner("STAGE 6: Summary")
    print("1. Paper's scheme reproduced: example matches, homomorphic addition correct.")
    print("2. Attack: plaintexts recovered from the public key, 100%.")
    print("3. Error vector restored: homomorphism fails after about one addition.")
    print("4. Benchmarks: Paillier and BFV as fair baselines.")
    print("5. Application: paper's scheme leaks readings; Paillier and BFV do not.")


STAGES = [stage1, stage2, stage3, stage4, stage5, stage6]
if args.stage:
    STAGES[args.stage - 1]()
else:
    for i, s in enumerate(STAGES):
        s()
        if i < len(STAGES) - 1:
            pause()