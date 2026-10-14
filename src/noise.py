"""Phase 4: noise growth when n noisy ciphertexts are XOR-aggregated.

Each ciphertext carries a weight-1 error. XOR-ing n of them XORs the errors,
so the aggregate error has weight up to min(n, 7), but Hamming(7,4) corrects
only weight 1.  We measure the decryption success rate empirically and also
compute it exactly with a Markov chain over the 2^7 error patterns.
"""
import numpy as np
from . import gf2
from .mceliece_he import (paper_keys, generate_keys, encrypt_noisy,
                          decrypt, aggregate, correct_errors)


def empirical_success(n, trials, seed=0):
    """Fraction of trials where Dec(XOR of n noisy ciphertexts) == XOR of plaintexts."""
    rng = np.random.default_rng(seed)
    ok = 0
    for t in range(trials):
        k = generate_keys(seed * 100003 + t)
        X = rng.integers(0, 2, (n, 4))
        phi = aggregate([encrypt_noisy(x, k.Psi, rng) for x in X])
        ok += int((decrypt(phi, k) == np.bitwise_xor.reduce(X, axis=0)).all())
    return ok / trials


def _decodes_correctly(pattern_int, k):
    """With plaintext 0, does error pattern (after P^-1) still decode to 0?"""
    e = np.array([(pattern_int >> i) & 1 for i in range(7)], dtype=np.uint8)
    fixed = correct_errors(e, k.H)
    return not gf2.mat_mul(fixed, k.R).any()     # data bits must be all 0


def exact_success(max_n):
    """Exact success probability for n = 1..max_n (list of floats).
    Error positions are uniform and P only permutes them, so the chain is
    the same in the permuted frame. Decoding is linear, so plaintext is irrelevant."""
    k = paper_keys()
    good = np.array([_decodes_correctly(s, k) for s in range(128)])
    T = np.zeros((128, 128))
    for s in range(128):
        for pos in range(7):
            T[s, s ^ (1 << pos)] += 1 / 7
    dist = np.zeros(128); dist[0] = 1.0
    out = []
    for _ in range(max_n):
        dist = dist @ T
        out.append(float(dist[good].sum()))
    return out
