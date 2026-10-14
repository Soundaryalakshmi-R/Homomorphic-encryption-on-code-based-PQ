"""Phase 3: key-less plaintext recovery against the paper's scheme.

The paper's encryption is c = x Psi (no error vector), which is a linear
system over GF(2). Given only the PUBLIC key Psi and a ciphertext c, solve
x Psi = c  <=>  Psi^T x^T = c^T  by Gaussian elimination. No private key
(S, G, P, H, R) is used anywhere in this file.
"""
from . import gf2


def recover_plaintext(c, Psi):
    """Return x such that x Psi = c, or None if c is not a valid ciphertext."""
    return gf2.solve(gf2.to_gf2(Psi).T, c)


def is_uniquely_solvable(Psi):
    """x is unique iff Psi has full row rank."""
    return gf2.rank(Psi) == gf2.to_gf2(Psi).shape[0]


def recover_plaintext_noisy(c, Psi):
    """Attack on the REPAIRED scheme (Phase 4): nearest-codeword search.
    The code has only 2^4 = 16 codewords, so try them all and keep the closest.
    Uses only the public key Psi. Works whenever wt(e) <= 1 (perfect code)."""
    import numpy as np
    from itertools import product
    c = gf2.to_gf2(c)
    best, best_d = None, None
    for bits in product([0, 1], repeat=gf2.to_gf2(Psi).shape[0]):
        x = np.array(bits, dtype=np.uint8)
        d = int(np.bitwise_xor(gf2.mat_mul(x, Psi), c).sum())
        if best_d is None or d < best_d:
            best, best_d = x, d
    return best