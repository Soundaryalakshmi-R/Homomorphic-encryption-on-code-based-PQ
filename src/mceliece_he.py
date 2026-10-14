"""Phase 2: the paper's scheme (Chen, 'Homomorphic Encryption Based on PQC').

Everything is over GF(2). Row-vector convention: ciphertext c = x @ Psi.
NOTE: this reproduces the paper faithfully, including its flaw (no error
vector e in Enc), which is attacked in Phase 3.
"""
from dataclasses import dataclass
from functools import reduce
import numpy as np
from . import gf2


@dataclass
class Keys:
    S: np.ndarray      # 4x4 scrambler (private)
    G: np.ndarray      # 4x7 generator matrix (private)
    P: np.ndarray      # 7x7 permutation (private)
    H: np.ndarray      # 7x3 parity-check, rows = binary of 1..7 (private)
    R: np.ndarray      # 7x4 decoder, picks the 4 data bits out of a codeword (private)
    Psi: np.ndarray    # 4x7 public key = S G P (public)


def paper_keys() -> Keys:
    """Matrices from Eqs. (13)-(18) of the paper."""
    S = np.array([[1,1,0,1],[1,0,0,1],[0,1,1,1],[1,1,0,0]], dtype=np.uint8)
    G = np.array([[1,1,1,0,0,0,0],
                  [1,0,0,1,1,0,0],
                  [0,1,0,1,0,1,0],
                  [1,1,0,1,0,0,1]], dtype=np.uint8)
    P = np.array([[0,1,0,0,0,0,0],
                  [0,0,0,1,0,0,0],
                  [0,0,0,0,0,0,1],
                  [1,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,1,0,0]], dtype=np.uint8)
    H = np.array([[1,0,0],[0,1,0],[1,1,0],[0,0,1],
                  [1,0,1],[0,1,1],[1,1,1]], dtype=np.uint8)
    R = np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,0],
                  [0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.uint8)
    Psi = gf2.mat_mul(gf2.mat_mul(S, G), P)          # Eq. (1)
    return Keys(S, G, P, H, R, Psi)


def encrypt(x, Psi):
    """Eq. (11): c = x Psi  (no error vector in the paper's version)."""
    return gf2.mat_mul(x, Psi)


def correct_errors(word, H):
    """Hamming(7,4) single-error correction using syndrome word @ H.
    Row i of H is the syndrome produced by an error in position i."""
    word = gf2.to_gf2(word).copy()
    syndrome = gf2.mat_mul(word, H)
    if syndrome.any():
        for i, row in enumerate(H):
            if (row == syndrome).all():
                word[i] ^= 1
                break
    return word


def decrypt(c, keys: Keys, correct=True):
    """Eqs. (3)-(6): undo P, fix errors with H, decode with R, undo S."""
    c = gf2.to_gf2(c)
    xSG = gf2.mat_mul(c, gf2.perm_inverse(keys.P))        # Eq. (3)
    if correct:
        xSG = correct_errors(xSG, keys.H)                 # Eq. (4)
    xS = gf2.mat_mul(xSG, keys.R)                         # Eq. (5)
    return gf2.mat_mul(xS, gf2.inverse(keys.S))           # Eq. (6)


def aggregate(ciphertexts):
    """Eq. (9)/(12): XOR of all ciphertexts."""
    return reduce(gf2.add, ciphertexts)


# ---- random key generation (added after Phase 2) ----
def random_invertible(n, rng):
    while True:
        S = rng.integers(0, 2, (n, n)).astype(np.uint8)
        if gf2.rank(S) == n:
            return S


def random_permutation(n, rng):
    return np.eye(n, dtype=np.uint8)[rng.permutation(n)]


def generate_keys(seed=None) -> Keys:
    """Random S and P; G, H, R stay fixed (Hamming(7,4))."""
    rng = np.random.default_rng(seed)
    base = paper_keys()
    S = random_invertible(4, rng)
    P = random_permutation(7, rng)
    Psi = gf2.mat_mul(gf2.mat_mul(S, base.G), P)
    return Keys(S, base.G, P, base.H, base.R, Psi)


# ---- Phase 4: error vector restored ----
def encrypt_noisy(x, Psi, rng, weight=1):
    """Proper McEliece-style encryption: y = x Psi + e, wt(e) = weight.
    Hamming(7,4) corrects at most weight 1."""
    n = gf2.to_gf2(Psi).shape[1]
    e = np.zeros(n, dtype=np.uint8)
    e[rng.choice(n, size=weight, replace=False)] = 1
    return gf2.add(gf2.mat_mul(x, Psi), e)