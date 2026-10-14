"""Phase 6: privacy-preserving smart-meter aggregation.

Actors
  households : each holds one 4-bit reading (0..15) per hour and encrypts it
  server     : honest-but-curious aggregator; sees only ciphertexts + public key
  utility    : holds the secret key; decrypts ONLY the per-hour aggregates

Goal: utility learns the hourly total; the server learns nothing about
individual households (a daily profile reveals when someone is home/away).
"""
import numpy as np
from . import gf2
from .mceliece_he import generate_keys, encrypt, decrypt, aggregate
from .attack import recover_plaintext


def make_readings(n_households, hours, seed=0):
    """Plausible hourly usage in 0..15: household base level + evening peak + noise."""
    rng = np.random.default_rng(seed)
    base = rng.integers(2, 7, (n_households, 1))
    t = np.arange(hours)
    daily = 4 * np.exp(-((t - 19) ** 2) / 8) + 2 * np.exp(-((t - 7) ** 2) / 6)
    noise = rng.normal(0, 1.2, (n_households, hours))
    return np.clip(np.rint(base + daily + noise), 0, 15).astype(int)


def _bits(v):
    return np.array([(v >> i) & 1 for i in range(4)], dtype=np.uint8)


def _from_bits(b):
    return int(sum(int(bit) << i for i, bit in enumerate(b)))


def run_paper_scheme(readings, seed=0):
    k = generate_keys(seed)
    N, T = readings.shape
    cts = [[encrypt(_bits(v), k.Psi) for v in row] for row in readings]

    # server aggregates each hour (XOR is the only operation the scheme supports)
    agg = [aggregate([cts[h][t] for h in range(N)]) for t in range(T)]
    # utility decrypts the aggregates
    decrypted = np.array([_from_bits(decrypt(a, k)) for a in agg])
    # the server (knows only Psi) attacks every individual ciphertext
    recovered = np.array([[_from_bits(recover_plaintext(cts[h][t], k.Psi))
                           for t in range(T)] for h in range(N)])
    return dict(decrypted=decrypted,
                true_sum=readings.sum(axis=0),
                true_xor=np.bitwise_xor.reduce(readings, axis=0),
                leak_rate=float((recovered == readings).mean()),
                recovered=recovered)


def run_paillier(readings, key_bits=2048, seed=0):
    from phe import paillier
    pub, priv = paillier.generate_paillier_keypair(n_length=key_bits)
    N, T = readings.shape
    cts = [[pub.encrypt(int(v)) for v in row] for row in readings]
    agg = []
    for t in range(T):
        total = cts[0][t]
        for h in range(1, N):
            total = total + cts[h][t]
        agg.append(total)
    decrypted = np.array([priv.decrypt(a) for a in agg])
    # server has no secret key: its best generic move is a blind guess
    rng = np.random.default_rng(seed)
    guess = rng.integers(0, 16, readings.shape)
    return dict(decrypted=decrypted, true_sum=readings.sum(axis=0),
                leak_rate=float((guess == readings).mean()))


def run_bfv(readings, seed=0):
    import tenseal as ts
    ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
    N, T = readings.shape
    cts = [[ts.bfv_vector(ctx, [int(v)]) for v in row] for row in readings]
    agg = []
    for t in range(T):
        total = cts[0][t]
        for h in range(1, N):
            total = total + cts[h][t]
        agg.append(total)
    decrypted = np.array([a.decrypt()[0] for a in agg])
    rng = np.random.default_rng(seed)
    guess = rng.integers(0, 16, readings.shape)
    return dict(decrypted=decrypted, true_sum=readings.sum(axis=0),
                leak_rate=float((guess == readings).mean()))
