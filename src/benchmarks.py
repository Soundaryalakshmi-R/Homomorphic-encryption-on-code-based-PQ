"""Phase 5: fair benchmark workload for additive homomorphic encryption.

Workload (same for every scheme): n clients each hold one 4-bit value.
Each client encrypts its value; a server aggregates the n ciphertexts;
the key holder decrypts the single aggregate.

Every function returns a dict:
  enc   total seconds to encrypt all n values
  agg   seconds for the server to combine the n ciphertexts
  dec   seconds to decrypt the ONE aggregate ciphertext
  ct_bits  size of one ciphertext in bits
  correct  whether the decrypted aggregate matched the plaintext aggregate
  measured_n  how many items were really run (== n unless capped)
Optional: dec_each  seconds to decrypt all n ciphertexts individually.

NOTE: XOR (ours) and integer sum (Paillier/BFV) are different "additions";
that is inherent to the comparison and should be stated in the report.
"""
import time
import numpy as np
from . import gf2
from .mceliece_he import generate_keys, encrypt, decrypt, aggregate


def _t():
    return time.perf_counter()


def make_values(n, seed=0):
    return np.random.default_rng(seed).integers(0, 16, n)


# ---------- the paper's scheme (insecure: no error vector) ----------
def bench_paper_scheme(values, seed=0):
    n = len(values)
    k = generate_keys(seed)
    X = np.array([[(v >> i) & 1 for i in range(4)] for v in values], dtype=np.uint8)

    t = _t(); cts = [encrypt(x, k.Psi) for x in X]; enc_loop = _t() - t
    t = _t(); C = gf2.mat_mul(X, k.Psi); enc_batch = _t() - t   # vectorised
    t = _t(); phi = aggregate(cts); agg_loop = _t() - t
    t = _t(); phi_b = np.bitwise_xor.reduce(C, axis=0); agg_batch = _t() - t
    t = _t(); out = decrypt(phi, k); dec = _t() - t
    t = _t(); [decrypt(c, k) for c in cts]; dec_each = _t() - t

    expect = np.bitwise_xor.reduce(X, axis=0)
    return dict(enc=enc_loop, enc_batch=enc_batch, agg=agg_loop, agg_batch=agg_batch,
                dec=dec, dec_each=dec_each, ct_bits=7,
                correct=bool((out == expect).all() and (phi_b == phi).all()),
                measured_n=n)


# ---------- Paillier (additive, not post-quantum) ----------
def bench_paillier(values, key_bits=2048):
    from phe import paillier
    t = _t(); pub, priv = paillier.generate_paillier_keypair(n_length=key_bits); keygen = _t() - t
    vals = [int(v) for v in values]
    t = _t(); cts = [pub.encrypt(v) for v in vals]; enc = _t() - t
    t = _t()
    total = cts[0]
    for c in cts[1:]:
        total = total + c
    agg = _t() - t
    t = _t(); out = priv.decrypt(total); dec = _t() - t
    return dict(enc=enc, agg=agg, dec=dec, ct_bits=2 * key_bits, keygen=keygen,
                correct=(out == sum(vals)), measured_n=len(vals))


# ---------- BFV via TenSEAL (lattice-based, post-quantum, leveled FHE) ----------
def _bfv_context():
    import tenseal as ts
    ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
    return ctx


def bench_bfv_per_client(values):
    """One value per ciphertext (each client encrypts alone). Fair to the other rows."""
    import tenseal as ts
    t = _t(); ctx = _bfv_context(); keygen = _t() - t
    vals = [int(v) for v in values]
    t = _t(); cts = [ts.bfv_vector(ctx, [v]) for v in vals]; enc = _t() - t
    t = _t()
    total = cts[0]
    for c in cts[1:]:
        total = total + c
    agg = _t() - t
    t = _t(); out = total.decrypt()[0]; dec = _t() - t
    return dict(enc=enc, agg=agg, dec=dec, keygen=keygen,
                ct_bits=8 * len(cts[0].serialize()),
                correct=(out == sum(vals)), measured_n=len(vals))


def bench_bfv_batched(values):
    """Best case: pack up to 4096 values per ciphertext (SIMD batching)."""
    import tenseal as ts
    ctx = _bfv_context()
    vals = [int(v) for v in values]
    slots = 4096
    chunks = [vals[i:i + slots] for i in range(0, len(vals), slots)]
    chunks = [ch + [0] * (slots - len(ch)) for ch in chunks]   # pad: all vectors must be same size
    t = _t(); cts = [ts.bfv_vector(ctx, ch) for ch in chunks]; enc = _t() - t
    t = _t()
    total = cts[0]
    for c in cts[1:]:
        total = total + c
    agg = _t() - t
    t = _t(); out = sum(total.decrypt()); dec = _t() - t
    return dict(enc=enc, agg=agg, dec=dec, ct_bits=8 * len(cts[0].serialize()),
                correct=(out == sum(vals)), measured_n=len(vals))
