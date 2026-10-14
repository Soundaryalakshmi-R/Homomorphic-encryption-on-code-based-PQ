"""GF(2) utilities: all arithmetic is mod 2 on uint8 numpy arrays."""
import numpy as np


def to_gf2(A):
    """Convert anything array-like into a uint8 array with entries in {0,1}."""
    return (np.asarray(A, dtype=np.int64) % 2).astype(np.uint8)


def identity(n):
    return np.eye(n, dtype=np.uint8)


def add(A, B):
    """Matrix/vector addition over GF(2) (element-wise XOR)."""
    return np.bitwise_xor(to_gf2(A), to_gf2(B))


def mat_mul(A, B):
    """Matrix product over GF(2). Works for vector @ matrix too."""
    A, B = to_gf2(A), to_gf2(B)
    return ((A.astype(np.int64) @ B.astype(np.int64)) % 2).astype(np.uint8)


def rref(A):
    """Reduced row echelon form over GF(2).
    Returns (R, pivot_columns)."""
    R = to_gf2(A).copy()
    rows, cols = R.shape
    pivots, r = [], 0
    for c in range(cols):
        if r >= rows:
            break
        hit = np.nonzero(R[r:, c])[0]
        if hit.size == 0:
            continue
        p = r + hit[0]
        if p != r:
            R[[r, p]] = R[[p, r]]
        for i in range(rows):
            if i != r and R[i, c]:
                R[i] ^= R[r]
        pivots.append(c)
        r += 1
    return R, pivots


def rank(A):
    return len(rref(A)[1])


def inverse(A):
    """Inverse of a square matrix over GF(2). Raises ValueError if singular."""
    A = to_gf2(A)
    n, m = A.shape
    if n != m:
        raise ValueError("matrix must be square")
    R, piv = rref(np.hstack([A, identity(n)]))
    if piv[:n] != list(range(n)):
        raise ValueError("matrix is singular over GF(2)")
    return R[:, n:]


def solve(A, b):
    """Solve A x = b over GF(2) for a 1-D b.
    Returns one solution (free variables set to 0), or None if inconsistent."""
    A = to_gf2(A)
    b = to_gf2(b).reshape(-1, 1)
    n_cols = A.shape[1]
    R, piv = rref(np.hstack([A, b]))
    if n_cols in piv:          # pivot in augmented column -> no solution
        return None
    x = np.zeros(n_cols, dtype=np.uint8)
    for row, c in enumerate(piv):
        x[c] = R[row, -1]
    return x


def is_permutation_matrix(P):
    P = to_gf2(P)
    return (P.ndim == 2 and P.shape[0] == P.shape[1]
            and (P.sum(axis=0) == 1).all() and (P.sum(axis=1) == 1).all())


def perm_inverse(P):
    """Inverse of a permutation matrix is its transpose."""
    if not is_permutation_matrix(P):
        raise ValueError("not a permutation matrix")
    return to_gf2(P).T.copy()
