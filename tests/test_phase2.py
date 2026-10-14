import numpy as np
from src import gf2
from src.mceliece_he import paper_keys, encrypt, decrypt, aggregate


def test_psi_matches_eq16():
    expected = np.array([[0,1,1,0,1,0,1],
                         [1,0,0,0,1,0,1],
                         [1,0,1,0,1,1,0],
                         [1,0,1,1,0,0,1]])
    assert (paper_keys().Psi == expected).all()


def test_G_and_H_are_consistent():
    k = paper_keys()
    assert not gf2.mat_mul(k.G, k.H).any()        # G H = 0
    assert (gf2.mat_mul(k.G, k.R) == gf2.identity(4)).all()   # G R = I


def test_paper_example():
    k = paper_keys()
    x1, x2 = np.array([0,1,0,0]), np.array([1,0,0,0])
    c1, c2 = encrypt(x1, k.Psi), encrypt(x2, k.Psi)
    assert (c1 == [1,0,0,0,1,0,1]).all()          # Eq. (19)
    assert (c2 == [0,1,1,0,1,0,1]).all()          # Eq. (20)
    phi = aggregate([c1, c2])
    assert (phi == [1,1,1,0,0,0,0]).all()         # Eq. (21)
    assert (decrypt(phi, k) == [1,1,0,0]).all()   # Eqs. (22)-(23)


def test_single_roundtrip_all_plaintexts():
    k = paper_keys()
    for v in range(16):
        x = np.array([(v >> i) & 1 for i in range(4)])
        assert (decrypt(encrypt(x, k.Psi), k) == x).all()


def test_random_additive_homomorphism():
    k = paper_keys()
    rng = np.random.default_rng(1)
    for n in (2, 10, 100, 1000):
        X = rng.integers(0, 2, (n, 4))
        phi = aggregate([encrypt(x, k.Psi) for x in X])
        theta = np.bitwise_xor.reduce(X, axis=0)
        assert (decrypt(phi, k) == theta).all()


def test_single_bit_error_is_corrected():
    k = paper_keys()
    x = np.array([1,0,1,1])
    c = encrypt(x, k.Psi)
    for pos in range(7):
        noisy = c.copy(); noisy[pos] ^= 1
        assert (decrypt(noisy, k) == x).all()


from src.mceliece_he import generate_keys

def test_random_keys_roundtrip_and_homomorphism():
    for seed in range(50):
        k = generate_keys(seed)
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 2, (20, 4))
        phi = aggregate([encrypt(x, k.Psi) for x in X])
        assert (decrypt(phi, k) == np.bitwise_xor.reduce(X, axis=0)).all()