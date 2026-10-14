import numpy as np
from src.mceliece_he import generate_keys, encrypt, encrypt_noisy, decrypt
from src.attack import recover_plaintext_noisy
from src.noise import empirical_success, exact_success


def test_single_noisy_ciphertext_decrypts():
    rng = np.random.default_rng(0)
    for seed in range(100):
        k = generate_keys(seed)
        for x in rng.integers(0, 2, (10, 4)):
            assert (decrypt(encrypt_noisy(x, k.Psi, rng), k) == x).all()


def test_exact_n1_is_one_and_matches_empirical():
    ex = exact_success(8)
    assert ex[0] == 1.0
    for n in (2, 3, 5):
        emp = empirical_success(n, trials=3000, seed=n)
        assert abs(emp - ex[n - 1]) < 0.04


def test_success_collapses_for_large_n():
    assert exact_success(60)[-1] < 0.15


def test_nearest_codeword_attack_breaks_single_noisy_ciphertext():
    rng = np.random.default_rng(1)
    k = generate_keys(1)
    x = np.array([1, 0, 1, 1])
    c = encrypt_noisy(x, k.Psi, rng)
    assert (recover_plaintext_noisy(c, k.Psi) == x).all()
