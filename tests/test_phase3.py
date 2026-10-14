import numpy as np
from src.mceliece_he import paper_keys, generate_keys, encrypt, aggregate
from src.attack import recover_plaintext, is_uniquely_solvable


def test_attack_on_paper_example():
    k = paper_keys()                      # attacker only uses k.Psi
    assert (recover_plaintext([1,0,0,0,1,0,1], k.Psi) == [0,1,0,0]).all()
    assert (recover_plaintext([0,1,1,0,1,0,1], k.Psi) == [1,0,0,0]).all()
    assert (recover_plaintext([1,1,1,0,0,0,0], k.Psi) == [1,1,0,0]).all()


def test_full_rank_means_unique_solution():
    assert is_uniquely_solvable(paper_keys().Psi)


def test_attack_always_succeeds_random_keys():
    for seed in range(100):
        k = generate_keys(seed)
        rng = np.random.default_rng(seed)
        for x in rng.integers(0, 2, (20, 4)):
            assert (recover_plaintext(encrypt(x, k.Psi), k.Psi) == x).all()


def test_attack_recovers_aggregate():
    k = generate_keys(7)
    X = np.random.default_rng(7).integers(0, 2, (50, 4))
    phi = aggregate([encrypt(x, k.Psi) for x in X])
    assert (recover_plaintext(phi, k.Psi) == np.bitwise_xor.reduce(X, axis=0)).all()
