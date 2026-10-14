import numpy as np
from src.app_demo import make_readings, run_paper_scheme, run_paillier, run_bfv


def test_working_schemes_give_correct_totals():
    R = make_readings(6, 5, seed=1)
    assert (run_paillier(R, key_bits=1024)["decrypted"] == R.sum(axis=0)).all()
    assert (run_bfv(R)["decrypted"] == R.sum(axis=0)).all()


def test_paper_scheme_leaks_everything_and_is_not_a_sum():
    R = make_readings(8, 6, seed=2)
    r = run_paper_scheme(R)
    assert r["leak_rate"] == 1.0
    assert (r["decrypted"] == r["true_xor"]).all()
    assert not (r["decrypted"] == r["true_sum"]).all()
