import numpy as np
import pytest
from src import gf2

# Paper matrices (Eqs. 13-15) used as realistic test data
S = np.array([[1,1,0,1],[1,0,0,1],[0,1,1,1],[1,1,0,0]])
P = np.array([[0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,0,1],
              [1,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,1,0,0]])


def test_add_is_xor():
    assert (gf2.add([1,1,0],[0,1,1]) == [1,0,1]).all()


def test_mat_mul_mod2():
    assert (gf2.mat_mul([[1,1]], [[1],[1]]) == [[0]]).all()


def test_inverse_of_paper_S():
    Si = gf2.inverse(S)
    assert (gf2.mat_mul(S, Si) == gf2.identity(4)).all()
    assert (gf2.mat_mul(Si, S) == gf2.identity(4)).all()


def test_inverse_singular_raises():
    with pytest.raises(ValueError):
        gf2.inverse([[1,1],[1,1]])


def test_random_inverse():
    rng = np.random.default_rng(0)
    done = 0
    while done < 20:
        A = rng.integers(0, 2, (8, 8))
        if gf2.rank(A) == 8:
            assert (gf2.mat_mul(A, gf2.inverse(A)) == gf2.identity(8)).all()
            done += 1


def test_solve_consistent_and_inconsistent():
    A = np.array([[1,0],[0,1],[1,1]])
    x = np.array([1,1])
    b = gf2.mat_mul(A, x.reshape(-1,1)).ravel()
    assert (gf2.mat_mul(A, gf2.solve(A, b).reshape(-1,1)).ravel() == b).all()
    assert gf2.solve(A, [1,0,0]) is None


def test_perm_inverse():
    assert (gf2.mat_mul(P, gf2.perm_inverse(P)) == gf2.identity(7)).all()
    with pytest.raises(ValueError):
        gf2.perm_inverse(S)


def test_rank():
    assert gf2.rank(gf2.identity(5)) == 5
    assert gf2.rank([[1,1],[1,1]]) == 1
