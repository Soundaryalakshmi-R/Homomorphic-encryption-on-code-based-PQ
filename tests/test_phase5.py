from src.benchmarks import (make_values, bench_paper_scheme, bench_paillier,
                            bench_bfv_per_client, bench_bfv_batched)


def test_all_schemes_aggregate_correctly():
    v = make_values(30, seed=3)
    for fn in (bench_paper_scheme, bench_bfv_per_client, bench_bfv_batched):
        assert fn(v)["correct"]
    assert bench_paillier(v, key_bits=1024)["correct"]
