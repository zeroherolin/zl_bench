import math

import pytest

import ab_stats

X = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]
Y = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29]


@pytest.mark.parametrize("x, y, alternative, u, p", [
    # exact null distribution: n <= 8 on one side, no ties (scipy.stats.mannwhitneyu)
    ([1, 2, 3], [4, 5, 6, 7], "two-sided", 0.0, 0.05714285714285714),
    ([1, 2, 3], [4, 5, 6, 7], "less", 0.0, 0.02857142857142857),
    ([1, 2, 3], [4, 5, 6, 7], "greater", 0.0, 1.0),
    ([1, 4, 6, 9, 10, 12], [2, 3, 5, 7, 8, 11], "two-sided", 21.0, 0.6991341991341992),
    # normal approximation: n > 8 on both sides, continuity correction
    (X, Y, "greater", 58.0, 0.06645972909265946),
    (X, Y, "two-sided", 58.0, 0.13291945818531892),
    # ties: normal approximation with tie correction even for small samples
    ([1, 2, 2, 3], [2, 3, 4, 4, 5], "two-sided", 2.5, 0.07854585095119067),
])
def test_mann_whitney_matches_scipy(x, y, alternative, u, p):
    got_u, got_p = ab_stats.mann_whitney_u(x, y, alternative)
    assert got_u == u
    assert got_p == pytest.approx(p, rel=1e-12)


def test_mann_whitney_rejects_bad_input():
    with pytest.raises(ValueError):
        ab_stats.mann_whitney_u([], [1.0])
    with pytest.raises(ValueError):
        ab_stats.mann_whitney_u([1.0], [2.0], "sideways")


def test_exact_distribution_sums_to_all_arrangements():
    counts = ab_stats._u_counts(4, 6)
    assert sum(counts) == math.comb(10, 4)
    assert counts == counts[::-1]  # symmetric around m*n/2


def _shifted(base, factor):
    return [x * factor for x in base]


BASE = [1.000, 1.002, 0.998, 1.001, 0.999, 1.003]


def test_compare_verdicts():
    slow_b = ab_stats.compare(BASE, _shifted(BASE, 1.05))
    assert slow_b.verdict == "slower" and slow_b.gap == pytest.approx(1 / 1.05 - 1)
    fast_b = ab_stats.compare(BASE, _shifted(BASE, 0.9))
    assert fast_b.verdict == "faster" and fast_b.p < ab_stats.ALPHA
    assert ab_stats.compare(BASE, _shifted(BASE, 1.001)).verdict == "same"  # within threshold
    assert ab_stats.compare(BASE[:5], _shifted(BASE[:5], 0.9)).verdict == "undecided"  # few rounds


def test_compare_overlapping_samples():
    a = [1.00, 1.10, 1.02, 1.12, 1.01, 1.11]
    b = [1.03, 1.09, 1.00, 1.13, 1.04, 1.08]
    r = ab_stats.compare(a, b)
    assert r.p >= ab_stats.CLEAR and r.verdict == "same"
    # A shift the unpaired test leaves in between, but every round points the same way:
    # the pairing decides it (a round runs both sides back to back on one GPU).
    a = [1.00, 1.04, 1.02, 1.06, 1.01, 1.05]
    b = [0.99, 1.01, 0.98, 1.03, 0.97, 1.00]
    r = ab_stats.compare(a, b)
    assert ab_stats.ALPHA <= r.p < ab_stats.CLEAR
    assert r.p_paired < ab_stats.ALPHA_PAIRED and r.verdict == "faster"
    # a gap of the same size with one round disagreeing is not decided by the pairing
    a2 = [1.0340, 1.0661, 1.0099, 1.0179, 1.0502, 1.0758]
    b2 = [1.0162, 1.0017, 1.0481, 0.9737, 1.0387, 0.9932]
    r2 = ab_stats.compare(a2, b2)
    assert r2.gap > ab_stats.THRESHOLD and r2.p_paired >= ab_stats.ALPHA_PAIRED
    assert r2.verdict == "undecided"
    # agreeing rounds below --threshold stay "same": the pairing adds evidence, not reach
    a3 = [1.0010, 1.0012, 1.0011, 1.0013, 1.0010, 1.0012]
    b3 = [1.0000, 1.0002, 1.0001, 1.0003, 1.0000, 1.0002]
    assert ab_stats.compare(a3, b3).verdict == "same"


# Six rounds on GPUs a few percent apart; B is 2% faster in every one of them.
GPU_SPREAD = [1.000, 1.030, 0.985, 1.040, 1.012, 0.970]


def test_pairing_decides_what_the_gpu_spread_hides():
    # The rank test pools the rounds, so the spread between GPUs buries the effect well
    # past --clear. The pairing is consulted before --clear can call the row "same".
    r = ab_stats.compare(GPU_SPREAD, _shifted(GPU_SPREAD, 0.98))
    assert r.p >= ab_stats.CLEAR
    assert r.p_paired < ab_stats.ALPHA_PAIRED and r.verdict == "faster"
    assert r.basis == "pairing"
    r = ab_stats.compare(_shifted(GPU_SPREAD, 0.98), GPU_SPREAD)
    assert r.verdict == "slower"
    # separated samples are shown by the rank test, whatever the pairing says
    assert ab_stats.compare(BASE, _shifted(BASE, 0.9)).basis == "rank"
    assert ab_stats.compare(BASE, _shifted(BASE, 1.001)).basis == ""


def test_pairing_needs_min_rounds_too():
    # --min-rounds is the floor for every verdict, whichever test would give it
    a = GPU_SPREAD
    assert ab_stats.compare(a, _shifted(a, 0.98), min_rounds=8).verdict == "undecided"
    assert ab_stats.compare(a[:5], _shifted(a[:5], 0.98), alpha_paired=0.1).verdict == "undecided"


def test_pairing_needs_every_round():
    # five rounds of six agree: the sign test is nowhere near, and the rank test decides
    b = _shifted(GPU_SPREAD, 0.98)
    b[3] = GPU_SPREAD[3] * 1.01
    r = ab_stats.compare(GPU_SPREAD, b)
    assert r.p_paired == pytest.approx(0.21875) and r.verdict == "same"
    # a tied round has no direction, so the rounds no longer all agree
    b = _shifted(GPU_SPREAD, 0.98)
    b[3] = GPU_SPREAD[3]
    assert not ab_stats.pairs_agree(GPU_SPREAD, b)
    assert ab_stats.compare(GPU_SPREAD, b).verdict == "same"


def test_unpaired_samples_use_the_rank_test_alone():
    r = ab_stats.compare(BASE + [1.004], _shifted(BASE, 0.9))
    assert math.isnan(r.p_paired) and r.verdict == "faster"
    r = ab_stats.compare(GPU_SPREAD + [1.0], _shifted(GPU_SPREAD, 0.98))
    assert math.isnan(r.p_paired) and r.verdict == "same"


@pytest.mark.parametrize("a, b, alternative, wins, n, p", [
    # scipy.stats.binomtest(k, n, 0.5, alternative=...)
    ([2] * 6, [1] * 6, "two-sided", 6, 6, 0.03125),
    ([2] * 6, [1] * 6, "greater", 6, 6, 0.015625),
    ([2] * 6, [1] * 6, "less", 6, 6, 1.0),
    ([1] * 5, [2] * 5, "less", 0, 5, 0.03125),
    ([2, 2, 2, 2, 2, 1], [1, 1, 1, 1, 1, 2], "two-sided", 5, 6, 0.21875),
    ([2, 2, 2, 2, 2, 1], [1, 1, 1, 1, 1, 2], "greater", 5, 6, 0.109375),
    ([2, 2, 2, 1, 1, 1], [1, 1, 1, 2, 2, 2], "two-sided", 3, 6, 1.0),
    ([2, 2, 1, 3], [1, 1, 1, 3], "two-sided", 2, 2, 0.5),  # ties are left out
    ([1, 1], [1, 1], "two-sided", 0, 0, 1.0),
])
def test_sign_test(a, b, alternative, wins, n, p):
    assert ab_stats.sign_test(a, b, alternative) == (wins, n, pytest.approx(p, rel=1e-12))


def test_sign_test_rejects_bad_input():
    with pytest.raises(ValueError):
        ab_stats.sign_test([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        ab_stats.sign_test([1.0], [2.0], "sideways")


def test_pairs_agree_follows_the_alternative():
    assert ab_stats.pairs_agree([2, 3], [1, 2])
    assert ab_stats.pairs_agree([2, 3], [1, 2], "greater")
    assert not ab_stats.pairs_agree([2, 3], [1, 2], "less")
    assert not ab_stats.pairs_agree([2, 1], [1, 2])
    assert not ab_stats.pairs_agree([], [])
    assert not ab_stats.pairs_agree([2, 3, 4], [1, 2])


def test_smallest_p_is_what_n_rounds_can_reach():
    # n separated samples per side: 1 / C(2n, n); it is below 0.05 only from 4 rounds on
    assert ab_stats.smallest_p(3) == pytest.approx(0.05)
    assert ab_stats.smallest_p(4) == pytest.approx(1 / 70)
    assert ab_stats.smallest_p(0) == 1.0
    for n in range(1, 9):
        x, y = [10.0 + i for i in range(n)], [1.0 + i for i in range(n)]
        reached = min(ab_stats.mann_whitney_u(x, y, "greater")[1], ab_stats.sign_test(x, y, "greater")[2])
        assert ab_stats.smallest_p(n) == pytest.approx(reached)


def test_samples_that_are_all_the_same_value_say_nothing():
    # scipy gives nan here; a p of 0 would read as the strongest evidence there is
    assert ab_stats.mann_whitney_u([1.0] * 9, [1.0] * 9)[1] == 1.0
    assert ab_stats.compare([1.0] * 9, [1.0] * 9).verdict == "same"
