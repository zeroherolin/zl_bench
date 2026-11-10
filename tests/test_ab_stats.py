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
