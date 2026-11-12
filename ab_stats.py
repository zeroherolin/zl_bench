"""Statistics of a repeated-measures A/B: per-row verdict from the samples of both sides.

Each round of an A/B (ab_run.py) measures every row once per side, so a row has one
latency sample per round and side. A row's verdict compares the two samples:

  gap      = median(A) / median(B) - 1   (> 0: B is faster)
  p        = two-sided Mann-Whitney U test of A against B
  p_paired = two-sided sign test over the rounds, each a pair (A, B) on one GPU
  verdict  = "undecided"  fewer than min_rounds samples on either side
             "same"       |gap| <= threshold
             "faster"     |gap| > threshold and a difference is shown, B faster
             "slower"     |gap| > threshold and a difference is shown, B slower
             "same"       no shift detectable (p >= clear)
             "undecided"  otherwise: more rounds are needed

A difference is shown by either test: the rank test (p < alpha), or the pairing, when
every round points the same way and p_paired < alpha_paired. The two answer different
noise. The rank test pools the rounds, so whatever the two runs of a round share (the
GPU the round landed on, slow drift) counts as noise: over GPUs a few percent apart, a
2% effect present in every round stays far from p < alpha. A pair cancels what its two
runs share, at the price of using only the direction of each round.

The Mann-Whitney test makes no assumption about the latency distribution (GPU timings
are skewed and sometimes bimodal) and matches scipy.stats.mannwhitneyu with its
defaults: the exact null distribution when either sample has at most 8 values and there
are no ties, else the normal approximation with tie and continuity correction. Pure
Python, so the tools need no scipy.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from functools import lru_cache
from typing import NamedTuple, Sequence, Tuple

MIN_ROUNDS = 6
THRESHOLD = 0.005
ALPHA = 0.01
CLEAR = 0.10
# Sign test over the paired rounds. Six agreeing rounds are p = 2 / 2**6 = 0.031, the
# only outcome of MIN_ROUNDS rounds below 0.05; five rounds cannot get there (0.0625).
ALPHA_PAIRED = 0.05

VERDICTS = ("faster", "slower", "same", "undecided")


def _ranks(values: Sequence[float]) -> list:
    """1-based ranks, ties get the average rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


@lru_cache(maxsize=None)
def _u_counts(m: int, n: int) -> Tuple[int, ...]:
    """Number of arrangements of m + n distinct values giving each U = 0 .. m*n."""
    if m == 0 or n == 0:
        return (1,)
    # A sample's largest value either belongs to the first sample (it beats all n of the
    # second: U grows by n) or to the second (U unchanged).
    a, b = _u_counts(m - 1, n), _u_counts(m, n - 1)
    out = [0] * (m * n + 1)
    for u, c in enumerate(a):
        out[u + n] += c
    for u, c in enumerate(b):
        out[u] += c
    return tuple(out)


def _exact_sf(u: int, m: int, n: int) -> float:
    """P(U >= u) under the null hypothesis."""
    counts = _u_counts(min(m, n), max(m, n))
    return sum(counts[max(u, 0):]) / sum(counts)


def mann_whitney_u(x: Sequence[float], y: Sequence[float],
                   alternative: str = "two-sided") -> Tuple[float, float]:
    """(U of x, p-value). alternative: "two-sided", "greater" (x tends to be larger) or
    "less"."""
    n1, n2 = len(x), len(y)
    if not n1 or not n2:
        raise ValueError("both samples need at least one value")
    combined = list(x) + list(y)
    ranks = _ranks(combined)
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    if alternative == "two-sided":
        u, factor = max(u1, u2), 2.0
    elif alternative == "greater":
        u, factor = u1, 1.0
    elif alternative == "less":
        u, factor = u2, 1.0
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    ties = [c for c in Counter(combined).values() if c > 1]
    if (n1 <= 8 or n2 <= 8) and not ties:
        p = _exact_sf(int(round(u)), n1, n2)
    else:
        n = n1 + n2
        tie_term = sum(t ** 3 - t for t in ties)
        s = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1))))
        if s == 0:
            return u1, 1.0  # every value is the same: the samples say nothing
        z = (u - n1 * n2 / 2 - 0.5) / s
        p = 0.5 * math.erfc(z / math.sqrt(2))
    return u1, min(1.0, max(0.0, p * factor))


def sign_test(a: Sequence[float], b: Sequence[float],
              alternative: str = "two-sided") -> Tuple[int, int, float]:
    """Exact sign test over paired samples: (pairs with a > b, decisive pairs, p).

    a[i] and b[i] are the two sides of round i. Pairs with a == b carry no direction and
    are left out, as usual. alternative: "two-sided", "greater" (a tends to be larger,
    i.e. B is faster) or "less".
    """
    if len(a) != len(b):
        raise ValueError("the sign test needs one b for every a")
    wins = sum(1 for x, y in zip(a, b, strict=True) if x > y)
    n = wins + sum(1 for x, y in zip(a, b, strict=True) if x < y)
    if alternative == "two-sided":
        k, factor = max(wins, n - wins), 2.0
    elif alternative == "greater":
        k, factor = wins, 1.0
    elif alternative == "less":
        k, factor = n - wins, 1.0
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n  # P(at least k of n)
    return wins, n, min(1.0, factor * tail)


def pairs_agree(a: Sequence[float], b: Sequence[float], alternative: str = "two-sided") -> bool:
    """Every round points the same way (for "greater": a > b in every round, for "less":
    a < b). A round with a == b, or samples that do not pair up, break the agreement."""
    if not a or len(a) != len(b):
        return False
    wins, n, _ = sign_test(a, b)
    if n != len(a):
        return False
    return {"two-sided": wins in (0, n), "greater": wins == n, "less": wins == 0}[alternative]


def pairing_shows(a: Sequence[float], b: Sequence[float], alpha: float,
                  alternative: str = "two-sided") -> bool:
    """The pairing alone shows a difference: every round agrees and the sign test is
    below alpha. Asking for every round, not just for enough of them, keeps a row that
    flips between two speeds from being decided by a lucky majority."""
    return pairs_agree(a, b, alternative) and sign_test(a, b, alternative)[2] < alpha


def smallest_p(n: int) -> float:
    """The smallest one-sided p that n rounds can give, by either test: n fully
    separated samples per side for the rank test, n agreeing rounds for the sign test.
    A level at or below it cannot be reached, however clear the difference."""
    return min(1 / math.comb(2 * n, n), 0.5 ** n) if n > 0 else 1.0


class RowStats(NamedTuple):
    n_a: int
    n_b: int
    median_a: float
    median_b: float
    gap: float  # median_a / median_b - 1, > 0: B faster
    p: float  # two-sided Mann-Whitney; nan below two samples per side
    verdict: str
    p_paired: float = math.nan  # two-sided sign test over the rounds; nan if they do not pair up
    basis: str = ""  # what showed a faster/slower verdict: "rank" or "pairing"


def compare(a: Sequence[float], b: Sequence[float], *, min_rounds: int = MIN_ROUNDS,
            threshold: float = THRESHOLD, alpha: float = ALPHA,
            clear: float = CLEAR, alpha_paired: float = ALPHA_PAIRED) -> RowStats:
    """Verdict of one row from its A and B latency samples (one per round).

    With as many A as B samples, a[i] and b[i] must be the two sides of one round: the
    pairing is then part of the rule. Samples of unequal length are judged by the rank
    test alone.
    """
    if not a or not b:
        raise ValueError("both sides need at least one sample")
    ma, mb = statistics.median(a), statistics.median(b)
    gap = ma / mb - 1
    paired = len(a) == len(b)
    p = mann_whitney_u(a, b)[1] if min(len(a), len(b)) >= 2 else math.nan
    # Rounds that all agree put the two medians in that order too, so the sign of the
    # gap is the direction of the rounds.
    direction = "faster" if gap > 0 else "slower"
    basis = ""
    if min(len(a), len(b)) < min_rounds:
        verdict = "undecided"
    elif abs(gap) <= threshold:
        verdict = "same"
    elif p < alpha:
        verdict, basis = direction, "rank"
    elif paired and pairing_shows(a, b, alpha_paired):
        verdict, basis = direction, "pairing"
    elif p >= clear:
        verdict = "same"
    else:
        verdict = "undecided"
    return RowStats(len(a), len(b), ma, mb, gap, p, verdict,
                    sign_test(a, b)[2] if paired else math.nan, basis)


def geomean(values: Sequence[float]) -> float:
    return math.exp(sum(map(math.log, values)) / len(values))
