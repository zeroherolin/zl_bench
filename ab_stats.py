"""Statistics of a repeated-measures A/B: per-row verdict from the samples of both sides.

Each round of an A/B (ab_run.py) measures every row once per side, so a row has one
latency sample per round and side. A row's verdict compares the two samples:

  gap      = median(A) / median(B) - 1   (> 0: B is faster)
  p        = two-sided Mann-Whitney U test of A against B
  verdict  = "undecided"  fewer than min_rounds samples on either side
             "same"       |gap| <= threshold, or no shift detectable (p >= clear)
             "faster"     |gap| > threshold and p < alpha, B faster
             "slower"     |gap| > threshold and p < alpha, B slower
             "undecided"  otherwise: more rounds are needed

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
ALPHA_PAIRED = 0.05   # sign test over the paired rounds: 6 agreeing pairs are p = 0.031

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
        z = (u - n1 * n2 / 2 - 0.5) / s if s > 0 else math.inf
        p = 0.5 * math.erfc(z / math.sqrt(2))
    return u1, min(1.0, max(0.0, p * factor))


def sign_test(a: Sequence[float], b: Sequence[float]) -> Tuple[int, int, float]:
    """Two-sided exact sign test over the paired rounds: (wins for B, decisive pairs, p).

    Each round runs both sides back to back on one GPU, so a pair cancels the GPU and the
    slow drift the unpaired rank test has to absorb. Six paired rounds that all point the
    same way are p = 0.031, while the unpaired test over 6 vs 6 samples cannot go below
    p = 0.002 and needs the samples to separate completely -- without the pairing, real
    and stable differences of a few percent stay undecided.
    """
    wins = sum(1 for x, y in zip(a, b, strict=True) if x > y)
    losses = sum(1 for x, y in zip(a, b, strict=True) if x < y)
    n = wins + losses
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return wins, n, min(1.0, 2.0 * tail / (2 ** n))


def _paired_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Sign-test p of the paired rounds; nan when the rounds do not pair up."""
    return sign_test(a, b)[2] if a and len(a) == len(b) else math.nan


def _decide_on_pairing(a, b, gap, verdict, threshold, alpha_paired) -> str:
    """Decide a row the unpaired test left open, using the pairing.

    Only when every round agrees and the median effect clears the same threshold the
    unpaired rule uses, so the pairing adds evidence without lowering the bar.
    """
    if verdict != "undecided" or len(a) != len(b) or abs(gap) <= threshold:
        return verdict
    if _paired_p(a, b) >= alpha_paired:
        return verdict
    ratios = [x / y for x, y in zip(a, b, strict=True)]
    if min(ratios) > 1 or max(ratios) < 1:
        return "faster" if gap > 0 else "slower"
    return verdict


class RowStats(NamedTuple):
    n_a: int
    n_b: int
    median_a: float
    median_b: float
    gap: float          # median_a / median_b - 1, > 0: B faster
    p: float            # two-sided Mann-Whitney; nan below two samples per side
    verdict: str
    p_paired: float = math.nan   # two-sided sign test over the paired rounds


def compare(a: Sequence[float], b: Sequence[float], *, min_rounds: int = MIN_ROUNDS,
            threshold: float = THRESHOLD, alpha: float = ALPHA,
            clear: float = CLEAR, alpha_paired: float = ALPHA_PAIRED) -> RowStats:
    """Verdict of one row from its A and B latency samples (one per round)."""
    if not a or not b:
        raise ValueError("both sides need at least one sample")
    ma, mb = statistics.median(a), statistics.median(b)
    gap = ma / mb - 1
    p = mann_whitney_u(a, b)[1] if min(len(a), len(b)) >= 2 else math.nan
    if min(len(a), len(b)) < min_rounds:
        verdict = "undecided"
    elif abs(gap) <= threshold:
        verdict = "same"
    elif p < alpha:
        verdict = "faster" if gap > 0 else "slower"
    elif p >= clear:
        verdict = "same"
    else:
        verdict = "undecided"
    verdict = _decide_on_pairing(a, b, gap, verdict, threshold, alpha_paired)
    return RowStats(len(a), len(b), ma, mb, gap, p, verdict, _paired_p(a, b))


def geomean(values: Sequence[float]) -> float:
    return math.exp(sum(map(math.log, values)) / len(values))
