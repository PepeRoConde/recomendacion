"""
    src/evaluation/metrics.py

    Pure metric functions for the RecSys 2018 MPD challenge:
        r_precision  : R-Precision
        ndcg         : Normalised Discounted Cumulative Gain
        clicks       : Recommended Songs Clicks
"""

import math


def r_precision(predicted, relevant_set):
    """
    R-Precision: fraction of ground-truth tracks found in the top-R predictions.

    R is defined as |relevant_set|.  Only the first R predictions are examined.

    Example
    -------
    predicted    = [A, B, C, D, E]
    relevant_set = {A, C, X}   →  R = 3
    hits in predicted[:3] = {A, C}  →  R-Prec = 2/3
    """
    R = len(relevant_set)
    if R == 0:
        return 0.0
    hits = sum(1 for t in predicted[:R] if t in relevant_set)
    return hits / R


def ndcg(predicted, relevant_set):
    """
    Normalised Discounted Cumulative Gain (binary relevance).

    DCG  = sum_{i=0}^{n-1}  rel_i / log2(i + 2)
    IDCG = sum_{i=0}^{R-1}  1     / log2(i + 2)   (ideal: all R hits first)
    NDCG = DCG / IDCG

    Positions are 0-indexed, so position 0 has weight 1/log2(2) = 1.
    """
    if not relevant_set:
        return 0.0

    dcg = 0.0
    for i, t in enumerate(predicted):
        if t in relevant_set:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG: first |relevant_set| positions all relevant
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(relevant_set)))

    return dcg / idcg if idcg > 0.0 else 0.0


def clicks(predicted, relevant_set):
    """
    Minimum number of times the user must refresh the 10-track recommendation
    list before seeing a relevant track.

    - Position 1-10  (index 0-9)  → 0 clicks
    - Position 11-20 (index 10-19)→ 1 click
    - ...
    - No relevant track found     → 51  (per challenge spec: 1 > max possible)
    """
    for i, t in enumerate(predicted):
        if t in relevant_set:
            return i // 10

    return 51   # challenge spec: "a value of 51 is picked"
