"""
    src/evaluation/evaluate.py

    Metrics
    -------
    R-Precision
        |predicted ∩ truth[:top_n]| / |truth[:top_n]|
        Rewards total relevant retrieved regardless of order.

    NDCG  (Normalised Discounted Cumulative Gain)
        DCG  = Σ 1/log2(i+2)  for each relevant track at rank i (0-indexed)
        IDCG = DCG of a perfect ranking (relevant tracks first)
        NDCG = DCG / IDCG   (0 if no relevant tracks)

    Recommended Songs Clicks
        floor(rank_of_first_relevant / 10)   (0-indexed rank, buckets of 10)
        51 if no relevant track found in predictions.
        Lower is better.

    All three metrics are averaged across playlists and reported per
    num_samples group and overall.

    evaluate() delegates scoring to the blended model via recommend_all()
    so that the alpha parameter is respected.
"""

import json
import os
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#  Data loading                                                                #
# --------------------------------------------------------------------------- #

def load_eval(eval_dir):
    """
    Load the eval split.

    Returns
    -------
    ground_truth   : dict  {pid -> list of track_uris}   ordered, capped later
    pid_to_samples : dict  {pid -> num_samples}
    """
    eval_path  = os.path.join(eval_dir, "test_eval_playlists.json")
    input_path = os.path.join(eval_dir, "test_input_playlists.json")

    with open(eval_path,  encoding="utf-8") as f:
        eval_data  = json.load(f)
    with open(input_path, encoding="utf-8") as f:
        input_data = json.load(f)

    # Keep as list (order doesn't matter for these metrics, but consistent)
    ground_truth = {
        p["pid"]: [t["track_uri"] for t in p["tracks"]]
        for p in eval_data["playlists"]
    }
    pid_to_samples = {
        p["pid"]: p["num_samples"]
        for p in input_data["playlists"]
    }
    return ground_truth, pid_to_samples


# --------------------------------------------------------------------------- #
#  Per-playlist metric functions                                               #
# --------------------------------------------------------------------------- #

def _r_precision(predicted_ranked, truth_set):
    """
    R-Precision: hits in predicted / |truth_set|.
    predicted_ranked is already capped to top_n before calling.
    truth_set        is already capped to top_n before calling.
    """
    if not truth_set:
        return 0.0
    hits = sum(1 for t in predicted_ranked if t in truth_set)
    return hits / len(truth_set)


def _ndcg(predicted_ranked, truth_set):
    """
    NDCG over the full predicted list.
    truth_set is already capped to top_n.
    """
    if not truth_set:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, t in enumerate(predicted_ranked)
        if t in truth_set
    )

    # Ideal: all relevant tracks placed first
    n_ideal = min(len(truth_set), len(predicted_ranked))
    idcg    = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))

    return dcg / idcg if idcg > 0 else 0.0


def _clicks(predicted_ranked, truth_set):
    """
    Recommended Songs Clicks: floor(rank_of_first_relevant / 10).
    Returns 51 if no relevant track found.
    """
    for rank, t in enumerate(predicted_ranked):
        if t in truth_set:
            return rank // 10
    return 51


# --------------------------------------------------------------------------- #
#  Main evaluation loop                                                        #
# --------------------------------------------------------------------------- #

def evaluate(ground_truth, pid_to_samples, pid_to_row, A, track_to_col,
             top_n=500, chunk_size=500, alpha=0.5):
    """
    Evaluate R-Precision, NDCG, and Clicks using the blended model.

    Parameters
    ----------
    ground_truth   : {pid -> list of track_uris}
    pid_to_samples : {pid -> num_samples}
    pid_to_row     : {pid -> matrix row index}
    A              : sparse playlist-track matrix
    track_to_col   : {track_uri -> col index}
    top_n          : recommendation list length (default 500, per challenge spec)
    chunk_size     : playlists per batch
    alpha          : CF vs reputation blend weight

    Returns
    -------
    overall  : dict  {metric -> mean_value}
    by_group : dict  {num_samples -> {metric -> mean_value, 'n' -> count}}
    """
    # Import here to avoid circular imports at module load time
    from src.models.blend import recommend_all

    test_pids = [pid for pid in ground_truth if pid in pid_to_row]
    missing   = [pid for pid in ground_truth if pid not in pid_to_row]

    if not test_pids:
        raise ValueError("No test pids found in the training matrix. "
                         "Rebuild with --input-playlists.")
    if missing:
        print(f"[warn] {len(missing)} test pids not in matrix — skipped")

    # Get blended recommendations for all test playlists in one pass
    print(f"[evaluate] running blended model (alpha={alpha:.2f}) …")
    all_recs = recommend_all(
        A, pid_to_row, track_to_col,
        top_n=top_n, chunk_size=chunk_size, alpha=alpha,
        pid_subset=test_pids,
    )

    # Accumulators: {group -> {metric -> running_sum}}
    group_sums  = defaultdict(lambda: defaultdict(float))
    group_count = defaultdict(int)

    for pid in tqdm(test_pids, desc="scoring", unit="playlist"):
        # Cap truth to top_n
        truth_list = ground_truth[pid][:top_n]
        truth_set  = set(truth_list)

        # Predicted: ordered list of uris
        predicted  = [uri for uri, _ in all_recs.get(pid, [])]

        group = pid_to_samples.get(pid, -1)
        group_sums[group]["r_precision"] += _r_precision(predicted, truth_set)
        group_sums[group]["ndcg"]        += _ndcg(predicted, truth_set)
        group_sums[group]["clicks"]      += _clicks(predicted, truth_set)
        group_count[group]               += 1

    # Aggregate
    by_group = {
        g: {
            "r_precision": group_sums[g]["r_precision"] / group_count[g],
            "ndcg":        group_sums[g]["ndcg"]        / group_count[g],
            "clicks":      group_sums[g]["clicks"]      / group_count[g],
            "n":           group_count[g],
        }
        for g in sorted(group_count)
    }

    n_total = sum(group_count.values())
    overall = {
        metric: sum(group_sums[g][metric] for g in group_count) / n_total
        for metric in ("r_precision", "ndcg", "clicks")
    }

    return overall, by_group


# --------------------------------------------------------------------------- #
#  Pretty printing                                                             #
# --------------------------------------------------------------------------- #

def print_results(overall, by_group, top_n):
    width = 72
    print(f"\n{'─' * width}")
    print(f"  {'num_samples':>12}  {'R-Prec':>10}  {'NDCG':>10}  {'Clicks':>10}  {'n':>7}")
    print(f"{'─' * width}")
    for group, metrics in by_group.items():
        print(f"  {group:>12}  "
              f"{metrics['r_precision']:>10.4f}  "
              f"{metrics['ndcg']:>10.4f}  "
              f"{metrics['clicks']:>10.4f}  "
              f"{metrics['n']:>7,}")
    print(f"{'─' * width}")
    print(f"  {'overall':>12}  "
          f"{overall['r_precision']:>10.4f}  "
          f"{overall['ndcg']:>10.4f}  "
          f"{overall['clicks']:>10.4f}  "
          f"{sum(m['n'] for m in by_group.values()):>7,}")
    print(f"{'─' * width}\n")
