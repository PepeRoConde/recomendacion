"""
    evaluate.py - Precision@N and Recall@N on the held-out eval set.

    Metrics are reported per num_samples group and overall.
"""

import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def load_eval(eval_dir):
    """
    Loads the eval split.

    Returns
    -------
    ground_truth  : dict  {pid -> set of track_uris}
    pid_to_samples: dict  {pid -> num_samples}  (for group breakdown)
    """
    eval_path = os.path.join(eval_dir, "test_eval_playlists.json")
    input_path = os.path.join(eval_dir, "test_input_playlists.json")

    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(input_path, encoding="utf-8") as f:
        input_data = json.load(f)

    ground_truth = {
        p["pid"]: {t["track_uri"] for t in p["tracks"]}
        for p in eval_data["playlists"]
    }
    pid_to_samples = {
        p["pid"]: p["num_samples"]
        for p in input_data["playlists"]
    }
    return ground_truth, pid_to_samples


def evaluate(ground_truth, pid_to_samples, pid_to_row, A, track_to_col,
             top_n=10, chunk_size=500):
    """
    Computes Precision@N and Recall@N, broken down by num_samples group.

    Returns
    -------
    overall   : (precision, recall)
    by_group  : dict  {num_samples -> (precision, recall, count)}
    """
    col_to_track = {v: k for k, v in track_to_col.items()}

    test_pids = [pid for pid in ground_truth if pid in pid_to_row]
    missing   = [pid for pid in ground_truth if pid not in pid_to_row]

    if not test_pids:
        raise ValueError("No test pids found in the training matrix. "
                         "Rebuild with --input-playlists.")
    if missing:
        print(f"[warn] {len(missing)} test pids not in matrix — skipped")

    # accumulators per group
    group_hits      = defaultdict(float)
    group_precision = defaultdict(float)
    group_recall    = defaultdict(float)
    group_count     = defaultdict(int)

    for start in tqdm(range(0, len(test_pids), chunk_size),
                      desc="evaluating", unit="chunk"):
        chunk_pids = test_pids[start:start + chunk_size]
        chunk_rows = [pid_to_row[pid] for pid in chunk_pids]

        A_chunk = A[chunk_rows]                   # (K x n_tracks)
        S       = A_chunk.dot(A.T)                # (K x n_playlists)

        # Zero self-similarity
        for local_i, global_row in enumerate(chunk_rows):
            S[local_i, global_row] = 0.0

        V = S.dot(A).toarray()                    # (K x n_tracks) dense
        already_in = A_chunk.toarray().astype(bool)
        V[already_in] = 0.0

        for local_i, pid in enumerate(chunk_pids):
            votes   = V[local_i]
            top_idx = np.argsort(votes)[::-1][:top_n]
            predicted = {col_to_track[idx] for idx in top_idx if votes[idx] > 0}

            truth   = ground_truth[pid]
            hits    = len(predicted & truth)
            prec    = hits / top_n
            rec     = hits / len(truth) if truth else 0.0

            group = pid_to_samples.get(pid, -1)
            group_precision[group] += prec
            group_recall[group]    += rec
            group_count[group]     += 1

    # Aggregate
    total_p = sum(group_precision.values())
    total_r = sum(group_recall.values())
    n_total = sum(group_count.values())

    overall  = (total_p / n_total, total_r / n_total)
    by_group = {
        g: (group_precision[g] / group_count[g],
            group_recall[g]    / group_count[g],
            group_count[g])
        for g in sorted(group_count)
    }

    return overall, by_group


def print_results(overall, by_group, top_n):
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {'num_samples':>12}  {'precision@'+str(top_n):>14}  {'recall@'+str(top_n):>12}  {'n':>7}")
    print(f"{'─' * width}")
    for group, (prec, rec, count) in by_group.items():
        print(f"  {group:>12}  {prec:>14.4f}  {rec:>12.4f}  {count:>7,}")
    print(f"{'─' * width}")
    print(f"  {'overall':>12}  {overall[0]:>14.4f}  {overall[1]:>12.4f}  {sum(c for _,_,c in by_group.values()):>7,}")
    print(f"{'─' * width}\n")
