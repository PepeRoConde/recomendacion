import numpy as np
from tqdm import tqdm

from .CF         import CFRecommender
from .reputation import ReputationModel


# --------------------------------------------------------------------------- #
#  Normalisation helpers                                                       #
# --------------------------------------------------------------------------- #

def _norm(arr):
    """Normalise a 1-D array to [0, 1]. Safe against all-zero input."""
    m = arr.max()
    return arr / m if m > 0 else arr


def _norm_rows(mat):
    """Normalise each row of a 2-D array to [0, 1] independently."""
    maxes = mat.max(axis=1, keepdims=True)
    maxes[maxes == 0] = 1      # rows that are all-zero stay all-zero
    return mat / maxes


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def recommend(pid, A, pid_to_row, track_to_col, top_n=10, alpha=0.5):
    """
    Recommend top_n tracks for a single playlist.

    Parameters
    ----------
    alpha : float in [0, 1]
        1.0 = pure CF, 0.0 = pure reputation.

    Returns
    -------
    list of (track_uri, blended_score)  length <= top_n
    """
    col_to_track = {v: k for k, v in track_to_col.items()}
    n_tracks     = A.shape[1]

    cf_model  = CFRecommender(A, pid_to_row, track_to_col)
    rep_model = ReputationModel(A, track_to_col)

    # Expand model outputs into full-length score vectors
    cf_raw  = _recs_to_array(cf_model.recommend(pid, A, pid_to_row, track_to_col, top_n=n_tracks),
                             track_to_col, n_tracks)
    rep_raw = _recs_to_array(rep_model.recommend(pid, A, pid_to_row, track_to_col, top_n=n_tracks),
                             track_to_col, n_tracks)

    # Safety mask (models should already zero these, but be defensive)
    row_i      = pid_to_row[pid]
    already_in = A[row_i].toarray().flatten().astype(bool)
    cf_raw[already_in]  = 0.0
    rep_raw[already_in] = 0.0

    blended = alpha * _norm(cf_raw) + (1.0 - alpha) * _norm(rep_raw)

    top_idx = np.argsort(blended)[::-1][:top_n]
    return [
        (col_to_track[idx], float(blended[idx]))
        for idx in top_idx
        if blended[idx] > 0
    ]


def recommend_all(A, pid_to_row, track_to_col, top_n=10, chunk_size=500,
                  alpha=0.5, pid_subset=None):
    """
    Recommend top_n tracks for every playlist (or a subset), in chunks.

    Parameters
    ----------
    pid_subset : list of pids or None
        If provided, only these playlists are scored. Their rows may be
        scattered across the matrix so we sort by row index and chunk
        those contiguous slices.  If None, all playlists are scored in
        natural row order.

    Returns
    -------
    dict  {pid -> [(track_uri, blended_score), ...]}
    """
    col_to_track = {v: k for k, v in track_to_col.items()}
    row_to_pid   = {v: k for k, v in pid_to_row.items()}

    # Build models once — ReputationModel precomputes popularity here
    cf_model  = CFRecommender(A, pid_to_row, track_to_col)
    rep_model = ReputationModel(A, track_to_col)

    # Determine which rows to score, in sorted order for efficient slicing
    if pid_subset is not None:
        target_rows = sorted(pid_to_row[pid] for pid in pid_subset if pid in pid_to_row)
    else:
        target_rows = list(range(A.shape[0]))

    results = {}

    for start in tqdm(range(0, len(target_rows), chunk_size),
                      desc="recommending", unit="chunk"):

        chunk_rows = target_rows[start:start + chunk_size]
        A_chunk    = A[chunk_rows]                  # sparse (K x n_tracks)

        V_cf  = cf_model.recommend_batch(A_chunk, chunk_rows, A, track_to_col, top_n)
        V_rep = rep_model.recommend_batch(A_chunk, chunk_rows, A, track_to_col, top_n)

        V_blended = alpha * _norm_rows(V_cf) + (1.0 - alpha) * _norm_rows(V_rep)

        for local_i, global_row in enumerate(chunk_rows):
            pid     = row_to_pid[global_row]
            scores  = V_blended[local_i]
            top_idx = np.argsort(scores)[::-1][:top_n]
            results[pid] = [
                (col_to_track[idx], float(scores[idx]))
                for idx in top_idx
                if scores[idx] > 0
            ]

    return results


# --------------------------------------------------------------------------- #
#  Internal utility                                                            #
# --------------------------------------------------------------------------- #

def _recs_to_array(recs, track_to_col, n_tracks):
    """Convert a list of (track_uri, score) into a dense score array."""
    arr = np.zeros(n_tracks)
    for uri, score in recs:
        arr[track_to_col[uri]] = score
    return arr
