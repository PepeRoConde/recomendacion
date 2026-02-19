"""
    Utility functions for the MPD recommendation system.

    - load_mpd: iterates over all slices, returns (playlist_list, track_index)
    - build_sparse_matrix: constructs a scipy CSR matrix (playlists x tracks)
    - recommend: given a playlist id, returns ranked song recommendations
"""

import os
import json
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm


def load_mpd(path, quick=False, max_files=5):
    """
    Loads the MPD dataset.

    Returns
    -------
    playlists : list of dict
        Raw playlist dicts (pid, name, tracks, …)
    track_to_col : dict  {track_uri -> column_index}
    """
    playlists = []
    track_to_col = {}

    filenames = sorted(
        f for f in os.listdir(path)
        if f.startswith("mpd.slice.") and f.endswith(".json")
    )

    # Slice *before* tqdm so total reflects actual work, not the full dataset
    if quick:
        filenames = filenames[:max_files]

    for filename in tqdm(filenames, desc="loading slices", unit="slice"):
        fullpath = os.path.join(path, filename)
        with open(fullpath, encoding="utf-8") as f:
            mpd_slice = json.load(f)

        for playlist in mpd_slice["playlists"]:
            playlists.append(playlist)
            for track in playlist["tracks"]:
                uri = track["track_uri"]
                if uri not in track_to_col:
                    track_to_col[uri] = len(track_to_col)

    return playlists, track_to_col

def build_sparse_matrix(playlists, track_to_col):
    """
    Builds a binary sparse matrix A of shape (num_playlists, num_tracks).
    A[i, j] = 1  iff playlist i contains track j.

    Rows are ordered by the position of each playlist in `playlists`.
    The mapping  pid -> row  is also returned.

    Returns
    -------
    A   : scipy.sparse.csr_matrix  (num_playlists x num_tracks)
    pid_to_row : dict {pid -> row_index}
    """
    n_playlists = len(playlists)
    n_tracks = len(track_to_col)

    pid_to_row = {pl["pid"]: row for row, pl in enumerate(playlists)}

    # lil_matrix is efficient for incremental construction
    A = lil_matrix((n_playlists, n_tracks), dtype=np.float32)

    for row, playlist in enumerate(playlists):
        for track in playlist["tracks"]:
            col = track_to_col[track["track_uri"]]
            A[row, col] = 1.0

    return csr_matrix(A), pid_to_row


def recommend(pid, playlists, A, pid_to_row, track_to_col, top_n=10):
    """
    Recommends tracks for the playlist identified by `pid`.

    Algorithm
    ---------
    Let A be the (playlists x tracks) binary matrix.

    1.  Compute the similarity of playlist i with all other playlists:
            sim = A @ A[i].T          (shape: n_playlists x 1)
        This is the i-th row of A A^T without explicitly forming the full matrix.

    2.  Zero out sim[i] (remove self-similarity).

    3.  Aggregate votes for each track:
            votes = sim.T @ A         (shape: 1 x n_tracks)
        Tracks that appear in many similar playlists get high votes.

    4.  Zero out tracks already in playlist i.

    5.  Return the top_n tracks sorted by vote count.

    Returns
    -------
    recommendations : list of (track_uri, vote_score)  length <= top_n
    """
    col_to_track = {v: k for k, v in track_to_col.items()}

    row_i = pid_to_row[pid]
    a_i = A[row_i]  # sparse row vector (1 x n_tracks)

    # Step 1 – similarity vector (n_playlists x 1)
    sim = A.dot(a_i.T)          # (n_playlists, n_tracks) @ (n_tracks, 1)

    # Step 2 – remove self
    sim[row_i] = 0.0

    # Step 3 – votes (1 x n_tracks)
    votes = sim.T.dot(A)        # (1, n_playlists) @ (n_playlists, n_tracks)
    votes = np.asarray(votes.toarray()).flatten()  # dense 1-D array of length n_tracks

    # Step 4 – mask tracks already in the playlist
    already_in = a_i.toarray().flatten().astype(bool)
    votes[already_in] = 0.0

    # Step 5 – rank
    top_indices = np.argsort(votes)[::-1][:top_n]

    recommendations = [
        (col_to_track[idx], float(votes[idx]))
        for idx in top_indices
        if votes[idx] > 0
    ]
    return recommendations
