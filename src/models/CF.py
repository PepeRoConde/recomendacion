"""
    src/models/CF.py  -  Collaborative filtering recommender.

    Algorithm
    ---------
    Let A be the (playlists x tracks) binary matrix.

    1.  sim   = A @ A[i].T   ->  similarity of every playlist to playlist i
    2.  Zero out sim[i]      ->  remove self-similarity
    3.  votes = sim.T @ A    ->  weighted sum of neighbour track vectors
    4.  Zero out tracks already in playlist i
    5.  argsort descending, return top_n
"""

import numpy as np
from tqdm import tqdm

from .base import BaseRecommender


class CFRecommender(BaseRecommender):

    def __init__(self, A, pid_to_row, track_to_col):
        self.A            = A
        self.pid_to_row   = pid_to_row
        self.track_to_col = track_to_col
        # Precompute reverse lookups once
        self.col_to_track = {v: k for k, v in track_to_col.items()}
        self.row_to_pid   = {v: k for k, v in pid_to_row.items()}

    # ------------------------------------------------------------------

    def recommend(self, pid, A, pid_to_row, track_to_col, top_n=10):
        """Single-playlist CF recommendations."""
        row_i = pid_to_row[pid]
        a_i   = A[row_i]                          # sparse (1 x n_tracks)

        sim   = A.dot(a_i.T)                      # (n_playlists x 1)
        sim[row_i] = 0.0                          # remove self

        votes = sim.T.dot(A).toarray().flatten()  # (n_tracks,) dense

        already_in = a_i.toarray().flatten().astype(bool)
        votes[already_in] = 0.0

        top_idx = np.argsort(votes)[::-1][:top_n]
        return [
            (self.col_to_track[idx], float(votes[idx]))
            for idx in top_idx
            if votes[idx] > 0
        ]

    # ------------------------------------------------------------------

    def recommend_batch(self, A_chunk, chunk_rows, A, track_to_col, top_n=10):
        """
        Vectorised CF scoring for a chunk of playlists.

        Returns
        -------
        V : np.ndarray  (K x n_tracks)  raw vote scores, already-in zeroed
        """
        S = A_chunk.dot(A.T)                      # sparse (K x n_playlists)

        # Zero self-similarity via lil to avoid SparseEfficiencyWarning
        S_lil = S.tolil()
        for local_i, global_row in enumerate(chunk_rows):
            S_lil[local_i, global_row] = 0.0
        S = S_lil.tocsr()

        V = S.dot(A).toarray()                    # dense (K x n_tracks)

        already_in = A_chunk.toarray().astype(bool)
        V[already_in] = 0.0

        return V
