import numpy as np

from .base import BaseRecommender


class ReputationModel(BaseRecommender):

    def __init__(self, A, track_to_col):
        self.track_to_col = track_to_col
        self.col_to_track = {v: k for k, v in track_to_col.items()}

        # Column sums -> global popularity vector, normalised to [0, 1].
        scores    = np.asarray(A.sum(axis=0)).flatten()   # (n_tracks,)
        max_score = scores.max()
        self.scores = scores / max_score if max_score > 0 else scores

    # ------------------------------------------------------------------

    def recommend(self, pid, A, pid_to_row, track_to_col, top_n=10):
        """Single-playlist reputation recommendations."""
        row_i      = pid_to_row[pid]
        already_in = A[row_i].toarray().flatten().astype(bool)

        scores             = self.scores.copy()
        scores[already_in] = 0.0

        top_idx = np.argsort(scores)[::-1][:top_n]
        return [
            (self.col_to_track[idx], float(scores[idx]))
            for idx in top_idx
            if scores[idx] > 0
        ]

    # ------------------------------------------------------------------

    def recommend_batch(self, A_chunk, chunk_rows, A, track_to_col, top_n=10):
        """
        Vectorised reputation scoring for a chunk of playlists.

        Returns
        -------
        V : np.ndarray  (K x n_tracks)  reputation scores, already-in zeroed
        """
        K = A_chunk.shape[0]
        V = np.tile(self.scores, (K, 1))              # broadcast global scores
        V[A_chunk.toarray().astype(bool)] = 0.0
        return V
