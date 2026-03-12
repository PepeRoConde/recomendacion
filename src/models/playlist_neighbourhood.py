"""
playlist_neighbourhood.py  —  Playlist-neighbourhood collaborative filter.

Batch inference formula
-----------------------
Let  B  (Q × T)  be the query matrix built from seed URIs.

    hat_B = θ_k( R Bᵀ )ᵀ · R       (Q × T)

Steps:
  1.  C  = R Bᵀ          (P × Q)   overlap of every training playlist with every query
  2.  θ_k(C)             (P × Q)   per query-column: top-k training playlists,
                                    containment-zeroed
  3.  hat_B = θ_k(C)ᵀ · R          (Q × T)   weighted neighbour votes

R Bᵀ is cheap (P×Q, Q ~ eval batch size ~ 10k), so nothing needs caching
beyond R itself which is already in memory.

Cold start (empty seed row in B): scores default to 0 for all tracks.
"""

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from .base import BaseRecommender
from .theta import theta_cols


class PlaylistNeighbourhoodRecommender(BaseRecommender):
    def fit(
        self,
        R: csr_matrix,
        track_to_col: dict,
        k: int = 20,
        cache_dir: str = "data",
        **kwargs,
    ):
        """
        Parameters
        ----------
        R            : csr_matrix  (P × T)
        track_to_col : dict  {track_uri -> col index}
        k            : neighbourhood size
        cache_dir    : unused for this model (no gram matrix to cache)
        """
        self.k = k
        self.track_to_col = track_to_col
        self.col_to_track = {v: u for u, v in track_to_col.items()}
        self.R = R.tocsr().astype(np.float32)

        # diag_norms[i] = number of tracks in playlist i  =  R[i,:].nnz
        # Used by theta_cols for containment detection.
        self.diag_norms = np.asarray(
            self.R.astype(bool).sum(axis=1), dtype=np.float32
        ).flatten()

        print(f"[PlaylistNeighbourhood] fit — R={self.R.shape}  k={k}")

    # ------------------------------------------------------------------
    def _build_B(self, seeds: list) -> csr_matrix:
        """Build query matrix B  (Q × T) from a list of seed-URI collections."""
        Q = len(seeds)
        T = self.R.shape[1]
        rows, cols = [], []
        for q, seed_uris in enumerate(seeds):
            for uri in seed_uris:
                col = self.track_to_col.get(uri)
                if col is not None:
                    rows.append(q)
                    cols.append(col)
        if not rows:
            return csr_matrix((Q, T), dtype=np.float32)
        data = np.ones(len(rows), dtype=np.float32)
        return csr_matrix(
            (data, (np.array(rows, np.int32), np.array(cols, np.int32))),
            shape=(Q, T),
        )

    def recommend_batch(self, seeds: list, top_n: int = 500) -> list[list[str]]:
        Q = len(seeds)
        B = self._build_B(seeds)  # (Q × T)

        # 1.  C = R Bᵀ   (P × Q)
        C = (self.R @ B.T).astype(np.float32)  # may be sparse or dense

        if not scipy.sparse.issparse(C):
            C = csr_matrix(C)

        # 2.  θ_k(C)  column-wise  (each column = one query playlist)
        #     diag_norms are row-entity (playlist) norms
        C_thresh = theta_cols(C, self.k, self.diag_norms)  # (P × Q)

        # 3.  hat_B = C_thresh.T @ R   (Q × T)
        hat_B = C_thresh.T @ self.R  # (Q × T)

        # 4.  Rank per query, exclude seeds
        results = []
        for q in range(Q):
            if scipy.sparse.issparse(hat_B):
                scores = np.asarray(hat_B[q].todense()).flatten()
            else:
                scores = np.asarray(hat_B[q]).flatten()

            # zero seed tracks
            for uri in seeds[q]:
                col = self.track_to_col.get(uri)
                if col is not None:
                    scores[col] = 0.0

            n_pos = int((scores > 0).sum())
            if n_pos == 0:
                results.append([])
                continue

            n = min(top_n, n_pos)
            top_cols = np.argpartition(scores, -n)[-n:]
            top_cols = top_cols[np.argsort(scores[top_cols])[::-1]]
            results.append([self.col_to_track[c] for c in top_cols])

        return results

    @property
    def name(self) -> str:
        return f"playlist-neighbourhood-k{self.k}"
