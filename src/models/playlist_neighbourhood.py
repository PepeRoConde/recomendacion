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
from scipy.sparse import csr_matrix

from .base import BaseRecommender


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
    def recommend_batch(self, seeds: list, top_n: int = 500) -> list[list[str]]:
        """Process one query at a time — never materialises the full Q×T score matrix."""
        return [self._recommend_one(s, top_n) for s in seeds]

    def _recommend_one(self, seed_uris, top_n: int) -> list[str]:
        seed_set = set(seed_uris)
        T = self.R.shape[1]

        # ── build seed vector b_q  (T,) ───────────────────────────────
        seed_cols = [self.track_to_col[u] for u in seed_uris if u in self.track_to_col]
        if not seed_cols:
            return []  # cold start — no signal available

        b_q = np.zeros(T, dtype=np.float32)
        b_q[seed_cols] = 1.0
        seed_size = len(seed_cols)

        # ── 1.  c_q = R b_q   (P,) ────────────────────────────────────
        c_q = self.R.dot(b_q)  # (P,)  dense float32 vector

        # ── 2a. containment zeroing ────────────────────────────────────
        # playlist i is strictly contained in seed when:
        #   c_q[i] == diag_norms[i]  (all of i's tracks are in the seed)
        #   AND diag_norms[i] != seed_size  (not an identical playlist)
        contained = (c_q == self.diag_norms) & (self.diag_norms != seed_size)
        c_q[contained] = 0.0

        # ── 2b. top-k threshold ────────────────────────────────────────
        nonzero_mask = c_q > 0
        n_nonzero = nonzero_mask.sum()
        if n_nonzero == 0:
            return []
        if n_nonzero > self.k:
            kth = np.partition(c_q, -self.k)[-self.k]
            c_q[c_q < kth] = 0.0

        # ── 3.  scores = c_q · R  but only over nonzero rows ──────────
        # Slice only the (at most k) neighbour rows — avoids a full P×T product
        neighbour_rows = np.flatnonzero(c_q)  # at most k indices
        weights = c_q[neighbour_rows]  # shape (k',)
        R_nbrs = self.R[neighbour_rows]  # (k' × T) sparse slice

        scores = np.asarray(weights @ R_nbrs).flatten()  # (T,)  dense

        # ── 4.  exclude seeds, rank ────────────────────────────────────
        for uri in seed_set:
            col = self.track_to_col.get(uri)
            if col is not None:
                scores[col] = 0.0

        n_pos = int((scores > 0).sum())
        if n_pos == 0:
            return []

        n = min(top_n, n_pos)
        top_cols = np.argpartition(scores, -n)[-n:]
        top_cols = top_cols[np.argsort(scores[top_cols])[::-1]]
        return [self.col_to_track[c] for c in top_cols]

    @property
    def name(self) -> str:
        return f"playlist-neighbourhood-k{self.k}"
