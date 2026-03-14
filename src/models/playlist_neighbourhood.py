"""
playlist_neighbourhood.py  —  Playlist-neighbourhood collaborative filter.

Batch inference formula
-----------------------
Let  B  (Q × T)  be the query matrix built from seed URIs.

    1.  C        = R @ Bᵀ           (P × Q)   overlap of every training playlist
                                               with every query playlist
    2.  θ_k(C)                      (P × Q)   top-k per column (in-place, no cache)
    3.  Ŝ        = θ_k(C)ᵀ @ R     (Q × T)   weighted neighbour votes

Containment filtering is handled at build time by zero_contained(R) — no
per-query containment check is needed here.

Cold start: a zero row in B produces a zero column in C → zero score row in Ŝ.
"""

import numpy as np
from scipy.sparse import csr_matrix

from .base import BaseRecommender
from .theta import topk_cols


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
        R            : csr_matrix  (P × T)  — should already be containment-filtered
        track_to_col : dict  {track_uri -> col index}
        k            : neighbourhood size (columns to keep in θ_k)
        cache_dir    : unused for this model
        """
        self.k = k
        self.track_to_col = track_to_col
        self.col_to_track = {v: u for u, v in track_to_col.items()}
        self.R = R.tocsr().astype(np.float32)

        print(f"[PlaylistNeighbourhood] fit — R={self.R.shape}  k={k}")

    # ------------------------------------------------------------------
    def recommend_batch(self, seeds: list, top_n: int = 500) -> list[list[str]]:
        """
        Native batch inference — builds B, runs the full matrix pipeline,
        then extracts top-n per row.

        Parameters
        ----------
        seeds  : list of Q seed-URI iterables (sets or lists)
        top_n  : tracks to return per query

        Returns
        -------
        list of Q lists of track URIs
        """
        T = self.R.shape[1]
        Q = len(seeds)

        # ── 1.  Build query matrix B  (Q × T) ─────────────────────────
        seed_cols_per_q = [
            [self.track_to_col[u] for u in s if u in self.track_to_col] for s in seeds
        ]

        # COO for B
        b_rows, b_cols = [], []
        for q_idx, cols in enumerate(seed_cols_per_q):
            for c in cols:
                b_rows.append(q_idx)
                b_cols.append(c)

        if not b_rows:
            return [[] for _ in seeds]

        B = csr_matrix(
            (np.ones(len(b_rows), dtype=np.float32), (b_rows, b_cols)),
            shape=(Q, T),
        )

        # ── 2.  C = R @ Bᵀ   (P × Q) ─────────────────────────────────
        C = self.R @ B.T  # (P × Q)  sparse

        # ── 3.  θ_k(C) column-wise   (P × Q) ──────────────────────────
        C_k = topk_cols(C, self.k)  # (P × Q)  sparser

        # ── 4.  Ŝ = C_kᵀ @ R   (Q × T) ───────────────────────────────
        scores = (C_k.T @ self.R).toarray()  # (Q × T)  dense float32

        # ── 5.  Zero seed tracks, extract top-n per query ──────────────
        results = []
        for q_idx, cols in enumerate(seed_cols_per_q):
            row = scores[q_idx]
            for c in cols:
                row[c] = 0.0

            n_pos = int((row > 0).sum())
            if n_pos == 0:
                results.append([])
                continue

            n = min(top_n, n_pos)
            top_c = np.argpartition(row, -n)[-n:]
            top_c = top_c[np.argsort(row[top_c])[::-1]]
            results.append([self.col_to_track[c] for c in top_c])

        return results

    @property
    def name(self) -> str:
        return f"playlist-neighbourhood-k{self.k}"
