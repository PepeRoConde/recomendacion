"""
track_neighbourhood.py  —  Track-neighbourhood collaborative filter.

Batch inference formula
-----------------------
Let  B  (Q × T)  be the query matrix built from seed URIs.

    hat_B = B · θ_k( Rᵀ R )        (Q × T)

Steps:
  1.  S_tt = Rᵀ R    (T × T)   track-track co-occurrence — query-independent,
                                 loaded/computed once via gram_cache
  2.  θ_k(S_tt)      (T × T)   per column: top-k tracks, containment-zeroed
  3.  hat_B = B · θ_k(S_tt)    (Q × T)   for each query, weighted track votes

The gram matrix S_tt is cached to disk with a checksum of R so it is
automatically recomputed if R changes and reused otherwise.

Containment rule: if the playlist-support of track a is a strict subset of
track b's playlist-support, then S_tt[a,b] = 0 (a does not influence b's score).
"""

import numpy as np
from scipy.sparse import csr_matrix

from .base import BaseRecommender
from .theta import theta_cols
from .gram_cache import load_or_compute_Stt


class TrackNeighbourhoodRecommender(BaseRecommender):
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
        k            : neighbourhood size (top-k tracks per column of S_tt)
        cache_dir    : directory for gram matrix cache
        """
        self.k = k
        self.track_to_col = track_to_col
        self.col_to_track = {v: u for u, v in track_to_col.items()}
        R = R.tocsr().astype(np.float32)

        # ── 1. S_tt = Rᵀ R  (loaded from cache or computed) ──────────
        S_tt_raw = load_or_compute_Stt(R, cache_dir=cache_dir)  # (T × T)

        # diag_norms[t] = S_tt[t,t] = number of playlists track t appears in
        # Used by theta_cols for containment detection.
        diag_norms = np.asarray(S_tt_raw.diagonal(), dtype=np.float32)

        # ── 2. θ_k(S_tt) ──────────────────────────────────────────────
        print(f"[TrackNeighbourhood] Applying θ_k  k={k} ...")
        self.S = theta_cols(S_tt_raw, k, diag_norms)  # (T × T)
        print(
            f"[TrackNeighbourhood] θ(S_tt)  nnz={self.S.nnz:,}  "
            f"sparsity={1 - self.S.nnz / (self.S.shape[0] ** 2):.4f}"
        )

    # ------------------------------------------------------------------
    def recommend_batch(self, seeds: list, top_n: int = 500) -> list[list[str]]:
        return [self._recommend_one(s, top_n) for s in seeds]

    def _recommend_one(self, seed_uris, top_n: int) -> list[str]:
        seed_set = set(seed_uris)
        T = self.S.shape[0]

        seed_cols = [self.track_to_col[u] for u in seed_uris if u in self.track_to_col]
        if not seed_cols:
            return []

        # b_q (1 × T) sparse seed vector
        b_q = csr_matrix(
            (
                np.ones(len(seed_cols), dtype=np.float32),
                (np.zeros(len(seed_cols), np.int32), np.array(seed_cols, np.int32)),
            ),
            shape=(1, T),
        )

        # scores = b_q · θ_k(S_tt)   (1 × T) → (T,)
        scores = np.asarray(b_q.dot(self.S).todense()).flatten()

        # zero seed tracks
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
        return f"track-neighbourhood-k{self.k}"
