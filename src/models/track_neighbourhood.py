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
import scipy.sparse
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
    def _build_B(self, seeds: list) -> csr_matrix:
        """Build query matrix B  (Q × T) from a list of seed-URI collections."""
        Q = len(seeds)
        T = self.S.shape[0]
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

        # hat_B = B · θ_k(S_tt)   (Q × T)
        hat_B = B @ self.S  # sparse × sparse → sparse

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
        return f"track-neighbourhood-k{self.k}"
