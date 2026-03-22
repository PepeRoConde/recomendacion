"""
playlist_neighbourhood.py  —  Playlist-neighbourhood collaborative filter.

Batch inference formula
-----------------------
Let  B  (Q × T)  be the query matrix built from seed URIs.
Implementation uses batched inference and builds Bᵀ directly in CSC.

    1.  C        = R @ Bᵀ           (P × Q)   overlap of every training playlist
                                               with every query playlist
    2.  θ_k(C)                      (P × Q)   top-k per column (in-place, no cache)
    3.  Ŝ        = θ_k(C)ᵀ @ R     (Q × T)   weighted neighbour votes

Containment filtering is handled at build time by zero_contained(R) — no
per-query containment check is needed here.

Cold start: when a query has no valid seed tracks (or no remaining candidates
after seed filtering), recommendations fall back to the global popularity
baseline.
"""

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .base import BaseRecommender
from .theta import topk_cols


class PlaylistNeighbourhoodRecommender(BaseRecommender):
    def _recommend_popularity_fallback(self, seed_cols: list[int], top_n: int) -> list[str]:
        seed_set = set(seed_cols)
        recs = []
        for c in self.pop_order:
            if c in seed_set:
                continue
            recs.append(self.col_to_track[c])
            if len(recs) == top_n:
                break
        return recs

    def fit(
        self,
        r: csr_matrix,
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
        pop_counts = np.asarray(r.astype(bool).sum(axis=0), dtype=np.int64).flatten()
        self.pop_order = np.argsort(pop_counts)[::-1]
        self.R = r.tocsr().astype(np.float32)
        self.R = normalize(self.R, axis=1, norm="l2", copy=False)
        if not isinstance(self.R, csr_matrix):
            self.R = self.R.tocsr()

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
        t = self.R.shape[1]
        batch_size = 200
        results = []
        total_batches = (len(seeds) + batch_size - 1) // batch_size

        for start in tqdm(
            range(0, len(seeds), batch_size),
            total=total_batches,
            desc="[PlaylistNeighbourhood] inferencia",
            unit="batch",
        ):
            end = min(start + batch_size, len(seeds))
            seeds_batch = seeds[start:end]
            q = len(seeds_batch)

            # ── 1.  Build query matrix B  (q × T) ─────────────────────
            seed_cols_per_q = [
                [self.track_to_col[u] for u in s if u in self.track_to_col]
                for s in seeds_batch
            ]

            bt_rows, bt_cols = [], []
            for q_idx, cols in enumerate(seed_cols_per_q):
                for c in cols:
                    bt_rows.append(c)
                    bt_cols.append(q_idx)

            # Cold-start batch: all rows empty.
            if not bt_rows:
                results.extend(
                    [
                        self._recommend_popularity_fallback(seed_cols, top_n)
                        for seed_cols in seed_cols_per_q
                    ]
                )
                continue

            # Build B^T directly in CSC: shape (T x q)
            b_t = csc_matrix(
                (np.ones(len(bt_rows), dtype=np.float32), (bt_rows, bt_cols)),
                shape=(t, q),
            )

            # Row-normalize B by normalizing columns of B^T.
            b_t = normalize(b_t, axis=0, norm="l2", copy=False)
            if not isinstance(b_t, csc_matrix):
                b_t = b_t.tocsc()

            # ── 2.  C = R @ Bᵀ   (P × q) ─────────────────────────────
            C = self.R @ b_t

            # ── 3.  θ_k(C) column-wise   (P × q) ──────────────────────
            c_k = topk_cols(C, self.k)

            # ── 4.  Ŝ = C_kᵀ @ R   (q × T) ───────────────────────────
            scores = (c_k.T @ self.R).tocsr()

            # ── 5.  Zero seed tracks, extract top-n per query ─────────
            for q_idx, cols in enumerate(seed_cols_per_q):
                row = scores.getrow(q_idx)
                cand_cols = row.indices
                cand_vals = row.data

                if cand_cols.size == 0:
                    results.append(self._recommend_popularity_fallback(cols, top_n))
                    continue

                # Remove tracks already present in the seed playlist.
                if cols:
                    seed_set = set(cols)
                    keep_mask = np.fromiter(
                        (c not in seed_set for c in cand_cols),
                        dtype=bool,
                        count=cand_cols.size,
                    )
                    cand_cols = cand_cols[keep_mask]
                    cand_vals = cand_vals[keep_mask]

                if cand_cols.size == 0:
                    results.append(self._recommend_popularity_fallback(cols, top_n))
                    continue

                n = min(top_n, cand_cols.size)
                take = np.argpartition(cand_vals, -n)[-n:]
                order = np.argsort(cand_vals[take])[::-1]
                top_cols = cand_cols[take][order]
                results.append([self.col_to_track[c] for c in top_cols])

            # Release large temporaries before next batch.
            del b_t, C, c_k, scores

        return results

    @property
    def name(self) -> str:
        return f"playlist-neighbourhood-k{self.k}"
