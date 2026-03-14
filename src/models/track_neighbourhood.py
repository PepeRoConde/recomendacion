"""
track_neighbourhood.py  —  Track-neighbourhood collaborative filter.

Algorithm
---------
IDF-weight each track:  idf[t] = 1 / playlist_count[t]

Batch inference formula
-----------------------
Let  B_idf  (Q × T)  be the IDF-weighted query matrix.

    1.  S    = topk_cols( Rᵀ @ R_idf,  k )   (T × T)   cached to disk
    2.  Ŝ    = B_idf @ S                      (Q × T)   weighted votes

Where  R_idf = R * idf[t]  (each column of R scaled by the track's IDF weight).

Cache
-----
S is stored as  <cache_dir>/RtR_k{k}.npz  with a checksum of R.
If R changes or k changes, the cache is ignored and recomputed.

Containment filtering is handled at build time by zero_contained(R).
"""

import pathlib
import pickle
import time

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, diags

from .base import BaseRecommender
from .theta import topk_cols


def _checksum_R(R: csr_matrix) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(np.array(R.shape, dtype=np.int64).tobytes())
    h.update(R.data.tobytes())
    h.update(R.indices.tobytes())
    h.update(R.indptr.tobytes())
    return h.hexdigest()


def _load_or_compute_S(
    R: csr_matrix,
    idf: np.ndarray,
    k: int,
    cache_dir: str,
) -> csr_matrix:
    """
    Load or compute  S = topk_cols( Rᵀ @ R_idf,  k ).

    Cache filename:  RtR_k{k}.npz   +   RtR_k{k}_meta.pkl
    Cache is invalidated when R's checksum changes.
    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / f"RtR_k{k}.npz"
    meta_path = cache_dir / f"RtR_k{k}_meta.pkl"

    checksum = _checksum_R(R)

    # ── try cache hit ──────────────────────────────────────────────────
    if npz_path.exists() and meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("checksum") == checksum:
            print(f"[TrackNeighbourhood] Cache hit — loading S from {npz_path}")
            t0 = time.time()
            S = scipy.sparse.load_npz(str(npz_path)).tocsr()
            print(
                f"[TrackNeighbourhood] Loaded in {time.time()-t0:.2f}s  shape={S.shape}  nnz={S.nnz:,}"
            )
            return S
        print(f"[TrackNeighbourhood] Checksum mismatch — recomputing S (k={k})")

    # ── compute ────────────────────────────────────────────────────────
    # R_idf: scale each column t of R by idf[t]
    print(
        f"[TrackNeighbourhood] Computing S = topk_cols(Rᵀ @ R_idf, k={k})  R={R.shape} ..."
    )
    t0 = time.time()
    R_idf = R @ diags(idf, format="csr")  # (P × T) IDF-scaled
    S_raw = (R.T @ R_idf).tocsr()  # (T × T)
    print(
        f"[TrackNeighbourhood] Rᵀ @ R_idf done in {time.time()-t0:.2f}s  nnz={S_raw.nnz:,}"
    )

    t1 = time.time()
    S = topk_cols(S_raw, k)
    print(
        f"[TrackNeighbourhood] topk_cols done in {time.time()-t1:.2f}s  nnz={S.nnz:,}"
    )

    # ── save ───────────────────────────────────────────────────────────
    scipy.sparse.save_npz(str(npz_path), S)
    with open(meta_path, "wb") as f:
        pickle.dump({"checksum": checksum, "k": k, "shape": S.shape, "nnz": S.nnz}, f)
    print(f"[TrackNeighbourhood] Saved S to {npz_path}  (total {time.time()-t0:.2f}s)")
    return S


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
        R            : csr_matrix  (P × T)  — should already be containment-filtered
        track_to_col : dict  {track_uri -> col index}
        k            : neighbourhood size (top-k per column of S)
        cache_dir    : directory for RtR_k{k}.npz cache
        """
        self.k = k
        self.track_to_col = track_to_col
        self.col_to_track = {v: u for u, v in track_to_col.items()}
        self.R = R.tocsr().astype(np.float32)

        # idf[t] = 1 / number of playlists track t appears in
        track_freq = np.asarray(
            self.R.astype(bool).sum(axis=0), dtype=np.float32
        ).flatten()  # (T,)
        track_freq[track_freq == 0] = 1.0  # avoid div-by-zero
        self.idf = 1.0 / track_freq  # (T,)

        # load or compute the cached similarity matrix S (T × T)
        self.S = _load_or_compute_S(self.R, self.idf, k, cache_dir)

        print(
            f"[TrackNeighbourhood] fit done — R={self.R.shape}  S={self.S.shape}  k={k}"
        )

    # ------------------------------------------------------------------
    def recommend_batch(self, seeds: list, top_n: int = 500) -> list[list[str]]:
        """
        Native batch inference.

        Parameters
        ----------
        seeds  : list of Q seed-URI iterables
        top_n  : tracks to return per query

        Returns
        -------
        list of Q lists of track URIs
        """
        T = self.R.shape[1]
        Q = len(seeds)

        # ── 1.  Build IDF-weighted query matrix B_idf  (Q × T) ────────
        seed_cols_per_q = [
            [self.track_to_col[u] for u in s if u in self.track_to_col] for s in seeds
        ]

        b_rows, b_cols, b_vals = [], [], []
        for q_idx, cols in enumerate(seed_cols_per_q):
            for c in cols:
                b_rows.append(q_idx)
                b_cols.append(c)
                b_vals.append(self.idf[c])

        if not b_rows:
            return [[] for _ in seeds]

        B_idf = csr_matrix(
            (np.array(b_vals, dtype=np.float32), (b_rows, b_cols)),
            shape=(Q, T),
        )

        # ── 2.  Ŝ = B_idf @ S   (Q × T) ──────────────────────────────
        scores = (B_idf @ self.S).toarray()  # (Q × T)  dense float32

        # ── 3.  Zero seed tracks, extract top-n per query ──────────────
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
        return f"track-neighbourhood-k{self.k}"
