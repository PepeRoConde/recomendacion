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
from sklearn.preprocessing import normalize

from .base import BaseRecommender
from .theta import topk_cols


def _checksum_r(r: csr_matrix) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(np.array(r.shape, dtype=np.int64).tobytes())
    h.update(r.data.tobytes())
    h.update(r.indices.tobytes())
    h.update(r.indptr.tobytes())
    return h.hexdigest()


def _compute_topk_cosine_blockwise(
    r_cos: csr_matrix,
    k_max: int,
    block_size: int = 16,
) -> csr_matrix:
    """
    Build a sparse track-track cosine matrix keeping top-k per column,
    without materializing the full R_cos.T @ R_cos matrix.

    This avoids large memory spikes on very large track vocabularies.
    """
    t = r_cos.shape[1]
    out_indices = []
    out_data = []
    out_indptr = [0]

    n_blocks = (t + block_size - 1) // block_size
    for b in range(n_blocks):
        c0 = b * block_size
        c1 = min((b + 1) * block_size, t)

        # Similarities from all tracks to this block of query tracks.
        sim_block = (r_cos.T @ r_cos[:, c0:c1]).tocsc()  # (T x B)

        for local_col, col_idx in enumerate(range(c0, c1)):
            start = sim_block.indptr[local_col]
            end = sim_block.indptr[local_col + 1]

            rows = sim_block.indices[start:end]
            vals = sim_block.data[start:end]

            if rows.size == 0:
                out_indptr.append(len(out_indices))
                continue

            # Remove self-similarity; seeds are removed later anyway.
            mask = rows != col_idx
            rows = rows[mask]
            vals = vals[mask]

            if rows.size == 0:
                out_indptr.append(len(out_indices))
                continue

            if rows.size > k_max:
                keep = np.argpartition(vals, -k_max)[-k_max:]
                rows = rows[keep]
                vals = vals[keep]

            # Keep by descending similarity so higher scores appear first.
            order = np.argsort(vals)[::-1]
            rows = rows[order]
            vals = vals[order]

            out_indices.extend(rows.tolist())
            out_data.extend(vals.astype(np.float32, copy=False).tolist())
            out_indptr.append(len(out_indices))

        if (b + 1) % 100 == 0 or b + 1 == n_blocks:
            print(
                f"[TrackNeighbourhood] blockwise top-k progress: {b+1}/{n_blocks} blocks"
            )

    s_csc = scipy.sparse.csc_matrix(
        (
            np.asarray(out_data, dtype=np.float32),
            np.asarray(out_indices, dtype=np.int32),
            np.asarray(out_indptr, dtype=np.int64),
        ),
        shape=(t, t),
        dtype=np.float32,
    )
    s_csc.eliminate_zeros()
    return s_csc.tocsr()


def _load_or_compute_s(
    r_cos: csr_matrix,
    k_max: int,
    use_idf: bool,
    cache_dir: str,
) -> csr_matrix:
    """
    Load or compute  S = topk_cols( R_cosᵀ @ R_cos,  k_max ).

    Cache filename:  RtR_k{k_max}_idf{use_idf}.npz   +   corresponding meta
    Cache is invalidated when R's checksum changes.
    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / f"RtR_k{k_max}_idf{use_idf}.npz"
    meta_path = cache_dir / f"RtR_k{k_max}_idf{use_idf}_meta.pkl"

    checksum = _checksum_r(r_cos)

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
        print(f"[TrackNeighbourhood] Checksum mismatch — recomputing S (k_max={k_max})")

    # ── compute ────────────────────────────────────────────────────────
    print(
        f"[TrackNeighbourhood] Computing S with blockwise cosine top-k (k_max={k_max})  R_cos={r_cos.shape} ..."
    )
    t0 = time.time()
    S = _compute_topk_cosine_blockwise(r_cos, k_max=k_max, block_size=16)
    print(
        f"[TrackNeighbourhood] blockwise top-k done in {time.time()-t0:.2f}s  nnz={S.nnz:,}"
    )

    # ── save ───────────────────────────────────────────────────────────
    scipy.sparse.save_npz(str(npz_path), S)
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "checksum": checksum,
                "k_max": k_max,
                "use_idf": use_idf,
                "shape": S.shape,
                "nnz": S.nnz,
            },
            f,
        )
    print(f"[TrackNeighbourhood] Saved S to {npz_path}  (total {time.time()-t0:.2f}s)")
    return S


class TrackNeighbourhoodRecommender(BaseRecommender):
    def fit(
        self,
        r: csr_matrix,
        track_to_col: dict,
        k: int = 20,
        cache_dir: str = "data",
        use_idf: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        R            : csr_matrix  (P × T)  — should already be containment-filtered
        track_to_col : dict  {track_uri -> col index}
        k            : neighbourhood size (top-k per column of S)
        cache_dir    : directory for RtR cache
        """
        self.k_max = 150
        self.k = k
        self.use_idf = use_idf
        self.track_to_col = track_to_col
        self.col_to_track = {v: u for u, v in track_to_col.items()}
        self.R = r.tocsr().astype(np.float32)

        # Optional IDF weighting before cosine normalization.
        if self.use_idf:
            track_freq = np.asarray(
                self.R.astype(bool).sum(axis=0), dtype=np.float32
            ).flatten()
            track_freq[track_freq == 0] = 1.0
            self.idf = 1.0 / track_freq
            r_work = self.R @ diags(self.idf, format="csr")
        else:
            self.idf = np.ones(self.R.shape[1], dtype=np.float32)
            r_work = self.R

        # Column-wise normalization: cosine similarities from R_cos.T @ R_cos.
        r_cos = normalize(r_work, axis=0, norm="l2", copy=False)
        if not scipy.sparse.isspmatrix_csr(r_cos):
            r_cos = r_cos.tocsr()

        # Load or compute a fixed-size similarity cache and recort at inference.
        self.S = _load_or_compute_s(r_cos, self.k_max, self.use_idf, cache_dir)
        self._S_runtime_k = None
        self._S_runtime = None

        print(
            f"[TrackNeighbourhood] fit done — R={self.R.shape}  S={self.S.shape}  k={k}  k_max={self.k_max}  use_idf={self.use_idf}"
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

        b_idf = csr_matrix(
            (np.array(b_vals, dtype=np.float32), (b_rows, b_cols)),
            shape=(Q, T),
        )

        # ── 2.  Dynamic top-k truncation and scoring ───────────────────
        k_runtime = min(self.k, self.k_max)
        if self._S_runtime_k != k_runtime or self._S_runtime is None:
            self._S_runtime = topk_cols(self.S, k_runtime)
            self._S_runtime_k = k_runtime

        scores = (b_idf @ self._S_runtime).toarray()  # (Q × T) dense float32

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
        idf_tag = "-idf" if self.use_idf else ""
        return f"track-neighbourhood-k{self.k}{idf_tag}"
