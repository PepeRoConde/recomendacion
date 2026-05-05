"""
track_neighbourhood.py  —  Track-neighbourhood collaborative filter.

Algorithm
---------
IDF-weight each track:  idf[t] = 1 / playlist_count[t]

Batch inference formula
-----------------------
Let  B_idf  (Q × T)  be the IDF-weighted query matrix.

    1.  Precompute/cache a sparse top-k cosine matrix S (T × T)
        from the training matrix using chunked construction.
    2.  Ŝ    = B_idf @ S_runtime               (Q × T)   weighted votes

Where  R_idf = R * idf[t]  (each column of R scaled by the track's IDF weight)
when use_idf=True. With use_idf=False, idf is all-ones (no extra weighting).

Cache
-----
S is stored as <cache_dir>/RtR_k{k_max}_idf{use_idf}.npz with a checksum of
the normalized training matrix used for cosine computation.
If that checksum changes, the cache is ignored and recomputed.

Cold start
----------
If a query has no valid seed tracks (or no remaining candidates after seed
filtering), recommendations fall back to the global popularity baseline.

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


def _load_or_compute_s(
    r: csr_matrix,
    k_max: int,
    use_idf: bool,
    cache_dir: str,
    chunk_size: int = 1000,
    idf: np.ndarray | None = None,
) -> csr_matrix:
    """
    Load or compute S in memory-safe chunks:
        1) optional IDF column scaling
        2) column-wise L2 normalization (cosine-ready)
        3) chunked multiplication R_cos.T @ R_cos[:, start:end]
        4) immediate top-k filter per chunk
        5) incremental CSC assembly from filtered chunk columns

    Cache filename:  RtR_k{k_max}_idf{use_idf}.npz   +   corresponding meta
    Cache is invalidated when R's checksum changes.
    """
    r_work = r.tocsr().astype(np.float32)
    if use_idf:
        if idf is None:
            track_freq = np.asarray(r_work.astype(bool).sum(axis=0), dtype=np.float32).flatten()
            track_freq[track_freq == 0] = 1.0
            idf = 1.0 / track_freq
        r_work = r_work @ diags(idf, format="csr")

    r_cos = normalize(r_work, axis=0, norm="l2", copy=False)
    if not scipy.sparse.isspmatrix_csr(r_cos):
        r_cos = r_cos.tocsr()

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
    print(f"[TrackNeighbourhood] Computing S in chunks (k_max={k_max}, chunk_size={chunk_size})  R_cos={r_cos.shape} ...")
    t0 = time.time()
    t = r_cos.shape[1]
    n_chunks = (t + chunk_size - 1) // chunk_size

    # Build CSC buffers directly to avoid high peak RAM from hstack(chunks).
    out_indices = []
    out_data = []
    out_indptr = [0]

    for chunk_idx, start in enumerate(range(0, t, chunk_size), start=1):
        end = min(start + chunk_size, t)

        # Similarity of all tracks vs current chunk only.
        chunk_s = (r_cos.T @ r_cos[:, start:end]).tocsr()

        # Keep top-k per chunk column immediately to cap memory.
        chunk_s = topk_cols(chunk_s, k_max)

        # Append this filtered block to global CSC buffers column by column.
        chunk_csc = chunk_s.tocsc()
        for local_col in range(end - start):
            global_col = start + local_col
            c0 = chunk_csc.indptr[local_col]
            c1 = chunk_csc.indptr[local_col + 1]

            rows = chunk_csc.indices[c0:c1]
            vals = chunk_csc.data[c0:c1]

            # Optional: drop self-similarity to keep only true neighbours.
            if rows.size:
                mask = rows != global_col
                rows = rows[mask]
                vals = vals[mask]

            out_indices.extend(rows.tolist())
            out_data.extend(vals.astype(np.float32, copy=False).tolist())
            out_indptr.append(len(out_indices))

        if chunk_idx % 10 == 0 or chunk_idx == n_chunks:
            print(f"[TrackNeighbourhood] chunk progress: {chunk_idx}/{n_chunks}")

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
    S = s_csc.tocsr()
    print(
        f"[TrackNeighbourhood] chunked top-k done in {time.time()-t0:.2f}s  nnz={S.nnz:,}"
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
                "chunk_size": chunk_size,
            },
            f,
        )
    print(f"[TrackNeighbourhood] Saved S to {npz_path}  (total {time.time()-t0:.2f}s)")
    return S


class TrackNeighbourhoodRecommender(BaseRecommender):
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

        pop_counts = np.asarray(self.R.astype(bool).sum(axis=0), dtype=np.int64).flatten()
        self.pop_order = np.argsort(pop_counts)[::-1]

        # Query weighting uses the same optional IDF choice as training similarities.
        if self.use_idf:
            track_freq = np.asarray(
                self.R.astype(bool).sum(axis=0), dtype=np.float32
            ).flatten()
            track_freq[track_freq == 0] = 1.0
            self.idf = 1.0 / track_freq
        else:
            self.idf = np.ones(self.R.shape[1], dtype=np.float32)

        # Load or compute a fixed-size similarity cache and recort at inference.
        self.S = _load_or_compute_s(
            self.R,
            self.k_max,
            self.use_idf,
            cache_dir,
            chunk_size=1000,
            idf=self.idf,
        )
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
            return [
                self._recommend_popularity_fallback(seed_cols, top_n)
                for seed_cols in seed_cols_per_q
            ]

        b_idf = csr_matrix(
            (np.array(b_vals, dtype=np.float32), (b_rows, b_cols)),
            shape=(Q, T),
        )

        # ── 2.  Dynamic top-k truncation and scoring ───────────────────
        k_runtime = min(self.k, self.k_max)
        if self._S_runtime_k != k_runtime or self._S_runtime is None:
            self._S_runtime = topk_cols(self.S, k_runtime)
            self._S_runtime_k = k_runtime

        scores = (b_idf @ self._S_runtime).tocsr()  # (Q × T) sparse float32

        # ── 3.  Zero seed tracks, extract top-n per query ──────────────
        results = []
        for q_idx, cols in enumerate(seed_cols_per_q):
            start = scores.indptr[q_idx]
            end = scores.indptr[q_idx + 1]

            cand_cols = scores.indices[start:end]
            cand_vals = scores.data[start:end]

            if cand_cols.size == 0:
                results.append(self._recommend_popularity_fallback(cols, top_n))
                continue

            # Remove seed tracks from the candidate list.
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

        return results

    @property
    def name(self) -> str:
        idf_tag = "-idf" if self.use_idf else ""
        return f"track-neighbourhood-k{self.k}{idf_tag}"
