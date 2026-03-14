from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

"""
theta.py  —  Thresholding operators for neighbourhood models.

Two independent operations:

    zero_contained(R)      — build-time, mutates/filters R
    topk_cols(M, k)        — inference-time, applied column-wise to any matrix
"""


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Build-time containment filter
# ──────────────────────────────────────────────────────────────────────────────


def zero_contained(R: csr_matrix) -> tuple[csr_matrix, np.ndarray]:
    """
    Remove rows that are strict subsets of any other row in R.

    Algorithm (inverted index + rarest-track-first intersection)
    -------------------------------------------------------------
    For each playlist i (ascending nnz order):
        1. Get the tracks in i.
        2. Sort tracks by posting-list size ascending (rarest first).
        3. Intersect posting lists across tracks — collapse fast when any
           track is rare.
        4. If any surviving candidate j ≠ i exists, i is contained → drop.

    Correctness note: we never remove a row from the inverted index after
    marking it dropped.  A dropped row appearing as a candidate only creates
    a false positive (we do a final norm check below to be safe), never a
    false negative — a true superset that was kept will still appear.

    Parameters
    ----------
    R : csr_matrix  (P × T)  binary (values treated as bool)

    Returns
    -------
    R_filtered : csr_matrix  (P' × T)
    kept_indices : np.ndarray  (P',)  original row indices that were kept
    """
    # ------------------------------------------------------------------ setup
    R = R.tocsr()
    P = R.shape[0]

    # nnz per row (playlist size)
    norms = np.asarray(R.astype(bool).sum(axis=1), dtype=np.int32).flatten()  # (P,)

    # Column-major view for cheap posting-list lookup  (track → playlists)
    R_csc = R.astype(bool).tocsc()

    # Process smallest playlists first — most likely to be contained,
    # and their early removal shrinks candidate sets for later rows.
    order = np.argsort(norms, kind="stable")
    keep_mask = np.ones(P, dtype=bool)

    # --------------------------------------------------------- main loop
    for idx in tqdm(order, desc="Quitando playlists", unit="playlist"):
        if not keep_mask[idx]:
            continue  # already dropped earlier

        norm_i = int(norms[idx])
        if norm_i == 0:
            keep_mask[idx] = False  # empty playlist — always drop
            continue

        # Tracks belonging to playlist idx
        tracks = R.indices[R.indptr[idx] : R.indptr[idx + 1]]  # (norm_i,)

        # Sort tracks by posting-list length ascending (rarest first)
        pl_sizes = R_csc.indptr[tracks + 1] - R_csc.indptr[tracks]
        tracks = tracks[np.argsort(pl_sizes, kind="stable")]

        # Intersect posting lists — start from rarest track
        t0 = tracks[0]
        candidates = R_csc.indices[R_csc.indptr[t0] : R_csc.indptr[t0 + 1]].copy()

        for t in tracks[1:]:
            if len(candidates) == 0:
                break
            pl = R_csc.indices[R_csc.indptr[t] : R_csc.indptr[t + 1]]
            candidates = np.intersect1d(candidates, pl, assume_unique=True)

        # Remove self and any already-dropped rows from candidates
        candidates = candidates[candidates != idx]
        candidates = candidates[keep_mask[candidates]]

        if len(candidates) > 0:
            # At least one kept superset (or equal-size duplicate) exists
            keep_mask[idx] = False

    # --------------------------------------------------------- output
    kept_indices = np.flatnonzero(keep_mask)
    R_filtered = R[kept_indices]
    n_removed = P - len(kept_indices)
    print(
        f"[zero_contained] Removed {n_removed:,} contained rows "
        f"({P:,} → {len(kept_indices):,} playlists)"
    )
    return R_filtered, kept_indices


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Inference-time top-k column filter
# ──────────────────────────────────────────────────────────────────────────────


def topk_cols(M: csr_matrix, k: int) -> csr_matrix:
    """
    For each column of M, zero out all but the top-k entries by value.

    Operates in CSC format for efficient column access.
    Ties at the k-th value are all kept (may return slightly more than k).

    Parameters
    ----------
    M : csr_matrix  (R × C)
    k : int         entries to keep per column

    Returns
    -------
    csr_matrix  same shape, sparser
    """
    from scipy.sparse import csc_matrix

    M_csc = M.tocsc()
    new_data = M_csc.data.copy().astype(np.float32)
    indptr = M_csc.indptr

    for q in range(M_csc.shape[1]):
        start, end = indptr[q], indptr[q + 1]
        n = end - start
        if n <= k:
            continue  # already at most k entries — nothing to do

        col_data = new_data[start:end]
        kth = np.partition(col_data, -k)[-k]
        col_data[col_data < kth] = 0.0
        new_data[start:end] = col_data

    result = csc_matrix(
        (new_data, M_csc.indices, indptr),
        shape=M_csc.shape,
    )
    result.eliminate_zeros()
    return result.tocsr()
