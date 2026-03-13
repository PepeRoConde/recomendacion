"""
theta.py  —  Thresholding operator θ_k.

Two variants, both operating on a (P × Q) or (T × T) matrix column-wise:

    theta_cols(M, k, diag_norms)

Given matrix M and a vector of "self-norms" (diagonal of the gram matrix
for each row entity), for each column q:
  1.  Zero row i if  M[i,q] == diag_norms[i]  AND  diag_norms[i] != col_nnz[q]
      (strict containment: supp of row-entity i ⊆ supp of col-entity q)
  2.  Keep only the top-k remaining nonzero entries; zero the rest.

This single function covers both models:
  - PlaylistNeighbourhood: M = R Bᵀ  (P×Q),  diag_norms = track-count per playlist
  - TrackNeighbourhood:    M = S_tt  (T×T),  diag_norms = playlist-count per track
"""

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def theta_cols(M: csr_matrix, k: int, diag_norms: np.ndarray) -> csr_matrix:
    """
    Apply θ_k column-wise to M.

    Parameters
    ----------
    M          : csr_matrix  (R × C)  — will be converted to CSC internally
    k          : int         neighbourhood size (nonzeros to keep per column)
    diag_norms : ndarray (R,) — self-similarity of each row entity
                 (number of tracks per playlist, or playlists per track)

    Returns
    -------
    csr_matrix  same shape as M, sparser
    """
    M_csc = M.tocsc()
    new_data = M_csc.data.copy().astype(np.float32)
    indptr = M_csc.indptr
    indices = M_csc.indices  # row indices for each stored value

    for q in range(M_csc.shape[1]):
        start, end = indptr[q], indptr[q + 1]
        if start == end:
            continue

        col_data = new_data[start:end]  # overlap values
        row_idx = indices[start:end]  # which row entities

        # ── 1. containment zeroing ──────────────────────────────────
        # entity i is strictly contained in query q when:
        #   overlap(i, q) == norm²(i)   [i's full support hits q]
        #   AND  norm²(i) != norm²(q)   [they are not equal supports]
        col_size = end - start  # nnz of this column (proxy for query size)
        norms_i = diag_norms[row_idx]
        strictly_contained = (col_data == norms_i) & (norms_i != col_size)
        col_data[strictly_contained] = 0.0

        # ── 2. top-k threshold ──────────────────────────────────────
        live = col_data > 0
        live_vals = col_data[live]
        if live_vals.size > k:
            kth = np.partition(live_vals, -k)[-k]
            col_data[col_data < kth] = 0.0

        new_data[start:end] = col_data

    # indptr has length n_cols+1 — must build a csc_matrix, not csr_matrix
    result = csc_matrix(
        (new_data, indices, indptr),
        shape=M_csc.shape,
    )
    result.eliminate_zeros()
    return result.tocsr()
