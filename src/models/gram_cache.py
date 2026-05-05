"""
gram_cache.py  —  Content-addressed disk cache for the track-track gram matrix.

Only  S_tt = Rᵀ R  (T × T)  is worth caching: it is query-independent and
expensive for large T.  S_pp = R Rᵀ is NOT cached because the playlist-
neighbourhood model computes  R Bᵀ  at inference time (where B is the query
batch), so R Rᵀ is never needed.

The cache key is a SHA-256 digest of R's arrays (shape + data + indices +
indptr).  If R changes for any reason, the cached file is automatically
ignored and recomputed — no flag needed in main.py.

Public API
----------
    load_or_compute_Stt(R, cache_dir="data") -> csr_matrix
"""

import hashlib
import pathlib
import pickle
import time

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix


def _checksum(R: csr_matrix) -> str:
    """SHA-256 of R's shape + data + indices + indptr."""
    h = hashlib.sha256()
    h.update(np.array(R.shape, dtype=np.int64).tobytes())
    h.update(R.data.tobytes())
    h.update(R.indices.tobytes())
    h.update(R.indptr.tobytes())
    return h.hexdigest()


def load_or_compute_Stt(R: csr_matrix, cache_dir: str = "data") -> csr_matrix:
    """
    Return  S_tt = Rᵀ R  (T × T), loading from disk if the cached version
    matches the current R, otherwise recomputing and saving.

    Parameters
    ----------
    R         : csr_matrix  (P × T)  binary playlist-track matrix
    cache_dir : directory for .npz and .pkl cache files

    Returns
    -------
    S_tt : csr_matrix  (T × T)
    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    T = R.shape[1]
    npz_path = cache_dir / f"gram_tt_{T}.npz"
    meta_path = cache_dir / f"gram_tt_{T}_meta.pkl"

    R = R.tocsr().astype(np.float32)
    checksum = _checksum(R)

    # ── try cache hit ──────────────────────────────────────────────────
    if npz_path.exists() and meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("checksum") == checksum:
            print(f"[gram_cache] Cache hit — cargando S_tt desde {npz_path}")
            t = time.time()
            S = scipy.sparse.load_npz(str(npz_path)).tocsr()
            print(
                f"[gram_cache] Cargado en {time.time()-t:.2f}s  "
                f"shape={S.shape}  nnz={S.nnz:,}"
            )
            return S
        print("[gram_cache] Checksum distinto — recalculando S_tt.")

    # ── compute ────────────────────────────────────────────────────────
    print(f"[gram_cache] Calculando S_tt = Rᵀ R  (R shape={R.shape}) ...")
    t = time.time()
    S = (R.T @ R).tocsr()
    print(
        f"[gram_cache] Calculado en {time.time()-t:.2f}s  "
        f"shape={S.shape}  nnz={S.nnz:,}  "
        f"densidad={S.nnz / (S.shape[0] * S.shape[1]):.2e}"
    )

    # ── save ───────────────────────────────────────────────────────────
    scipy.sparse.save_npz(str(npz_path), S)
    with open(meta_path, "wb") as f:
        pickle.dump({"checksum": checksum, "shape": S.shape, "nnz": S.nnz}, f)
    print(f"[gram_cache] Guardado en {npz_path}")
    return S
