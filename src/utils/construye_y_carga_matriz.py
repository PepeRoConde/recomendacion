import os
import time
import json
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.models.theta import zero_contained


def construye_y_carga_matriz(path, max_jsons=5):
    """
    Loads the MPD dataset, builds the sparse matrix in a single pass,
    and filters out contained rows.

    Returns
    -------
    R            : csr_matrix  (n_playlists x n_tracks), binary float32, containment-filtered
    pid_to_row   : dict  {pid -> row index}   (indices into the *filtered* R)
    track_to_col : dict  {track_uri -> col index}
    track_info   : dict  {track_uri -> (name, artist)}
    """

    filenames = sorted(
        f
        for f in os.listdir(path)
        if f.startswith("mpd.slice.") and f.endswith(".json")
    )

    if max_jsons != -1:
        filenames = filenames[:max_jsons]

    pid_to_row_raw = {}  # pid -> row index before filtering
    track_to_col = {}

    rows = []
    cols = []
    pids_list = []  # pids in row order, for rebuilding pid_to_row after filter
    track_info = {}

    row_idx = 0  # global, never resets between files

    inicio = time.time()

    for filename in tqdm(filenames, desc="cargando JSONs", unit="json"):
        fullpath = os.path.join(path, filename)
        with open(fullpath, encoding="utf-8") as f:
            mpd_slice = json.load(f)

        for playlist in mpd_slice["playlists"]:
            pid = playlist["pid"]
            pid_to_row_raw[pid] = row_idx
            pids_list.append(pid)

            for track in playlist["tracks"]:
                uri = track["track_uri"]
                if uri not in track_to_col:
                    track_to_col[uri] = len(track_to_col)
                    track_info[uri] = (track["track_name"], track["artist_name"])
                rows.append(row_idx)
                cols.append(track_to_col[uri])

            row_idx += 1

    print(f"Parseáronse {len(filenames)} ficheiros en {time.time() - inicio:.3f}s")

    n_playlists = row_idx
    n_tracks = len(track_to_col)

    data = np.ones(len(rows), dtype=np.float32)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)

    t0 = time.time()
    R_raw = csr_matrix((data, (rows, cols)), shape=(n_playlists, n_tracks))
    print(
        f"Matriz R construída en {time.time()-t0:.3f}s  shape={R_raw.shape}  nnz={R_raw.nnz:,}"
    )

    # ── containment filter ─────────────────────────────────────────────
    t0 = time.time()
    R, kept_indices = zero_contained(R_raw)
    print(f"Filtrado de contención en {time.time()-t0:.3f}s")

    pids_list_arr = np.array(pids_list)
    kept_pids = pids_list_arr[kept_indices]
    pid_to_row = {pid: new_idx for new_idx, pid in enumerate(kept_pids)}

    print(
        f"Cargouse a matriz R con shape={R.shape}  nnz={R.nnz:,}  "
        f"densidade={R.nnz / (R.shape[0] * R.shape[1]):.6f}"
    )
    return R, pid_to_row, track_to_col, track_info
