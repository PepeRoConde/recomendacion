import os
import time
import json
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


def construye_y_carga_matriz(path, max_jsons=5):
    """
    Loads the MPD dataset and builds the sparse matrix in a single pass.

    Returns
    -------
    R            : csr_matrix  (n_playlists x n_tracks), binary float32
    pid_to_row   : dict  {pid -> row index}
    track_to_col : dict  {track_uri -> col index}
    track_info   : dict  {track_uri -> (name, artist)}
    """

    # lista de los nombres de los JSONs
    filenames = sorted(
        f
        for f in os.listdir(path)
        if f.startswith("mpd.slice.") and f.endswith(".json")
    )

    if max_jsons != -1:
        filenames = filenames[:max_jsons]

    pid_to_row = {}
    track_to_col = {}

    rows = []  # COO row indices
    cols = []  # COO col indices
    track_info = {}  # uri -> (track_name, artist_name)

    for filename in tqdm(filenames, desc="loading slices", unit="slice"):
        fullpath = os.path.join(path, filename)
        with open(fullpath, encoding="utf-8") as f:
            mpd_slice = json.load(f)

        for i, playlist in enumerate(mpd_slice["playlists"]):
            pid_to_row[playlist["pid"]] = i
            for track in playlist["tracks"]:
                uri = track["track_uri"]

                # si es la primera vez que procesamos esa cancion
                if uri not in track_to_col:
                    track_to_col[uri] = len(track_to_col) # le asignamos columna
                    track_info[uri] = (track["track_name"], track["artist_name"]) # y metadata

                rows.append(i) # por cada cancion de la playlist i, metemos i en _rows_
                cols.append(track_to_col[uri]) # metemos indice de columna de esa cancion en _cols_

    n_playlists = len(pid_to_row.keys())
    n_tracks = len(track_to_col.keys())

    data = np.ones(len(rows), dtype=np.float32) # tantos unos como ocurrencias de canciones en playlists
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    
    inicio = time.time()
    R = csr_matrix((data, (rows, cols)), shape=(n_playlists, n_tracks))
    print(f'La matriz R tardo {time.time() - inicio}"  en construirse con csr_matrix()')

    print(f"Se cargo la matriz R con  shape={R.shape},  nnz={R.nnz:,} densidad={R.nnz / (R.shape[0] * R.shape[1]):.6f}")

    return R, pid_to_row, track_to_col, track_info
