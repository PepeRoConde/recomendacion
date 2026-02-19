import os
import json
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

def _ingest_playlist(playlist, playlists, pid_to_row, track_to_col, track_info, rows, cols):
    row = len(playlists)
    pid_to_row[playlist["pid"]] = row
    for track in playlist["tracks"]:
        uri = track["track_uri"]
        if uri not in track_to_col:
            track_to_col[uri] = len(track_to_col)
            track_info[uri] = (track["track_name"], track["artist_name"])
        rows.append(row)
        cols.append(track_to_col[uri])
    playlists.append({k: v for k, v in playlist.items() if k != "tracks"})

def load_and_build(path, quick=False, max_files=5, input_playlists_path=None):
    """
    Loads the MPD dataset and builds the sparse matrix in a single pass.

    Instead of storing playlists and iterating again, we accumulate COO
    triplets (row, col, 1) directly while parsing, then call csr_matrix(...)
    once at the end — no lil_matrix, no second loop.

    Returns
    -------
    A            : csr_matrix  (n_playlists x n_tracks), binary float32
    playlists    : list of dict  — metadata only, tracks list is stripped
    pid_to_row   : dict  {pid -> row index}
    track_to_col : dict  {track_uri -> col index}
    """
    filenames = sorted(
        f for f in os.listdir(path)
        if f.startswith("mpd.slice.") and f.endswith(".json")
    )
    if quick:
        filenames = filenames[:max_files]

    playlists    = []
    pid_to_row   = {}
    track_to_col = {}

    rows = []         # COO row indices
    cols = []         # COO col indices
    track_info = {}   # uri -> (track_name, artist_name)

    for filename in tqdm(filenames, desc="loading slices", unit="slice"):
        fullpath = os.path.join(path, filename)
        with open(fullpath, encoding="utf-8") as f:
            mpd_slice = json.load(f)

        for playlist in mpd_slice["playlists"]:
            row = len(playlists)
            pid_to_row[playlist["pid"]] = row

            for track in playlist["tracks"]:
                uri = track["track_uri"]
                if uri not in track_to_col:
                    track_to_col[uri] = len(track_to_col)
                    track_info[uri] = (track["track_name"], track["artist_name"])
                rows.append(row)
                cols.append(track_to_col[uri])

            # Keep only metadata — drop the tracks list to save memory
            playlists.append({k: v for k, v in playlist.items() if k != "tracks"})


    if input_playlists_path:
        with open(input_playlists_path, encoding="utf-8") as f:
            input_data = json.load(f)
        for playlist in tqdm(input_data["playlists"], desc="loading input playlists", unit="pl"):
            if playlist["pid"] not in pid_to_row:
                _ingest_playlist(playlist, playlists, pid_to_row, track_to_col, track_info, rows, cols)
        print(f"[input]  added {len(input_data['playlists']):,} eval input playlists")

    n_playlists = len(playlists)
    n_tracks    = len(track_to_col)

    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix(
        (data, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n_playlists, n_tracks),
    )

    return A, playlists, pid_to_row, track_to_col, track_info


