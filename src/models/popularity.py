"""
    popularity.py - Non-personalised popularity baseline.

    build_popularity   : counts how many training playlists contain each track,
                         returns a ranked list (most popular first).
    recommend_popular  : for a given seed, returns the top-N popular tracks
                         that are NOT already in the seed.
    write_submission   : writes the challenge submission CSV.
"""

import numpy as np


def build_popularity(A, track_to_col):
    """
    Computes track popularity as the number of playlists that contain it.

    Uses the column sums of the binary matrix A (one sum per track column).

    Parameters
    ----------
    A            : csr_matrix  (n_playlists x n_tracks)
    track_to_col : dict  {track_uri -> column index}

    Returns
    -------
    popularity_list : list of (track_uri, count) sorted by count descending
    """
    col_to_track = {v: k for k, v in track_to_col.items()}

    # A.sum(axis=0)  →  (1 x n_tracks) matrix  →  flatten to 1-D array
    counts = np.asarray(A.sum(axis=0)).flatten()

    order = np.argsort(counts)[::-1]   # indices sorted by count, descending
    return [(col_to_track[idx], int(counts[idx])) for idx in order]


def recommend_popular(seed_uris, popularity_list, top_n=500):
    """
    Returns the top_n most popular tracks that are NOT in seed_uris.

    Parameters
    ----------
    seed_uris       : iterable of track URIs already in the playlist
    popularity_list : output of build_popularity()
    top_n           : number of tracks to recommend (default 500)

    Returns
    -------
    list of track_uri strings, length <= top_n
    """
    seed_set = set(seed_uris)
    recs = []
    for uri, _ in popularity_list:
        if uri not in seed_set:
            recs.append(uri)
            if len(recs) == top_n:
                break
    return recs


def write_submission(predictions, output_path,
                     team_name="team", contact="contact@email.com",
                     description="popularity baseline"):
    """
    Writes the challenge submission file.

    Format
    ------
    team_info,<team_name>,<contact>,<description>
    <pid>,<uri_1>,<uri_2>,...,<uri_500>
    <pid>,<uri_1>,<uri_2>,...,<uri_500>
    ...

    Parameters
    ----------
    predictions : dict  {pid -> [track_uri, ...]}
    output_path : str   path to the output CSV file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"team_info,{team_name},{contact},{description}\n")
        for pid in sorted(predictions):
            tracks = ",".join(predictions[pid])
            f.write(f"{pid},{tracks}\n")
    print(f"[submission] written to {output_path}  ({len(predictions):,} playlists)")
