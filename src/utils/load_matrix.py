
def load_matrix(stem):
    stem = pathlib.Path(stem)
    A = scipy.sparse.load_npz(str(stem) + ".npz")
    with open(str(stem) + "_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    n = len(meta["playlists"])
    print(f"[loaded] {n:,} playlists x {A.shape[1]:,} tracks")
    return A, meta["playlists"], meta["pid_to_row"], meta["track_to_col"], meta["track_info"]
