
def save_matrix(stem, A, playlists, pid_to_row, track_to_col, track_info):
    stem = pathlib.Path(stem)
    scipy.sparse.save_npz(str(stem) + ".npz", A)
    with open(str(stem) + "_meta.pkl", "wb") as f:
        pickle.dump({"playlists":    playlists,
                     "pid_to_row":   pid_to_row,
                     "track_to_col": track_to_col,
                     "track_info":   track_info}, f)
    print(f"[saved] {stem}.npz  +  {stem}_meta.pkl")
