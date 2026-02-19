import numpy as np

def recommend(pid, A, pid_to_row, track_to_col, top_n=10):
    """
    Recommends tracks for the playlist identified by `pid`.

    Algorithm
    ---------
    Let A be the (playlists x tracks) binary matrix.

    1.  sim = A @ A[i].T  ->  similarity of every playlist to playlist i
    2.  Zero out sim[i]   ->  remove self
    3.  votes = sim.T @ A ->  weighted sum of neighbour track vectors
    4.  Zero out tracks already in playlist i
    5.  argsort descending, return top_n

    Returns
    -------
    list of (track_uri, vote_score), length <= top_n
    """
    col_to_track = {v: k for k, v in track_to_col.items()}

    row_i = pid_to_row[pid]
    a_i   = A[row_i]                        # sparse (1 x n_tracks)

    # Step 1 - playlist similarities
    sim = A.dot(a_i.T)                      # (n_playlists x 1)

    # Step 2 - remove self
    sim[row_i] = 0.0

    # Step 3 - votes per track
    votes = sim.T.dot(A)                    # (1 x n_tracks), still sparse
    votes = votes.toarray().flatten()       # dense 1-D

    # Step 4 - mask tracks already present
    already_in = a_i.toarray().flatten().astype(bool)
    votes[already_in] = 0.0

    # Step 5 - rank
    top_idx = np.argsort(votes)[::-1][:top_n]

    return [
        (col_to_track[idx], float(votes[idx]))
        for idx in top_idx
        if votes[idx] > 0
    ]

