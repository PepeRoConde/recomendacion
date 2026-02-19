import numpy as np
from tqdm import tqdm


def recommend_all(A, pid_to_row, track_to_col, top_n=10, chunk_size=500):
    """
    Returns recommendations for every playlist using chunked matrix multiplication.

    Algorithm (per chunk of K playlists):
        S_chunk = A_chunk @ A.T          (K x n_playlists) similarities
        zero diagonal entries             remove self-similarity
        V_chunk = S_chunk @ A            (K x n_tracks) vote matrix
        mask tracks already in playlist
        argsort each row, take top_n

    Parameters
    ----------
    chunk_size : number of playlists to process per batch (tune to your RAM)

    Returns
    -------
    results : dict  {pid -> [(track_uri, score), ...]}
    """
    col_to_track = {v: k for k, v in track_to_col.items()}
    row_to_pid   = {v: k for k, v in pid_to_row.items()}

    n_playlists = A.shape[0]
    results = {}

    for start in tqdm(range(0, n_playlists, chunk_size),
                      desc="recommending (all)", unit="chunk"):
        end        = min(start + chunk_size, n_playlists)
        A_chunk    = A[start:end]                    # (K x n_tracks)

        S_chunk    = A_chunk.dot(A.T)                # (K x n_playlists)

        # Zero self-similarity: playlist at local row i is global row start+i
        for local_i in range(end - start):
            S_chunk[local_i, start + local_i] = 0.0

        V = S_chunk.dot(A)                           # (K x n_tracks)
        V = V.toarray()                              # dense

        # Mask already-present tracks
        already_in = A_chunk.toarray().astype(bool)
        V[already_in] = 0.0

        for local_i in range(end - start):
            global_row = start + local_i
            pid        = row_to_pid[global_row]
            votes      = V[local_i]
            top_idx    = np.argsort(votes)[::-1][:top_n]
            results[pid] = [
                (col_to_track[idx], float(votes[idx]))
                for idx in top_idx
                if votes[idx] > 0
            ]

    return results
