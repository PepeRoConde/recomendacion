from abc import ABC, abstractmethod
import numpy as np


class BaseRecommender(ABC):

    @abstractmethod
    def recommend(self, pid, A, pid_to_row, track_to_col, top_n=10):
        """
        Return top_n track recommendations for a single playlist.

        Parameters
        ----------
        pid          : playlist id
        A            : sparse (n_playlists x n_tracks) binary matrix
        pid_to_row   : dict {pid -> row index}
        track_to_col : dict {track_uri -> col index}
        top_n        : number of results to return

        Returns
        -------
        list of (track_uri, score)  length <= top_n, descending by score
        """

    @abstractmethod
    def recommend_batch(self, A_chunk, chunk_rows, A, track_to_col, top_n=10):
        """
        Return a dense score matrix for a chunk of playlists.

        Parameters
        ----------
        A_chunk    : sparse matrix  (K x n_tracks)  rows for this chunk
        chunk_rows : list of global row indices, length K
        A          : full sparse matrix  (n_playlists x n_tracks)
        track_to_col : dict {track_uri -> col index}
        top_n      : hint — models may use this to limit internal work

        Returns
        -------
        V : np.ndarray  (K x n_tracks)
            Raw (unnormalised) scores. Already-present tracks should be
            zeroed out. Normalisation and blending are handled externally.
        """
