import numpy as np
from .base import BaseRecommender

class PopularityRecommender(BaseRecommender):

    def fit(self, A, track_to_col, **kwargs):
        col_to_track = {v: k for k, v in track_to_col.items()}
        counts = np.asarray(A.sum(axis=0)).flatten()
        order = np.argsort(counts)[::-1]
        self.popularity_list = [(col_to_track[idx], int(counts[idx])) for idx in order]

    def recommend(self, seed_uris, top_n=500):
        seed_set = set(seed_uris)
        recs = []
        for uri, _ in self.popularity_list:
            if uri not in seed_set:
                recs.append(uri)
                if len(recs) == top_n:
                    break
        return recs

    def recommend_batch(self, seeds, top_n=500):
        return [self.recommend(seed, top_n=top_n) for seed in seeds]

    @property
    def name(self):
        return "popularity-baseline"
