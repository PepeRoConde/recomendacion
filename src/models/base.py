from abc import ABC, abstractmethod

class BaseRecommender(ABC):

    @abstractmethod
    def fit(self, A, track_to_col, **kwargs): ...

    @abstractmethod
    def recommend_batch(self, seeds, top_n=500) -> list[list[str]]: ...

    def recommend(self, seed_uris, top_n=500) -> list[str]:
        return self.recommend_batch([seed_uris], top_n=top_n)[0]

    @property
    @abstractmethod
    def name(self) -> str: ...
