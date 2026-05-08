from .popularity import PopularityRecommender
from .playlist_neighbourhood import PlaylistNeighbourhoodRecommender
from .pure_svd import PureSVDRecommender
from .track_neighbourhood import TrackNeighbourhoodRecommender
from .SSLIM import SSLIM
from .FISM import FISM

models = {
    "popularity": PopularityRecommender,
    "playlist-neighbourhood": PlaylistNeighbourhoodRecommender,
    "pure-svd": PureSVDRecommender,
    "track-neighbourhood": TrackNeighbourhoodRecommender,
    "SSLIM": SSLIM,
    "FISM": FISM,
}
