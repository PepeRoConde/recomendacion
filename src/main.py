import argparse
import time

from src.utils.carga_o_construye_matriz import carga_o_construye_matriz

from src.models.popularity import PopularityRecommender
from src.models.playlist_neighbourhood import PlaylistNeighbourhoodRecommender
from src.models.pure_svd import PureSVDRecommender
from src.models.track_neighbourhood import TrackNeighbourhoodRecommender
from src.evaluation.evaluate import load_eval, evaluate
from src.evaluation.logging import print_results, write_results


MODELOS = {
    "popularity": PopularityRecommender,
    "playlist-neighbourhood": PlaylistNeighbourhoodRecommender,
    "pure-svd": PureSVDRecommender,
    "track-neighbourhood": TrackNeighbourhoodRecommender,
}


def parse_args():
    p = argparse.ArgumentParser(description="Sistemas de Recomendación — Marcos & Pepe")

    # ── matrix ────────────────────────────────────────────────────────
    p.add_argument(
        "--train-dir",
        metavar="DIR",
        default=None,
        help="Ruta aos JSONs descomprimidos de MPD (mpd.slice.*.json). Só necesario a primeira vez (cando non hai caché).",
    )
    p.add_argument(
        "--data-dir",
        metavar="DIR",
        default="data",
        help="Directorio onde gardar/cargar matrices e cachés (default: data/)",
    )
    p.add_argument(
        "--max_jsons",
        type=int,
        default=-1,
        help="Número de JSONs a usar (-1 = todos). Determina o nome do ficheiro de caché.",
    )

    # ── model ─────────────────────────────────────────────────────────
    p.add_argument(
        "--modelo",
        choices=MODELOS.keys(),
        default="popularity",
    )
    p.add_argument(
        "--k",
        type=int,
        default=20,
        help="Tamaño do vecindario para modelos neighbourhood (default: 20)",
    )
    p.add_argument(
        "--use-idf",
        action="store_true",
        help="Activa ponderación IDF (default: False)",
    )
    p.add_argument(
        "--folding-in",
        action="store_true",
        help="Activa o modo cold-start puro do PureSVD (default: False)",
    )
    p.add_argument(
        "--use-ann",
        action="store_true",
        help="Activa recuperación aproximada de candidatos con HNSW sobre PureSVD (requiere folding-in)",
    )
    p.add_argument(
        "--ann-candidates",
        type=int,
        default=20000,
        help="Número de candidatos ANN por playlist antes del reranking exacto (default: 20000)",
    )

    # ── evaluation ────────────────────────────────────────────────────
    p.add_argument(
        "--eval-dir",
        metavar="DIR",
        default=None,
        help="Directorio con test_input_playlists.json e test_eval_playlists.json descomprimidos",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="Canciones a recomendar por playlist (default: 500)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # ── load / build R ────────────────────────────────────────────────
    R, _pid_to_row, track_to_col, _track_info = carga_o_construye_matriz(
        args.train_dir, args.data_dir, args.max_jsons
    )

    # ── fit ───────────────────────────────────────────────────────────
    modelo = MODELOS[args.modelo]()

    t0 = time.time()
    modelo.fit(
        R,
        track_to_col,
        k=args.k,
        cache_dir=args.data_dir,
        use_idf=args.use_idf,
        folding_in=args.folding_in,
        use_ann=args.use_ann,
        ann_candidates=args.ann_candidates,
    )
    print(f"Tardouse {time.time()-t0:.3f}s en axustar o modelo '{modelo.name}'")

    # ── evaluate ──────────────────────────────────────────────────────
    if args.eval_dir:
        ground_truth, pid_to_uris, pid_to_num_samples = load_eval(args.eval_dir)
        print(f"Atopáronse {len(ground_truth):,} playlists de test en {args.eval_dir}")

        overall, by_group = evaluate(
            modelo,
            ground_truth,
            pid_to_uris,
            pid_to_num_samples,
            top_n=args.top_n,
        )

        print_results(overall, by_group, top_n=args.top_n)
        write_results(
            overall, by_group, args.top_n, modelo.name, args.max_jsons, args.k
        )


if __name__ == "__main__":
    main()
