import sys
import argparse
import time

from src.utils import carga_matriz, construye_y_carga_matriz, guarda_matriz
from src.models.popularity import PopularityRecommender
from src.models.playlist_neighbourhood import PlaylistNeighbourhoodRecommender
from src.models.track_neighbourhood import TrackNeighbourhoodRecommender
from src.evaluation.evaluate import load_eval, evaluate, print_results


MODELOS = {
    "popularity": PopularityRecommender,
    "playlist-neighbourhood": PlaylistNeighbourhoodRecommender,
    "track-neighbourhood": TrackNeighbourhoodRecommender,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Practica de Marcos y Pepe para la asignatura Sistemas de Recomendación"
    )

    group = p.add_mutually_exclusive_group()

    # Carga de datos
    # ruta_mpd esta por defecto a None para que, por defecto, no se construya la matriz desde los JSONs
    # y en vez se carga la matriz desde '--carga_matriz'
    group.add_argument(
        "--construye_matriz",
        metavar="RUTA_MPD_TRAIN",
        default=None,
        help="Ruta al dataset de entrenamiento de MPD. Si se especifica se construira la matriz R y otra de metadatos y se guardarán con el nombre especificado en --guarda_matriz",
    )

    group.add_argument(
        "--carga_matriz",
        metavar="ARCHIVO",
        default=None,
        help="Ruta para cargar la matriz y metadata desde ARCHIVO.npz / ARCHIVO_meta.pkl",
    )

    p.add_argument(
        "--guarda_matriz",
        metavar="ARCHIVO",
        default="data/matriz",
        help="Ruta para guardar la matriz y metadata a ARCHIVO.npz / ARCHIVO_meta.pkl que se construyo con los JSONs de --ruta_mpd",
    )

    p.add_argument(
        "--max_jsons",
        type=int,
        default=-1,
        help="Numero de archivos JSON que se usaran para construir R, si no se especifica se usaran todos",
    )

    # Evaluacion
    p.add_argument("--modelo", choices=MODELOS.keys(), default="popularity")

    p.add_argument(
        "--k",
        type=int,
        default=20,
        help="Tamaño do vecindario para modelos baseados en vecindario",
    )
    p.add_argument(
        "--cache_dir",
        metavar="DIR",
        default="data/caches/",
        help="Directorio para a caché de matrices de similitud",
    )

    p.add_argument(
        "--eval-dir",
        metavar="DIR",
        help="Directorio con los JSONs test_input/test_eval",
    )
    p.add_argument(
        "--salida",
        metavar="ARCHIVO",
        default="salida",
        help="Escribe la salida del modelo a ARCHIVO.csv, por defecto 'salida.csv' ",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="Numero de canciones para recomendar por playlist",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.carga_matriz:
        R, pid_to_row, track_to_col, track_info = carga_matriz(args.carga_matriz)

    elif args.construye_matriz:
        R, pid_to_row, track_to_col, track_info = construye_y_carga_matriz(
            args.construye_matriz, max_jsons=args.max_jsons
        )

        guarda_matriz(args.guarda_matriz, R, pid_to_row, track_to_col, track_info)
    else:
        print(
            "Error cargando a matriz R. Especificaches ben os argumentos? Corre python -m src.main -h"
        )
        sys.exit(1)

    modelo = MODELOS[args.modelo]()

    inicio = time.time()
    modelo.fit(R, track_to_col, k=args.k, cache_dir=args.cache_dir)
    print(f"Axustouse o modelo {args.modelo} en  {time.time() - inicio:.3f}s")

    if args.eval_dir:
        ground_truth, pid_to_uris, pid_to_num_samples = load_eval(args.eval_dir)
        print(
            f"Atopáronse #{len(ground_truth):,} playlists de test no directorio {args.eval_dir}"
        )

        overall, by_group = evaluate(
            modelo, ground_truth, pid_to_uris, pid_to_num_samples, top_n=args.top_n
        )

        # write_submission(predictions, args.output)

        print_results(overall, by_group, top_n=args.top_n)


if __name__ == "__main__":
    main()
