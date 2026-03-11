import sys
import argparse
import pickle
import pathlib
import scipy.sparse

from src.utils.load_and_build import load_and_build_matrix_R
from src.popularity            import build_popularity, recommend_popular, write_submission
from src.evaluation.evaluate   import load_eval, evaluate_all, print_results

def parse_args():
    p = argparse.ArgumentParser(description="Practica de Marcos y Pepe para la asignatura Sistemas de Recomendación")

    # Data
    p.add_argument("--mpd_path",     
                   default="data/dataset/train",
                   help="Ruta al dataset de entrenamiento de MPD")
    p.add_argument("--guarda_matriz",     
                   metavar="ARCHIVO",
                    help="Ruta para guuardar la matrix y metadata a ARCHIVO.npz / ARCHIVO_meta.pkl. Si no se especifica no se guarda.")
    p.add_argument("--carga-matriz",     
                   metavar="ARCHIVO",
                    help="Carga la matriz desde una ruta (si hay una guardada ahí)")
    p.add_argument("--test_playlists_path", metavar="FILE",
                                        help="Path to test_input_playlists.json — adds eval pids into the matrix")
    p.add_argument("--quick",           action="store_true",
                                        help="Only load first --max-files slices")
    p.add_argument("--max-files",       type=int, default=5)

    # Baseline / evaluation
    p.add_argument("--baseline",        action="store_true",
                                        help="Run popularity baseline recommender")
    p.add_argument("--eval-dir",        metavar="DIR",
                                        help="Directory with test_input/test_eval JSON files")
    p.add_argument("--output",          metavar="FILE",
                                        help="Write challenge submission CSV to FILE")
    p.add_argument("--top-n",           type=int, default=500,
                                        help="Tracks to recommend per playlist (default: 500)")

    return p.parse_args()


def main():
    args = parse_args()

    # --- Load / build ------------------------------------------------
    if args.load_matrix:
        A, playlists, pid_to_row, track_to_col, track_info = load_matrix(args.load_matrix)
    elif args.mpd_path:
        A, playlists, pid_to_row, track_to_col, track_info = load_and_build(
            args.mpd_path, quick=args.quick, max_files=args.max_files,
            test_playlists_path=args.test_playlists_path,
        )
    else:
        print("Error: provide --path or --load-matrix")
        sys.exit(1)

    print(f"[matrix]  shape={A.shape}  nnz={A.nnz:,}  "
          f"density={A.nnz / (A.shape[0] * A.shape[1]):.6f}")

    if args.guarda_matriz:
        save_matrix(args.save_matrix, A, playlists, pid_to_row, track_to_col, track_info)

    # --- Mode: popularity baseline -----------------------------------
    if args.baseline:
        print("\n[baseline] computing track popularity …")
        popularity_list = build_popularity(A, track_to_col)
        print(f"[baseline] top-3 tracks: "
              f"{', '.join(uri for uri, _ in popularity_list[:3])}")

        if args.eval_dir:
            print(f"[baseline] loading eval data from {args.eval_dir} …")
            ground_truth, seed_tracks, pid_to_samples = load_eval(args.eval_dir)
            print(f"[baseline] {len(ground_truth):,} test playlists found")

            if args.output:
                print(f"[baseline] generating {args.top_n} recommendations per playlist …")
                predictions = {
                    pid: recommend_popular(seed_tracks.get(pid, set()),
                                          popularity_list, top_n=args.top_n)
                    for pid in ground_truth
                }
                write_submission(predictions, args.output)

            overall, by_group = evaluate_all(
                ground_truth, seed_tracks, pid_to_samples,
                popularity_list, top_n=args.top_n
            )
            print_results(overall, by_group, top_n=args.top_n)

        elif args.output:
            print("Error: --output requires --eval-dir to obtain the playlist seeds")
            sys.exit(1)


if __name__ == "__main__":
    main()
