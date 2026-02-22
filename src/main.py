"""
    main.py - MPD playlist recommendation  (iteration 0 — popularity baseline)

    Usage
    -----
    # Build matrix from raw data:
        python main.py --path dataset/data/train

    # Quick mode (only first N slices):
        python main.py --path dataset/data/train --quick --max-files 5

    # Save/load matrix to skip re-parsing:
        python main.py --path dataset/data/train --save-matrix matrix
        python main.py --load-matrix matrix

    # Popularity baseline + evaluate + write submission:
        python main.py --load-matrix matrix \\
                       --baseline \\
                       --eval-dir dataset/data/eval \\
                       --output submission.csv \\
                       --top-n 500
"""

import sys
import argparse
import pickle
import pathlib
import scipy.sparse

from src.utils.load_and_build import load_and_build
from src.popularity            import build_popularity, recommend_popular, write_submission
from src.evaluation.evaluate   import load_eval, evaluate_all, print_results


def parse_args():
    p = argparse.ArgumentParser(description="MPD collaborative-filtering recommender")

    # Data
    p.add_argument("--path",            help="Path to MPD training data directory")
    p.add_argument("--save-matrix",     metavar="FILE",
                                        help="Save matrix + metadata to FILE.npz / FILE_meta.pkl")
    p.add_argument("--load-matrix",     metavar="FILE",
                                        help="Load a previously saved matrix")
    p.add_argument("--input-playlists", metavar="FILE",
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


# ------------------------------------------------------------------ #
#  Matrix persistence                                                  #
# ------------------------------------------------------------------ #

def save_matrix(stem, A, playlists, pid_to_row, track_to_col, track_info):
    stem = pathlib.Path(stem)
    scipy.sparse.save_npz(str(stem) + ".npz", A)
    with open(str(stem) + "_meta.pkl", "wb") as f:
        pickle.dump({"playlists":    playlists,
                     "pid_to_row":   pid_to_row,
                     "track_to_col": track_to_col,
                     "track_info":   track_info}, f)
    print(f"[saved] {stem}.npz  +  {stem}_meta.pkl")


def load_matrix(stem):
    stem = pathlib.Path(stem)
    A = scipy.sparse.load_npz(str(stem) + ".npz")
    with open(str(stem) + "_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    n = len(meta["playlists"])
    print(f"[loaded] {n:,} playlists x {A.shape[1]:,} tracks")
    return A, meta["playlists"], meta["pid_to_row"], meta["track_to_col"], meta["track_info"]


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    # --- Load / build ------------------------------------------------
    if args.load_matrix:
        A, playlists, pid_to_row, track_to_col, track_info = load_matrix(args.load_matrix)
    elif args.path:
        A, playlists, pid_to_row, track_to_col, track_info = load_and_build(
            args.path, quick=args.quick, max_files=args.max_files,
            input_playlists_path=args.input_playlists,
        )
    else:
        print("Error: provide --path or --load-matrix")
        sys.exit(1)

    print(f"[matrix]  shape={A.shape}  nnz={A.nnz:,}  "
          f"density={A.nnz / (A.shape[0] * A.shape[1]):.6f}")

    if args.save_matrix:
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
