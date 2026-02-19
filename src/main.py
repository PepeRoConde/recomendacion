import sys
import argparse
import pickle
import pathlib
import numpy as np
import scipy.sparse

from src.utils.load_and_build import load_and_build
from src.models import recommend, recommend_all
from src.evaluation.evaluate import load_eval, evaluate, print_results


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

    # Modes (mutually exclusive)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pid",          type=int,
                                        help="Recommend for a single playlist id")
    mode.add_argument("--all",          action="store_true",
                                        help="Recommend for ALL playlists (chunked matrix ops)")
    mode.add_argument("--evaluate",     metavar="EVAL_DIR",
                                        help="Evaluate R-Precision, NDCG, and Clicks")

    # Output
    p.add_argument("--top",             type=int, default=10,
                                        help="Tracks to recommend for --pid/--all (default: 10). "
                                             "For --evaluate, defaults to 500 per challenge spec.")
    p.add_argument("--chunk-size",      type=int, default=500,
                                        help="Playlists per chunk (default: 500)")

    # Blending
    p.add_argument("--alpha",           type=float, default=0.5,
                                        help="CF vs reputation blend weight: "
                                             "1.0 = pure CF, 0.0 = pure popularity (default: 0.5)")
    # Evaluation
    p.add_argument("--eval-sample", type=int, default=None,
                   help="Randomly sample N playlists from the eval set for a quick test")
    p.add_argument("--eval-seed",   type=int, default=42,
                   help="Random seed for --eval-sample (default: 42)")


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

    if not 0.0 <= args.alpha <= 1.0:
        print("Error: --alpha must be between 0.0 and 1.0")
        sys.exit(1)

    # --- Load / build ------------------------------------------------
    if args.load_matrix:
        A, playlists, pid_to_row, track_to_col, track_info = load_matrix(args.load_matrix)
    elif args.path:
        A, playlists, pid_to_row, track_to_col, track_info = load_and_build(
            args.path, quick=args.quick, max_files=args.max_files,
            input_playlists_path=args.input_playlists,
        )
        print(f"[matrix]  shape={A.shape}  nnz={A.nnz:,}  "
              f"density={A.nnz / (A.shape[0] * A.shape[1]):.6f}")
    else:
        print("Error: provide --path or --load-matrix")
        sys.exit(1)

    if args.save_matrix:
        save_matrix(args.save_matrix, A, playlists, pid_to_row, track_to_col, track_info)

    alpha_label = (
        "pure CF"         if args.alpha == 1.0 else
        "pure reputation" if args.alpha == 0.0 else
        f"CF {args.alpha:.0%} / reputation {1-args.alpha:.0%}"
    )
    print(f"[blend]   alpha={args.alpha:.2f}  ({alpha_label})")

    # --- Mode: single pid --------------------------------------------
    if args.pid is not None:
        pid = args.pid
        if pid not in pid_to_row:
            print(f"Error: pid {pid} not found in loaded data")
            sys.exit(1)

        playlist = playlists[pid_to_row[pid]]
        print(f"\n{'='*60}")
        print(f"  Playlist {pid}: \"{playlist['name']}\"")
        print(f"  {playlist['num_tracks']} tracks  |  {playlist['num_followers']} followers")
        print(f"{'='*60}")
        print(f"\n  Top {args.top} recommendations:")

        recs = recommend(pid, A, pid_to_row, track_to_col,
                         top_n=args.top, alpha=args.alpha)
        if not recs:
            print("    (no recommendations — try loading more data)")
        else:
            for rank, (uri, score) in enumerate(recs, 1):
                name, artist = track_info.get(uri, (uri, "?"))
                print(f"    {rank:>3}. [{score:.4f}]  {name}  -  {artist}")
        print()

    # --- Mode: all playlists -----------------------------------------
    elif args.all:
        print(f"\n[recommend-all] top {args.top} per playlist, "
              f"chunk_size={args.chunk_size}")
        results = recommend_all(
            A, pid_to_row, track_to_col,
            top_n=args.top, chunk_size=args.chunk_size, alpha=args.alpha
        )
        print(f"\n[done] recommendations generated for {len(results):,} playlists")
        sample_pid = next(iter(results))
        print(f"\nSample — playlist {sample_pid}:")
        for rank, (uri, score) in enumerate(results[sample_pid], 1):
            name, artist = track_info.get(uri, (uri, "?"))
            print(f"  {rank:>3}. [{score:.4f}]  {name}  -  {artist}")

    # --- Mode: evaluate ----------------------------------------------
    elif args.evaluate:
        # Default to 500 for evaluation (challenge spec) unless overridden
        eval_top_n = args.top if args.top != 10 else 500
    
        print(f"\n[evaluate] loading ground truth from {args.evaluate} …")
        ground_truth, pid_to_samples = load_eval(args.evaluate)
        print(f"[evaluate] {len(ground_truth):,} test playlists found  |  top_n={eval_top_n}")
    
        if args.eval_sample:
            import random
            rng     = random.Random(args.eval_seed)
            pids    = rng.sample(list(ground_truth.keys()),
                                 min(args.eval_sample, len(ground_truth)))
            ground_truth   = {p: ground_truth[p]   for p in pids}
            pid_to_samples = {p: pid_to_samples[p] for p in pids if p in pid_to_samples}
            print(f"[evaluate] sampled {len(ground_truth):,} playlists  (seed={args.eval_seed})")
    
        overall, by_group = evaluate(
            ground_truth, pid_to_samples, pid_to_row, A, track_to_col,
            top_n=eval_top_n, chunk_size=args.chunk_size, alpha=args.alpha,
        )
        print_results(overall, by_group, top_n=eval_top_n)

if __name__ == "__main__":
    main()
