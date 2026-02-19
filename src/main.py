"""
    main.py – MPD playlist recommendation

    Usage
    -----
    # Full dataset, recommend 10 tracks for playlist 0:
        python main.py --path dataset/data --pid 0

    # Quick mode (only first 5 slices) with 20 recommendations:
        python main.py --path dataset/data --pid 0 --quick --top 20

    # Save the sparse matrix for reuse:
        python main.py --path dataset/data --pid 0 --save-matrix matrix.npz

    # Load a previously saved matrix (skip re-parsing JSONs):
        python main.py --load-matrix matrix.npz --pid 0
"""

import sys
import argparse
import numpy as np
import scipy.sparse

from src.utils import load_mpd, build_sparse_matrix, recommend


def parse_args():
    p = argparse.ArgumentParser(description="MPD collaborative-filtering recommender")
    p.add_argument("--path",         help="Path to MPD data directory")
    p.add_argument("--pid",          type=int, required=True,
                                     help="Playlist ID to generate recommendations for")
    p.add_argument("--top",          type=int, default=10,
                                     help="Number of tracks to recommend (default: 10)")
    p.add_argument("--quick",        action="store_true",
                                     help="Only load first 5 slices (for testing)")
    p.add_argument("--max-files",    type=int, default=5,
                                     help="Number of slices to load in quick mode (default: 5)")
    p.add_argument("--save-matrix",  metavar="FILE",
                                     help="Save the sparse matrix + metadata to a .npz file")
    p.add_argument("--load-matrix",  metavar="FILE",
                                     help="Load a previously saved matrix (skip JSON parsing)")
    return p.parse_args()


def save_matrix(path, A, playlists, pid_to_row, track_to_col):
    """Persist the matrix and index mappings so you don't have to rebuild."""
    import pickle, pathlib
    stem = pathlib.Path(path).with_suffix("")
    scipy.sparse.save_npz(str(stem) + ".npz", A)
    with open(str(stem) + "_meta.pkl", "wb") as f:
        pickle.dump({"playlists": playlists,
                     "pid_to_row": pid_to_row,
                     "track_to_col": track_to_col}, f)
    print(f"[saved] matrix → {stem}.npz   metadata → {stem}_meta.pkl")


def load_matrix(path):
    import pickle, pathlib
    stem = pathlib.Path(path).with_suffix("")
    A = scipy.sparse.load_npz(str(stem) + ".npz")
    with open(str(stem) + "_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    print(f"[loaded] {A.shape[0]} playlists × {A.shape[1]} tracks")
    return A, meta["playlists"], meta["pid_to_row"], meta["track_to_col"]


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    #  1. Load / build the sparse matrix                                   #
    # ------------------------------------------------------------------ #
    if args.load_matrix:
        A, playlists, pid_to_row, track_to_col = load_matrix(args.load_matrix)
    elif args.path:
        print(f"[loading] {'quick mode, ' + str(args.max_files) + ' slices' if args.quick else 'full dataset'} …")
        playlists, track_to_col = load_mpd(
            args.path, quick=args.quick, max_files=args.max_files
        )
        print(f"[loaded]  {len(playlists):,} playlists  |  {len(track_to_col):,} unique tracks")

        print("[building] sparse matrix …")
        A, pid_to_row = build_sparse_matrix(playlists, track_to_col)
        print(f"[matrix]  shape={A.shape}  nnz={A.nnz:,}  "
              f"density={A.nnz / (A.shape[0] * A.shape[1]):.6f}")
    else:
        print("Error: provide --path or --load-matrix")
        sys.exit(1)

    if args.save_matrix:
        save_matrix(args.save_matrix, A, playlists, pid_to_row, track_to_col)

    # ------------------------------------------------------------------ #
    #  2. Recommend                                                        #
    # ------------------------------------------------------------------ #
    pid = args.pid
    if pid not in pid_to_row:
        print(f"Error: pid {pid} not found in loaded data")
        sys.exit(1)

    row = pid_to_row[pid]
    playlist = playlists[row]

    print(f"\n{'='*60}")
    print(f"  Playlist {pid}: {playlist['name']}")
    print(f"  {playlist['num_tracks']} tracks  |  {playlist['num_followers']} followers")
    print(f"{'='*60}")

    print(f"\n  Current tracks (first 10 shown):")
    for track in playlist["tracks"][:10]:
        print(f"    • {track['track_name']}  –  {track['artist_name']}")
    if playlist["num_tracks"] > 10:
        print(f"    … and {playlist['num_tracks'] - 10} more")

    print(f"\n  Top {args.top} recommendations:")
    recs = recommend(pid, playlists, A, pid_to_row, track_to_col, top_n=args.top)

    if not recs:
        print("    (no recommendations found – try loading more data)")
    else:
        # Build a uri->name lookup from loaded playlists
        uri_to_name = {}
        for pl in playlists:
            for t in pl["tracks"]:
                if t["track_uri"] not in uri_to_name:
                    uri_to_name[t["track_uri"]] = (t["track_name"], t["artist_name"])

        for rank, (uri, score) in enumerate(recs, 1):
            name, artist = uri_to_name.get(uri, (uri, "?"))
            print(f"    {rank:>3}. [{score:6.0f} votes]  {name}  –  {artist}")

    print()


if __name__ == "__main__":
    main()
