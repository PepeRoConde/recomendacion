import json
import os
from collections import defaultdict

from tqdm import tqdm

from src.evaluation.metrics import r_precision, ndcg, clicks
from src.models.popularity import recommend_popular


def load_eval(eval_dir):
    """
    Loads the evaluation split.

    Parameters
    ----------
    eval_dir : directory containing:
               - test_input_playlists.json  (seed tracks shown to the system)
               - test_eval_playlists.json   (withheld tracks = ground truth)

    Returns
    -------
    ground_truth   : dict  {pid -> list of withheld track_uris}
    seed_tracks    : dict  {pid -> set  of seed track_uris}
    pid_to_samples : dict  {pid -> num_samples}  — for group breakdown
    """
    # eval tiene: las canciones de las playlists que se quieren predecir
    eval_path  = os.path.join(eval_dir, "test_eval_playlists.json")
    # input tiene: las playlists que se quieren predecir (con alguna canción a veces)
    input_path = os.path.join(eval_dir, "test_input_playlists.json")

    with open(eval_path,  encoding="utf-8") as f:
        eval_data  = json.load(f)
    with open(input_path, encoding="utf-8") as f:
        input_data = json.load(f)

    ground_truth = {
        p["pid"]: [t["track_uri"] for t in p["tracks"]]
        for p in eval_data["playlists"]
    }
    seed_tracks = {
        p["pid"]: {t["track_uri"] for t in p["tracks"]}
        for p in input_data["playlists"]
    }
    pid_to_samples = {
        p["pid"]: p["num_samples"]
        for p in input_data["playlists"]
    }
    return ground_truth, seed_tracks, pid_to_samples


def evaluate_all(ground_truth, seed_tracks, pid_to_samples,
                 popularity_list, top_n=500):
    """
    Runs the three metrics over every test playlist using the popularity
    baseline recommender.

    Parameters
    ----------
    ground_truth    : dict  {pid -> list of withheld track_uris}
    seed_tracks     : dict  {pid -> set  of seed track_uris}
    pid_to_samples  : dict  {pid -> num_samples}
    popularity_list : output of popularity.build_popularity()
    top_n           : recommendations per playlist (default 500)

    Returns
    -------
    overall  : (r_prec, ndcg_score, clicks_score)
    by_group : dict  {num_samples -> (r_prec, ndcg, clicks, count)}
    """

    group_rp    = defaultdict(float)
    group_ndcg  = defaultdict(float)
    group_clk   = defaultdict(float)
    group_count = defaultdict(int)

    for pid, relevant in tqdm(ground_truth.items(), desc="evaluating", unit="pl"):
        seed    = seed_tracks.get(pid, set())
        recs    = recommend_popular(seed, popularity_list, top_n=top_n)
        rel_set = set(relevant)

        rp = r_precision(recs, rel_set)
        ng = ndcg(recs, rel_set)
        cl = clicks(recs, rel_set)

        g = pid_to_samples.get(pid, -1)
        group_rp[g]    += rp
        group_ndcg[g]  += ng
        group_clk[g]   += cl
        group_count[g] += 1

    n_total = sum(group_count.values())
    overall = (
        sum(group_rp.values())   / n_total,
        sum(group_ndcg.values()) / n_total,
        sum(group_clk.values())  / n_total,
    )
    by_group = {
        g: (group_rp[g]   / group_count[g],
            group_ndcg[g] / group_count[g],
            group_clk[g]  / group_count[g],
            group_count[g])
        for g in sorted(group_count)
    }
    return overall, by_group


# ------------------------------------------------------------------ #
#  Pretty printing                                                     #
# ------------------------------------------------------------------ #

def print_results(overall, by_group, top_n):
    w = 72
    print(f"\n{'─' * w}")
    print(f"  {'samples':>8}  {'R-Prec@'+str(top_n):>12}  "
          f"{'NDCG@'+str(top_n):>12}  {'Clicks':>8}  {'n':>7}")
    print(f"{'─' * w}")
    for g, (rp, ng, cl, cnt) in by_group.items():
        print(f"  {g:>8}  {rp:>12.4f}  {ng:>12.4f}  {cl:>8.4f}  {cnt:>7,}")
    print(f"{'─' * w}")
    rp, ng, cl = overall
    n = sum(cnt for _, _, _, cnt in by_group.values())
    print(f"  {'overall':>8}  {rp:>12.4f}  {ng:>12.4f}  {cl:>8.4f}  {n:>7,}")
    print(f"{'─' * w}\n")
