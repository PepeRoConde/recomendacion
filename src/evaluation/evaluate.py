import json
import os
import time
from collections import defaultdict

from src.evaluation.metrics import r_precision, ndcg, clicks


def load_eval(eval_dir):
    # eval tiene: las canciones de las playlists que se quieren predecir, es
    # decir, lo que vale para comparar las predicciones del modelo
    eval_path = os.path.join(eval_dir, "test_eval_playlists.json")
    # input tiene: las playlists que se quieren predecir (con alguna canción a veces),
    # vale para saber que se tiene que predecir
    input_path = os.path.join(eval_dir, "test_input_playlists.json")

    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(input_path, encoding="utf-8") as f:
        input_data = json.load(f)

    ground_truth = {
        p["pid"]: [t["track_uri"] for t in p["tracks"]] for p in eval_data["playlists"]
    }
    pid_to_uris = {
        p["pid"]: {t["track_uri"] for t in p["tracks"]} for p in input_data["playlists"]
    }
    # num_samples es: cuantas canciones tenia esa playlist en entrenamiento.
    # puede ser 0, en ese caso hay un _cold_start_
    pid_to_num_samples = {p["pid"]: p["num_samples"] for p in input_data["playlists"]}

    return ground_truth, pid_to_uris, pid_to_num_samples


def evaluate(model, ground_truth, pid_to_uri, pid_to_num_samples, top_n=500):
    """
    Parameters
    ----------
    model          : BaseRecommender  — must have .recommend_batch(seeds, top_n)
    ground_truth   : dict  {pid -> list of withheld track_uris}
    pid_to_uri    : dict  {pid -> set  of seed track_uris}
    pid_to_num_samples : dict  {pid -> num_samples}
    top_n          : recommendations per playlist (default 500)

    Returns
    -------
    overall  : (r_prec, ndcg_score, clicks_score)
    by_group : dict  {num_samples -> (r_prec, ndcg, clicks, count)}
    """
    group_rp = defaultdict(float)
    group_ndcg = defaultdict(float)
    group_clk = defaultdict(float)
    group_count = defaultdict(int)

    pids = list(ground_truth.keys())
    seeds = [pid_to_uri.get(pid, set()) for pid in pids]

    print(f"Evaluando o modelo {model.name} con {len(pids):,} playlists de test...")
    inicio = time.time()
    all_recs = model.recommend_batch(seeds, top_n=top_n)  # list[list[str]]
    print(f"... tardou {time.time() - inicio:.3f}s")

    for pid, recs in zip(pids, all_recs):
        rel_set = set(ground_truth[pid])

        rp = r_precision(recs, rel_set)
        ng = ndcg(recs, rel_set)
        cl = clicks(recs, rel_set)

        g = pid_to_num_samples.get(pid, -1)
        group_rp[g] += rp
        group_ndcg[g] += ng
        group_clk[g] += cl
        group_count[g] += 1

    n_total = sum(group_count.values())
    overall = (
        sum(group_rp.values()) / n_total,
        sum(group_ndcg.values()) / n_total,
        sum(group_clk.values()) / n_total,
    )
    by_group = {
        g: (
            group_rp[g] / group_count[g],
            group_ndcg[g] / group_count[g],
            group_clk[g] / group_count[g],
            group_count[g],
        )
        for g in sorted(group_count)
    }
    return overall, by_group
