import csv
import pathlib


def write_results(overall, by_group, top_n, model_name, max_jsons, k):
    """
    Write results to  results/{model_name}_{max_jsons}jsons_{k}k.csv

    Columns: samples, r_precision, ndcg, clicks, n
    One row per num_samples group, plus a final 'overall' row.
    """

    jsons_tag = f"{max_jsons}" if max_jsons != -1 else "all"
    pathlib.Path("results").mkdir(exist_ok=True)
    path = pathlib.Path(f"results/{model_name}_{jsons_tag}jsons_{k}k.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["samples", f"r_precision@{top_n}", f"ndcg@{top_n}", "clicks", "n"]
        )
        for g, (rp, ng, cl, cnt) in by_group.items():
            writer.writerow([g, round(rp, 6), round(ng, 6), round(cl, 6), cnt])
        rp, ng, cl = overall
        n = sum(cnt for _, _, _, cnt in by_group.values())
        writer.writerow(["overall", round(rp, 6), round(ng, 6), round(cl, 6), n])

    print(f"Resultados gardados en {path}")


def print_results(overall, by_group, top_n):
    w = 72
    print(f"\n{'─' * w}")
    print(
        f"  {'samples':>8}  {'R-Prec@'+str(top_n):>12}  "
        f"{'NDCG@'+str(top_n):>12}  {'Clicks':>8}  {'n':>7}"
    )
    print(f"{'─' * w}")
    for g, (rp, ng, cl, cnt) in by_group.items():
        print(f"  {g:>8}  {rp:>12.4f}  {ng:>12.4f}  {cl:>8.4f}  {cnt:>7,}")
    print(f"{'─' * w}")
    rp, ng, cl = overall
    n = sum(cnt for _, _, _, cnt in by_group.values())
    print(f"  {'overall':>8}  {rp:>12.4f}  {ng:>12.4f}  {cl:>8.4f}  {n:>7,}")
    print(f"{'─' * w}\n")
