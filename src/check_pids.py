import json, pickle

with open("data/matrix_meta.pkl", "rb") as f:
    meta = pickle.load(f)

pid_to_row = meta["pid_to_row"]

with open("data/dataset/eval/test_eval_playlists.json") as f:
    eval_data = json.load(f)

eval_pids = [p["pid"] for p in eval_data["playlists"]]

found    = sum(1 for p in eval_pids if p in pid_to_row)
missing  = sum(1 for p in eval_pids if p not in pid_to_row)

print(f"Total eval pids : {len(eval_pids)}")
print(f"Found in matrix : {found}")
print(f"Missing         : {missing}")
print(f"\nFirst 10 eval pids : {eval_pids[:10]}")
print(f"Matrix pid range   : {min(pid_to_row)} – {max(pid_to_row)}")
