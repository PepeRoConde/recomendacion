import pathlib
import scipy
import pickle
import time


def carga_matriz(stem):
    stem = pathlib.Path(stem)

    inicio = time.time()
    R = scipy.sparse.load_npz(str(stem) + ".npz")
    with open(str(stem) + "_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    print(
        f"Tardouse {time.time() - inicio:.3f}s en cargar a matriz R con: shape={R.shape},  nnz={R.nnz:,} densidade={R.nnz / (R.shape[0] * R.shape[1]):.6f}"
    )

    return (
        R,
        meta["pid_to_row"],
        meta["track_to_col"],
        meta["track_info"],
    )
