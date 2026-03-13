import pathlib
import sys

from .carga_matriz import carga_matriz
from .guarda_matriz import guarda_matriz
from .construye_y_carga_matriz import construye_y_carga_matriz


def _matriz_stem(data_dir: pathlib.Path, max_jsons: int) -> pathlib.Path:
    """Canonical path stem for a matrix built from max_jsons slices (or 'all')."""
    suffix = str(max_jsons) if max_jsons != -1 else "all"
    return data_dir / f"matriz_{suffix}"


def carga_o_construye_matriz(train_dir: str, data_dir: str, max_jsons: int):
    """
    Load R from disk if a matrix for this max_jsons already exists,
    otherwise build it from train_dir, save it, and return it.

    Files on disk:  data/matriz_20.npz / data/matriz_20_meta.pkl
                    data/matriz_all.npz / data/matriz_all_meta.pkl  (max_jsons=-1)
    """
    data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stem = _matriz_stem(data_dir, max_jsons)

    npz_path = pathlib.Path(str(stem) + ".npz")

    if npz_path.exists():
        print(f"[matriz] Atopouse caché en {npz_path} — cargando.")
        return carga_matriz(str(stem))

    if train_dir is None:
        print(
            f"[matriz] Non se atopou {npz_path} e non se especificou --train-dir.\n"
            "Usa --train-dir para construír a matriz por primeira vez.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[matriz] Non se atopou caché — construíndo desde {train_dir} ...")
    R, pid_to_row, track_to_col, track_info = construye_y_carga_matriz(
        train_dir, max_jsons=max_jsons
    )
    guarda_matriz(str(stem), R, pid_to_row, track_to_col, track_info)
    return R, pid_to_row, track_to_col, track_info
