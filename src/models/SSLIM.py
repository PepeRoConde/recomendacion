import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

from tqdm import tqdm

# from src.utils import matriz_a_tensor, inicializa_S
from src.utils.matriz_a_tensor import matriz_a_tensor
from src.utils.inicializa import inicializa
from src.models.base import BaseRecommender


class SSLIM(BaseRecommender):
    # pide dim para ser compatible  pero me parece una chambonada hacerlo asi
    def fit(self, R, track_to_col, epochs, dim=0, lr=0.01, positiva=False):
        @tf.function
        def train_step(R, S, optimizer, lmb1=1, lmb2=1, lmbdiag=1):
            with tf.GradientTape() as t:
                diag = tf.norm(tf.linalg.diag_part(S), ord=2)
                RS = tf.sparse.sparse_dense_matmul(R, S)
                residuo = tf.norm(tf.sparse.add(R, -RS), ord=2)
                l1, l2 = tf.norm(S, ord=1), tf.norm(S, ord=2)
                perdida = residuo + lmb1 * l1 + lmb2 * l2 + lmbdiag * diag

            g = t.gradient(perdida, S)
            optimizer.apply_gradients([(g, S)])
            return perdida

        self.R = matriz_a_tensor(R)  # ambas dispersas, claro
        self.track_to_col = track_to_col
        self.col_to_track = {v: u for u, v in track_to_col.items()}
        self.S = inicializa(n=self.R.numcol, matriz="S", positiva=positiva)

        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

        print("Entramos no bucle de adestramento, isto vai levar...")
        progreso = tqdm(range(epochs))
        for i in progreso:
            perdida = train_step(self.R, self.S, optimizer)
            progreso.set_description(f"Época {i} || Pérdida {perdida.numpy().sum()}")

    def recommend_batch(self, seeds, top_n=500, batch_size=200) -> list[list[str]]:
        n_tracks = int(self.S.shape[0])

        results: list[list[str]] = []

        for start in range(0, len(seeds), batch_size):
            end = min(start + batch_size, len(seeds))
            seeds_batch = seeds[start:end]

            seed_cols_per_q = [
                [self.track_to_col[u] for u in s if u in self.track_to_col]
                for s in seeds_batch
            ]

            rows = []
            cols = []
            for q_idx, seed_cols in enumerate(seed_cols_per_q):
                for c in seed_cols:
                    rows.append(q_idx)
                    cols.append(c)

            if not rows:
                print("no hay rows")
                continue

            data = np.ones(len(rows), dtype=np.float32)
            b_csr = csr_matrix((data, (rows, cols)), shape=(len(seeds_batch), n_tracks))
            b_tf = matriz_a_tensor(b_csr)

            scores_tf = tf.sparse.sparse_dense_matmul(b_tf, self.S)
            scores = scores_tf.numpy()

            for q_idx, seed_cols in enumerate(seed_cols_per_q):
                row_scores = scores[q_idx]

                if seed_cols:
                    row_scores = row_scores.copy()
                    row_scores[seed_cols] = -np.inf

                if top_n >= row_scores.size:
                    top_cols = np.argsort(row_scores)[::-1]
                else:
                    take = np.argpartition(row_scores, -top_n)[-top_n:]
                    top_cols = take[np.argsort(row_scores[take])[::-1]]

                recs = [
                    self.col_to_track[int(c)]
                    for c in top_cols
                    if np.isfinite(row_scores[int(c)])
                ]

                results.append(recs[:top_n])

        return results

    def recommend(self, seed_uris, top_n=500) -> list[str]:
        return self.recommend_batch([seed_uris], top_n=top_n, batch_size=200)[0]

    @property
    def name(self) -> str:
        return "SSLIM"
