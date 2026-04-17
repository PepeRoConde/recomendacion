from __future__ import annotations

import os

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import svds

from .base import BaseRecommender

try:
    import hnswlib
except ImportError:  # pragma: no cover - optional dependency
    hnswlib = None

try:
    from pynndescent import NNDescent
except ImportError:  # pragma: no cover - optional dependency
    NNDescent = None


class PureSVDRecommender(BaseRecommender):
    def _recommend_popularity_fallback(self, seed_cols: list[int], top_n: int) -> list[str]:
        seed_set = set(seed_cols)
        recs = []
        for c in self.pop_order:
            if c in seed_set:
                continue
            recs.append(self.col_to_track[c])
            if len(recs) == top_n:
                break
        return recs

    def _fill_with_popularity(
        self,
        seed_cols: list[int],
        selected_cols: np.ndarray,
        selected_recs: list[str],
        top_n: int,
    ) -> list[str]:
        seed_set = set(seed_cols)
        selected_set = {int(c) for c in selected_cols}

        for c in self.pop_order:
            if c in seed_set or c in selected_set:
                continue
            selected_recs.append(self.col_to_track[c])
            if len(selected_recs) == top_n:
                break

        return selected_recs

    def _build_query_matrix(self, seed_cols_per_q: list[list[int]], n_tracks: int) -> csr_matrix:
        rows = []
        cols = []

        for q_idx, seed_cols in enumerate(seed_cols_per_q):
            for c in seed_cols:
                rows.append(q_idx)
                cols.append(c)

        if not rows:
            return csr_matrix((len(seed_cols_per_q), n_tracks), dtype=np.float32)

        data = np.ones(len(rows), dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=(len(seed_cols_per_q), n_tracks))

    def _compute_sorted_svd(self, matrix: csr_matrix, k: int):
        u, s, vt = svds(matrix, k=k, random_state=42)
        order = np.argsort(s)[::-1]
        return (
            u[:, order].astype(np.float32, copy=False),
            s[order].astype(np.float32, copy=False),
            vt[order, :].astype(np.float32, copy=False),
        )

    def _merge_top_candidates(
        self,
        current_cols: np.ndarray,
        current_scores: np.ndarray,
        new_cols: np.ndarray,
        new_scores: np.ndarray,
        top_n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if new_cols.size == 0:
            return current_cols, current_scores

        if current_cols.size == 0:
            if new_cols.size > top_n:
                take = np.argpartition(new_scores, -top_n)[-top_n:]
                new_cols = new_cols[take]
                new_scores = new_scores[take]
            order = np.argsort(new_scores)[::-1]
            return new_cols[order], new_scores[order]

        merged_cols = np.concatenate((current_cols, new_cols))
        merged_scores = np.concatenate((current_scores, new_scores))

        if merged_cols.size > top_n:
            take = np.argpartition(merged_scores, -top_n)[-top_n:]
            merged_cols = merged_cols[take]
            merged_scores = merged_scores[take]

        order = np.argsort(merged_scores)[::-1]
        return merged_cols[order], merged_scores[order]

    def _finalize_recommendations(
        self,
        seed_cols_per_q: list[list[int]],
        best_cols_per_q: list[np.ndarray],
        best_scores_per_q: list[np.ndarray],
        top_n: int,
    ) -> list[list[str]]:
        results: list[list[str]] = []

        for seed_cols, cand_cols, cand_scores in zip(
            seed_cols_per_q, best_cols_per_q, best_scores_per_q
        ):
            if cand_cols.size == 0:
                results.append(self._recommend_popularity_fallback(seed_cols, top_n))
                continue

            if cand_cols.size > top_n:
                take = np.argpartition(cand_scores, -top_n)[-top_n:]
                cand_cols = cand_cols[take]
                cand_scores = cand_scores[take]

            order = np.argsort(cand_scores)[::-1]
            cand_cols = cand_cols[order]

            recs = [self.col_to_track[int(c)] for c in cand_cols]
            if len(recs) < top_n:
                recs = self._fill_with_popularity(seed_cols, cand_cols, recs, top_n)

            results.append(recs)

        return results

    def _scores_to_topn(
        self,
        user_factors: np.ndarray,
        seed_cols_per_q: list[list[int]],
        top_n: int,
    ) -> list[list[str]]:
        q = user_factors.shape[0]
        n_tracks = self.vt.shape[1]
        chunk_size = self.score_chunk_size

        best_cols_per_q = [np.empty(0, dtype=np.int32) for _ in range(q)]
        best_scores_per_q = [np.empty(0, dtype=np.float32) for _ in range(q)]

        for start in range(0, n_tracks, chunk_size):
            end = min(start + chunk_size, n_tracks)
            vt_block = self.vt[:, start:end]
            block_scores = user_factors @ vt_block
            block_cols = np.arange(start, end, dtype=np.int32)

            for q_idx, seed_cols in enumerate(seed_cols_per_q):
                row_scores = np.asarray(block_scores[q_idx]).ravel()

                if seed_cols:
                    seed_set = set(seed_cols)
                    keep_mask = np.fromiter(
                        (c not in seed_set for c in block_cols),
                        dtype=bool,
                        count=block_cols.size,
                    )
                    if not keep_mask.any():
                        continue
                    row_scores = row_scores[keep_mask]
                    row_cols = block_cols[keep_mask]
                else:
                    row_cols = block_cols

                if row_cols.size == 0:
                    continue

                current_cols, current_scores = self._merge_top_candidates(
                    best_cols_per_q[q_idx],
                    best_scores_per_q[q_idx],
                    row_cols.astype(np.int32, copy=False),
                    row_scores.astype(np.float32, copy=False),
                    top_n,
                )
                best_cols_per_q[q_idx] = current_cols
                best_scores_per_q[q_idx] = current_scores

        return self._finalize_recommendations(
            seed_cols_per_q, best_cols_per_q, best_scores_per_q, top_n
        )

    def _fallback_batch(self, seed_cols_per_q: list[list[int]], top_n: int) -> list[list[str]]:
        return [
            self._recommend_popularity_fallback(seed_cols, top_n)
            for seed_cols in seed_cols_per_q
        ]

    def _recommend_batch_folding_in(
        self,
        seed_cols_per_q: list[list[int]],
        active_seed_cols: list[list[int]],
        top_n: int,
    ) -> list[list[str]]:
        if self.vt is None or self.sigma is None or self._effective_k == 0:
            return self._fallback_batch(seed_cols_per_q, top_n)

        b_active = self._build_query_matrix(active_seed_cols, self.R.shape[1])
        user_factors = (b_active @ self.vt.T).astype(np.float32, copy=False)
        user_factors /= self.sigma
        scores_factors = user_factors * self.sigma
        active_results = self._scores_to_topn(scores_factors, active_seed_cols, top_n)
        return self._interleave_active_and_fallback(seed_cols_per_q, active_results, top_n)

    def _build_ann_index(self):
        if hnswlib is None and NNDescent is None:
            self.ann_index = None
            self.item_vectors = None
            return

        self.item_vectors = np.ascontiguousarray(self.vt.T.astype(np.float32, copy=False))
        if hnswlib is not None:
            dim = self.item_vectors.shape[1]
            index = hnswlib.Index(space="ip", dim=dim)
            index.init_index(
                max_elements=self.item_vectors.shape[0],
                ef_construction=self.ann_ef_construction,
                M=self.ann_m,
            )
            index.add_items(
                self.item_vectors,
                np.arange(self.item_vectors.shape[0], dtype=np.int32),
                num_threads=self.ann_num_threads,
            )
            index.set_ef(self.ann_ef_search)
            self.ann_index = index
            self.ann_backend = "hnswlib"
            return

        n_neighbors = min(max(30, self.ann_n_neighbors), max(1, self.item_vectors.shape[0] - 1))
        self.ann_index = NNDescent(
            self.item_vectors,
            metric="dot",
            n_neighbors=n_neighbors,
            n_trees=self.ann_n_trees,
            n_jobs=self.ann_num_threads,
            random_state=42,
            low_memory=True,
            verbose=False,
        )
        self.ann_backend = "pynndescent"

    def _recommend_batch_ann(
        self,
        seed_cols_per_q: list[list[int]],
        active_seed_cols: list[list[int]],
        top_n: int,
    ) -> list[list[str]]:
        if self.ann_index is None or self.item_vectors is None:
            return self._recommend_batch_folding_in(seed_cols_per_q, active_seed_cols, top_n)

        b_active = self._build_query_matrix(active_seed_cols, self.R.shape[1])
        query_vectors = (b_active @ self.vt.T).astype(np.float32, copy=False)

        k_query = min(
            self.ann_candidates + max(top_n, 50),
            self.item_vectors.shape[0],
        )
        if self.ann_backend == "hnswlib":
            labels, _ = self.ann_index.knn_query(
                query_vectors,
                k=k_query,
                num_threads=self.ann_num_threads,
            )
        else:
            labels, _ = self.ann_index.query(query_vectors, k=k_query)

        active_results = []
        for q_idx, seed_cols in enumerate(active_seed_cols):
            cand_cols = labels[q_idx]
            cand_cols = cand_cols[cand_cols >= 0]

            if seed_cols:
                seed_set = set(seed_cols)
                keep_mask = np.fromiter(
                    (c not in seed_set for c in cand_cols),
                    dtype=bool,
                    count=cand_cols.size,
                )
                cand_cols = cand_cols[keep_mask]

            if cand_cols.size == 0:
                active_results.append(self._recommend_popularity_fallback(seed_cols, top_n))
                continue

            cand_vecs = self.item_vectors[cand_cols]
            cand_scores = cand_vecs @ query_vectors[q_idx]

            if cand_cols.size > top_n:
                take = np.argpartition(cand_scores, -top_n)[-top_n:]
                cand_cols = cand_cols[take]
                cand_scores = cand_scores[take]

            order = np.argsort(cand_scores)[::-1]
            cand_cols = cand_cols[order]

            recs = [self.col_to_track[int(c)] for c in cand_cols]
            if len(recs) < top_n:
                recs = self._fill_with_popularity(seed_cols, cand_cols, recs, top_n)
            active_results.append(recs)

        return self._interleave_active_and_fallback(seed_cols_per_q, active_results, top_n)

    def _recommend_batch_full_svd(
        self,
        seed_cols_per_q: list[list[int]],
        active_seed_cols: list[list[int]],
        top_n: int,
    ) -> list[list[str]]:
        b_active = self._build_query_matrix(active_seed_cols, self.R.shape[1])
        stacked = vstack([self.R, b_active], format="csr")
        min_dim = min(stacked.shape)
        if min_dim <= 1:
            return self._fallback_batch(seed_cols_per_q, top_n)

        k_eff = min(self.k, min_dim - 1)
        if k_eff < 1:
            return self._fallback_batch(seed_cols_per_q, top_n)

        u, sigma, vt = self._compute_sorted_svd(stacked, k_eff)
        user_factors = u[-len(active_seed_cols) :].astype(np.float32, copy=False)
        user_factors *= sigma
        self.vt = vt

        active_results = self._scores_to_topn(user_factors, active_seed_cols, top_n)
        return self._interleave_active_and_fallback(seed_cols_per_q, active_results, top_n)

    def _interleave_active_and_fallback(
        self,
        seed_cols_per_q: list[list[int]],
        active_results: list[list[str]],
        top_n: int,
    ) -> list[list[str]]:
        batch_results = []
        active_iter = iter(active_results)
        for seed_cols in seed_cols_per_q:
            if seed_cols:
                batch_results.append(next(active_iter))
            else:
                batch_results.append(self._recommend_popularity_fallback(seed_cols, top_n))
        return batch_results

    def fit(
        self,
        r: csr_matrix,
        track_to_col: dict,
        k: int = 100,
        folding_in: bool = False,
        use_ann: bool = False,
        ann_candidates: int = 20000,
        ann_m: int = 16,
        ann_ef_construction: int = 200,
        ann_ef_search: int = 200,
        ann_num_threads: int | None = None,
        query_batch_size: int = 256,
        score_chunk_size: int = 4096,
        **kwargs,
    ):
        self.k = int(k)
        self.folding_in = bool(folding_in)
        self.use_ann = bool(use_ann)
        self.ann_candidates = max(1, int(ann_candidates))
        self.ann_m = int(ann_m)
        self.ann_ef_construction = int(ann_ef_construction)
        self.ann_ef_search = int(ann_ef_search)
        self.ann_n_neighbors = 50
        self.ann_n_trees = 10
        self.ann_num_threads = int(ann_num_threads or (os.cpu_count() or 1))
        self.query_batch_size = max(1, int(query_batch_size))
        self.score_chunk_size = max(1, int(score_chunk_size))
        self.track_to_col = track_to_col
        self.col_to_track = {v: u for u, v in track_to_col.items()}
        self.R = r.tocsr().astype(np.float32, copy=False)

        pop_counts = np.asarray(self.R.astype(bool).sum(axis=0), dtype=np.int64).ravel()
        self.pop_order = np.argsort(pop_counts)[::-1]

        self.sigma = None
        self.vt = None
        self._effective_k = None
        self.ann_index = None
        self.item_vectors = None
        self.ann_backend = None

        if self.folding_in:
            min_dim = min(self.R.shape)
            if min_dim <= 1:
                self._effective_k = 0
                return

            self._effective_k = min(self.k, min_dim - 1)
            if self._effective_k < 1:
                self._effective_k = 0
                return

            _, self.sigma, self.vt = self._compute_sorted_svd(self.R, self._effective_k)
            if self.use_ann:
                if hnswlib is None and NNDescent is None:
                    print("[PureSVD] ANN requested but no ANN backend is available; falling back to exact scoring.")
                    self.use_ann = False
                else:
                    self._build_ann_index()

        print(
            f"[PureSVD] fit - R={self.R.shape}  k={self.k}  folding_in={self.folding_in}  use_ann={self.use_ann}  ann_backend={self.ann_backend}"
        )

    def recommend_batch(self, seeds: list, top_n: int = 500) -> list[list[str]]:
        if not seeds:
            return []

        results: list[list[str]] = []

        for start in range(0, len(seeds), self.query_batch_size):
            end = min(start + self.query_batch_size, len(seeds))
            batch_seeds = seeds[start:end]

            seed_cols_per_q = [
                sorted({self.track_to_col[u] for u in seed if u in self.track_to_col})
                for seed in batch_seeds
            ]

            active_positions = [
                idx for idx, seed_cols in enumerate(seed_cols_per_q) if seed_cols
            ]
            if not active_positions:
                results.extend(self._fallback_batch(seed_cols_per_q, top_n))
                continue

            active_seed_cols = [seed_cols_per_q[idx] for idx in active_positions]

            if self.folding_in and self.use_ann:
                batch_results = self._recommend_batch_ann(
                    seed_cols_per_q, active_seed_cols, top_n
                )
            elif self.folding_in:
                batch_results = self._recommend_batch_folding_in(
                    seed_cols_per_q, active_seed_cols, top_n
                )
            else:
                batch_results = self._recommend_batch_full_svd(
                    seed_cols_per_q, active_seed_cols, top_n
                )

            results.extend(batch_results)

        return results

    @property
    def name(self) -> str:
        tag = "-folding" if self.folding_in else ""
        if self.use_ann:
            tag += "-ann"
        return f"pure-svd-k{self.k}{tag}"