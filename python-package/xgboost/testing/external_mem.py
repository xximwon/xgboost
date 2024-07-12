import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple

import numpy as np
from sklearn.datasets import make_regression

from ..core import Booster, DataIter, DMatrix, QuantileDMatrix
from ..training import train
from .data import make_dense_regression


class EmTestIterator(DataIter):
    """A custom iterator for profiling external memory."""

    def __init__(self, file_paths: List[Tuple[str, str]], on_host: bool) -> None:
        self._file_paths = file_paths
        self._it = 0
        super().__init__(cache_prefix="cache", on_host=on_host)

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self._file_paths[self._it]
        X = np.lib.format.open_memmap(filename=X_path, mode="r")
        y = np.lib.format.open_memmap(filename=y_path, mode="r")
        assert X.shape[0] == y.shape[0]
        return X, y

    def next(self, input_data: Callable) -> int:
        print("Next", flush=True)
        if self._it == len(self._file_paths):
            return 0

        X, y = self.load_file()
        input_data(data=X, label=y)
        self._it += 1
        return 1

    def reset(self) -> None:
        self._it = 0


def make_batches(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    reuse: bool,
    tmpdir: str,
) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []

    if reuse:
        for i in range(n_batches):
            X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
            y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                files = []
                break
            files.append((X_path, y_path))
        return files

    assert not files

    n_workers = min(n_batches, 36)
    futures = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for i in range(n_batches):
            fut = executor.submit(
                make_regression,
                n_samples=n_samples_per_batch,
                n_features=n_features,
                random_state=i,
            )
            futures.append(fut)

    for i, f in enumerate(futures):
        X, y = f.result()
        X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
        y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
        print(f"Save to X_path: {X_path}", flush=True)
        np.save(X_path, X)
        np.save(y_path, y)
        files.append((X_path, y_path))

    return files


def run_external_memory(
    tmpdir: str, reuse: bool, n_samples_per_batch: int = 2**22
) -> Booster:
    files = make_batches(n_samples_per_batch, 242, 8, reuse, tmpdir)
    it = EmTestIterator(files, on_host=True)
    Xy = DMatrix(it, missing=np.nan, enable_categorical=False)

    booster = train(
        {"tree_method": "hist", "max_depth": 6, "device": "cuda"},
        Xy,
        # evals=[(Xy, "Train")],
        num_boost_round=6,
    )
    return booster


def run_over_subscription(tmpdir: str, reuse: bool, n_samples: int = 2**22) -> Booster:
    import rmm

    rmm.reinitialize(
        pool_allocator=True,
        system_memory=True,
        system_memory_headroom_size=2 * 1024 * 1024 * 1024,
    )

    X_path = os.path.join(tmpdir, "over_subscription-X.npy")
    y_path = os.path.join(tmpdir, "over_subscription-y.npy")
    if reuse and os.path.exists(X_path) and os.path.exists(y_path):
        X = np.lib.format.open_memmap(filename=X_path, mode="r")
        y = np.lib.format.open_memmap(filename=y_path, mode="r")
    else:
        X, y = make_dense_regression(n_samples, n_features=242)

    Xy = QuantileDMatrix(X, y)
    booster = train(
        {"tree_method": "hist", "max_depth": 6, "device": "cuda"},
        Xy,
        # evals=[(Xy, "Train")],
        num_boost_round=6,
    )
    return booster
