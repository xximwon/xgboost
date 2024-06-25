"""
Experimental support for external memory
========================================

This is similar to the one in `quantile_data_iterator.py`, but for external memory
instead of Quantile DMatrix.  The feature is not ready for production use yet.

    .. versionadded:: 1.5.0


See :doc:`the tutorial </tutorials/external_memory>` for more details.

"""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple

import numpy as np
import xgboost
from sklearn.datasets import make_regression
from xgboost.compat import concat

import rmm


rmm.reinitialize(pool_allocator=True)


def make_batches(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    tmpdir: str,
) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    n_workers = min(n_batches, 36)
    futures = []

    # for i in range(n_batches):
    #     X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
    #     y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
    #     files.append((X_path, y_path))
    # return files

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


class Iterator(xgboost.DataIter):
    """A custom iterator for loading files in batches."""

    def __init__(self, file_paths: List[Tuple[str, str]], on_host: bool) -> None:
        self._file_paths = file_paths
        self._it = 0
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix="cache", on_host=on_host)

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self._file_paths[self._it]
        # X = np.load(X_path)
        # y = np.load(y_path)
        X = np.lib.format.open_memmap(filename=X_path, mode="r")
        y = np.lib.format.open_memmap(filename=y_path, mode="r")
        assert X.shape[0] == y.shape[0]
        return X, y

    def next(self, input_data: Callable) -> int:
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        """
        print("Next", flush=True)
        if self._it == len(self._file_paths):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the similar signature to
        # the ``DMatrix`` constructor.
        X, y = self.load_file()
        input_data(data=X, label=y)
        self._it += 1
        return 1

    def reset(self) -> None:
        """Reset the iterator to its beginning"""
        self._it = 0


def main(tmpdir: str) -> xgboost.Booster:
    # generate some random data for demo
    files = make_batches(2 ** 22, 242, 16, tmpdir)
    it = Iterator(files, on_host=True)
    # For non-data arguments, specify it here once instead of passing them by the `next`
    # method.
    missing = np.nan
    Xy = xgboost.DMatrix(it, missing=missing, enable_categorical=False)

    # ``approx`` is also supported, but less efficient due to sketching. GPU behaves
    # differently than CPU tree methods as it uses a hybrid approach. See tutorial in
    # doc for details.
    booster = xgboost.train(
        {"tree_method": "hist", "max_depth": 3, "device": "cuda"},
        Xy,
        # evals=[(Xy, "Train")],
        num_boost_round=10,
    )
    return booster


if __name__ == "__main__":
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    with xgboost.config_context(verbosity=3, use_rmm=True):
        main(data_dir)
