"""
Experimental support for external memory
========================================

This is similar to the one in `quantile_data_iterator.py`, but for external memory
instead of Quantile DMatrix.  The feature is not ready for production use yet.

    .. versionadded:: 1.5.0


See :doc:`the tutorial </tutorials/external_memory>` for more details.

"""

import multiprocessing
import os
import tempfile
from typing import Callable, List, Tuple

import numpy as np
import xgboost
from sklearn.datasets import make_regression
from xgboost import RabitTracker


def make_batches(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    tmpdir: str,
) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    rng = np.random.RandomState(1994)
    for i in range(n_batches):
        X, y = make_regression(n_samples_per_batch, n_features, random_state=rng)
        X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
        y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
        np.save(X_path, X)
        np.save(y_path, y)
        files.append((X_path, y_path))
    return files


class Iterator(xgboost.DataIter):
    """A custom iterator for loading files in batches."""

    def __init__(self, file_paths: List[Tuple[str, str]]):
        self._file_paths = file_paths
        self._it = 0
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join("/dev/shm/", "cache"))

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self._file_paths[self._it]
        X = np.load(X_path)
        y = np.load(y_path)
        assert X.shape[0] == y.shape[0]
        return X, y

    def next(self, input_data: Callable) -> int:
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        """
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


def run_rabit_worker(rabit_args) -> None:
    with xgboost.collective.CommunicatorContext(**rabit_args), xgboost.config_context(
        verbosity=2
    ), tempfile.TemporaryDirectory() as tmpdir:
        # generate some random data for demo
        files = make_batches(1024, 17, 2, tmpdir)
        it = Iterator(files)
        # For non-data arguments, specify it here once instead of passing them by the `next`
        # method.
        Xy = xgboost.DMatrix(it, missing=np.NaN, enable_categorical=False)
        booster = xgboost.train({"device": "cuda"}, Xy, evals=[(Xy, "Train")])


def main() -> None:
    world_size = 2
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=world_size)
    tracker.start(n_workers=2)

    workers = []
    for w_id in range(world_size):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(w_id)
        worker = multiprocessing.Process(
            target=run_rabit_worker, args=(tracker.worker_envs(),)
        )
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0


if __name__ == "__main__":
    main()
