"""Tests for dask shared by different test modules."""

from typing import List, Literal

import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from distributed import Client, LocalCluster

import xgboost as xgb
from xgboost.testing.updater import get_basescore


def check_init_estimation_clf(
    tree_method: str, device: Literal["cpu", "cuda"], client: Client
) -> None:
    """Test init estimation for classsifier."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=4096 * 2, n_features=32, random_state=1994)
    clf = xgb.XGBClassifier(
        n_estimators=1, max_depth=1, tree_method=tree_method, device=device
    )
    clf.fit(X, y)
    base_score = get_basescore(clf)

    dx = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dclf = xgb.dask.DaskXGBClassifier(
        n_estimators=1,
        max_depth=1,
        tree_method=tree_method,
        device=device,
    )
    dclf.client = client
    dclf.fit(dx, dy)
    dbase_score = get_basescore(dclf)
    np.testing.assert_allclose(base_score, dbase_score)


def check_init_estimation_reg(
    tree_method: str, device: Literal["cpu", "cuda"], client: Client
) -> None:
    """Test init estimation for regressor."""
    from sklearn.datasets import make_regression

    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(n_samples=4096 * 2, n_features=32, random_state=1994)
    reg = xgb.XGBRegressor(
        n_estimators=1, max_depth=1, tree_method=tree_method, device=device
    )
    reg.fit(X, y)
    base_score = get_basescore(reg)

    dx = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dreg = xgb.dask.DaskXGBRegressor(
        n_estimators=1, max_depth=1, tree_method=tree_method, device=device
    )
    dreg.client = client
    dreg.fit(dx, dy)
    dbase_score = get_basescore(dreg)
    np.testing.assert_allclose(base_score, dbase_score)


def check_init_estimation(
    tree_method: str, device: Literal["cpu", "cuda"], client: Client
) -> None:
    """Test init estimation."""
    check_init_estimation_reg(tree_method, device, client)
    check_init_estimation_clf(tree_method, device, client)


def check_uneven_nan(
    client: Client, tree_method: str, device: Literal["cpu", "cuda"], n_workers: int
) -> None:
    """Issue #9271, not every worker has missing value."""
    assert n_workers >= 2

    with client.as_current():
        clf = xgb.dask.DaskXGBClassifier(tree_method=tree_method, device=device)
        X = pd.DataFrame({"a": range(10000), "b": range(10000, 0, -1)})
        y = pd.Series([*[0] * 5000, *[1] * 5000])

        X.loc[:3000:1000, "a"] = np.nan

        client.wait_for_workers(n_workers=n_workers)

        clf.fit(
            dd.from_pandas(X, npartitions=n_workers),
            dd.from_pandas(y, npartitions=n_workers),
        )


def check_rabit_bootstrap(scheduler: str) -> None:
    from functools import partial, update_wrapper

    import numpy as np

    from xgboost import collective
    from xgboost.dask import CommunicatorContext, _get_dask_config, _get_rabit_args

    def local_test(worker_id: int, rabit_args: dict) -> int:
        with CommunicatorContext(**rabit_args):
            a = 1
            assert collective.is_distributed()
            arr = np.array([a])
            reduced = collective.allreduce(arr, collective.Op.SUM)
            assert reduced[0] == n_workers

            arr = np.array([worker_id])
            reduced = collective.allreduce(arr, collective.Op.MAX)
            assert reduced == n_workers - 1

            return 1

    def get_client_workers(client: Client) -> List[str]:
        "Get workers from a dask client."
        workers = client.scheduler_info()["workers"]
        return list(workers.keys())

    with Client(address=scheduler) as client:
        workers = get_client_workers(client)
        args = client.sync(_get_rabit_args, len(workers), _get_dask_config(), client)
        assert not collective.is_distributed()
        n_workers = len(workers)

        fn = update_wrapper(partial(local_test, rabit_args=args), local_test)

        futures = client.map(fn, range(len(workers)), workers=workers)
        results = client.gather(futures)
        assert sum(results) == n_workers


def check_rabit_bootstrap_local() -> None:
    with LocalCluster(n_workers=16) as cluster:
        check_rabit_bootstrap(cluster)
