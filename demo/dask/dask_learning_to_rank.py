"""
Learning to rank with the Dask Interface
========================================

"""

from __future__ import annotations

import argparse
import os

import numpy as np
from distributed import Client, LocalCluster, wait
from sklearn.datasets import load_svmlight_file
from xgboost import dask as dxgb

from dask import array as da
from dask import dataframe as dd


def load_mlsr_10k(
    device: str, data_path: str, cache_path: str
) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
    """Load the MSLR10k dataset from data_path and save parquet files in the cache_path."""
    root_path = os.path.expanduser(args.data)
    cache_path = os.path.expanduser(args.cache)

    # Use only the Fold1 for demo:
    # Train,      Valid, Test
    # {S1,S2,S3}, S4,    S5
    fold = 1
    nparts = 32

    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        fold_path = os.path.join(root_path, f"Fold{fold}")
        train_path = os.path.join(fold_path, "train.txt")
        valid_path = os.path.join(fold_path, "vali.txt")
        test_path = os.path.join(fold_path, "test.txt")

        X_train, y_train, qid_train = load_svmlight_file(
            train_path, query_id=True, dtype=np.float32
        )
        columns = [f"f{i}" for i in range(X_train.shape[1])]
        X_train = dd.from_array(X_train.toarray(), columns=columns)
        y_train = y_train.astype(np.int32)
        qid_train = qid_train.astype(np.int32)

        X_train["y"] = dd.from_dict({"y": y_train}, npartitions=nparts).y
        X_train["qid"] = dd.from_dict({"qid": qid_train}, npartitions=nparts).qid
        X_train.to_parquet(os.path.join(cache_path, "train"))

        X_valid, y_valid, qid_valid = load_svmlight_file(
            valid_path, query_id=True, dtype=np.float32
        )
        X_valid = dd.from_array(X_valid.toarray(), columns=columns)
        y_valid = y_valid.astype(np.int32)
        qid_valid = qid_valid.astype(np.int32)

        X_valid["y"] = dd.from_dict({"y": y_valid}, npartitions=nparts).y
        X_valid["qid"] = dd.from_dict({"qid": qid_valid}, npartitions=nparts).qid
        X_valid.to_parquet(os.path.join(cache_path, "valid"))

        X_test, y_test, qid_test = load_svmlight_file(
            test_path, query_id=True, dtype=np.float32
        )

        X_test = dd.from_array(X_test.toarray(), columns=columns)
        y_test = y_test.astype(np.int32)
        qid_test = qid_test.astype(np.int32)

        X_test["y"] = dd.from_dict({"y": y_test}, npartitions=nparts).y
        X_test["qid"] = dd.from_dict({"qid": qid_test}, npartitions=nparts).qid
        X_test.to_parquet(os.path.join(cache_path, "test"))

    df_train = dd.read_parquet(
        os.path.join(cache_path, "train"), calculate_divisions=True
    )
    df_valid = dd.read_parquet(
        os.path.join(cache_path, "valid"), calculate_divisions=True
    )
    df_test = dd.read_parquet(
        os.path.join(cache_path, "test"), calculate_divisions=True
    )

    if device == "cuda":
        df_train = df_train.to_backend("cudf")
        df_valid = df_valid.to_backend("cudf")
        df_test = df_test.to_backend("cudf")

    return df_train, df_valid, df_test


def ranking_demo(client: Client, args: argparse.Namespace) -> None:
    df_train, df_valid, df_test = load_mlsr_10k(args.device, args.data, args.cache)

    X_train: dd.DataFrame = df_train[df_train.columns.difference(["y", "qid"])]
    y_train = df_train[["y", "qid"]]
    Xy_train = dxgb.DaskQuantileDMatrix(client, X_train, y_train.y, qid=y_train.qid)

    X_valid: dd.DataFrame = df_valid[df_valid.columns.difference(["y", "qid"])]
    y_valid = df_valid[["y", "qid"]]
    Xy_valid = dxgb.DaskQuantileDMatrix(
        client, X_valid, y_valid.y, qid=y_valid.qid, ref=Xy_train
    )

    dxgb.train(
        client,
        {"objective": "rank:ndcg", "device": args.device},
        Xy_train,
        evals=[(Xy_train, "Train"), (Xy_valid, "Valid")],
    )


def no_group_split(client: Client, args: argparse.Namespace) -> None:
    df_train, df_valid, df_test = load_mlsr_10k(args.device, args.data, args.cache)
    print("divisions:", df_train.divisions)
    df_train = df_train.sort_values(by="qid")
    print("df_train.columns:", list(df_train.columns))
    cnt = df_train.groupby("qid").qid.count()
    div = da.cumsum(cnt.to_dask_array(lengths=True)).compute()
    print(div, type(div), div.shape)
    div = np.concatenate([np.zeros(shape=(1,), dtype=div.dtype), div])
    print(div, div.shape, "npartitions:", df_train.npartitions)
    df_train = df_train.set_index("qid", divisions=list(div)).persist()
    df_train = df_train.persist()
    wait([df_train])

    Xy_train = dxgb.DaskQuantileDMatrix(
        client,
        df_train[df_train.columns.difference(["y"])],
        df_train.y,
        qid=df_train.index,
    )
    dxgb.train(
        client,
        {"objective": "rank:ndcg", "device": args.device},
        Xy_train,
        evals=[(Xy_train, "Train")],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstration of learning to rank using XGBoost."
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Root directory of the MSLR-WEB10K data.",
        required=True,
    )
    parser.add_argument(
        "--cache",
        type=str,
        help="Directory for caching processed data.",
        required=True,
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    match args.device:
        case "cuda":
            from dask_cuda import LocalCUDACluster

            with LocalCUDACluster() as cluster:
                with Client(cluster) as client:
                    # ranking_demo(client, args)
                    no_group_split(client, args)
        case "cpu":
            with LocalCluster(n_workers=2) as cluster:
                with Client(cluster) as client:
                    # ranking_demo(client, args)
                    no_group_split(client, args)
