"""
Getting started with learning to rank
=====================================

  .. versionadded:: 2.0.0

This is a demonstration of using XGBoost for learning to rank tasks using the
MSLR_10k_letor dataset. For more infomation about the dataset, please visit its
`description page <https://www.microsoft.com/en-us/research/project/mslr/>`_.

This is a two-part demo, the first one contains a basic example of using XGBoost to
train on relevance degree, and the second part simulates click data and enable the
position debiasing training.

For an overview of learning to rank in XGBoost, please see
:doc:`Learning to Rank </tutorials/learning_to_rank>`.
"""
from __future__ import annotations

import argparse
import os
import pickle as pkl
from collections import namedtuple
from time import time
from typing import List, NamedTuple, Tuple, TypedDict

import numpy as np
from numpy import typing as npt
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file

import xgboost as xgb


class PBM:
    """Simulate click data with position bias model. There are other models available in
    `ULTRA <https://github.com/ULTR-Community/ULTRA.git>`_ like the cascading model.

    References
    ----------
    Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm

    """

    def __init__(self, eta: float) -> None:
        # click probability for each relevance degree. (from 0 to 4)
        self.click_prob = np.array([0.1, 0.16, 0.28, 0.52, 1.0])
        exam_prob = np.array(
            [0.68, 0.61, 0.48, 0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06]
        )
        self.exam_prob = np.power(exam_prob, eta)

    def sample_clicks_for_query(
        self, labels: npt.NDArray[np.int32], position: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int32]:
        """Sample clicks for one query based on input relevance degree and position.

        Parameters
        ----------

        labels :
            relevance_degree

        """
        labels = np.array(labels, copy=True)

        click_prob = np.zeros(labels.shape)
        # minimum
        labels[labels < 0] = 0
        # maximum
        labels[labels >= len(self.click_prob)] = -1
        click_prob = self.click_prob[labels]

        exam_prob = np.zeros(labels.shape)
        assert position.size == labels.size
        ranks = np.array(position, copy=True)
        # maximum
        ranks[ranks >= self.exam_prob.size] = -1
        exam_prob = self.exam_prob[ranks]

        rng = np.random.default_rng(1994)
        prob = rng.random(size=labels.shape[0], dtype=np.float32)

        clicks: npt.NDArray[np.int32] = np.zeros(labels.shape, dtype=np.int32)
        clicks[prob < exam_prob * click_prob] = 1
        return clicks


# relevance degree data
RelData = Tuple[csr_matrix, npt.NDArray[np.int32], npt.NDArray[np.int32]]


class RelDataCV(NamedTuple):
    train: RelData
    valid: RelData
    test: RelData


def load_mlsr_10k(data_path: str, cache_path: str) -> RelDataCV:
    """Load the MSLR10k dataset from data_path and cache a pickle object in cache_path.

    Returns
    -------

    A list of tuples [(X, y, qid), ...].

    """
    root_path = os.path.expanduser(args.data)
    cacheroot_path = os.path.expanduser(args.cache)
    cache_path = os.path.join(cacheroot_path, "MSLR_10K_LETOR.pkl")

    # Use only the Fold1 for demo:
    # Train,      Valid, Test
    # {S1,S2,S3}, S4,    S5
    fold = 1

    if not os.path.exists(cache_path):
        fold_path = os.path.join(root_path, f"Fold{fold}")
        train_path = os.path.join(fold_path, "train.txt")
        valid_path = os.path.join(fold_path, "vali.txt")
        test_path = os.path.join(fold_path, "test.txt")
        X_train, y_train, qid_train = load_svmlight_file(
            train_path, query_id=True, dtype=np.float32
        )
        y_train = y_train.astype(np.int32)
        qid_train = qid_train.astype(np.int32)

        X_valid, y_valid, qid_valid = load_svmlight_file(
            valid_path, query_id=True, dtype=np.float32
        )
        y_valid = y_valid.astype(np.int32)
        qid_valid = qid_valid.astype(np.int32)

        X_test, y_test, qid_test = load_svmlight_file(
            test_path, query_id=True, dtype=np.float32
        )
        y_test = y_test.astype(np.int32)
        qid_test = qid_test.astype(np.int32)

        data = RelDataCV(
            train=(X_train, y_train, qid_train),
            valid=(X_valid, y_valid, qid_valid),
            test=(X_test, y_test, qid_test),
        )

        with open(cache_path, "wb") as fd:
            pkl.dump(data, fd)

    with open(cache_path, "rb") as fd:
        data = pkl.load(fd)

    return data


def ranking_demo(args: argparse.Namespace) -> None:
    """Demonstration for learning to rank with relevance degree."""
    data = load_mlsr_10k(args.data, args.cache)

    X_train, y_train, qid_train = data.train
    sorted_idx = np.argsort(qid_train)
    X_train = X_train[sorted_idx]
    y_train = y_train[sorted_idx]
    qid_train = qid_train[sorted_idx]

    X_valid, y_valid, qid_valid = data.valid
    sorted_idx = np.argsort(qid_valid)
    X_valid = X_valid[sorted_idx]
    y_valid = y_valid[sorted_idx]
    qid_valid = qid_valid[sorted_idx]

    ranker = xgb.XGBRanker(
        tree_method="gpu_hist",
        lambdarank_pair_method="topk",
        lambdarank_num_pair=13,
        eval_metric=["ndcg@1", "ndcg@8"],
    )
    ranker.fit(
        X_train,
        y_train,
        qid=qid_train,
        eval_set=[(X_valid, y_valid)],
        eval_qid=[qid_valid],
        verbose=True,
    )


def rlencode(x: npt.NDArray[np.int32]) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Run length encoding using numpy, modified from:
    https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66

    """
    x = np.asarray(x)
    n = x.size
    starts = np.r_[0, np.flatnonzero(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    indptr = np.append(starts, np.array([x.size]))

    return indptr, lengths, values


ClickFold = namedtuple("ClickFold", ("X", "y", "q", "s", "c", "p"))


def simulate_clicks(args: argparse.Namespace) -> ClickFold:
    """Simulate click data using position biased model (PBM)."""

    def init_rank_score(
        X: csr_matrix,
        y: npt.NDArray[np.int32],
        qid: npt.NDArray[np.int32],
        sample_rate: float = 0.01,
    ) -> npt.NDArray[np.float32]:
        """We use XGBoost to generate the initial score instead of SVMRank for
        simplicity.

        """
        # random sample
        _rng = np.random.default_rng(1994)
        n_samples = int(X.shape[0] * sample_rate)
        index = np.arange(0, X.shape[0], dtype=np.uint64)
        _rng.shuffle(index)
        index = index[:n_samples]

        X_train = X[index]
        y_train = y[index]
        qid_train = qid[index]

        # Sort training data based on query id, required by XGBoost.
        sorted_idx = np.argsort(qid_train)
        X_train = X_train[sorted_idx]
        y_train = y_train[sorted_idx]
        qid_train = qid_train[sorted_idx]

        ltr = xgb.XGBRanker(objective="rank:ndcg", tree_method="gpu_hist")
        ltr.fit(X_train, y_train, qid=qid_train)

        # Use the original order of the data.
        scores = ltr.predict(X)
        return scores

    def simulate_one_fold(fold, scores_fold: npt.NDArray[np.float32]) -> ClickFold:
        """Simulate clicks for one fold."""
        X_fold, y_fold, qid_fold = fold
        assert qid_fold.dtype == np.int32
        indptr, lengths, values = rlencode(qid_fold)

        qids = np.unique(qid_fold)

        position = np.empty((y_fold.size,), dtype=np.int64)
        clicks = np.empty((y_fold.size,), dtype=np.int32)
        pbm = PBM(eta=1.0)

        # Avoid grouping by qid as we want to preserve the original data partition by
        # the dataset authors.
        for q in qids:
            qid_mask = q == qid_fold
            query_scores = scores_fold[qid_mask]
            # Initial rank list, scores sorted to decreasing order
            query_position = np.argsort(query_scores)[::-1]
            position[qid_mask] = query_position
            # get labels
            relevance_degrees = y_fold[qid_mask]
            query_clicks = pbm.sample_clicks_for_query(
                relevance_degrees, query_position
            )
            clicks[qid_mask] = query_clicks

        assert X_fold.shape[0] == qid_fold.shape[0], (X_fold.shape, qid_fold.shape)
        assert X_fold.shape[0] == clicks.shape[0], (X_fold.shape, clicks.shape)

        return ClickFold(X_fold, y_fold, qid_fold, scores_fold, clicks, position)

    cache_path = os.path.join(
        os.path.expanduser(args.cache), "MSLR_10K_LETOR_Clicks.pkl"
    )
    if os.path.exists(cache_path):
        print("Found existing cache for clicks.")
        with open(cache_path, "rb") as fdr:
            new_folds = pkl.load(fdr)
            return new_folds

    cv_data = load_mlsr_10k(args.data, args.cache)
    X, y, qid = list(zip(cv_data.train, cv_data.valid, cv_data.test))

    indptr = np.array([0] + [v.shape[0] for v in X])
    indptr = np.cumsum(indptr)

    assert len(indptr) == 3 + 1  # train, valid, test
    X_full = sparse.vstack(X)
    y_full = np.concatenate(y)
    qid_full = np.concatenate(qid)

    # Skip the data cleaning here for demonstration purposes.

    # Obtain initial relevance score for click simulation
    scores_full = init_rank_score(X_full, y_full, qid_full)
    # partition it back to train,valid,test tuple
    scores = [scores_full[indptr[i - 1] : indptr[i]] for i in range(1, indptr.size)]

    (
        X_full,
        y_full,
        qid_full,
        scores_ret,
        clicks_full,
        position_full,
    ) = simulate_one_fold((X_full, y_full, qid_full), scores_full)

    scores_check_1 = [
        scores_ret[indptr[i - 1] : indptr[i]] for i in range(1, indptr.size)
    ]
    for i in range(3):
        assert (scores_check_1[i] == scores[i]).all()

    position = [position_full[indptr[i - 1] : indptr[i]] for i in range(1, indptr.size)]
    clicks = [clicks_full[indptr[i - 1] : indptr[i]] for i in range(1, indptr.size)]

    with open(cache_path, "wb") as fdw:
        data = ClickFold(X, y, qid, scores, clicks, position)
        pkl.dump(data, fdw)

    return data


def sort_samples(
    X: csr_matrix,
    y: npt.NDArray[np.int32],
    qid: npt.NDArray[np.int32],
    clicks: npt.NDArray[np.int32],
    pos: npt.NDArray[np.int64],
    cache_path: str,
) -> Tuple[
    csr_matrix, npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]
]:
    """Sort data based on query index and position."""
    if os.path.exists(cache_path):
        print(f"Found existing cache: {cache_path}")
        with open(cache_path, "rb") as fdr:
            data = pkl.load(fdr)
            return data

    s = time()
    sorted_idx = np.argsort(qid)
    X = X[sorted_idx]
    clicks = clicks[sorted_idx]
    qid = qid[sorted_idx]
    pos = pos[sorted_idx]

    indptr, lengths, values = rlencode(qid)

    for i in range(1, indptr.size):
        beg = indptr[i - 1]
        end = indptr[i]

        assert beg < end, (beg, end)
        assert np.unique(qid[beg:end]).size == 1, (beg, end)

        query_pos = pos[beg:end]
        assert query_pos.min() == 0, query_pos.min()
        assert query_pos.max() >= query_pos.size - 1, (
            query_pos.max(),
            query_pos.size,
            i,
            np.unique(qid[beg:end]),
        )
        sorted_idx = np.argsort(query_pos)

        X[beg:end] = X[beg:end][sorted_idx]
        clicks[beg:end] = clicks[beg:end][sorted_idx]
        y[beg:end] = y[beg:end][sorted_idx]
        # not necessary
        qid[beg:end] = qid[beg:end][sorted_idx]

    e = time()
    print("Sort samples:", e - s)
    data = X, clicks, y, qid

    with open(cache_path, "wb") as fdw:
        pkl.dump(data, fdw)

    return data


def click_data_demo(args: argparse.Namespace) -> None:
    """Demonstration for learning to rank with click data."""
    folds = simulate_clicks(args)

    train = [pack[0] for pack in folds]
    valid = [pack[1] for pack in folds]
    test = [pack[2] for pack in folds]

    X_train, y_train, qid_train, scores_train, clicks_train, position_train = train
    assert X_train.shape[0] == clicks_train.size
    X_valid, y_valid, qid_valid, scores_valid, clicks_valid, position_valid = valid
    assert X_valid.shape[0] == clicks_valid.size
    assert scores_valid.dtype == np.float32
    assert clicks_valid.dtype == np.int32
    cache_path = os.path.expanduser(args.cache)

    X_train, clicks_train, y_train, qid_train = sort_samples(
        X_train,
        y_train,
        qid_train,
        clicks_train,
        position_train,
        os.path.join(cache_path, "sorted.train.pkl"),
    )
    X_valid, clicks_valid, y_valid, qid_valid = sort_samples(
        X_valid,
        y_valid,
        qid_valid,
        clicks_valid,
        position_valid,
        os.path.join(cache_path, "sorted.valid.pkl"),
    )

    ranker = xgb.XGBRanker(
        n_estimators=512,
        tree_method="hist",
        boost_from_average=0,
        grow_policy="lossguide",
        learning_rate=0.1,
        # LTR specific parameters
        objective="rank:ndcg",
        lambdarank_unbiased=True,
        lambdarank_bias_norm=0.5,
        lambdarank_num_pair=8,
        lambdarank_pair_method="topk",
        ndcg_exp_gain=True,
        eval_metric=["ndcg@1", "ndcg@8", "ndcg@32"],
    )
    ranker.fit(
        X_train,
        clicks_train,
        qid=qid_train,
        eval_set=[(X_train, clicks_train), (X_valid, y_valid)],
        eval_qid=[qid_train, qid_valid],
        verbose=True,
    )
    X_test = test[0]
    ranker.predict(X_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstration of learning to rank using XGBoost."
    )
    parser.add_argument(
        "--data", type=str, help="Root directory of the MSLR data.", required=True
    )
    parser.add_argument(
        "--cache",
        type=str,
        help="Directory for caching processed data.",
        required=True,
    )
    args = parser.parse_args()

    ranking_demo(args)
    click_data_demo(args)
