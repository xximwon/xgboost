################
Learning to Rank
################

**Contents**

.. contents::
  :local:
  :backlinks: none

********
Overview
********
Often in the context of information retrieval, learning to rank aims to train a model that arranges a set of query results into an ordered list `[1] <#references>`__. For surprivised learning to rank, the predictors are sample documents encoded as feature matrix, and the labels are relevance degree for each sample. Relevance degree can be multi-level (graded) or binary (relevant or not). The training samples are often grouped by their query index with each query group containing multiple query results.

XGBoost implements learning to rank through a set of objective functions and performane metrics. The default objective is ``rank:ndcg`` based on the ``LambdaMART`` `[2] <#references>`__ algorithm, which in turn is an adaptation of the ``LambdaRank`` `[3] <#references>`__ framework to gradient boosting trees. For a history and a summary of the algorithm, see `[5] <#references>`__. The implementation in XGBoost features deterministic GPU computation, distributed training, position debiasing and two different pair construction strategies.

************************************
Training with the Pariwise Objective
************************************
``LambdaMART`` is a pairwise ranking model, meaning that it compares the relevance degree for every pair of samples in a query group and calculate a proxy gradient for each pair. The default objective ``rank:ndcg`` is using the surrogate gradient derived from the ``ndcg`` metric. To train a XGBoost model, we need an additional sorted array called ``qid`` for specifying the query group of input samples. An example input would look like this:

+-------+-----------+---------------+
|   QID |   Label   |   Features    |
+=======+===========+===============+
|   1   |   0       |   :math:`x_1` |
+-------+-----------+---------------+
|   1   |   1       |   :math:`x_2` |
+-------+-----------+---------------+
|   1   |   0       |   :math:`x_3` |
+-------+-----------+---------------+
|   2   |   0       |   :math:`x_4` |
+-------+-----------+---------------+
|   2   |   1       |   :math:`x_5` |
+-------+-----------+---------------+
|   2   |   1       |   :math:`x_6` |
+-------+-----------+---------------+
|   2   |   1       |   :math:`x_7` |
+-------+-----------+---------------+

Notice that the samples are sorted based on their query index in an non-decreasing order. Here the first three samples belong to the first query and the next four samples belong to the second. For the sake of simplicity, we will use a pseudo binary learning to rank dataset in the following snippets, with binary labels representing whether the result is relevant or not, and randomly assign the query group index to each sample. For an example that uses a real world dataset, please see :ref:`sphx_glr_python_examples_learning_to_rank.py`.

.. code-block:: python

  from sklearn.datasets import make_classification
  import numpy as np

  import xgboost as xgb

  # Make a pseudo ranking dataset for demonstration
  X, y = make_classification(random_state=rng)
  rng = np.random.default_rng(1994)
  n_query_groups = 3
  qid = rng.integers(0, 3, size=X.shape[0])

  # Sort the inputs based on query index
  sorted_idx = np.argsort(qid)
  X = X[sorted_idx, :]
  y = y[sorted_idx]

The simpliest way to train a ranking model is by using the sklearn estimator interface. Please note that, as of writing, there's no learning to rank interface in sklearn. As a result, the :py:class:`xgboost.XGBRanker` does not fully conform the sklearn estimator guideline and can not be used with some of its utility functions. For instance, the ``auc_score`` and ``ndcg_score`` in sklearn cannot be directly used for XGBoost without some modifications and care, and the ``KFold`` also ignores group information. Continuing the previous snippet, we can now train a simple ranking model:

.. code-block:: python

  ranker = xgb.XGBRanker(tree_method="hist", lambdarank_num_pair=8, objective="rank:ndcg", lambdarank_pair_method="topk")
  ranker.fit(X, y, qid=qid)

The above snippet builds a model using ``LambdaMART`` with the ``NDCG@8`` metric. There's also a ``group`` parameter in the :py:meth:`xgboost.XGBRanker.fit` method, which is an alternative to the ``qid``. We will stick with ``qid`` here as it's easier to demonstrate. The output of ranker are relevance scores:

.. code-block:: python

  scores = ranker.predict(X)
  sorted_idx = np.argsort(scores)[::-1]
  # Sort the relevance scores from most relevant to least relevant
  scores = scores[sorted_idx]


*************
Position Bias
*************
Real relevance degree for query result is difficult to obtain as it often requires human judegs to examine the content of query results. When such labeled data is absent, we might want to train the model on ground truth data like user clicks. Another upside of using click data directly is that it can relect the up-to-date relevance status `[1] <#references>`__. However, user clicks are often nosiy and biased as users tend to choose results displayed in higher position. To ameliorate this issue, XGBoost implements the ``Unbiased LambdaMART`` `[4] <#references>`__ algorithm to debias the position-dependent click data. The feature can be enabled by the ``lambdarank_unbiased`` parameter, see :ref:`ltr-param` for related options and :ref:`sphx_glr_python_examples_learning_to_rank.py` for a worked example with simulated user clicks.

****
Loss
****

XGBoost implements different ``LambdaMART`` objectives based on different metrics. We list them here as a reference. Other than those used as objective function, XGBoost also implements metrics like ``pre`` (for precision) for evaluation. See :doc:`parameters </parameter>` for available options and the following sections for how to choose these objectives based of the amount of effective pairs.

NDCG
----
`Normalized Discounted Cumulative Gain` ``NDCG`` can be used with both binary relevance and multi-level relevance. If you are not sure about your data, this metric can be used as the default. The name for the objective is ``rank:ndcg``.


MAP
---
`Mean average precision` ``MAP`` is a binary measure. It can be used when the relevance label is 0 or 1. The name for the objective is ``rank:map``.


Pairwise
--------
The `LambdaMART` algorithm scales the logistic loss with learning to rank metrics like ``NDCG`` in the hope of including ranking infomation into the loss function. The ``rank:pairwise`` loss is the orginal version of the pairwise loss, also known as the `RankNet loss` `[7] <#references>`__ or the `pairwise logistic loss`. Unlike the ``rank:map`` and the ``rank:ndcg``, no scaling is applied (:math:`|\Delta Z_{ij}| = 1`).

Whether scaling with a LTR metric is actually more effective is still up for debate, `[8] <#references>`__ provides a theoretical foundation for general lambda loss functions and some insights into the framework.

******************
Constructing Pairs
******************

There are two implemented strategies for constructing document pairs for :math:`\lambda`-gradient calculation. The first one is the ``mean`` method, another one is the ``topk`` method. The preferred strategy can be specified by the ``lambdarank_pair_method`` parameter.

For the ``mean`` strategy, XGBoost samples ``lambdarank_num_pair`` pairs for each document in a query list. For example, given a list of 3 documents and ``lambdarank_num_pair`` is set to 2, XGBoost will randomly sample 6 pairs assuming the labels for these documents are different. On the other hand, if the pair method is set to ``topk``, XGBoost constructs about :math:`k \times |query|` number of pairs with :math:`|query|` pairs for each sample at the top :math:`k = lambdarank\_num\_pair` position. The number of pairs counted here is an approximation since we skip pairs that have the sample label.

*********************
Obtaining Good Result
*********************

Learning to rank is a sophisticated task and a field of heated research. It's not trivial to train a model that generalizes well. There are multiple loss functions available in XGBoost along with a set of hyper-parameters. This section contains some hints for how to choose those parameters as a starting point. One can further optimize the model by tuning these parameters.

The first question would be how to choose an objective that matches the task at hand. If your input data is multi-level relevance degree, then either ``rank:ndcg`` or ``rank:pairwise`` should be used. However, when the input is binary we have multiple options based on the target metric. `[6] <#references>`__ provides some guidelines on this topic and users are encouraged to see the analysis done in their work. The choice should be based on the number of `effective pairs`, which refers to the number of pairs that can generate non-zero gradient and contribute to training. `LambdaMART` with ``MRR`` has the least amount of effective pairs as the :math:`\lambda`-gradient is only non-zero when the pair contains a non-relevant document ranked higher than the top relevant document. As a result, it's not implemented in XGBoost. Since ``NDCG`` is a multi-level metric, it can generate more effective pairs than ``MAP``.

However, when there's a sufficient amount of effective pairs, it's shown in `[6] <#references>`__ that matching the target metric with the objective is of significance. When your targeted metric is ``MAP`` and you are using a large dataset that can provide a sufficient amount of effective pairs, ``rank:map`` can in theory yield higher ``MAP`` value than the ``rank:ndcg``.

The choice of pair method (``lambdarank_pair_method``) and the number of pairs for each sample (``lambdarank_num_pair``) is similar, as the mean-``NDCG`` considers more pairs than ``NDCG@10``, it can generate more effective pairs and provide more granularity. Also, using the ``mean`` strategy can help the model generalize with random sampling. However, one might want to focus the training on the top :math:`k` documents instead of using all pairs in practice, the tradeoff should be made based on the user's goal.

When using mean value instead of targeting a specific position by calculating the target metric (like ``NDCG``) over the whole query list, user can specify how many pairs they want in each query by setting the ``lambdarank_num_pair`` and XGBoost will randomly sample this amount of pairs for each element in the query group (:math:`|pairs| = |query| \times num\_pairsample`). Often time, setting it to 1 can produce reasonable result, with higher value producing more pairs (with the hope that a reasonable amount of them being effective). On the other hand, if you are prioritizing top :math:`k` documents, the ``lambdarank_num_pair`` should be set to slightly higher than :math:`k` (with a few more documents) to obtain better training result.

In summary, to start off the training, if you have a large dataset, consider using the target-matching objective, otherwise ``NDCG`` might be preferred. With the same target metric, use the ``lambdarank_num_pair`` to specify the top :math:`k` documents for training if your dataset is large, and use the mean value version otherwise. When mean value is used, ``lambdarank_num_pair`` can be used to control the amount of pairs.

********************
Distributed Training
********************
XGBoost implements distributed learning-to-rank with integration of multiple frameworks including dask, spark, and pyspark. The interface is similar to single node. Please refer to document of the respective XGBoost interface for details. Scattering a query group onto multiple workers is theoretically sound but can affect the model accuracy. For most of the use cases, the small discrepancy is not an issue since when distributed training is involved the dataset should be sufficiently large. As a result, users don't need to partition the data based on group information given the dataset is correctly sorted, XGBoost can aggregate sample gradients accordingly.

*******************
Reproducbile Result
*******************

Like any other tasks, XGBoost should generate reproducbile results given the same hardware and software environments, along with data partitions if distributed interface is used. However, since the ``lambdarank_pair_method`` uses sampling, and the random generator used on Windows (MSVC) is different from the one used on other platforms like Linux (GCC, Clang), the output varies between these platforms.

**********
References
**********

[1] Tie-Yan Liu. 2009. "`Learning to Rank for Information Retrieval`_". Found. Trends Inf. Retr. 3, 3 (March 2009), 225–331.

[2] Christopher J. C. Burges, Robert Ragno, and Quoc Viet Le. 2006. "`Learning to rank with nonsmooth cost functions`_". In Proceedings of the 19th International Conference on Neural Information Processing Systems (NIPS'06). MIT Press, Cambridge, MA, USA, 193–200.

[3] Wu, Q., Burges, C.J.C., Svore, K.M. et al. "`Adapting boosting for information retrieval measures`_". Inf Retrieval 13, 254–270 (2010).

[4] Ziniu Hu, Yang Wang, Qu Peng, Hang Li. "`Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm`_". Proceedings of the 2019 World Wide Web Conference.

[5] Burges, Chris J.C. "`From RankNet to LambdaRank to LambdaMART: An Overview`_". MSR-TR-2010-82

[6] Pinar Donmez, Krysta M. Svore, and Christopher J.C. Burges. 2009. "`On the local optimality of LambdaRank`_". In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (SIGIR '09). Association for Computing Machinery, New York, NY, USA, 460–467.

[7] Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005. "`Learning to rank using gradient descent`_". In Proceedings of the 22nd international conference on Machine learning (ICML '05). Association for Computing Machinery, New York, NY, USA, 89–96.

[8] Xuanhui Wang and Cheng Li and Nadav Golbandi and Mike Bendersky and Marc Najork. 2018. "`The LambdaLoss Framework for Ranking Metric Optimization`_". Proceedings of The 27th ACM International Conference on Information and Knowledge Management (CIKM '18).

.. _`Learning to Rank for Information Retrieval`: https://doi.org/10.1561/1500000016
.. _`Learning to rank with nonsmooth cost functions`: https://dl.acm.org/doi/10.5555/2976456.2976481
.. _`Adapting boosting for information retrieval measures`: https://doi.org/10.1007/s10791-009-9112-1
.. _`Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm`: https://dl.acm.org/doi/10.1145/3308558.3313447
.. _`From RankNet to LambdaRank to LambdaMART: An Overview`: https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
.. _`On the local optimality of LambdaRank`: https://doi.org/10.1145/1571941.1572021
.. _`Learning to rank using gradient descent`:  https://doi.org/10.1145/1102351.1102363
