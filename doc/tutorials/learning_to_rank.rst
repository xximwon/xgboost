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

XGBoost supports learning to rank through a set of objective functions and evaluation metrics. Both binary relevance and multi-level relevance are supported.

*************
Deterministic
*************

GPU implementation of the LambdaMART loss function is not deterministic, it may return varying results between runs.

References:
