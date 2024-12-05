/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICT_FN_H_
#define XGBOOST_PREDICTOR_PREDICT_FN_H_
#include "../common/categorical.h"
#include "../common/math.h"      // for CheckNAN
#include "../data/adapter.h"     // for COOTuple
#include "../encoder/ordinal.h"  // for MappingView
#include "xgboost/tree_model.h"

namespace xgboost::predictor {
/** @brief Whether it should traverse to the left branch of a tree. */
template <bool has_categorical>
XGBOOST_DEVICE bool GetDecision(RegTree::Node const &node, bst_node_t nid, float fvalue,
                                RegTree::CategoricalSplitMatrix const &cats) {
  if (has_categorical && common::IsCat(cats.split_type, nid)) {
    auto node_categories = cats.categories.subspan(cats.node_ptr[nid].beg, cats.node_ptr[nid].size);
    return common::Decision(node_categories, fvalue);
  } else {
    return fvalue < node.SplitCond();
  }
}

template <bool has_missing, bool has_categorical>
inline XGBOOST_DEVICE bst_node_t GetNextNode(const RegTree::Node &node, const bst_node_t nid,
                                             float fvalue, bool is_missing,
                                             RegTree::CategoricalSplitMatrix const &cats) {
  if (has_missing && is_missing) {
    return node.DefaultChild();
  } else {
    return node.LeftChild() + !GetDecision<has_categorical>(node, nid, fvalue, cats);
  }
}

template <bool has_missing, bool has_categorical>
inline XGBOOST_DEVICE bst_node_t GetNextNodeMulti(MultiTargetTree const &tree,
                                                  bst_node_t const nidx, float fvalue,
                                                  bool is_missing,
                                                  RegTree::CategoricalSplitMatrix const &cats) {
  if (has_missing && is_missing) {
    return tree.DefaultChild(nidx);
  } else {
    if (has_categorical && common::IsCat(cats.split_type, nidx)) {
      auto node_categories =
          cats.categories.subspan(cats.node_ptr[nidx].beg, cats.node_ptr[nidx].size);
      return common::Decision(node_categories, fvalue) ? tree.LeftChild(nidx)
                                                       : tree.RightChild(nidx);
    } else {
      return tree.LeftChild(nidx) + !(fvalue < tree.SplitCond(nidx));
    }
  }
}

struct CatAccessor {
  enc::MappingView const &enc;
  [[nodiscard]] XGBOOST_DEVICE float operator()(data::COOTuple const &e) {
    auto fvalue = e.value;
    if (!common::CheckNAN(fvalue) && !enc.Empty() && !enc[e.column_idx].empty()) {
      auto f_mapping = enc[e.column_idx];
      if (fvalue >= 0 && fvalue < f_mapping.size()) {
        fvalue = f_mapping[fvalue];
      }
    }
    return fvalue;
  }
};

struct NoOpAccessor {
  XGBOOST_DEVICE explicit NoOpAccessor(enc::MappingView const &) {}
  NoOpAccessor() = default;
  [[nodiscard]] XGBOOST_DEVICE float operator()(data::COOTuple const &e) const { return e.value; }
};
}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_PREDICT_FN_H_
