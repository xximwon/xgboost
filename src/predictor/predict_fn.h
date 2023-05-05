/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICT_FN_H_
#define XGBOOST_PREDICTOR_PREDICT_FN_H_

#include "../common/categorical.h"            // for IsCat, Decision
#include "xgboost/base.h"                     // for bst_note_t
#include "xgboost/context.h"                  // for Context
#include "xgboost/host_device_vector.h"       // for HostDeviceVector
#include "xgboost/multi_target_tree_model.h"  // for MultiTargetTree
#include "xgboost/tree_model.h"               // for RegTree

namespace xgboost {
struct PredictionCacheEntry;
class DMatrix;

namespace gbm {
struct GBTreeModel;
}  // namespace gbm
}  // namespace xgboost

namespace xgboost::predictor {
template <bool has_missing, bool has_categorical>
inline XGBOOST_DEVICE bst_node_t GetNextNode(const RegTree::Node& node, const bst_node_t nid,
                                             float fvalue, bool is_missing,
                                             RegTree::CategoricalSplitMatrix const& cats) {
  if (has_missing && is_missing) {
    return node.DefaultChild();
  } else {
    if (has_categorical && common::IsCat(cats.split_type, nid)) {
      auto node_categories =
          cats.categories.subspan(cats.node_ptr[nid].beg, cats.node_ptr[nid].size);
      return common::Decision(node_categories, fvalue) ? node.LeftChild() : node.RightChild();
    } else {
      return node.LeftChild() + !(fvalue < node.SplitCond());
    }
  }
}

template <bool has_missing, bool has_categorical>
inline XGBOOST_DEVICE bst_node_t GetNextNodeMulti(MultiTargetTree const& tree,
                                                  bst_node_t const nidx, float fvalue,
                                                  bool is_missing,
                                                  RegTree::CategoricalSplitMatrix const& cats) {
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

namespace cuda_impl {
bool InplacePredict(Context const* ctx, std::shared_ptr<DMatrix> p_m, const gbm::GBTreeModel& model,
                    float missing, PredictionCacheEntry* out_preds, bst_tree_t tree_begin,
                    bst_tree_t tree_end);

void PredictInteractionContributions(Context const* ctx, DMatrix* p_fmat,
                                     HostDeviceVector<float>* out_contribs,
                                     const gbm::GBTreeModel& model, unsigned tree_end,
                                     std::vector<float> const* tree_weights, bool approximate);

void PredictLeaf(Context const* ctx, DMatrix* p_fmat, HostDeviceVector<float>* predictions,
                 const gbm::GBTreeModel& model, bst_tree_t tree_end);
}  // namespace cuda_impl
}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_PREDICT_FN_H_
