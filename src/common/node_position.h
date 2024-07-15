/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "xgboost/base.h"                // for bst_node_t
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::common {
class NodePosition {
  // The node position for each row, 1 HDV for each tree in the forest. Note that the
  // position is negated if the row is sampled out.
  std::vector<HostDeviceVector<bst_node_t>> position_;

 public:
  void Resize(std::size_t n_new_trees) { this->position_.resize(n_new_trees); }

  HostDeviceVector<bst_node_t> const& operator[](std::size_t tree_idx) const {
    return position_[tree_idx];
  }
  HostDeviceVector<bst_node_t>& operator[](std::size_t tree_idx) { return position_[tree_idx]; }

  // We negate the node index if the sample is not used (not sampled).
  XGBOOST_DEVICE static bst_node_t EncodeMissing(bst_node_t nidx) { return ~nidx; }
  XGBOOST_DEVICE static bool NonMissing(bst_node_t nidx) { return nidx >= 0; }
};
}  // namespace xgboost::common
