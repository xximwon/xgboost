/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#pragma once

#include <algorithm>
#include <cstdint>  // std::int32_t
#include <limits>
#include <vector>   // std::vector

#include "../collective/communicator-inl.h"
#include "../common/common.h"
#include "xgboost/base.h"                // bst_node_t
#include "xgboost/context.h"             // Context
#include "xgboost/data.h"                // MetaInfo
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/tree_model.h"          // RegTree

namespace xgboost::obj {
namespace detail {
inline void FillMissingLeaf(std::vector<bst_node_t> const& maybe_missing,
                            std::vector<bst_node_t>* p_nidx, std::vector<size_t>* p_nptr) {
  auto& h_node_idx = *p_nidx;
  auto& h_node_ptr = *p_nptr;

  for (auto leaf : maybe_missing) {
    if (std::binary_search(h_node_idx.cbegin(), h_node_idx.cend(), leaf)) {
      continue;
    }
    auto it = std::upper_bound(h_node_idx.cbegin(), h_node_idx.cend(), leaf);
    auto pos = it - h_node_idx.cbegin();
    h_node_idx.insert(h_node_idx.cbegin() + pos, leaf);
    h_node_ptr.insert(h_node_ptr.cbegin() + pos, h_node_ptr[pos]);
  }
}

void UpdateLeafValues(linalg::MatrixView<float> quantiles, std::vector<bst_node_t> const& nidx,
                      RegTree* p_tree);

inline std::size_t IdxY(MetaInfo const& info, bst_group_t group_idx) {
  std::size_t y_idx{0};
  if (info.labels.Shape(1) > 1) {
    y_idx = group_idx;
  }
  CHECK_LE(y_idx, info.labels.Shape(1));
  return y_idx;
}

void UpdateTreeLeafDevice(Context const* ctx, common::Span<bst_node_t const> position,
                          std::int32_t group_idx, MetaInfo const& info, float learning_rate,
                          HostDeviceVector<float> const& predt, float alpha, RegTree* p_tree);

void UpdateTreeLeafHost(Context const* ctx, std::vector<bst_node_t> const& position,
                        std::int32_t group_idx, MetaInfo const& info, float learning_rate,
                        HostDeviceVector<float> const& predt, common::Span<float const> alphas,
                        RegTree* p_tree);
}  // namespace detail

inline void UpdateTreeLeaf(Context const* ctx, HostDeviceVector<bst_node_t> const& position,
                           std::int32_t group_idx, MetaInfo const& info, float learning_rate,
                           HostDeviceVector<float> const& predt, common::Span<float const> alphas,
                           RegTree* p_tree) {
  if (ctx->IsCPU()) {
    detail::UpdateTreeLeafHost(ctx, position.ConstHostVector(), group_idx, info, learning_rate,
                               predt, alphas, p_tree);
  } else {
    position.SetDevice(ctx->gpu_id);
    // detail::UpdateTreeLeafDevice(ctx, position.ConstDeviceSpan(), group_idx, info, learning_rate,
    //                              predt, alpha, p_tree);
  }
}
}  // namespace xgboost::obj
