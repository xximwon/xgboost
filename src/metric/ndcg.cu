/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <thrust/iterator/counting_iterator.h>  // make_counting_iterator

#include <cstddef>                              // std::size_t

#include "../common/cuda_context.cuh"           // CUDAContext
#include "../common/device_helpers.cuh"         // device_vector
#include "../common/optional_weight.h"          // MakeOptionalWeights
#include "../common/ranking_utils.cuh"          // CalcQueriesInvIDCG
#include "xgboost/base.h"                       // bst_group_t
#include "xgboost/context.h"                    // Context
#include "xgboost/data.h"                       // MetaInfo
#include "xgboost/host_device_vector.h"         // HostDeviceVector
#include "xgboost/linalg.h"                     // MakeTensorView

namespace xgboost {
namespace metric {
namespace cuda_impl {
double NDCGScore(Context const* ctx, MetaInfo const& info, HostDeviceVector<float> const& predt,
                 std::size_t k) {
  auto d_weight = common::MakeOptionalWeights(ctx, info.weights_);
  auto d_labels = info.labels.View(ctx->gpu_id).Slice(linalg::All(), 0);
  predt.SetDevice(ctx->gpu_id);
  auto d_predts = linalg::MakeTensorView(predt.ConstDeviceSpan(), {predt.Size()}, ctx->gpu_id);
  dh::device_vector<bst_group_t> group_ptr(info.group_ptr_.size());
  auto d_group_ptr = dh::ToSpan(group_ptr);
  auto const* cuctx = ctx->CUDACtx();
  dh::safe_cuda(cudaMemcpyAsync(d_group_ptr.data(), info.group_ptr_.data(),
                                d_group_ptr.size_bytes(), cudaMemcpyHostToDevice, cuctx->Stream()));
  auto n_groups = info.group_ptr_.size() - 1;

  dh::device_vector<double> inv_IDCG(n_groups);  // NOLINT
  auto d_inv_idcg = dh::ToSpan(inv_IDCG);
  CalcQueriesInvIDCG(ctx, d_labels, d_group_ptr, d_inv_idcg, k);
  dh::device_vector<std::size_t> sorted_idx(info.labels.Shape(0));
  auto d_sorted_idx = dh::ToSpan(sorted_idx);

  dh::SegmentedArgSort<false>(d_predts.Values(), d_group_ptr, d_sorted_idx);
  dh::device_vector<double> out_dcg(n_groups);
  auto d_out_dcg = dh::ToSpan(out_dcg);
  ltr::cuda_impl::CalcQueriesDCG(ctx, d_labels, d_sorted_idx, d_group_ptr, k, d_out_dcg);

  auto it = dh::MakeTransformIterator<double>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) { return d_out_dcg[i] * d_inv_idcg[i]; });
  // fixme: weight
  return thrust::reduce(cuctx->CTP(), it, it + d_out_dcg.size());
}
}  // namespace cuda_impl
}  // namespace metric
}  // namespace xgboost
