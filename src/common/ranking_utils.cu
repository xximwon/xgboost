/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <cstddef>  // std::size_t

#include "device_helpers.cuh"
#include "ranking_utils.cuh"
#include "xgboost/context.h"  // Context
#include "xgboost/linalg.h"   // MatrixView
#include "xgboost/span.h"     // Span

namespace xgboost {
namespace ltr {
namespace cuda_impl {
void CalcQueriesDCG(Context const* ctx, linalg::VectorView<float const> d_labels,
                    common::Span<std::size_t const> d_sorted_idx,
                    common::Span<bst_group_t const> d_group_ptr, std::size_t k,
                    common::Span<double> out_dcg) {
  CHECK_EQ(d_group_ptr.size() - 1, out_dcg.size());
  using IdxGroup = thrust::pair<std::size_t, std::size_t>;
  auto group_it = dh::MakeTransformIterator<IdxGroup>(
      thrust::make_counting_iterator(0ull), [=] XGBOOST_DEVICE(std::size_t idx) {
        return thrust::make_pair(idx, dh::SegmentId(d_group_ptr, idx));
      });
  auto value_it = dh::MakeTransformIterator<float>(
      group_it,
      [d_labels, d_group_ptr, k, d_sorted_idx] XGBOOST_DEVICE(IdxGroup const& l) -> double {
        auto g_begin = d_group_ptr[l.second];
        auto idx_in_group = l.first - g_begin;
        if (idx_in_group >= k) {
          return 0.0;
        }

        auto gain = d_labels(d_sorted_idx[l.first]);
        double discount = CalcNDCGDiscount(idx_in_group);
        return gain * discount;
      });
  thrust::reduce_by_key(
      ctx->CUDACtx()->CTP(), group_it, group_it + d_sorted_idx.size(), value_it,
      thrust::make_discard_iterator(), dh::tbegin(out_dcg),
      [] XGBOOST_DEVICE(IdxGroup const& l, IdxGroup const& r) { return l.second == r.second; },
      thrust::plus<float>{});
}
}  // namespace cuda_impl
}  // namespace ltr
}  // namespace xgboost
