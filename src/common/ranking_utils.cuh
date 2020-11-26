/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_RANKING_UTILS_CUH_
#define XGBOOST_COMMON_RANKING_UTILS_CUH_

#include <cstddef>
#include <cub/cub.cuh>

#include "./math.h"
#include "algorithm.cuh"
#include "device_helpers.cuh"
#include "ranking_utils.h"
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace ltr {
namespace cuda_impl {
void CalcQueriesDCG(Context const *ctx, linalg::VectorView<float const> d_labels,
                    common::Span<std::size_t const> d_sorted_idx,
                    common::Span<bst_group_t const> d_group_ptr, std::size_t k,
                    common::Span<double> out_dcg);
}  // namespace cuda_impl
}  // namespace ltr

namespace common {
/**
 * \param n Number of items (length of the base)
 * \param h hight
 */
XGBOOST_DEVICE inline size_t DiscreteTrapezoidArea(size_t n, size_t h) {
  n -= 1;              // without diagonal entries
  h = std::min(n, h);  // Specific for ranking.
  size_t total = ((n - (h - 1)) + n) * h / 2;
  return total;
}

/**
 * Used for mapping many groups of trapezoid shaped computation onto CUDA blocks.  The
 * trapezoid must be on upper right corner.
 *
 * Equivalent to loops like:
 *
 * \code
 *   for (size i = 0; i < h; ++i) {
 *     for (size_t j = i + 1; j < n; ++j) {
 *        do_something();
 *     }
 *   }
 * \endcode
 */
template <typename U>
inline size_t SegmentedTrapezoidThreads(xgboost::common::Span<U> group_ptr,
                                        xgboost::common::Span<size_t> out_group_threads_ptr,
                                        size_t h) {
  CHECK_GE(group_ptr.size(), 1);
  CHECK_EQ(group_ptr.size(), out_group_threads_ptr.size());
  dh::LaunchN(group_ptr.size(), [=] XGBOOST_DEVICE(size_t idx) {
    if (idx == 0) {
      out_group_threads_ptr[0] = 0;
      return;
    }

    size_t cnt = static_cast<size_t>(group_ptr[idx] - group_ptr[idx - 1]);
    out_group_threads_ptr[idx] = DiscreteTrapezoidArea(cnt, h);
  });
  dh::InclusiveSum(out_group_threads_ptr.data(), out_group_threads_ptr.data(),
                   out_group_threads_ptr.size());
  size_t total = 0;
  dh::safe_cuda(cudaMemcpy(&total, out_group_threads_ptr.data() + out_group_threads_ptr.size() - 1,
                           sizeof(total), cudaMemcpyDeviceToHost));
  return total;
}

/**
 * Called inside kernel to obtain coordinate from trapezoid grid.
 */
XGBOOST_DEVICE inline void UnravelTrapeziodIdx(size_t i_idx, size_t n, size_t *out_i,
                                               size_t *out_j) {
  auto &i = *out_i;
  auto &j = *out_j;
  double idx = static_cast<double>(i_idx);
  double N = static_cast<double>(n);

  i = std::ceil(-(0.5 - N + std::sqrt(common::Sqr(N - 0.5) + 2.0 * (-idx - 1.0)))) - 1.0;

  auto I = static_cast<double>(i);
  size_t n_elems = -0.5 * common::Sqr(I) + (N - 0.5) * I;

  j = idx - n_elems + i + 1;
}

inline void CalcQueriesInvIDCG(Context const *ctx, linalg::VectorView<float const> d_labels,
                               common::Span<bst_group_t const> d_group_ptr,
                               common::Span<double> out_inv_IDCG, size_t truncation) {
  CHECK_GE(d_group_ptr.size(), 2ul);
  size_t n_groups = d_group_ptr.size() - 1;
  CHECK_EQ(out_inv_IDCG.size(), n_groups);
  dh::device_vector<std::size_t> sorted_idx(d_labels.Size());
  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  dh::SegmentedArgSort<false>(d_labels.Values(), d_group_ptr, d_sorted_idx);
  ltr::cuda_impl::CalcQueriesDCG(ctx, d_labels, d_sorted_idx, d_group_ptr, truncation,
                                 out_inv_IDCG);
  dh::LaunchN(out_inv_IDCG.size(), [out_inv_IDCG] XGBOOST_DEVICE(size_t idx) {
    double idcg = out_inv_IDCG[idx];
    // fixme: extract this
    out_inv_IDCG[idx] = idcg == 0.0 ? 0.0 : 1.0 / idcg;
  });
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_RANKING_UTILS_CUH_
