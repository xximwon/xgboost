/*!
 * Copyright 2020 by XGBoost Contributors
 */
#ifndef HISTOGRAM_CUH_
#define HISTOGRAM_CUH_
#include <thrust/transform.h>

#include "feature_groups.cuh"

#include "../../data/ellpack_page.cuh"

namespace xgboost {
namespace tree {

template <typename GradientSumT>
GradientSumT CreateRoundingFactor(common::Span<GradientPair const> gpair);

template <typename T>
XGBOOST_DEV_INLINE T TruncateWithRoundingFactor(T const rounding_factor, float const x) {
  return (rounding_factor + static_cast<T>(x)) - rounding_factor;
}

template <typename GradientSumT>
class GPUHistogramBuilder {
  dim3 grids_;
  uint32_t block_threads_;
  size_t smem_size_;
  bool shared_;

 public:
  explicit GPUHistogramBuilder(FeatureGroupsAccessor const& feature_groups, uint32_t tunning_threads = 0);
  void Build(EllpackDeviceAccessor const &matrix,
             FeatureGroupsAccessor const &feature_groups,
             common::Span<GradientPair const> gpair,
             common::Span<const uint32_t> ridx,
             common::Span<GradientSumT> histogram, GradientSumT rounding);
};
}  // namespace tree
}  // namespace xgboost

#endif  // HISTOGRAM_CUH_
