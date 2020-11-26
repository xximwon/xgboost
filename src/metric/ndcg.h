#ifndef XGBOOST_METRIC_NDCG_H_
#define XGBOOST_METRIC_NDCG_H_
/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <cstddef>  // std::size_t

#include "../common/common.h"  // AssertGPUSupport
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"

namespace xgboost {
namespace metric {
namespace cuda_impl {
double NDCGScore(Context const* ctx, MetaInfo const& info, HostDeviceVector<float> const& predt,
                 std::size_t k);
#if !defined(XGBOOST_USE_CUDA)
inline double NDCGScore(Context const*, MetaInfo const&, HostDeviceVector<float> const&,
                        std::size_t) {
  common::AssertGPUSupport();
  return 0;
}
#endif
}  // namespace cuda_impl
}  // namespace metric
}  // namespace xgboost
#endif  // XGBOOST_METRIC_NDCG_H_
