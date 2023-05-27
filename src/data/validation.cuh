/**
 *  Copyright 2023, XGBoost Contributors
 */
#ifndef XGBOOST_DATA_VALIDATION_CUH_
#define XGBOOST_DATA_VALIDATION_CUH_
#include <thrust/functional.h>  // for logical_and

#include <iterator>     // for iterator_traits, distance
#include <type_traits>  // for enable_if_t, is_same_v

#include "../common/cuda_context.cuh"
#include "../common/device_helpers.cuh"  // for Reduce
#include "xgboost/context.h"             // for Context

namespace xgboost::data::cuda_impl {
// The default implementation in thrust optimizes any_of/none_of/all_of by using small
// intervals to early stop. But we expect all data to be valid here, using small intervals
// only decreases performance due to excessive kernel launch and stream synchronization.
template <typename It, typename Pred>
bool AllOfValid(Context const* ctx, It first, It last, Pred is_valid) {
  auto const* cuctx = ctx->CUDACtx();
  auto it = dh::MakeTransformIterator<bool>(first,
                                            [=] __device__(auto v) -> bool { return is_valid(v); });
  auto n = std::distance(first, last);
  return dh::Reduce(cuctx->CTP(), it, it + n, true, thrust::logical_and<>{});
}

template <typename It, typename Pred>
bool NoneOfInvalid(Context const* ctx, It first, It last, Pred is_invalid) {
  auto const* cuctx = ctx->CUDACtx();
  // turns it into all of valid
  auto it = dh::MakeTransformIterator<bool>(
      first, [=] __device__(auto v) -> bool { return !is_invalid(v); });
  auto n = std::distance(first, last);
  return dh::Reduce(cuctx->CTP(), it, it + n, true, thrust::logical_and<>{});
}
}  // namespace xgboost::data::cuda_impl
#endif  // XGBOOST_DATA_VALIDATION_CUH_
