/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#pragma once

#include <cub/cub.cuh>         // DispatchSegmentedRadixSort,NullType,DoubleBuffer

#include "cuda_context.cuh"    // CUDAContext
#include "device_helpers.cuh"  // TemporaryArray
#include "xgboost/base.h"
#include "xgboost/context.h"   // Context
#include "xgboost/span.h"      // Span,byte

namespace xgboost {
namespace common {
namespace cuda_impl {
namespace detail {
// Wrapper around cub sort to define is_decending
template <bool IS_DESCENDING, typename KeyT, typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
static void DeviceSegmentedRadixSortKeys(CUDAContext const *ctx, void *d_temp_storage,
                                         size_t &temp_storage_bytes, const KeyT *d_keys_in,
                                         KeyT *d_keys_out, int num_items, int num_segments,
                                         BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT d_end_offsets, int begin_bit = 0,
                                         int end_bit = sizeof(KeyT) * 8,
                                         bool debug_synchronous = false) {
  typedef int OffsetT;

  // Null value type
  cub::DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
  cub::DoubleBuffer<cub::NullType> d_values;

  dh::safe_cuda((cub::DispatchSegmentedRadixSort<
                 IS_DESCENDING, KeyT, cub::NullType, BeginOffsetIteratorT, EndOffsetIteratorT,
                 OffsetT>::Dispatch(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
                                    num_segments, d_begin_offsets, d_end_offsets, begin_bit,
                                    end_bit, false, ctx->Stream(), debug_synchronous)));
}
}  // namespace detail

template <bool descending, typename U, typename V>
inline void SegmentedSortKeys(Context const *ctx, Span<V const> group_ptr,
                              xgboost::common::Span<U> out_sorted_values) {
  CHECK_GE(group_ptr.size(), 1ul);
  size_t n_groups = group_ptr.size() - 1;
  size_t bytes = 0;
  auto const *cuctx = ctx->CUDACtx();
  CHECK(cuctx);
  detail::DeviceSegmentedRadixSortKeys<descending>(
      cuctx, nullptr, bytes, out_sorted_values.data(), out_sorted_values.data(),
      out_sorted_values.size(), n_groups, group_ptr.data(), group_ptr.data() + 1);
  dh::TemporaryArray<xgboost::common::byte> temp_storage(bytes);
  detail::DeviceSegmentedRadixSortKeys<descending>(
      cuctx, temp_storage.data().get(), bytes, out_sorted_values.data(), out_sorted_values.data(),
      out_sorted_values.size(), n_groups, group_ptr.data(), group_ptr.data() + 1);
}
}  // namespace cuda_impl
}  // namespace common
}  // namespace xgboost
