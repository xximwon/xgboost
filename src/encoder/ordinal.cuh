#pragma once

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/find.h>  // for find_if
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>          // for size_t
#include <cstdint>          // for int32_t, int8_t
#include <cuda/functional>  // for proclaim_return_type
#include <cuda/std/variant>
#include <sstream>    // for stringstream
#include <stdexcept>  // for logic_error

#include "../common/algorithm.cuh"
#include "../common/device_helpers.cuh"
#include "ordinal.h"
#include "types.h"  // for Overloaded

namespace enc::cuda_impl {
struct SegmentedSearchSortedStrOp {
  DeviceColumnsView haystack_v;             // The training set
  Span<std::int32_t const> ref_sorted_idx;  // Sorted index for the training set
  DeviceColumnsView needles_v;              // Keys
  std::size_t f_idx;                        // Feature (segment) index

  [[nodiscard]] __device__ std::int32_t operator()(std::int32_t i) const {
    using detail::SearchKey;
    auto haystack = cuda::std::get<CatStrArrayView>(haystack_v.columns[f_idx]);
    auto needles = cuda::std::get<CatStrArrayView>(needles_v.columns[f_idx]);
    // Get the search key
    auto idx = i - needles_v.feature_segments[f_idx];  // index local to the feature
    auto begin = needles.offsets[idx];
    auto end = needles.offsets[idx + 1];
    auto needle = needles.values.subspan(begin, end - begin);

    // Search the key from the training set
    auto it = thrust::make_counting_iterator(0);
    auto f_sorted_idx = ref_sorted_idx.subspan(
        haystack_v.feature_segments[f_idx],
        haystack_v.feature_segments[f_idx + 1] - haystack_v.feature_segments[f_idx]);
    auto end_it = it + f_sorted_idx.size();
    auto ret_it = thrust::lower_bound(thrust::seq, it, end_it, SearchKey(), [&](auto l, auto r) {
      Span<std::int8_t const> l_str;
      if (l == SearchKey()) {
        l_str = needle;
      } else {
        auto l_idx = f_sorted_idx[l];
        auto l_beg = haystack.offsets[l_idx];
        auto l_end = haystack.offsets[l_idx + 1];
        l_str = haystack.values.subspan(l_beg, l_end - l_beg);
      }

      Span<std::int8_t const> r_str;
      if (r == SearchKey()) {
        r_str = needle;
      } else {
        auto r_idx = f_sorted_idx[r];
        auto r_beg = haystack.offsets[r_idx];
        auto r_end = haystack.offsets[r_idx + 1];
        r_str = haystack.values.subspan(r_beg, r_end - r_beg);
      }

      return l_str < r_str;
    });
    if (ret_it == it + f_sorted_idx.size()) {
      return -1;  // not found
    }
    return *ret_it;
  }
};

template <typename T>
struct SegmentedSearchSortedNumOp {
  DeviceColumnsView haystack_v;             // The training set
  Span<std::int32_t const> ref_sorted_idx;  // Sorted index for the training set
  DeviceColumnsView needles_v;              // Keys
  std::size_t f_idx;                        // Feature (segment) index

  [[nodiscard]] __device__ std::int32_t operator()(std::int32_t i) const {
    using detail::SearchKey;
    auto haystack = cuda::std::get<Span<T const>>(haystack_v.columns[f_idx]);
    auto needles = cuda::std::get<Span<T const>>(needles_v.columns[f_idx]);
    // Get the search key
    auto idx = i - needles_v.feature_segments[f_idx];  // index local to the feature
    auto needle = needles[idx];
    // Search the key from the training set
    auto it = thrust::make_counting_iterator(0);
    auto f_sorted_idx = ref_sorted_idx.subspan(
        haystack_v.feature_segments[f_idx],
        haystack_v.feature_segments[f_idx + 1] - haystack_v.feature_segments[f_idx]);
    auto end_it = it + f_sorted_idx.size();
    auto ret_it = thrust::lower_bound(thrust::seq, it, end_it, SearchKey(), [&](auto l, auto r) {
      T l_value = l == SearchKey() ? needle : haystack[ref_sorted_idx[l]];
      T r_value = r == SearchKey() ? needle : haystack[ref_sorted_idx[r]];
      return l_value < r_value;
    });
    if (ret_it == it + f_sorted_idx.size()) {
      return SearchKey();  // not found
    }
    return *ret_it;
  }
};

// fixme: the sort only needs to be done once.

// Sort the names of the original encoding, returns the sorted index with the size of the
// number of categories in the original encoding.
inline thrust::device_vector<std::int32_t> SortNames(DeviceColumnsView orig_enc) {
  auto n_total_cats = orig_enc.n_total_cats;
  thrust::device_vector<std::int32_t> sorted_idx(n_total_cats);
  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  xgboost::Context ctx;
  ctx.Init(xgboost::Args{{"device", "cuda"}});
  xgboost::common::SegmentedSequence(&ctx, orig_enc.feature_segments, d_sorted_idx);

  // <fidx, sorted_idx>
  using Pair = cuda::std::pair<std::int32_t, std::int32_t>;
  thrust::device_vector<Pair> keys(n_total_cats);
  auto key_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      cuda::proclaim_return_type<Pair>([=] __device__(std::int32_t i) {
        auto seg = dh::SegmentId(orig_enc.feature_segments, i);
        auto idx = d_sorted_idx[i];
        return cuda::std::make_pair(static_cast<std::int32_t>(seg), idx);
      }));
  thrust::copy(key_it, key_it + n_total_cats, keys.begin());

  // fixme: thrust exec
  thrust::sort(thrust::device, keys.begin(), keys.end(),
               cuda::proclaim_return_type<bool>([=] __device__(Pair const& l, Pair const& r) {
                 if (l.first == r.first) {  // same feature
                   auto const& col = orig_enc.columns[l.first];
                   return cuda::std::visit(
                       Overloaded{[&l, &r](CatStrArrayView const& str) -> bool {
                                    auto l_beg = str.offsets[l.second];
                                    auto l_end = str.offsets[l.second + 1];
                                    auto l_str = str.values.subspan(l_beg, l_end - l_beg);

                                    auto r_beg = str.offsets[r.second];
                                    auto r_end = str.offsets[r.second + 1];
                                    auto r_str = str.values.subspan(r_beg, r_end - r_beg);
                                    return l_str < r_str;
                                  },
                                  [&](auto&& values) {
                                    return values[l.second] < values[r.second];
                                  }},
                       col);
                 }
                 return l.first < r.first;
               }));

  // Extract the sorted index out from sorted keys.
  auto s_keys = dh::ToSpan(keys);
  auto it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      cuda::proclaim_return_type<decltype(Pair{}.second)>(
          [=] __device__(std::int32_t i) { return s_keys[i].second; }));
  thrust::copy(it, it + sorted_idx.size(), sorted_idx.begin());
  return sorted_idx;
}

class CudaCategoryRecoder {
 public:
  thrust::device_vector<std::int32_t> Recode(DeviceColumnsView orig_enc,
                                             DeviceColumnsView new_enc) const {
    CHECK_EQ(orig_enc.feature_segments.size(), orig_enc.columns.size() + 1);
    /**
     * Sort the reference encoding.
     */
    auto ref_sorted_idx = SortNames(orig_enc);
    auto d_sorted_idx = dh::ToSpan(ref_sorted_idx);

    /**
     * search the index for the new encoding
     */
    thrust::device_vector<std::int32_t> mapping(new_enc.n_total_cats, -1);
    auto d_mapping = dh::ToSpan(mapping);

    thrust::for_each_n(
        thrust::make_counting_iterator(0), new_enc.n_total_cats, [=] __device__(std::int32_t i) {
          auto f_idx = dh::SegmentId(new_enc.feature_segments, i);
          std::int32_t searched_idx{-1};
          auto const& col = orig_enc.columns[f_idx];
          cuda::std::visit(
              Overloaded{
                  [&](CatStrArrayView const& str) {
                    auto op = SegmentedSearchSortedStrOp{orig_enc, d_sorted_idx, new_enc, f_idx};
                    searched_idx = op(i);
                  },
                  [&](auto&& values) {
                    using T = typename std::decay_t<decltype(values)>::value_type;
                    auto op = SegmentedSearchSortedNumOp<T>{orig_enc, d_sorted_idx, new_enc, f_idx};
                    searched_idx = op(i);
                  }},
              col);

          auto f_sorted_idx = d_sorted_idx.subspan(
              orig_enc.feature_segments[f_idx],
              orig_enc.feature_segments[f_idx + 1] - orig_enc.feature_segments[f_idx]);

          std::int32_t idx = -1;
          if (searched_idx != -1) {
            idx = f_sorted_idx[searched_idx];
          }

          auto f_beg = new_enc.feature_segments[f_idx];
          auto f_end = new_enc.feature_segments[f_idx + 1];
          auto f_mapping = d_mapping.subspan(f_beg, f_end - f_beg);
          f_mapping[i - f_beg] = idx;
        });

    auto err_it = thrust::find_if(
        mapping.cbegin(), mapping.cend(),
        cuda::proclaim_return_type<bool>([=] __device__(std::int32_t v) { return v == -1; }));

    if (err_it != mapping.cend()) {
      std::vector<decltype(mapping)::value_type> h_mapping(mapping.size());
      thrust::copy(mapping.cbegin(), mapping.cend(), h_mapping.begin());
      std::vector<decltype(new_enc.feature_segments)::value_type> h_feature_segments(
          new_enc.feature_segments.size());
      thrust::copy(dh::tcbegin(new_enc.feature_segments), dh::tcend(new_enc.feature_segments),
                   h_feature_segments.begin());
      auto h_idx = std::distance(mapping.cbegin(), err_it);
      auto f_idx = dh::SegmentId(Span<std::int32_t const>{h_feature_segments}, h_idx);
      CHECK_LT(f_idx, new_enc.Size()) << h_idx;
      auto f_beg = h_feature_segments[f_idx];
      auto f_local_idx = h_idx - f_beg;

      std::vector<DeviceColumnsView::VariantT> h_columns(new_enc.columns.size());
      thrust::copy_n(dh::tcbegin(new_enc.columns), new_enc.columns.size(), h_columns.begin());

      std::stringstream name;
      auto const& col = h_columns[f_idx];
      cuda::std::visit(
          Overloaded{[&](CatStrArrayView str) {
                       std::vector<CatCharT> values(str.values.size());
                       std::vector<std::int32_t> offsets(str.offsets.size());
                       thrust::copy_n(dh::tcbegin(str.values), str.values.size(), values.data());
                       thrust::copy_n(dh::tcbegin(str.offsets), str.offsets.size(), offsets.data());

                       auto cat = Span{values}.subspan(
                           offsets[f_local_idx], offsets[f_local_idx + 1] - offsets[f_local_idx]);
                       for (auto v : cat) {
                         name.put(v);
                       }
                     },
                     [&](auto&& values) {
                       auto cat = values[f_local_idx];
                       name << cat;
                     }},
          col);

      std::stringstream ss;
      ss << "Found a category not in the training set for the " << f_idx << "th (0-based) column: `"
         << name.str() << "`";
      throw std::logic_error{ss.str()};
    }
    return mapping;
  }
};
}  // namespace enc::cuda_impl
