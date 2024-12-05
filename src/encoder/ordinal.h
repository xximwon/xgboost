/**
 * @brief Orindal re-coder for categorical features.
 *
 * For training with dataframes, we use the default encoding provided by the dataframe
 * implementation. However, we need a way to ensure the encoding is consistent at test
 * time, which is often not the case. This module re-code the test data given the train
 * time encoding (mapping between categories to dense discrete integers starting from 0).
 *
 * The algorithm proceeds as follow:
 *
 * Given the categories used for training [c, b, d, a], the ordering of this list is the
 * encoding, c maps to 0, b maps to 1, so on and so forth. At test time, we recieve an
 * encoding [c, a, b], which differs from the encoding used for training and we need to
 * re-code the data.
 *
 * First, we perform an `argsort` on the training categories in the increasing order,
 * obtaining a list of index: [3, 1, 0, 2], which corresponds to [a, b, c, d] as a sorted
 * list. Then we perform binary search for each category in the test time encoding [c, a,
 * b] with the training encoding as the sorted haystack. Since c is the third item of
 * sorted training encoding, we have an index 2 (0-based) for c, index 0 for a, and index
 * 1 for b. After the bianry search, we obtain a new list of index [2, 0, 1]. Using this
 * index list, we can recover the training encoding for the test dataset [0, 3, 1]. This
 * has O(NlogN) complexity with N as the number of categories (assuming the length of the
 * strings as constant). Originally, the encoding for test data set is [0, 1, 2] for [c,
 * a, b], now we have a mapping {0 -> 0, 1 -> 3, 2 -> 1} for re-coding the data.
 */

#pragma once
#include <algorithm>
#include <cstdint>    // for int32_t, int8_t
#include <numeric>    // for accumulate, iota
#include <sstream>    // for stringstream
#include <stdexcept>  // for logic_error
#include <variant>    // for variant
#include <vector>     // for vector

#include "../common/transform_iterator.h"
#include "types.h"         // for Overloaded
#include "xgboost/span.h"  // for Span

#if defined(XGBOOST_USE_CUDA)
#include <cuda/std/variant>  // for variant
#endif                       // defined(XGBOOST_USE_CUDA)

namespace enc {
using xgboost::common::MakeIndexTransformIter;
using xgboost::common::Span;

using CatCharT = std::int8_t;

/**
 * @brief String names of categorical data. Represented in the arrow StringArray format.
 */
struct CatStrArrayView {
  Span<std::int32_t const> offsets;
  Span<CatCharT const> values;

  [[nodiscard]] XGBOOST_DEVICE bool empty() const { return offsets.empty(); }  // NOLINT
  [[nodiscard]] XGBOOST_DEVICE std::size_t size() const {                      // NOLINT
    return this->empty() ? 0 : this->offsets.size() - 1;
  }

  [[nodiscard]] std::size_t SizeBytes() const {
    return this->offsets.size_bytes() + values.size_bytes();
  }
};
/**
 * @brief All the types supported by the encoder.
 */
using CatIndexViewTypes =
    std::tuple<enc::CatStrArrayView, Span<std::int8_t const>, Span<std::int16_t const>,
               Span<std::int32_t const>, Span<std::int64_t const>, Span<float const>,
               Span<double const>>;

/**
 * @brief Host categories view for a single column.
 */
using HostCatIndexView = cpu_impl::TupToVarT<CatIndexViewTypes>;

#if defined(XGBOOST_USE_CUDA)
/**
 * @brief Device categories view for a single column.
 */
using DeviceCatIndexView = cuda_impl::TupToVarT<CatIndexViewTypes>;
#endif  // defined(XGBOOST_USE_CUDA)

namespace detail {
constexpr std::int32_t SearchKey() { return -1; };

template <typename Variant>
struct ColumnsViewImpl {
  using VariantT = Variant;

  Span<Variant const> columns;

  // Segment pointer for features, each segment represents the number of categories in a feature.
  Span<std::int32_t const> feature_segments;
  // The total number of cats in all features, equals feature_segments.back()
  std::int32_t n_total_cats{0};

  [[nodiscard]] std::size_t Size() const { return columns.size(); }
  [[nodiscard]] bool Empty() const { return this->Size() == 0; }
  [[nodiscard]] auto operator[](std::size_t i) const { return columns[i]; }
};
}  // namespace detail

/**
 * @brief Host view for all columns
 */
using HostColumnsView = detail::ColumnsViewImpl<HostCatIndexView>;
#if defined(XGBOOST_USE_CUDA)
/**
 * @brief Device view for all columns
 */
using DeviceColumnsView = detail::ColumnsViewImpl<DeviceCatIndexView>;
#endif  // defined(XGBOOST_USE_CUDA)

/**
 * @brief The result encoding.
 */
struct MappingView {
  Span<std::int32_t const> offsets;
  Span<std::int32_t const> cats_mapping;

  /**
   * @brief Get the encoding for a specific feature.
   */
  [[nodiscard]] XGBOOST_DEVICE auto operator[](std::size_t f_idx) const {
    return cats_mapping.subspan(offsets[f_idx], offsets[f_idx + 1] - offsets[f_idx]);
  }
  [[nodiscard]] XGBOOST_DEVICE bool Empty() const { return offsets.empty(); }
};

template <typename InIt, typename OutIt, typename Comp>
void ArgSort(InIt in_first, InIt in_last, OutIt out_first, Comp comp = std::less{}) {
  auto n = std::distance(in_first, in_last);
  using Idx = typename std::iterator_traits<OutIt>::value_type;

  auto out_last = out_first + n;
  std::iota(out_first, out_last, 0);
  auto op = [&](Idx const &l, Idx const &r) {
    return comp(in_first[l], in_first[r]);
  };
  std::stable_sort(out_first, out_last, op);
}

inline std::vector<std::int32_t> SortNames(HostCatIndexView const &cats) {
  auto it = MakeIndexTransformIter([](auto i) { return i; });
  auto n_categories = std::visit([](auto &&arg) { return arg.size(); }, cats);
  std::vector<std::int32_t> sorted_idx(n_categories);

  std::visit(
      Overloaded{[&](CatStrArrayView const &str) {
                   ArgSort(it, it + str.size(), sorted_idx.begin(),
                           [&](std::size_t l, std::size_t r) {
                             auto l_beg = str.offsets[l];
                             auto l_str = str.values.subspan(l_beg, str.offsets[l + 1] - l_beg);

                             auto r_beg = str.offsets[r];
                             auto r_str = str.values.subspan(r_beg, str.offsets[r + 1] - r_beg);

                             return l_str < r_str;
                           });
                 },
                 [&](auto &&values) {
                   ArgSort(it, it + values.size(), sorted_idx.begin(),
                           [&](std::size_t l, std::size_t r) { return values[l] < values[r]; });
                 }},
      cats);

  return sorted_idx;
}

[[nodiscard]] inline std::size_t SearchSorted(CatStrArrayView haystack,
                                              Span<std::int32_t const> ref_sorted_idx,
                                              Span<std::int8_t const> needle) {
  auto it = MakeIndexTransformIter([](auto i) { return static_cast<std::int32_t>(i); });
  auto const h_off = haystack.offsets;
  auto const h_data = haystack.values;
  using detail::SearchKey;
  auto ret_it = std::lower_bound(it, it + haystack.size(), SearchKey(), [&](auto l, auto r) {
    Span<std::int8_t const> l_str;
    if (l == SearchKey()) {
      l_str = needle;
    } else {
      auto l_idx = ref_sorted_idx[l];
      auto l_beg = h_off[l_idx];
      auto l_end = h_off[l_idx + 1];
      l_str = h_data.subspan(l_beg, l_end - l_beg);
    }

    Span<std::int8_t const> r_str;
    if (r == SearchKey()) {
      r_str = needle;
    } else {
      auto r_idx = ref_sorted_idx[r];
      auto r_beg = h_off[r_idx];
      auto r_end = h_off[r_idx + 1];
      r_str = h_data.subspan(r_beg, r_end - r_beg);
    }

    return l_str < r_str;
  });
  if (ret_it == it + haystack.size()) {
    std::stringstream ss;
    ss << "Found a category not in the training set: `";
    for (auto c : needle) {
      ss.put(c);
    }
    ss.put('`');
    throw std::logic_error{ss.str()};
  }
  return *ret_it;
}

template <typename T>
[[nodiscard]] std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>, std::size_t>
SearchSorted(Span<T const> haystack, Span<std::int32_t const> ref_sorted_idx, T needle) {
  using detail::SearchKey;
  auto it = MakeIndexTransformIter([](auto i) { return static_cast<std::int32_t>(i); });
  auto ret_it = std::lower_bound(it, it + haystack.size(), SearchKey(), [&](auto l, auto r) {
    T l_value = l == SearchKey() ? needle : haystack[ref_sorted_idx[l]];
    T r_value = r == SearchKey() ? needle : haystack[ref_sorted_idx[r]];
    return l_value < r_value;
  });
  if (ret_it == it + haystack.size()) {
    std::stringstream ss;
    ss << "Found a category not in the training set: `" << needle << "`";
    throw std::logic_error{ss.str()};
  }
  return *ret_it;
}

class CategoryRecoder {
 public:
  std::vector<std::int32_t> Recode(HostColumnsView const orig_enc,
                                   HostColumnsView const new_enc) const {
    if (orig_enc.Size() != new_enc.Size()) {
      throw std::logic_error{"Invalid encoding."};
    }

    std::vector<std::int32_t> mapping;
    for (std::size_t fidx = 0, n_features = orig_enc.Size(); fidx < n_features; fidx++) {
      bool is_empty = std::visit([](auto &&arg) { return arg.empty(); }, orig_enc.columns[fidx]);
      if (is_empty) {
        continue;
      }

      auto ref_sorted_idx = SortNames(orig_enc.columns[fidx]);

      auto n_new_categories =
          std::visit([](auto &&arg) { return arg.size(); }, new_enc.columns[fidx]);
      std::vector<std::int32_t> searched_idx(n_new_categories, -1);
      auto const &col = new_enc.columns[fidx];
      std::visit(Overloaded{[&](CatStrArrayView const &str) {
                              for (std::size_t j = 1, m = n_new_categories + 1; j < m; ++j) {
                                auto begin = str.offsets[j - 1];
                                auto end = str.offsets[j];
                                auto needle = str.values.subspan(begin, end - begin);
                                // fixme: check new cat type matches the old cat type
                                searched_idx[j - 1] =
                                    SearchSorted(std::get<CatStrArrayView>(orig_enc.columns[fidx]),
                                                 ref_sorted_idx, needle);
                              }
                            },
                            [&](auto &&values) {
                              for (std::size_t j = 0; j < n_new_categories; ++j) {
                                auto needle = values[j];
                                using T = typename std::decay_t<decltype(values)>::value_type;
                                searched_idx[j] =
                                    SearchSorted(std::get<Span<T const>>(orig_enc.columns[fidx]),
                                                 ref_sorted_idx, needle);
                              }
                            }},
                 col);

      std::vector<std::int32_t> code_idx(searched_idx.size());
      for (std::size_t i = 0, n = searched_idx.size(); i < n; ++i) {
        auto idx = ref_sorted_idx[searched_idx[i]];
        code_idx[i] = idx;
      }
      mapping.insert(mapping.end(), code_idx.cbegin(), code_idx.cend());
      // mapping.emplace_back(std::move(code_idx));
    }

    return mapping;
  }
};
}  // namespace enc
