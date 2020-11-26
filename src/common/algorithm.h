/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_ALGORITHM_H_
#define XGBOOST_COMMON_ALGORITHM_H_
#include <algorithm>   // std::upper_bound
#include <cinttypes>   // std::size_t
#include <functional>  // std::less
#include <iterator>    // std::iterator_traits,distance
#include <vector>      // std::vector

#include "numeric.h"          // Iota
#include "xgboost/context.h"  // Context

namespace xgboost {
namespace common {
template <typename It, typename Idx>
auto SegmentId(It first, It last, Idx idx) {
  std::size_t segment_id = std::upper_bound(first, last, idx) - 1 - first;
  return segment_id;
}

template <typename Idx, typename Iter, typename V = typename std::iterator_traits<Iter>::value_type,
          typename Comp = std::less<V>>
std::vector<Idx> ArgSort(Context const *ctx, Iter begin, Iter end, Comp comp = std::less<V>{}) {
  CHECK(ctx->IsCPU());
  auto n = std::distance(begin, end);
  std::vector<Idx> result(n);
  Iota(ctx, result.begin(), result.end(), 0);
  auto op = [&begin, comp](Idx const &l, Idx const &r) { return comp(begin[l], begin[r]); };
  if (omp_in_parallel()) {
    std::stable_sort(result.begin(), result.end(), op);
  } else {
    XGBOOST_PARALLEL_STABLE_SORT(result.begin(), result.end(), op);
  }
  return result;
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_ALGORITHM_H_
