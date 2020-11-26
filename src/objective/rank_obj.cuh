/**
 * Copyright 2023 by XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_RANK_OBJ_CUH_
#define XGBOOST_OBJECTIVE_RANK_OBJ_CUH_

#include <algorithm>       // std::min
#include <cstddef>         // std::size_t

#include "xgboost/base.h"  // XGBOOST_DEVICE

namespace xgboost {
namespace obj {
template <bool with_diagonal = false>
XGBOOST_DEVICE size_t TrapezoidArea(std::size_t n, std::size_t h) {
  if (!with_diagonal) {
    n -= 1;
  }
  h = std::min(n, h);  // Specific for ranking.
  std::size_t total = ((n - (h - 1)) + n) * h / 2;
  return total;
}
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_RANK_OBJ_CUH_
