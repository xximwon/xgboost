/**
 * Copyright 2023, XGBoost Contributors
 */
#ifndef XGBOOST_DATA_BATCH_UTILS_H_
#define XGBOOST_DATA_BATCH_UTILS_H_

#include "xgboost/data.h"  // for BatchParam

namespace xgboost::data::detail {
// At least one batch parameter is initialized.
inline void CheckEmpty(BatchParam const& l, BatchParam const& r) {
  if (!l.Initialized()) {
    CHECK(r.Initialized()) << "Batch parameter is not initialized.";
  }
}

/**
 * \brief Should we regenerate the gradient index?
 *
 * \param old Parameter stored in DMatrix.
 * \param p   New parameter passed in by caller.
 */
inline bool RegenGHist(BatchParam old, BatchParam p) {
  // Parameter is renewed or caller requests a regen
  if (!p.Initialized()) {
    // Empty parameter is passed in, don't regenerate so that we can use gindex in
    // predictor, which doesn't have any training parameter.
    return false;
  }
  return p.regen || old.ParamNotEqual(p);
}

/**
 * @brief Re-order the row pointer according to a sorted index array.
 */
template <typename InRowPtr, typename OutRowPtr>
void ReorderOffset(InRowPtr const& in, std::vector<std::size_t> const& sorted_idx,
                   OutRowPtr* p_out) {
  auto& out = *p_out;
  CHECK_EQ(in.size(), out.size());
  auto p_idx = sorted_idx.data();

  out[0] = 0;
  for (std::size_t i = 0; i < out.size() - 1; ++i) {
    auto ridx = p_idx[i];
    auto length = in[ridx + 1] - in[ridx];
    out[i + 1] = length;
  }
  std::partial_sum(out.cbegin(), out.cend(), out.begin());
}
}  // namespace xgboost::data::detail
#endif  // XGBOOST_DATA_BATCH_UTILS_H_
