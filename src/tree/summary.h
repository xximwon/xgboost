/**
 * Copyright 2023, XGBoost Contributors
 */
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/linalg.h"
#include "xgboost/parameter.h"

namespace xgboost::tree {
struct SummarizerParam : public XGBoostParameter<SummarizerParam> {
  bst_target_t num_grad;
};

inline linalg::Matrix<GradientPair> SummarizeGradient(Context const* ctx,
                                                      linalg::Matrix<GradientPair> const& in,
                                                      SummarizerParam const& p) {
  linalg::Matrix<GradientPair> out{
      linalg::Empty<GradientPair>(ctx, linalg::kF, in.Shape(0), p.num_grad)};
  if (ctx->IsCPU()) {
    auto h_in = in.HostView();
    auto h_out = out.HostView();
  } else {
    LOG(FATAL) << "Not implemented";
  }
  return out;
}
}  // namespace xgboost::tree
