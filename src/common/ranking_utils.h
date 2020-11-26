/**
 * Copyright 2023 by XGBoost contributors
 */
#ifndef XGBOOST_COMMON_RANKING_UTILS_H_
#define XGBOOST_COMMON_RANKING_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>  // std::size_t
#include <cstdint>  // std::uint32_t
#include <limits>

#include "dmlc/parameter.h"
#include "xgboost/base.h"
#include "xgboost/context.h"  // Context
#include "xgboost/data.h"
#include "xgboost/linalg.h"  // Tensor,VectorView
#include "xgboost/logging.h"
#include "xgboost/parameter.h"

namespace xgboost {
namespace ltr {
using rel_degree_t = std::int32_t;  // NOLINT
constexpr std::size_t MaxRel() { return sizeof(rel_degree_t) * 8; }
}  // namespace ltr

XGBOOST_DEVICE inline float CalcNDCGGain(std::uint32_t label) {
  label = std::min(31u, label);
  return (1u << label) - 1;
}

XGBOOST_DEVICE inline double CalcNDCGDiscount(std::size_t idx) {
  return 1.0 / std::log2(static_cast<double>(idx) + 2.0);
}

inline float CalcDCGAtK(linalg::VectorView<float const> scores, std::size_t k) {
  double sumdcg = 0;
  for (size_t i = 0; i < std::min(scores.Size(), k); ++i) {
    float gain = CalcNDCGGain(scores(i));
    double discount = CalcNDCGDiscount(i);
    sumdcg += gain * discount;
  }
  return sumdcg;
}

inline double CalcInvIDCG(linalg::VectorView<float const> sorted_labels, std::size_t p) {
  double sumdcg = 0;
  for (std::size_t i = 0; i < std::min(p, sorted_labels.Size()); ++i) {
    double gain = CalcNDCGGain(sorted_labels(i));
    double discount = CalcNDCGDiscount(i);
    sumdcg += gain * discount;
  }
  // When IDCG is 0
  sumdcg = sumdcg == 0.0f ? 0.0f : 1.0 / sumdcg;
  return sumdcg;
}
namespace ltr {
namespace cpu_impl {
void CalcInvIDCG(Context const* ctx, MetaInfo const& info, std::size_t p,
                 linalg::Vector<double>* out);
}  // namespace cpu_impl

inline void CalcInvIDCG(Context const* ctx, MetaInfo const& info, std::size_t p,
                        linalg::Vector<double>* out) {
  if (ctx->IsCPU()) {
    cpu_impl::CalcInvIDCG(ctx, info, p, out);
  } else {
    LOG(FATAL) << "";
  }
}
}  // namespace ltr

struct LambdaMARTParam : public XGBoostParameter<LambdaMARTParam> {
  // Top k result used for evaluating document ranks.
  std::size_t lambdamart_truncation;
  bool lambdamart_unbiased;
  bool lambdamart_exp_gain;

  DMLC_DECLARE_PARAMETER(LambdaMARTParam) {
    DMLC_DECLARE_FIELD(lambdamart_truncation)
        .set_default(std::numeric_limits<std::size_t>::max())
        .set_lower_bound(1)
        .describe("Ranking truncation level.");
    DMLC_DECLARE_FIELD(lambdamart_unbiased)
        .set_default(true)
        .describe("Unbiased lambda mart. Use IPW to debias click position");
    DMLC_DECLARE_FIELD(lambdamart_exp_gain)
        .set_default(true)
        .describe("When set to true, the label gain is 2^rel - 1, otherwise it's rel.");
  }
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_RANKING_UTILS_H_
