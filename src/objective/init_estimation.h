#ifndef XGBOOST_OBJECTIVE_ESSTIMATION_H_
#define XGBOOST_OBJECTIVE_ESSTIMATION_H_
#include "../common/linalg_op.h"
#include "../common/stats.h"
#include "rabit/rabit.h"
#include "xgboost/data.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/learner.h"

namespace xgboost {
namespace obj {

template <int32_t D>
void ValidateBaseMarginShape(linalg::Tensor<float, D> const& margin, bst_row_t n_samples,
                             bst_group_t n_groups) {
  // FIXME: Bindings other than Python doesn't have shape.
  std::string expected{"Invalid shape of base_margin. Expected: (" + std::to_string(n_samples) +
                       ", " + std::to_string(n_groups) + ")"};
  CHECK_EQ(margin.Shape(0), n_samples) << expected;
  CHECK_EQ(margin.Shape(1), n_groups) << expected;
}

struct MultiTargetMixIn {
  static uint32_t Targets(MetaInfo const& info) {
    return std::max(static_cast<size_t>(1), info.labels.Shape(1));
  }
};

inline void InitPrediction(Context const* ctx, MetaInfo const& info, LearnerModelParam const* model,
                           HostDeviceVector<float>* out_predt) {
  size_t n_groups = model->num_output_group;
  size_t n = n_groups * info.num_row_;
  if (!ctx->IsCPU()) {
    out_predt->SetDevice(ctx->gpu_id);
  }
  out_predt->Resize(n);
  HostDeviceVector<bst_float> const* base_margin = info.base_margin_.Data();
  if (base_margin->Size() != 0) {
    ValidateBaseMarginShape(info.base_margin_, info.num_row_, 1);
    out_predt->Copy(*base_margin);
  }
}

struct InitialEstimationRegression : public MultiTargetMixIn {
  static void Constant(Context const* ctx, MetaInfo const& info, LearnerModelParam const* model,
                       HostDeviceVector<float>* out_predt) {
    InitPrediction(ctx, info, model, out_predt);
    HostDeviceVector<bst_float> const* base_margin = info.base_margin_.Data();
    if (!ctx->IsCPU()) {
      out_predt->SetDevice(ctx->gpu_id);
    }
    if (base_margin->Empty()) {
      // cannot rely on the Resize to fill as it might skip if the size is already correct.
      out_predt->Fill(model->base_score);
    }
  }

  static void QuantileGPU(Context const* ctx, float alpha, MetaInfo const& info,
                          LearnerModelParam const* model, HostDeviceVector<float>* out_predt);

  static float Quantile(Context const* ctx, float alpha, MetaInfo const& info,
                        LearnerModelParam const* model, HostDeviceVector<float>* out_predt) {
    InitPrediction(ctx, info, model, out_predt);
    // fixme: skip if base margin is set.
    out_predt->Fill(0);
    auto const& h_labels = info.labels.HostView();
    auto n_targets = Targets(info);
    CHECK_EQ(n_targets, 1);
    float q{0};
    for (size_t t = 0; t < n_targets; ++t) {
      auto iter = common::MakeIndexTransformIter([&](size_t i) { return h_labels(i, t); });
      if (info.weights_.Empty()) {
        q = common::Quantile(alpha, iter, iter + h_labels.Shape(0));
      } else {
        auto const& h_weight = info.weights_.HostVector();
        q = common::WeightedQuantile(alpha, iter, iter + h_labels.Shape(0), h_weight.cbegin());
      }
      auto h_predt =
          linalg::MakeTensorView(out_predt->HostSpan(), h_labels.Shape(), Context::kCpuId);
      rabit::Allreduce<rabit::op::Sum>(&q, 1);
      q /= static_cast<float>(rabit::GetWorldSize());
    }
    return q;
  }

  static void MeanCUDA(Context const* ctx, MetaInfo const& info, LearnerModelParam const* model,
                       HostDeviceVector<float>* out_predt);

  static void Mean(Context const* ctx, MetaInfo const& info, LearnerModelParam const* model,
                   HostDeviceVector<float>* out_predt) {
    InitPrediction(ctx, info, model, out_predt);
    auto const& h_labels = info.labels.HostView();
    auto n_targets = Targets(info);
    for (size_t t = 0; t < n_targets; ++t) {
      auto const n = static_cast<double>(h_labels.Shape(0));
      auto weights = common::OptionalWeights{info.weights_.ConstHostSpan()};
      auto wit = common::MakeIndexTransformIter([&](size_t i) -> double { return weights[i]; });
      auto wn = std::accumulate(wit, wit + n, 0.0);
      auto it = common::MakeIndexTransformIter(
          [&](size_t i) { return h_labels(i, t) * weights[i] / wn; });
      auto mean = std::accumulate(it, it + n, 0.f);

      auto h_predt =
          linalg::MakeTensorView(out_predt->HostSpan(), h_labels.Shape(), Context::kCpuId);
      std::array<double, 2> results{mean, wn};
      rabit::Allreduce<rabit::op::Sum>(results.data(), 2);
      for (size_t i = 0; i < h_labels.Shape(0); ++i) {
        h_predt(i, t) = mean;
      }
    }
  }
};
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_ESSTIMATION_H_
