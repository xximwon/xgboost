/*!
 * Copyright 2019 by Contributors
 * \file probabilistic_forecasting.cc
 */
#include "xgboost/base.h"
#include "xgboost/json.h"
#include "xgboost/span.h"
#include "xgboost/logging.h"
#include "xgboost/objective.h"
#include "xgboost/parameter.h"

#include "../common/distributions.h"

namespace xgboost {

struct MLETrainParameter : public XGBoostParameter<MLETrainParameter> {
  std::string distribution;
  bool fisher_as_hessian;

  DMLC_DECLARE_PARAMETER(MLETrainParameter) {
    DMLC_DECLARE_FIELD(distribution)
        .set_default("normal");
    DMLC_DECLARE_FIELD(fisher_as_hessian)
        .set_default(false);
  }
};

DMLC_REGISTER_PARAMETER(MLETrainParameter);

class MaximumLikelihoodEstimation : public ObjFunction {
  MLETrainParameter param_;

 public:
  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);
  }

  void GetGradient(const HostDeviceVector<bst_float> &preds,
                   const MetaInfo &info, int iteration,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    out_gpair->Resize(info.num_row_ * 2);
    auto& h_gpair = out_gpair->HostVector();
    auto const& h_preds = preds.HostVector();
    auto const& h_labels = info.labels_.HostVector();

    bst_row_t y_iter = 0;
    if (param_.fisher_as_hessian) {
      for (bst_row_t i = 0; i < preds.Size(); i += 2) {
        float fi_0, fi_1;
        std::tie(fi_0, fi_1) =
            dist::Normal::FisherInfo(common::Span<float const>{h_preds}[i], h_preds[i + 1]);
        float sample_mean, sample_var;
        std::tie(sample_mean, sample_var) = dist::Normal::GradientNLL(
            h_preds[i], h_preds[i + 1],
            common::Span<float const>{h_labels}[y_iter]);

        h_gpair[i] = GradientPair {sample_mean, fi_0};
        h_gpair[i+1] = GradientPair {sample_var, fi_1};
        y_iter ++;
      }
    } else {
      for (bst_row_t i = 0; i < preds.Size(); i += 2) {
        float fi_0, fi_1;
        std::tie(fi_0, fi_1) =
            dist::Normal::FisherInfo(h_preds[i], h_preds[i + 1]);
        float sample_mean, sample_var;
        std::tie(sample_mean, sample_var) =
            dist::Normal::GradientNLL(h_preds[i], h_preds[i + 1], h_labels[y_iter]);
        // G^{-1} * g = G_n
        auto det = 1 / (fi_0 * fi_1);
        auto fi_inv_0 = det * fi_1;
        auto fi_inv_1 = det * fi_0;
        // | fi_inv_0, 0        | \times | f | = | fi_inv_0 * f |
        // | 0       , fi_inv_1 |        | s |   | fi_inv_1 * s |
        h_gpair[i] = GradientPair{fi_inv_0 * sample_mean, 1};
        h_gpair[i + 1] = GradientPair{fi_inv_1 * sample_var, 1};
        y_iter ++;
      }
    }
  }
  void PredTransform(HostDeviceVector<float> *io_preds) override {}
  float ProbToMargin(float base_score) const override {
    return base_score;
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("mle");
  }
  void LoadConfig(Json const&) override {}

  char const* DefaultEvalMetric() const override { return "None"; }
};

XGBOOST_REGISTER_OBJECTIVE(MaximumLikelihoodEstimation, "mle")
.describe("Maximum Likelihood Estimation")
.set_body([]() { return new MaximumLikelihoodEstimation(); });

class ContinuousRankedProbabilityScore : ObjFunction {
 public:
  void Configure(Args const& args) override {}

  void GetGradient(const HostDeviceVector<bst_float> &preds,
                   const MetaInfo &info, int iteration,
                   HostDeviceVector<GradientPair> *out_gpair) override {
  }
};

}  // namespace xgboost
