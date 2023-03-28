/**
 * Copyright 2015-2023, XGBoost Contributors
 * \file multi_class.cc
 * \brief Definition of multi-class classification objectives.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "../common/common.h"
#include "../common/linalg_op.h"
#if defined(XGBOOST_USE_CUDA)
#include "../common/linalg_op.cuh"
#endif  // defined(XGBOOST_USE_CUDA)
#include "../common/math.h"
#include "../common/optional_weight.h"
#include "../common/transform.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/linalg.h"
#include "xgboost/logging.h"
#include "xgboost/objective.h"
#include "xgboost/parameter.h"

namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(multiclass_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

struct SoftmaxMultiClassParam : public XGBoostParameter<SoftmaxMultiClassParam> {
  int num_class;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1).describe(
        "Number of output class in the multi-class classification.");
  }
};

class SoftmaxMultiClassObj : public ObjFunction {
 public:
  explicit SoftmaxMultiClassObj(bool output_prob) : output_prob_(output_prob) {}

  void Configure(Args const& args) override { param_.UpdateAllowUnknown(args); }

  ObjInfo Task() const override { return ObjInfo::kClassification; }
  void GetGradient(const HostDeviceVector<float>& predt, MetaInfo const& info, int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {}

  void GetGradient(const HostDeviceVector<float>& predt, MetaInfo const& info, int iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    auto const n_classes = param_.num_class;
    if (iter == 0) {
      auto h_label = info.labels.HostView();
      auto valid = std::all_of(linalg::cbegin(h_label), linalg::cend(h_label),
                               [n_classes](auto v) { return v >= 0 && v < n_classes; });
      CHECK(valid) << "label must be in [0, num_class).";
    }

    if (info.labels.Size() == 0) {
      return;
    }
    CHECK(predt.Size() == (static_cast<size_t>(param_.num_class) * info.labels.Size()))
        << "SoftmaxMultiClassObj: label size and pred size does not match.\n"
        << "label.Size() * num_class: "
        << info.labels.Size() * static_cast<size_t>(param_.num_class) << "\n"
        << "num_class: " << param_.num_class << "\n"
        << "preds.Size(): " << predt.Size();

    const auto ndata = static_cast<std::uint64_t>(predt.Size() / n_classes);

    auto device = ctx_->gpu_id;
    out_gpair->SetDevice(device);
    info.labels.SetDevice(device);
    info.weights_.SetDevice(device);
    predt.SetDevice(device);

    if (out_gpair->Ord() != linalg::kF) {
      // make sure it's not repeatly initialized.
      CHECK_EQ(out_gpair->Size(), 0);
      *out_gpair = linalg::MakeTensor<GradientPair>(ctx_, linalg::kF, info.num_row_, n_classes);
    }
    out_gpair->Reshape(info.num_row_, n_classes);

    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }

    auto h_predt = predt.ConstHostSpan();
    auto h_label = info.labels.HostView();
    auto h_weight = common::MakeOptionalWeights(ctx_, info.weights_);
    auto h_gpair = out_gpair->HostView();
    linalg::ElementWiseKernel(ctx_, h_label, [=] XGBOOST_DEVICE(std::size_t idx, float y) mutable {
      common::Span<float const> point = h_predt.subspan(idx * n_classes, n_classes);
      float wmax = std::numeric_limits<float>::min();
      for (auto const i : point) {
        wmax = fmaxf(i, wmax);
      }
      double wsum = 0.0f;
      for (auto const i : point) {
        wsum += expf(i - wmax);
      }
      float wt = h_weight[idx];

      for (std::int32_t k = 0; k < n_classes; ++k) {
        // Computation duplicated to avoid creating a cache.
        float p = expf(point[k] - wmax) / static_cast<float>(wsum);
        const float eps = 1e-16f;
        const float h = fmax(2.0f * p * (1.0f - p) * wt, eps);
        p = y == k ? p - 1.0f : p;
        h_gpair(idx, k) = GradientPair(p * wt, h);
      }
    });
  }
  void PredTransform(HostDeviceVector<float>* io_preds) const override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(HostDeviceVector<float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override { return "mlogloss"; }

  inline void Transform(HostDeviceVector<float>* io_preds, bool prob) const {
    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(io_preds->Size() / nclass);

    auto device = io_preds->DeviceIdx();
    if (prob) {
      common::Transform<>::Init(
          [=] XGBOOST_DEVICE(size_t _idx, common::Span<float> _preds) {
            common::Span<float> point = _preds.subspan(_idx * nclass, nclass);
            common::Softmax(point.begin(), point.end());
          },
          common::Range{0, ndata}, this->ctx_->Threads(), device)
          .Eval(io_preds);
    } else {
      io_preds->SetDevice(device);
      HostDeviceVector<float> max_preds;
      max_preds.SetDevice(device);
      max_preds.Resize(ndata);
      common::Transform<>::Init(
          [=] XGBOOST_DEVICE(size_t _idx, common::Span<const float> _preds,
                             common::Span<float> _max_preds) {
            common::Span<const float> point = _preds.subspan(_idx * nclass, nclass);
            _max_preds[_idx] = common::FindMaxIndex(point.cbegin(), point.cend()) - point.cbegin();
          },
          common::Range{0, ndata}, this->ctx_->Threads(), device)
          .Eval(io_preds, &max_preds);
      io_preds->Resize(max_preds.Size());
      io_preds->Copy(max_preds);
    }
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    if (this->output_prob_) {
      out["name"] = String("multi:softprob");
    } else {
      out["name"] = String("multi:softmax");
    }
    out["softmax_multiclass_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override { FromJson(in["softmax_multiclass_param"], &param_); }

 private:
  // output probability
  bool output_prob_;
  // parameter
  SoftmaxMultiClassParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(SoftmaxMultiClassParam);

XGBOOST_REGISTER_OBJECTIVE(SoftmaxMultiClass, "multi:softmax")
    .describe("Softmax for multi-class classification, output class index.")
    .set_body([]() { return new SoftmaxMultiClassObj(false); });

XGBOOST_REGISTER_OBJECTIVE(SoftprobMultiClass, "multi:softprob")
    .describe("Softmax for multi-class classification, output probability distribution.")
    .set_body([]() { return new SoftmaxMultiClassObj(true); });

}  // namespace obj
}  // namespace xgboost
