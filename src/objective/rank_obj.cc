/**
 * Copyright 2019-2023 by XGBoost contributors
 */
#include "rank_obj.h"

#include <dmlc/registry.h>  // DMLC_REGISTRY_FILE_TAG

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../common/charconv.h"
#include "../common/common.h"     // AssertGPUSupport
#include "../common/linalg_op.h"  // cbegin
#include "../common/math.h"
#include "../common/ranking_utils.h"
#include "dmlc/omp.h"
#include "init_estimation.h"  // FitIntercept
#include "xgboost/base.h"     // GradientPair
#include "xgboost/data.h"     // MetaInfo
#include "xgboost/json.h"     // Json
#include "xgboost/linalg.h"
#include "xgboost/objective.h"  // ObjFunction
#include "xgboost/parameter.h"

namespace xgboost {
namespace obj {

// fixme: Maybe we name it lambda rank since we have linear model?
class LambdaMARTNDCG : public FitIntercept {
 private:
  LambdaMARTParam ndcg_param_;
  struct NDCGCache {
    std::size_t truncation{0};
    MetaInfo const* p_info;
    std::vector<double> inv_idcg;
  } h_cache_;
  std::shared_ptr<DeviceNDCGCache> d_cache_{nullptr};

 public:
  static char const* Name() { return "lambdamart:ndcg"; }
  bst_target_t Targets(MetaInfo const& info) const override {
    CHECK_LE(info.labels.Shape(1), 1) << " multi-output for LTR is not yet supported.";
    return 1;
  }
  void Configure(Args const& args) override { ndcg_param_.UpdateAllowUnknown(args); }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Name());
    out["ndcg_param"] = ToJson(ndcg_param_);
  }

  void LoadConfig(Json const& in) override {
    auto const& obj = get<Object const>(in);
    if (obj.find("ndcg_param") != obj.cend()) {
      FromJson(in["ndcg_param"], &ndcg_param_);
    } else {
      // Being compatible with XGBoost version < 1.6.
      auto const& j_parameter = get<Object const>(obj.at("lambda_rank_param"));
      ndcg_param_.lambdamart_truncation =
          std::stol(get<String const>(j_parameter.at("num_pairsample")));
    }
  }

  void CalcLambdaForGroup(common::Span<float const> predt, linalg::VectorView<float const> label,
                          common::Span<GradientPair> gpair, MetaInfo const& info,
                          bst_group_t query_id) {
    auto cnt = info.group_ptr_.at(query_id + 1) - info.group_ptr_.at(query_id);
    std::fill(gpair.begin(), gpair.end(), GradientPair{});
    const double inv_IDCG = h_cache_.inv_idcg[query_id];
    auto sorted_idx = common::ArgSort<size_t>(predt, std::greater<>{});
    for (size_t i = 0; i < cnt - 1 && i < ndcg_param_.lambdamart_truncation; ++i) {
      for (size_t j = i + 1; j < cnt; ++j) {
        if (label(sorted_idx[i]) == label(sorted_idx[j])) {
          continue;
        }
        LambdaNDCG(label, predt, sorted_idx, i, j, inv_IDCG, gpair);
      }
    }
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds, const MetaInfo& info,
                   std::int32_t iter, HostDeviceVector<GradientPair>* out_gpair) override {
    if (iter == 0) {
      CHECK_EQ(info.labels.Size(), preds.Size());
      CHECK(!info.group_ptr_.empty());
    }
    if (ctx_->IsCUDA()) {
      LambdaMARTGetGradientNDCGGPUKernel(ctx_, preds, info, ndcg_param_.lambdamart_truncation,
                                         &d_cache_, out_gpair);
      return;
    }

    bst_group_t n_groups = info.group_ptr_.size() - 1;
    out_gpair->Resize(info.num_row_);
    auto h_gpair = out_gpair->HostSpan();
    auto h_predt = preds.ConstHostSpan();
    auto h_label = info.labels.HostView();
    auto h_weight = info.weights_.ConstHostSpan();
    auto make_range = [&](bst_group_t g) {
      return linalg::Range(info.group_ptr_[g], info.group_ptr_[g + 1]);
    };

    if (h_cache_.p_info != &info || h_cache_.truncation != ndcg_param_.lambdamart_truncation) {
      h_cache_.inv_idcg.clear();
      h_cache_.inv_idcg.resize(n_groups, std::numeric_limits<float>::quiet_NaN());
      CheckNDCGLabelsCPUKernel(ndcg_param_, h_label.Values());
      common::ParallelFor(n_groups, ctx_->Threads(), common::Sched::Guided(), [&](auto g) {
        auto label = h_label.Slice(make_range(g), 0);
        linalg::Vector<float> sorted_labels;
        sorted_labels.Reshape(label.Size());
        auto h_sorted_labels = sorted_labels.HostView();
        std::copy(linalg::cbegin(label), linalg::cend(label), linalg::begin(h_sorted_labels));
        auto s_sorted_labels = h_sorted_labels.Values();
        if (omp_in_parallel()) {
          std::stable_sort(s_sorted_labels.data(), s_sorted_labels.data() + s_sorted_labels.size(),
                           std::greater<>{});
        } else {
          XGBOOST_PARALLEL_STABLE_SORT(s_sorted_labels.data(),
                                       s_sorted_labels.data() + s_sorted_labels.size(),
                                       std::greater<>{});
        }
        double inv_IDCG = CalcInvIDCG(h_sorted_labels, ndcg_param_.lambdamart_truncation);
        h_cache_.inv_idcg[g] = inv_IDCG;
      });
      h_cache_.p_info = &info;
      h_cache_.truncation = ndcg_param_.lambdamart_truncation;
    }
    // fixme: calculate n_threads for each group.
    common::ParallelFor(n_groups, ctx_->Threads(), [&](auto g) {
      size_t cnt = info.group_ptr_.at(g + 1) - info.group_ptr_[g];
      auto predts = h_predt.subspan(info.group_ptr_[g], cnt);
      auto gpairs = h_gpair.subspan(info.group_ptr_[g], cnt);
      auto labels = h_label.Slice(make_range(g), 0);
      this->CalcLambdaForGroup(predts, labels, gpairs, info, g);

      if (!h_weight.empty()) {
        CHECK_EQ(h_weight.size(), info.group_ptr_.size() - 1);
        std::transform(gpairs.begin(), gpairs.end(), gpairs.begin(),
                       [&](GradientPair const& gpair) { return gpair * h_weight[g]; });
      }
    });
  }

  const char* DefaultEvalMetric() const override { return "ndcg-loss"; }
  const char* DefaultEvalMetric(Json* config) const override {
    this->SaveConfig(config);
    (*config)["name"] = String{this->DefaultEvalMetric()};
    return this->DefaultEvalMetric();
  }

  ObjInfo Task() const override { return ObjInfo{ObjInfo::kRanking}; }
};

class LambdaMARTPairwise : public ObjFunction {};

class LambdaMARTMaps : public ObjFunction {};

void CheckNDCGLabelsCPUKernel(LambdaMARTParam const& p, common::Span<float const> labels) {
  auto label_is_integer =
      std::none_of(labels.data(), labels.data() + labels.size(), [](auto const& v) {
        auto l = std::floor(v);
        return std::fabs(l - v) > kRtEps || v < 0.0f;
      });
  CHECK(label_is_integer) << "When using relevance degree as target, labels "
                             "must be either 0 or positive integer.";
  if (p.lambdamart_exp_gain) {
    auto label_is_valid = std::none_of(labels.data(), labels.data() + labels.size(),
                                       [](ltr::rel_degree_t v) { return v >= 32; });
    CHECK(label_is_valid)
        << "Relevance degress must be smaller than 32 when the exponential gain function is used.";
  }
}

#if !defined(XGBOOST_USE_CUDA)
void LambdaMARTGetGradientNDCGGPUKernel(Context const*, const HostDeviceVector<bst_float>&,
                                        const MetaInfo&, size_t, std::shared_ptr<DeviceNDCGCache>*,
                                        HostDeviceVector<GradientPair>*) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)

auto GetParamInfo(std::string const& name) {
  return LambdaMARTParam::__MANAGER__()->Find(name)->GetFieldInfo();
}

XGBOOST_REGISTER_OBJECTIVE(LambdaMARTNDCG, LambdaMARTNDCG::Name())
    .describe("LambdaMART with NDCG as objective")
    .add_argument("unbiased", GetParamInfo("lambdamart_unbiased").type,
                  GetParamInfo("lambdamart_unbiased").description)
    .add_argument("truncation", GetParamInfo("lambdamart_truncation").type,
                  GetParamInfo("lambdamart_truncation").description)
    .add_argument("exp_gain", GetParamInfo("lambdamart_exp_gain").type,
                  GetParamInfo("lambdamart_exp_gain").description)
    .set_body([]() { return new LambdaMARTNDCG(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaMARTNDCG_obsolated, "rank:ndcg")
    .describe("LambdaMART with NDCG as objective")
    .set_body([]() { return new LambdaMARTNDCG(); });

DMLC_REGISTRY_FILE_TAG(rank_obj);
}  // namespace obj
}  // namespace xgboost
