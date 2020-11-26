/**
 * Copyright 2023 by XGBoost Contributors
 */
#include "ndcg.h"

#include <algorithm>                     // std::min
#include <cstddef>                       // std::size_t
#include <cstdint>                       // std::uint32_t
#include <functional>                    // std::greater
#include <memory>                        // std::unique_ptr
#include <string>                        // std::to_string
#include <vector>                        // std::vector

#include "../common/algorithm.h"         // ArgSort
#include "../common/common.h"            // OptionalWeights
#include "../common/linalg_op.h"         // cbegin,cend
#include "../common/optional_weight.h"   // OptionalWeights
#include "../common/ranking_utils.h"     // LambdaMARTParam
#include "../common/threading_utils.h"   // ParallelFor
#include "xgboost/base.h"                // Args,bst_group_t
#include "xgboost/context.h"             // Context
#include "xgboost/data.h"                // MetaInfo
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/json.h"                // Json,String,ToJson,FromJson,get
#include "xgboost/linalg.h"              // MakeTensorView,Vector
#include "xgboost/metric.h"              // Metric,XGBOOST_REGISTER_METRIC
#include "xgboost/span.h"                // Span

namespace xgboost {
namespace ltr {
inline double CalcDCGDiscount(rel_degree_t rel) {
  auto d = 1.0 / std::log2(static_cast<double>(rel) + 2.0);
  return d;
}

class NDCGCache {
  HostDeviceVector<double> discounts_;
  linalg::Vector<double> inv_idcg_cache_;

 public:
  explicit NDCGCache(Context const* ctx, MetaInfo const& info, std::size_t k) {
    CHECK_GE(info.group_ptr_.size(), 2);

    bst_group_t max_group_size{0};
    for (std::size_t i = 1; i < info.group_ptr_.size(); ++i) {
      auto n = info.group_ptr_[i] - info.group_ptr_[i - 1];
      max_group_size = std::max(max_group_size, n);
    }
    discounts_.Resize(max_group_size, 0);
    auto& h_discounts = discounts_.HostVector();
    for (std::size_t i = 0; i < max_group_size; ++i) {
      h_discounts[i] = CalcDCGDiscount(i);
    }

    auto n_groups = info.group_ptr_.size() - 1;
    auto h_labels = info.labels.HostView();

    inv_idcg_cache_.Reshape(n_groups);
    auto h_inv_idcg = inv_idcg_cache_.HostView();

    common::ParallelFor(n_groups, ctx->Threads(), [&](auto g) {
      auto g_labels = h_labels.Slice(linalg::Range(info.group_ptr_[g], info.group_ptr_[g + 1]), 0);
      auto sorted_idx = common::ArgSort<std::size_t>(ctx, linalg::cbegin(g_labels),
                                                     linalg::cend(g_labels), std::greater<>{});

      double idcg{0.0};
      for (std::size_t i = 0; i < std::min(g_labels.Size(), k); ++i) {
        // idcg += h_discounts[i] * CalcNDCGGain(g_labels(sorted_idx[i]));
        idcg += h_discounts[i] * g_labels(sorted_idx[i]);
      }
      auto inv_idcg = (idcg == 0.0 ? 0.0 : (1.0 / idcg));
      h_inv_idcg(g) = inv_idcg;
    });
  }

  linalg::VectorView<double const> InvIDCG(Context const* ctx) const {
    return inv_idcg_cache_.View(ctx->gpu_id);
  }
  common::Span<double const> Discounts(Context const* ctx) const {
    return ctx->IsCPU() ? discounts_.ConstHostSpan() : discounts_.ConstDeviceSpan();
  }
};
}  // namespace ltr

namespace metric {
/**
 * \brief Implement the NDCG score function for learning to rank.
 *
 *     Ties are ignored.
 */
template <bool neg>
class EvalNDCGScore : public Metric {
  // fixme: how do we decouple the evaluation parameter with the objective parameter?
  // we may use only 1 training parameter for obj, but can hvae multiple evaluation metrics.
  LambdaMARTParam ndcg_param_;
  std::map<MetaInfo const*, std::unique_ptr<ltr::NDCGCache>> ndcg_cache_;

 public:
  void Configure(Args const& args) override { ndcg_param_.UpdateAllowUnknown(args); }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String{this->Name()};
    out["ndcg_param"] = ToJson(ndcg_param_);
  }
  void LoadConfig(Json const& in) override {
    CHECK_EQ(this->Name(), get<String const>(in["name"]));
    FromJson(in["ndcg_param"], &ndcg_param_);
  }
  const char* Name() const override { return neg ? "ndcg-loss" : "ndcg-score"; }

  double Eval(const HostDeviceVector<float>& preds, MetaInfo const& info) override {
    // todos:
    // - extract log scale utilities
    // - ties
    // - cache discounts
    // - cache idcg
    // - truncation, dcg and idcg
    // - GPU impl
    //   + document non-deterministic
    // - tests
    // - compatibility with old metrics
    if (ctx_->IsCUDA()) {
      return cuda_impl::NDCGScore(ctx_, info, preds, ndcg_param_.lambdamart_truncation);
    }
    bst_group_t n_groups = info.group_ptr_.size() - 1;
    if (ndcg_cache_.find(&info) == ndcg_cache_.cend()) {
      ndcg_cache_[&info] =
          std::make_unique<ltr::NDCGCache>(ctx_, info, ndcg_param_.lambdamart_truncation);
    }

    // group local ndcg
    std::vector<double> ndcg_gloc(n_groups, 0.0);
    std::vector<double> weight_gloc(n_groups, 0.0);

    auto h_inv_idcg = ndcg_cache_.at(&info)->InvIDCG(ctx_);
    auto h_discounts = ndcg_cache_.at(&info)->Discounts(ctx_);

    auto h_labels = info.labels.HostView();
    auto h_predt =
        linalg::MakeTensorView(ctx_->IsCPU() ? preds.ConstHostSpan() : preds.ConstDeviceSpan(),
                               {preds.Size()}, ctx_->gpu_id);
    auto weight = common::MakeOptionalWeights(ctx_, info.weights_);

    common::ParallelFor(n_groups, ctx_->Threads(), [&](auto g) {
      auto g_predt = h_predt.Slice(linalg::Range(info.group_ptr_[g], info.group_ptr_[g + 1]));
      auto g_labels = h_labels.Slice(linalg::Range(info.group_ptr_[g], info.group_ptr_[g + 1]), 0);
      auto sorted_idx = common::ArgSort<std::size_t>(ctx_, linalg::cbegin(g_predt),
                                                     linalg::cend(g_predt), std::greater<>{});
      double dcg{.0};
      for (std::size_t i = 0; i < std::min(g_labels.Size(), ndcg_param_.lambdamart_truncation);
           ++i) {
        // dcg += h_discounts[i] * CalcNDCGGain(g_labels(sorted_idx[i]));
        dcg += h_discounts[i] * g_labels(sorted_idx[i]);
      }

      weight_gloc[g] = weight[g];
      ndcg_gloc[g] = h_inv_idcg(g) * dcg * weight[g];
    });
    auto sum_w = std::accumulate(weight_gloc.cbegin(), weight_gloc.cend(), 0.0);
    auto ndcg = std::accumulate(ndcg_gloc.cbegin(), ndcg_gloc.cend(), 0.0) / sum_w;
    CHECK_LE(ndcg, 1.0 + kRtEps);
    ndcg = std::min(1.0, ndcg);
    if (neg) {
      return 1.0 - ndcg;
    }
    return ndcg;
  }
};

XGBOOST_REGISTER_METRIC(NDCGScore, "ndcg-score")
    .describe("ndcg@k for rank.")
    .set_body([](const char*) { return new EvalNDCGScore<false>{}; });
XGBOOST_REGISTER_METRIC(NDCGLoss, "ndcg-loss")
    .describe("ndcg@k for rank.")
    .set_body([](const char*) { return new EvalNDCGScore<true>{}; });

XGBOOST_REGISTER_METRIC(NDCG, "ndcg").describe("ndcg@k for rank.").set_body([](const char* param) {
  Metric* m;
  std::uint32_t topn;
  if (param != nullptr) {
    if (param[strlen(param) - 1] == '-') {
      m = new EvalNDCGScore<true>{};
    } else {
      m = new EvalNDCGScore<false>{};
    }

    std::ostringstream os;
    if (sscanf(param, "%u[-]?", &topn) == 1) {
      m->Configure(Args{{"lambdamart_truncation", std::to_string(topn)}});
    }
  } else {
    m = new EvalNDCGScore<false>{};
  }
  CHECK(m);
  return m;
});
}  // namespace metric
}  // namespace xgboost
