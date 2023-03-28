#include <memory>
#include <vector>

#include "../common/linalg_op.h"  // for begin
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"  // for ObjFunction

namespace xgboost::obj {
/**
 * \brief A special objective function for multi-task.
 */
class ObjFunctionContainer : public ObjFunction {
  std::vector<std::unique_ptr<ObjFunction>> mem_;

 public:
  void Configure(Args const& args) override {
    CHECK(!mem_.empty());
    for (auto const& obj : mem_) {
      obj->Configure(args);
    }
  }
  void SaveConfig(Json* p_out) const override {
    CHECK(!mem_.empty());
    std::vector<Json> config(mem_.size(), Json{Object{}});
    for (std::size_t i = 0; i < mem_.size(); ++i) {
      mem_[i]->SaveConfig(&config[i]);
    }
    *p_out = Array{config};
  }
  void LoadConfig(Json const& in) override {
    CHECK(!mem_.empty());
    auto const& config = get<Array const>(in);
    for (std::size_t i = 0; i < mem_.size(); ++i) {
      mem_[i]->LoadConfig(config[i]);
    }
  }
  bst_target_t Targets(MetaInfo const& info) const override {
    CHECK(!mem_.empty());
    bst_target_t n_targets{0};
    // fixme: we should pass a subset of info
    for (auto const& obj : mem_) {
      n_targets += obj->Targets(info);
    }
    return n_targets;
  }
  const char* DefaultEvalMetric() const override { return mem_.front()->DefaultEvalMetric(); }
  ObjInfo Task() const override { return mem_.front()->Task(); }

  void GetGradient(HostDeviceVector<float> const& predt, const MetaInfo& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) {
    // fixme: turn it into a vector of matrices.
    std::vector<HostDeviceVector<GradientPair>> gpair(mem_.size());
    for (std::size_t i = 0; i < mem_.size(); ++i) {
      mem_[i]->GetGradient(predt, info, iter, &gpair[i]);
    }
    out_gpair->Reshape(info.num_row_, this->Targets(info));
    auto h_out_gpair = out_gpair->HostView();
    // fixme: f-order is necessary
    bst_target_t processed_targets{0};
    for (std::size_t i = 0; i < mem_.size(); ++i) {
      // fixme: subset of info
      auto n_targets = mem_[i]->Targets(info);
      auto const& h_vec = gpair[i].ConstHostVector();
      auto t_gpair = h_out_gpair.Slice(linalg::All(), linalg::Range(processed_targets, n_targets));
      std::copy(h_vec.cbegin(), h_vec.cend(), linalg::begin(t_gpair));
      processed_targets += n_targets;
    }
  }
};
}  // namespace xgboost::obj
