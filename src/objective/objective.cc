/*!
 * Copyright 2015 by Contributors
 * \file objective.cc
 * \brief Registry of all objective functions.
 */
#include <dmlc/registry.h>
#include <xgboost/objective.h>

#include <sstream>

#include "init_estimation.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/learner.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::ObjFunctionReg);
}  // namespace dmlc

namespace xgboost {
// implement factory functions
ObjFunction* ObjFunction::Create(Context const* ctx, const std::string& name) {
  auto* e = ::dmlc::Registry< ::xgboost::ObjFunctionReg>::Get()->Find(name);
  if (e == nullptr) {
    std::stringstream ss;
    for (const auto& entry : ::dmlc::Registry< ::xgboost::ObjFunctionReg>::List()) {
      ss << "Objective candidate: " << entry->name << "\n";
    }
    LOG(FATAL) << "Unknown objective function: `" << name << "`\n" << ss.str();
  }
  auto pobj = (e->body)();
  pobj->ctx_ = ctx;
  return pobj;
}

float ObjFunction::InitEstimation(MetaInfo const& info, LearnerModelParam const* model,
                                  HostDeviceVector<float>* out_predt) const {
  obj::InitialEstimationRegression::Constant(ctx_, info, model, out_predt);
  return model->base_score;
}
}  // namespace xgboost

namespace xgboost {
namespace obj {
// List of files that will be force linked in static links.
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(regression_obj_gpu);
DMLC_REGISTRY_LINK_TAG(hinge_obj_gpu);
DMLC_REGISTRY_LINK_TAG(multiclass_obj_gpu);
DMLC_REGISTRY_LINK_TAG(rank_obj_gpu);
#else
DMLC_REGISTRY_LINK_TAG(regression_obj);
DMLC_REGISTRY_LINK_TAG(hinge_obj);
DMLC_REGISTRY_LINK_TAG(multiclass_obj);
DMLC_REGISTRY_LINK_TAG(rank_obj);
#endif  // XGBOOST_USE_CUDA
}  // namespace obj
}  // namespace xgboost
