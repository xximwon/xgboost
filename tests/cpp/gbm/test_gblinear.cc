/**
 * Copyright 2019-2023, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <memory>
#include <sstream>

#include "../helpers.h"
#include "xgboost/context.h"
#include "xgboost/gbm.h"
#include "xgboost/json.h"
#include "xgboost/learner.h"
#include "xgboost/logging.h"

namespace xgboost::gbm {

TEST(GBLinear, JsonIO) {
  size_t constexpr kRows = 16, kCols = 16;

  Context ctx;
  LearnerModelParam mparam{MakeMP(kCols, .5, 1, &ctx)};

  std::unique_ptr<GradientBooster> gbm{
      CreateTrainedGBM("gblinear", Args{}, kRows, kCols, &mparam, &ctx)};
  Json model { Object() };
  gbm->SaveModel(&model);
  ASSERT_TRUE(IsA<Object>(model));

  std::string model_str;
  Json::Dump(model, &model_str);

  model = Json::Load(StringView{model_str.c_str(), model_str.size()});
  ASSERT_TRUE(IsA<Object>(model));

  {
    model = model["model"];
    auto weights = get<Array>(model["weights"]);
    ASSERT_EQ(weights.size(), 17);
  }
}
}  // namespace xgboost::gbm
