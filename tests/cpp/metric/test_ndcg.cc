/**
 * Copyright 2016-2023 by XGBoost Contributors
 */
#include <xgboost/metric.h>  // Metric
#include <gtest/gtest.h>
#include <memory>  // unique_ptr
#include "../helpers.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace metric {
TEST(Metric, DeclareUnifiedTest(NDCG)) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("ndcg", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}));
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.6509f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.3868f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@-", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.6509f, 0.001f);
  delete metric;
  metric = xgboost::Metric::Create("ndcg-", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.6509f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2-", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg@2-");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.3868f, 0.001f);

  delete metric;
}

TEST(Metric, DeclareUnifiedTest(NDCGBasic)) {
  Context ctx = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<Metric> metric{Metric::Create("ndcg", &ctx)};
  metric->Configure(Args{});
  MetaInfo info;
  info.labels = linalg::Tensor<float, 2>{{10.0f, 0.0f, 0.0f, 1.0f, 5.0f}, {5}, ctx.gpu_id};
  info.num_row_ = info.labels.Shape(0);
  info.group_ptr_.resize(2);
  info.group_ptr_[0] = 0;
  info.group_ptr_[1] = info.num_row_;
  HostDeviceVector<float> predt{{0.1, 0.2, 0.3, 4, 70}};
  auto ndcg = metric->Eval(predt, info);
  std::cout << "ndcg:" << ndcg << std::endl;
  // 0.6956940443813076 or 0.409738, depending on the gain type.
  predt.HostVector() = info.labels.Data()->HostVector();
  ndcg = metric->Eval(predt, info);
  std::cout << "ndcg:" << ndcg << std::endl;
}
}  // namespace metric
}  // namespace xgboost
