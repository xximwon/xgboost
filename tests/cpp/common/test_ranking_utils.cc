/**
 * Copyright 2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // Context

#include "../../../src/common/ranking_utils.h"

namespace xgboost {
TEST(RankingUtils, IDCG) {
  linalg::Vector<float> scores{{2, 2, 1, 0}, {4}, Context::kCpuId};
  auto h_scores = scores.HostView();
  float IDCG = CalcInvIDCG(h_scores, h_scores.Size());
  ASSERT_FLOAT_EQ(IDCG, 1.0f / 5.39279f);
  // float ndcg = CalcNDCGAtK(h_scores, h_scores, scores.Size());
  // ASSERT_EQ(ndcg, 1);
}
}  // namespace xgboost
