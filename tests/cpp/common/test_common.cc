#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/generic_parameters.h>
#include "../../../src/common/common.h"
#include "../helpers.h"

namespace xgboost {
namespace common {
TEST(ArgSort, Basic) {
  std::vector<float> inputs {3.0, 2.0, 1.0};
  auto ret = ArgSort<bst_feature_t>(inputs);
  std::vector<bst_feature_t> sol{2, 1, 0};
  ASSERT_EQ(ret, sol);
}

TEST(StridedCopy, Basic) {
  auto g = GenerateRandomGradients(100);
  ASSERT_EQ(g.DeviceIdx(), GenericParameter::kCpuId);
  HostDeviceVector<GradientPair> out(100);

  StridedCopy(&out, g, 1, 1);

  HostDeviceVector<GradientPair> solution(100);
  solution.Copy(g);

  ASSERT_EQ(solution.HostVector(), g.HostVector());
}
}  // namespace common
}  // namespace xgboost
