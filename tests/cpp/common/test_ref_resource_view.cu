/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/equal.h>                       // for equal
#include <thrust/fill.h>                        // for fill_n
#include <thrust/iterator/constant_iterator.h>  // for make_constant_iterator

#include "../../../src/common/device_helpers.cuh"  // for CachingThrustPolicy
#include "../../../src/common/ref_resource_view.cuh"

namespace xgboost::common {
TEST(RefResourceView, CudaGrowOnly) {
  auto ref = MakeCudaGrowOnly<double>(1024);
  ASSERT_EQ(ref.size_bytes(), 1024 * sizeof(double));
  auto res = std::dynamic_pointer_cast<common::CudaGrowOnlyResource>(ref.Resource());
  CHECK(res);
  res->Resize(2048);

  auto ref1 = RefResourceView<double>(res->DataAs<double>(), res->Size(), ref.Resource());
  ASSERT_EQ(ref1.size(), 2048);
  ASSERT_EQ(res->Size(), 2048);
  thrust::fill_n(dh::CachingThrustPolicy(), ref1.data(), ref1.size(), 1.0);
  ASSERT_TRUE(thrust::equal(dh::CachingThrustPolicy(), ref1.cbegin(), ref.cend(),
                            thrust::make_constant_iterator(1.0)));
}
}  // namespace xgboost::common
