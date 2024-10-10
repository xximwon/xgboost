/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/equal.h>                       // for equal
#include <thrust/fill.h>                        // for fill_n
#include <thrust/iterator/constant_iterator.h>  // for make_constant_iterator
#include <thrust/sequence.h>                    // for sequence

#include "../../../src/common/ref_resource_view.cuh"
#include "../helpers.h"  // for MakeCUDACtx

namespace xgboost::common {
class TestCudaGrowOnly : public ::testing::TestWithParam<std::size_t> {
 public:
  void TestGrow(std::size_t m, std::size_t n) {
    auto ctx = MakeCUDACtx(0);
    ctx.CUDACtx()->Stream().Sync();

    auto ref = MakeCudaGrowOnly<double>(m);
    ASSERT_EQ(ref.size_bytes(), m * sizeof(double));
    thrust::sequence(ctx.CUDACtx()->CTP(), ref.begin(), ref.end(), 0.0);
    auto res = std::dynamic_pointer_cast<common::CudaGrowOnlyResource>(ref.Resource());
    CHECK(res);
    res->Resize(n);

    auto ref1 = RefResourceView<double>(res->DataAs<double>(), res->Size(), ref.Resource());
    ASSERT_EQ(res->Size(), n);
    ASSERT_EQ(ref1.size(), n);
    // thrust::sequence(ctx.CUDACtx()->CTP(), ref1.begin() + m, ref1.end(), static_cast<double>(m));
    std::cout << "dis:" << std::distance(ref1.begin(), ref1.end()) << std::endl;
    thrust::sequence(ctx.CUDACtx()->CTP(), ref1.begin(), ref1.end(), static_cast<double>(0.0));
    ctx.CUDACtx()->Stream().Sync();
    std::vector<double> h_vec(ref1.size());
    dh::safe_cuda(cudaMemcpyAsync(h_vec.data(), ref1.data(), ref1.size(), cudaMemcpyDefault));
    ctx.CUDACtx()->Stream().Sync();
    for (std::size_t i = 0; i < h_vec.size(); ++i) {
      ASSERT_EQ(h_vec[i], i) << "i:" << i << " m:" << m << std::endl;
    }
  }

  void Run(std::size_t n) {
    this->TestGrow(1024, n);
    // this->TestGrow(n, 2 * n);
    // this->TestGrow(1024, 2 * n);
  }
};

TEST_P(TestCudaGrowOnly, Resize) { this->Run(this->GetParam()); }

INSTANTIATE_TEST_SUITE_P(RefResourceView, TestCudaGrowOnly, ::testing::Values(1  << 20));
}  // namespace xgboost::common
