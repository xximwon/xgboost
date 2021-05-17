#include <gtest/gtest.h>

#include "../../../src/common/algorithm.cuh"
#include "../../../src/common/device_helpers.cuh"

void TestSegmentedInclusiveScan() {
  size_t n_elem = 100, n_segments = 3;
  dh::device_vector<float> data(n_elem, 0);
  thrust::sequence(data.begin(), data.end(), 0);

  dh::device_vector<size_t> segments(n_segments + 1, 0);
  segments[1] = 10;
  segments[2] = 66;
  segments[3] = 100;

  dh::device_vector<float> out(n_elem, 0);

  dh::SegmentedInclusiveScan(dh::ToSpan(segments), data.begin(), data.end(),
                             out.begin(),
                             [] __device__(auto l, size_t l_seg, auto r,
                                           size_t r_seg) { return l + r; });

  auto s_0 = (0 + 9) * 10 / 2;
  auto s_1 = (10 + 65) * (65 - 10 + 1) / 2;
  auto s_2 = (66 + 99) * (99 - 66 + 1) / 2;

  ASSERT_EQ(out[0], 0);
  ASSERT_EQ(out[10], 10);
  ASSERT_EQ(out[66], 66);

  ASSERT_EQ(out[9], s_0);
  ASSERT_EQ(out[65], s_1);
  ASSERT_EQ(out[99], s_2);
}

TEST(GPUAlgorithm, SegmentedInclusiveScan) {
  TestSegmentedInclusiveScan();
}
