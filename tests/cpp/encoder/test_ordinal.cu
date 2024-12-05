#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include "../../src/encoder/ordinal.cuh"

namespace enc::cuda_impl {
TEST(GpuCategoricalEncoder, Str) {
  CudaCategoryRecoder encoder;

  thrust::device_vector<std::int8_t> orig_names{'c', 'b', 'd', 'a'};
  thrust::device_vector<std::int32_t> orig_offsets{0, 1, 2, 3, 4};
  std::vector<enc::cuda_impl::TupToVarT<enc::CatIndexViewTypes>> h_orig_columns;
  thrust::device_vector<enc::cuda_impl::TupToVarT<enc::CatIndexViewTypes>> orig_columns(1);
  thrust::device_vector<std::int32_t> orig_feature_segments{0, 4};

  DeviceColumnsView orig_dict;
  h_orig_columns.emplace_back(CatStrArrayView{dh::ToSpan(orig_offsets), dh::ToSpan(orig_names)});
  thrust::copy(h_orig_columns.begin(), h_orig_columns.end(), orig_columns.begin());
  orig_dict.columns = dh::ToSpan(orig_columns);
  orig_dict.feature_segments = dh::ToSpan(orig_feature_segments);
  orig_dict.n_total_cats = 4;

  // New one
  thrust::device_vector<std::int8_t> new_names{'c', 'a', 'b'};
  thrust::device_vector<std::int32_t> new_offsets{0, 1, 2, 3};
  std::vector<enc::cuda_impl::TupToVarT<enc::CatIndexViewTypes>> h_new_columns;
  thrust::device_vector<enc::cuda_impl::TupToVarT<enc::CatIndexViewTypes>> new_columns(1);
  thrust::device_vector<std::int32_t> new_feature_segments{0, 3};

  DeviceColumnsView new_dict;
  h_new_columns.emplace_back(CatStrArrayView{dh::ToSpan(new_offsets), dh::ToSpan(new_names)});
  thrust::copy(h_new_columns.begin(), h_new_columns.end(), new_columns.begin());
  new_dict.columns = dh::ToSpan(new_columns);
  new_dict.feature_segments = dh::ToSpan(new_feature_segments);
  new_dict.n_total_cats = 3;

  auto mapping = encoder.Recode(orig_dict, new_dict);
  ASSERT_EQ(mapping.size(), 3);

  std::vector<std::int32_t> h_mapping(mapping.size());
  thrust::copy(mapping.cbegin(), mapping.cend(), h_mapping.begin());
  auto sol = std::remove_reference_t<decltype(h_mapping)>{0, 3, 1};
  ASSERT_EQ(h_mapping, sol);

  {
    thrust::device_vector<std::int8_t> new_names{'c', 'a', 'e'};
    thrust::device_vector<std::int32_t> new_offsets{0, 1, 2, 3};
    thrust::device_vector<std::int32_t> new_feature_segments{0, 3};

    DeviceColumnsView new_dict;
    std::vector<enc::cuda_impl::TupToVarT<enc::CatIndexViewTypes>> h_new_columns;
    h_new_columns.emplace_back(CatStrArrayView{dh::ToSpan(new_offsets), dh::ToSpan(new_names)});
    thrust::device_vector<enc::cuda_impl::TupToVarT<enc::CatIndexViewTypes>> new_columns(1);
    thrust::copy(h_new_columns.begin(), h_new_columns.end(), new_columns.begin());
    new_dict.columns = dh::ToSpan(new_columns);
    new_dict.feature_segments = dh::ToSpan(new_feature_segments);
    new_dict.n_total_cats = 3;

    ASSERT_THROW({ encoder.Recode(orig_dict, new_dict); }, std::logic_error);
    try {
      encoder.Recode(orig_dict, new_dict);
    } catch (std::logic_error const& e) {
      std::string msg = e.what();
      ASSERT_NE(msg.find("0th"), std::string::npos);
      ASSERT_NE(msg.find("`e`"), std::string::npos);
    }
  }
}
}  // namespace enc::cuda_impl
