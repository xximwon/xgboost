/**
 * Copyright 2019-2024, XGBoost Contributors
 */
#include "../common/cuda_rt_utils.h"
#include "device_adapter.cuh"

namespace xgboost::data {
CudfAdapter::CudfAdapter(StringView cuda_arrinf) {
  Json interfaces = Json::Load(cuda_arrinf);
  std::vector<Json> const& jcolumns = get<Array>(interfaces);
  std::size_t n_columns = jcolumns.size();
  CHECK_GT(n_columns, 0) << "The number of columns must not equal to 0.";

  std::vector<ArrayInterface<1>> columns;
  std::vector<std::int32_t> feature_segments{0};
  for (auto const& jcol : jcolumns) {
    std::int32_t n_cats{0};
    if (IsA<Array>(jcol)) {
      // This is a dictionary type.
      auto const& first = get<Object const>(jcol[0]);
      if (first.find("offsets") == first.cend()) {
        n_cats = GetArrowNumericIndex(first, &cats_);
      } else {
        n_cats = GetArrowDictionary(jcol, &cats_, &columns, &n_bytes_, &num_rows_);
      }
    } else {
      auto col = ArrayInterface<1>(get<Object const>(jcol));
      columns.push_back(col);
      this->cats_.emplace_back();
      this->num_rows_ = std::max(num_rows_, col.Shape<0>());
      CHECK_EQ(device_.ordinal, dh::CudaGetPointerDevice(col.data))
          << "All columns should use the same device.";
      CHECK_EQ(num_rows_, col.Shape<0>()) << "All columns should have same number of rows.";
      n_bytes_ += col.ElementSize() * col.Shape<0>();
    }
    feature_segments.emplace_back(n_cats);
  }
  // Categories
  std::partial_sum(feature_segments.cbegin(), feature_segments.cend(), feature_segments.begin());
  this->n_total_cats_ = feature_segments.back();
  this->cat_segments_ = std::move(feature_segments);
  this->d_cats_ = this->cats_;  // thrust copy

  CHECK(!columns.empty());
  device_ = DeviceOrd::CUDA(dh::CudaGetPointerDevice(columns.front().data));
  CHECK(device_.IsCUDA());
  curt::SetDevice(device_.ordinal);

  this->columns_ = columns;
  batch_ = CudfAdapterBatch(dh::ToSpan(columns_), num_rows_);
}
}  // namespace xgboost::data
