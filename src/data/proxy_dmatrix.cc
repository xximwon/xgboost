/**
 * Copyright 2021-2023, XGBoost Contributors
 * \file proxy_dmatrix.cc
 */

#include "proxy_dmatrix.h"

namespace xgboost::data {
void DMatrixProxy::SetArrayData(char const *c_interface) {
  std::shared_ptr<ArrayAdapter> adapter{new ArrayAdapter(StringView{c_interface})};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  this->ctx_ = this->ctx_.MakeCPU();
}

void DMatrixProxy::SetCSRData(char const *c_indptr, char const *c_indices,
                              char const *c_values, bst_feature_t n_features, bool on_host) {
  CHECK(on_host) << "Not implemented on device.";
  std::shared_ptr<CSRArrayAdapter> adapter{new CSRArrayAdapter(
      StringView{c_indptr}, StringView{c_indices}, StringView{c_values}, n_features)};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  this->ctx_ = this->ctx_.MakeCPU();
}
}  // namespace xgboost::data
