/**
 * Copyright 2021-2023, XGBoost contributors
 */
#include "sparse_page_source.h"
#include "proxy_dmatrix.cuh"
#include "simple_dmatrix.cuh"

namespace xgboost {
namespace data {

namespace detail {
std::size_t NSamplesDevice(DMatrixProxy *proxy) {
  return Dispatch(proxy, [](auto const &value) { return value.NumRows(); });
}

std::size_t NFeaturesDevice(DMatrixProxy *proxy) {
  return Dispatch(proxy, [](auto const &value) { return value.NumCols(); });
}
}  // namespace detail

void DevicePush(DMatrixProxy *proxy, float missing, SparsePage *page) {
  Context ctx;
  if (proxy->Ctx()->IsCPU()) {
    auto device = dh::CurrentDevice();
    ctx = proxy->Ctx()->MakeCUDA(device);
  } else {
    ctx = *proxy->Ctx();
  }

  Dispatch(proxy, [&](auto const &value) { CopyToSparsePage(&ctx, value, missing, page); });
}
}  // namespace data
}  // namespace xgboost
