/**
 * Copyright 2021-2023, XGBoost contributors
 */
#include "sparse_page_source.h"
#include "proxy_dmatrix.cuh"
#include "simple_dmatrix.cuh"

namespace xgboost::data {
namespace detail {
std::size_t NSamplesDevice(DMatrixProxy *proxy) {
  return Dispatch(proxy, [](auto const &value) { return value.NumRows(); });
}

std::size_t NFeaturesDevice(DMatrixProxy *proxy) {
  return Dispatch(proxy, [](auto const &value) { return value.NumCols(); });
}
}  // namespace detail

void DevicePush(DMatrixProxy *proxy, float missing, SparsePage *page) {
  auto ctx = proxy->Ctx();
  std::int32_t device{ctx->Ordinal()};
  if (ctx->IsCPU()) {
    device = dh::CurrentDevice();
  }
  CHECK_GE(device, 0);

  Dispatch(proxy, [&](auto const &value) { CopyToSparsePage(value, device, missing, page); });
}
}  // namespace xgboost::data
