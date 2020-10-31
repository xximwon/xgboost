/*!
 * Copyright 2018 XGBoost contributors
 */
#include "common.h"
#include "device_helpers.cuh"

namespace xgboost {
namespace common {

int AllVisibleGPUs() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDeviceCount will fail.
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
  } catch(const dmlc::Error &except) {
    return 0;
  }
  return n_visgpus;
}

void DeviceStridedCopyGradient(Span<GradientPair> out, Span<GradientPair const> in,
                               size_t stride, size_t step, int device) {
  dh::LaunchN(device, out.size(), [=] XGBOOST_DEVICE(size_t idx) {
    size_t in_idx = idx * stride + step;
    out[idx] = in[in_idx];
  });
}
}  // namespace common
}  // namespace xgboost
