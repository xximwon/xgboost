/**
 * Copyright 2018-2022, XGBoost contributors
 */
#include "common.h"
#include "device_helpers.cuh"

namespace xgboost::common {
void SetDevice(std::int32_t device) {
  if (device >= 0) {
    dh::safe_cuda(cudaSetDevice(device));
  }
}

int AllVisibleGPUs() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDeviceCount will fail.
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
  } catch (const dmlc::Error &) {
    cudaGetLastError();  // reset error.
    return 0;
  }
  return n_visgpus;
}

bst_d_ordinal_t CurrentDeviceOrd() { return static_cast<bst_d_ordinal_t>(dh::CurrentDevice()); }
}  // namespace xgboost::common
