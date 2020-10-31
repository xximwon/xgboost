/*!
 * Copyright 2015-2019 by Contributors
 * \file common.cc
 * \brief Enable all kinds of global variables in common.
 */
#include <dmlc/thread_local.h>
#include <xgboost/logging.h>
#include "xgboost/generic_parameters.h"
#include "common.h"
#include "./random.h"

namespace xgboost {
namespace common {
/*! \brief thread local entry for random. */
struct RandomThreadLocalEntry {
  /*! \brief the random engine instance. */
  GlobalRandomEngine engine;
};

using RandomThreadLocalStore = dmlc::ThreadLocalStore<RandomThreadLocalEntry>;

GlobalRandomEngine& GlobalRandom() {
  return RandomThreadLocalStore::Get()->engine;
}

#if !defined(XGBOOST_USE_CUDA)
int AllVisibleGPUs() {
  return 0;
}
#endif  // !defined(XGBOOST_USE_CUDA)

#if defined(XGBOOST_USE_CUDA)
void DeviceStridedCopyGradient(Span<GradientPair> dst, Span<GradientPair const> in,
                               size_t stride, size_t step, int device);
#endif  // defined(XGBOOST_USE_CUDA)

void StridedCopy(HostDeviceVector<GradientPair> *out,
                 HostDeviceVector<GradientPair> const &in, size_t stride,
                 size_t step) {
  auto device = in.DeviceIdx();

#if defined(XGBOOST_USE_CUDA)
  if (device != GenericParameter::kCpuId) {
    out->SetDevice(device);
    DeviceStridedCopyGradient(out->DeviceSpan(), in.ConstDeviceSpan(), stride, step, device);
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)

  size_t n_out = out->Size();
  auto p_in = in.ConstHostPointer();
  auto p_out = out->HostPointer();
#pragma omp parallel for schedule(static)
  for (bst_omp_uint i = 0; i < n_out; ++i) {
    p_out[i] = p_in[i * stride + step];
  }
}
}  // namespace common
}  // namespace xgboost
