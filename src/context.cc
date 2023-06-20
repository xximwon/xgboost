/**
 * Copyright 2014-2023 by XGBoost Contributors
 *
 * \brief Context object used for controlling runtime parameters.
 */
#include "xgboost/context.h"

#include "common/common.h"  // AssertGPUSupport
#include "common/threading_utils.h"
#include "xgboost/string_view.h"

namespace xgboost {

DMLC_REGISTER_PARAMETER(Context);

bst_d_ordinal_t constexpr Context::kCpuId;
std::int64_t constexpr Context::kDefaultSeed;

Context::Context() : cfs_cpu_count_{common::GetCfsCPUCount()} {}

std::int32_t ConfigureGpuId(std::int32_t gpu_id, bool fail_on_invalid) {
#if defined(XGBOOST_USE_CUDA)
  // When booster is loaded from a memory image (Python pickle or R
  // raw model), number of available GPUs could be different.  Wrap around it.
  std::int32_t n_gpus = common::AllVisibleGPUs();
  if (n_gpus == 0) {
    if (gpu_id != Context::kCpuId) {
      LOG(WARNING) << "No visible GPU is found, setting `ordinal` to -1";
    }
    gpu_id = Context::kCpuId;
  } else if (fail_on_invalid) {
    CHECK(gpu_id == Context::kCpuId || gpu_id < n_gpus)
        << "Only " << n_gpus << " GPUs are visible, ordinal " << gpu_id << " is invalid.";
  } else if (gpu_id != Context::kCpuId && gpu_id >= n_gpus) {
    gpu_id = gpu_id % n_gpus;
    LOG(WARNING) << "Only " << n_gpus << " GPUs are visible, setting `ordinal` to " << gpu_id;
  }
#else
  // Just set it to CPU, don't think about it.
  gpu_id = Context::kCpu;
  (void)(fail_on_invalid);
#endif  // defined(XGBOOST_USE_CUDA)

  if (gpu_id != Context::kCpuId) {
    common::SetDevice(gpu_id);
  }
  return gpu_id;
}

void Context::ParseDeviceOrdinal() {
  auto const& original = this->device;
  std::int32_t gpu_id{Context::kCpuId};
  StringView msg{R"(Invalid argument for `device`. Expected to be one of the following:
- CPU
- CUDA
- CUDA:<device ordinal>  # e.g. CUDA:0
)"};
  std::string device{original.c_str(), original.size()};
  // being lenient toward case.
  std::transform(device.cbegin(), device.cend(), device.begin(),
                 [](auto c) { return std::toupper(c); });
  auto split_it = std::find(device.cbegin(), device.cend(), ':');
  gpu_id = -2;  // mark it invalid for check.
  if (split_it == device.cend()) {
    // no ordinal.
    if (device == "CPU") {
      gpu_id = Context::kCpuId;
    } else if (device == "CUDA") {
      gpu_id = 0;  // use 0 as default;
    } else {
      LOG(FATAL) << msg << "Got: " << original;
    }
  } else {
    // must be CUDA when ordinal is specifed.
    auto splited = common::Split(device, ':');
    CHECK_EQ(splited.size(), 2) << msg;
    device = splited[0];
    CHECK_EQ(device, "CUDA") << msg << "Got: " << original;

    // boost::lexical_cast should be used instead, but for now some basic checks will do
    auto ordinal = splited[1];
    CHECK_GE(ordinal.size(), 1) << msg << "Got: " << original;
    bool valid =
        std::all_of(ordinal.cbegin(), ordinal.cend(), [](auto c) { return std::isdigit(c); });
    CHECK(valid) << msg << "Got: " << original;
    try {
      gpu_id = std::stoi(splited[1]);
    } catch (std::exception const& e) {
      LOG(FATAL) << msg << "Got: " << original;
    }
  }
  CHECK_GE(gpu_id, Context::kCpuId) << msg;

  gpu_id = ConfigureGpuId(gpu_id, this->fail_on_invalid_gpu_id);

  if (gpu_id == kCpuId) {
    this->device_ = Device{Device::kCPU, kCpuId};
  } else {
    CHECK_LE(gpu_id, std::numeric_limits<bst_d_ordinal_t>::max()) << "Ordinal value too large.";
    this->device_ = Device{Device::kCUDA, static_cast<bst_d_ordinal_t>(gpu_id)};
  }

  if (this->IsCPU()) {
    CHECK_EQ(this->device_.ordinal, kCpuId);
  }
}

std::int32_t Context::Threads(bool config) const {
  if (!config) {
    return nthread;
  }

  auto n_threads = common::OmpGetNumThreads(nthread);
  if (cfs_cpu_count_ > 0) {
    n_threads = std::min(n_threads, cfs_cpu_count_);
  }
  return n_threads;
}

#if !defined(XGBOOST_USE_CUDA)
CUDAContext const* Context::CUDACtx() const {
  common::AssertGPUSupport();
  return nullptr;
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
