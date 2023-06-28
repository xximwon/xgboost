/**
 * Copyright 2014-2023 by XGBoost Contributors
 *
 * \brief Context object used for controlling runtime parameters.
 */
#include "xgboost/context.h"

#include "common/common.h"     // AssertGPUSupport
#include "common/error_msg.h"  // InvalidOrdinalError
#include "common/threading_utils.h"
#include "xgboost/string_view.h"

namespace xgboost {

DMLC_REGISTER_PARAMETER(Context);

bst_d_ordinal_t constexpr Context::kCpuId;
std::int64_t constexpr Context::kDefaultSeed;

Context::Context() : cfs_cpu_count_{common::GetCfsCPUCount()} {}

namespace {
Device ConfigureDeviceOrd(std::int32_t cu_ordinal, bool fail_on_invalid) {
  CHECK_GE(cu_ordinal, 0);

#if defined(XGBOOST_USE_CUDA)
  // When booster is loaded from a memory image (Python pickle or R raw model), number of
  // available GPUs could be different.  Wrap around it.
  std::int32_t n_gpus = common::AllVisibleGPUs();
  if (n_gpus == 0) {
    LOG(WARNING) << "No visible GPU is found, setting `ordinal` to -1";
    cu_ordinal = Context::kCpuId;
  } else if (fail_on_invalid) {
    CHECK_LT(cu_ordinal, n_gpus) << "Only " << n_gpus << " GPUs are visible, ordinal " << cu_ordinal
                                 << " is invalid.";
  } else if (cu_ordinal >= n_gpus) {
    cu_ordinal = cu_ordinal % n_gpus;
    LOG(WARNING) << "Only " << n_gpus << " GPUs are visible, setting `ordinal` to " << cu_ordinal;
  }
#else
  // Just set it to CPU.
  gpu_id = Context::kCpu;
  (void)(fail_on_invalid);
#endif  // defined(XGBOOST_USE_CUDA)

  if (cu_ordinal == Context::kCpuId) {
    return Device::CPU();
  }

  common::SetDevice(cu_ordinal);
  CHECK_LE(cu_ordinal, std::numeric_limits<bst_d_ordinal_t>::max())
      << "Device ordinal value too large.";
  return Device::CUDA(static_cast<bst_d_ordinal_t>(cu_ordinal));
}

void ThrowIf(bool cond, StringView original) {
  if (XGBOOST_EXPECT(cond, false)) {
    error::InvalidOrdinal(StringView{original});
  }
}
}  // anonymous namespace

void Context::ParseDeviceOrdinal() {
  auto original = StringView{this->device};
  std::string device{original.c_str(), original.size()};
  // being lenient toward case.
  std::transform(device.cbegin(), device.cend(), device.begin(),
                 [](auto c) { return std::toupper(c); });

  /** Ordinal is not specified: CPU/CUDA */
  if (std::find(device.cbegin(), device.cend(), ':') == device.cend()) {
    if (device == "CPU") {
      this->device_ = Device::CPU();
    } else if (device == "CUDA") {
      auto current_d = common::CurrentDeviceOrd();
      if (current_d == kCpuId) {
        LOG(WARNING) << "XGBoost not compiled with CUDA, setting device to CPU instead.";
        this->device_ = Device::CPU();
      } else {
        this->device_ = Device::CUDA(current_d);
      }
    } else {
      error::InvalidOrdinal(original);
    }

    return;
  }

  /** Ordinal is specified: CUDA:<ordinal> */
  // Use 32bit int to hold larger value for validation.
  std::int32_t cu_ordinal{-2};
  auto splited = common::Split(device, ':');
  ThrowIf(splited.size() != 2, original);
  device = splited[0];
  // must be CUDA when ordinal is specifed.
  ThrowIf(device != "CUDA", original);
  // boost::lexical_cast should be used instead, but for now some basic checks will do
  auto ordinal = splited[1];
  ThrowIf(ordinal.empty(), original);
  ThrowIf(!std::all_of(ordinal.cbegin(), ordinal.cend(), [](auto c) { return std::isdigit(c); }),
          original);
  try {
    cu_ordinal = std::stoi(splited[1]);
  } catch (std::exception const& e) {
    error::InvalidOrdinal(original);
  }

  ThrowIf(cu_ordinal < 0, original);
  this->device_ = ConfigureDeviceOrd(cu_ordinal, this->fail_on_invalid_gpu_id);
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
