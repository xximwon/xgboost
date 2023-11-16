/**
 * Copyright 2023 by XGBoost contributors
 */
#include "error_msg.h"

#include <mutex>    // for call_once, once_flag
#include <sstream>  // for stringstream

#include "../collective/communicator-inl.h"  // for GetRank
#include "xgboost/context.h"                 // for Context
#include "xgboost/logging.h"

namespace xgboost::error {
[[nodiscard]] std::string DeprecatedFunc(StringView old, StringView since, StringView replacement) {
  std::stringstream ss;
  ss << "`" << old << "` is deprecated since" << since << ", use `" << replacement << "` instead.";
  return ss.str();
}

void WarnDeprecatedGPUHist() {
  auto msg =
      "The tree method `gpu_hist` is deprecated since 2.0.0. To use GPU training, set the `device` "
      R"(parameter to CUDA instead.

    E.g. tree_method = "hist", device = "cuda"
)";
  LOG(WARNING) << msg;
}

void WarnManualUpdater() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    LOG(WARNING)
        << "You have manually specified the `updater` parameter. The `tree_method` parameter "
           "will be ignored. Incorrect sequence of updaters will produce undefined "
           "behavior. For common uses, we recommend using `tree_method` parameter instead."
           "\n"
           "The warning will only be shown once";
  });
}

void WarnDeprecatedGPUId() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    auto msg = DeprecatedFunc("gpu_id", "2.0.0", "device");
    msg += " E.g. device=cpu/cuda/cuda:0";
    LOG(WARNING) << msg;
  });
}

void WarnEmptyDataset() {
  static std::once_flag flag;
  std::call_once(flag,
                 [] { LOG(WARNING) << "Empty dataset at worker: " << collective::GetRank(); });
}

void MismatchedDevices(Context const* booster, Context const* data) {
  static std::once_flag flag;
  std::call_once(flag, [&] {
    LOG(WARNING)
        << "Falling back to prediction using DMatrix due to mismatched devices. This might "
           "lead to higher memory usage and slower performance. XGBoost is running on: "
        << booster->DeviceName() << ", while the input data is on: " << data->DeviceName() << ".\n"
        << R"(Potential solutions:
- Use a data structure that matches the device ordinal in the booster.
- Set the device for booster before call to inplace_predict.

This warning will only be shown once.
)";
  });
}

std::string QidWeight() {
  // FIXME: Remove this constraint.
  std::stringstream ss;
  ss << "Starting from 2.1.0, when QID is used along with group-based weight, QID should start"
        " from zero. Possible solutions:";
  ss << R"(
- Change the query index to be zero-based.
- Use sample weight instead (one weight per sample).

)";
  if (collective::IsDistributed()) {
    ss << "Since you are using distributed system, using sample weight might be easier since we"
          " don't need to change the query index for each worker.";
  }
  return ss.str();
}
}  // namespace xgboost::error
