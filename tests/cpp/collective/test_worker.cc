/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include "test_worker.h"

#include <xgboost/base.h>     // for Args
#include <xgboost/context.h>  // for Context

#include <cstdint>  // for int32_t
#include <fstream>  // for ifstream
#include <thread>   // for thread

#include "../../../src/collective/communicator-inl.h"  // for GetRank
#include "../helpers.h"                                // for FileExists

namespace xgboost::collective {
bool SocketTest::SkipTest() {
  std::string path{"/sys/module/ipv6/parameters/disable"};
  if (FileExists(path)) {
    std::ifstream fin(path);
    if (!fin) {
      return true;
    }
    std::string s_value;
    fin >> s_value;
    auto value = std::stoi(s_value);
    if (value != 0) {
      return true;
    }
  } else {
    return true;
  }
  return false;
}

[[nodiscard]] Context MakeCtxForDistributedTest(bool use_cuda) {
  std::int32_t n_workers = collective::GetWorldSize();
  auto rank = collective::GetRank();
  auto n_total_threads = std::thread::hardware_concurrency();
  Context ctx;
  if (use_cuda) {
    ctx = MakeCUDACtx(common::AllVisibleGPUs() == 1 ? 0 : rank);
  }
  ctx.UpdateAllowUnknown(Args{{"nthread", std::to_string(n_total_threads / n_workers)}});
  return ctx;
}
}  // namespace xgboost::collective
