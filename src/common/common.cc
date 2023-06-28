/*!
 * Copyright 2015-2019 by Contributors
 * \file common.cc
 * \brief Enable all kinds of global variables in common.
 */
#include "common.h"

#include <dmlc/thread_local.h>
#include <xgboost/logging.h>

#include "./random.h"
#include "xgboost/context.h"  // for Context

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

bst_d_ordinal_t CurrentDeviceOrd() { return Context::kCpuId; }
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace common
}  // namespace xgboost
