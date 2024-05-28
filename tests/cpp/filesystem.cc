/**
 * Copyright 2024, XGBoost Contributors
 */
#include "./filesystem.h"

// #include <process.h>
#include <xgboost/logging.h>

#include <filesystem>  // for temp_directory_path
#include <string>      // for string

namespace xgboost {
TemporaryDirectory::TemporaryDirectory() {
  auto tmpdir = std::filesystem::temp_directory_path();

  // std::int32_t pid = getpid();
  {
    auto name = tmpdir / std::filesystem::path{std::to_string(0)};
    std::filesystem::create_directory(name);
  }

  std::string dirtemplate = tmpdir.string() + "/tmpdir.XXXXXX";
  char* name = mkdtemp(dirtemplate.data());
  if (!name) {
    LOG(FATAL) << "TemporaryDirectory(): "
               << "Could not create temporary directory";
  }
  path = std::string(name);
}
}  // namespace xgboost
