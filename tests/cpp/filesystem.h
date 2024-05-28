/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#ifndef XGBOOST_TESTS_CPP_FILESYSTEM_H
#define XGBOOST_TESTS_CPP_FILESYSTEM_H

#include <iostream>

namespace xgboost {
struct TemporaryDirectory {
  std::string path;

  TemporaryDirectory();
  ~TemporaryDirectory() {
    std::cout << "path:" << path << std::endl;
    // if (!path.empty()) {
    //   std::filesystem::remove_all(path);
    // }
  }
};
}  // namespace xgboost

#endif  // XGBOOST_TESTS_CPP_FILESYSTEM_H
