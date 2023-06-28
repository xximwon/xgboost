/*!
 * Copyright 2014-2022 by Contributors
 * \file context.h
 */
#ifndef XGBOOST_CONTEXT_H_
#define XGBOOST_CONTEXT_H_

#include <xgboost/base.h>  // for bst_d_ordinal_t
#include <xgboost/logging.h>
#include <xgboost/parameter.h>

#include <memory>  // std::shared_ptr
#include <string>

namespace xgboost {

struct CUDAContext;

struct DeviceOrd {
  enum Type : std::int16_t { kCPU = 0, kCUDA = 1 } device{kCPU};
  // CUDA device ordinal.
  bst_d_ordinal_t ordinal{-1};

  [[nodiscard]] bool IsCUDA() const { return device == kCUDA; }
  [[nodiscard]] bool IsCPU() const { return device == kCPU; }

  DeviceOrd() = default;
  constexpr DeviceOrd(Type type, bst_d_ordinal_t ord) : device{type}, ordinal{ord} {}

  DeviceOrd(DeviceOrd const& that) = default;
  DeviceOrd& operator=(DeviceOrd const& that) = default;
  DeviceOrd(DeviceOrd&& that) = default;
  DeviceOrd& operator=(DeviceOrd&& that) = default;

  constexpr static auto CPU() { return DeviceOrd{kCPU, -1}; }
  static auto CUDA(bst_d_ordinal_t ordinal) { return DeviceOrd{kCPU, ordinal}; }

  bool operator==(DeviceOrd const& that) const {
    return device == that.device && ordinal == that.ordinal;
  }
  bool operator!=(DeviceOrd const& that) const { return !(*this == that); }

  [[nodiscard]] std::string Name() const {
    switch (device) {
      case DeviceOrd::kCPU:
        return "CPU";
      case DeviceOrd::kCUDA:
        return "CUDA:" + std::to_string(ordinal);
      default: {
        LOG(FATAL) << "Unknown device.";
        return "";
      }
    }
  }
};

static_assert(sizeof(DeviceOrd) == 4);

struct Context : public XGBoostParameter<Context> {
 private:
  std::string device;  // NOLINT
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  std::int32_t nthread{0};  // NOLINT
  DeviceOrd device_{DeviceOrd::CPU()};

 public:
  // Constant representing the device ID of CPU.
  static bst_d_ordinal_t constexpr kCpuId = -1;
  bst_d_ordinal_t constexpr InvalidOrdinal() { return -2; }
  static std::int64_t constexpr kDefaultSeed = 0;

 public:
  Context();

  template <typename Container>
  Args UpdateAllowUnknown(Container const& kwargs) {
    auto args = XGBoostParameter<Context>::UpdateAllowUnknown(kwargs);
    this->ParseDeviceOrdinal();
    for (auto const& kv : args) {
      CHECK_NE(kv.first, "gpu_id");
    }
    return args;
  }

  // stored random seed
  std::int64_t seed{kDefaultSeed};
  // whether seed the PRNG each iteration
  bool seed_per_iteration{false};
  // fail when gpu_id is invalid
  bool fail_on_invalid_gpu_id{false};

  bool validate_parameters{false};
  void ParseDeviceOrdinal();
  /**
   * \brief Return automatically configured number of threads.
   *
   * \param config Whether we should configure the number of threads based on the runtime
   *               environment. Only used for testing.
   */
  std::int32_t Threads(bool config = true) const;

  bool IsCPU() const { return device_.device == DeviceOrd::kCPU; }
  bool IsCUDA() const { return !IsCPU(); }
  DeviceOrd Device() const { return device_; }
  /**
   * \brief Returns CUDA device ordinal.
   */
  bst_d_ordinal_t Ordinal() const { return device_.ordinal; }
  /**
   * \brief Name of the current device.
   */
  std::string DeviceName() const { return this->device_.Name(); }

  CUDAContext const* CUDACtx() const;
  // Make a CUDA context based on the current context.
  Context MakeCUDA(bst_d_ordinal_t device = 0) const {
    Context ctx = *this;
    ctx.device_ = DeviceOrd{DeviceOrd::kCUDA, device};
    return ctx;
  }
  Context MakeCPU() const {
    Context ctx = *this;
    ctx.device_ = DeviceOrd{DeviceOrd::kCPU, kCpuId};
    return ctx;
  }

  // declare parameters
  DMLC_DECLARE_PARAMETER(Context) {
    DMLC_DECLARE_FIELD(seed)
        .set_default(kDefaultSeed)
        .describe("Random number seed during training.");
    DMLC_DECLARE_ALIAS(seed, random_state);
    DMLC_DECLARE_FIELD(seed_per_iteration)
        .set_default(false)
        .describe("Seed PRNG determnisticly via iterator number.");
    DMLC_DECLARE_FIELD(device).set_default("CPU").describe("Device ordinal.");
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe("Number of threads to use.");
    DMLC_DECLARE_ALIAS(nthread, n_jobs);
    DMLC_DECLARE_FIELD(fail_on_invalid_gpu_id)
        .set_default(false)
        .describe("Fail with error when gpu_id is invalid.");
    DMLC_DECLARE_FIELD(validate_parameters)
        .set_default(false)
        .describe("Enable checking whether parameters are used or not.");
  }

 private:
  // mutable for lazy initialization for cuda context to avoid initializing CUDA at load.
  // shared_ptr is used instead of unique_ptr as with unique_ptr it's difficult to define p_impl
  // while trying to hide CUDA code from host compiler.
  mutable std::shared_ptr<CUDAContext> cuctx_;
  // cached value for CFS CPU limit. (used in containerized env)
  std::int32_t cfs_cpu_count_;  // NOLINT
};
}  // namespace xgboost

#endif  // XGBOOST_CONTEXT_H_
