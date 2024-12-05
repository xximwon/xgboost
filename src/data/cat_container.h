/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once

#include <cstdint>  // for int32_t, int8_t
#include <memory>   // for unique_ptr
#include <variant>
#include <vector>  // for vector

#include "../encoder/ordinal.h"          // for DictionaryView
#include "../encoder/types.h"            // for Overloaded
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json

#if defined(XGBOOST_USE_CUDA)

#include <cuda/std/variant>  // for variant

#endif  // defined(XGBOOST_USE_CUDA)

namespace xgboost {
namespace cpu_impl {
struct CatStrArray {
  std::vector<std::int32_t> offsets;
  std::vector<enc::CatCharT> values;
};

template <typename T>
struct ViewToStorageImpl;

template <>
struct ViewToStorageImpl<enc::CatStrArrayView> {
  using Type = CatStrArray;
};

template <typename T>
struct ViewToStorageImpl<common::Span<T const>> {
  using Type = std::vector<T>;
};

template <typename... Ts>
struct ViewToStorage;

template <typename... Ts>
struct ViewToStorage<std::tuple<Ts...>> {
  using Type = std::tuple<typename ViewToStorageImpl<Ts>::Type...>;
};

using CatIndexTypes = ViewToStorage<enc::CatIndexViewTypes>::Type;

struct CatContainerImpl {
  std::vector<enc::cpu_impl::TupToVarT<CatIndexTypes>> columns;
  // View
  std::vector<enc::HostCatIndexView> columns_v;

  void Finalize() {
    this->columns_v.clear();
    for (auto const& col : this->columns) {
      std::visit(enc::Overloaded{[this](CatStrArray const& str) {
                                   this->columns_v.emplace_back(
                                       enc::CatStrArrayView{str.offsets, str.values});
                                 },
                                 [this](auto&& values) {
                                   this->columns_v.emplace_back(common::Span{values});
                                 }},
                 col);
    }
  }

  void Copy(CatContainerImpl const* that) { this->columns = that->columns; }
};
};  // namespace cpu_impl

namespace cuda_impl {
struct CatContainerImpl;
}

class CatContainer {
  void CopyCommon(CatContainer const& that) {
    this->sorted_idx_.Resize(0);

    this->cpu_impl->Copy(that.cpu_impl.get());
    this->cpu_impl->Finalize();

    this->feature_segments_.SetDevice(that.feature_segments_.Device());
    this->feature_segments_.Resize(that.feature_segments_.Size());
    this->feature_segments_.Copy(that.feature_segments_);

    this->n_total_cats_ = that.n_total_cats_;
  }

  void FinalizeCommon() {
    this->sorted_idx_.Resize(0);

    auto& h_feature_segments = this->feature_segments_.HostVector();
    std::partial_sum(h_feature_segments.cbegin(), h_feature_segments.cend(),
                     h_feature_segments.begin());
    CHECK_EQ(h_feature_segments.back(), n_total_cats_);

    this->cpu_impl->Finalize();
  }

 public:
  CatContainer();
  explicit CatContainer(enc::HostColumnsView const& df);
#if defined(XGBOOST_USE_CUDA)
  explicit CatContainer(DeviceOrd device, enc::DeviceColumnsView const& df);
#endif  // defined(XGBOOST_USE_CUDA)
  ~CatContainer();

  void Copy(CatContainer const& that);

  [[nodiscard]] bool Empty() const { return this->cpu_impl->columns.empty(); }
  [[nodiscard]] std::size_t Size() const { return this->cpu_impl->columns.size(); }

  void Sort(Context const* ctx);

  [[nodiscard]] common::Span<std::int32_t const> RefSortedIndex(Context const* ctx) const {
    if (ctx->IsCPU()) {
      return this->sorted_idx_.ConstHostSpan();
    } else {
      return this->sorted_idx_.ConstDeviceSpan();
    }
  }

  void Save(Json* out) const;
  void Load(Json const& in);

  [[nodiscard]] enc::HostColumnsView HostView() const {
    CHECK_EQ(this->cpu_impl->columns.size(), this->cpu_impl->columns_v.size());
    return {common::Span{this->cpu_impl->columns_v}, this->feature_segments_.ConstHostSpan(),
            this->n_total_cats_};
  }

#if defined(XGBOOST_USE_CUDA)
  [[nodiscard]] enc::DeviceColumnsView DeviceView(Context const* ctx) const;
#endif  // defined(XGBOOST_USE_CUDA)

  void Finalize();

  std::unique_ptr<cpu_impl::CatContainerImpl> cpu_impl;

 private:
  HostDeviceVector<std::int32_t> feature_segments_;
  std::int32_t n_total_cats_{0};

#if defined(XGBOOST_USE_CUDA)
  std::unique_ptr<cuda_impl::CatContainerImpl> cu_impl_;
  HostDeviceVector<std::int32_t> sorted_idx_;
#endif  // defined(XGBOOST_USE_CUDA)
};

std::ostream& operator<<(std::ostream& os, enc::CatStrArrayView const& strings);
}  // namespace xgboost
