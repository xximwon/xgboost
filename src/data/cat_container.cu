/**
 * Copyright 2024, XGBoost Contributors
 */
#include <thrust/copy.h>  // for copy

#include <cstdint>  // for int32_t
#include <memory>   // for make_unique
#include <vector>   // for vector

#include "../common/device_helpers.cuh"  // for ToSpan
#include "../common/device_vector.cuh"   // for device_vector
#include "../encoder/ordinal.cuh"        // for SortNames
#include "../encoder/ordinal.h"          // for DictionaryView
#include "../encoder/types.h"            // for Overloaded
#include "cat_container.h"               // for CatContainer
#include "xgboost/span.h"                // for Span

namespace xgboost {
namespace cuda_impl {
namespace {
struct CatStrArray {
  dh::device_vector<std::int32_t> offsets;
  dh::device_vector<enc::CatCharT> values;

  CatStrArray() = default;
  CatStrArray(CatStrArray const& that) = delete;
  CatStrArray& operator=(CatStrArray const& that) = delete;

  CatStrArray(CatStrArray&& that) = default;
  CatStrArray& operator=(CatStrArray&& that) = default;
};

template <typename T>
struct ViewToStorageImpl;

template <>
struct ViewToStorageImpl<enc::CatStrArrayView> {
  using Type = CatStrArray;
};

template <typename T>
struct ViewToStorageImpl<common::Span<T const>> {
  using Type = dh::device_vector<T>;
};

template <typename... Ts>
struct ViewToStorage;

template <typename... Ts>
struct ViewToStorage<std::tuple<Ts...>> {
  using Type = std::tuple<typename ViewToStorageImpl<Ts>::Type...>;
};

using CatIndexTypes = ViewToStorage<enc::CatIndexViewTypes>::Type;
}  // namespace

struct CatContainerImpl {
  std::vector<enc::cpu_impl::TupToVarT<CatIndexTypes>> columns;
  dh::device_vector<enc::cuda_impl::TupToVarT<enc::CatIndexViewTypes>> columns_v;

  void Finalize(cpu_impl::CatContainerImpl const* cpu_impl) {
    this->columns.resize(cpu_impl->columns.size());
    this->columns_v.resize(cpu_impl->columns_v.size());
    CHECK_EQ(this->columns.size(), this->columns_v.size());

    std::vector<decltype(columns_v)::value_type> h_columns_v(this->columns_v.size());
    // FIXME(jiamingy): We can eliminate most of the copies once cuDF exposes the device
    // `to_arrow` method.
    for (std::size_t f_idx = 0, n = cpu_impl->columns.size(); f_idx < n; ++f_idx) {
      auto const& col_v = cpu_impl->columns_v[f_idx];
      std::visit(enc::Overloaded{
                     [this, f_idx, &h_columns_v](enc::CatStrArrayView const& str) {
                       this->columns[f_idx].emplace<CatStrArray>();
                       auto& col = std::get<CatStrArray>(this->columns[f_idx]);
                       // Handle the offsets
                       col.offsets.resize(str.offsets.size());
                       thrust::copy_n(str.offsets.data(), str.offsets.size(), col.offsets.data());
                       // Handle the values
                       col.values.resize(str.values.size());
                       thrust::copy_n(str.values.data(), str.values.size(), col.values.data());
                       // Create the view
                       h_columns_v[f_idx].emplace<enc::CatStrArrayView>();
                       auto& col_v = cuda::std::get<enc::CatStrArrayView>(h_columns_v[f_idx]);
                       col_v = {dh::ToSpan(col.offsets), dh::ToSpan(col.values)};
                     },
                     [this, f_idx, &h_columns_v](auto&& values) {
                       using T =
                           std::remove_cv_t<typename std::decay_t<decltype(values)>::value_type>;

                       this->columns[f_idx].emplace<dh::device_vector<T>>();
                       auto& col = std::get<dh::device_vector<T>>(this->columns[f_idx]);

                       col.resize(values.size());
                       thrust::copy_n(values.data(), values.size(), col.data());

                       using V = common::Span<std::add_const_t<T>>;
                       h_columns_v[f_idx].emplace<V>();
                       auto& col_v = cuda::std::get<V>(h_columns_v[f_idx]);
                       col_v = dh::ToSpan(col);
                     }},
                 col_v);
    }
    thrust::copy_n(h_columns_v.data(), h_columns_v.size(), this->columns_v.data());
  }
};
}  // namespace cuda_impl

CatContainer::CatContainer()
    : cpu_impl{std::make_unique<cpu_impl::CatContainerImpl>()},
      cu_impl_{std::make_unique<cuda_impl::CatContainerImpl>()} {}

CatContainer::CatContainer(DeviceOrd device, enc::DeviceColumnsView const& df) : CatContainer{} {
  this->n_total_cats_ = df.n_total_cats;

  this->feature_segments_.SetDevice(device);
  this->feature_segments_.Resize(df.feature_segments.size());
  auto d_segs = this->feature_segments_.DeviceSpan();
  thrust::copy_n(dh::tcbegin(df.feature_segments), df.feature_segments.size(), dh::tbegin(d_segs));

  // FIXME(jiamingy): We can use a single kernel for copying data once cuDF can return
  // device data.
  // For now, this is copy and pasted from the CPU implementation with `cuda::std::visit` instead of
  // `std::visit`.
  for (auto const& col : df.columns) {
    cuda::std::visit(
        enc::Overloaded{
            [this](enc::CatStrArrayView str) {
              using T = typename cpu_impl::ViewToStorageImpl<enc::CatStrArrayView>::Type;
              this->cpu_impl->columns.emplace_back();
              this->cpu_impl->columns.back().emplace<T>();
              auto& v = std::get<T>(this->cpu_impl->columns.back());
              v.offsets.resize(str.offsets.size());
              v.values.resize(str.values.size());
              std::copy_n(str.offsets.data(), str.offsets.size(), v.offsets.data());
              std::copy_n(str.values.data(), str.values.size(), v.values.data());
            },
            [this](auto&& values) {
              using T = typename cpu_impl::ViewToStorageImpl<std::decay_t<decltype(values)>>::Type;
              this->cpu_impl->columns.emplace_back();
              this->cpu_impl->columns.back().emplace<T>();
              auto& v = std::get<T>(this->cpu_impl->columns.back());
              v.resize(values.size());
              std::copy_n(values.data(), values.size(), v.data());
            }},
        col);
  }

  this->sorted_idx_.SetDevice(device);
  this->sorted_idx_.Resize(0);

  this->cpu_impl->Finalize();
  this->cu_impl_->Finalize(this->cpu_impl.get());
}

CatContainer::~CatContainer() = default;

void CatContainer::Copy(CatContainer const& that) {
  this->CopyCommon(that);
  this->cu_impl_->Finalize(this->cpu_impl.get());
}

void CatContainer::Sort(Context const* ctx) {
  if (ctx->IsCPU()) {
    auto view = this->HostView();
    for (std::size_t i = 0; i < view.Size(); ++i) {
      this->sorted_idx_.HostVector() = enc::SortNames(view.columns[i]);
    }
  } else {
    auto view = this->DeviceView(ctx);
    auto sorted_idx = enc::cuda_impl::SortNames(view);
    this->sorted_idx_.SetDevice(ctx->Device());
    this->sorted_idx_.Resize(sorted_idx.size());
    thrust::copy_n(sorted_idx.cbegin(), sorted_idx.size(),
                   dh::tbegin(this->sorted_idx_.DeviceSpan()));
  }
}

[[nodiscard]] enc::DeviceColumnsView CatContainer::DeviceView(Context const* ctx) const {
  CHECK(ctx->IsCUDA());
  this->feature_segments_.SetDevice(ctx->Device());
  return {dh::ToSpan(this->cu_impl_->columns_v), this->feature_segments_.ConstDeviceSpan(),
          this->n_total_cats_};
}

void CatContainer::Finalize() {
  this->FinalizeCommon();
  this->cu_impl_->Finalize(this->cpu_impl.get());
}
}  // namespace xgboost
