/**
 * Copyright 2024, XGBoost Contributors
 */
#include "cat_container.h"

#include <algorithm>  // for copy
#include <cstddef>    // for size_t
#include <memory>     // for make_unique
#include <vector>     // for vector

#include "../encoder/types.h"  // for Overloaded
#include "xgboost/json.h"      // for Json

namespace xgboost {
CatContainer::CatContainer(enc::HostColumnsView const& df) : CatContainer{} {
  this->n_total_cats_ = df.n_total_cats;

  this->feature_segments_.Resize(df.feature_segments.size());
  auto& seg = this->feature_segments_.HostVector();
  std::copy_n(df.feature_segments.data(), df.feature_segments.size(), seg.begin());

  for (auto const& col : df.columns) {
    std::visit(enc::Overloaded{
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
                     using T =
                         typename cpu_impl::ViewToStorageImpl<std::decay_t<decltype(values)>>::Type;
                     this->cpu_impl->columns.emplace_back();
                     this->cpu_impl->columns.back().emplace<T>();
                     auto& v = std::get<T>(this->cpu_impl->columns.back());
                     v.resize(values.size());
                     std::copy_n(values.data(), values.size(), v.data());
                   }},
               col);
  }

  this->sorted_idx_.Resize(0);
  this->cpu_impl->Finalize();
}

void CatContainer::Save(Json* p_out) const {
  auto& out = *p_out;

  auto const& columns = this->cpu_impl->columns;
  std::vector<Json> arr(this->cpu_impl->columns.size());
  for (std::size_t fidx = 0, n_features = columns.size(); fidx < n_features; ++fidx) {
    auto& f_out = arr[fidx];

    auto const& col = columns[fidx];
    std::visit(enc::Overloaded{
                   [&f_out](cpu_impl::CatStrArray const& str) {
                     f_out = Object{};
                     I32Array joffsets{str.offsets.size()};
                     auto const& f_offsets = str.offsets;
                     std::copy(f_offsets.cbegin(), f_offsets.cend(), joffsets.GetArray().begin());
                     f_out["offsets"] = std::move(joffsets);

                     I8Array jnames{str.values.size()};  // fixme: uint8
                     auto const& f_names = str.values;
                     std::copy(f_names.cbegin(), f_names.cend(), jnames.GetArray().begin());
                     f_out["values"] = std::move(jnames);
                   },
                   [&f_out](auto&& values) {
                     I32Array array{values.size()};
                     f_out = std::move(array);
                   }},
               col);
  }

  auto jf_segments = I32Array{this->feature_segments_.Size()};
  auto const& hf_segments = this->feature_segments_.ConstHostVector();
  std::copy(hf_segments.cbegin(), hf_segments.cend(), jf_segments.GetArray().begin());

  out = Object{};
  out["feature_segments"] = std::move(jf_segments);
  out["enc"] = arr;
}

void CatContainer::Load(Json const& in) {
  auto array = get<Array const>(in["enc"]);
  auto n_features = array.size();

  auto& columns = this->cpu_impl->columns;
  for (std::size_t fidx = 0; fidx < n_features; ++fidx) {
    if (IsA<Object>(array[fidx])) {
      auto const& column = get<Object const>(array[fidx]);
      // str
      // fixme: load json
      auto joffset = get<I32Array const>(column.at("offsets"));
      auto jnames = get<I8Array const>(column.at("values"));

      cpu_impl::CatStrArray str{};
      str.offsets = joffset;
      str.values = jnames;

      columns.emplace_back(str);
    }
  }

  auto jf_segments = get<I32Array const>(in["feature_segments"]);
  auto& hf_segments = this->feature_segments_.HostVector();
  hf_segments.resize(jf_segments.size());
  std::copy(jf_segments.cbegin(), jf_segments.cend(), hf_segments.begin());
  CHECK(!hf_segments.empty());
  this->n_total_cats_ = hf_segments.back();

  this->Finalize();
}

std::ostream& operator<<(std::ostream& os, enc::CatStrArrayView const& strings) {
  auto const& offset = strings.offsets;
  auto const& data = strings.values;
  os << "[";
  for (std::size_t i = 1, n = offset.size(); i < n; ++i) {
    auto begin = offset[i - 1];
    auto end = offset[i];
    auto ptr = reinterpret_cast<char const*>(data.data()) + begin;
    os << StringView{ptr, static_cast<std::size_t>(end - begin)};
    if (i != n - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

#if !defined(XGBOOST_USE_CUDA)
CatContainer::CatContainer() : cpu_impl{std::make_unique<cpu_impl::CatContainerImpl>()} {}

CatContainer::~CatContainer() = default;

void CatContainer::Copy(CatContainer const& that) { this->CopyCommon(that); }

void CatContainer::Finalize() { this->FinalizeCommon(); }
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
