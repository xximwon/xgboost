/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <dmlc/registry.h>
#include <nvtx3/nvToolsExt.h>

#include <cstddef>  // for size_t
#include <cstdint>  // for uint64_t

#include "../common/io.h"                 // for AlignedResourceReadStream, AlignedFileWriteStream
#include "../common/ref_resource_view.h"  // for ReadVec, WriteVec
#include "ellpack_page.cuh"               // for EllpackPage
#include "ellpack_page_raw_format.h"
#include "ellpack_page_source.h"

namespace xgboost::data {
DMLC_REGISTRY_FILE_TAG(ellpack_page_raw_format);

namespace {
template <typename T>
[[nodiscard]] bool ReadDeviceVec(common::AlignedResourceReadStream* fi, HostDeviceVector<T>* vec) {
  std::uint64_t n{0};
  if (!fi->Read(&n)) {
    return false;
  }
  if (n == 0) {
    return true;
  }

  auto expected_bytes = sizeof(T) * n;

  auto [ptr, n_bytes] = fi->Consume(expected_bytes);
  if (n_bytes != expected_bytes) {
    return false;
  }

  vec->Resize(n);
  auto d_vec = vec->DeviceSpan();
  dh::safe_cuda(
      cudaMemcpyAsync(d_vec.data(), ptr, n_bytes, cudaMemcpyDefault, dh::DefaultStream()));
  return true;
}
}  // namespace

[[nodiscard]] bool EllpackPageRawFormat::Read(EllpackPage* page,
                                              common::AlignedResourceReadStream* fi) {
  auto* impl = page->Impl();
  impl->SetCuts(this->cuts_);
  if (!fi->Read(&impl->n_rows)) {
    return false;
  }
  if (!fi->Read(&impl->is_dense)) {
    return false;
  }
  if (!fi->Read(&impl->row_stride)) {
    return false;
  }
  impl->gidx_buffer.SetDevice(device_);
  if (!ReadDeviceVec(fi, &impl->gidx_buffer)) {
    return false;
  }
  if (!fi->Read(&impl->base_rowid)) {
    return false;
  }
  return true;
}

[[nodiscard]] std::size_t EllpackPageRawFormat::Write(const EllpackPage& page,
                                                      common::AlignedFileWriteStream* fo) {
  std::size_t bytes{0};
  auto* impl = page.Impl();
  bytes += fo->Write(impl->n_rows);
  bytes += fo->Write(impl->is_dense);
  bytes += fo->Write(impl->row_stride);
  CHECK(!impl->gidx_buffer.ConstHostVector().empty());
  bytes += common::WriteVec(fo, impl->gidx_buffer.HostVector());
  bytes += fo->Write(impl->base_rowid);
  dh::DefaultStream().Sync();
  return bytes;
}

[[nodiscard]] bool EllpackPageRawFormat::Read(EllpackPage* page, EllpackHostCacheStream* fi) const {
  auto nvtxid = nvtxRangeStartA(__func__);
  auto* impl = page->Impl();
  CHECK(this->cuts_->cut_values_.DeviceCanRead());
  impl->SetCuts(this->cuts_);

  // Read vec
  auto read_vec = [&] {
    bst_idx_t n{0};
    if (!fi->Read(&n)) {
      return false;
    }
    if (n == 0) {
      return true;
    }

    impl->gidx_buffer.SetDevice(device_);
    impl->gidx_buffer.Resize(n);
    auto span = impl->gidx_buffer.DeviceSpan();
    if (!fi->Read(span.data(), span.size_bytes())) {
      return false;
    }
    return true;
  };

  if (!read_vec()) {
    return false;
  }

  if (!fi->Read(&impl->n_rows)) {
    return false;
  }
  if (!fi->Read(&impl->is_dense)) {
    return false;
  }
  if (!fi->Read(&impl->row_stride)) {
    return false;
  }
  if (!fi->Read(&impl->base_rowid)) {
    return false;
  }

  // dh::DefaultStream().Sync();
  fi->Sync();
  nvtxRangeEnd(nvtxid);
  return true;
}

[[nodiscard]] std::size_t EllpackPageRawFormat::Write(const EllpackPage& page,
                                                      EllpackHostCacheStream* fo) const {
  auto nvtxid = nvtxRangeStartA(__func__);
  bst_idx_t bytes{0};
  auto* impl = page.Impl();
  fo->Sync();
  // Write vector
  auto write_vec = [&] {
    bst_idx_t n = impl->gidx_buffer.Size();
    bytes += fo->Write(n);

    if (!impl->gidx_buffer.Empty()) {
      auto span = impl->gidx_buffer.ConstDeviceSpan();
      bytes += fo->Write(span.data(), span.size_bytes());
    }
  };

  auto wv_nvtx = nvtxRangeStartA("write_vec");
  write_vec();
  nvtxRangeEnd(wv_nvtx);

  bytes += fo->Write(impl->n_rows);
  bytes += fo->Write(impl->is_dense);
  bytes += fo->Write(impl->row_stride);
  bytes += fo->Write(impl->base_rowid);

  fo->Sync();
  nvtxRangeEnd(nvtxid);
  return bytes;
}
}  // namespace xgboost::data
