/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <sys/sysinfo.h>         // for sysinfo

#include <cstddef>  // for size_t
#include <cstdint>  // for int8_t, uint64_t, uint32_t
#include <memory>   // for shared_ptr, make_unique, make_shared
#include <numeric>  // for accumulate
#include <utility>  // for move

#include "../common/common.h"               // for safe_cuda
#include "../common/common.h"               // for HumanMemUnit
#include "../common/cuda_rt_utils.h"        // for SetDevice
#include "../common/device_helpers.cuh"     // for CUDAStreamView, DefaultStream
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/resource.cuh"           // for PrivateCudaMmapConstStream
#include "../common/transform_iterator.h"   // for MakeIndexTransformIter
#include "ellpack_page.cuh"                 // for EllpackPageImpl
#include "ellpack_page.h"                   // for EllpackPage
#include "ellpack_page_source.h"
#include "proxy_dmatrix.cuh"            // for Dispatch
#include "xgboost/base.h"               // for bst_idx_t
#include "xgboost/collective/socket.h"  // fixme

namespace xgboost::data {

#if !defined(xgboost_CHECK_SYS_CALL)
#define xgboost_CHECK_SYS_CALL(exp, expected)         \
  do {                                                \
    if (XGBOOST_EXPECT((exp) != (expected), false)) { \
      ::xgboost::system::ThrowAtError(#exp);          \
    }                                                 \
  } while (false)
#endif  // !defined(xgboost_CHECK_SYS_CALL)

[[nodiscard]] std::size_t AvailableHostMemory() {
  struct sysinfo info;
  xgboost_CHECK_SYS_CALL(sysinfo(&info), 0);
  return info.freeram;
}

/**
 * Cache
 */
EllpackHostCache::EllpackHostCache(bst_idx_t n_batches, bool prefer_device,
                                   std::vector<std::size_t> cache_mapping,
                                   std::vector<std::size_t> buffer_bytes,
                                   std::vector<std::size_t> base_rows,
                                   std::vector<bst_idx_t> buffer_rows)
    : total_available_mem{dh::TotalMemory(curt::CurrentDevice())},
      n_batches_orig{n_batches},
      prefer_device{prefer_device},
      cache_mapping{std::move(cache_mapping)},
      buffer_bytes{std::move(buffer_bytes)},
      base_rows{std::move(base_rows)},
      buffer_rows{std::move(buffer_rows)} {};

EllpackHostCache::~EllpackHostCache() = default;

[[nodiscard]] std::size_t EllpackHostCache::SizeBytes() const {
  auto it = common::MakeIndexTransformIter([&](auto i) { return pages.at(i)->MemCostBytes(); });
  return std::accumulate(it, it + pages.size(), 0ul);
}

EllpackPageImpl const* EllpackHostCache::At(std::int32_t k) {
  return this->pages.at(k).get();
}

/**
 * Cache stream.
 */
class EllpackHostCacheStreamImpl {
  std::shared_ptr<EllpackHostCache> cache_;
  std::int32_t ptr_{0};

 public:
  explicit EllpackHostCacheStreamImpl(std::shared_ptr<EllpackHostCache> cache)
      : cache_{std::move(cache)} {}

  auto Share() { return cache_; }

  void Seek(bst_idx_t offset_bytes) {
    std::size_t n_bytes{0};
    std::int32_t k{-1};
    for (std::size_t i = 0, n = cache_->pages.size(); i < n; ++i) {
      if (n_bytes == offset_bytes) {
        k = i;
        break;
      }
      n_bytes += cache_->pages[i]->MemCostBytes();
    }
    if (offset_bytes == n_bytes && k == -1) {
      k = this->cache_->pages.size();  // seek end
    }
    CHECK_NE(k, -1) << "Invalid offset:" << offset_bytes;
    ptr_ = k;
  }

  [[nodiscard]] bool Write(EllpackPage const& page) {
    auto impl = page.Impl();

    auto ctx = Context{}.MakeCUDA(dh::CurrentDevice());

    this->cache_->sizes_orig.push_back(page.Impl()->MemCostBytes());
    auto orig_ptr = this->cache_->sizes_orig.size() - 1;

    auto cache_idx = this->cache_->cache_mapping.at(orig_ptr);
    // Wrap up the previous page if this is a new page, or this is the last page.
    auto new_page = cache_idx == this->cache_->pages.size();
    auto last_page = (orig_ptr + 1) == this->cache_->n_batches_orig;

    auto commit_page = [this, &ctx] {
      auto const& d_page = this->cache_->pages.back();

      auto& old_impl = this->cache_->pages.back();
      CHECK_EQ(old_impl->gidx_buffer.Resource()->Type(), common::ResourceHandler::kCudaMalloc);

      auto new_impl = std::make_unique<EllpackPageImpl>();
      new_impl->CopyInfo(d_page.get());
      new_impl->gidx_buffer = common::MakeFixedVecWithPinnedMalloc<common::CompressedByteT>(
          old_impl->gidx_buffer.size());
      dh::safe_cuda(cudaMemcpyAsync(new_impl->gidx_buffer.data(), old_impl->gidx_buffer.data(),
                                    old_impl->gidx_buffer.size_bytes(), cudaMemcpyDefault));

      old_impl.reset();
      this->cache_->pages.back() = std::move(new_impl);
      LOG(INFO) << "Create cache page with size:"
                << common::HumanMemUnit(this->cache_->pages.back()->MemCostBytes());
    };

    if (new_page) {
      if (!this->cache_->pages.empty()) {
        // New to wrap up the previous page.
        commit_page();
      }
      // Push a new page
      auto n_bytes = this->cache_->buffer_bytes.at(this->cache_->pages.size());
      auto n_samples = this->cache_->buffer_rows.at(this->cache_->pages.size());
      auto new_impl = std::make_unique<EllpackPageImpl>(&ctx, impl->CutsShared(), impl->IsDense(),
                                                        impl->info.row_stride, n_samples);
      new_impl->base_rowid = this->cache_->base_rows.at(orig_ptr);
      CHECK_EQ(this->cache_->base_rows.at(orig_ptr), impl->base_rowid);
      new_impl->SetNumSymbols(impl->NumSymbols());
      new_impl->gidx_buffer = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(n_bytes);
      dh::safe_cuda(cudaMemsetAsync(new_impl->gidx_buffer.data(), '\0',
                                    new_impl->gidx_buffer.size_bytes(), ctx.CUDACtx()->Stream()));
      auto offset = new_impl->Copy(&ctx, impl, 0);

      this->cache_->offsets.push_back(offset);
      this->cache_->pages.push_back(std::move(new_impl));
    } else {
      CHECK(!this->cache_->pages.empty());
      CHECK_EQ(cache_idx, this->cache_->pages.size() - 1);
      auto& new_impl = this->cache_->pages.back();
      auto offset = new_impl->Copy(&ctx, impl, this->cache_->offsets.back());
      this->cache_->offsets.back() += offset;
      if (last_page) {
        commit_page();
      }
    }

    return new_page;
  }

  void Read(EllpackPage* out, bool prefetch_copy) const {
    auto page = this->cache_->At(ptr_);

    auto impl = out->Impl();
    if (prefetch_copy) {
      impl->gidx_buffer =
          common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(page->gidx_buffer.size());
      dh::safe_cuda(cudaMemcpyAsync(impl->gidx_buffer.data(), page->gidx_buffer.data(),
                                    page->gidx_buffer.size_bytes(), cudaMemcpyDefault));
    } else {
      auto res = page->gidx_buffer.Resource();
      impl->gidx_buffer = common::RefResourceView<common::CompressedByteT>{
          res->DataAs<common::CompressedByteT>(), page->gidx_buffer.size(), res};
    }

    impl->CopyInfo(page);
  }
};

/**
 * EllpackHostCacheStream
 */
EllpackHostCacheStream::EllpackHostCacheStream(std::shared_ptr<EllpackHostCache> cache)
    : p_impl_{std::make_unique<EllpackHostCacheStreamImpl>(std::move(cache))} {}

EllpackHostCacheStream::~EllpackHostCacheStream() = default;

std::shared_ptr<EllpackHostCache> EllpackHostCacheStream::Share() { return p_impl_->Share(); }

void EllpackHostCacheStream::Seek(bst_idx_t offset_bytes) { this->p_impl_->Seek(offset_bytes); }

void EllpackHostCacheStream::Read(EllpackPage* page, bool prefetch_copy) const {
  this->p_impl_->Read(page, prefetch_copy);
}

[[nodiscard]] bool EllpackHostCacheStream::Write(EllpackPage const& page) {
  return this->p_impl_->Write(page);
}

/**
 * EllpackCacheStreamPolicy
 */
template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackCacheStreamPolicy<S, F>::WriterT>
EllpackCacheStreamPolicy<S, F>::CreateWriter(StringView, std::uint32_t iter) {
  if (!this->p_cache_) {
    this->p_cache_ = std::make_shared<EllpackHostCache>(this->OrigBatches(), this->PreferDevice(),
                                                        this->CacheMapping(), this->BufferBytes(),
                                                        this->BaseRows(), this->BufferRows());
  }
  auto fo = std::make_unique<EllpackHostCacheStream>(this->p_cache_);
  if (iter == 0) {
    CHECK(this->p_cache_->Empty());
  } else {
    fo->Seek(this->p_cache_->SizeBytes());
  }
  return fo;
}

template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackCacheStreamPolicy<S, F>::ReaderT>
EllpackCacheStreamPolicy<S, F>::CreateReader(StringView, bst_idx_t offset, bst_idx_t) const {
  auto fi = std::make_unique<ReaderT>(this->p_cache_);
  fi->Seek(offset);
  return fi;
}

// Instantiation
template std::unique_ptr<
    typename EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::WriterT>
EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateWriter(StringView name,
                                                                         std::uint32_t iter);

template std::unique_ptr<
    typename EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::ReaderT>
EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateReader(StringView name,
                                                                         bst_idx_t offset,
                                                                         bst_idx_t length) const;

/**
 * EllpackMmapStreamPolicy
 */

template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackMmapStreamPolicy<S, F>::ReaderT>
EllpackMmapStreamPolicy<S, F>::CreateReader(StringView name, bst_idx_t offset,
                                            bst_idx_t length) const {
  if (has_hmm_) {
    return std::make_unique<common::PrivateCudaMmapConstStream>(name, offset, length);
  } else {
    return std::make_unique<common::PrivateMmapConstStream>(name, offset, length);
  }
}

// Instantiation
template std::unique_ptr<
    typename EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>::ReaderT>
EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateReader(StringView name,
                                                                        bst_idx_t offset,
                                                                        bst_idx_t length) const;

void CalcCachInfo(Context const* ctx, bool is_dense,
                  std::shared_ptr<common::HistogramCuts const> cuts, double min_page_bytes,
                  ExternalDataInfo const& ext_info) {
  auto ell_info = CalcNumSymbols(ctx, ext_info.row_stride, is_dense, cuts);
  std::vector<std::size_t> cache_size;
  CHECK_EQ(ext_info.base_rows.size(), ext_info.n_batches + 1);
  std::vector<std::size_t> cache_mapping(ext_info.base_rows.size(), 0);
  std::vector<std::size_t> cache_rows;

  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    auto n_samples = ext_info.base_rows.at(i + 1) - ext_info.base_rows[i];
    auto n_bytes = common::CompressedBufferWriter::CalculateBufferSize(
        ext_info.row_stride * n_samples, ell_info.n_symbols);
    if (cache_size.empty()) {
      cache_size.push_back(n_bytes);
      cache_rows.push_back(n_samples);
    } else if (cache_size.back() < min_page_bytes) {
      cache_size.back() += n_bytes;
      cache_rows.back() += n_samples;
    } else {
      cache_size.push_back(n_bytes);
      cache_rows.push_back(n_samples);
    }
    cache_mapping[i] = cache_size.size() - 1;
  }
}

/**
 * EllpackPageSourceImpl
 */
template <typename F>
void EllpackPageSourceImpl<F>::Fetch() {
  curt::SetDevice(this->Device().ordinal);
  if (!this->ReadCache()) {
    if (this->Iter() != 0 && !this->sync_) {
      // source is initialized to be the 0th page during construction, so when count_ is 0
      // there's no need to increment the source.
      ++(*this->source_);
    }
    // This is not read from cache so we still need it to be synced with sparse page source.
    CHECK_EQ(this->Iter(), this->source_->Iter());
    auto const& csr = this->source_->Page();
    this->page_.reset(new EllpackPage{});
    auto* impl = this->page_->Impl();
    Context ctx = Context{}.MakeCUDA(this->Device().ordinal);
    *impl = EllpackPageImpl{&ctx, this->GetCuts(), *csr, is_dense_, row_stride_, feature_types_};
    this->page_->SetBaseRowId(csr->base_rowid);
    LOG(INFO) << "Generated an Ellpack page with size: "
              << common::HumanMemUnit(impl->MemCostBytes())
              << " from a SparsePage with size:" << common::HumanMemUnit(csr->MemCostBytes());
    this->WriteCache();
  }
}

// Instantiation
template void
EllpackPageSourceImpl<DefaultFormatStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
EllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
EllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();

/**
 * ExtEllpackPageSourceImpl
 */
template <typename F>
void ExtEllpackPageSourceImpl<F>::Fetch() {
  curt::SetDevice(this->Device().ordinal);
  if (!this->ReadCache()) {
    auto iter = this->source_->Iter();
    CHECK_EQ(this->Iter(), iter);
    cuda_impl::Dispatch(proxy_, [this](auto const& value) {
      CHECK(this->proxy_->Ctx()->IsCUDA()) << "All batches must use the same device type.";
      proxy_->Info().feature_types.SetDevice(dh::GetDevice(this->ctx_));
      auto d_feature_types = proxy_->Info().feature_types.ConstDeviceSpan();
      auto n_samples = value.NumRows();

      dh::device_vector<size_t> row_counts(n_samples + 1, 0);
      common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
      GetRowCounts(this->ctx_, value, row_counts_span, dh::GetDevice(this->ctx_), this->missing_);
      this->page_.reset(new EllpackPage{});
      *this->page_->Impl() = EllpackPageImpl{this->ctx_,
                                             value,
                                             this->missing_,
                                             this->info_->IsDense(),
                                             row_counts_span,
                                             d_feature_types,
                                             this->ext_info_.row_stride,
                                             n_samples,
                                             this->GetCuts()};
      this->info_->Extend(proxy_->Info(), false, true);
    });
    // The size of ellpack is logged in write cache.
    LOG(INFO) << "Estimated batch size:"
              << cuda_impl::Dispatch<false>(proxy_, [](auto const& adapter) {
                   return common::HumanMemUnit(adapter->SizeBytes());
                 });
    this->page_->SetBaseRowId(this->ext_info_.base_rows.at(iter));
    this->WriteCache();
  }
}

// Instantiation
template void
ExtEllpackPageSourceImpl<DefaultFormatStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
ExtEllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
ExtEllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
}  // namespace xgboost::data
