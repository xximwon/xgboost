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
EllpackHostCache::EllpackHostCache(bst_idx_t n_batches, double ratio, bool prefer_device,
                                   double max_cache_ratio)
    : total_available_mem{dh::TotalMemory(curt::CurrentDevice())},
      max_cache_page_ratio{ratio},
      n_batches_orig{n_batches},
      prefer_device{prefer_device},
      max_cache_ratio{max_cache_ratio} {};
EllpackHostCache::~EllpackHostCache() = default;

[[nodiscard]] std::size_t EllpackHostCache::SizeBytes() const {
  auto it = common::MakeIndexTransformIter([&](auto i) { return pages.at(i)->MemCostBytes(); });
  return std::accumulate(it, it + pages.size(), 0ul);
}

[[nodiscard]] std::size_t EllpackHostCache::DeviceSizeBytes() const {
  auto it = common::MakeIndexTransformIter(
      [&](auto i) -> bst_idx_t { return this->on_host.at(i) ? 0ul : pages.at(i)->MemCostBytes(); });
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
      if (this->cache_->sizes_orig.empty()) {
        n_bytes += cache_->pages[i]->MemCostBytes();
      } else {
        n_bytes += cache_->sizes_orig[i];
      }
    }
    if (offset_bytes == n_bytes && k == -1) {
      k = this->cache_->pages.size();  // seek end
    }
    CHECK_NE(k, -1) << "Invalid offset:" << offset_bytes;
    ptr_ = k;
  }
  // Whether we should copy the cache page during commit.
  [[nodiscard]] bool KeepCache() const {
    bool is_on_host = this->cache_->on_host.back();
    if (is_on_host) {
      return true;  // Don't copy
    }
    if (this->cache_->prefer_device) {
      auto avail = dh::AvailableMemory(dh::CurrentDevice());
      return this->cache_->DeviceSizeBytes() < (this->cache_->max_cache_ratio * avail);
    }
    return false;
  }
  [[nodiscard]] bool Write(EllpackPage const& page) {
    auto impl = page.Impl();

    // Have a new page
    auto new_impl = std::make_unique<EllpackPageImpl>();
    new_impl->CopyInfo(impl);

    auto ctx = Context{}.MakeCUDA(dh::CurrentDevice());

    if (this->cache_->pages.empty() || this->cache_->written.back()) {
      // First page, use host memory if there's no concatenation.
      if (this->cache_->max_cache_page_ratio == 0.0) {
        new_impl->gidx_buffer =
            common::MakeFixedVecWithPinnedMalloc<common::CompressedByteT>(impl->gidx_buffer.size());
        this->cache_->on_host.push_back(true);
      } else {
#if defined(XGBOOST_USE_RMM)
        new_impl->gidx_buffer =
            common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(impl->gidx_buffer.size());
#else
        new_impl->gidx_buffer =
            common::MakeCudaGrowOnly<common::CompressedByteT>(impl->gidx_buffer.size());
#endif
        this->cache_->on_host.push_back(false);
      }

      dh::safe_cuda(cudaMemcpyAsync(new_impl->gidx_buffer.data(), impl->gidx_buffer.data(),
                                    impl->gidx_buffer.size_bytes(), cudaMemcpyDefault));
      this->cache_->pages.emplace_back(std::move(new_impl));
      this->cache_->written.push_back(false);
    } else {
      CHECK(!this->cache_->on_host.back());
      CHECK(!this->cache_->written.back());
      std::cout << "this:" << this->cache_->pages.back()->gidx_buffer.size_bytes()
                << " that:" << impl->gidx_buffer.size_bytes() << std::endl;
      this->cache_->pages.back()->Extend(&ctx, impl);
    }
    this->cache_->sizes_orig.push_back(impl->MemCostBytes());

    auto AtEnd = [this] {
      return this->cache_->sizes_orig.size() == this->cache_->n_batches_orig;
    };
    auto ShouldCommit = [this, AtEnd] {
      auto max_page_size = this->cache_->total_available_mem * this->cache_->max_cache_page_ratio;
      bool page_full = this->cache_->pages.back()->MemCostBytes() >= max_page_size;
      bool at_end = AtEnd();
      return page_full || at_end;
    };

    // Move the page to the host memory if it's large enough
    auto at_end = AtEnd();
    bool commit = ShouldCommit();
    if (commit && !this->KeepCache()) {
      auto& orig = this->cache_->pages.back();

      auto full_page = std::make_unique<EllpackPageImpl>();
      full_page->CopyInfo(orig.get());
      full_page->gidx_buffer =
          common::MakeFixedVecWithPinnedMalloc<common::CompressedByteT>(orig->gidx_buffer.size());
      dh::safe_cuda(cudaMemcpyAsync(full_page->gidx_buffer.data(), orig->gidx_buffer.data(),
                                    orig->gidx_buffer.size_bytes(), cudaMemcpyDefault));
      this->cache_->pages.back().reset();
      this->cache_->pages.back() = std::move(full_page);
      this->cache_->on_host.back() = true;
    }
    if (commit) {
      this->cache_->written.back() = true;
    }
    if (at_end) {
      this->cache_->sizes_orig.clear();
    }

    CHECK_EQ(this->cache_->pages.size(), this->cache_->on_host.size());
    CHECK_EQ(this->cache_->pages.size(), this->cache_->written.size());
    return commit;
  }

  void Read(EllpackPage* out, bool prefetch_copy) const {
    auto page = this->cache_->At(ptr_);

    auto impl = out->Impl();
    CHECK(this->cache_->written.at(this->ptr_));
    if (prefetch_copy && this->cache_->on_host.at(this->ptr_)) {
#if defined(XGBOOST_USE_RMM)
      impl->gidx_buffer =
          common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(page->gidx_buffer.size());
#else
      impl->gidx_buffer =
          common::MakeCudaGrowOnly<common::CompressedByteT>(page->gidx_buffer.size());
#endif
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
    this->p_cache_ =
        std::make_shared<EllpackHostCache>(this->OrigBatches(), this->MaxCachePageRatio(),
                                           this->PreferDevice(), this->MaxCacheRatio());
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
