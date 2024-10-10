/**
 * Copyright 2019-2024, XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_

#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr
#include <utility>  // for move
#include <vector>   // for vector

#include "../common/cuda_rt_utils.h"  // for SupportsPageableMem, SupportsAts
#include "../common/hist_util.h"      // for HistogramCuts
#include "ellpack_page.h"             // for EllpackPage
#include "ellpack_page_raw_format.h"  // for EllpackPageRawFormat
#include "sparse_page_source.h"       // for PageSourceIncMixIn
#include "xgboost/base.h"             // for bst_idx_t
#include "xgboost/context.h"          // for DeviceOrd
#include "xgboost/data.h"             // for BatchParam
#include "xgboost/span.h"             // for Span

namespace xgboost::data {
// We need to decouple the storage and the view of the storage so that we can implement
// concurrent read. As a result, there are two classes, one for cache storage, another one
// for stream.
struct EllpackHostCache {
  std::size_t const total_available_mem;
  double const max_cache_page_ratio;
  bool const prefer_device;
  double const max_cache_ratio;

  std::vector<std::unique_ptr<EllpackPageImpl>> pages;
  std::vector<std::size_t> offsets;
  // Number of batches before concatenation.
  bst_idx_t const n_batches_orig;
  // Size of each batch before concatenation.
  std::vector<bst_idx_t> sizes_orig;
  std::vector<std::size_t> const cache_mapping;
  std::vector<std::size_t> const buffer_bytes;
  std::vector<std::size_t> const base_rows;
  std::vector<bst_idx_t> const buffer_rows;

  explicit EllpackHostCache(bst_idx_t n_batches, double ratio, bool prefer_device,
                            double max_cache_ratio, std::vector<std::size_t> cache_mapping,
                            std::vector<std::size_t> buffer_bytes,
                            std::vector<std::size_t> base_rows, std::vector<bst_idx_t> buffer_rows);
  ~EllpackHostCache();

  // The number of bytes for the entire cache.
  [[nodiscard]] std::size_t SizeBytes() const;

  bool Empty() const { return this->SizeBytes() == 0; }

  EllpackPageImpl const* At(std::int32_t k);
};

// Pimpl to hide CUDA calls from the host compiler.
class EllpackHostCacheStreamImpl;

/**
 * @brief A view of the actual cache implemented by `EllpackHostCache`.
 */
class EllpackHostCacheStream {
  std::unique_ptr<EllpackHostCacheStreamImpl> p_impl_;

 public:
  explicit EllpackHostCacheStream(std::shared_ptr<EllpackHostCache> cache);
  ~EllpackHostCacheStream();
  /**
   * @brief Get a shared handler to the cache.
   */
  std::shared_ptr<EllpackHostCache> Share();
  /**
   * @brief Stream seek.
   *
   * @param offset_bytes This must align to the actual cached page size.
   */
  void Seek(bst_idx_t offset_bytes);
  /**
   * @brief Read a page from the cache.
   *
   * The read page might be concatenated during page write.
   *
   * @param page[out] The returned page.
   * @param prefetch_copy[in] Does the stream need to copy the page?
   */
  void Read(EllpackPage* page, bool prefetch_copy) const;
  /**
   * @brief Add a new page to the host cache.
   *
   * This method might append the input page to a previously stored page to increase
   * individual page size.
   *
   * @return Whether a new cache page is create. False if the new page is appended to the
   * previous one.
   */
  [[nodiscard]] bool Write(EllpackPage const& page);
};

template <typename S>
class EllpackFormatPolicy {
  std::shared_ptr<common::HistogramCuts const> cuts_{nullptr};
  DeviceOrd device_;
  bool has_hmm_{curt::SupportsPageableMem()};
  bst_idx_t n_orig_batches_{0};
  double max_cache_page_ratio_{0};
  double max_cache_ratio_{0};
  bool prefer_device_{false};
  std::vector<std::size_t> cache_mapping_;
  std::vector<std::size_t> buffer_bytes_;
  std::vector<std::size_t> base_rows_;
  static_assert(std::is_same_v<S, EllpackPage>);

 public:
  using FormatT = EllpackPageRawFormat;

 public:
  EllpackFormatPolicy() {
    StringView msg{" The overhead of iterating through external memory might be significant."};
    if (!has_hmm_) {
      LOG(WARNING) << "CUDA heterogeneous memory management is not available." << msg;
    } else if (!curt::SupportsAts()) {
      LOG(WARNING) << "CUDA address translation service is not available." << msg;
    }
#if !defined(XGBOOST_USE_RMM)
    LOG(WARNING) << "XGBoost is not built with RMM support." << msg;
#endif
    if (!GlobalConfigThreadLocalStore::Get()->use_rmm) {
      LOG(WARNING) << "`use_rmm` is set to false." << msg;
    }
    std::int32_t major{0}, minor{0};
    curt::DrVersion(&major, &minor);
    if (!(major >= 12 && minor >= 7) && curt::SupportsAts()) {
      // Use ATS, but with an old kernel driver.
      LOG(WARNING) << "Using an old kernel driver with supported CTK<12.7."
                   << "The latest version of CTK supported by the current driver: " << major << "."
                   << minor << "." << msg;
    }
  }
  // For testing with the HMM flag.
  explicit EllpackFormatPolicy(bool has_hmm) : has_hmm_{has_hmm} {}

  [[nodiscard]] auto CreatePageFormat(BatchParam const& param) const {
    CHECK_EQ(cuts_->cut_values_.Device(), device_);
    std::unique_ptr<FormatT> fmt{new EllpackPageRawFormat{cuts_, device_, param, has_hmm_}};
    return fmt;
  }
  void SetCuts(std::shared_ptr<common::HistogramCuts const> cuts, DeviceOrd device,
               bst_idx_t n_batches, double ratio, bool prefer_device, double cache_ratio,
               std::vector<std::size_t> cache_mapping, std::vector<std::size_t> buffer_bytes,
               std::vector<std::size_t> base_rows) {
    std::swap(this->cuts_, cuts);
    this->device_ = device;
    CHECK(this->device_.IsCUDA());
    this->n_orig_batches_ = n_batches;
    this->max_cache_page_ratio_ = ratio;
    this->prefer_device_ = prefer_device;
    this->max_cache_ratio_ = cache_ratio;
    this->cache_mapping_ = std::move(cache_mapping);
    this->buffer_bytes_ = std::move(buffer_bytes);
    this->base_rows_ = std::move(base_rows);
  }
  [[nodiscard]] auto GetCuts() {
    CHECK(cuts_);
    return cuts_;
  }
  // Get the original number of batches from the input iterator
  [[nodiscard]] auto OrigBatches() const { return this->n_orig_batches_; }
  [[nodiscard]] auto MaxCachePageRatio() const { return this->max_cache_page_ratio_; }
  [[nodiscard]] auto MaxCacheRatio() const { return this->max_cache_ratio_; }
  [[nodiscard]] auto const& CacheMapping() const { return this->cache_mapping_; }
  [[nodiscard]] auto const& BufferBytes() const { return this->buffer_bytes_; }
  [[nodiscard]] auto const& BaseRows() const { return this->base_rows_; }

  [[nodiscard]] auto Device() const { return device_; }
  [[nodiscard]] auto PreferDevice() const { return prefer_device_; }
};

template <typename S, template <typename> typename F>
class EllpackCacheStreamPolicy : public F<S> {
  std::shared_ptr<EllpackHostCache> p_cache_;

 public:
  using WriterT = EllpackHostCacheStream;
  using ReaderT = EllpackHostCacheStream;

 public:
  EllpackCacheStreamPolicy() = default;
  [[nodiscard]] std::unique_ptr<WriterT> CreateWriter(StringView name, std::uint32_t iter);

  [[nodiscard]] std::unique_ptr<ReaderT> CreateReader(StringView name, bst_idx_t offset,
                                                      bst_idx_t length) const;
};

template <typename S, template <typename> typename F>
class EllpackMmapStreamPolicy : public F<S> {
  bool has_hmm_{curt::SupportsPageableMem()};

 public:
  using WriterT = common::AlignedFileWriteStream;
  using ReaderT = common::AlignedResourceReadStream;

 public:
  EllpackMmapStreamPolicy() = default;
  // For testing with the HMM flag.
  template <
      typename std::enable_if_t<std::is_same_v<F<S>, EllpackFormatPolicy<EllpackPage>>>* = nullptr>
  explicit EllpackMmapStreamPolicy(bool has_hmm) : F<S>{has_hmm}, has_hmm_{has_hmm} {}

  [[nodiscard]] std::unique_ptr<WriterT> CreateWriter(StringView name, std::uint32_t iter) {
    std::unique_ptr<common::AlignedFileWriteStream> fo;
    if (iter == 0) {
      fo = std::make_unique<common::AlignedFileWriteStream>(name, "wb");
    } else {
      fo = std::make_unique<common::AlignedFileWriteStream>(name, "ab");
    }
    return fo;
  }

  [[nodiscard]] std::unique_ptr<ReaderT> CreateReader(StringView name, bst_idx_t offset,
                                                      bst_idx_t length) const;
};

struct EllpackSourceConfig {
  BatchParam param;
  bool prefer_device;
  float missing;
  double max_cache_page_ratio;
  double max_cache_ratio;
  std::vector<std::size_t> cache_mapping;
  std::vector<std::size_t> buffer_bytes;
};

/**
 * @brief Ellpack source with sparse pages as the underlying source.
 */
template <typename F>
class EllpackPageSourceImpl : public PageSourceIncMixIn<EllpackPage, F> {
  using Super = PageSourceIncMixIn<EllpackPage, F>;
  bool is_dense_;
  bst_idx_t row_stride_;
  BatchParam param_;
  common::Span<FeatureType const> feature_types_;

 public:
  EllpackPageSourceImpl(std::int32_t nthreads, bst_feature_t n_features,
                        std::size_t n_batches, std::shared_ptr<Cache> cache,
                        std::shared_ptr<common::HistogramCuts> cuts, bool is_dense,
                        bst_idx_t row_stride, common::Span<FeatureType const> feature_types,
                        std::shared_ptr<SparsePageSource> source, DeviceOrd device,
                        EllpackSourceConfig const& config)
      : Super{config.missing, nthreads, n_features, n_batches, cache, false},
        is_dense_{is_dense},
        row_stride_{row_stride},
        param_{std::move(config.param)},
        feature_types_{feature_types} {
    this->source_ = source;
    cuts->SetDevice(device);
    this->SetCuts(std::move(cuts), device, n_batches, config.max_cache_page_ratio,
                  config.prefer_device, config.max_cache_ratio, config.cache_mapping,
                  config.buffer_bytes, {});
    this->Fetch();
  }

  void Fetch() final;
};

// Cache to host
using EllpackPageHostSource =
    EllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

// Cache to disk
using EllpackPageSource =
    EllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

/**
 * @brief Ellpack source directly interfaces with user-defined iterators.
 */
template <typename FormatCreatePolicy>
class ExtEllpackPageSourceImpl : public ExtQantileSourceMixin<EllpackPage, FormatCreatePolicy> {
  using Super = ExtQantileSourceMixin<EllpackPage, FormatCreatePolicy>;

  Context const* ctx_;
  BatchParam p_;
  DMatrixProxy* proxy_;
  MetaInfo* info_;
  ExternalDataInfo ext_info_;
  std::vector<std::size_t> cache_mapping_;

 public:
  ExtEllpackPageSourceImpl(
      Context const* ctx, MetaInfo* info, ExternalDataInfo ext_info, std::shared_ptr<Cache> cache,
      std::shared_ptr<common::HistogramCuts> cuts,
      std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> source,
      DMatrixProxy* proxy, EllpackSourceConfig const& config)
      : Super{config.missing, ctx->Threads(), static_cast<bst_feature_t>(info->num_col_), source,
              cache},
        ctx_{ctx},
        p_{std::move(config.param)},
        proxy_{proxy},
        info_{info},
        ext_info_{std::move(ext_info)},
        cache_mapping_{config.cache_mapping} {
    cuts->SetDevice(ctx->Device());
    this->SetCuts(std::move(cuts), ctx->Device(), ext_info.n_batches, config.max_cache_page_ratio,
                  config.prefer_device, config.max_cache_ratio, config.cache_mapping,
                  config.buffer_bytes, ext_info_.base_rows);
    CHECK(!this->cache_info_->written);
    this->source_->Reset();
    CHECK(this->source_->Next());
    this->Fetch();
  }

  void Fetch() final;
  // Need a specialized end iter as we can concatenate pages.
  void EndIter() final {
    if (this->cache_info_->written) {
      CHECK_EQ(this->Iter(), this->cache_info_->Size());
    } else {
      CHECK_LE(this->cache_info_->Size(), this->ext_info_.n_batches);
    }
    this->cache_info_->Commit();
    CHECK_GE(this->count_, 1);
    this->count_ = 0;
  }
};

// Cache to host
using ExtEllpackPageHostSource =
    ExtEllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

// Cache to disk
using ExtEllpackPageSource =
    ExtEllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

#if !defined(XGBOOST_USE_CUDA)
template <typename F>
inline void EllpackPageSourceImpl<F>::Fetch() {
  // silent the warning about unused variables.
  (void)(row_stride_);
  (void)(is_dense_);
  common::AssertGPUSupport();
}

template <typename F>
inline void ExtEllpackPageSourceImpl<F>::Fetch() {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data

#endif  // XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
