/**
 * Copyright 2024, XGBoost Contributors
 */
#include <memory>   // for shared_ptr
#include <variant>  // for visit

#include "../common/cuda_rt_utils.h"  // for xgboost_NVTX_FN_RANGE
#include "batch_utils.h"              // for CheckParam, RegenGHist
#include "ellpack_page.cuh"           // for EllpackPage
#include "extmem_quantile_dmatrix.h"
#include "proxy_dmatrix.h"    // for DataIterProxy
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for BatchParam
#include "../common/cuda_rt_utils.h"

namespace xgboost::data {
void ExtMemQuantileDMatrix::InitFromCUDA(
    Context const *ctx,
    std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
    DMatrixHandle proxy_handle, BatchParam const &p, float missing, std::shared_ptr<DMatrix> ref) {
  xgboost_NVTX_FN_RANGE();

  // A handle passed to external iterator.
  auto proxy = MakeProxy(proxy_handle);
  CHECK(proxy);

  /**
   * Generate quantiles
   */
  auto cuts = std::make_shared<common::HistogramCuts>();
  ExternalDataInfo ext_info;
  cuda_impl::MakeSketches(ctx, iter.get(), proxy, ref, p, missing, cuts, this->Info(), &ext_info);
  ext_info.SetInfo(ctx, &this->info_);

  /**
   * Calculate cache info
   */
  auto ell_info = CalcNumSymbols(ctx, ext_info.row_stride, this->Info().IsDense(), cuts);
  // FIXME: This can be done in the source
  std::vector<std::size_t> cache_size;
  CHECK_EQ(ext_info.base_rows.size(), ext_info.n_batches + 1);
  std::vector<std::size_t> cache_mapping(ext_info.base_rows.size(), 0);
  std::vector<std::size_t> cache_rows;
  auto n_total_bytes = dh::TotalMemory(curt::CurrentDevice());
  auto page_bytes = n_total_bytes * cuda_impl::CachePageRatio();

  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    auto n_samples = ext_info.base_rows.at(i + 1) - ext_info.base_rows[i];
    auto n_bytes = common::CompressedBufferWriter::CalculateBufferSize(
        ext_info.row_stride * n_samples, ell_info.n_symbols);
    if (cache_size.empty()) {
      cache_size.push_back(n_bytes);
      cache_rows.push_back(n_samples);
    } else if (cache_size.back() < page_bytes) {
      cache_size.back() += n_bytes;
      cache_rows.back() += n_samples;
    } else {
      cache_size.push_back(n_bytes);
      cache_rows.push_back(n_samples);
    }
    cache_mapping[i] = cache_size.size() - 1;
  }

  auto n_batches = cache_size.size();
  std::cout << "New n_batches:" << n_batches << std::endl;

  /**
   * Generate gradient index
   */
  auto id = MakeCache(this, ".ellpack.page", this->on_host_, cache_prefix_, &cache_info_);
  if (on_host_ && std::get_if<EllpackHostPtr>(&ellpack_page_source_) == nullptr) {
    ellpack_page_source_.emplace<EllpackHostPtr>(nullptr);
  }

  std::visit(
      [&](auto &&ptr) {
        using SourceT = typename std::remove_reference_t<decltype(ptr)>::element_type;
        // We can't hide the data load overhead for inference. Prefer device cache for
        // validation datasets.
        auto config = EllpackSourceConfig{.param = p,
                                          .prefer_device = (ref != nullptr),
                                          .missing = missing,
                                          .max_cache_page_ratio = this->max_cache_page_ratio_,
                                          .max_cache_ratio = this->max_device_cache_ratio_,
                                          .cache_mapping = cache_mapping,
                                          .buffer_bytes = cache_size,
                                          .buffer_rows = cache_rows};
        ptr = std::make_shared<SourceT>(ctx, &this->Info(), ext_info, cache_info_.at(id), cuts,
                                        iter, proxy, config);
      },
      ellpack_page_source_);

  /**
   * Force initialize the cache and do some sanity checks along the way
   */
  bst_idx_t batch_cnt = 0, k = 0;
  bst_idx_t n_total_samples = 0;
  for (auto const &page : this->GetEllpackPageImpl()) {
    n_total_samples += page.Size();
    CHECK_EQ(page.Impl()->base_rowid, ext_info.base_rows[k]);
    CHECK_EQ(page.Impl()->info.row_stride, ext_info.row_stride);
    ++k, ++batch_cnt;
  }
  CHECK_EQ(batch_cnt, ext_info.n_batches);
  CHECK_EQ(n_total_samples, ext_info.accumulated_rows);

  CHECK_EQ(this->cache_info_.at(id)->Size(), n_batches);
  this->n_batches_ = this->cache_info_.at(id)->Size();
}

[[nodiscard]] BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackPageImpl() {
  auto batch_set =
      std::visit([this](auto &&ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                 this->ellpack_page_source_);
  return batch_set;
}

BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackBatches(Context const *,
                                                               const BatchParam &param) {
  if (param.Initialized()) {
    detail::CheckParam(this->batch_, param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }

  std::visit(
      [this, param](auto &&ptr) {
        CHECK(ptr)
            << "The `ExtMemQuantileDMatrix` is initialized using CPU data, cannot be used for GPU.";
        ptr->Reset(param);
      },
      this->ellpack_page_source_);

  return this->GetEllpackPageImpl();
}
}  // namespace xgboost::data
