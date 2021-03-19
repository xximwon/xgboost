/*!
 * Copyright 2018~2020 XGBoost contributors
 */

#include <xgboost/logging.h>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "device_helpers.cuh"
#include "hist_util.h"
#include "hist_util.cuh"
#include "math.h"  // NOLINT
#include "quantile.h"
#include "categorical.h"
#include "xgboost/host_device_vector.h"


namespace xgboost {
namespace common {

constexpr float SketchContainer::kFactor;

namespace detail {
size_t RequiredSampleCutsPerColumn(int max_bins_per_feature, size_t num_rows) {
  double eps = 1.0 / (WQSketch::kFactor * max_bins_per_feature);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(
      num_rows, eps, &dummy_nlevel, &num_cuts);
  return std::min(num_cuts, num_rows);
}

size_t RequiredSampleCuts(bst_row_t num_rows, bst_feature_t num_columns,
                          size_t max_bins_per_feature, size_t nnz) {
  auto per_column = RequiredSampleCutsPerColumn(max_bins_per_feature, num_rows);
  auto if_dense = num_columns * per_column;
  auto result = std::min(nnz, if_dense);
  return result;
}

size_t ConstantMemoryPerWindow(size_t num_rows, bst_feature_t num_columns,
                               size_t max_bins_per_feature, size_t nnz) {
  // 0. Allocate cut pointer in quantile container by increasing: n_columns + 1
  size_t total = (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  // 3. Allocate colomn size scan by increasing: n_columns + 1
  total += (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  // 4. Allocate cut pointer by increasing: n_columns + 1
  total += (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  total +=
      RequiredSampleCuts(num_rows, num_columns, max_bins_per_feature, nnz) *
      sizeof(SketchEntry);
  return total;
}

void SortByWeight(dh::device_vector<float>* weights,
                  dh::device_vector<Entry>* sorted_entries) {
  // Sort both entries and wegihts.
  dh::XGBDeviceAllocator<char> alloc;
  thrust::sort_by_key(thrust::cuda::par(alloc), sorted_entries->begin(),
                      sorted_entries->end(), weights->begin(),
                      detail::EntryCompareOp());

  // Scan weights
  dh::XGBCachingDeviceAllocator<char> caching;
  thrust::inclusive_scan_by_key(thrust::cuda::par(caching),
                                sorted_entries->begin(), sorted_entries->end(),
                                weights->begin(), weights->begin(),
                                [=] __device__(const Entry& a, const Entry& b) {
                                  return a.index == b.index;
                                });
}

struct IsCatOp {
  XGBOOST_DEVICE bool operator()(FeatureType ft) { return ft == FeatureType::kCategorical; }
};

void RemoveDuplicatedCategories(
    int32_t device, MetaInfo const &info, Span<bst_row_t> d_cuts_ptr,
    dh::device_vector<Entry> *p_sorted_entries,
    dh::caching_device_vector<size_t>* p_column_sizes_scan) {
  auto d_feature_types = info.feature_types.ConstDeviceSpan();
  auto& column_sizes_scan = *p_column_sizes_scan;
  if (!info.feature_types.Empty() &&
      thrust::any_of(dh::tbegin(d_feature_types), dh::tend(d_feature_types),
                     IsCatOp{})) {
    auto& sorted_entries = *p_sorted_entries;
    // Removing duplicated entries in categorical features.
    dh::caching_device_vector<size_t> new_column_scan(column_sizes_scan.size());
    dh::SegmentedUnique(
        column_sizes_scan.data().get(),
        column_sizes_scan.data().get() + column_sizes_scan.size(),
        sorted_entries.begin(), sorted_entries.end(),
        new_column_scan.data().get(), sorted_entries.begin(),
        [=] __device__(Entry const &l, Entry const &r) {
          if (l.index == r.index) {
            if (IsCat(d_feature_types, l.index)) {
              return l.fvalue == r.fvalue;
            }
          }
          return false;
        });

    // Renew the column scan and cut scan based on categorical data.
    auto d_old_column_sizes_scan = dh::ToSpan(column_sizes_scan);
    dh::caching_device_vector<SketchContainer::OffsetT> new_cuts_size(
        info.num_col_ + 1);
    auto d_new_cuts_size = dh::ToSpan(new_cuts_size);
    auto d_new_columns_ptr = dh::ToSpan(new_column_scan);
    CHECK_EQ(new_column_scan.size(), new_cuts_size.size());
    dh::LaunchN(device, new_column_scan.size(), [=] __device__(size_t idx) {
      d_old_column_sizes_scan[idx] = d_new_columns_ptr[idx];
      if (idx == d_new_columns_ptr.size() - 1) {
        return;
      }
      if (IsCat(d_feature_types, idx)) {
        // Cut size is the same as number of categories in input.
        d_new_cuts_size[idx] =
            d_new_columns_ptr[idx + 1] - d_new_columns_ptr[idx];
      } else {
        d_new_cuts_size[idx] = d_cuts_ptr[idx] - d_cuts_ptr[idx];
      }
    });
    // Turn size into ptr.
    thrust::exclusive_scan(thrust::device, new_cuts_size.cbegin(),
                           new_cuts_size.cend(), d_cuts_ptr.data());
  }
}

size_t EstimateBatchSize(size_t num_rows, bst_feature_t num_columns,
                         size_t max_bins_per_feature, size_t nnz,
                         size_t memory_limit, int32_t device, bool weighted) {
  // assuming dense cuts
  int64_t avail =
      memory_limit == 0 ? dh::AvailableMemory(device) : memory_limit;
  size_t lower_bound =
      ConstantMemoryPerWindow(num_rows, num_columns, max_bins_per_feature, nnz);
  CHECK_LT(static_cast<int64_t>(lower_bound), avail);
  int64_t a = static_cast<int64_t>(BytesPerElement(weighted));
  int64_t c = (-avail + static_cast<int64_t>(lower_bound));
  size_t n_elements = std::sqrt(-4 * a * c) / 2 * a;
  return n_elements;
}
}  // namespace detail

void ProcessBatch(int device, MetaInfo const &info, const SparsePage &page,
                  size_t begin, size_t end, SketchContainer *sketch_container,
                  int num_cuts_per_feature, size_t num_columns) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  const auto& host_data = page.data.ConstHostVector();
  dh::device_vector<Entry> sorted_entries(host_data.begin() + begin,
                                          host_data.begin() + end);
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
               sorted_entries.end(), detail::EntryCompareOp());

  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  dh::caching_device_vector<size_t> column_sizes_scan;
  data::IsValidFunctor dummy_is_valid(std::numeric_limits<float>::quiet_NaN());
  auto batch_it = dh::MakeTransformIterator<data::COOTuple>(
      sorted_entries.data().get(),
      [] __device__(Entry const &e) -> data::COOTuple {
        return {0, e.index, e.fvalue};  // row_idx is not needed for scanning column size.
      });
  detail::GetColumnSizesScan(device, num_columns, num_cuts_per_feature,
                             batch_it, dummy_is_valid,
                             0, sorted_entries.size(),
                             &cuts_ptr, &column_sizes_scan);
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  detail::RemoveDuplicatedCategories(device, info, d_cuts_ptr, &sorted_entries,
                                     &column_sizes_scan);

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  CHECK_EQ(d_cuts_ptr.size(), column_sizes_scan.size());

  // add cuts into sketches
  sketch_container->Push(dh::ToSpan(sorted_entries), dh::ToSpan(column_sizes_scan),
                         d_cuts_ptr, h_cuts_ptr.back());
  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
  CHECK_EQ(sorted_entries.capacity(), 0);
  CHECK_NE(cuts_ptr.Size(), 0);
}

void ProcessWeightedBatch(int device, const SparsePage& page,
                          MetaInfo const& info, size_t begin, size_t end,
                          SketchContainer* sketch_container, int num_cuts_per_feature,
                          size_t num_columns,
                          bool is_ranking, Span<bst_group_t const> d_group_ptr) {
  auto weights = info.weights_.ConstDeviceSpan();

  dh::XGBCachingDeviceAllocator<char> alloc;
  const auto& host_data = page.data.ConstHostVector();
  dh::device_vector<Entry> sorted_entries(host_data.begin() + begin,
                                          host_data.begin() + end);

  // Binary search to assign weights to each element
  dh::device_vector<float> temp_weights(sorted_entries.size());
  auto d_temp_weights = temp_weights.data().get();
  page.offset.SetDevice(device);
  auto row_ptrs = page.offset.ConstDeviceSpan();
  size_t base_rowid = page.base_rowid;
  if (is_ranking) {
    CHECK_GE(d_group_ptr.size(), 2)
        << "Must have at least 1 group for ranking.";
    CHECK_EQ(weights.size(), d_group_ptr.size() - 1)
        << "Weight size should equal to number of groups.";
    dh::LaunchN(device, temp_weights.size(), [=] __device__(size_t idx) {
        size_t element_idx = idx + begin;
        size_t ridx = dh::SegmentId(row_ptrs, element_idx);
        bst_group_t group_idx = dh::SegmentId(d_group_ptr, ridx + base_rowid);
        d_temp_weights[idx] = weights[group_idx];
      });
  } else {
    dh::LaunchN(device, temp_weights.size(), [=] __device__(size_t idx) {
        size_t element_idx = idx + begin;
        size_t ridx = dh::SegmentId(row_ptrs, element_idx);
        d_temp_weights[idx] = weights[ridx + base_rowid];
      });
  }
  detail::SortByWeight(&temp_weights, &sorted_entries);

  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  dh::caching_device_vector<size_t> column_sizes_scan;
  data::IsValidFunctor dummy_is_valid(std::numeric_limits<float>::quiet_NaN());
  auto batch_it = dh::MakeTransformIterator<data::COOTuple>(
      sorted_entries.data().get(),
      [] __device__(Entry const &e) -> data::COOTuple {
        return {0, e.index, e.fvalue};  // row_idx is not needed for scaning column size.
      });
  detail::GetColumnSizesScan(device, num_columns, num_cuts_per_feature,
                             batch_it, dummy_is_valid,
                             0, sorted_entries.size(),
                             &cuts_ptr, &column_sizes_scan);
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  detail::RemoveDuplicatedCategories(device, info, d_cuts_ptr, &sorted_entries,
                                     &column_sizes_scan);
  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();

  // Extract cuts
  sketch_container->Push(dh::ToSpan(sorted_entries),
                         dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back(), dh::ToSpan(temp_weights));
  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
}

HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins, size_t memory_limit) {
  dmat->Info().feature_types.SetDevice(device);
  dmat->Info().feature_types.ConstDevicePointer();  // pull to device early
  // Configure batch size based on available memory
  bool has_weights = dmat->Info().weights_.Size() > 0;
  size_t num_cuts_per_feature =
      detail::RequiredSampleCutsPerColumn(max_bins, dmat->Info().num_row_);

  HistogramCuts cuts;
  SketchContainer sketch_container(dmat->Info().feature_types, max_bins,
                                   dmat->Info().num_col_, dmat->Info().num_row_,
                                   device);

  dmat->Info().weights_.SetDevice(device);
  for (const auto &batch : dmat->GetBatches<SparsePage>()) {
    size_t batch_nnz = batch.data.Size();
    auto const &info = dmat->Info();

    size_t remaining = batch_nnz;
    size_t begin = 0;

    do {
      size_t sketch_batch_num_elements = detail::EstimateBatchSize(
          info.num_row_, info.num_col_, num_cuts_per_feature, info.num_nonzero_,
          memory_limit, device, has_weights);
      size_t end =
          std::min(batch_nnz, size_t(begin + sketch_batch_num_elements));

      if (has_weights) {
        bool is_ranking = HostSketchContainer::UseGroup(dmat->Info());
        dh::caching_device_vector<uint32_t> groups(info.group_ptr_.cbegin(),
                                                   info.group_ptr_.cend());
        ProcessWeightedBatch(device, batch, dmat->Info(), begin, end,
                             &sketch_container, num_cuts_per_feature,
                             dmat->Info().num_col_, is_ranking,
                             dh::ToSpan(groups));
      } else {
        ProcessBatch(device, dmat->Info(), batch, begin, end, &sketch_container,
                     num_cuts_per_feature, dmat->Info().num_col_);
      }
      remaining -= (end - begin);
      begin = end;
    } while (remaining > 0);
  }
  sketch_container.MakeCuts(&cuts);
  return cuts;
}
}  // namespace common
}  // namespace xgboost
