/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include "evaluate_splits.cuh"
#include <limits>

namespace xgboost {
namespace tree {

// With constraints
template <typename GradientPairT>
XGBOOST_DEVICE float
LossChangeMissing(const GradientPairT &scan, const GradientPairT &missing,
                  const GradientPairT &parent_sum,
                  const GPUTrainingParam &param,
                  bst_node_t nidx,
                  bst_feature_t fidx,
                  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                  bool &missing_left_out) { // NOLINT
  float parent_gain = CalcGain(param, parent_sum);
  float missing_left_gain =
      evaluator.CalcSplitGain(param, nidx, fidx, GradStats(scan + missing),
                              GradStats(parent_sum - (scan + missing)));
  float missing_right_gain = evaluator.CalcSplitGain(
      param, nidx, fidx, GradStats(scan), GradStats(parent_sum - scan));

  if (missing_left_gain >= missing_right_gain) {
    missing_left_out = true;
    return missing_left_gain - parent_gain;
  } else {
    missing_left_out = false;
    return missing_right_gain - parent_gain;
  }
}

/*!
 * \brief
 *
 * \tparam ReduceT     BlockReduce Type.
 * \tparam TempStorage Cub Shared memory
 *
 * \param begin
 * \param end
 * \param temp_storage Shared memory for intermediate result.
 */
template <int BLOCK_THREADS, typename ReduceT, typename TempStorageT,
          typename GradientSumT>
__device__ GradientSumT
ReduceFeature(common::Span<const GradientSumT> feature_histogram,
              TempStorageT* temp_storage) {
  __shared__ cub::Uninitialized<GradientSumT> uninitialized_sum;
  GradientSumT& shared_sum = uninitialized_sum.Alias();

  GradientSumT local_sum = GradientSumT();
  // For loop sums features into one block size
  auto begin = feature_histogram.data();
  auto end = begin + feature_histogram.size();
  for (auto itr = begin; itr < end; itr += BLOCK_THREADS) {
    bool thread_active = itr + threadIdx.x < end;
    // Scan histogram
    GradientSumT bin = thread_active ? *(itr + threadIdx.x) : GradientSumT();
    local_sum += bin;
  }
  local_sum = ReduceT(temp_storage->sum_reduce).Reduce(local_sum, cub::Sum());
  // Reduction result is stored in thread 0.
  if (threadIdx.x == 0) {
    shared_sum = local_sum;
  }
  __syncthreads();
  return shared_sum;
}

/*! \brief Find the thread with best gain. */
template <int BLOCK_THREADS, typename ReduceT, typename ScanT,
          typename MaxReduceT, typename TempStorageT, typename GradientSumT>
__device__ void EvaluateFeature(
    int fidx, EvaluateSplitInputs<GradientSumT> inputs,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    DeviceSplitCandidate* best_split,  // shared memory storing best split
    TempStorageT* temp_storage         // temp memory for cub operations
) {
  // Use pointer from cut to indicate begin and end of bins for each feature.
  uint32_t gidx_begin = inputs.feature_segments[fidx];  // begining bin
  uint32_t gidx_end =
      inputs.feature_segments[fidx + 1];  // end bin for i^th feature

  // Sum histogram bins for current feature
  GradientSumT const feature_sum =
      ReduceFeature<BLOCK_THREADS, ReduceT, TempStorageT, GradientSumT>(
          inputs.gradient_histogram.subspan(gidx_begin, gidx_end - gidx_begin),
          temp_storage);

  GradientSumT const missing = inputs.parent_sum - feature_sum;
  float const null_gain = -std::numeric_limits<bst_float>::infinity();

  SumCallbackOp<GradientSumT> prefix_op = SumCallbackOp<GradientSumT>();
  for (int scan_begin = gidx_begin; scan_begin < gidx_end;
       scan_begin += BLOCK_THREADS) {
    bool thread_active = (scan_begin + threadIdx.x) < gidx_end;

    // Gradient value for current bin.
    GradientSumT bin = thread_active
                           ? inputs.gradient_histogram[scan_begin + threadIdx.x]
                           : GradientSumT();
    ScanT(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

    // Whether the gradient of missing values is put to the left side.
    bool missing_left = true;
    float gain = null_gain;
    if (thread_active) {
      gain = LossChangeMissing(bin, missing, inputs.parent_sum, inputs.param,
                               inputs.nidx,
                               fidx,
                               evaluator,
                               missing_left);
    }

    __syncthreads();

    // Find thread with best gain
    cub::KeyValuePair<int, float> tuple(threadIdx.x, gain);
    cub::KeyValuePair<int, float> best =
        MaxReduceT(temp_storage->max_reduce).Reduce(tuple, cub::ArgMax());

    __shared__ cub::KeyValuePair<int, float> block_max;
    if (threadIdx.x == 0) {
      block_max = best;
    }

    __syncthreads();

    // Best thread updates split
    if (threadIdx.x == block_max.key) {
      int split_gidx = (scan_begin + threadIdx.x) - 1;
      float fvalue;
      if (split_gidx < static_cast<int>(gidx_begin)) {
        fvalue = inputs.min_fvalue[fidx];
      } else {
        fvalue = inputs.feature_values[split_gidx];
      }
      GradientSumT left = missing_left ? bin + missing : bin;
      GradientSumT right = inputs.parent_sum - left;
      best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue,
                         fidx, GradientPair(left), GradientPair(right),
                         inputs.param);
    }
    __syncthreads();
  }
}

template <int BLOCK_THREADS, typename GradientSumT>
__global__ void EvaluateSplitsKernel(
    EvaluateSplitInputs<GradientSumT> left,
    EvaluateSplitInputs<GradientSumT> right,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    common::Span<DeviceSplitCandidate> out_candidates) {
  // KeyValuePair here used as threadIdx.x -> gain_value
  using ArgMaxT = cub::KeyValuePair<int, float>;
  using BlockScanT =
      cub::BlockScan<GradientSumT, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>;
  using MaxReduceT = cub::BlockReduce<ArgMaxT, BLOCK_THREADS>;

  using SumReduceT = cub::BlockReduce<GradientSumT, BLOCK_THREADS>;

  union TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  // Aligned && shared storage for best_split
  __shared__ cub::Uninitialized<DeviceSplitCandidate> uninitialized_split;
  DeviceSplitCandidate& best_split = uninitialized_split.Alias();
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    best_split = DeviceSplitCandidate();
  }

  __syncthreads();

  // If this block is working on the left or right node
  bool is_left = blockIdx.x < left.feature_set.size();
  EvaluateSplitInputs<GradientSumT>& inputs = is_left ? left : right;

  // One block for each feature. Features are sampled, so fidx != blockIdx.x
  int fidx = inputs.feature_set[is_left ? blockIdx.x
                                        : blockIdx.x - left.feature_set.size()];

  EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT>(
      fidx, inputs, evaluator, &best_split, &temp_storage);

  __syncthreads();

  if (threadIdx.x == 0) {
    // Record best loss for each feature
    out_candidates[blockIdx.x] = best_split;
  }
}

__device__ DeviceSplitCandidate operator+(const DeviceSplitCandidate& a,
                                          const DeviceSplitCandidate& b) {
  return b.loss_chg > a.loss_chg ? b : a;
}

template <typename T>
void PrintDeviceSpan(common::Span<T> values, std::string name = "") {
  using V = std::remove_cv_t<T>;
  std::vector<V> h_left_histogram(values.size());
  dh::CopyDeviceSpanToVector(&h_left_histogram, values);
  std::cout << name << ": " << std::endl;
  for (auto v : h_left_histogram) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
}

template <typename GradientSumT>
size_t EvaluateNode(TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                    EvaluateSplitInputs<GradientSumT> input,
                    common::Span<GradientSumT> d_left_histogram,
                    common::Span<float> d_gain_histogram) {
  auto d_columns_ptr = input.feature_segments;
  auto scan_it = dh::MakeTransformIterator<size_t>(
      thrust::make_counting_iterator(0ul),
      [=] __device__(size_t idx) { return dh::SegmentId(d_columns_ptr, idx); });
  using Tuple = thrust::tuple<size_t, GradientSumT>;
  auto val_it = thrust::make_zip_iterator(thrust::make_tuple(
      thrust::make_counting_iterator(0ul), dh::tbegin(input.gradient_histogram)));
  dh::caching_device_vector<Tuple> histogram(input.gradient_histogram.size());
  thrust::fill(histogram.begin(), histogram.end(), Tuple{0, GradientSumT{}});
  thrust::inclusive_scan_by_key(
      thrust::device, scan_it, scan_it + input.gradient_histogram.size(),
      val_it, histogram.begin(), thrust::equal_to<size_t>(),
      [=] __device__(Tuple const &l, Tuple const &r) -> Tuple {
        auto left_sum = thrust::get<1>(l);
        auto right_val = thrust::get<1>(r);
        size_t idx = thrust::get<0>(l);
        size_t columnd_id = dh::SegmentId(d_columns_ptr, idx);
        auto gain = evaluator.CalcSplitGain(
            input.param, input.nidx, columnd_id, GradStats(left_sum),
            GradStats(input.parent_sum - left_sum));
        d_gain_histogram[idx] = gain;
        if (idx == 0) {
          d_gain_histogram[0] = input.min_fvalue[columnd_id];
        }
        return thrust::make_tuple(thrust::get<0>(r), left_sum + right_val);
      });
  thrust::transform(thrust::device, histogram.begin(), histogram.end(),
                    d_left_histogram.data(),
                    [] __device__(auto v) { return thrust::get<1>(v); });
  auto max_it =
      thrust::max_element(thrust::device, d_gain_histogram.data(),
                          d_gain_histogram.data() + d_gain_histogram.size());
  auto idx = std::distance(d_gain_histogram.data(), max_it);
  return idx;
}

template <typename GradientSumT>
void EvaluateSplits(common::Span<DeviceSplitCandidate> out_splits,
                    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                    EvaluateSplitInputs<GradientSumT> left,
                    EvaluateSplitInputs<GradientSumT> right) {
  dh::caching_device_vector<GradientSumT> left_histogram(left.gradient_histogram.size());
  auto d_left_histogram = dh::ToSpan(left_histogram);
  dh::caching_device_vector<float> left_gain_histogram(left.gradient_histogram.size(), 0);
  auto left_t = EvaluateNode(evaluator, left, d_left_histogram, dh::ToSpan(left_gain_histogram));

  dh::caching_device_vector<GradientSumT> right_histogram(left.gradient_histogram.size());
  auto d_right_histogram = dh::ToSpan(right_histogram);
  dh::caching_device_vector<float> right_gain_histogram(left.gradient_histogram.size(), 0);
  auto right_t = EvaluateNode(evaluator, right, d_right_histogram, dh::ToSpan(right_gain_histogram));

  std::cout << "left parent_sum: " << left.parent_sum << std::endl;
  dh::LaunchN(0, 2, [=]__device__(size_t idx) {
    decltype(left_t) split;
    common::Span<GradientSumT> left_sum_histogram;
    bst_feature_t fidx = 0;
    if (idx == 0) {
      split = left_t;
      left_sum_histogram = d_left_histogram;
    } else {
      split = right_t;
      left_sum_histogram = d_right_histogram;
    }
    // auto split_idx = thrust::get<2>(split);
    // if (idx == 0) {
    //   fidx = dh::SegmentId(left.feature_segments, split_idx);
    // } else {
    //   fidx = dh::SegmentId(left.feature_segments, split_idx);
    // }
    // out_splits[idx].Update(
    //     thrust::get<0>(split), kRightDir, left.feature_values[thrust::get<2>(split)],
    //     fidx, GradientPair{left_sum_histogram[split_idx]},
    //     GradientPair{left.parent_sum - left_sum_histogram[split_idx]}, left.param);
  });
}

template <typename GradientSumT>
void EvaluateSingleSplit(common::Span<DeviceSplitCandidate> out_split,
                         TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                         EvaluateSplitInputs<GradientSumT> input) {
  dh::caching_device_vector<GradientSumT> left_histogram(input.gradient_histogram.size());
  thrust::fill(left_histogram.begin(), left_histogram.end(), GradientSumT{});
  auto d_left_histogram = dh::ToSpan(left_histogram);
  dh::caching_device_vector<float> left_gain_histogram(input.gradient_histogram.size(), 0);
  thrust::fill(left_gain_histogram.begin(), left_gain_histogram.end(), 0);
  auto d_gain_histogram = dh::ToSpan(left_gain_histogram);
  auto left_t = EvaluateNode(evaluator, input, d_left_histogram, dh::ToSpan(left_gain_histogram));
  dh::LaunchN(0, 1, [=] __device__(size_t idx) {
    common::Span<GradientSumT> left_sum_histogram;
    bst_feature_t fidx = dh::SegmentId(input.feature_segments, left_t);
    left_sum_histogram = d_left_histogram;

    auto gain = d_gain_histogram[left_t];
    out_split[0].Update(
        gain, kRightDir, input.feature_values[left_t], fidx,
        GradientPair{left_sum_histogram[left_t]},
        GradientPair{input.parent_sum - left_sum_histogram[left_t]},
        input.param);
  });
}

template void EvaluateSplits<GradientPair>(
    common::Span<DeviceSplitCandidate> out_splits,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientPair> left,
    EvaluateSplitInputs<GradientPair> right);
template void EvaluateSplits<GradientPairPrecise>(
    common::Span<DeviceSplitCandidate> out_splits,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientPairPrecise> left,
    EvaluateSplitInputs<GradientPairPrecise> right);
template void EvaluateSingleSplit<GradientPair>(
    common::Span<DeviceSplitCandidate> out_split,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientPair> input);
template void EvaluateSingleSplit<GradientPairPrecise>(
    common::Span<DeviceSplitCandidate> out_split,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientPairPrecise> input);
}  // namespace tree
}  // namespace xgboost
