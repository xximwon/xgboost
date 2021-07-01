/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <limits>
#include "evaluate_splits.cuh"
#include "../../common/categorical.h"
#include "../../common/algorithm.cuh"

namespace xgboost {
namespace tree {
namespace {

template <typename GradientSumT>
struct ScanElem {
  size_t idx;
  GradientSumT grad;
  DeviceSplitCandidate candidate;

  ScanElem() = default;
  XGBOOST_DEVICE ScanElem(size_t _idx, GradientSumT _grad,
                          DeviceSplitCandidate _candidate)
      : idx{_idx}, grad{_grad}, candidate{_candidate} {}
  XGBOOST_DEVICE explicit ScanElem(thrust::tuple<size_t, GradientSumT, DeviceSplitCandidate> const& tu) {
    idx = thrust::get<0>(tu);
    grad = thrust::get<1>(tu);
    candidate = thrust::get<2>(tu);
  }
};

// FIXME: add bool need_backward.
template <typename GradientSumT, typename ItemTy = ScanElem<GradientSumT>>
struct ScanOp : public thrust::binary_function<ItemTy, ItemTy, ItemTy> {
  EvaluateSplitInputs<GradientSumT> left;
  EvaluateSplitInputs<GradientSumT> right;
  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator;

  XGBOOST_DEVICE ScanOp(EvaluateSplitInputs<GradientSumT> l,
                        EvaluateSplitInputs<GradientSumT> r,
                        TreeEvaluator::SplitEvaluator<GPUTrainingParam> e)
      : left{std::move(l)}, right{std::move(r)}, evaluator{std::move(e)} {}

  template <bool forward, bool is_cat>
  XGBOOST_DEVICE DeviceSplitCandidate
  DoIt(EvaluateSplitInputs<GradientSumT> input, size_t idx,
       GradientSumT l_gpair, GradientSumT r_gpair,
       DeviceSplitCandidate l_split, bst_feature_t fidx) const {
    DeviceSplitCandidate best;
    float gain = evaluator.CalcSplitGain(
        input.param, input.nidx, fidx, GradStats{l_gpair}, GradStats{r_gpair});
    best.Update(l_split, input.param);
    float parent_gain = CalcGain(input.param, input.parent_sum);  // FIXME: get it out
    float loss_chg = gain - parent_gain;
    float fvalue = input.feature_values[idx];
    if (forward) {
      bool replaced =
          best.Update(loss_chg, kRightDir, fvalue, fidx, GradientPair{l_gpair},
                      GradientPair{r_gpair}, is_cat, input.param);
      if (idx == 1) {
        printf("replaced: %d, be: %f, af: %f \n", int(replaced), best.loss_chg, loss_chg);
      }
    } else {
      best.Update(loss_chg, kLeftDir, fvalue, fidx, GradientPair{r_gpair},
                  GradientPair{l_gpair}, is_cat, input.param);
    }

    return best;
  }

  template <bool forward>
  XGBOOST_DEVICE ScanElem<GradientSumT> Scan(ScanElem<GradientSumT> const &l, ScanElem<GradientSumT> const &r) const {
    DeviceSplitCandidate l_split = l.candidate;

    if (l.idx < left.gradient_histogram.size()) {
      // Left node
      auto r_idx = r.idx;

      auto l_fidx = dh::SegmentId(left.feature_segments, l.idx);
      auto r_fidx = dh::SegmentId(left.feature_segments, r.idx);
      /* Segmented scan with 2 segments
       * *****|******
       * 0, 1 |  2, 3
       *   /|_|_/| /|
       * 0, 1 |  2, 5
       * *****|******
       */
      if (l_fidx != r_fidx) {
        // Segmented scan
        // if (forward) {
        //   printf("segment: l.idx: %lu, r.idx: %lu, %f\n", l.idx, r.idx, r.candidate.loss_chg);
        // }
        return r;
      }

      assert(!left.feature_set.empty());
      if ((left.feature_set.size() != left.feature_segments.size() - 1) &&
          !thrust::binary_search(thrust::seq, left.feature_set.cbegin(),
                                 left.feature_set.cend(), l_fidx)) {
        // column sampling
        return {r_idx, r.grad, DeviceSplitCandidate{}};
      }

      if (common::IsCat(left.feature_types, l_fidx)) {
        auto l_gpair = left.gradient_histogram[l.idx];
        auto r_gpair = left.parent_sum - l_gpair;
        auto best = DoIt<forward, true>(left, l.idx, l_gpair, r_gpair, l_split, l_fidx);
        return {r_idx, r_gpair, best};
      } else {
        auto l_gpair = l.grad;
        auto r_gpair = left.parent_sum - l_gpair;
        DeviceSplitCandidate best = DoIt<forward, false>(left, l.idx, l_gpair, r_gpair, l_split, l_fidx);
        // if (forward && best.IsValid() && l.idx != 0) {
        //   auto g = l_gpair + r.grad;
        //   printf("l_idx: %d, g: %f, h: %f, chg: %f\n", int(l.idx), l_gpair.GetGrad(), l_gpair.GetHess(), best.loss_chg);
        // }
        return {r_idx, l_gpair + r.grad, best};
      }
    } else {
      // Right node
      if (left.gradient_histogram.size() != right.gradient_histogram.size()) {
        printf("left.gradient_histogram.size(): %lu, "
               "right.gradient_histogram.size(): %lu, idx: %lu, forward: %d\n",
               left.gradient_histogram.size(), right.gradient_histogram.size(),
               l.idx, int(forward));
      }
      assert(left.gradient_histogram.size() == right.gradient_histogram.size());
      auto l_idx = l.idx - left.gradient_histogram.size();
      auto r_idx = r.idx - left.gradient_histogram.size();

      auto l_fidx = dh::SegmentId(right.feature_segments, l_idx);
      auto r_fidx = dh::SegmentId(right.feature_segments, r_idx);
      if (l_fidx != r_fidx) {
        // Segmented scan
        return {r.idx, r.grad, r.candidate};
      }

      assert(!right.feature_segments.empty());
      if ((right.feature_set.size() != right.feature_segments.size()) &&
          !thrust::binary_search(thrust::seq, right.feature_set.cbegin(),
                                 right.feature_set.cend(), l_fidx)) {
        // column sampling
        return {r_idx, r.grad, DeviceSplitCandidate{}};
      }

      if (common::IsCat(right.feature_types, l_fidx)) {
        auto l_gpair = right.gradient_histogram[l_idx];
        auto r_gpair = right.parent_sum - l_gpair;
        auto best = DoIt<forward, true>(right, l_idx, l_gpair, r_gpair, l_split, l_fidx);
        return {r_idx, r_gpair, best};
      } else {
        auto l_gpair = l.grad;
        auto r_gpair = right.parent_sum - l_gpair;
        auto best = DoIt<forward, false>(right, l_idx, l_gpair, r_gpair, l_split, l_fidx);
        return {r_idx, l.grad + r.grad, best};
      }
    }
  }

  using Ty = thrust::tuple<ItemTy, ItemTy>;

  XGBOOST_DEVICE Ty operator()(Ty const &l, Ty const &r) const {
    auto fw = Scan<true>(thrust::get<0>(l), thrust::get<0>(r));
    auto bw = Scan<false>(thrust::get<1>(l), thrust::get<1>(r));
    return thrust::make_tuple(fw, bw);
  }
};

template <typename GradientSumT, typename Tu = thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>>
class DiscardOverload : public thrust::discard_iterator<Tu> {
 public:
  using value_type = Tu;  // NOLINT
};

template <typename GradientSumT>
struct WriteScan {
  using Tuple = typename DiscardOverload<GradientSumT>::value_type;
  using ScanElemTy = ScanElem<GradientSumT>;
  EvaluateSplitInputs<GradientSumT> left;
  EvaluateSplitInputs<GradientSumT> right;
  common::Span<ScanElem<GradientSumT>> d_out_scan;
  size_t n_features;

  template <bool forward>
  XGBOOST_DEVICE void DoIt(ScanElemTy const& candidate) {
    size_t offset = 0;
    size_t beg_idx = 0;
    size_t end_idx = 0;

    auto fidx = candidate.candidate.findex;
    auto idx = candidate.idx;

    if (idx < left.gradient_histogram.size()) {
      // left node
      beg_idx = left.feature_segments[fidx];
      auto f_size = left.feature_segments[fidx + 1] - beg_idx;
      f_size = f_size == 0 ? 0 : f_size - 1;
      end_idx = beg_idx + f_size;
    } else {
      // right node
      beg_idx = right.feature_segments[fidx];
      auto f_size = right.feature_segments[fidx + 1] - beg_idx;
      f_size = f_size == 0 ? 0 : f_size - 1;
      end_idx = beg_idx + f_size;
      offset = n_features * 2;
    }
    // printf("idx: %lu, fidx: %d, beg: %lu, end: %lu, chg: %f, g: %f, f: %d \n",
    //        idx, int(fidx), beg_idx, end_idx, candidate.candidate.loss_chg,
    //        candidate.grad.GetGrad(), int(forward));
    if (forward) {
      if (end_idx == idx) {
        d_out_scan[offset + fidx] = candidate;
      }
    } else {
      if (beg_idx == idx) {
        d_out_scan[offset + n_features + fidx] = candidate;
      }
    }
  }

  XGBOOST_DEVICE Tuple operator()(Tuple const &tu) {
    ScanElem<GradientSumT> const &fw = thrust::get<0>(tu);
    ScanElem<GradientSumT> const &bw = thrust::get<1>(tu);
    if (fw.candidate.findex != -1) {
      DoIt<true>(fw);
    }
    if (bw.candidate.findex != -1) {
      DoIt<false>(bw);
    }
    return {};  // discard
  }
};

template <typename GradientSumT, bool forward>
struct ScanValueOp {
  EvaluateSplitInputs<GradientSumT> left;
  EvaluateSplitInputs<GradientSumT> right;
  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator;

  XGBOOST_DEVICE ScanElem<GradientSumT> operator()(size_t idx) {
    ScanElem<GradientSumT> ret;
    ret.idx = idx;
    float fvalue;
    size_t fidx;
    bool is_cat;
    float loss_chg;
    if (idx < left.gradient_histogram.size()) {
      // left node
      ret.grad = left.gradient_histogram[idx];
      fvalue = left.feature_values[idx];
      fidx = dh::SegmentId(left.feature_segments, idx);
      is_cat = common::IsCat(left.feature_types, fidx);

      float parent_gain =
          CalcGain(left.param, left.parent_sum); // FIXME: get it out
      float gain = evaluator.CalcSplitGain(left.param, left.nidx, fidx,
                                           GradStats{ret.grad},
                                           GradStats{left.parent_sum - ret.grad});
      loss_chg = gain - parent_gain;
    } else {
      // right node
      idx -= left.gradient_histogram.size();
      ret.grad = right.gradient_histogram[idx];
      fvalue = right.feature_values[idx];
      fidx = dh::SegmentId(right.feature_segments, idx);
      is_cat = common::IsCat(right.feature_types, fidx);

      float parent_gain =
          CalcGain(right.param, right.parent_sum); // FIXME: get it out
      float gain = evaluator.CalcSplitGain(right.param, right.nidx, fidx,
                                           GradStats{ret.grad},
                                           GradStats{left.parent_sum - ret.grad});
      loss_chg = gain - parent_gain;
    };
    if (forward) {
      ret.candidate.Update(
          loss_chg, kRightDir, fvalue, fidx, GradientPair{ret.grad},
          GradientPair{left.parent_sum - ret.grad}, is_cat, left.param);
    } else {
      ret.candidate.Update(loss_chg, kLeftDir, fvalue, fidx,
                           GradientPair{left.parent_sum - ret.grad},
                           GradientPair{ret.grad}, is_cat, left.param);
    }
    return ret;
  }
};
}  // anonymous namespace

template <typename GradientSumT>
void EvaluateSplits(common::Span<DeviceSplitCandidate> out_splits,
                    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                    EvaluateSplitInputs<GradientSumT> left,
                    EvaluateSplitInputs<GradientSumT> right) {
  /**
   * Validate inputs
   */
  CHECK(left.gradient_histogram.size() == right.gradient_histogram.size() ||
        right.gradient_histogram.empty());
  CHECK(left.feature_segments.size() == right.feature_segments.size() ||
        right.feature_segments.empty());
  if (left.feature_segments.empty()) {
    CHECK(left.gradient_histogram.empty());
  }
  if (right.feature_segments.empty()) {
    CHECK(right.gradient_histogram.empty());
  }

  std::cout << "left: " << left.gradient_histogram.size()
            << ", right: " << right.gradient_histogram.size() << std::endl;

  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  CHECK(r_n_features == 0 || l_n_features == r_n_features);
  auto n_features = l_n_features + r_n_features;

  /**
   * Handle trivial input
   */
  if (n_features == 0) {
    dh::LaunchN(out_splits.size(), [=]XGBOOST_DEVICE(size_t idx) {
      out_splits[idx] = DeviceSplitCandidate{};
    });
    return;
  }

  std::cout << "n_features:" << n_features << std::endl;

  size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();

  auto for_counting = thrust::make_counting_iterator(0ul);
  auto rev_counting =
      thrust::make_reverse_iterator(thrust::make_counting_iterator(size));
  auto for_value_iter = dh::MakeTransformIterator<ScanElem<GradientSumT>>(
      for_counting, ScanValueOp<GradientSumT, true>{left, right, evaluator});
  auto rev_value_iter = dh::MakeTransformIterator<ScanElem<GradientSumT>>(
      rev_counting, ScanValueOp<GradientSumT, false>{left, right, evaluator});

  auto value_iter = thrust::make_zip_iterator(thrust::make_tuple(for_value_iter, rev_value_iter));
  using FBTuple = thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>;
  dh::device_vector<ScanElem<GradientSumT>> out_scan(n_features * 2); // x2 due to forward and backward
  auto d_out_scan = dh::ToSpan(out_scan);

  using Discard = DiscardOverload<GradientSumT>;
  auto out_it = thrust::make_transform_output_iterator(
      Discard(), WriteScan<GradientSumT>{left, right, d_out_scan, l_n_features});

  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, value_iter, out_it,
                                 ScanOp<GradientSumT>{left, right, evaluator},
                                 size);
  dh::TemporaryArray<int8_t> temp(temp_bytes);
  cub::DeviceScan::InclusiveScan(
      temp.data().get(), temp_bytes, value_iter, out_it,
      ScanOp<GradientSumT>{left, right, evaluator}, size);

  // {
  //   // debug
  //   std::cout << "beg debug" << std::endl;
  //   dh::device_vector<FBTuple> out_scan(size);
  //   size_t temp_bytes = 0;
  //   cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, value_iter, out_scan.begin(),
  //                                  ScanOp<GradientSumT>{left, right, evaluator},
  //                                  size);
  //   dh::TemporaryArray<int8_t> temp(temp_bytes);
  //   cub::DeviceScan::InclusiveScan(
  //       temp.data().get(), temp_bytes, value_iter, out_scan.begin(),
  //       ScanOp<GradientSumT>{left, right, evaluator}, size);
  //   dh::DebugSyncDevice();
  //   for (size_t i = 0; i < out_scan.size(); ++i) {
  //     auto fw = thrust::get<0>(FBTuple(out_scan[i]));
  //     auto bw = thrust::get<1>(FBTuple(out_scan[i]));
  //     std::cout << "i: " << i << ", grad: " << fw.grad << "\n"
  //               << fw.candidate << bw.candidate << std::endl;
  //   }
  //   std::cout << "end debug" << std::endl;
  // }

  dh::DebugSyncDevice();
  for (size_t i = 0; i < out_scan.size(); ++i) {
    auto candidate = ScanElem<GradientSumT>(out_scan[i]);
    std::cout << "i: " << i << ", grad: " << candidate.grad << "\n"
              << candidate.candidate << std::endl;
  }

  auto reduce_key = dh::MakeTransformIterator<int>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(bst_feature_t fidx) -> int {
        if (fidx < l_n_features * 2) {
          return 0;  // left node
        } else {
          return 1;  // right node
        }
      });
  auto reduce_val = dh::MakeTransformIterator<DeviceSplitCandidate>(
      thrust::make_counting_iterator(0),
      [d_out_scan] __device__(size_t idx) {
        // No need to distinguish left and right node as we are just extracting values.
        ScanElem<GradientSumT> candidate = d_out_scan[idx];
        return candidate.candidate;
      });
  thrust::reduce_by_key(
      thrust::device, reduce_key, reduce_key + out_scan.size(),
      reduce_val, thrust::make_discard_iterator(), out_splits.data(),
      thrust::equal_to<int>{},
      [=] XGBOOST_DEVICE(DeviceSplitCandidate l, DeviceSplitCandidate r) {
        l.Update(r, left.param);
        return l;
      });
}

template <typename GradientSumT>
void EvaluateSingleSplit(common::Span<DeviceSplitCandidate> out_split,
                         TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                         EvaluateSplitInputs<GradientSumT> input) {
  EvaluateSplits(out_split, evaluator, input, {});
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
