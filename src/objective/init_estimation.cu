#include "../common/stats.cuh"
#include "init_estimation.h"

namespace xgboost {
namespace obj {
void InitialEstimationRegression::QuantileGPU(Context const* ctx, float alpha, MetaInfo const& info,
                                              LearnerModelParam const* model,
                                              HostDeviceVector<float>* out_predt) {
  auto n_targets = Targets(info);
  HostDeviceVector<float> quantiles;
  HostDeviceVector<size_t> segments(n_targets + 1, 0);
  auto& h_segments = segments.HostVector();
  for (size_t i = 1; i < h_segments.size(); ++i) {
    h_segments[i] = h_segments[i - 1] + info.labels.Shape(0);
  }
  auto d_segments = segments.ConstDeviceSpan();
  if (info.weights_.Empty()) {
    /* fixme: incorrect */
    auto seg_beg = thrust::make_counting_iterator(0ul);
    auto seg_end = thrust::make_counting_iterator(info.labels.Size());
    /* adapt to row major matrix. */
    auto d_label = info.labels.View(ctx->gpu_id);
    auto val_beg = dh::MakeTransformIterator<float>(
        thrust::make_counting_iterator(0ul), [=] __device__(size_t i) {
          auto coord = linalg::UnravelIndex(i, d_label.Shape());
          return linalg::detail::Apply(d_label, coord);
        });
    auto val_end = val_beg + info.labels.Size();
    common::SegmentedQuantile(ctx, alpha, dh::tcbegin(d_segments), dh::tcend(d_segments), val_beg,
                              val_end, &quantiles);
  } else {
    info.weights_.SetDevice(ctx->gpu_id);
    auto d_weights = info.weights_.ConstDeviceSpan();
    CHECK_EQ(d_weights.size(), d_row_index.size());
    auto w_it = thrust::make_permutation_iterator(dh::tcbegin(d_weights), dh::tcbegin(d_row_index));
    common::SegmentedWeightedQuantile(ctx, alpha, seg_beg, seg_end, val_beg, val_end, w_it,
                                      w_it + d_weights.size(), &quantiles);
  }
}

void InitialEstimationRegression::MeanCUDA(Context const* ctx, MetaInfo const& info,
                                           LearnerModelParam const* model,
                                           HostDeviceVector<float>* out_predt) {

}
}  // namespace obj
}  // namespace xgboost
