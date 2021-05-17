#include <cub/cub.cuh>
#include <iterator>  // std::distance
#include "./device_helpers.cuh"
#include "xgboost/span.h"
#include <thrust/tuple.h>

namespace dh {
namespace detail {
template <typename T, typename Op> struct SegmentedInclusiveOp {
  Op op;
  xgboost::common::Span<size_t> segments;

  XGBOOST_DEVICE thrust::tuple<size_t, T>
  operator()(thrust::tuple<size_t, T> const &l,
             thrust::tuple<size_t, T> const &r) {
    auto l_sidx = dh::SegmentId(segments, thrust::get<0>(l));
    auto r_sidx = dh::SegmentId(segments, thrust::get<0>(r));

    if (l_sidx != r_sidx) {
      return r;
    }

    auto ret = op(thrust::get<1>(l), l_sidx, thrust::get<1>(r), r_sidx);
    auto tuple = thrust::make_tuple(thrust::get<0>(r), ret);
    return tuple;
  }
};
}  // namespace detail

template <typename In, typename Out, typename Op>
void SegmentedInclusiveScan(xgboost::common::Span<size_t> segments, In in_first,
                            In in_second, Out out_first, Op op) {
  auto n_elements = std::distance(in_first, in_second);

  auto in = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator(0ul), in_first));
  auto out = thrust::make_transform_output_iterator(
      out_first,
      [] __device__(auto zipped_it) { return thrust::get<1>(zipped_it); });
  detail::SegmentedInclusiveOp<typename thrust::iterator_traits<In>::value_type,
                               Op>
      seg_op{op, segments};

  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, in, out, seg_op,
                                 n_elements);
  dh::TemporaryArray<int8_t> temp(temp_bytes);
  cub::DeviceScan::InclusiveScan(temp.data().get(), temp_bytes, in, out, seg_op,
                                 n_elements);
}
}  // namespace dh
