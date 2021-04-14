#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include "../../../src/data/device_adapter.cuh"

namespace xgboost {
struct IndexFlagTuple {
  size_t idx;
  size_t flag;
};

struct IndexFlagOp {
  __device__ IndexFlagTuple operator()(const IndexFlagTuple& a,
                                       const IndexFlagTuple& b) const {
    return {b.idx, a.flag + b.flag};
  }
};

template <typename ValIt, typename Idx>
struct WriteResultsFunctor {
  uint32_t bit;
  ValIt it;
  common::Span<Idx> address;
  int64_t* d_left_count;

  __device__ IndexFlagTuple operator()(const IndexFlagTuple& x) {
    // the ex_scan_result represents how many rows have been assigned to left
    // node so far during scan.
    int scatter_address;
    if (*(it + x.idx) & bit) {
      scatter_address = x.flag - 1;  // -1 because inclusive scan
    } else {
      scatter_address = (x.idx - x.flag) + *d_left_count;
    }
    address[x.idx] = scatter_address;

    // Discard
    return {};
  }
};

class DiscardOverload : public thrust::discard_iterator<IndexFlagTuple> {
 public:
  using value_type = IndexFlagTuple;  // NOLINT
};

template <typename ValIt, typename Idx>
void Argsort(ValIt val_first, ValIt val_last,
             dh::device_vector<Idx> &address_a) {
  int constexpr kValueSize =
      sizeof(typename thrust::iterator_traits<ValIt>::value_type);
  dh::Iota(dh::ToSpan(address_a));
  dh::device_vector<Idx> address_b(thrust::distance(val_first, val_last));
  dh::DoubleBuffer<Idx> buffer{&address_a, &address_b};

  dh::PinnedMemory left_count;
  auto d_left_count = left_count.GetSpan<int64_t>(1);

  for (size_t i = 0; i < kValueSize; ++i) {
    auto bit = 1u << i;

    auto current = buffer.Current();
    auto other = buffer.Other();

    WriteResultsFunctor<ValIt, Idx> write_results{
        bit, val_first, common::Span<Idx>{other, buffer.Size()},
        d_left_count.data()};
    auto discard_write_iterator = thrust::make_transform_output_iterator(
        DiscardOverload(), write_results);
    auto counting = thrust::make_counting_iterator(0llu);
    auto input_iterator = dh::MakeTransformIterator<IndexFlagTuple>(
        counting, [=] __device__(size_t idx) {
          return IndexFlagTuple{
              idx, static_cast<size_t>(*(val_first + current[idx]) & bit)};
        });

    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, input_iterator,
                                   discard_write_iterator, IndexFlagOp(),
                                   buffer.Size());
    dh::TemporaryArray<int8_t> temp(temp_bytes);
    cub::DeviceScan::InclusiveScan(temp.data().get(), temp_bytes,
                                   input_iterator, discard_write_iterator,
                                   IndexFlagOp(), buffer.Size());

    buffer.Alternate();
  }
}
}  // namespace xgboost

namespace xgboost {

void TestDeviceArgSort() {
  dh::device_vector<float> values(10000);
  dh::device_vector<uint32_t> indices(10000);

  auto d_values = dh::ToSpan(values);
  auto d_indices = dh::ToSpan(indices);

  dh::device_vector<uint32_t> sorted_idx(values.size());
  auto it = dh::MakeTransformIterator<uint64_t>(
      thrust::make_counting_iterator(0ul), [=] __device__(size_t idx) {
        uint64_t value = d_values[idx];
        return value;
      });
  Argsort(it, it + sorted_idx.size(), sorted_idx);
}
TEST(Helper, Argsort) {
  TestDeviceArgSort();
}
}
