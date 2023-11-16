/**
 * Copyright 2019-2023, XGBoost Contributors
 *
 * \file data.cu
 * \brief Handles setting metainfo from array interface.
 */
#include "../common/algorithm.cuh"  // for ArgSort
#include "../common/cuda_context.cuh"
#include "../common/device_helpers.cuh"
#include "../common/linalg_op.cuh"
#include "array_interface.h"
#include "device_adapter.cuh"
#include "simple_dmatrix.h"
#include "validation.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/linalg.h"  // for Tensor, TensorView, VectorView
#include "xgboost/logging.h"

namespace xgboost {
namespace {
auto SetDeviceToPtr(void const* ptr) {
  cudaPointerAttributes attr;
  dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
  int32_t ptr_device = attr.device;
  dh::safe_cuda(cudaSetDevice(ptr_device));
  return ptr_device;
}

template <typename T, int32_t D>
void CopyTensorInfoImpl(CUDAContext const* ctx, Json arr_interface, linalg::Tensor<T, D>* p_out) {
  ArrayInterface<D> array(arr_interface);
  if (array.n == 0) {
    p_out->SetDevice(DeviceOrd::CUDA(0));
    p_out->Reshape(array.shape);
    return;
  }
  CHECK_EQ(array.valid.Capacity(), 0)
      << "Meta info like label or weight can not have missing value.";
  auto ptr_device = DeviceOrd::CUDA(SetDeviceToPtr(array.data));
  p_out->SetDevice(ptr_device);

  if (array.is_contiguous && array.type == ToDType<T>::kType) {
    p_out->ModifyInplace([&](HostDeviceVector<T>* data, common::Span<size_t, D> shape) {
      // set shape
      std::copy(array.shape, array.shape + D, shape.data());
      // set data
      data->Resize(array.n);
      dh::safe_cuda(cudaMemcpyAsync(data->DevicePointer(), array.data, array.n * sizeof(T),
                                    cudaMemcpyDefault, ctx->Stream()));
    });
    return;
  }
  p_out->Reshape(array.shape);
  auto t = p_out->View(ptr_device);
  linalg::ElementWiseTransformDevice(
      t,
      [=] __device__(size_t i, T) {
        return linalg::detail::Apply(TypedIndex<T, D>{array},
                                     linalg::UnravelIndex<D>(i, array.shape));
      },
      ctx->Stream());
}

void CopyGroupInfoImpl(Context const* ctx, ArrayInterface<1> column,
                       std::vector<bst_group_t>* out) {
  CHECK(column.type != ArrayInterfaceHandler::kF4 && column.type != ArrayInterfaceHandler::kF8)
      << "Expected integer for group info.";

  auto ptr_device = SetDeviceToPtr(column.data);
  CHECK_EQ(ptr_device, dh::CurrentDevice());
  dh::TemporaryArray<bst_group_t> temp(column.Shape(0));
  auto d_tmp = temp.data().get();

  dh::LaunchN(column.Shape(0), ctx->CUDACtx()->Stream(),
              [=] __device__(size_t idx) { d_tmp[idx] = TypedIndex<size_t, 1>{column}(idx); });
  auto length = column.Shape(0);
  out->resize(length + 1);
  out->at(0) = 0;
  thrust::copy(temp.data(), temp.data() + length, out->begin() + 1);
  std::partial_sum(out->begin(), out->end(), out->begin());
}

void CopyQidImpl(Context const* ctx, ArrayInterface<1> array_interface,
                 std::vector<bst_group_t>* p_group_ptr) {
  // fixme: allow non-sorted qid.
  auto cuctx = ctx->CUDACtx();

  auto &group_ptr_ = *p_group_ptr;
  auto it = dh::MakeTransformIterator<uint32_t>(
      thrust::make_counting_iterator(0ul), [array_interface] __device__(std::size_t i) {
        return TypedIndex<uint32_t, 1>{array_interface}(i);
      });
  dh::caching_device_vector<bool> flag(1);
  auto d_flag = dh::ToSpan(flag);
  auto d = DeviceOrd::CUDA(SetDeviceToPtr(array_interface.data));
  dh::LaunchN(1, cuctx->Stream(), [=] __device__(size_t) { d_flag[0] = true; });
  dh::LaunchN(array_interface.Shape(0) - 1, cuctx->Stream(), [=] __device__(std::size_t i) {
    auto typed = TypedIndex<uint32_t, 1>{array_interface};
    if (typed(i) > typed(i + 1)) {
      d_flag[0] = false;
    }
  });
  bool non_dec = true;
  dh::safe_cuda(cudaMemcpyAsync(&non_dec, flag.data().get(), sizeof(bool), cudaMemcpyDeviceToHost,
                                cuctx->Stream()));
  CHECK(non_dec) << "`qid` must be sorted in increasing order along with data.";
  size_t bytes = 0;
  dh::caching_device_vector<uint32_t> out(array_interface.Shape(0));
  dh::caching_device_vector<uint32_t> cnt(array_interface.Shape(0));
  HostDeviceVector<int> d_num_runs_out(1, 0, d);
  cub::DeviceRunLengthEncode::Encode(nullptr, bytes, it, out.begin(), cnt.begin(),
                                     d_num_runs_out.DevicePointer(), array_interface.Shape(0),
                                     cuctx->Stream());
  dh::caching_device_vector<char> tmp(bytes);
  cub::DeviceRunLengthEncode::Encode(tmp.data().get(), bytes, it, out.begin(), cnt.begin(),
                                     d_num_runs_out.DevicePointer(), array_interface.Shape(0),
                                     cuctx->Stream());

  auto h_num_runs_out = d_num_runs_out.HostSpan()[0];
  group_ptr_.clear();
  group_ptr_.resize(h_num_runs_out + 1, 0);
  thrust::inclusive_scan(cuctx->CTP(), cnt.begin(), cnt.begin() + h_num_runs_out, cnt.begin());
  thrust::copy(cnt.begin(), cnt.begin() + h_num_runs_out, group_ptr_.begin() + 1);
}
}  // namespace

void MetaInfo::SetInfoFromCUDA(Context const& ctx, StringView key, Json array) {
  // multi-dim float info
  if (key == "base_margin") {
    CopyTensorInfoImpl(ctx.CUDACtx(), array, &base_margin_);
    return;
  } else if (key == "label") {
    CopyTensorInfoImpl(ctx.CUDACtx(), array, &labels);
    auto ptr = labels.Data()->ConstDevicePointer();
    auto valid = thrust::none_of(thrust::device, ptr, ptr + labels.Size(), data::LabelsCheck{});
    CHECK(valid) << "Label contains NaN, infinity or a value too large.";
    return;
  }
  // uint info
  if (key == "group") {
    ArrayInterface<1> array_interface{array};
    CopyGroupInfoImpl(&ctx, array_interface, &group_ptr_);
    data::ValidateQueryGroup(group_ptr_);
    return;
  } else if (key == "qid") {
    ArrayInterface<1> array_interface{array};
    CopyQidImpl(&ctx, array_interface, &group_ptr_);
    data::ValidateQueryGroup(group_ptr_);
    return;
  }
  // float info
  linalg::Tensor<float, 1> t;
  CopyTensorInfoImpl(ctx.CUDACtx(), array, &t);
  if (key == "weight") {
    this->weights_ = std::move(*t.Data());
    auto ptr = weights_.ConstDevicePointer();
    auto valid = thrust::none_of(thrust::device, ptr, ptr + weights_.Size(), data::WeightsCheck{});
    CHECK(valid) << "Weights must be positive values.";
  } else if (key == "label_lower_bound") {
    this->labels_lower_bound_ = std::move(*t.Data());
  } else if (key == "label_upper_bound") {
    this->labels_upper_bound_ = std::move(*t.Data());
  } else if (key == "feature_weights") {
    this->feature_weights = std::move(*t.Data());
    auto d_feature_weights = feature_weights.ConstDeviceSpan();
    auto valid =
        thrust::none_of(ctx.CUDACtx()->CTP(), d_feature_weights.data(),
                        d_feature_weights.data() + d_feature_weights.size(), data::WeightsCheck{});
    CHECK(valid) << "Feature weight must be greater than 0.";
  } else {
    LOG(FATAL) << "Unknown key for MetaInfo: " << key;
  }
}

namespace cuda_impl {
void SortSparsePageByQid(Context const* ctx, MetaInfo const& info,
                         HostDeviceVector<bst_row_t>* io_offset, HostDeviceVector<Entry>* io_data) {
  auto qid = info.qid.View(ctx->Device()).Values();
  dh::device_vector<std::size_t> idx(info.num_row_);
  auto d_sorted_idx = dh::ToSpan(idx);
  common::ArgSort<true>(ctx, qid, d_sorted_idx);
  auto cuctx = ctx->CUDACtx();

  HostDeviceVector<Entry> out;
  out.SetDevice(ctx->Device());
  out.Resize(io_data->Size());
  auto d_out = out.DeviceSpan();

  io_offset->SetDevice(ctx->Device());
  auto d_offset = io_offset->ConstDeviceSpan();

  dh::LaunchN(idx.size(), cuctx->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    auto src_beg = d_offset[d_sorted_idx[i]];
    auto src_end = d_offset[d_sorted_idx[i] + 1];
  });
}

namespace {
template <typename T, typename U = std::add_const_t<T>>
void PermuByQid(CUDAContext const* cuctx, common::Span<bst_group_t> qid, common::Span<U> data,
                common::Span<T> buffer) {
  CHECK_EQ(qid.size(), data.size());
  CHECK_EQ(data.size(), buffer.size());
  auto it = thrust::make_permutation_iterator(dh::tcbegin(data), dh::tcbegin(qid));
  // Sort data into buffer.
  thrust::copy_n(cuctx->CTP(), it, data.size(), dh::tbegin(buffer));
  // Copy it back to data.
  thrust::copy_n(cuctx->CTP(), dh::tbegin(buffer), buffer.size(), dh::tbegin(data));
}

template <typename T, std::int32_t kDim>
void PermuByQid(CUDAContext const* cuctx, linalg::VectorView<bst_group_t> qid,
                linalg::TensorView<T, kDim> data, linalg::TensorView<T, kDim> buffer) {
  CHECK_EQ(qid.Size(), data.Shape(0));
  CHECK_EQ(data.Size(), buffer.Size());
  static_assert(kDim == 2);

  // Sort data into buffer.
  thrust::for_each_n(cuctx->CTP(), thrust::make_counting_iterator(0ul), data.Size(),
                     [=] __device__(std::size_t i) mutable {
                       for (std::size_t j = 0; j < data.Shape(1); ++j) {
                         buffer(i, j) = data(qid(i), j);
                       }
                     });
  // Copy it back to data.
  thrust::copy_n(cuctx->CTP(), dh::tcbegin(buffer.Values()), buffer.Size(),
                 dh::tbegin(data.Values()));
}
}  // namespace

void SortByQid(Context const* ctx, MetaInfo* p_info) {
  auto& info = *p_info;
  CHECK(!info.qid.Empty());
  auto cuctx = ctx->CUDACtx();

  auto d_qid = info.qid.View(ctx->Device());
  dh::device_vector<std::size_t> sorted_idx(d_qid.Size());
  common::ArgSort<true>(ctx, d_qid.Values(), dh::ToSpan(sorted_idx));
  if (info.labels.Empty()) {
    CHECK_NE(info.num_col_, 0);
  }

  {
    // base_margin
    linalg::Tensor<float, 2> buffer{info.base_margin_.Shape(), ctx->Device()};
    auto d_buffer = buffer.View(ctx->Device());
    auto d_bm = info.base_margin_.View(ctx->Device());
    PermuByQid(cuctx, d_qid, d_bm, d_buffer);
  }
  {
    // label
    linalg::Tensor<float, 2> buffer{info.labels.Shape(), ctx->Device()};
    auto d_buffer = buffer.View(ctx->Device());
    auto d_y = info.labels.View(ctx->Device());
    PermuByQid(cuctx, d_qid, d_y, d_buffer);
  }
  {
    // qid itself
    linalg::Tensor<float, 2> buffer{info.qid.Shape(), ctx->Device()};
    auto d_buffer = buffer.View(ctx->Device()).Values();
    PermuByQid(cuctx, d_qid.Values(), d_qid.Values(), d_buffer);
  }
  {
    // rebuild group_ptr
    auto jarray = linalg::ArrayInterface(d_qid);
    ArrayInterface<1> array{jarray};
    CopyQidImpl(ctx, array, &info.group_ptr_);
    data::ValidateQueryGroup(info.group_ptr_);
  }
  // weight, see the CPU variant for details
  if (info.weights_.Size() == info.qid.Size()) {
    auto d_weight = info.weights_.DeviceSpan();
    dh::device_vector<float> buffer(d_weight.size());
    thrust::copy_n(dh::tcbegin(d_weight), d_weight.size(), buffer.begin());
    auto d_buffer = dh::ToSpan(buffer);
    PermuByQid(cuctx, d_qid.Values(), d_buffer, d_weight);

    dh::device_vector<bst_group_t> d_group_ptr(info.group_ptr_.size());
    dh::safe_cuda(cudaMemcpyAsync(d_group_ptr.data().get(), info.group_ptr_.data(),
                                  dh::ToSpan(d_group_ptr).size_bytes(), cudaMemcpyHostToHost,
                                  cuctx->Stream()));

    std::size_t n_bytes{0};
    cub::DeviceSegmentedReduce::Sum(nullptr, n_bytes, dh::tbegin(d_weight), dh::tend(d_weight),
                                    info.group_ptr_.size() - 1, d_group_ptr.data(),
                                    d_group_ptr.data() + 1, cuctx->Stream());
    dh::TemporaryArray<char> temp(n_bytes);
    cub::DeviceSegmentedReduce::Sum(temp.data().get(), n_bytes, dh::tbegin(d_weight),
                                    dh::tend(d_weight), info.group_ptr_.size() - 1,
                                    d_group_ptr.data(), d_group_ptr.data() + 1, cuctx->Stream());

    auto sd_group_ptr = dh::ToSpan(d_group_ptr);
    thrust::for_each_n(cuctx->CTP(), thrust::make_counting_iterator(0ul), d_weight.size(),
                       [=] XGBOOST_DEVICE(std::size_t i) {
                         float n = sd_group_ptr[i + 1] - sd_group_ptr[i];
                         d_weight[i] /= n;
                       });
  } else {
    auto min_qid =
        thrust::reduce(cuctx->CTP(), linalg::tcbegin(d_qid), linalg::tcend(d_qid),
                       std::numeric_limits<bst_group_t>::max(),
                       [] XGBOOST_DEVICE(bst_group_t a, bst_group_t b) { return std::min(a, b); });
    CHECK_EQ(min_qid, 0) << error::QidWeight();
  }
}
}  // namespace cuda_impl

template <typename AdapterT>
DMatrix* DMatrix::Create(AdapterT* adapter, float missing, int nthread,
                         const std::string& cache_prefix, DataSplitMode data_split_mode) {
  CHECK_EQ(cache_prefix.size(), 0)
      << "Device memory construction is not currently supported with external "
         "memory.";
  return new data::SimpleDMatrix(adapter, missing, nthread, data_split_mode);
}

template DMatrix* DMatrix::Create<data::CudfAdapter>(
    data::CudfAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, DataSplitMode data_split_mode);
template DMatrix* DMatrix::Create<data::CupyAdapter>(
    data::CupyAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, DataSplitMode data_split_mode);
}  // namespace xgboost
