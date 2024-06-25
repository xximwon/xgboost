/**
 * Copyright 2022-2024, XGBoost Contributors
 *
 * @brief cuda pinned allocator for usage with thrust containers
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>  // for size_t
#include <functional>
#include <limits>  // for numeric_limits

#include "common.h"
#include "type.h"
#include "xgboost/span.h"

namespace xgboost::common::cuda_impl {
// \p pinned_allocator is a CUDA-specific host memory allocator
//  that employs \c cudaMallocHost for allocation.
//
// This implementation is ported from the experimental/pinned_allocator
// that Thrust used to provide.
//
//  \see https://en.cppreference.com/w/cpp/memory/allocator
template <typename T>
struct PinnedAllocPolicy {
  using pointer = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;  // NOLINT: The type returned by address()
  using size_type = std::size_t;   // NOLINT: The type used for the size of the allocation
  using value_type = T;            // NOLINT: The type of the elements in the allocator

  size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) const {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_alloc{};
    }  // end if

    pointer result(nullptr);
    dh::safe_cuda(cudaMallocHost(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFreeHost(p)); }  // NOLINT
};

template <typename T>
struct ManagedAllocPolicy {
  using pointer = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;  // NOLINT: The type returned by address()
  using size_type = std::size_t;   // NOLINT: The type used for the size of the allocation
  using value_type = T;            // NOLINT: The type of the elements in the allocator

  size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) const {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_alloc{};
    }  // end if

    pointer result(nullptr);
    dh::safe_cuda(cudaMallocManaged(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFree(p)); }  // NOLINT
};

// This is actually a pinned memory allocator in disguise. We utilize HMM or ATS for
// efficient tracked memory allocation.
template <typename T>
struct SamAllocPolicy {
  using pointer = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;  // NOLINT: The type returned by address()
  using size_type = std::size_t;   // NOLINT: The type used for the size of the allocation
  using value_type = T;            // NOLINT: The type of the elements in the allocator

  size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) const {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_alloc{};
    }  // end if

    size_type n_bytes = cnt * sizeof(value_type);
    pointer result = reinterpret_cast<pointer>(std::malloc(n_bytes));
    if (!result) {
      throw std::bad_alloc{};
    }
    dh::safe_cuda(cudaHostRegister(result, n_bytes, cudaHostRegisterDefault));
    return result;
  }

  void deallocate(pointer p, size_type) {  // NOLINT
    dh::safe_cuda(cudaHostUnregister(p));
    std::free(p);
  }
};

template <typename T, template <typename> typename Policy>
class CudaHostAllocatorImpl : public Policy<T> {
 public:
  using typename Policy<T>::value_type;
  using typename Policy<T>::pointer;
  using typename Policy<T>::const_pointer;
  using typename Policy<T>::size_type;

  using reference = value_type&;              // NOLINT: The parameter type for address()
  using const_reference = const value_type&;  // NOLINT: The parameter type for address()

  using difference_type = std::ptrdiff_t;  // NOLINT: The type of the distance between two pointers

  template <typename U>
  struct rebind {                                    // NOLINT
    using other = CudaHostAllocatorImpl<U, Policy>;  // NOLINT: The rebound type
  };

  CudaHostAllocatorImpl() = default;
  ~CudaHostAllocatorImpl() = default;
  CudaHostAllocatorImpl(CudaHostAllocatorImpl const&) = default;

  CudaHostAllocatorImpl& operator=(CudaHostAllocatorImpl const& that) = default;
  CudaHostAllocatorImpl& operator=(CudaHostAllocatorImpl&& that) = default;

  template <typename U>
  CudaHostAllocatorImpl(CudaHostAllocatorImpl<U, Policy> const&) {}  // NOLINT

  pointer address(reference r) { return &r; }              // NOLINT
  const_pointer address(const_reference r) { return &r; }  // NOLINT

  bool operator==(CudaHostAllocatorImpl const&) const { return true; }

  bool operator!=(CudaHostAllocatorImpl const& x) const { return !operator==(x); }
};

template <typename T>
using PinnedAllocator = CudaHostAllocatorImpl<T, PinnedAllocPolicy>;  // NOLINT

template <typename T>
using ManagedAllocator = CudaHostAllocatorImpl<T, ManagedAllocPolicy>;  // NOLINT

template <typename T>
using SamAllocator = CudaHostAllocatorImpl<T, SamAllocPolicy>;

template <bool managed>
class GrowOnlyPinnedVector {
  using T = std::int8_t;

 public:
  using size_type = typename PinnedAllocator<T>::size_type;              // NOLINT
  using pointer = typename PinnedAllocator<T>::pointer;                  // NOLINT
  using const_pointer = typename PinnedAllocator<T>::const_pointer;      // NOLINT
  using reference = typename PinnedAllocator<T>::reference;              // NOLINT
  using const_reference = typename PinnedAllocator<T>::const_reference;  // NOLINT

 private:
  std::unique_ptr<T, std::function<void(T*)>> data_;
  size_type capacity_{0};
  size_type n_{0};

 public:
  GrowOnlyPinnedVector() { this->Resize(1024); }

  void Resize(size_type n_bytes) {
    if (n_bytes > capacity_) {
      T* new_ptr{nullptr};
      auto new_size = std::max(n_bytes * 2, capacity_ * 2);
      if (managed) {
        dh::safe_cuda(cudaMallocManaged(&new_ptr, new_size));
      } else {
        dh::safe_cuda(cudaMallocHost(&new_ptr, new_size));
      }

      auto old_ptr = data_.get();
      dh::safe_cuda(cudaMemcpyAsync(new_ptr, old_ptr, n_, cudaMemcpyDefault));

      data_ = decltype(data_){new_ptr, [](T* ptr) {
                                if (managed) {
                                  dh::safe_cuda(cudaFree(ptr));
                                } else {
                                  dh::safe_cuda(cudaFreeHost(ptr));
                                }
                              }};
      capacity_ = new_size;
    }
    n_ = n_bytes;
  }
  [[nodiscard]] size_type Size() const { return n_; }
  [[nodiscard]] const_pointer Data() const { return data_.get(); }
  [[nodiscard]] pointer Data() { return data_.get(); }

  reference operator[](size_type i) { return Data()[i]; }
  const_reference operator[](size_type i) const { return Data()[i]; }

  template <typename U>
  [[nodiscard]] Span<U> GetSpan(size_type n) {
    auto n_bytes = n * sizeof(U);
    this->Resize(n_bytes);
    return RestoreType<U>(Span{data_.get(), n_bytes});
  }
};
}  // namespace xgboost::common::cuda_impl
