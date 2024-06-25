/*!
 * Copyright 2022-2024, XGBoost Contributors
 * \file common.h
 * \brief cuda pinned allocator for usage with thrust containers
 */

#pragma once

#include <cstddef>
#include <limits>

#include "common.h"
#include "type.h"
#include "xgboost/span.h"

namespace xgboost::common {
namespace cuda {

// \p pinned_allocator is a CUDA-specific host memory allocator
//  that employs \c cudaMallocHost for allocation.
//
// This implementation is ported from the experimental/pinned_allocator
// that Thrust used to provide.
//
//  \see https://en.cppreference.com/w/cpp/memory/allocator
template <typename T>
class pinned_allocator;  // NOLINT

template <>
class pinned_allocator<void> {
 public:
  using value_type      = void;            // NOLINT: The type of the elements in the allocator
  using pointer         = void*;           // NOLINT: The type returned by address() / allocate()
  using const_pointer   = const void*;     // NOLINT: The type returned by address()
  using size_type       = std::size_t;     // NOLINT: The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  // NOLINT: The type of the distance between two pointers

  template <typename U>
  struct rebind {                       // NOLINT
    using other = pinned_allocator<U>;  // NOLINT: The rebound type
  };
};

template <typename T>
class pinned_allocator {
 public:
  using value_type      = T;               // NOLINT: The type of the elements in the allocator
  using pointer         = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer   = const T*;        // NOLINT: The type returned by address()
  using reference       = T&;              // NOLINT: The parameter type for address()
  using const_reference = const T&;        // NOLINT: The parameter type for address()
  using size_type       = std::size_t;     // NOLINT: The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  // NOLINT: The type of the distance between two pointers

  template <typename U>
  struct rebind {                       // NOLINT
    using other = pinned_allocator<U>;  // NOLINT: The rebound type
  };

  XGBOOST_DEVICE inline pinned_allocator() {}; // NOLINT: host/device markup ignored on defaulted functions
  XGBOOST_DEVICE inline ~pinned_allocator() {} // NOLINT: host/device markup ignored on defaulted functions
  XGBOOST_DEVICE inline pinned_allocator(pinned_allocator const&) {} // NOLINT: host/device markup ignored on defaulted functions

  pinned_allocator& operator=(pinned_allocator const& that) = default;
  pinned_allocator& operator=(pinned_allocator&& that) = default;

  template <typename U>
  pinned_allocator(pinned_allocator<U> const&) {} // NOLINT

  [[nodiscard]] pointer address(reference r) { return &r; }              // NOLINT
  [[nodiscard]] const_pointer address(const_reference r) { return &r; }  // NOLINT

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_alloc();
    }

    pointer result(nullptr);
    dh::safe_cuda(cudaMallocHost(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFreeHost(p)); } // NOLINT

  [[nodiscard]] size_type max_size() const { // NOLINT
    return (std::numeric_limits<size_type>::max)() / sizeof(T);
  }

  [[nodiscard]] bool operator==(pinned_allocator const& x) const { return true; }

  [[nodiscard]] bool operator!=(pinned_allocator const& x) const { return !operator==(x); }
};

template <typename T>
class managed_allocator {
 public:
  using value_type = T;                    // NOLINT: The type of the elements in the allocator
  using pointer = T*;                      // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;          // NOLINT: The type returned by address()
  using reference = T&;                    // NOLINT: The parameter type for address()
  using const_reference = const T&;        // NOLINT: The parameter type for address()
  using size_type = std::size_t;           // NOLINT: The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  // NOLINT: The type of the distance between two pointers

  template <typename U>
  struct rebind {                       // NOLINT
    using other = pinned_allocator<U>;  // NOLINT: The rebound type
  };

  managed_allocator() {};  // NOLINT: host/device markup ignored on defaulted functions
  ~managed_allocator() {}  // NOLINT: host/device markup ignored on defaulted functions
  managed_allocator(managed_allocator const&) = default;

  template <typename U>
  managed_allocator(managed_allocator<U> const&) {}  // NOLINT

  [[nodiscard]] pointer address(reference r) { return &r; }              // NOLINT
  [[nodiscard]] const_pointer address(const_reference r) { return &r; }  // NOLINT

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_alloc();
    }

    pointer result(nullptr);
    dh::safe_cuda(cudaMallocManaged(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFree(p)); }  // NOLINT

  [[nodiscard]] size_type max_size() const {  // NOLINT
    return (std::numeric_limits<size_type>::max)() / sizeof(T);
  }

  [[nodiscard]] bool operator==(managed_allocator const& x) const { return true; }

  [[nodiscard]] bool operator!=(managed_allocator const& x) const { return !operator==(x); }
};

template <bool managed>
class GrowOnlyPinnedVector {
  using T = std::int8_t;

 public:
  using size_type = typename pinned_allocator<T>::size_type;              // NOLINT
  using pointer = typename pinned_allocator<T>::pointer;                  // NOLINT
  using const_pointer = typename pinned_allocator<T>::const_pointer;      // NOLINT
  using reference = typename pinned_allocator<T>::reference;              // NOLINT
  using const_reference = typename pinned_allocator<T>::const_reference;  // NOLINT

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
}  // namespace cuda
}  // namespace xgboost::common
