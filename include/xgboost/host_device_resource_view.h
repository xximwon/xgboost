#pragma once
#include <xgboost/context.h>
#include <xgboost/span.h>

#include <cstddef>  // size_t
#include <memory>   // for shared_ptr

namespace xgboost {
namespace common {
class ResourceHandler;
}

template <typename T>
class HostDeviceResourceView {
 public:
  using value_type = T;           // NOLINT
  using size_type = std::size_t;  // NOLINT

  HostDeviceResourceView() = default;
  HostDeviceResourceView(value_type* ptr, size_type n,
                         std::shared_ptr<common::ResourceHandler> mem);
  HostDeviceResourceView(value_type* ptr, size_type n, std::shared_ptr<common::ResourceHandler> mem,
                         value_type const& init);
  void SetDevice(std::int32_t device) const;
  [[nodiscard]] size_type Size() const;
  [[nodiscard]] bool Empty() const { return Size() == 0; }
  [[nodiscard]] int DeviceIdx() const;

  common::Span<T> DeviceSpan();
  common::Span<const T> ConstDeviceSpan() const;
  common::Span<const T> DeviceSpan() const { return ConstDeviceSpan(); }

  T* DevicePointer();
  T const* ConstDevicePointer() const;
  T const* DevicePointer() const { return ConstDevicePointer(); }

  common::Span<T> HostSpan() { return common::Span<T>{this->HostPointer(), this->Size()}; }
  common::Span<T const> HostSpan() const {
    return common::Span<T const>{this->HostPointer(), this->Size()};
  }
  common::Span<T const> ConstHostSpan() const { return HostSpan(); }

  T* HostPointer();
  T const* HostPointer() const { return ConstHostPointer(); }
  T const* ConstHostPointer() const;

  [[nodiscard]] bool HostCanRead() const;
  [[nodiscard]] bool HostCanWrite() const;
  [[nodiscard]] bool DeviceCanRead() const;
  [[nodiscard]] bool DeviceCanWrite() const;
};
}  // namespace xgboost
