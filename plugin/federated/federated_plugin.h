/**
 * Copyright 2024, XGBoost contributors
 */
#pragma once

#include <functional>   // for function
#include <memory>       // for unique_ptr
#include <string_view>  // for string_view

#include "xgboost/json.h"  // for Json
#include "xgboost/span.h"  // for Span

typedef void *FederatedPluginHandle;  // NOLINT

namespace xgboost::collective {
namespace federated {
// API exposed by the plugin
using CreateFn = FederatedPluginHandle(int, char const **);
using CloseFn = int(FederatedPluginHandle);
using ErrorFn = char const *();
// Gradient
using EncryptFn = int(FederatedPluginHandle handle, float const *in_gpair, size_t n_in,
                      uint8_t **out_gpair, size_t *n_out);
using SyncEncryptFn = int(FederatedPluginHandle handle, uint8_t const *in_gpair, size_t n_bytes,
                          uint8_t const **out_gpair, size_t *n_out);
// Vert Histogram
using ResetHistCtxVertFn = int(FederatedPluginHandle handle, std::uint32_t const *cutptrs,
                               std::size_t cutptr_len, std::int32_t const *bin_idx,
                               std::size_t n_idx);
using BuildHistVertFn = int(FederatedPluginHandle handle, uint64_t const **ridx,
                            size_t const *sizes, int32_t const *nidx, size_t len,
                            uint8_t **out_hist, size_t *out_len);
using SyncHistVertFn = int(FederatedPluginHandle handle, uint8_t *buf, size_t len, double **out,
                           size_t *out_len);
// Hori Histogram
using BuildHistHoriFn = int(FederatedPluginHandle handle, double const *in_histogram, size_t len,
                            uint8_t **out_hist, size_t *out_len);
using SyncHistHoriFn = int(FederatedPluginHandle handle, std::uint8_t const *buffer,
                           std::size_t len, double **out_hist, std::size_t *out_len);
}  // namespace federated

// Base class, only used for testing.
class FederatedPluginBase {
  std::vector<std::uint8_t> grad_;
  std::vector<std::uint8_t> hist_enc_;
  std::vector<double> hist_plain_;

 public:
  [[nodiscard]] virtual common::Span<std::uint8_t> EncryptGradient(common::Span<float const> data) {
    grad_.resize(data.size_bytes());
    auto casted =
        common::Span{reinterpret_cast<std::uint8_t const *>(data.data()), data.size_bytes()};
    std::copy_n(casted.data(), casted.size(), grad_.data());
    return grad_;
  }
  virtual void SyncEncryptedGradient(common::Span<std::uint8_t const>) {
    // nothing
  }

  // Vertical histogram
  virtual void Reset(common::Span<std::uint32_t const>, common::Span<std::int32_t const>) {}

  [[nodiscard]] virtual common::Span<std::uint8_t> BuildEncryptedHistVert(
      common::Span<std::uint64_t const *> rowptrs, common::Span<std::size_t const> sizes,
      common::Span<bst_node_t const> nids) {
    int total_bin_size = cuts_.back();
    int histo_size = total_bin_size * 2;
    *size = kPrefixLen + 8 * histo_size * nodes.size();
    int64_t buf_size = *size;
    int8_t *buf = static_cast<int8_t *>(calloc(buf_size, 1));
    memcpy(buf, kSignature, strlen(kSignature));
    memcpy(buf + 8, &buf_size, 8);
    memcpy(buf + 16, &kDataTypeHisto, 8);

    double *histo = reinterpret_cast<double *>(buf + kPrefixLen);
    for (const auto &node : nodes) {
      auto rows = node.second;
      for (const auto &row_id : rows) {
        auto num = cuts_.size() - 1;
        for (std::size_t f = 0; f < num; f++) {
          int slot = slots_[f + num * row_id];
          if ((slot < 0) || (slot >= total_bin_size)) {
            continue;
          }

          auto g = (*gh_pairs_)[row_id * 2];
          auto h = (*gh_pairs_)[row_id * 2 + 1];
          histo[slot * 2] += g;
          histo[slot * 2 + 1] += h;
        }
      }
      histo += histo_size;
    }

    return buf;
  }

  [[nodiscard]] virtual common::Span<double> SyncEncryptedHistVert(
      common::Span<std::uint8_t> hist) {
    std::vector<double> result = std::vector<double>();

    int8_t *ptr = static_cast<int8_t *>(buffer);
    auto rest_size = buf_size;

    while (rest_size > kPrefixLen) {
      if (!ValidDam(ptr, rest_size)) {
        break;
      }
      int64_t *size_ptr = reinterpret_cast<int64_t *>(ptr + 8);
      double *array_start = reinterpret_cast<double *>(ptr + kPrefixLen);
      auto array_size = (*size_ptr - kPrefixLen) / 8;
      result.insert(result.end(), array_start, array_start + array_size);
      rest_size -= *size_ptr;
      ptr = ptr + *size_ptr;
    }

    return result;
  }

  // Horizontal histogram
  [[nodiscard]] virtual common::Span<std::uint8_t> BuildEncryptedHistHori(
      common::Span<double const> hist) {
    hist_enc_.resize(hist.size_bytes());
    std::copy_n(reinterpret_cast<std::uint8_t const *>(hist.data()), hist.size_bytes(),
                hist_enc_.data());
    return hist_enc_;
  }
  [[nodiscard]] virtual common::Span<double> SyncEncryptedHistHori(
      common::Span<std::uint8_t const> hist) {
    hist_plain_.resize(hist.size_bytes());
    std::copy_n(hist.data(), hist.size_bytes(),
                reinterpret_cast<std::uint8_t const *>(hist_plain_.data()));
    return hist_plain_;
  }
};

/**
 * @brief Bridge for plugins that handle encryption.
 */
class FederatedPlugin : public FederatedPluginBase {
  // Federated plugin shared object, for dlopen.
  std::unique_ptr<void, std::function<void(void *)>> plugin_;

  federated::CreateFn *PluginCreate_{nullptr};
  federated::CloseFn *PluginClose_{nullptr};
  federated::ErrorFn *ErrorMsg_{nullptr};
  // Gradient
  federated::EncryptFn *Encrypt_{nullptr};
  federated::SyncEncryptFn *SyncEncrypt_{nullptr};
  // Vert Histogram
  federated::ResetHistCtxVertFn *ResetHistCtxVert_{nullptr};
  federated::BuildHistVertFn *BuildEncryptedHistVert_{nullptr};
  federated::SyncHistVertFn *SyncEncryptedHistVert_{nullptr};
  // Hori Histogram
  federated::BuildHistHoriFn *BuildEncryptedHistHori_{nullptr};
  federated::SyncHistHoriFn *SyncEncryptedHistHori_;

  // Object handle of the plugin.
  std::unique_ptr<void, std::function<void(void *)>> plugin_handle_;

  void CheckRC(std::int32_t rc, std::string_view msg) {
    if (rc != 0) {
      auto err_msg = ErrorMsg_();
      LOG(FATAL) << msg << ":" << err_msg;
    }
  }

 public:
  explicit FederatedPlugin(std::string_view path, Json config);
  ~FederatedPlugin();
  // Gradient
  [[nodiscard]] common::Span<std::uint8_t> EncryptGradient(
      common::Span<float const> data) override {
    std::uint8_t *ptr{nullptr};
    std::size_t n{0};
    auto rc = Encrypt_(this->plugin_handle_.get(), data.data(), data.size(), &ptr, &n);
    CheckRC(rc, "Failed to encrypt gradient");
    return {ptr, n};
  }
  void SyncEncryptedGradient(common::Span<std::uint8_t const> data) override {
    uint8_t const *out;
    std::size_t n{0};
    auto rc = SyncEncrypt_(this->plugin_handle_.get(), data.data(), data.size(), &out, &n);
    CheckRC(rc, "Failed to sync encrypt gradient");
  }

  // Vertical histogram
  void Reset(common::Span<std::uint32_t const> cutptrs,
             common::Span<std::int32_t const> bin_idx) override {
    auto rc = ResetHistCtxVert_(this->plugin_handle_.get(), cutptrs.data(), cutptrs.size(),
                                bin_idx.data(), bin_idx.size());
    CheckRC(rc, "Failed to set the data context for federated learning");
  }
  [[nodiscard]] common::Span<std::uint8_t> BuildEncryptedHistVert(
      common::Span<std::uint64_t const *> rowptrs, common::Span<std::size_t const> sizes,
      common::Span<bst_node_t const> nids) override {
    std::uint8_t *ptr{nullptr};
    std::size_t n{0};
    auto rc = BuildEncryptedHistVert_(this->plugin_handle_.get(), rowptrs.data(), sizes.data(),
                                      nids.data(), nids.size(), &ptr, &n);
    CheckRC(rc, "Failed to build the encrypted hist");
    return {ptr, n};
  }
  [[nodiscard]] common::Span<double> SyncEncryptedHistVert(
      common::Span<std::uint8_t> hist) override {
    double *ptr{nullptr};
    std::size_t n{0};
    auto rc =
        SyncEncryptedHistVert_(this->plugin_handle_.get(), hist.data(), hist.size(), &ptr, &n);
    CheckRC(rc, "Failed to sync the encrypted hist");
    return {ptr, n};
  }

  // Horizontal histogram
  [[nodiscard]] common::Span<std::uint8_t> BuildEncryptedHistHori(
      common::Span<double const> hist) override {
    std::uint8_t *ptr{nullptr};
    std::size_t n{0};
    auto rc =
        BuildEncryptedHistHori_(this->plugin_handle_.get(), hist.data(), hist.size(), &ptr, &n);
    CheckRC(rc, "Failed to build the encrypted hist");
    return {ptr, n};
  }
  [[nodiscard]] common::Span<double> SyncEncryptedHistHori(
      common::Span<std::uint8_t const> hist) override {
    double *ptr{nullptr};
    std::size_t n{0};
    auto rc =
        SyncEncryptedHistHori_(this->plugin_handle_.get(), hist.data(), hist.size(), &ptr, &n);
    CheckRC(rc, "Failed to sync the encrypted hist");
    return {ptr, n};
  }
};
}  // namespace xgboost::collective
