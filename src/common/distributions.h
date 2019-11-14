#ifndef XGBOOST_COMMON_DISTRIBUTIONS_H_
#define XGBOOST_COMMON_DISTRIBUTIONS_H_

#include <iterator>

#include "xgboost/base.h"
#include "random.h"
#include "./math.h"

namespace xgboost {
namespace dist {

template <typename R>
struct TransformIter {
  std::function<R(size_t i)> f_;
  size_t i;

  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::size_t;
  using value_type = R;

  template <typename F>
  TransformIter(F f, size_t _i) : f_{f}, i{_i} {}
  auto operator*() -> decltype(f_(i)) {
    return f_(i);
  }
  R operator[](size_t i) const {
    return *(TransformIter {f_, i});
  }
  TransformIter& operator++() {
    i++;
    return *this;
  }
  TransformIter operator++(int) {
    auto ret = *this;
    ++(*this);
    return ret;
  }
  bool operator==(TransformIter const &that) const {
    return i == that.i;
  }
  bool operator!=(TransformIter const &that) const {
    return i != that.i;
  }
  size_t operator-(TransformIter const &that) const {
    return i - that.i;
  }
};

struct Normal {
  static std::pair<float, float> FisherInfo(float mean, float var) {
    return {1 / var, 2 / var};
  }
  static float NegativeLogLikelihood(float mean, float var) { return 0; }

  static std::pair<float, float> GradientNLL(float mean, float var, float x) {
    auto sloc = (mean - x) / var;
    auto svar = 1 - (mean - x) * (mean - x) / var;
    return {sloc, svar};
  }
  uint32_t static constexpr ParameterLength() { return 2; }

  /*
   * \brief Maximum likelihood estimation.
   */
  static std::pair<float, float> Fit(std::vector<float> y) {
    if (y.size() == 0) { return {std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::quiet_NaN()}; }
    float mean = common::Mean(y.cbegin(), y.cend());
    auto f = [&y, mean](bst_row_t i) { return (y[i] - mean) * (y[i] - mean); };
    TransformIter<float> beg{f, 0};
    TransformIter<float> end{f, y.size()};
    auto scale = std::sqrt(common::Mean(beg, end));
    return {mean, scale};
  }
};

}  // namespace dist
}  // namespace xgboost

#endif  // XGBOOST_COMMON_DISTRIBUTIONS_H_
