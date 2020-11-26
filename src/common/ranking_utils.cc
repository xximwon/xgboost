/**
 * Copyright 2023 by XGBoost contributors
 */
#include "ranking_utils.h"

#include <dmlc/omp.h>  // omp_in_parallel

#include <algorithm>  // std::copy
#include <cstddef>    // std::size_t
#include <vector>     // std::vector

#include "linalg_op.h"          // cbegin,cend
#include "threading_utils.h"    // common::ParallelFor,Guided
#include "xgboost/base.h"       // XGBOOST_PARALLEL_STABLE_SORT
#include "xgboost/context.h"    // Context
#include "xgboost/data.h"       // MetaInfo
#include "xgboost/linalg.h"     // Tensor,Vector,Range,VectorView
#include "xgboost/parameter.h"  // DMLC_REGISTER_PARAMETER

namespace xgboost {
namespace ltr {
namespace cpu_impl {
/**
 * \brief Calculate IDCG at position k
 *
 *   Ties are averaged out.
 *
 * \param sorted_labels Sorted labels (relevance degree) in a single query group.
 * \param k             Truncation level
 *
 * \return IDCG
 */
double CalcIDCGAtK(linalg::VectorView<float const> sorted_labels, std::size_t k) {
  double ret = 0.0f;
  std::size_t group_size = sorted_labels.Size();
  // counts for all labels
  std::vector<std::size_t> label_cnt(ltr::MaxRel(), 0);
  for (std::size_t i = 0; i < group_size; ++i) {
    ++label_cnt[static_cast<ltr::rel_degree_t>(sorted_labels(i))];
  }
  auto top_label = static_cast<ltr::rel_degree_t>(ltr::MaxRel()) - 1;

  // start from top label, and accumulate DCG
  for (std::size_t j = 0; j < std::min(k, group_size); ++j) {
    while (top_label > 0 && label_cnt[top_label] == 0) {
      top_label -= 1;
    }
    if (top_label < 0) {
      break;
    }
    ret += CalcNDCGDiscount(j) * CalcNDCGGain(top_label);
    label_cnt[top_label] -= 1;
  }
  return ret;
}

void CalcInvIDCG(Context const* ctx, MetaInfo const& info, std::size_t p,
                 linalg::Vector<double>* out) {
  auto h_labels = info.labels.HostView();
  auto n_groups = info.group_ptr_.size() - 1;
  out->Reshape(n_groups);
  auto h_out = out->HostView();
  auto make_range = [&](bst_group_t g) {
    return linalg::Range(info.group_ptr_[g], info.group_ptr_[g + 1]);
  };
  common::ParallelFor(n_groups, ctx->Threads(), common::Sched::Guided(), [&](auto g) {
    auto label = h_labels.Slice(make_range(g), 0);
    linalg::Vector<float> sorted_labels;
    sorted_labels.Reshape(label.Size());
    auto h_sorted_labels = sorted_labels.HostView();
    std::copy(linalg::cbegin(label), linalg::cend(label), linalg::begin(h_sorted_labels));
    auto s_sorted_labels = h_sorted_labels.Values();
    if (omp_in_parallel()) {
      std::stable_sort(s_sorted_labels.data(), s_sorted_labels.data() + s_sorted_labels.size(),
                       std::greater<>{});
    } else {
      XGBOOST_PARALLEL_STABLE_SORT(s_sorted_labels.data(),
                                   s_sorted_labels.data() + s_sorted_labels.size(),
                                   std::greater<>{});
    }
    auto idcg = CalcIDCGAtK(h_sorted_labels, p);
    double inv_idcg = idcg == 0.0 ? 1.0 : 1.0 / idcg;
    h_out(g) = inv_idcg;
  });
}
}  // namespace cpu_impl
}  // namespace ltr
DMLC_REGISTER_PARAMETER(LambdaMARTParam);
}  // namespace xgboost
