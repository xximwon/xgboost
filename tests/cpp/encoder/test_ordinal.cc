#include <gtest/gtest.h>

#include <vector>

#include "../../../src/encoder/ordinal.h"

namespace enc {
TEST(CategoricalEncoder, Str) {
  CategoryRecoder encoder;

  std::vector<std::int8_t> orig_names{'c', 'b', 'd', 'a'};
  std::vector<std::int32_t> orig_offsets{0, 1, 2, 3, 4};
  std::vector<HostCatIndexView> orig_columns;
  orig_columns.emplace_back(CatStrArrayView{orig_offsets, orig_names});
  HostColumnsView orig_dict;
  orig_dict.columns = orig_columns;

  std::vector<std::int8_t> new_names{'c', 'a', 'b'};
  std::vector<std::int32_t> new_offsets{0, 1, 2, 3};

  HostColumnsView new_dict;
  std::vector<HostCatIndexView> new_columns;
  new_columns.emplace_back(CatStrArrayView{new_offsets, new_names});
  new_dict.columns = new_columns;

  auto mapping = encoder.Recode(orig_dict, new_dict);
  ASSERT_EQ(mapping.size(), 3);
  auto sol = decltype(mapping){0, 3, 1};
  ASSERT_EQ(mapping, sol);
}
}  // namespace enc
