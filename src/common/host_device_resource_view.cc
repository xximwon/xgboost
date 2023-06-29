#include "xgboost/host_device_resource_view.h"

#include <cstdint>  // for int32_t, uint32_t, uint64_t, uint8_t

#include "ref_resource_view.h"
#include "xgboost/base.h"        // for GradientPair, GradientPairPrecise
#include "xgboost/data.h"        // for FeatureType, Entry
#include "xgboost/tree_model.h"  // for Node, Segment, RTreeNodeStat

namespace xgboost {
// explicit instantiations are required, as HostDeviceResourceView isn't header-only
template class HostDeviceResourceView<float>;
template class HostDeviceResourceView<double>;
template class HostDeviceResourceView<GradientPair>;
template class HostDeviceResourceView<GradientPairPrecise>;
template class HostDeviceResourceView<int32_t>;  // bst_node_t
template class HostDeviceResourceView<uint8_t>;
template class HostDeviceResourceView<FeatureType>;
template class HostDeviceResourceView<Entry>;
template class HostDeviceResourceView<uint64_t>;  // bst_row_t
template class HostDeviceResourceView<uint32_t>;  // bst_feature_t
template class HostDeviceResourceView<RegTree::Node>;
template class HostDeviceResourceView<RegTree::CategoricalSplitMatrix::Segment>;
template class HostDeviceResourceView<RTreeNodeStat>;

#if defined(__APPLE__)
/*
 * On OSX:
 *
 * typedef unsigned int         uint32_t;
 * typedef unsigned long long   uint64_t;
 * typedef unsigned long       __darwin_size_t;
 */
template class HostDeviceResourceView<std::size_t>;
#endif  // defined(__APPLE__)
}  // namespace xgboost
