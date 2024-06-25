#include "common.h"  // for safe_cuda
#include "cuda_rt_utils.h"
#include "device_helpers.cuh"

namespace xgboost::common {
void CudaPrefetch(CudaPrefetchConfig const& config, void* ptr, std::size_t n_bytes) {
  auto device = CurrentDevice();
  std::cout << "r:" << config.read_mostly << " p:" << config.preferred
            << " a:" << config.accessed_by << " p:" << config.prefetch << std::endl;
  if (config.read_mostly) {
    dh::safe_cuda(cudaMemAdvise(ptr, n_bytes, cudaMemAdviseSetReadMostly, device));
  }
  if (config.preferred) {
    dh::safe_cuda(cudaMemAdvise(ptr, n_bytes, cudaMemAdviseSetPreferredLocation, device));
  }
  if (config.accessed_by) {
    dh::safe_cuda(cudaMemAdvise(ptr, n_bytes, cudaMemAdviseSetAccessedBy, device));
  }
  if (config.prefetch) {
    dh::safe_cuda(cudaMemPrefetchAsync(ptr, n_bytes, device, dh::DefaultStream()));
  }
}
}  // namespace xgboost::common
