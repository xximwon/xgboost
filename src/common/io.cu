#include <cufile.h>
#include <fcntl.h>     // for open, O_DIRECT
#include <sys/mman.h>  // for mmap, mmap64, munmap
#include <unistd.h>    // for close

#include <cstdint>

#include "common.h"
#include "io.h"
#include "xgboost/string_view.h"

namespace xgboost {
void SafeCUFile(CUfileError_t status) {
  if (IS_CUFILE_ERR(status.err)) {
    if (status.err != CU_FILE_SUCCESS) {
      auto err = CUFILE_ERRSTR(status.err);
      LOG(FATAL) << err;
    }
  } else {
    if (status.cu_err != CUDA_SUCCESS) {

    }
  }
  if (IS_CUDA_ERR(status)) {

  }
}

void CUFileRead(StringView path) {
  std::int32_t fd = open(path.c_str(), O_DIRECT);
  CUfileHandle_t fh;

  CUfileDescr_t desc;
  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  CUfileError_t status = cuFileHandleRegister(&fh, &desc);
  if (status.err != CU_FILE_SUCCESS) {
    LOG(FATAL) << "";
    // leaking
  }

  void* gpumem_buf;
  cudaMalloc(&gpumem_buf, 12);
  cuFileRead(fh, gpumem_buf, 12, 0, 0);
  cudaFree(gpumem_buf);

  cuFileHandleDeregister(fh);
  close(fd);
}
}  // namespace xgboost
