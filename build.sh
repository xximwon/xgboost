CUDA_VERSION=12.4.1
NCCL_VERSION=2.22.3-1
RAPIDS_VERSION=24.08
DEV_RAPIDS_VERSION=24.10
SPARK_VERSION=3.5.1
JDK_VERSION=8
R_VERSION=4.3.2
is_pull_request=0
is_release_branch=1
BUILDKITE_COMMIT=86291473566330d2335cfee16082f134b4bbb8d4

WHEEL_TAG=manylinux_2_28_x86_64
arch_flag=""


command_wrapper="tests/ci_build/ci_build.sh gpu_build_rockylinux8 --build-arg "`
                `"CUDA_VERSION_ARG=$CUDA_VERSION --build-arg "`
                `"NCCL_VERSION_ARG=$NCCL_VERSION --build-arg "`
                `"RAPIDS_VERSION_ARG=$RAPIDS_VERSION"

echo "--- Build libxgboost from the source"
$command_wrapper tests/ci_build/build_via_cmake.sh \
		 -DCMAKE_PREFIX_PATH="/opt/grpc;/opt/rmm;/opt/rmm/lib64/rapids/cmake" \
		 -DUSE_CUDA=ON \
		 -DUSE_OPENMP=ON \
		 -DHIDE_CXX_SYMBOLS=ON \
		 -DPLUGIN_FEDERATED=ON \
		 -DPLUGIN_RMM=ON \
		 -DUSE_NCCL=ON \
		 -DUSE_NCCL_LIB_PATH=ON \
		 -DNCCL_INCLUDE_DIR=/usr/include \
		 -DUSE_DLOPEN_NCCL=ON \
  ${arch_flag}
echo "--- Build binary wheel"
$command_wrapper bash -c \
  "cd python-package && rm -rf dist/* && pip wheel --no-deps -v . --wheel-dir dist/"
$command_wrapper python tests/ci_build/rename_whl.py  \
  --wheel-path python-package/dist/*.whl  \
  --commit-hash ${BUILDKITE_COMMIT}  \
  --platform-tag ${WHEEL_TAG}

echo "--- Audit binary wheel to ensure it's compliant with ${WHEEL_TAG} standard"
tests/ci_build/ci_build.sh manylinux_2_28_x86_64 auditwheel repair \
  --plat ${WHEEL_TAG} python-package/dist/*.whl
$command_wrapper python tests/ci_build/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${BUILDKITE_COMMIT}  \
  --platform-tag ${WHEEL_TAG}
mv -v wheelhouse/*.whl python-package/dist/
# Make sure that libgomp.so is vendored in the wheel
tests/ci_build/ci_build.sh manylinux_2_28_x86_64 bash -c \
  "unzip -l python-package/dist/*.whl | grep libgomp  || exit -1"
