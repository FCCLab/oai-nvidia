export AERIAL_VERSION_TAG=24-2.1-cubb
export AERIAL_PLATFORM=arm64
export PYAERIAL_IMAGE=pyaerial:${USER}-${AERIAL_VERSION_TAG}-${AERIAL_PLATFORM}

cd cuBB

export cuBB_SDK=`pwd`
$cuBB_SDK/pyaerial/container/run.sh

cd $cuBB_SDK
cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native -DNVIPC_FMTLOG_ENABLE=OFF -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
cmake --build build -t _pycuphy pycuphycpp
./pyaerial/scripts/install_dev_pkg.sh
