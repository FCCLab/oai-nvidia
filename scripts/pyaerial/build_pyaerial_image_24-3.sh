#!/bin/bash

docker pull nvcr.io/qhrjhjrvlsbu/aerial-cuda-accelerated-ran:24-3-cubb

set -e

rm -rf cuBB
docker run --rm -d --name cuBB nvcr.io/qhrjhjrvlsbu/aerial-cuda-accelerated-ran:24-3-cubb
docker cp cuBB:/opt/nvidia/cuBB cuBB
docker stop cuBB
cd cuBB

pip install hpccm

export PATH=$HOME/.local/bin:$PATH

export cuBB_SDK=`pwd`
AERIAL_BASE_IMAGE=nvcr.io/qhrjhjrvlsbu/aerial-cuda-accelerated-ran:24-3-cubb $cuBB_SDK/pyaerial/container/build.sh

$cuBB_SDK/pyaerial/container/run.sh

# cd $cuBB_SDK
# cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native -DNVIPC_FMTLOG_ENABLE=OFF -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
# cmake --build build -t _pycuphy pycuphycpp
# ./pyaerial/scripts/install_dev_pkg.sh

# $cuBB_SDK/pyaerial/scripts/run_unit_tests.sh
