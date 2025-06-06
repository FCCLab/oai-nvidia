#!/bin/bash

# Save current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Build Flex-ric
cd $SCRIPT_DIR
cd ../openairinterface5g/openair2/E2AP/flexric
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install

# Monitor the log file
LOG_FILE="/home/vantuan_ngo/oai-nvidia-sera/openairinterface5g/cmake_targets/log/all.txt"
[ -f "$LOG_FILE" ] || { echo "Log file not found"; exit 1; }
tail -f "$LOG_FILE" &

# Change to the build directory
cd $SCRIPT_DIR
cd ../openairinterface5g/cmake_targets
# Run the build in the background, saving output to a log file
BUILD_COMMAND="./build_oai -w AERIAL --gNB --ninja --build-e2"
bash -c "$BUILD_COMMAND"
