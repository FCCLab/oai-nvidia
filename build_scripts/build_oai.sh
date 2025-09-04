#!/bin/bash

# Monitor the log file
LOG_FILE="/home/vantuan_ngo/oai-nvidia-sera/openairinterface5g/cmake_targets/log/all.txt"
[ -f "$LOG_FILE" ] || mkdir -p "$(dirname "$LOG_FILE")" && touch "$LOG_FILE"
tail -f "$LOG_FILE" &

# Change to the build directory
cd ../openairinterface5g/cmake_targets || { echo "Directory not found"; exit 1; } ; 

# Run the build in the background, saving output to a log file
BUILD_COMMAND="./build_oai -w AERIAL --gNB --ninja"
bash -c "$BUILD_COMMAND"
