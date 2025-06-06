#!/bin/bash

echo "Setting CPU idle to 0"
sudo cpupower idle-set -D 0

echo "Setting CPU to performance" 
sudo tuned-adm profile realtime

echo "Setting CPU frequency to performance"
for ((i=0;i<$(nproc);i++)); do sudo cpufreq-set -c $i -r -g performance; done
