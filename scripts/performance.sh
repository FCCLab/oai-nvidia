sudo cpupower idle-set -D 0
# Get current CPU power mode
current_mode=$(cpupower frequency-info | grep "current governor" | awk '{print $3}')
echo "Current CPU power mode: $current_mode"

# Set the tuned profile to realtime
sudo tuned-adm profile realtime
current_profile=$(tuned-adm active)
echo "Current tuned profile: $current_profile"

# # Set the CPU power mode to performance
# sudo cpupower frequency-set -g performance
# current_mode=$(cpupower frequency-info | grep "current CPU frequency" | awk '{print $4}')
# echo "Current CPU frequency: $current_mode"

