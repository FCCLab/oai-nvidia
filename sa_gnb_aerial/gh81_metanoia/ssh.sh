#!/bin/bash

# Check if the network interface exists
INTERFACE="aerial00"
if ! ip link show "$INTERFACE" >/dev/null 2>&1; then
  echo "Error: Network interface $INTERFACE does not exist."
  exit 1
fi

# Configure the network interface
sudo ifconfig "$INTERFACE" 192.168.1.100/24 || {
  echo "Error: Failed to configure $INTERFACE with IP 192.168.1.100/24."
  exit 1
}

# Check if sshpass is installed
if ! command -v sshpass >/dev/null 2>&1; then
  echo "Error: sshpass is not installed. Please install it (e.g., 'sudo apt install sshpass')."
  exit 1
fi

# Ping the target host until successful
SSH_HOST="192.168.1.3"
MAX_ATTEMPTS=30  # Maximum number of ping attempts
ATTEMPT=1

echo "Attempting to ping $SSH_HOST at $(date)"
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "Ping attempt $ATTEMPT of $MAX_ATTEMPTS..."
    if ping -c 1 "$SSH_HOST" >/dev/null 2>&1; then
        echo "Successfully pinged $SSH_HOST at $(date)"
        break
    fi
    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        echo "Error: Failed to ping $SSH_HOST after $MAX_ATTEMPTS attempts."
        exit 1
    fi
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
done

# SSH to the target host
SSH_USER="root"
SSH_PASS="root"  # Replace with a secure method (e.g., SSH keys or prompt)

sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no "$SSH_USER@$SSH_HOST" || {
  echo "Error: Failed to SSH to $SSH_USER@$SSH_HOST."
  exit 1
}