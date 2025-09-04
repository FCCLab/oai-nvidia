#!/bin/bash

echo "Copying network files to /etc/systemd/network/"
sudo cp 20-aerial0* /etc/systemd/network/

echo "Copying network files to /etc/netplan/"
sudo cp 99-aerial.yaml /etc/netplan/

echo "Applying network configuration"
sudo netplan apply

echo "Done"
