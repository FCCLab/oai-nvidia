#!/bin/bash

echo "Disabling NTP"
sudo timedatectl set-ntp false

echo "Copying ptp.conf to /etc/ptp.conf"
sudo cp ptp.conf /etc/ptp.conf

echo "Copying ptp4l.service to /lib/systemd/system/"
sudo cp ptp4l.service /lib/systemd/system/

echo "Reloading systemd"
sudo systemctl daemon-reload
sudo systemctl restart ptp4l.service
sudo systemctl enable ptp4l.service

echo "Copying phc2sys.service to /lib/systemd/system/"
sudo cp phc2sys.service /lib/systemd/system/

echo "Reloading systemd"
sudo systemctl daemon-reload
sudo systemctl restart phc2sys.service
sudo systemctl enable phc2sys.service

# sudo systemctl status ptp4l.service

sudo tail -f /var/log/syslog
