#!/bin/bash

sudo mount /dev/nvme1n1 /home/data

sudo chown -R 472:472 grafana*
docker compose up -d
docker ps -a | grep wirelessdb
