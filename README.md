# OAI NVIDIA DELTA IOT

## Clone
```
git submodule update --init --recursive
```

## Build
```
cd sa_gnb_aerial 
docker compose up -d
docker exec -it nv-cubb bash
mkdir build
cd build/
cmake ..
make -j32
```

```
cd openairinterface5g/cmake_targets
./build_oai -w AERIAL --gNB --ninja --build-e2
```

## Run

### HIGH PHY
```
docker exec -it nv-cubb bash
./aerial_l1_entrypoint.sh delta
```

### MAC+
```
cd ran_build/build                     
sudo ./nr-softmodem -O ../../../ci-scripts/conf_files/gnb-vnf.sa.band78.273prb.aerial.conf --log_config.global_log_options level,nocolor,time
```

## PTP
```
ethtool -T aerial00| grep PTP | awk '{print $4}'
cat /lib/systemd/system/phc2sys.service
```

## N2N3
```
sudo ip address add 192.168.120.115/24 dev ens1f1
sudo ip address add 5.5.5.115/24 dev ens1f1
```

```
jupyter lab password
jupyter notebook
```

# pyAerial
```
docker run --rm -d --name cuBB <container image file>
docker cp cuBB:/opt/nvidia/cuBB cuBB
docker stop cuBB
cd cuBB

export cuBB_SDK=`pwd`
AERIAL_BASE_IMAGE=nvcr.io/qhrjhjrvlsbu/aerial-cuda-accelerated-ran:24-3-cubb $cuBB_SDK/pyaerial/container/build.sh

export cuBB_SDK=`pwd`
$cuBB_SDK/pyaerial/container/run.sh

cd $cuBB_SDK/pyaerial/notebooks
jupyter lab --ip=0.0.0.0
```

# Liteon RU

## 01 DHCP
```
cd dhcp
./liteon.sh
sshpass -p user ssh user@192.168.1.101
```

## 02 Static IP
```
sudo ip addr add 10.101.131.100/24 dev aerial0.d9
sshpass -p user ssh user@10.101.131.194
```

## Common
```
en
liteon168
# configure terminal

Entering configuration mode...
(config)# bandwidth 50000000
Old Band Width = 100000000
New Band Width = 50000000
(config)# 
```

DU MAC = DU MAC Address
```
# show running-config 
Band Width = 50000000
Center Frequency = 3425010000
Compression Bit = 9
Control and User Plane vlan = 3
M Plane vlan = 0
default gateway = 10.1.7.1
dpd mode : Enable
DU MAC Address = 9c63c0a70832
phase compensation mode : Enable
RX attenuation = 14
TX attenuation = 0
subcarrier spacing = 1
rj45_vlan_ip = 10.102.131.61
SFP_vlan_ip = 10.102.131.62
SFP_non_vlan_static_ip = 10.101.131.194
prach eAxC-id port 0, 1, 2, 3 = 0x0000, 0x0001, 0x0002, 0x0003
slotid = 0x00000001
jumboframe = 0x00000000
sync source : PTP
```

RU MAC = MAC of eth1
```
# show eth-info 
eth0      Link encap:Ethernet  HWaddr e8:c7:4f:25:7f:7c  
          inet addr:10.1.7.194  Bcast:10.1.7.255  Mask:255.255.255.0
          inet6 addr: fe80::eac7:4fff:fe25:7f7c/64 Scope:Link
          UP BROADCAST MULTICAST  MTU:1500  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:3 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:0 (0.0 B)  TX bytes:278 (278.0 B)
          Interrupt:29 

eth1      Link encap:Ethernet  HWaddr e8:c7:4f:25:89:40  
          inet addr:10.101.131.194  Bcast:10.101.131.255  Mask:255.255.255.0
          inet6 addr: fe80::eac7:4fff:fe25:8940/64 Scope:Link
          UP BROADCAST RUNNING  MTU:1500  Metric:1
          RX packets:1577 errors:0 dropped:4 overruns:0 frame:0
          TX packets:693 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:117732 (114.9 KiB)  TX bytes:49115 (47.9 KiB)

lo        Link encap:Local Loopback  
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:2 errors:0 dropped:0 overruns:0 frame:0
          TX packets:2 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:176 (176.0 B)  TX bytes:176 (176.0 B)

# 
```
