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