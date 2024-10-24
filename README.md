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
./build_oai -w AERIAL --gNB --ninja
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
