# Run container without dev mode
**This is for cuBB 24-3**
### Build containers
##### OAI gNB
```bash
cd sa_gnb_aerial_gh
cp nvipc_src.1970.01.01.tar.gz ../openairinterface5g/
cd ../openairinterface5g
docker build . -f docker/Dockerfile.base.ubuntu22 --tag ran-base:latest
docker build . -f docker/Dockerfile.gNB.aerial.ubuntu22 --tag oai-gnb-aerial:latest
```
##### Create cubb-build:24-3
```bash
docker run -it  --name nv-cubb-tmp nvcr.io/qhrjhjrvlsbu/aerial-cuda-accelerated-ran:24-3-cubb  /bin/bash
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native
make -j$nproc
```
Commnit the container
```bash
docker commit nv-cubb-tmp cubb-build:24-3
```
##### Core network
Follow this: https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed/-/blob/master/docs/BUILD_IMAGES.md
Or below steps:
```bash
cd
git clone  https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed.git
cd oai-cn5g-fed
git checkout develop
./scripts/syncComponents.sh --nrf-branch develop --amf-branch develop \
                              --smf-branch develop --upf-branch develop \
                              --ausf-branch develop --udm-branch develop \
                              --udr-branch develop --upf-vpp-branch develop \
                              --nssf-branch develop --nef-branch develop \
                              --pcf-branch develop --lmf-branch develop
docker build --target oai-amf --tag oai-amf:develop \
               --file component/oai-amf/docker/Dockerfile.amf.ubuntu \
               --build-arg BASE_IMAGE=ubuntu:focal \
               component/oai-amf
docker build --target oai-smf --tag oai-smf:develop \
               --file component/oai-smf/docker/Dockerfile.smf.ubuntu \
               --build-arg BASE_IMAGE=ubuntu:22.04 \
               component/oai-smf
docker build --target oai-nrf --tag oai-nrf:develop \
               --file component/oai-nrf/docker/Dockerfile.nrf.ubuntu \
               --build-arg BASE_IMAGE=ubuntu:jammy \
               component/oai-nrf
docker build --target oai-upf --tag oai-upf:develop \
               --file component/oai-upf/docker/Dockerfile.upf.ubuntu \
               --build-arg BASE_IMAGE=ubuntu:22.04 \
               component/oai-upf
docker build --target oai-ausf --tag oai-ausf:develop \
               --file component/oai-ausf/docker/Dockerfile.ausf.ubuntu \
               component/oai-ausf
docker build --target oai-udm --tag oai-udm:develop \
               --file component/oai-udm/docker/Dockerfile.udm.ubuntu \
               component/oai-udm
docker build --target oai-udr --tag oai-udr:develop \
               --file component/oai-udr/docker/Dockerfile.udr.ubuntu \
               component/oai-udr
# docker build --target oai-upf-vpp --tag oai-upf-vpp:develop \
#                --file component/oai-upf-vpp/docker/Dockerfile.upf-vpp.ubuntu \
#                component/oai-upf-vpp
docker build --target oai-nssf --tag oai-nssf:develop \
               --file component/oai-nssf/docker/Dockerfile.nssf.ubuntu \
               component/oai-nssf

```
### Run
- Core network
```bash
cd oai-cn5g
docker compose -f docker-compose-build.yaml up -d
```
- gNB
```bash
cd sa_gnb_aerial_gh
# Modify the 
docker compose up -d
```