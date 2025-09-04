#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# USAGE: PYAERIAL_IMAGE=<image> $cuBB_SDK/pyaerial/container/run.sh

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
host_cuBB_SDK=$(builtin cd $SCRIPT_DIR/../..;pwd)
echo $SCRIPT starting...
source $host_cuBB_SDK/cuPHY-CP/container/setup.sh


if [ -z "$1" ]; then
   echo Start container instance at bash prompt
   CMDS="/bin/bash"
else
   CMDS="$@"
   echo Run command then exit container
   echo Command: $CMDS
fi

if [[ ! $(grep -i "export PS1=\"\[host:" ~/.bashrc) ]]; then
    CMDS="echo 'export PS1=\"[host: $host_cuBB_SDK] \$PS1\"' >> ~/.bashrc && $CMDS"
fi

AERIAL_PLATFORM=${AERIAL_PLATFORM:-amd64}
TARGETARCH=$(basename $AERIAL_PLATFORM)

if [[ -z $PYAERIAL_IMAGE ]]; then
   PYAERIAL_IMAGE=pyaerial:${USER}-${AERIAL_VERSION_TAG}-${TARGETARCH}
fi

docker run --privileged \
            -it --rm \
            $AERIAL_EXTRA_FLAGS \
            --gpus all \
            --name pyaerial_$USER \
            --hostname pyaerial_$USER \
            --add-host pyaerial_$USER:127.0.0.1 \
            --network host --shm-size=4096m \
            --device=/dev/gdrdrv:/dev/gdrdrv \
            -u $USER_ID:$GROUP_ID \
            -w /opt/nvidia/cuBB \
            -v $host_cuBB_SDK:/opt/nvidia/cuBB \
            -v $host_cuBB_SDK:/opt/nvidia/aerial_sdk \
            -v $(echo ~):$(echo ~) \
            -v /dev/hugepages:/dev/hugepages \
            -v /lib/modules:/lib/modules \
            -v /var/log/aerial:/var/log/aerial \
            -e host_cuBB_SDK=$host_cuBB_SDK \
            --userns=host --ipc=host \
            $PYAERIAL_IMAGE fixuid -q /bin/bash -c "$CMDS"
