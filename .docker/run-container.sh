#!/bin/bash

HOST_USERNAME=$(echo $USER | sed "s/@[[:print:]]*//")
HOST_UID=$(id -u)
HOST_GID=$(id -g)

docker run \
    -it \
    --rm \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    -e HOST_UID=$HOST_UID \
    -e HOST_GID=$HOST_GID \
    -e HOST_USERNAME=$HOST_USERNAME \
    -v $(pwd):/LitDreamerV3 \
    -w /LitDreamerV3 \
    --name LitDreamerV3 \
    litdreamerv3:latest \
    bash
