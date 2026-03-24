#!/bin/bash
NAME=sk047
IMAGE_NAME="myservice-backend"
VERSION="1.0.4"

CPU_PLATFORM=amd64

# Docker 이미지 빌드
docker build \
  --tag ${NAME}-${IMAGE_NAME}:${VERSION} \
  --file Dockerfile-backend \
  --platform linux/${CPU_PLATFORM} \
  ${IS_CACHE} .
