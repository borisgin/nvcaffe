#!/usr/bin/env bash
# CMake configuration

mkdir -p build
cd build

#Travis doesn't have CUDA driver installed
ARGS="-DCMAKE_BUILD_TYPE=Release -DBLAS=Open -DNO_NVML=On"

if $WITH_PYTHON3 ; then
  ARGS="$ARGS -Dpython_version=3"
fi

if $WITH_IO ; then
  ARGS="$ARGS -DUSE_LMDB=On -DUSE_LEVELDB=On"
else
  ARGS="$ARGS -DUSE_LMDB=Off -DUSE_LEVELDB=Off"
fi

if $WITH_CUDA ; then
  # Only build SM50
  ARGS="$ARGS -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN=\"50\" -DCUDA_ARCH_PTX=\"\""
fi

if $WITH_CUDNN ; then
  ARGS="$ARGS -DUSE_CUDNN=On"
else
  ARGS="$ARGS -DUSE_CUDNN=Off"
fi

cmake .. $ARGS

