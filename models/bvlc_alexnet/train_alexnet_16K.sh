#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_16K.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_larc_B16K_lr16_e100.log
