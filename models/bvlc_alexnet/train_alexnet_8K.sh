#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_8K.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_larc_B8K_lr16_fp16.log
