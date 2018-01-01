#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_googlenet/solver_lars.prototxt -gpu=all \
    2>&1 | tee models/bvlc_googlenet/logs/googlenet_lars_fp16_b8K_lr6.log
