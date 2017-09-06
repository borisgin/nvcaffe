#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/alexnet_bn/solver_32k.prototxt -gpu=all \
    2>&1 | tee models/alexnet_bn/logs/alexnetbn_32k_lr32_fp32.log

