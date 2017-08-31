#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/alexnet_bn/solver_16k.prototxt -gpu=all \
    2>&1 | tee models/alexnet_bn/logs/alexnetbn_16k_lr16_fp24.log

