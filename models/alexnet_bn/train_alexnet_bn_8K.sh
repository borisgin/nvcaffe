#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/alexnet_bn/solver_8K.prototxt -gpu=all \
    2>&1 | tee models/alexnet_bn/logs/alexnetbn_8K_lr10_fp16.log

