#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_augment.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_augment_B1024_lr2.0.log
