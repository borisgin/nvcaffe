#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/alexnet_bn/solver.prototxt -gpu=all \
    2>&1 | tee models/alexnet_bn/logs/alexnet_bn_base2_lr0.08_wd0.0005_l4.log
