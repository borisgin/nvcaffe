#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_ls.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_ls.log
