#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_googlenet/solver_parallel.prototxt -gpu=all \
    2>&1 | tee models/bvlc_googlenet/logs/googlenet_parallel_b256_lr2.log
