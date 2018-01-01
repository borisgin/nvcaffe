#!/usr/bin/env sh

./build/tools/caffe train --solver=models/resnet50/solver_lars.prototxt -gpu=all \
    2>&1 | tee models/resnet50/logs/resnet50_lars_b1024_lr9.0_e100_fp16_aug.log
