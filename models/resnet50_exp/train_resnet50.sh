#!/usr/bin/env sh

./build/tools/caffe train --solver=models/resnet50_exp/solver.prototxt -gpu=all \
    2>&1 | tee models/resnet50_exp/logs/resnet50_nobn_lr0.1.log
