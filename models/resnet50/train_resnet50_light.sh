#!/usr/bin/env sh

./build/tools/caffe train --solver=models/resnet50/solver_light.prototxt -gpu=all \
    2>&1 | tee models/resnet50/logs/resnet50_light_hot0.5.log
