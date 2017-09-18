#!/usr/bin/env sh

./build/tools/caffe train --solver=models/resnet50/solver_lars.prototxt -gpu=all \
    2>&1 | tee models/resnet50/logs/resnet50_lars_b64_lr0.05_e150.log
