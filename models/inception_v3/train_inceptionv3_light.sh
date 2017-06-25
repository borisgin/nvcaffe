#!/usr/bin/env sh

./build/tools/caffe train --solver=models/inception_v3/solver_light.prototxt -gpu=all \
    2>&1 | tee models/inception_v3/logs/inceptionv3_light.log
