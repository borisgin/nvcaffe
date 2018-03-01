#!/usr/bin/env bash
# build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if ! $WITH_CMAKE ; then
  make -j"$(nproc)" all test pycaffe warn
else
  cd build
  make -j"$(nproc)" all test.testbin
fi
make lint
