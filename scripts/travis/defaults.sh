#!/usr/bin/env bash
# set default environment variables

set -e

WITH_CMAKE=${WITH_CMAKE:-false}
WITH_PYTHON3=${WITH_PYTHON3:-false}
WITH_CUDA=${WITH_CUDA:-true}
WITH_CUDNN=${WITH_CUDNN:-false}
