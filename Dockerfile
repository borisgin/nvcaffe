FROM nvdl.githost.io:4678/dgx/cuda:8.0-cudnn6-devel-ubuntu14.04
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

ENV CAFFE_VERSION 0.16.0
LABEL com.nvidia.caffe.version="0.16.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        cython \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-all-dev \
        python-dateutil \
        python-gflags \
        python-h5py \
        python-leveldb \
        python-matplotlib \
        python-networkx \
        python-numpy \
        python-opencv \
        python-pandas \
        python-pil \
        python-pip \
        python-protobuf \
        python-scipy \
        python-skimage \
        python-yaml && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DUSE_NCCL=ON -DUSE_CUDNN=ON -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="52 60 61" -DCUDA_ARCH_PTX="61" .. && \
    make -j"$(nproc)" install && \
    make clean && \
    cd .. && rm -rf build && \
    ldconfig

ENV PYTHONPATH $PYTHONPATH:/usr/local/python

RUN chmod -R a+w /workspace

################################################################################
# Show installed packages
################################################################################

RUN echo "------------------------------------------------------" && \
    echo "-- INSTALLED PACKAGES --------------------------------" && \
    echo "------------------------------------------------------" && \
    echo "[[dpkg -l]]" && \
    dpkg -l && \
    echo "" && \
    echo "[[pip list]]" && \
    pip list && \
    echo "" && \
    echo "------------------------------------------------------" && \
    echo "-- FILE SIZE, DATE, HASH -----------------------------" && \
    echo "------------------------------------------------------" && \
    echo "[[find /usr/bin /usr/sbin /usr/lib /usr/local /workspace -type f | xargs ls -al]]" && \
    (find /usr/bin /usr/sbin /usr/lib /usr/local /workspace -type f | xargs ls -al || true) && \
    echo "" && \
    echo "[[find /usr/bin /usr/sbin /usr/lib /usr/local /workspace -type f | xargs md5sum]]" && \
    (find /usr/bin /usr/sbin /usr/lib /usr/local /workspace -type f | xargs md5sum || true)
